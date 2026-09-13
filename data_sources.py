#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一多数据源管理器 v3.0
========================
经过实际测试验证（含海外服务器 Streamlit Cloud）的数据源优先级：

新闻：
  1. 东方财富 F10 个股资讯/公告直连API（emweb，纯urllib，稳定）
  2. 东方财富搜索API（search-api-web，备选）
  3. akshare stock_news_em（兜底，容错 ArrowInvalid 等版本问题）

资金流：
  1. 东方财富 push2his 多镜像HTTP轮询（14个数字镜像，实测5/5成功）
     —— 关键：海外环境 HTTPS 会被 RemoteDisconnected，必须用 HTTP + 多镜像
  2. akshare stock_individual_fund_flow（兜底）

实时行情：腾讯财经直连 → 东方财富多镜像直连

所有直连API使用纯urllib实现，不依赖akshare，海外服务器可正常访问。
"""
import time
import json
import gzip
import zlib
import ssl
import socket
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 兼容：pandas 2.2+ 默认可能用 pyarrow 字符串后端，会让 akshare 内部正则
# 报 "ArrowInvalid: Invalid regular expression: invalid escape sequence \u"。
# 关闭字符串的 pyarrow 推断，回退到 Python/object 字符串后端（正则走 re 模块，更宽松）。
try:
    pd.options.future.infer_string = False
except Exception:
    pass
try:
    pd.options.mode.string_storage = 'python'
except Exception:
    pass


def _decode_response(raw, encoding='utf-8'):
    """
    解码HTTP响应bytes，自动处理 gzip/deflate 压缩。
    东财服务器偶尔返回gzip（魔数0x1f 0x8b），urllib不会自动解压，需手动处理。
    """
    if raw is None:
        return ''
    if isinstance(raw, str):
        return raw
    # gzip 魔数 0x1f 0x8b
    if len(raw) >= 2 and raw[:2] == b'\x1f\x8b':
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
    else:
        # zlib/deflate
        try:
            raw = zlib.decompress(raw)
        except Exception:
            # 可能是无zlib头的deflate，或本身就是明文
            try:
                raw = zlib.decompress(raw, -zlib.MAX_WBITS)
            except Exception:
                pass
    return raw.decode(encoding, errors='ignore')

# 不校验SSL（部分海外环境证书链有问题）
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_source_status = {}

# 东方财富 push2his 资金流镜像列表（主域名 + 数字镜像，海外环境单个不稳定需轮询）
_PUSH2HIS_MIRRORS = [
    'push2his', '1.push2his', '5.push2his', '10.push2his', '15.push2his',
    '20.push2his', '25.push2his', '30.push2his', '40.push2his', '50.push2his',
    '60.push2his', '70.push2his', '80.push2his', '90.push2his',
]
# 东方财富 push2 实时行情镜像
_PUSH2_MIRRORS = [
    'push2', '1.push2', '5.push2', '10.push2', '20.push2',
    '30.push2', '50.push2', '70.push2', '90.push2',
]


def _record_status(source_name, success, detail=''):
    _source_status[source_name] = {
        'success': success, 'detail': detail,
        'time': datetime.now().strftime('%H:%M:%S')
    }


def get_source_status():
    return dict(_source_status)


def clear_source_status():
    _source_status.clear()


def get_source_status_list():
    """获取数据源状态列表（用于前端展示）"""
    return [
        {'source': name, 'success': st['success'], 'detail': st['detail'], 'time': st['time']}
        for name, st in get_source_status().items()
    ]


def _http_get(url, timeout=10, headers=None, retries=1):
    """统一HTTP GET请求（SSL不校验，自动解压gzip/deflate，支持简单重试），返回解压后的bytes"""
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
    }
    if headers:
        default_headers.update(headers)
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=default_headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                raw = resp.read()
                enc = (resp.headers.get('Content-Encoding') or '').lower()
                # 按响应头解压；若无头但魔数是gzip也解压
                if enc == 'gzip' or raw[:2] == b'\x1f\x8b':
                    raw = gzip.decompress(raw)
                elif enc == 'deflate':
                    try:
                        raw = zlib.decompress(raw)
                    except Exception:
                        raw = zlib.decompress(raw, -zlib.MAX_WBITS)
                return raw
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.4)
    raise last_err


def _get_secid(code):
    """东方财富secid: 沪市1.代码, 深市0.代码"""
    if code.startswith(('60', '68', '90', '11', '13', '51', '58')):
        return f"1.{code}"
    return f"0.{code}"


def _get_em_market(code):
    """东财F10接口市场前缀: SH/SZ"""
    if code.startswith(('60', '68', '90', '11', '13', '51', '58')):
        return 'SH'
    return 'SZ'


def _is_index(code):
    """判断是否为指数代码"""
    return code.startswith(('0000', '3990', '399', '880')) and len(code) == 6


def _retry(func, retries=3, delay=1.0):
    """带重试的函数执行器"""
    last_error = None
    for attempt in range(retries):
        try:
            result = func()
            if result is not None and (not isinstance(result, pd.DataFrame) or len(result) > 0):
                return result
            last_error = Exception("返回空结果")
        except Exception as e:
            last_error = e
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    raise last_error if last_error else Exception("未知错误")


# ============================================================
# 新闻数据源
# ============================================================
def _fetch_news_emweb(code, start_date, end_date, max_items=100):
    """
    东方财富 F10 个股资讯+公告直连API（首选，实测稳定）。
    接口: https://emweb.securities.eastmoney.com/PC_HSF10/NewsBulletin/PageAjax
    返回 gszx(公司资讯) + gsgg(公司公告)
    """
    try:
        # 指数没有F10资讯，直接返回空
        if _is_index(code):
            _record_status('东财F10资讯', False, '指数无个股资讯')
            return pd.DataFrame()

        market_code = f"{_get_em_market(code)}{code}"
        url = (f"https://emweb.securities.eastmoney.com/PC_HSF10/NewsBulletin/PageAjax?"
               f"code={market_code}")
        t0 = time.time()
        raw = _http_get(url, timeout=12, headers={'Referer': 'https://emweb.securities.eastmoney.com/'})
        data = json.loads(raw.decode('utf-8'))
        elapsed = time.time() - t0

        from news_analysis import classify_news_type, classify_news_timing
        results = []

        # 1. 公司资讯 gszx.data.items（时间为毫秒时间戳）
        try:
            items = data.get('gszx', {}).get('data', {}).get('items', [])
            for art in items:
                try:
                    ts = art.get('showDateTime')
                    dt = datetime.fromtimestamp(int(ts) / 1000) if ts else pd.to_datetime(art.get('publishDate', ''))
                    title = str(art.get('title', '')).strip()
                    if not title:
                        continue
                    results.append({
                        'date': dt, 'title': title,
                        'url': art.get('uniqueUrl') or art.get('url', ''),
                        'source': '东财资讯',
                        'news_type': classify_news_type(title),
                        'timing': classify_news_timing(dt),
                        'category': '资讯',
                    })
                except Exception:
                    continue
        except Exception:
            pass

        # 2. 公司公告 gsgg（直接是list，时间为字符串 notice_date/display_time）
        try:
            anns = data.get('gsgg', [])
            if isinstance(anns, list):
                for art in anns:
                    try:
                        time_str = art.get('notice_date') or art.get('display_time', '')
                        dt = pd.to_datetime(str(time_str).split(':')[0] if ':' in str(time_str) else time_str)
                        title = str(art.get('title', '')).strip()
                        if not title:
                            continue
                        art_code = art.get('art_code', '')
                        ann_url = f"https://data.eastmoney.com/notices/detail/{code}/{art_code}.html" if art_code else ''
                        results.append({
                            'date': dt, 'title': title, 'url': ann_url,
                            'source': '东财公告',
                            'news_type': classify_news_type(title),
                            'timing': classify_news_timing(dt),
                            'category': '公告',
                        })
                    except Exception:
                        continue
        except Exception:
            pass

        if not results:
            _record_status('东财F10资讯', False, f'返回空, {elapsed:.1f}s')
            return pd.DataFrame()

        df = pd.DataFrame(results).drop_duplicates(subset=['title'])
        # 日期过滤
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date) + timedelta(days=1)
        df = df[(df['date'] >= s) & (df['date'] <= e)]
        df = df.sort_values('date').reset_index(drop=True)

        if len(df) > 0:
            _record_status('东财F10资讯', True, f'{len(df)}条(资讯+公告), {elapsed:.1f}s')
            return df.tail(max_items)
        _record_status('东财F10资讯', False, f'日期过滤后空, {elapsed:.1f}s')
        return pd.DataFrame()
    except Exception as e:
        _record_status('东财F10资讯', False, f'{type(e).__name__}: {str(e)[:80]}')
        return pd.DataFrame()


def _fetch_news_search_api(code, start_date, end_date, max_items=100):
    """东方财富搜索API（备选2，按代码搜索，部分股票可能为空）"""
    try:
        param = {
            "uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
            "client": "web", "clientType": "web", "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {
                "searchScope": "default", "sort": "time",
                "pageIndex": 1, "pageSize": max_items, "preTag": "", "postTag": ""
            }}
        }
        encoded_param = urllib.parse.quote(json.dumps(param, ensure_ascii=False))
        url = f"https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&param={encoded_param}"
        t0 = time.time()
        text = _http_get(url, timeout=10, headers={'Referer': 'https://so.eastmoney.com/'}).decode('utf-8')
        start = text.find('(') + 1
        end = text.rfind(')')
        if start <= 0 or end <= start:
            raise ValueError("JSONP解析失败")
        data = json.loads(text[start:end])
        articles = data.get('result', {}).get('cmsArticleWebOld', [])
        if not isinstance(articles, list) or not articles:
            _record_status('东财搜索资讯', False, f'返回空, {time.time()-t0:.1f}s')
            return pd.DataFrame()

        from news_analysis import classify_news_type, classify_news_timing
        results = []
        for art in articles:
            try:
                dt = pd.to_datetime(art.get('date', ''))
                title = str(art.get('title', '')).replace('<em>', '').replace('</em>', '').strip()
                results.append({
                    'date': dt, 'title': title, 'url': art.get('url', ''),
                    'source': '东财搜索', 'news_type': classify_news_type(title),
                    'timing': classify_news_timing(dt), 'category': '资讯',
                })
            except Exception:
                continue
        if not results:
            _record_status('东财搜索资讯', False, '解析后空')
            return pd.DataFrame()
        df = pd.DataFrame(results).drop_duplicates(subset=['title'])
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date) + timedelta(days=1)
        df = df[(df['date'] >= s) & (df['date'] <= e)].sort_values('date').reset_index(drop=True)
        if len(df):
            _record_status('东财搜索资讯', True, f'{len(df)}条')
            return df.tail(max_items)
        _record_status('东财搜索资讯', False, '过滤后空')
        return pd.DataFrame()
    except Exception as e:
        _record_status('东财搜索资讯', False, f'{type(e).__name__}: {str(e)[:60]}')
        return pd.DataFrame()


def _fetch_news_akshare(code, start_date, end_date, max_items=100):
    """akshare东方财富新闻（兜底3，容错ArrowInvalid等版本问题）"""
    try:
        import akshare as ak
        t0 = time.time()
        df = ak.stock_news_em(symbol=code)
        elapsed = time.time() - t0
        if df is None or len(df) == 0:
            _record_status('akshare新闻', False, f'返回空, {elapsed:.1f}s')
            return pd.DataFrame()
        from news_analysis import classify_news_type, classify_news_timing
        df = df.rename(columns={'发布时间': 'date', '新闻标题': 'title',
                                '新闻链接': 'url', '文章来源': 'source'})
        results = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row['date'])
                title = str(row['title'])
                results.append({
                    'date': dt, 'title': title, 'url': str(row.get('url', '')),
                    'source': 'akshare', 'news_type': classify_news_type(title),
                    'timing': classify_news_timing(dt), 'category': '资讯',
                })
            except Exception:
                continue
        if not results:
            _record_status('akshare新闻', False, f'解析后空, {elapsed:.1f}s')
            return pd.DataFrame()
        result_df = pd.DataFrame(results).drop_duplicates(subset=['title'])
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date) + timedelta(days=1)
        result_df = result_df[(result_df['date'] >= s) & (result_df['date'] <= e)]
        result_df = result_df.sort_values('date').reset_index(drop=True)
        if len(result_df):
            _record_status('akshare新闻', True, f'{len(result_df)}条, {elapsed:.1f}s')
            return result_df.tail(max_items)
        _record_status('akshare新闻', False, f'过滤后空, {elapsed:.1f}s')
        return pd.DataFrame()
    except Exception as e:
        # 特别标注 ArrowInvalid（pyarrow版本不兼容）问题
        ename = type(e).__name__
        detail = str(e)[:80]
        if 'ArrowInvalid' in ename or 'escape sequence' in detail:
            detail = 'pyarrow版本不兼容(建议pip install "pyarrow<15"): ' + detail
        _record_status('akshare新闻', False, f'{ename}: {detail}')
        return pd.DataFrame()


def fetch_news(code, start_date, end_date, max_items=100):
    """
    多源获取个股新闻/公告：
      东财F10资讯(首选) → 东财搜索(备选) → akshare(兜底)
    """
    clear_source_status()

    # 源1：东财F10（资讯+公告）
    df = _fetch_news_emweb(code, start_date, end_date, max_items)
    if len(df) > 0:
        return df

    # 源2：东财搜索API
    df = _fetch_news_search_api(code, start_date, end_date, max_items)
    if len(df) > 0:
        return df

    # 源3：akshare兜底
    df = _fetch_news_akshare(code, start_date, end_date, max_items)
    if len(df) > 0:
        return df

    _record_status('全部新闻源', False, '3个源均失败')
    return pd.DataFrame(columns=['date', 'title', 'url', 'source', 'news_type', 'timing', 'category'])


# ============================================================
# 资金流数据源
# ============================================================
def _request_mirrors_json(mirrors, path_suffix, protocol='http', timeout=6, rounds=2):
    """
    东财多镜像轮询请求JSON（核心容错机制）。
    单个镜像在海外环境不稳定（约30-60%成功率），遍历多个镜像可保证总体成功。
    mirrors: 镜像前缀列表; path_suffix: 域名后的路径
    返回: (json数据, 成功的host, 尝试次数)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'http://data.eastmoney.com/',
        'Connection': 'close',
    }
    attempts = 0
    last_err = None
    for rnd in range(rounds):
        for prefix in mirrors:
            host = f"{prefix}.eastmoney.com"
            attempts += 1
            try:
                url = f"{protocol}://{host}{path_suffix}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                    raw = resp.read()
                    enc = (resp.headers.get('Content-Encoding') or '').lower()
                    if enc == 'gzip' or raw[:2] == b'\x1f\x8b':
                        raw = gzip.decompress(raw)
                    elif enc == 'deflate':
                        try:
                            raw = zlib.decompress(raw)
                        except Exception:
                            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
                    text = raw.decode('utf-8', errors='ignore')
                    data = json.loads(text)
                    if data is not None:
                        return data, host, attempts
            except Exception as e:
                last_err = e
                continue
        time.sleep(0.3)
    raise last_err if last_err else Exception("所有镜像均失败")


# ---------- 雪球资金流（全球CDN，海外/美国服务器可达，作为东财被封时的可靠信源）----------
import http.cookiejar
_XQ_STATE = {'opener': None, 'token_time': 0, 'last_req': 0}
_XQ_UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
_XQ_TOKEN_TTL = 1200      # token 缓存20分钟
_XQ_MIN_INTERVAL = 0.8    # 两次请求最小间隔（秒），规避限流


def _xq_symbol(code):
    """A股个股代码转雪球代码：沪市(60/68/90及沪基金/债)SH，其余(000/002/300等)SZ。
    注：资金流仅针对个股；指数(如000001上证指数)无主力资金流概念，故此处000默认按深市个股处理。"""
    c = str(code).zfill(6)
    if c.startswith(('60', '68', '90', '11', '13', '50', '51', '56', '58')):
        return 'SH' + c
    return 'SZ' + c


def _xq_ensure_token(force=False):
    """获取/复用雪球 cookie token（xq_a_token），带缓存"""
    now = time.time()
    if (not force and _XQ_STATE['opener'] is not None
            and now - _XQ_STATE['token_time'] < _XQ_TOKEN_TTL):
        return _XQ_STATE['opener']
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=_SSL_CTX))
    opener.addheaders = [('User-Agent', _XQ_UA), ('Accept-Language', 'zh-CN,zh;q=0.9')]
    # 访问行情页，Set-Cookie 会下发 xq_a_token
    opener.open('https://xueqiu.com/hq', timeout=10).read()
    has_token = any('token' in c.name for c in cj)
    if not has_token:
        # 再访问个股页兜底
        time.sleep(0.6)
        opener.open('https://xueqiu.com/S/' + _xq_symbol('600118'), timeout=10).read()
    _XQ_STATE['opener'] = opener
    _XQ_STATE['token_time'] = now
    _XQ_STATE['last_req'] = now
    return opener


def _xq_get_json(path, tries=3):
    """带限速、退避重试、token自动刷新的雪球JSON GET"""
    last_err = None
    for i in range(tries):
        try:
            # 限速
            wait = _XQ_MIN_INTERVAL - (time.time() - _XQ_STATE['last_req'])
            if wait > 0:
                time.sleep(wait)
            opener = _xq_ensure_token(force=(i == 1))  # 第2次尝试时强制刷新token
            req = urllib.request.Request(
                'https://stock.xueqiu.com' + path,
                headers={'User-Agent': _XQ_UA, 'Accept': 'application/json',
                         'Referer': 'https://xueqiu.com/'})
            _XQ_STATE['last_req'] = time.time()
            raw = opener.open(req, timeout=10).read()
            if raw[:2] == b'\x1f\x8b':
                raw = gzip.decompress(raw)
            text = raw.decode('utf-8', errors='ignore')
            # 限流时可能返回空或HTML登录页
            if not text.strip().startswith('{'):
                raise ValueError('非JSON响应(可能限流)')
            data = json.loads(text)
            if data and data.get('data') is not None:
                return data['data']
            raise ValueError('data为空')
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))  # 退避
    raise last_err if last_err else RuntimeError('雪球请求失败')


def _fetch_flow_xueqiu(code, start_date, end_date):
    """
    雪球资金流（全球CDN，美国服务器可达）。
    history.json: 最近20个交易日 主力净流入(amount,元) 的日级序列；
    assort.json : 最新交易日 大/中/小单买卖结构（雪球为3档，无独立超大单）。
    注意：雪球"主力"口径(大单阈值)与东财5档不同，故用 flow_source 列标注，不与东财混用。
    """
    try:
        sym = _xq_symbol(code)
        t0 = time.time()
        hist = _xq_get_json(f'/v5/stock/capital/history.json?symbol={sym}&size=20')
        items = hist.get('items', []) if hist else []
        if not items:
            _record_status('雪球资金流', False, 'history返回空')
            return pd.DataFrame()

        # 最新交易日的分档结构（可选，失败不影响主序列）
        breakdown = {}
        try:
            time.sleep(_XQ_MIN_INTERVAL)
            asrt = _xq_get_json(f'/v5/stock/capital/assort.json?symbol={sym}', tries=2)
            if asrt:
                bd_date = pd.to_datetime(asrt.get('timestamp'), unit='ms').normalize()
                breakdown = {
                    'date': bd_date,
                    'large_net': (asrt.get('buy_large') or 0) - (asrt.get('sell_large') or 0),
                    'medium_net': (asrt.get('buy_medium') or 0) - (asrt.get('sell_medium') or 0),
                    'small_net': (asrt.get('buy_small') or 0) - (asrt.get('sell_small') or 0),
                }
        except Exception:
            breakdown = {}

        records = []
        for it in items:
            dt = pd.to_datetime(it.get('timestamp'), unit='ms').normalize()
            row = {
                'date': dt,
                'main_net_inflow': float(it.get('amount') or 0),
                'small_net': np.nan, 'medium_net': np.nan,
                'large_net': np.nan, 'super_large_net': np.nan,
                'main_net_pct': np.nan, 'close': np.nan,
            }
            if breakdown and dt == breakdown['date']:
                row['large_net'] = breakdown['large_net']
                row['medium_net'] = breakdown['medium_net']
                row['small_net'] = breakdown['small_net']
            records.append(row)

        df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
        df = df[(df['date'] >= s) & (df['date'] <= e)].reset_index(drop=True)
        df['flow_source'] = '雪球(全球CDN·近20交易日)'
        if len(df):
            _record_status('雪球资金流', True, f'{len(df)}天(近20交易日), {time.time()-t0:.1f}s')
            return df
        _record_status('雪球资金流', False, '日期过滤后空')
        return pd.DataFrame()
    except Exception as ex:
        _record_status('雪球资金流', False, f'{type(ex).__name__}: {str(ex)[:70]}')
        return pd.DataFrame()


def _fetch_flow_eastmoney_direct(code, start_date, end_date):
    """
    东方财富个股资金流直连API（多镜像HTTP轮询）。
    字段: f51日期 f52主力净 f53小单 f54中单 f55大单 f56超大单 f57-f61各类占比 ...
    注意：东财 push2his 域名在美国数据中心可能被断开(RemoteDisconnected)，故仅作信源之一。
    """
    try:
        secid = _get_secid(code)
        fields2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65"
        path = (f"/api/qt/stock/fflow/daykline/get?secid={secid}"
                f"&fields1=f1,f2,f3,f7&fields2={fields2}&lmt=0&klt=101")
        t0 = time.time()
        data, host, attempts = _request_mirrors_json(_PUSH2HIS_MIRRORS, path, timeout=6, rounds=2)
        elapsed = time.time() - t0

        klines = (data.get('data') or {}).get('klines', [])
        if not klines:
            _record_status('东财直连资金流', False, f'返回空(尝试{attempts}镜像), {elapsed:.1f}s')
            return pd.DataFrame()

        records = []
        for line in klines:
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                dt = pd.to_datetime(parts[0])
                records.append({
                    'date': dt,
                    'main_net_inflow': float(parts[1]),
                    'small_net': float(parts[2]),
                    'medium_net': float(parts[3]),
                    'large_net': float(parts[4]),
                    'super_large_net': float(parts[5]),
                    'main_net_pct': float(parts[6]) if len(parts) > 6 and parts[6] not in ('-', '') else 0.0,
                    'close': float(parts[11]) if len(parts) > 11 and parts[11] not in ('-', '') else 0.0,
                })
            except Exception:
                continue

        if not records:
            _record_status('东财直连资金流', False, f'解析失败, {elapsed:.1f}s')
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
        s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
        df = df[(df['date'] >= s) & (df['date'] <= e)].reset_index(drop=True)
        if len(df):
            df['flow_source'] = '东方财富(5档)'
            _record_status('东财直连资金流', True,
                           f'{len(df)}天 via {host.split(".")[0]}(试{attempts}), {elapsed:.1f}s')
            return df
        _record_status('东财直连资金流', False, f'日期过滤后空(试{attempts}), {elapsed:.1f}s')
        return pd.DataFrame()
    except Exception as e:
        _record_status('东财直连资金流', False, f'{type(e).__name__}: {str(e)[:80]}')
        return pd.DataFrame()


def _fetch_flow_akshare(code, start_date, end_date):
    """akshare资金流（兜底）"""
    try:
        import akshare as ak
        market = 'sh' if code.startswith(('60', '68', '90')) else 'sz'
        t0 = time.time()
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        elapsed = time.time() - t0
        if df is None or len(df) == 0:
            _record_status('akshare资金流', False, f'返回空, {elapsed:.1f}s')
            return pd.DataFrame()
        col_map = {}
        for col in df.columns:
            cs = str(col)
            if '日期' in cs: col_map[col] = 'date'
            elif '主力净流入-净额' in cs: col_map[col] = 'main_net_inflow'
            elif '超大单净流入-净额' in cs: col_map[col] = 'super_large_net'
            elif '大单净流入-净额' in cs: col_map[col] = 'large_net'
            elif '中单净流入-净额' in cs: col_map[col] = 'medium_net'
            elif '小单净流入-净额' in cs: col_map[col] = 'small_net'
            elif '主力净流入-净占比' in cs: col_map[col] = 'main_net_pct'
            elif '收盘价' in cs: col_map[col] = 'close'
        df = df.rename(columns=col_map)
        if 'date' not in df.columns:
            _record_status('akshare资金流', False, '缺少date列')
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        for c in ['main_net_inflow', 'super_large_net', 'large_net', 'medium_net', 'small_net', 'main_net_pct', 'close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask].sort_values('date').reset_index(drop=True)
        if len(df) > 0:
            df['flow_source'] = 'akshare(东财口径)'
            _record_status('akshare资金流', True, f'{len(df)}天, {elapsed:.1f}s')
        else:
            _record_status('akshare资金流', False, f'过滤后空, {elapsed:.1f}s')
        return df
    except Exception as e:
        _record_status('akshare资金流', False, f'{type(e).__name__}: {str(e)[:80]}')
        return pd.DataFrame()


def fetch_capital_flow(code, start_date, end_date):
    """
    多源获取个股资金流，按可达性/质量依次降级：
      源1 东财push2his镜像（5档完整历史，国内最优；美国被封时快速失败）
      源2 雪球Xueqiu（全球CDN，美国服务器可达；近20交易日主力净流入+最新分档）
      源3 akshare（兜底）
    任一源成功即返回，结果带 flow_source 列标注口径；全失败返回空DataFrame（不阻断主流程）。
    """
    clear_source_status()
    # 源1：东财多镜像直连（RemoteDisconnected为即时失败，不会长时间卡住；外层只重试1次）
    try:
        df = _retry(lambda: _fetch_flow_eastmoney_direct(code, start_date, end_date),
                    retries=1, delay=0.6)
        if len(df) > 0:
            return df
    except Exception:
        pass
    # 源2：雪球（全球CDN，海外/美国可达）
    try:
        df = _fetch_flow_xueqiu(code, start_date, end_date)
        if len(df) > 0:
            return df
    except Exception:
        pass
    # 源3：akshare兜底
    df = _fetch_flow_akshare(code, start_date, end_date)
    if len(df) > 0:
        return df
    _record_status('全部资金流源', False, '3个源均失败（东财/雪球/akshare）')
    return pd.DataFrame()


# ============================================================
# 实时行情数据源
# ============================================================
def fetch_realtime_quote(code):
    """多源获取实时行情：腾讯财经直连 → 东方财富多镜像直连"""
    clear_source_status()
    # 源1: 腾讯财经直连
    try:
        market = 'sh' if code.startswith(('60', '68', '90', '11', '13', '51', '58')) else 'sz'
        url = f"http://qt.gtimg.cn/q={market}{code}"
        raw = _http_get(url, timeout=8)
        text = raw.decode('gbk', errors='ignore')
        parts = text.split('~')
        if len(parts) > 37 and parts[3]:
            result = {
                'price': float(parts[3]), 'open': float(parts[5]) if parts[5] else 0,
                'high': float(parts[33]) if parts[33] else 0,
                'low': float(parts[34]) if parts[34] else 0,
                'volume': float(parts[6]) if parts[6] else 0,
                'amount': float(parts[37]) if len(parts) > 37 and parts[37] else 0,
                'change_pct': float(parts[32]) if parts[32] else 0, 'source': '腾讯财经'
            }
            _record_status('腾讯实时行情', True, f'¥{result["price"]}')
            return result
        _record_status('腾讯实时行情', False, '解析失败')
    except Exception as e:
        _record_status('腾讯实时行情', False, f'{type(e).__name__}: {str(e)[:60]}')

    # 源2: 东方财富多镜像直连
    try:
        secid = _get_secid(code)
        path = (f"/api/qt/stock/get?secid={secid}"
                f"&fields=f43,f44,f45,f46,f47,f48,f57,f58,f170")
        data, host, _ = _request_mirrors_json(_PUSH2_MIRRORS, path, protocol='http', timeout=6, rounds=1)
        d = data.get('data', {})
        if d and d.get('f43'):
            # 东财价格单位为厘（×100），需除以100
            result = {
                'price': float(d['f43']) / 100,
                'open': float(d.get('f46', 0)) / 100,
                'high': float(d.get('f44', 0)) / 100,
                'low': float(d.get('f45', 0)) / 100,
                'volume': float(d.get('f47', 0)),
                'amount': float(d.get('f48', 0)),
                'change_pct': float(d.get('f170', 0)) / 100, 'source': '东方财富'
            }
            _record_status('东财实时行情', True, f'¥{result["price"]} via {host.split(".")[0]}')
            return result
        _record_status('东财实时行情', False, '无数据')
    except Exception as e:
        _record_status('东财实时行情', False, f'{type(e).__name__}: {str(e)[:60]}')

    _record_status('全部实时行情源', False, '2个源均失败')
    return None
