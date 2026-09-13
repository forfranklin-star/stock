#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股股票分析核心逻辑模块 v2.0
================================
增强功能：
  - 55+ 技术指标（交易员常用指标全覆盖）
  - 多周期支持：日线/60分钟/120分钟/30分钟/15分钟/周线
  - 大盘指数分析：上证/深证/创业板/沪深300/中证500等
  - 改进数学模型：全指标相关性扫描 → 自动筛选 → 淘汰低相关指标
  - 模型灵敏度分级：保守/均衡/激进
  - 模型命名存储与加载
被 Flask (app.py) 和 Streamlit (streamlit_app.py) 共同复用。
"""

import os
import json
import time
import hashlib
import warnings
from datetime import datetime, timedelta
from functools import wraps
from http.client import RemoteDisconnected

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 回测参数
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.00025
STAMP_TAX_RATE = 0.001
MIN_COMMISSION = 5.0

# 支持的周期
TIMEFRAMES = {
    'daily':   {'name': '日线',   'ak_period': 'daily',  'min_em': None,  'warmup': 60},
    '60min':   {'name': '60分钟', 'ak_period': None,     'min_em': '60',  'warmup': 60},
    '120min':  {'name': '120分钟','ak_period': None,     'min_em': '120', 'warmup': 60},
    '30min':   {'name': '30分钟', 'ak_period': None,     'min_em': '30',  'warmup': 60},
    '15min':   {'name': '15分钟', 'ak_period': None,     'min_em': '15',  'warmup': 60},
    'weekly':  {'name': '周线',   'ak_period': 'weekly', 'min_em': None,  'warmup': 60},
}

# 大盘指数代码映射
INDEX_MAP = {
    '000001': {'name': '上证指数', 'symbol': '000001', 'market': 'sh'},
    '399001': {'name': '深证成指', 'symbol': '399001', 'market': 'sz'},
    '399006': {'name': '创业板指', 'symbol': '399006', 'market': 'sz'},
    '000300': {'name': '沪深300',  'symbol': '000300', 'market': 'sh'},
    '000905': {'name': '中证500',  'symbol': '000905', 'market': 'sh'},
    '000688': {'name': '科创50',   'symbol': '000688', 'market': 'sh'},
    '000016': {'name': '上证50',   'symbol': '000016', 'market': 'sh'},
    '000852': {'name': '中证1000', 'symbol': '000852', 'market': 'sh'},
    '399005': {'name': '中小板指', 'symbol': '399005', 'market': 'sz'},
}

# 全部技术指标列（55个）
INDICATOR_COLS = [
    # 均线系统 (11)
    'MA5', 'MA10', 'MA20', 'MA60', 'MA120',
    'EMA5', 'EMA10', 'EMA20', 'EMA60', 'EMA120', 'WMA20',
    # MACD (3)
    'DIF', 'DEA', 'MACD',
    # RSI (3)
    'RSI6', 'RSI14', 'RSI24',
    # KDJ (3)
    'K', 'D', 'J',
    # 布林带 (5)
    'BOLL_MID', 'BOLL_UP', 'BOLL_LOW', 'BOLL_PCTB', 'BOLL_BW',
    # 量能 (4)
    'VOL_MA5', 'VOL_MA10', 'VOL_RATE', 'VOL_RATIO',
    # 能量/资金流 (4)
    'OBV', 'AD_LINE', 'MFI14', 'VWAP',
    # 超买超卖 (3)
    'WR', 'CCI', 'STOCHRSI',
    # 趋势 (6)
    'ADX14', 'PDI14', 'MDI14', 'SAR', 'TRIX12', 'DMA',
    # 波动率 (4)
    'ATR14', 'NATR14', 'STD20', 'DONCHIAN_W',
    # 动量/变化率 (4)
    'ROC12', 'MOM10', 'CMO14', 'ULTOSC',
    # 乖离/心理 (5)
    'BIAS5', 'BIAS10', 'BIAS20', 'BIAS60', 'PSY12',
    # 一目均衡 (4)
    'TENKAN', 'KIJUN', 'SENKOU_A', 'SENKOU_B',
    # 价格位置 (2)
    'CLOSE_MA20_RATIO', 'CLOSE_MA60_RATIO',
]


# ============================================================
# 磁盘缓存装饰器
# ============================================================
def cache_pickle(expire_seconds=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = hashlib.md5(
                f"{func.__name__}|{args}|{sorted(kwargs.items())}".encode()
            ).hexdigest()
            cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
            if os.path.exists(cache_file):
                age = time.time() - os.path.getmtime(cache_file)
                if age < expire_seconds:
                    try:
                        return pd.read_pickle(cache_file)
                    except Exception:
                        pass
            result = func(*args, **kwargs)
            try:
                result.to_pickle(cache_file)
            except Exception:
                pass
            return result
        return wrapper
    return decorator


# ============================================================
# 网络层：超时 + 重试 + 浏览器UA
# ============================================================
def _set_default_timeout(timeout=30):
    import requests
    if not hasattr(requests, '_original_get'):
        requests._original_get = requests.get
        requests._original_session_get = requests.Session.get
    def _get(url, **kw):
        kw.setdefault('timeout', timeout)
        return requests._original_get(url, **kw)
    def _sget(self, url, **kw):
        kw.setdefault('timeout', timeout)
        return requests._original_session_get(self, url, **kw)
    requests.get = _get
    requests.Session.get = _sget


def _setup_requests_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET","POST"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    })
    return session


def _fetch_with_retry(fetch_func, max_retries=3, base_delay=2):
    import requests
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fetch_func()
            if result is not None and (not isinstance(result, pd.DataFrame) or len(result) > 0):
                return result
            last_error = ValueError("返回数据为空")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError, ConnectionError, TimeoutError,
                RemoteDisconnected, OSError) as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"[数据获取] 第{attempt}次失败 ({type(e).__name__})，{delay}秒后重试...")
                time.sleep(delay)
            continue
        except Exception:
            raise
    raise ConnectionError(f"重试{max_retries}次后仍失败: {type(last_error).__name__}: {str(last_error)[:150]}")


def _standardize_dataframe(df, source_name):
    col_map = {
        '日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low',
        '成交量':'volume','成交额':'amount','涨跌幅':'pct_change','换手率':'turnover',
        'date':'date','open':'open','high':'high','low':'low','close':'close','volume':'volume',
        'outstanding_share':'amount',
        'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume',
        'Adj Close':'adj_close','Dividends':'dividends','Stock Splits':'splits',
    }
    df = df.rename(columns=col_map)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=col_map)
    if 'date' not in df.columns and df.index.name in ('Date','date',None):
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]:'date'})
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # 对于日线数据，去掉时间部分
    if all(' 00:00:00' in str(d) for d in df['date'].head(10)):
        df['date'] = df['date'].str[:10]
    for col in ['open','high','low','close','volume']:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
    keep = ['date','open','high','low','close','volume','amount','pct_change']
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values('date').drop_duplicates(subset='date').reset_index(drop=True)
    # 关键：丢弃OHLC缺失的行，绝不用相邻价格填充（避免伪造数据）
    before = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[数据标准化] 警告: 丢弃{dropped}条OHLC缺失的无效数据（不填充）")
    # volume为0或NaN的行保留（停牌等情况），但amount/pct_change缺失用0填充（非价格字段）
    if 'amount' in df.columns:
        df['amount'] = df['amount'].fillna(0)
    if 'pct_change' in df.columns:
        df['pct_change'] = df['pct_change'].fillna(0)
    print(f"[数据标准化] 数据源={source_name}, 获得{len(df)}条有效数据")
    return df


# ============================================================
# 数据源：AkShare-东方财富（支持日线/周线/分钟线）
# ============================================================
def _fetch_akshare_eastmoney(code, start_date, end_date, timeframe='daily', is_index=False):
    import akshare as ak
    s = start_date.replace('-', '')
    e = end_date.replace('-', '')
    tf = TIMEFRAMES.get(timeframe, TIMEFRAMES['daily'])

    if is_index:
        # 大盘指数：使用 index_zh_a_hist
        df = ak.index_zh_a_hist(symbol=code, period=tf['ak_period'] or 'daily',
                                 start_date=s, end_date=e)
    elif tf['ak_period']:
        # 日线/周线：stock_zh_a_hist
        df = ak.stock_zh_a_hist(symbol=code, period=tf['ak_period'],
                                 start_date=s, end_date=e, adjust="qfq")
    else:
        # 分钟线：stock_zh_a_hist_min_em
        df = ak.stock_zh_a_hist_min_em(symbol=code, period=tf['min_em'],
                                        start_date=start_date + ' 09:30:00',
                                        end_date=end_date + ' 15:00:00', adjust="qfq")
    return _standardize_dataframe(df, f"akshare-eastmoney-{timeframe}")


# ============================================================
# 数据源：AkShare-新浪财经（备用）
# ============================================================
def _fetch_akshare_sina(code, start_date, end_date, timeframe='daily', is_index=False):
    import akshare as ak
    if timeframe != 'daily' or is_index:
        raise ValueError("新浪数据源仅支持个股日线")
    prefix = 'sh' if code.startswith('6') else 'sz'
    symbol = f"{prefix}{code}"
    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
    return _standardize_dataframe(df, "akshare-sina")


# ============================================================
# 数据源：Yahoo Finance（海外终极回退）
# ============================================================
def _to_yfinance_ticker(code, is_index=False):
    """
    转换为 Yahoo Finance ticker 格式。
    中国个股和指数统一使用 .SS(沪市)/.SZ(深市)/.BJ(北交所) 后缀。
    注意：^ 前缀仅用于美股指数（如^GSPC），中国指数不用 ^。
    """
    if is_index:
        # 中国指数：深市(399开头)用.SZ，其余(沪市)用.SS
        if code.startswith('399'):
            return f"{code}.SZ"
        return f"{code}.SS"
    if code.startswith('6'):
        return f"{code}.SS"
    elif code.startswith(('0','3')):
        return f"{code}.SZ"
    elif code.startswith(('4','8')):
        return f"{code}.BJ"
    return f"{code}.SS"


def _resample_ohlc(df, target_freq):
    """
    将OHLCV分钟数据重采样到目标周期（如60min→120min）。
    open取首值, high取最大值, low取最小值, close取末值, volume求和。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if 'amount' in df.columns:
        agg_dict['amount'] = 'sum'
    resampled = df.resample(target_freq).agg(agg_dict).dropna(subset=['open', 'close'])
    resampled = resampled.reset_index()
    return resampled


def _fetch_yfinance(code, start_date, end_date, timeframe='daily', is_index=False):
    """
    通过 Yahoo Finance 获取数据（海外服务器最稳定的数据源）。
    支持：日线、周线、60分钟、30分钟、15分钟、120分钟（由60分钟重采样）。

    yfinance 分钟数据可用范围：
      - 60m/1h: 最近730天（约2年）
      - 30m/15m: 最近60天
      - 120m: 由60m重采样，同样2年范围
    """
    import yfinance as yf

    ticker = _to_yfinance_ticker(code, is_index)
    end_inclusive = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    # 周期 → yfinance interval 映射
    yf_interval_map = {
        'daily': '1d', 'weekly': '1wk',
        '60min': '60m', '30min': '30m', '15min': '15m',
        '120min': '60m',  # 120分钟由60分钟重采样
    }
    interval = yf_interval_map.get(timeframe, '1d')

    # 分钟数据用 period 参数获取最大可用范围，然后按日期过滤
    if timeframe in ('60min', '120min'):
        period = '2y'
    elif timeframe in ('30min', '15min'):
        period = '60d'
    else:
        period = None

    if period:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, threads=False)
    else:
        df = yf.download(ticker, start=start_date, end=end_inclusive, interval=interval,
                         auto_adjust=True, progress=False, threads=False)

    if df is None or len(df) == 0:
        raise ValueError(f"Yahoo Finance 未找到 {ticker} ({interval}) 的数据")

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 日线索引列名是Date，分钟线是Datetime
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={date_col: 'date', 'Open': 'open', 'High': 'high',
                             'Low': 'low', 'Close': 'close', 'Volume': 'volume'})

    # 120分钟：由60分钟数据重采样
    if timeframe == '120min':
        df = _resample_ohlc(df, '120min')
        if len(df) == 0:
            raise ValueError("120分钟重采样后无数据")

    # 按用户请求的日期范围过滤
    df['date'] = pd.to_datetime(df['date'])
    # 移除时区信息（Yahoo Finance分钟线返回带时区的datetime，如Asia/Shanghai，
    # 与无时区的start_date比较会报TypeError）
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_inclusive))
    df = df[mask].copy()
    if len(df) == 0:
        raise ValueError(f"Yahoo Finance {timeframe} 数据在 {start_date}~{end_date} 范围内为空"
                         f"（分钟数据仅支持最近2年，30m/15m仅支持最近60天）")

    # 统一日期时间格式
    if timeframe in ('daily', 'weekly'):
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    else:
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return _standardize_dataframe(df, f"yfinance-{timeframe}")


def _fetch_efinance(code, start_date, end_date, timeframe='daily', is_index=False):
    """
    通过 efinance（同花顺数据源）获取数据。
    支持日线/周线及各分钟周期，前复权。
    作为 AkShare 和 Yahoo Finance 之间的中间数据源。
    """
    import efinance as ef

    # efinance K线类型: 101=日, 102=周, 103=月, 5=5分, 15=15分, 30=30分, 60=60分
    klt_map = {'daily': 101, 'weekly': 102, 'monthly': 103,
               '5min': 5, '15min': 15, '30min': 30, '60min': 60}
    klt = klt_map.get(timeframe)

    if timeframe == '120min':
        klt = 60  # 用60分钟获取后重采样

    if klt is None:
        raise ValueError(f"efinance 不支持周期: {timeframe}")

    s = start_date.replace('-', '')
    e = end_date.replace('-', '')
    fqt = 0 if is_index else 1  # 指数不复权

    try:
        df = ef.stock.get_quote_history(code, beg=s, end=e, klt=klt, fqt=fqt)
    except Exception:
        df = ef.stock.get_quote_history(code, beg=s, end=e, klt=klt)

    if df is None or len(df) == 0:
        raise ValueError(f"efinance(同花顺) 返回空数据: {code} {timeframe}")

    col_map = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high',
                '最低': 'low', '成交量': 'volume', '成交额': 'amount',
                '涨跌幅': 'pct_change', '换手率': 'turnover'}
    df = df.rename(columns=col_map)

    if timeframe == '120min' and 'date' in df.columns:
        df = _resample_ohlc(df, '120min')

    return _standardize_dataframe(df, f"efinance-{timeframe}")


def _fetch_baostock(code, start_date, end_date, timeframe='daily', is_index=False):
    """
    通过 baostock 获取数据（TCP协议，非HTTP，海外服务器可能可访问）。
    支持日线/周线/月线/5分钟/15分钟/30分钟/60分钟，前复权。
    120分钟由60分钟重采样。
    """
    import baostock as bs

    # baostock 代码格式: sh.600519, sz.000001
    if is_index:
        bs_code = f"sz.{code}" if code.startswith('399') else f"sh.{code}"
    else:
        if code.startswith(('60', '68', '90', '11', '13', '51', '58')):
            bs_code = f"sh.{code}"
        else:
            bs_code = f"sz.{code}"

    # 周期映射
    freq_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm',
                '5min': '5', '15min': '15', '30min': '30', '60min': '60'}
    frequency = freq_map.get(timeframe, '60' if timeframe == '120min' else None)
    if frequency is None:
        raise ValueError(f"baostock 不支持周期: {timeframe}")

    # 复权: 1=后复权, 2=前复权, 3=不复权; 指数不需要复权
    adjustflag = "3" if is_index else "2"

    lg = bs.login()
    if lg.error_code != '0':
        raise ConnectionError(f"baostock 登录失败: {lg.error_msg}")

    try:
        # 分钟线不支持pctChg字段；日线/周线不需要time字段
        is_minute = timeframe not in ('daily', 'weekly', 'monthly')
        if is_minute:
            fields = "date,time,open,high,low,close,volume,amount"
        else:
            fields = "date,open,high,low,close,volume,amount,pctChg"

        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date=start_date, end_date=end_date,
            frequency=frequency, adjustflag=adjustflag
        )
        if rs.error_code != '0':
            raise ValueError(f"baostock 查询失败: {rs.error_msg}")

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
    finally:
        bs.logout()

    if not data_list:
        raise ValueError(f"baostock 返回空数据: {bs_code} {timeframe}")

    df = pd.DataFrame(data_list, columns=fields.split(','))
    num_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    if 'pctChg' in df.columns:
        num_cols.append('pctChg')
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 分钟线合并日期时间
    if timeframe not in ('daily', 'weekly', 'monthly') and 'time' in df.columns:
        def _fmt_time(t):
            t = str(t).zfill(6)
            return f"{t[:2]}:{t[2:4]}:{t[4:6]}"
        df['date'] = df['date'] + ' ' + df['time'].apply(_fmt_time)

    df = df.rename(columns={'pctChg': 'pct_change'})
    if 'time' in df.columns:
        df = df.drop(columns=['time'])

    # 120分钟重采样
    if timeframe == '120min':
        df = _resample_ohlc(df, '120min')

    return _standardize_dataframe(df, f"baostock-{timeframe}")


def _fetch_tencent(code, start_date, end_date, timeframe='daily', is_index=False):
    """
    通过腾讯财经直连HTTP API获取数据（非akshare封装，可能绕过部分封锁）。
    支持日线(前复权)/60分钟/30分钟/15分钟，120分钟由60分钟重采样。
    """
    import requests

    # 腾讯代码格式: sh600519, sz000001
    if is_index:
        prefix = 'sz' if code.startswith('399') else 'sh'
    else:
        prefix = 'sh' if code.startswith(('60', '68', '90', '11', '13', '51', '58')) else 'sz'
    qt_code = f"{prefix}{code}"

    # 计算需要的数据条数
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    days = max((end_dt - start_dt).days, 10)
    if timeframe == 'daily':
        datalen = min(days + 50, 2000)
    else:
        datalen = min(days * 8 + 100, 5000)

    # 构造URL
    if timeframe == 'daily':
        url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={qt_code},day,,,{datalen},qfq"
        data_key = 'qfqday'
    elif timeframe in ('60min', '120min'):
        url = f"http://web.ifzq.gtimg.cn/appstock/app/kline/mkline?param={qt_code},m60,,{datalen}"
        data_key = 'm60'
    elif timeframe == '30min':
        url = f"http://web.ifzq.gtimg.cn/appstock/app/kline/mkline?param={qt_code},m30,,{datalen}"
        data_key = 'm30'
    elif timeframe == '15min':
        url = f"http://web.ifzq.gtimg.cn/appstock/app/kline/mkline?param={qt_code},m15,,{datalen}"
        data_key = 'm15'
    else:
        raise ValueError(f"腾讯接口不支持周期: {timeframe}")

    resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
    if resp.status_code != 200:
        raise ConnectionError(f"腾讯接口HTTP {resp.status_code}")
    data = resp.json()

    if data.get('code') != 0:
        raise ValueError(f"腾讯接口返回错误: {data.get('msg', 'unknown')}")

    stock_data = data.get('data', {}).get(qt_code, {})
    kline = stock_data.get(data_key, stock_data.get('day', []))

    if not kline:
        raise ValueError(f"腾讯接口无K线数据: {qt_code} {timeframe}")

    # 每条数据: [日期, 开, 收, 高, 低, 成交量] (注意顺序: 开收高低)
    df = pd.DataFrame(kline)
    if df.shape[1] >= 6:
        df = df.iloc[:, :6]
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
    elif df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = ['date', 'open', 'close', 'high', 'low']
        df['volume'] = 0
    else:
        raise ValueError(f"腾讯接口数据格式异常: {df.shape}")

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 按日期范围过滤
    df['date'] = df['date'].astype(str)
    mask = (df['date'] >= start_date) & (df['date'] <= end_date + ' 23:59:59')
    df = df[mask].copy()
    if len(df) == 0:
        raise ValueError(f"腾讯接口在 {start_date}~{end_date} 范围内无数据")

    # 120分钟重采样
    if timeframe == '120min':
        df = _resample_ohlc(df, '120min')

    return _standardize_dataframe(df, f"tencent-{timeframe}")


# ============================================================
# 数据源优先级配置（支持周期和指数参数）
# 国外服务器优化顺序：Yahoo(最稳) → baostock(TCP) → 腾讯直连 → 同花顺 → 东方财富 → 新浪
# ============================================================
DATA_SOURCES = [
    ("Yahoo Finance", _fetch_yfinance, 2, 3),
    ("baostock", _fetch_baostock, 1, 2),
    ("腾讯财经", _fetch_tencent, 1, 2),
    ("efinance-同花顺", _fetch_efinance, 1, 1),
    ("AkShare-东方财富", _fetch_akshare_eastmoney, 2, 2),
    ("AkShare-新浪财经", _fetch_akshare_sina, 1, 1),
]


@cache_pickle(expire_seconds=86400)
def fetch_stock_data(code, start_date, end_date, timeframe='daily', is_index=False):
    """
    获取股票/指数K线数据（多数据源自动降级 + 多周期 + 指数支持）。

    参数:
      code: 6位代码，如 '600519'（个股）或 '000001'（上证指数）
      start_date/end_date: 'YYYY-MM-DD'
      timeframe: 'daily'/'60min'/'120min'/'30min'/'15min'/'weekly'
      is_index: 是否为大盘指数
    """
    _set_default_timeout(30)
    _setup_requests_session()

    if timeframe not in TIMEFRAMES:
        raise ValueError(f"不支持的周期: {timeframe}，可选: {list(TIMEFRAMES.keys())}")

    errors = []
    for source_name, fetch_func, max_retries, base_delay in DATA_SOURCES:
        try:
            print(f"\n[数据源] 尝试 {source_name} (周期={timeframe}, 指数={is_index})...")
            df = _fetch_with_retry(
                lambda: fetch_func(code, start_date, end_date, timeframe, is_index),
                max_retries=max_retries, base_delay=base_delay
            )
            if df is not None and len(df) > 0:
                print(f"[数据源] ✅ {source_name} 获取成功，共 {len(df)} 条数据")
                return df
            errors.append(f"{source_name}: 返回空数据")
        except Exception as e:
            err_msg = f"{source_name}: {type(e).__name__}: {str(e)[:100]}"
            errors.append(err_msg)
            print(f"[数据源] ❌ {err_msg}")
            continue

    # 构建友好的错误提示
    tf_name = TIMEFRAMES.get(timeframe, {}).get('name', timeframe)
    suggestion = ""
    if timeframe in ('60min', '120min', '30min', '15min'):
        suggestion = (f"\n\n💡 建议：{tf_name}数据在海外服务器获取较困难。"
                      f"\n   - Yahoo Finance 60分钟数据仅支持最近2年，30/15分钟仅支持最近60天"
                      f"\n   - 请缩短日期范围至最近2年内（60分钟）或最近60天内（30/15分钟）"
                      f"\n   - 或改用日线/周线周期，数据更稳定")
    else:
        suggestion = "\n\n💡 建议：检查代码是否正确，或稍后重试（数据源可能临时不可用）"

    raise ConnectionError(
        f"{code} ({tf_name}) 数据获取失败，所有数据源均不可用：\n  "
        + "\n  ".join(errors) + suggestion
    )


def fetch_recent_data(code, days=150, timeframe='daily', is_index=False):
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days * 2)).strftime('%Y-%m-%d')
    @cache_pickle(expire_seconds=7200)
    def _fetch(c, s, e, tf, idx):
        return fetch_stock_data(c, s, e, tf, idx)
    return _fetch(code, start, end, timeframe, is_index)


# ============================================================
# 技术指标计算（55个指标）
# ============================================================
def calc_indicators(df):
    d = df.copy()
    c = d['close']
    h = d['high']
    l = d['low']
    v = d['volume']
    n = len(d)

    # --- 均线系统 ---
    for w in [5,10,20,60,120]:
        d[f'MA{w}'] = c.rolling(w).mean()
        d[f'EMA{w}'] = c.ewm(span=w, adjust=False).mean()
    d['WMA20'] = c.rolling(20).apply(lambda x: np.dot(x, np.arange(1,21)) / np.arange(1,21).sum(), raw=True)

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['DIF'] = ema12 - ema26
    d['DEA'] = d['DIF'].ewm(span=9, adjust=False).mean()
    d['MACD'] = 2 * (d['DIF'] - d['DEA'])

    # --- RSI ---
    for period in [6,14,24]:
        delta = c.diff()
        gain = delta.where(delta>0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta<0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        d[f'RSI{period}'] = 100 - 100/(1+rs)

    # --- KDJ ---
    low9 = l.rolling(9).min()
    high9 = h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    d['K'] = rsv.ewm(com=2, adjust=False).mean()
    d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    d['J'] = 3*d['K'] - 2*d['D']

    # --- 布林带 ---
    d['BOLL_MID'] = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d['BOLL_UP'] = d['BOLL_MID'] + 2*std20
    d['BOLL_LOW'] = d['BOLL_MID'] - 2*std20
    d['BOLL_PCTB'] = (c - d['BOLL_LOW']) / (d['BOLL_UP'] - d['BOLL_LOW']).replace(0, np.nan)
    d['BOLL_BW'] = (d['BOLL_UP'] - d['BOLL_LOW']) / d['BOLL_MID'].replace(0, np.nan) * 100

    # --- 量能 ---
    d['VOL_MA5'] = v.rolling(5).mean()
    d['VOL_MA10'] = v.rolling(10).mean()
    d['VOL_RATE'] = (v - v.shift(1)) / v.shift(1).replace(0, np.nan) * 100
    d['VOL_RATIO'] = v / d['VOL_MA5'].replace(0, np.nan)

    # --- 能量/资金流 ---
    direction = np.sign(c.diff()).fillna(0)
    d['OBV'] = (direction * v).cumsum()
    # Accumulation/Distribution Line
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    d['AD_LINE'] = (clv * v).cumsum()
    # MFI (Money Flow Index)
    typical = (h + l + c) / 3
    money_flow = typical * v
    pos_flow = money_flow.where(typical > typical.shift(1), 0)
    neg_flow = money_flow.where(typical < typical.shift(1), 0)
    mfi_ratio = pos_flow.rolling(14).sum() / neg_flow.rolling(14).sum().replace(0, np.nan)
    d['MFI14'] = 100 - 100/(1+mfi_ratio)
    # VWAP
    d['VWAP'] = (typical * v).cumsum() / v.cumsum().replace(0, np.nan)

    # --- 超买超卖 ---
    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    d['WR'] = (hh14 - c) / (hh14 - ll14).replace(0, np.nan) * -100
    tp = (h + l + c) / 3
    ma_tp = tp.rolling(14).mean()
    md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d['CCI'] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))
    # Stochastic RSI
    rsi14 = d['RSI14']
    rsi_low = rsi14.rolling(14).min()
    rsi_high = rsi14.rolling(14).max()
    d['STOCHRSI'] = (rsi14 - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan) * 100

    # --- 趋势指标 ---
    # ADX / +DI / -DI
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr14 = tr.rolling(14).mean()
    d['PDI14'] = 100 * pd.Series(plus_dm, index=d.index).rolling(14).mean() / atr14.replace(0, np.nan)
    d['MDI14'] = 100 * pd.Series(minus_dm, index=d.index).rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (d['PDI14'] - d['MDI14']).abs() / (d['PDI14'] + d['MDI14']).replace(0, np.nan)
    d['ADX14'] = dx.rolling(14).mean()
    # Parabolic SAR
    sar = np.zeros(n)
    af = 0.02
    ep = h.iloc[0]
    sar[0] = l.iloc[0]
    trend = 1  # 1=up, -1=down
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend == 1:
            if l.iloc[i] < sar[i]:
                trend = -1; sar[i] = ep; ep = l.iloc[i]; af = 0.02
            else:
                if h.iloc[i] > ep: ep = h.iloc[i]; af = min(af+0.02, 0.2)
        else:
            if h.iloc[i] > sar[i]:
                trend = 1; sar[i] = ep; ep = h.iloc[i]; af = 0.02
            else:
                if l.iloc[i] < ep: ep = l.iloc[i]; af = min(af+0.02, 0.2)
    d['SAR'] = sar
    # TRIX
    ema1 = c.ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    d['TRIX12'] = (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan) * 100
    # DMA (Different of Moving Average)
    d['DMA'] = c.rolling(10).mean() - c.rolling(50).mean()

    # --- 波动率 ---
    d['ATR14'] = atr14
    d['NATR14'] = atr14 / c.replace(0, np.nan) * 100
    d['STD20'] = std20
    # Donchian Channel width
    don_up = h.rolling(20).max()
    don_low = l.rolling(20).min()
    d['DONCHIAN_W'] = (don_up - don_low) / c.replace(0, np.nan) * 100

    # --- 动量/变化率 ---
    d['ROC12'] = (c - c.shift(12)) / c.shift(12).replace(0, np.nan) * 100
    d['MOM10'] = c - c.shift(10)
    # CMO (Chande Momentum Oscillator)
    delta_cmo = c.diff()
    su = delta_cmo.where(delta_cmo > 0, 0).rolling(14).sum()
    sd = (-delta_cmo.where(delta_cmo < 0, 0)).rolling(14).sum()
    d['CMO14'] = 100 * (su - sd) / (su + sd).replace(0, np.nan)
    # Ultimate Oscillator
    bp = c - np.minimum(l, c.shift(1))
    tr_uo = np.maximum(h, c.shift(1)) - np.minimum(l, c.shift(1))
    avg7 = bp.rolling(7).sum() / tr_uo.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr_uo.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr_uo.rolling(28).sum().replace(0, np.nan)
    d['ULTOSC'] = 100 * (4*avg7 + 2*avg14 + avg28) / 7

    # --- 乖离/心理 ---
    for w in [5,10,20,60]:
        d[f'BIAS{w}'] = (c - c.rolling(w).mean()) / c.rolling(w).mean().replace(0, np.nan) * 100
    # PSY (Psychological Line)
    d['PSY12'] = (c.diff() > 0).rolling(12).sum() / 12 * 100

    # --- 一目均衡表 (Ichimoku Cloud) ---
    d['TENKAN'] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    d['KIJUN'] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    d['SENKOU_A'] = (d['TENKAN'] + d['KIJUN']) / 2
    d['SENKOU_B'] = (h.rolling(52).max() + l.rolling(52).min()) / 2

    # --- 价格位置 ---
    d['CLOSE_MA20_RATIO'] = (c - d['MA20']) / d['MA20'].replace(0, np.nan) * 100
    d['CLOSE_MA60_RATIO'] = (c - d['MA60']) / d['MA60'].replace(0, np.nan) * 100

    # 去除预热期（MA120需要120根K线，一目均衡表需要52根）
    warmup = 120
    d = d.iloc[warmup:].reset_index(drop=True)
    # 替换inf为NaN
    d = d.replace([np.inf, -np.inf], np.nan)
    # 仅前向填充（不后向填充，避免用未来数据填充早期值的未来函数）
    d = d.ffill()
    # 仍有NaN的行（如前向填充不到的首行）直接丢弃，绝不伪造数据
    before = len(d)
    d = d.dropna()
    dropped = before - len(d)
    if dropped > 0:
        print(f"[指标计算] 丢弃{dropped}条仍含NaN的行（不填充）")
    return d


# ============================================================
# 最优买卖点（贪心峰谷法）
# ============================================================
def find_optimal_trades(df, min_profit_pct=1.0):
    closes = df['close'].values
    dates = df['date'].values
    n = len(closes)
    if n < 2:
        return [], 0.0, 0.0
    trades = []
    i = 0
    while i < n - 1:
        while i < n - 1 and closes[i+1] <= closes[i]:
            i += 1
        if i >= n - 1: break
        buy_idx = i
        buy_price = closes[buy_idx]
        while i < n - 1 and closes[i+1] >= closes[i]:
            i += 1
        sell_idx = i
        sell_price = closes[sell_idx]
        profit = (sell_price - buy_price) / buy_price * 100
        if profit >= min_profit_pct:
            trades.append({
                'buy_date': str(dates[buy_idx]), 'buy_price': round(float(buy_price),2),
                'sell_date': str(dates[sell_idx]), 'sell_price': round(float(sell_price),2),
                'profit_rate': round(float(profit),2), 'holding_bars': int(sell_idx - buy_idx)
            })
    total = 1.0
    for t in trades:
        total *= (1 + t['profit_rate']/100)
    total_return = round((total-1)*100, 2)
    buy_hold = round((closes[-1]-closes[0])/closes[0]*100, 2)
    return trades, total_return, buy_hold


# ============================================================
# 相关性分析（全指标扫描）
# ============================================================
def analyze_correlation(df, trades):
    n = len(df)
    buy_dates = set(t['buy_date'] for t in trades)
    sell_dates = set(t['sell_date'] for t in trades)
    buy_labels = np.array([1 if d in buy_dates else 0 for d in df['date']])
    sell_labels = np.array([1 if d in sell_dates else 0 for d in df['date']])

    def _corr(labels):
        results = []
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        for col in INDICATOR_COLS:
            if col not in df.columns: continue
            vals = df[col].values.astype(float)
            if np.isnan(vals).all() or np.nanstd(vals) == 0: continue
            valid = ~np.isnan(vals)
            if valid.sum() < 10: continue
            r, p = stats.pointbiserialr(labels[valid], vals[valid])
            if np.isnan(r) or np.isnan(p): continue
            mean_at = float(np.nanmean(vals[labels==1])) if n_pos > 0 else 0
            mean_ot = float(np.nanmean(vals[labels==0])) if n_neg > 0 else 0
            results.append({
                'indicator': col, 'corr': round(float(r),4), 'abs_corr': round(abs(float(r)),4),
                'pvalue': round(float(p),6), 'significant': bool(p<0.05),
                'mean_at_event': round(mean_at,4), 'mean_other': round(mean_ot,4),
                'direction': 'high' if mean_at > mean_ot else 'low'
            })
        results.sort(key=lambda x: x['abs_corr'], reverse=True)
        return results

    buy_corr = _corr(buy_labels)
    sell_corr = _corr(sell_labels)
    return buy_corr, sell_corr


# ============================================================
# 模型灵敏度配置
# ============================================================
SENSITIVITY_CONFIG = {
    'conservative': {
        'name': '保守',
        'min_corr': 0.15,       # 最低相关系数阈值
        'max_pvalue': 0.01,      # 最高p值
        'min_indicators': 8,     # 最少指标数
        'max_indicators': 15,    # 最多指标数
        'prob_threshold': 0.70,  # 信号概率阈值
        'min_prob_gap': 0.15,    # 买卖概率最小差距（防止两者同时高）
        'description': '高门槛筛选，仅保留强相关指标，信号少但准'
    },
    'balanced': {
        'name': '均衡',
        'min_corr': 0.08,
        'max_pvalue': 0.05,
        'min_indicators': 5,
        'max_indicators': 12,
        'prob_threshold': 0.60,
        'min_prob_gap': 0.10,
        'description': '标准筛选，兼顾信号数量与质量'
    },
    'aggressive': {
        'name': '激进',
        'min_corr': 0.04,
        'max_pvalue': 0.10,
        'min_indicators': 3,
        'max_indicators': 10,
        'prob_threshold': 0.50,
        'min_prob_gap': 0.05,
        'description': '低门槛筛选，信号频繁但需注意假信号'
    }
}


# ============================================================
# 改进数学模型：全指标扫描 → 自动筛选 → 淘汰低相关
# ============================================================
def build_model(df, buy_corr, sell_corr, sensitivity='balanced', model_name=None):
    """
    构建可解释数学模型（改进版）。

    流程：
      1. 扫描全部55个指标与买卖点的相关性
      2. 按灵敏度配置筛选：相关系数≥min_corr 且 p值≤max_pvalue
      3. 淘汰不满足条件的指标，保留Top N（max_indicators）
      4. 对保留指标构建阈值加权打分规则
      5. 返回模型（含被淘汰指标列表，便于展示）
    """
    cfg = SENSITIVITY_CONFIG.get(sensitivity, SENSITIVITY_CONFIG['balanced'])

    def _select_and_build(corr_list, event_type):
        # 筛选：满足相关性和显著性条件
        selected = [c for c in corr_list
                    if c['abs_corr'] >= cfg['min_corr'] and c['pvalue'] <= cfg['max_pvalue']]
        eliminated = [c for c in corr_list if c not in selected]

        # 如果筛选后不足min_indicators，放宽条件补足
        if len(selected) < cfg['min_indicators']:
            for c in corr_list:
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= cfg['min_indicators']:
                        break

        # 限制最多max_indicators
        selected = selected[:cfg['max_indicators']]
        eliminated = [c for c in corr_list if c not in selected]

        # 构建规则
        rules = []
        for item in selected:
            col = item['indicator']
            if col not in df.columns: continue
            weight = item['abs_corr']
            direction = item['direction']
            threshold = (item['mean_at_event'] + item['mean_other']) / 2
            std_val = float(df[col].std())
            if std_val == 0 or np.isnan(std_val): std_val = 1.0
            rules.append({
                'indicator': col, 'weight': round(weight,4),
                'direction': direction, 'threshold': round(float(threshold),4),
                'std': round(std_val,4), 'corr': item['corr'], 'pvalue': item['pvalue'],
                'mean_at_event': item['mean_at_event'], 'mean_other': item['mean_other'],
                'event_type': event_type
            })
        return rules, selected, eliminated

    buy_rules, buy_selected, buy_eliminated = _select_and_build(buy_corr, 'buy')
    sell_rules, sell_selected, sell_eliminated = _select_and_build(sell_corr, 'sell')

    # 规则去冲突：同一指标不能同时出现在买点和卖点规则中
    # 保留相关性更强（abs_corr更大）的一方，另一方移入被淘汰列表
    buy_indicators = {r['indicator']: r for r in buy_rules}
    sell_indicators = {r['indicator']: r for r in sell_rules}
    conflict_indicators = set(buy_indicators.keys()) & set(sell_indicators.keys())
    conflict_removed_buy = []
    conflict_removed_sell = []
    for ind in conflict_indicators:
        b_corr = abs(buy_indicators[ind].get('corr', 0))
        s_corr = abs(sell_indicators[ind].get('corr', 0))
        if b_corr >= s_corr:
            # 保留买点，移除卖点
            sell_rules = [r for r in sell_rules if r['indicator'] != ind]
            conflict_removed_sell.append({'indicator': ind, 'corr': sell_indicators[ind].get('corr', 0),
                                           'pvalue': sell_indicators[ind].get('pvalue', 0),
                                           'reason': '与买点规则冲突（买点相关性更强）'})
        else:
            # 保留卖点，移除买点
            buy_rules = [r for r in buy_rules if r['indicator'] != ind]
            conflict_removed_buy.append({'indicator': ind, 'corr': buy_indicators[ind].get('corr', 0),
                                          'pvalue': buy_indicators[ind].get('pvalue', 0),
                                          'reason': '与卖点规则冲突（卖点相关性更强）'})
    buy_eliminated = conflict_removed_buy + buy_eliminated
    sell_eliminated = conflict_removed_sell + sell_eliminated

    model = {
        'name': model_name or f"模型_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'type': '阈值加权打分系统 v2.0（全指标自动筛选）',
        'sensitivity': sensitivity,
        'sensitivity_name': cfg['name'],
        'sensitivity_config': cfg,
        'description': (
            f"扫描全部{len(INDICATOR_COLS)}个技术指标，按{cfg['name']}灵敏度"
            f"(|r|≥{cfg['min_corr']}, p≤{cfg['max_pvalue']})自动筛选，"
            f"淘汰低相关指标后构建阈值加权打分模型。"
        ),
        'buy_rules': buy_rules,
        'sell_rules': sell_rules,
        'buy_selected': [c['indicator'] for c in buy_selected],
        'sell_selected': [c['indicator'] for c in sell_selected],
        'buy_eliminated': [{'indicator': c['indicator'], 'corr': c['corr'], 'pvalue': c['pvalue']}
                           for c in buy_eliminated],
        'sell_eliminated': [{'indicator': c['indicator'], 'corr': c['corr'], 'pvalue': c['pvalue']}
                            for c in sell_eliminated],
        'buy_threshold_prob': cfg['prob_threshold'],
        'sell_threshold_prob': cfg['prob_threshold'],
        'min_prob_gap': cfg['min_prob_gap'],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'formula': (
            'score = Σ [ weight_i × sigmoid( (value_i - threshold_i) / std_i × sign_i ) ]\n'
            'prob = score / Σ weight_i\n'
            'sign_i = +1 (direction=high), -1 (direction=low)\n'
            '信号触发: prob > threshold AND prob > other_prob + min_prob_gap'
        )
    }
    return model


def compute_model_signals(df, model):
    buy_rules = model['buy_rules']
    sell_rules = model['sell_rules']
    buy_w_sum = sum(r['weight'] for r in buy_rules) or 1
    sell_w_sum = sum(r['weight'] for r in sell_rules) or 1
    n = len(df)

    def _calc(rules, w_sum):
        probs = np.zeros(n)
        for r in rules:
            col = r['indicator']
            if col not in df.columns: continue
            vals = df[col].values.astype(float)
            sign = 1 if r['direction'] == 'high' else -1
            z = sign * (vals - r['threshold']) / (r['std'] or 1.0)
            sig = 1 / (1 + np.exp(-np.clip(z, -10, 10)))
            probs += r['weight'] * sig
        return probs / w_sum

    return _calc(buy_rules, buy_w_sum), _calc(sell_rules, sell_w_sum)


# ============================================================
# 模型命名存储与加载
# ============================================================
def save_model(model, metrics=None, backtest_info=None, custom_name=None):
    """将模型保存为JSON文件，支持命名存储。
    custom_name: 如提供，则覆盖模型名称用于保存和存储。
    """
    if custom_name:
        model = dict(model)  # 浅拷贝，避免修改原对象
        model['name'] = custom_name
    model_data = {
        'model': model,
        'metrics': metrics or {},
        'backtest_info': backtest_info or {},
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in model['name'])
    filepath = os.path.join(MODEL_DIR, f"{safe_name}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"[模型存储] 已保存: {filepath}")
    return filepath


def load_model(model_name):
    """按名称加载已保存的模型。"""
    safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in model_name)
    filepath = os.path.join(MODEL_DIR, f"{safe_name}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型不存在: {model_name}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['model'], data.get('metrics', {}), data.get('backtest_info', {})


def list_saved_models():
    """列出所有已保存的模型。"""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.endswith('.json'):
            filepath = os.path.join(MODEL_DIR, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                m = data.get('model', {})
                models.append({
                    'name': m.get('name', fname.replace('.json','')),
                    'sensitivity': m.get('sensitivity_name', ''),
                    'sensitivity_key': m.get('sensitivity', ''),
                    'buy_indicators': len(m.get('buy_rules', [])),
                    'sell_indicators': len(m.get('sell_rules', [])),
                    'created_at': m.get('created_at', ''),
                    'saved_at': data.get('saved_at', ''),
                    'total_return': data.get('metrics', {}).get('total_return', ''),
                    'win_rate': data.get('metrics', {}).get('win_rate', ''),
                    'filename': fname
                })
            except Exception:
                continue
    return models


def delete_model(model_name):
    """删除已保存的模型。"""
    safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in model_name)
    filepath = os.path.join(MODEL_DIR, f"{safe_name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


# ============================================================
# 历史回测
# ============================================================
def backtest(df, model, buy_probs, sell_probs):
    buy_th = model['buy_threshold_prob']
    sell_th = model['sell_threshold_prob']
    min_gap = model.get('min_prob_gap', 0.1)  # 买卖概率最小差距，防止两者同时高
    n = len(df)
    cash = INITIAL_CAPITAL
    shares = 0
    equity = np.zeros(n)
    trades = []
    position = None

    for i in range(n):
        if i < n - 1:
            exec_price = float(df['open'].iloc[i+1])
            exec_date = str(df['date'].iloc[i+1])
            # 买入条件：概率≥阈值 且 买概率 > 卖概率 + 最小差距
            if position is None and buy_probs[i] >= buy_th and buy_probs[i] > sell_probs[i] + min_gap:
                max_cost = cash / (1 + COMMISSION_RATE)
                buy_shares = int(max_cost / exec_price / 100) * 100
                if buy_shares >= 100:
                    cost = buy_shares * exec_price
                    commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
                    total_cost = cost + commission
                    if total_cost <= cash:
                        cash -= total_cost
                        shares = buy_shares
                        position = {'buy_date': exec_date, 'buy_price': round(exec_price,2),
                                    'shares': buy_shares, 'buy_cost': round(total_cost,2),
                                    'buy_prob': round(float(buy_probs[i]),4)}
            # 卖出条件：概率≥阈值 且 卖概率 > 买概率 + 最小差距
            elif position is not None and sell_probs[i] >= sell_th and sell_probs[i] > buy_probs[i] + min_gap:
                revenue = shares * exec_price
                commission = max(revenue * COMMISSION_RATE, MIN_COMMISSION)
                stamp_tax = revenue * STAMP_TAX_RATE
                net_revenue = revenue - commission - stamp_tax
                cash += net_revenue
                profit = net_revenue - position['buy_cost']
                profit_rate = profit / position['buy_cost'] * 100
                trades.append({
                    'buy_date': position['buy_date'], 'buy_price': position['buy_price'],
                    'sell_date': exec_date, 'sell_price': round(exec_price,2),
                    'shares': position['shares'], 'profit': round(profit,2),
                    'profit_rate': round(profit_rate,2),
                    'buy_prob': position['buy_prob'], 'sell_prob': round(float(sell_probs[i]),4)
                })
                shares = 0; position = None
        equity[i] = cash + shares * float(df['close'].iloc[i])

    if position is not None and shares > 0:
        last_price = float(df['close'].iloc[-1])
        revenue = shares * last_price
        commission = max(revenue * COMMISSION_RATE, MIN_COMMISSION)
        stamp_tax = revenue * STAMP_TAX_RATE
        net_revenue = revenue - commission - stamp_tax
        cash += net_revenue
        profit = net_revenue - position['buy_cost']
        profit_rate = profit / position['buy_cost'] * 100
        trades.append({
            'buy_date': position['buy_date'], 'buy_price': position['buy_price'],
            'sell_date': str(df['date'].iloc[-1])+'(期末平仓)', 'sell_price': round(last_price,2),
            'shares': position['shares'], 'profit': round(profit,2),
            'profit_rate': round(profit_rate,2),
            'buy_prob': position['buy_prob'], 'sell_prob': 1.0
        })
        equity[-1] = cash; shares = 0; position = None

    equity_series = pd.Series(equity)
    total_return = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    years = len(df) / 252 if df['date'].iloc[0].count('-') == 2 and len(df['date'].iloc[0]) == 10 else len(df) / (252*4)
    annual_return = ((equity[-1]/INITIAL_CAPITAL)**(1/years) - 1)*100 if years > 0 else 0
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = float(drawdown.min())
    win_trades = [t for t in trades if t['profit'] > 0]
    win_rate = len(win_trades)/len(trades)*100 if trades else 0
    avg_win = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
    loss_trades = [t for t in trades if t['profit'] <= 0]
    avg_loss = abs(np.mean([t['profit'] for t in loss_trades])) if loss_trades else 0
    pl_ratio = round(avg_win/avg_loss, 2) if avg_loss > 0 else '∞'
    bh_return = (float(df['close'].iloc[-1]) - float(df['close'].iloc[0])) / float(df['close'].iloc[0]) * 100

    metrics = {
        'initial_capital': INITIAL_CAPITAL, 'final_equity': round(float(equity[-1]),2),
        'total_return': round(float(total_return),2), 'annual_return': round(float(annual_return),2),
        'max_drawdown': round(float(max_drawdown),2), 'win_rate': round(float(win_rate),2),
        'profit_loss_ratio': pl_ratio, 'total_trades': len(trades),
        'win_trades': len(win_trades), 'loss_trades': len(loss_trades),
        'buy_hold_return': round(float(bh_return),2),
        'excess_return': round(float(total_return - bh_return),2)
    }
    step = max(1, n // 500)
    equity_curve = [{'date': str(df['date'].iloc[i]), 'equity': round(float(equity[i]),2)}
                     for i in range(0, n, step)]
    if equity_curve[-1]['date'] != str(df['date'].iloc[-1]):
        equity_curve.append({'date': str(df['date'].iloc[-1]), 'equity': round(float(equity[-1]),2)})
    return equity_curve, trades, metrics


# ============================================================
# 预测准确率
# ============================================================
def evaluate_prediction_accuracy(df, buy_probs, sell_probs, horizon=5):
    n = len(df)
    closes = df['close'].values
    buy_th = 0.5 if not hasattr(evaluate_prediction_accuracy, '_th') else evaluate_prediction_accuracy._th
    sell_th = buy_th
    buy_signals = []; sell_signals = []
    for i in range(n - horizon):
        if buy_probs[i] >= buy_th:
            future_max = float(np.max(closes[i+1:i+1+horizon]))
            current = float(closes[i])
            buy_signals.append({'date': str(df['date'].iloc[i]), 'prob': round(float(buy_probs[i]),4),
                                'future_return': round((future_max-current)/current*100,2),
                                'correct': future_max > current*1.01})
        if sell_probs[i] >= sell_th:
            future_min = float(np.min(closes[i+1:i+1+horizon]))
            current = float(closes[i])
            sell_signals.append({'date': str(df['date'].iloc[i]), 'prob': round(float(sell_probs[i]),4),
                                 'future_return': round((future_min-current)/current*100,2),
                                 'correct': future_min < current*0.99})
    buy_acc = sum(1 for s in buy_signals if s['correct'])/len(buy_signals)*100 if buy_signals else 0
    sell_acc = sum(1 for s in sell_signals if s['correct'])/len(sell_signals)*100 if sell_signals else 0
    return {
        'horizon': horizon, 'buy_signal_count': len(buy_signals),
        'buy_accuracy': round(buy_acc,2), 'sell_signal_count': len(sell_signals),
        'sell_accuracy': round(sell_acc,2),
        'buy_signals_sample': buy_signals[:20], 'sell_signals_sample': sell_signals[:20]
    }


# ============================================================
# 实时信号（支持灵敏度模型和已保存模型）
# ============================================================
def generate_realtime_signal(code, timeframe='daily', is_index=False, model=None, sensitivity='balanced'):
    """
    基于最新行情生成实时买卖信号。
    可传入已构建/已加载的模型，或指定灵敏度自动构建。
    """
    df_raw = fetch_recent_data(code, days=200, timeframe=timeframe, is_index=is_index)
    if len(df_raw) < 130:
        raise ValueError(f"近期数据不足（仅{len(df_raw)}条），无法计算指标")

    df = calc_indicators(df_raw)
    trades, _, _ = find_optimal_trades(df, min_profit_pct=1.0)
    if len(trades) < 2:
        trades, _, _ = find_optimal_trades(df, min_profit_pct=0.5)

    if model is None:
        buy_corr, sell_corr = analyze_correlation(df, trades)
        model = build_model(df, buy_corr, sell_corr, sensitivity=sensitivity,
                            model_name=f"实时_{code}_{sensitivity}")

    buy_probs, sell_probs = compute_model_signals(df, model)
    latest = df.iloc[-1]
    lb = float(buy_probs[-1]); ls = float(sell_probs[-1])
    min_gap = model.get('min_prob_gap', 0.1)
    buy_th = model['buy_threshold_prob']
    sell_th = model['sell_threshold_prob']

    # 信号判定：概率≥阈值 且 高概率方 > 低概率方 + 最小差距
    # 防止买卖概率同时偏高（信号冲突）
    conflicting = (lb >= buy_th and ls >= sell_th)  # 两者都超阈值=冲突
    if lb >= buy_th and lb > ls + min_gap:
        signal = '买入'; strength = lb
    elif ls >= sell_th and ls > lb + min_gap:
        signal = '卖出'; strength = ls
    else:
        signal = '持有'; strength = max(lb, ls)

    # 构建全部买卖指标的触发状态（不只是前8个，全部展示）
    def _check_rule(rule):
        col = rule['indicator']
        if col not in df.columns:
            return None
        val = float(latest[col])
        sign = 1 if rule['direction'] == 'high' else -1
        triggered = bool(sign * (val - rule['threshold']) > 0)
        return {
            'indicator': col, 'value': round(val, 4),
            'threshold': round(rule['threshold'], 4),
            'direction': rule['direction'], 'weight': rule['weight'],
            'corr': round(rule.get('corr', 0), 4),
            'pvalue': round(rule.get('pvalue', 0), 4),
            'triggered': triggered,
            'contribution': round(rule['weight'] * (1 if triggered else 0), 4)
        }

    all_buy_indicators = [r for r in (_check_rule(r) for r in model['buy_rules']) if r]
    all_sell_indicators = [r for r in (_check_rule(r) for r in model['sell_rules']) if r]

    buy_triggered = sum(1 for r in all_buy_indicators if r['triggered'])
    sell_triggered = sum(1 for r in all_sell_indicators if r['triggered'])

    # 生成操作建议说明
    gap = abs(lb - ls)
    if signal == '买入':
        action_desc = (f"买点概率 {lb*100:.1f}% ≥ 阈值 {buy_th*100:.0f}%，"
                       f"且高于卖点概率 {ls*100:.1f}%（差距{gap*100:.1f}% ≥ 最小差距{min_gap*100:.0f}%）。"
                       f"{buy_triggered}/{len(all_buy_indicators)} 个买点指标触发。")
    elif signal == '卖出':
        action_desc = (f"卖点概率 {ls*100:.1f}% ≥ 阈值 {sell_th*100:.0f}%，"
                       f"且高于买点概率 {lb*100:.1f}%（差距{gap*100:.1f}% ≥ 最小差距{min_gap*100:.0f}%）。"
                       f"{sell_triggered}/{len(all_sell_indicators)} 个卖点指标触发。")
    else:
        if conflicting:
            action_desc = (f"⚠️ 信号冲突：买点概率 {lb*100:.1f}% 和卖点概率 {ls*100:.1f}% 均超过阈值，"
                           f"但差距仅{gap*100:.1f}% < 最小差距{min_gap*100:.0f}%，多空分歧大。"
                           f"买{buy_triggered}/{len(all_buy_indicators)}触发，卖{sell_triggered}/{len(all_sell_indicators)}触发。建议观望。")
        else:
            action_desc = (f"买点概率 {lb*100:.1f}% / 卖点概率 {ls*100:.1f}%，均未达到阈值"
                           f"（买≥{buy_th*100:.0f}% / 卖≥{sell_th*100:.0f}%，且需差距≥{min_gap*100:.0f}%）。"
                           f"买{buy_triggered}/{len(all_buy_indicators)}触发，卖{sell_triggered}/{len(all_sell_indicators)}触发。建议观望。")

    # 波段顶/底概率（无未来函数，基于当前实时指标）
    swing_state = None
    try:
        import swing_detection
        swing_state = swing_detection.calculate_current_swing_probability(df)
    except Exception as e:
        print(f"[实时信号] 波段状态计算失败: {e}")

    return {
        'code': code, 'timeframe': timeframe, 'is_index': is_index,
        'date': str(latest['date']), 'close': round(float(latest['close']),2),
        'open': round(float(latest['open']),2), 'high': round(float(latest['high']),2),
        'low': round(float(latest['low']),2), 'volume': int(latest['volume']),
        'signal': signal, 'signal_strength': round(strength,4),
        'buy_probability': round(lb,4), 'sell_probability': round(ls,4),
        'prob_gap': round(gap,4), 'min_prob_gap': min_gap, 'conflicting': conflicting,
        'action_description': action_desc,
        'buy_indicators': all_buy_indicators,
        'sell_indicators': all_sell_indicators,
        'buy_triggered_count': buy_triggered,
        'sell_triggered_count': sell_triggered,
        'buy_total_count': len(all_buy_indicators),
        'sell_total_count': len(all_sell_indicators),
        'key_indicators': all_buy_indicators[:5] + all_sell_indicators[:5],  # 兼容旧字段
        'model_name': model.get('name',''), 'sensitivity': model.get('sensitivity_name',''),
        'model_threshold_buy': model['buy_threshold_prob'],
        'model_threshold_sell': model['sell_threshold_prob'],
        'swing_state': swing_state,
    }


# ============================================================
# 完整分析流水线
# ============================================================
def run_full_analysis(code, start_date, end_date, timeframe='daily', is_index=False,
                      sensitivity='balanced', model_name=None, min_profit_pct=1.0):
    """执行完整分析流水线，返回所有结果。"""
    sd = datetime.strptime(start_date, '%Y-%m-%d')
    fetch_start = (sd - timedelta(days=200)).strftime('%Y-%m-%d')
    df_raw = fetch_stock_data(code, fetch_start, end_date, timeframe=timeframe, is_index=is_index)
    if len(df_raw) < 130:
        raise ValueError(f'数据量不足（仅{len(df_raw)}条），请扩大日期范围')

    df_full = calc_indicators(df_raw)
    # 截取用户选定时间段（支持分钟线日期时间格式）
    if timeframe == 'daily' or timeframe == 'weekly':
        df = df_full[df_full['date'] >= start_date].reset_index(drop=True)
    else:
        df = df_full[df_full['date'] >= start_date].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f'选定时间段内有效数据不足（仅{len(df)}条），请扩大范围')

    trades, total_return, buy_hold_return = find_optimal_trades(df, min_profit_pct=min_profit_pct)
    buy_corr, sell_corr = analyze_correlation(df, trades)
    model = build_model(df, buy_corr, sell_corr, sensitivity=sensitivity, model_name=model_name)
    buy_probs, sell_probs = compute_model_signals(df, model)
    equity_curve, bt_trades, metrics = backtest(df, model, buy_probs, sell_probs)

    # 设置预测准确率的阈值
    evaluate_prediction_accuracy._th = model['buy_threshold_prob']
    accuracy = evaluate_prediction_accuracy(df, buy_probs, sell_probs, horizon=5)

    # ============================================================
    # 模型权重自更新（持续学习：基于回测表现调整指标权重）
    # ============================================================
    weight_learning_info = None
    try:
        import learning
        if len(bt_trades) >= 3 and metrics.get('total_trades', 0) >= 3:
            model = learning.update_model_weights_from_backtest(model, bt_trades, metrics)
            weight_learning_info = model.get('weight_learning', {})
            print(f"[持续学习] 模型权重已自更新，调整了{weight_learning_info.get('total_adjustments', 0)}个指标权重")
    except Exception as e:
        print(f"[持续学习] 权重自更新失败: {e}")

    # K线数据
    step = max(1, len(df) // 800)
    kline_data = []
    for i in range(0, len(df), step):
        row = df.iloc[i]
        kline_data.append({'date': str(row['date']), 'open': round(float(row['open']),2),
                           'close': round(float(row['close']),2), 'low': round(float(row['low']),2),
                           'high': round(float(row['high']),2), 'volume': int(row['volume'])})
    buy_markers = [{'date': t['buy_date'], 'price': t['buy_price']} for t in trades]
    sell_markers = [{'date': t['sell_date'], 'price': t['sell_price']} for t in trades]
    prob_curve = [{'date': str(df['date'].iloc[i]), 'buy_prob': round(float(buy_probs[i]),4),
                    'sell_prob': round(float(sell_probs[i]),4)} for i in range(0, len(df), step)]

    # ============================================================
    # 新闻/公告事件研究（仅日线和非指数）
    # ============================================================
    news_result = None
    if timeframe == 'daily' and not is_index:
        try:
            import news_analysis
            news_result = news_analysis.analyze_news_sensitivity(
                code, df_full, start_date, end_date, holding_days=5)
        except Exception as e:
            print(f"[主流程] 新闻分析失败: {e}")
            news_result = {'error': str(e), 'news_count': 0}

    # ============================================================
    # 资金流分析（仅日线和非指数）
    # ============================================================
    capital_flow_result = None
    if timeframe == 'daily' and not is_index:
        # 新闻和资金流都调用东方财富API，间隔2秒避免频率限制
        import time as _time
        _time.sleep(2)
        try:
            import capital_flow_analysis
            capital_flow_result = capital_flow_analysis.analyze_capital_flow_full(
                code, df_full, start_date, end_date)
        except Exception as e:
            print(f"[主流程] 资金流分析失败: {e}")
            capital_flow_result = {'error': str(e), 'flow_count': 0}

    # ============================================================
    # 异常信号捕捉（应跌未跌/应涨未涨 + 量价 + 资金流）+ 回测验证
    # ============================================================
    anomaly_result = None
    try:
        import anomaly_signal
        # 从资金流结果中提取资金流DataFrame
        cf_df = None
        if capital_flow_result and capital_flow_result.get('flow_data'):
            cf_df = pd.DataFrame(capital_flow_result['flow_data'])
            if 'date' in cf_df.columns:
                cf_df['date'] = pd.to_datetime(cf_df['date'])
        anomaly_signals = anomaly_signal.detect_anomaly_signals(
            df, capital_flow_df=cf_df, lookback=5, min_triggers=2)
        anomaly_summary = anomaly_signal.get_signal_summary(anomaly_signals)

        # 回测验证异常信号的历史表现
        anomaly_backtest = None
        if len(anomaly_signals) > 0 and df_full is not None and len(df_full) > 0:
            anomaly_backtest = anomaly_signal.backtest_anomaly_signals(
                anomaly_signals, df_full, holding_periods=[1, 3, 5, 10, 20])

        anomaly_result = {
            'signals': anomaly_signals,
            'summary': anomaly_summary,
            'backtest': anomaly_backtest,
        }
        print(f"[主流程] 异常信号: {anomaly_summary['total']}个 "
              f"(看涨{anomaly_summary['bullish']}/看跌{anomaly_summary['bearish']}, "
              f"高置信{anomaly_summary['high_confidence']})")
        if anomaly_backtest and 'overall' in anomaly_backtest:
            bt5 = anomaly_backtest['overall'].get('5d_avg_ret', 'N/A')
            bt5wr = anomaly_backtest['overall'].get('5d_win_rate', 'N/A')
            print(f"[主流程] 异常信号回测: 5日平均{bt5}%, 胜率{bt5wr}%")
    except Exception as e:
        print(f"[主流程] 异常信号检测失败: {e}")
        anomaly_result = {'signals': [], 'summary': {'total': 0}, 'error': str(e)}

    # ============================================================
    # 波段顶/底识别（含参数网格搜索+回测验证）
    # ============================================================
    swing_result = None
    try:
        import swing_detection
        swing_result = swing_detection.run_swing_analysis(
            df, timeframe=timeframe, holding_days=5)
        if 'error' not in swing_result:
            print(f"[主流程] 波段识别: 顶部{swing_result['top_count']}个, "
                  f"底部{swing_result['bottom_count']}个")
            bt = swing_result.get('best_backtest', {})
            if '5d' in bt.get('tops', {}):
                print(f"[主流程] 波段回测: 顶部5日下跌率{bt['tops']['5d']['decline_rate']}%, "
                      f"底部5日上涨率{bt['bottoms']['5d']['rise_rate']}%")
    except Exception as e:
        print(f"[主流程] 波段识别失败: {e}")
        swing_result = {'error': str(e)}

    return {
        'code': code, 'start_date': start_date, 'end_date': end_date,
        'timeframe': timeframe, 'is_index': is_index, 'sensitivity': sensitivity,
        'data_points': len(df),
        'optimal_trades': trades, 'optimal_total_return': total_return,
        'optimal_buy_hold_return': buy_hold_return, 'trade_count': len(trades),
        'kline_data': kline_data, 'buy_markers': buy_markers, 'sell_markers': sell_markers,
        'buy_correlation': buy_corr, 'sell_correlation': sell_corr,
        'model': model, 'probability_curve': prob_curve,
        'equity_curve': equity_curve, 'backtest_trades': bt_trades,
        'metrics': metrics, 'accuracy': accuracy,
        'weight_learning': weight_learning_info,
        'news_analysis': news_result,
        'capital_flow_analysis': capital_flow_result,
        'anomaly_signals': anomaly_result,
        'swing_detection': swing_result,
    }
