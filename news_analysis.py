#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公告与新闻事件研究模块
========================
功能：
  1. 自动检索个股公告和公开新闻（akshare多源）
  2. 区分盘中消息(9:30-15:00)与盘后消息(15:00-次日9:30)
  3. 事件研究法：计算消息发布后N日股价反应
  4. 新闻敏感度分析：不同类型消息的平均股价影响
  5. 持续学习：将学习结果存入learning模块，随每次分析更新
"""
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import learning

# A股交易时间
MARKET_OPEN = datetime.strptime('09:30', '%H:%M').time()
MARKET_CLOSE = datetime.strptime('15:00', '%H:%M').time()

# 新闻类型关键词分类
NEWS_TYPE_KEYWORDS = {
    '业绩预告': ['业绩预告', '业绩快报', '业绩修正', '预增', '预减', '扭亏', '首亏', '续亏', '续盈'],
    '增持回购': ['增持', '回购', '股东增持', '高管增持', '股权激励', '员工持股'],
    '减持': ['减持', '股东减持', '高管减持', '清仓减持', '拟减持'],
    '重组并购': ['重组', '并购', '收购', '借壳', '重大资产重组', '发行股份购买资产', '吸收合并'],
    '合同订单': ['中标', '合同', '订单', '战略合作', '框架协议', '重大合同'],
    '监管问询': ['问询函', '关注函', '监管函', '警示函', '立案', '行政处罚', '纪律处分', '监管措施', '警示'],
    '分红送转': ['分红', '派息', '送转', '高送转', '利润分配', '除权除息'],
    '停复牌': ['停牌', '复牌', '临时停牌', '终止上市', '恢复上市'],
    '股东变动': ['股权转让', '实际控制人变更', '控股股东变更', '表决权委托', '一致行动人'],
    '融资': ['定增', '非公开发行', '可转债', '发债', '配股', 'IPO', '上市'],
}


def classify_news_type(title):
    """根据标题关键词分类新闻类型"""
    if not title:
        return '其他'
    title = str(title)
    for news_type, keywords in NEWS_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in title:
                return news_type
    return '其他'


def classify_news_timing(dt):
    """
    区分盘中还是盘后消息。
    盘中：交易日 9:30-15:00
    盘后：交易日 15:00-次日9:30，以及非交易日全天
    返回: 'intraday' 或 'after_hours'
    """
    if dt is None:
        return 'after_hours'
    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt)
        except Exception:
            return 'after_hours'
    t = dt.time()
    weekday = dt.weekday()
    # 周末或节假日视为盘后
    if weekday >= 5:
        return 'after_hours'
    if MARKET_OPEN <= t <= MARKET_CLOSE:
        return 'intraday'
    return 'after_hours'


def fetch_news_akshare(code, start_date, end_date, max_items=200):
    """
    通过统一多数据源管理器获取个股新闻和公告。
    数据源优先级：AkShare-东财 → efinance-同花顺 → AkShare-新浪 → 巨潮公告
    返回 DataFrame: date, title, url, source, news_type, timing
    """
    import data_sources
    df = data_sources.fetch_news(code, start_date, end_date, max_items=max_items)
    # 记录数据源状态用于调试
    status = data_sources.get_source_status()
    for src, st in status.items():
        if not st['success']:
            print(f"[新闻] {src} 失败: {st['detail']}")
    return df


def event_study(df_price, news_events, holding_days=5):
    """
    事件研究法：计算每条新闻发布后holding_days日的股价反应。

    参数:
      df_price: 价格DataFrame（含date, close列）
      news_events: 新闻DataFrame（含date, title, news_type, timing列）
      holding_days: 持有天数

    返回:
      DataFrame: 每条新闻的事件研究结果
    """
    if news_events is None or len(news_events) == 0:
        return pd.DataFrame()

    df_price = df_price.copy()
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price = df_price.sort_values('date').reset_index(drop=True)
    closes = df_price['close'].values
    dates = df_price['date'].values

    results = []
    for _, news in news_events.iterrows():
        news_dt = pd.to_datetime(news['date'])
        # 盘后消息从下一个交易日开始计算
        if news['timing'] == 'after_hours':
            news_dt = news_dt + timedelta(days=1)
        # 找到新闻后的第一个交易日
        future_dates = df_price[df_price['date'] >= news_dt]
        if len(future_dates) < 2:
            continue
        start_idx = future_dates.index[0]
        end_idx = min(start_idx + holding_days, len(df_price) - 1)
        if end_idx <= start_idx:
            continue
        # 用事件日开盘价作为买入价（更真实），事件后N日收盘价作为卖出价
        buy_price = closes[start_idx]
        sell_price = closes[end_idx]
        ret = (sell_price - buy_price) / buy_price * 100
        # 最大回撤（事件期间）
        period_closes = closes[start_idx:end_idx + 1]
        peak = np.maximum.accumulate(period_closes)
        drawdown = np.min((period_closes - peak) / peak) * 100
        results.append({
            'news_date': news['date'],
            'title': news['title'],
            'news_type': news['news_type'],
            'timing': news['timing'],
            'source': news.get('source', ''),
            'holding_days': holding_days,
            'buy_price': round(buy_price, 2),
            'sell_price': round(sell_price, 2),
            'return_pct': round(ret, 4),
            'max_drawdown_pct': round(drawdown, 4),
        })
    return pd.DataFrame(results)


def analyze_news_sensitivity(code, df_price, start_date, end_date, holding_days=5):
    """
    完整的新闻敏感度分析流水线。

    返回:
      dict: 包含新闻列表、事件研究结果、敏感度统计、学习更新状态
    """
    print(f"\n[新闻分析] 开始分析 {code} {start_date}~{end_date}")

    # 1. 获取新闻
    try:
        news = fetch_news_akshare(code, start_date, end_date)
    except Exception as e:
        print(f"[新闻分析] 新闻获取失败: {e}")
        news = pd.DataFrame(columns=['date', 'title', 'url', 'source', 'news_type', 'timing'])

    if len(news) == 0:
        print("[新闻分析] 未获取到新闻数据")
        # 获取数据源详细状态用于诊断
        try:
            import data_sources
            source_status = data_sources.get_source_status_list()
        except Exception:
            source_status = []
        return {
            'news_count': 0, 'news_list': [], 'event_results': [],
            'sensitivity_by_type': [], 'sensitivity_by_timing': [],
            'learning_updated': False,
            'error': '未获取到新闻数据',
            'source_status': source_status
        }

    print(f"[新闻分析] 获取到 {len(news)} 条新闻/公告")

    # 2. 事件研究
    event_results = event_study(df_price, news, holding_days)
    print(f"[新闻分析] 完成 {len(event_results)} 条事件研究")

    # 3. 按新闻类型统计敏感度
    sensitivity_by_type = []
    if len(event_results) > 0:
        for ntype, group in event_results.groupby('news_type'):
            sensitivity_by_type.append({
                'news_type': ntype,
                'count': len(group),
                'avg_return': round(group['return_pct'].mean(), 4),
                'std_return': round(group['return_pct'].std(), 4) if len(group) > 1 else 0,
                'win_rate': round((group['return_pct'] > 0).mean() * 100, 2),
                'avg_drawdown': round(group['max_drawdown_pct'].mean(), 4),
                'best_return': round(group['return_pct'].max(), 4),
                'worst_return': round(group['return_pct'].min(), 4),
            })
        sensitivity_by_type.sort(key=lambda x: abs(x['avg_return']), reverse=True)

    # 4. 按盘中/盘后统计
    sensitivity_by_timing = []
    if len(event_results) > 0:
        for timing, group in event_results.groupby('timing'):
            sensitivity_by_timing.append({
                'timing': '盘中' if timing == 'intraday' else '盘后',
                'count': len(group),
                'avg_return': round(group['return_pct'].mean(), 4),
                'win_rate': round((group['return_pct'] > 0).mean() * 100, 2),
                'std_return': round(group['return_pct'].std(), 4) if len(group) > 1 else 0,
            })

    # 5. 持续学习：更新新闻敏感度参数
    learning_updated = False
    try:
        for _, er in event_results.iterrows():
            learning.update_news_sensitivity(
                code=code, news_type=er['news_type'], timing=er['timing'],
                price_reaction=er['return_pct'], holding_days=holding_days
            )
        learning_updated = True
        print(f"[新闻分析] 已更新学习参数（{len(event_results)}条事件）")
    except Exception as e:
        print(f"[新闻分析] 学习更新失败: {e}")

    # 6. 汇总新闻列表（限制条数）
    news_list = []
    for _, n in news.head(100).iterrows():
        news_list.append({
            'date': str(n['date']),
            'title': n['title'],
            'news_type': n['news_type'],
            'timing': '盘中' if n['timing'] == 'intraday' else '盘后',
            'source': n.get('source', ''),
            'category': n.get('category', '资讯'),
            'url': str(n.get('url', '')),
        })

    # 获取数据源状态
    try:
        import data_sources
        source_status = data_sources.get_source_status_list()
    except Exception:
        source_status = []

    return {
        'news_count': len(news),
        'news_list': news_list,
        'event_results': event_results.to_dict('records') if len(event_results) > 0 else [],
        'sensitivity_by_type': sensitivity_by_type,
        'sensitivity_by_timing': sensitivity_by_timing,
        'learning_updated': learning_updated,
        'holding_days': holding_days,
        'source_status': source_status,
    }
