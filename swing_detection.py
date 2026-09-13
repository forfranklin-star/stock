#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波段顶/底识别与参数优化模块
============================
多种算法识别波段顶和底，通过参数网格搜索+回测验证找到最优参数组合。

支持算法：
  1. 摆动高低点（Swing High/Low）— 左右N根K线极值
  2. 分形（Fractal）— 5根K线中间极值
  3. ZigZag — 价格回撤百分比转折点
  4. MACD背离 — 价格创新高/低但MACD不创新高/低
  5. RSI超买超卖+反转确认
  6. KDJ超买超卖+死叉/金叉
  7. 布林带触碰+反转
  8. 成交量确认（顶部放量滞涨/底部放量抗跌）
  9. 综合打分系统（多算法加权投票）

参数优化：
  - 摆动点左右K线数：2,3,5
  - ZigZag回撤：5%,8%,10%,15%
  - RSI阈值：70/30, 75/25
  - 综合打分阈值：3,4,5
  - 回测持有期：3,5,10日
"""
import numpy as np
import pandas as pd
from itertools import product


# ============================================================
# 算法1：摆动高低点（Swing High/Low）
# ============================================================

def detect_swing_points(df, left=3, right=3):
    """
    检测摆动高点和低点。
    摆动高点：中间K线高点 > 左右各left/right根K线的高点
    摆动低点：中间K线低点 < 左右各left/right根K线的低点

    返回: (swing_highs_indices, swing_lows_indices)
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    swing_highs = []
    swing_lows = []

    for i in range(left, n - right):
        # 摆动高点
        is_high = True
        for j in range(1, left + 1):
            if highs[i] <= highs[i - j]:
                is_high = False
                break
        if is_high:
            for j in range(1, right + 1):
                if highs[i] <= highs[i + j]:
                    is_high = False
                    break
        if is_high:
            swing_highs.append(i)

        # 摆动低点
        is_low = True
        for j in range(1, left + 1):
            if lows[i] >= lows[i - j]:
                is_low = False
                break
        if is_low:
            for j in range(1, right + 1):
                if lows[i] >= lows[i + j]:
                    is_low = False
                    break
        if is_low:
            swing_lows.append(i)

    return swing_highs, swing_lows


# ============================================================
# 算法2：分形（Fractal）
# ============================================================

def detect_fractals(df):
    """
    检测分形顶和分形底（5根K线）。
    分形顶：第3根K线高点 > 第1、2、4、5根K线高点
    分形底：第3根K线低点 < 第1、2、4、5根K线低点
    """
    return detect_swing_points(df, left=2, right=2)


# ============================================================
# 算法3：ZigZag转折点
# ============================================================

def detect_zigzag(df, retracement_pct=0.08):
    """
    ZigZag算法识别主要转折点。
    从一个极值点开始，价格反向变动超过retracement_pct则确认转折点。

    返回: (zigzag_highs_indices, zigzag_lows_indices)
    """
    closes = df['close'].values
    n = len(df)
    if n < 5:
        return [], []

    zigzag_highs = []
    zigzag_lows = []

    # 找到第一个极值点
    trend = 0  # 1=上升, -1=下降
    extreme_idx = 0
    extreme_price = closes[0]

    for i in range(1, n):
        if trend == 0:
            # 初始趋势判断
            if closes[i] > extreme_price * (1 + retracement_pct * 0.5):
                trend = 1
                extreme_idx = i
                extreme_price = closes[i]
            elif closes[i] < extreme_price * (1 - retracement_pct * 0.5):
                trend = -1
                extreme_idx = i
                extreme_price = closes[i]
            continue

        if trend == 1:
            # 上升趋势中，更新最高点
            if closes[i] > extreme_price:
                extreme_idx = i
                extreme_price = closes[i]
            # 回撤超过阈值，确认顶部
            elif closes[i] < extreme_price * (1 - retracement_pct):
                zigzag_highs.append(extreme_idx)
                trend = -1
                extreme_idx = i
                extreme_price = closes[i]
        else:
            # 下降趋势中，更新最低点
            if closes[i] < extreme_price:
                extreme_idx = i
                extreme_price = closes[i]
            # 反弹超过阈值，确认底部
            elif closes[i] > extreme_price * (1 + retracement_pct):
                zigzag_lows.append(extreme_idx)
                trend = 1
                extreme_idx = i
                extreme_price = closes[i]

    return zigzag_highs, zigzag_lows


# ============================================================
# 算法4：MACD背离
# ============================================================

def detect_macd_divergence(df, lookback=20):
    """
    检测MACD顶背离和底背离。
    顶背离：价格创新高但MACD_DIF不创新高
    底背离：价格创新低但MACD_DIF不创新低
    """
    # 兼容多种列名命名
    dif_col = None
    for col in ['MACD_DIF', 'DIF', 'macd_dif', 'dif']:
        if col in df.columns:
            dif_col = col
            break
    if dif_col is None:
        return [], []

    closes = df['close'].values
    dif = df[dif_col].values
    n = len(df)
    bearish_div = []  # 顶背离
    bullish_div = []  # 底背离

    for i in range(lookback, n):
        # 近期价格极值
        recent_high_idx = i - lookback + np.argmax(closes[i - lookback:i])
        recent_low_idx = i - lookback + np.argmin(closes[i - lookback:i])

        # 当前是否创新高
        if closes[i] >= closes[recent_high_idx] * 0.995:
            # 价格创新高，但DIF没有创新高 → 顶背离
            if dif[i] < dif[recent_high_idx] * 0.98:
                bearish_div.append(i)

        # 当前是否创新低
        if closes[i] <= closes[recent_low_idx] * 1.005:
            # 价格创新低，但DIF没有创新低 → 底背离
            if dif[i] > dif[recent_low_idx] * 1.02:
                bullish_div.append(i)

    return bearish_div, bullish_div


# ============================================================
# 算法5：RSI超买超卖+反转
# ============================================================

def detect_rsi_reversal(df, overbought=70, oversold=30):
    """
    检测RSI超买后回落（顶部信号）和超卖后回升（底部信号）。
    """
    rsi_col = None
    for col in ['RSI_14', 'RSI14', 'rsi_14', 'rsi14', 'RSI']:
        if col in df.columns:
            rsi_col = col
            break
    if rsi_col is None:
        return [], []

    rsi = df[rsi_col].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(2, n):
        # 顶部：RSI从超买区回落
        if rsi[i - 1] > overbought and rsi[i] < rsi[i - 1]:
            tops.append(i)
        # 底部：RSI从超卖区回升
        if rsi[i - 1] < oversold and rsi[i] > rsi[i - 1]:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 算法6：KDJ超买超卖+死叉/金叉
# ============================================================

def detect_kdj_signals(df, overbought=80, oversold=20):
    """
    检测KDJ超买死叉（顶部）和超卖金叉（底部）。
    """
    k_col = None
    d_col = None
    for col in ['KDJ_K', 'K', 'kdj_k', 'k']:
        if col in df.columns:
            k_col = col
            break
    for col in ['KDJ_D', 'D', 'kdj_d', 'd']:
        if col in df.columns:
            d_col = col
            break
    if k_col is None or d_col is None:
        return [], []

    k = df[k_col].values
    d = df[d_col].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(1, n):
        # 顶部：K在超买区死叉D
        if k[i - 1] > overbought and k[i] < d[i] and k[i - 1] >= d[i - 1]:
            tops.append(i)
        # 底部：K在超卖区金叉D
        if k[i - 1] < oversold and k[i] > d[i] and k[i - 1] <= d[i - 1]:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 算法7：布林带触碰+反转
# ============================================================

def detect_bollinger_reversal(df):
    """
    检测价格触碰布林上轨后回落（顶部）和触碰下轨后回升（底部）。
    """
    upper_col = None
    lower_col = None
    for col in ['BOLL_UPPER', 'BOLL_UP', 'boll_upper', 'boll_up', 'UPPER', 'UP']:
        if col in df.columns:
            upper_col = col
            break
    for col in ['BOLL_LOWER', 'BOLL_LOW', 'boll_lower', 'boll_low', 'LOWER', 'LOW']:
        if col in df.columns:
            lower_col = col
            break
    if upper_col is None or lower_col is None:
        return [], []

    close = df['close'].values
    upper = df[upper_col].values
    lower = df[lower_col].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(1, n):
        # 顶部：前一日触碰/突破上轨，今日回落
        if close[i - 1] >= upper[i - 1] * 0.99 and close[i] < close[i - 1]:
            tops.append(i)
        # 底部：前一日触碰/跌破下轨，今日回升
        if close[i - 1] <= lower[i - 1] * 1.01 and close[i] > close[i - 1]:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 算法8：成交量确认
# ============================================================

def detect_volume_confirmation(df, vol_ratio_threshold=1.5):
    """
    顶部放量滞涨和底部放量抗跌。
    """
    if 'volume' not in df.columns:
        return [], []

    close = df['close'].values
    volume = df['volume'].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(5, n):
        avg_vol = np.mean(volume[i - 5:i])
        if avg_vol == 0:
            continue
        vol_ratio = volume[i] / avg_vol
        pct_change = (close[i] - close[i - 1]) / close[i - 1] * 100 if close[i - 1] > 0 else 0

        # 顶部：放量但涨幅小（滞涨）
        if vol_ratio > vol_ratio_threshold and 0 <= pct_change < 1.0:
            tops.append(i)
        # 底部：放量但跌幅小（抗跌）
        if vol_ratio > vol_ratio_threshold and -1.0 < pct_change <= 0:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 综合打分系统
# ============================================================

def detect_composite_swing_points(df, params=None):
    """
    综合多种算法的加权打分系统识别波段顶/底。

    参数:
      params: dict，包含各算法权重和阈值
        - swing_left/right: 摆动点左右K线数
        - zigzag_pct: ZigZag回撤百分比
        - rsi_ob/os: RSI超买超卖阈值
        - kdj_ob/os: KDJ超买超卖阈值
        - vol_ratio: 量比阈值
        - top_threshold/bottom_threshold: 综合打分阈值
        - weights: 各算法权重

    返回: (top_indices, bottom_indices, top_scores, bottom_scores)
    """
    if params is None:
        params = {
            'swing_left': 3, 'swing_right': 3,
            'zigzag_pct': 0.08,
            'rsi_ob': 70, 'rsi_os': 30,
            'kdj_ob': 80, 'kdj_os': 20,
            'vol_ratio': 1.5,
            'top_threshold': 3, 'bottom_threshold': 3,
            'weights': {
                'swing': 2.0, 'fractal': 1.0, 'zigzag': 2.0,
                'macd_div': 1.5, 'rsi': 1.0, 'kdj': 1.0,
                'bollinger': 1.0, 'volume': 1.5,
            }
        }

    w = params['weights']
    n = len(df)
    top_scores = np.zeros(n)
    bottom_scores = np.zeros(n)

    # 1. 摆动点
    sh, sl = detect_swing_points(df, params['swing_left'], params['swing_right'])
    for i in sh:
        top_scores[i] += w['swing']
    for i in sl:
        bottom_scores[i] += w['swing']

    # 2. 分形
    fh, fl = detect_fractals(df)
    for i in fh:
        top_scores[i] += w['fractal']
    for i in fl:
        bottom_scores[i] += w['fractal']

    # 3. ZigZag
    zh, zl = detect_zigzag(df, params['zigzag_pct'])
    for i in zh:
        top_scores[i] += w['zigzag']
    for i in zl:
        bottom_scores[i] += w['zigzag']

    # 4. MACD背离
    mh, ml = detect_macd_divergence(df)
    for i in mh:
        top_scores[i] += w['macd_div']
    for i in ml:
        bottom_scores[i] += w['macd_div']

    # 5. RSI
    rh, rl = detect_rsi_reversal(df, params['rsi_ob'], params['rsi_os'])
    for i in rh:
        top_scores[i] += w['rsi']
    for i in rl:
        bottom_scores[i] += w['rsi']

    # 6. KDJ
    kh, kl = detect_kdj_signals(df, params['kdj_ob'], params['kdj_os'])
    for i in kh:
        top_scores[i] += w['kdj']
    for i in kl:
        bottom_scores[i] += w['kdj']

    # 7. 布林带
    bh, bl = detect_bollinger_reversal(df)
    for i in bh:
        top_scores[i] += w['bollinger']
    for i in bl:
        bottom_scores[i] += w['bollinger']

    # 8. 成交量
    vh, vl = detect_volume_confirmation(df, params['vol_ratio'])
    for i in vh:
        top_scores[i] += w['volume']
    for i in vl:
        bottom_scores[i] += w['volume']

    # 筛选超过阈值的点
    top_indices = [i for i in range(n) if top_scores[i] >= params['top_threshold']]
    bottom_indices = [i for i in range(n) if bottom_scores[i] >= params['bottom_threshold']]

    return top_indices, bottom_indices, top_scores, bottom_scores


# ============================================================
# 回测验证
# ============================================================

def backtest_swing_points(df, top_indices, bottom_indices, holding_periods=[3, 5, 10]):
    """
    回测波段顶/底识别的准确率。

    顶部信号后N日：下跌概率、平均跌幅、最大跌幅
    底部信号后N日：上涨概率、平均涨幅、最大涨幅
    """
    closes = df['close'].values
    n = len(df)
    results = {'tops': {}, 'bottoms': {}}

    for days in holding_periods:
        # 顶部回测
        top_rets = []
        for i in top_indices:
            if i + days < n:
                ret = (closes[i + days] - closes[i]) / closes[i] * 100
                top_rets.append(ret)

        if top_rets:
            results['tops'][f'{days}d'] = {
                'count': len(top_rets),
                'decline_rate': round(sum(1 for r in top_rets if r < 0) / len(top_rets) * 100, 1),
                'avg_return': round(np.mean(top_rets), 2),
                'avg_decline': round(np.mean([r for r in top_rets if r < 0]), 2) if any(r < 0 for r in top_rets) else 0,
                'max_decline': round(min(top_rets), 2),
                'max_rise': round(max(top_rets), 2),
            }

        # 底部回测
        bottom_rets = []
        for i in bottom_indices:
            if i + days < n:
                ret = (closes[i + days] - closes[i]) / closes[i] * 100
                bottom_rets.append(ret)

        if bottom_rets:
            results['bottoms'][f'{days}d'] = {
                'count': len(bottom_rets),
                'rise_rate': round(sum(1 for r in bottom_rets if r > 0) / len(bottom_rets) * 100, 1),
                'avg_return': round(np.mean(bottom_rets), 2),
                'avg_rise': round(np.mean([r for r in bottom_rets if r > 0]), 2) if any(r > 0 for r in bottom_rets) else 0,
                'max_rise': round(max(bottom_rets), 2),
                'max_decline': round(min(bottom_rets), 2),
            }

    return results


# ============================================================
# 参数网格搜索
# ============================================================

def grid_search_params(df, holding_days=5, max_combos=50):
    """
    参数网格搜索，找到识别率最高的参数组合。

    搜索空间：
      - swing_left/right: 2,3,5
      - zigzag_pct: 0.05, 0.08, 0.10, 0.15
      - top/bottom_threshold: 2,3,4,5
      - rsi_ob/os: (70,30), (75,25)
      - kdj_ob/os: (80,20), (85,15)

    评估指标：顶部下跌率 + 底部上涨率（加权平均）
    """
    # 参数网格
    swing_vals = [2, 3, 5]
    zigzag_vals = [0.05, 0.08, 0.10, 0.15]
    threshold_vals = [2, 3, 4, 5]
    rsi_vals = [(70, 30), (75, 25)]
    kdj_vals = [(80, 20), (85, 15)]

    # 生成所有组合（限制数量）
    all_combos = list(product(swing_vals, zigzag_vals, threshold_vals, rsi_vals, kdj_vals))
    if len(all_combos) > max_combos:
        # 均匀采样
        step = len(all_combos) // max_combos
        all_combos = all_combos[::step][:max_combos]

    results = []
    for swing_lr, zz_pct, thresh, (rsi_ob, rsi_os), (kdj_ob, kdj_os) in all_combos:
        params = {
            'swing_left': swing_lr, 'swing_right': swing_lr,
            'zigzag_pct': zz_pct,
            'rsi_ob': rsi_ob, 'rsi_os': rsi_os,
            'kdj_ob': kdj_ob, 'kdj_os': kdj_os,
            'vol_ratio': 1.5,
            'top_threshold': thresh, 'bottom_threshold': thresh,
            'weights': {
                'swing': 2.0, 'fractal': 1.0, 'zigzag': 2.0,
                'macd_div': 1.5, 'rsi': 1.0, 'kdj': 1.0,
                'bollinger': 1.0, 'volume': 1.5,
            }
        }

        try:
            tops, bottoms, _, _ = detect_composite_swing_points(df, params)
            bt = backtest_swing_points(df, tops, bottoms, [holding_days])

            top_rate = bt['tops'].get(f'{holding_days}d', {}).get('decline_rate', 0)
            bottom_rate = bt['bottoms'].get(f'{holding_days}d', {}).get('rise_rate', 0)
            top_count = bt['tops'].get(f'{holding_days}d', {}).get('count', 0)
            bottom_count = bt['bottoms'].get(f'{holding_days}d', {}).get('count', 0)

            # 综合评分：顶部下跌率和底部上涨率的加权平均，信号数量适中
            total_signals = top_count + bottom_count
            if total_signals > 0:
                # 信号数量惩罚：太多信号（>数据量10%）扣分
                signal_penalty = max(0, (total_signals / len(df) - 0.1) * 100)
                # 数量太少（<5）也扣分
                count_penalty = max(0, (5 - min(top_count, bottom_count)) * 2)
                score = (top_rate + bottom_rate) / 2 - signal_penalty - count_penalty
            else:
                score = -100

            results.append({
                'params': params,
                'top_rate': top_rate,
                'bottom_rate': bottom_rate,
                'top_count': top_count,
                'bottom_count': bottom_count,
                'score': round(score, 2),
            })
        except Exception:
            continue

    # 按评分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ============================================================
# 主入口：完整波段分析
# ============================================================

def run_swing_analysis(df, timeframe='daily', holding_days=5):
    """
    完整的波段顶/底分析流程。

    1. 使用默认参数检测波段点
    2. 参数网格搜索找到最优参数
    3. 使用最优参数重新检测
    4. 回测验证
    5. 返回详细结果
    """
    if df is None or len(df) < 30:
        return {'error': f'数据不足（仅{len(df) if df is not None else 0}根K线），至少需要30根'}

    df = df.reset_index(drop=True)
    data_warning = None
    if len(df) < 60:
        data_warning = f'K线数据较少（{len(df)}根），部分长周期指标可能不可用，识别精度可能降低'

    # 根据数据量自动调整参数
    if len(df) < 60:
        default_swing = 2
        default_zigzag = 0.05
        default_threshold = 2
        max_grid_combos = 15
    else:
        default_swing = 3
        default_zigzag = 0.08
        default_threshold = 3
        max_grid_combos = 30 if timeframe == '60min' else 50

    # 1. 默认参数检测
    default_params = {
        'swing_left': default_swing, 'swing_right': default_swing,
        'zigzag_pct': default_zigzag,
        'rsi_ob': 70, 'rsi_os': 30,
        'kdj_ob': 80, 'kdj_os': 20,
        'vol_ratio': 1.5,
        'top_threshold': default_threshold, 'bottom_threshold': default_threshold,
        'weights': {
            'swing': 2.0, 'fractal': 1.0, 'zigzag': 2.0,
            'macd_div': 1.5, 'rsi': 1.0, 'kdj': 1.0,
            'bollinger': 1.0, 'volume': 1.5,
        }
    }

    default_tops, default_bottoms, default_top_scores, default_bottom_scores = \
        detect_composite_swing_points(df, default_params)
    default_bt = backtest_swing_points(df, default_tops, default_bottoms, [3, 5, 10])

    # 2. 参数网格搜索（限制组合数量，60分钟图数据多可以多搜一些）
    max_combos = 30 if timeframe == '60min' else 50
    grid_results = grid_search_params(df, holding_days=holding_days, max_combos=max_combos)

    # 3. 使用最优参数重新检测
    best_result = grid_results[0] if grid_results else None
    best_params = best_result['params'] if best_result else default_params

    best_tops, best_bottoms, best_top_scores, best_bottom_scores = \
        detect_composite_swing_points(df, best_params)
    best_bt = backtest_swing_points(df, best_tops, best_bottoms, [3, 5, 10, 20])

    # 4. 整理波段点详情
    def format_points(indices, scores, df):
        points = []
        for i in indices:
            points.append({
                'date': str(df.iloc[i]['date']),
                'index': i,
                'price': round(float(df.iloc[i]['close']), 2),
                'high': round(float(df.iloc[i]['high']), 2),
                'low': round(float(df.iloc[i]['low']), 2),
                'score': round(float(scores[i]), 2),
            })
        return points

    top_points = format_points(best_tops, best_top_scores, df)
    bottom_points = format_points(best_bottoms, best_bottom_scores, df)

    # 5. 各算法单独表现（用于展示）
    algo_performance = {}
    algos = {
        '摆动点(3,3)': lambda: detect_swing_points(df, 3, 3),
        '分形': lambda: detect_fractals(df),
        'ZigZag(8%)': lambda: detect_zigzag(df, 0.08),
        'MACD背离': lambda: detect_macd_divergence(df),
        'RSI反转': lambda: detect_rsi_reversal(df),
        'KDJ信号': lambda: detect_kdj_signals(df),
        '布林反转': lambda: detect_bollinger_reversal(df),
        '量价确认': lambda: detect_volume_confirmation(df),
    }

    for name, func in algos.items():
        try:
            tops, bottoms = func()
            bt = backtest_swing_points(df, tops, bottoms, [holding_days])
            algo_performance[name] = {
                'top_count': len(tops),
                'bottom_count': len(bottoms),
                'top_decline_rate': bt['tops'].get(f'{holding_days}d', {}).get('decline_rate', 0),
                'bottom_rise_rate': bt['bottoms'].get(f'{holding_days}d', {}).get('rise_rate', 0),
            }
        except Exception:
            algo_performance[name] = {'error': '计算失败'}

    # 6. 最优参数对比
    params_comparison = []
    for r in grid_results[:10]:
        p = r['params']
        params_comparison.append({
            'score': r['score'],
            'top_rate': r['top_rate'],
            'bottom_rate': r['bottom_rate'],
            'top_count': r['top_count'],
            'bottom_count': r['bottom_count'],
            'swing': p['swing_left'],
            'zigzag': f"{p['zigzag_pct']*100:.0f}%",
            'threshold': p['top_threshold'],
            'rsi': f"{p['rsi_ob']}/{p['rsi_os']}",
            'kdj': f"{p['kdj_ob']}/{p['kdj_os']}",
        })

    return {
        'timeframe': timeframe,
        'default_params': default_params,
        'best_params': best_params,
        'default_backtest': default_bt,
        'best_backtest': best_bt,
        'top_points': top_points,
        'bottom_points': bottom_points,
        'top_count': len(top_points),
        'bottom_count': len(bottom_points),
        'algo_performance': algo_performance,
        'grid_search_top10': params_comparison,
        'holding_days': holding_days,
        'current_state': calculate_current_swing_probability(df),
    }


# ============================================================
# 当前波段状态识别（无未来函数）
# ============================================================

def calculate_current_swing_probability(df, lookback=20):
    """
    计算当前时刻处于波段顶部/底部的概率（无未来函数，仅用当前和历史数据）。

    纳入12个维度：
      1. 价格位置（距N日高低点、回撤幅度）
      2. 当日涨跌幅度（大阳/大阴、连续涨跌天数）
      3. 波动幅度（当日振幅、ATR、波动率）
      4. 量能（量比、成交量变化、缩量/放量）
      5. RSI（超买超卖、偏低偏高档位）
      6. KDJ（超买超卖、偏低偏高档位）
      7. MACD（零轴上下、金叉死叉、背离）
      8. 布林带（触碰上下轨、%b位置、带宽）
      9. 均线系统（多空排列、乖离率、偏离度）
      10. CCI/WR/MFI（多指标超买超卖确认）
      11. PSY心理线（上涨天数比例）
      12. K线形态（长上影/下影、十字星、吞没）

    返回: dict
    """
    if df is None or len(df) < 10:
        return {
            'top_probability': 50, 'bottom_probability': 50,
            'state': '数据不足', 'top_factors': [], 'bottom_factors': [],
            'top_score': 0, 'bottom_score': 0,
        }

    df = df.reset_index(drop=True)
    i = len(df) - 1

    close = _safe_get_swing(df, 'close', i)
    high = _safe_get_swing(df, 'high', i)
    low = _safe_get_swing(df, 'low', i)
    open_p = _safe_get_swing(df, 'open', i)
    volume = _safe_get_swing(df, 'volume', i)
    prev_close = _safe_get_swing(df, 'close', i - 1) if i > 0 else close

    if any(pd.isna(x) for x in [close, high, low, open_p]):
        return {
            'top_probability': 50, 'bottom_probability': 50,
            'state': '数据不足', 'top_factors': [], 'bottom_factors': [],
            'top_score': 0, 'bottom_score': 0,
        }

    top_score = 0
    bottom_score = 0
    top_factors = []
    bottom_factors = []

    # ============================================================
    # 1. 价格位置
    # ============================================================
    if i >= lookback:
        recent_high = df['high'].iloc[i - lookback:i + 1].max()
        recent_low = df['low'].iloc[i - lookback:i + 1].min()
        if recent_high > 0 and recent_low > 0:
            dist_to_high = (recent_high - close) / recent_high * 100
            dist_to_low = (close - recent_low) / recent_low * 100
            drawdown = (recent_high - close) / recent_high * 100  # 从高点回撤
            rebound = (close - recent_low) / recent_low * 100  # 从低点反弹

            # 顶部：接近高点
            if dist_to_high < 2:
                top_score += 18; top_factors.append(f"接近{lookback}日高点(距{dist_to_high:.1f}%)")
            elif dist_to_high < 5:
                top_score += 10; top_factors.append(f"靠近{lookback}日高点(距{dist_to_high:.1f}%)")

            # 底部：接近低点
            if dist_to_low < 2:
                bottom_score += 18; bottom_factors.append(f"接近{lookback}日低点(距{dist_to_low:.1f}%)")
            elif dist_to_low < 5:
                bottom_score += 10; bottom_factors.append(f"靠近{lookback}日低点(距{dist_to_low:.1f}%)")

            # 顶部：从低点反弹幅度大
            if rebound > 30:
                top_score += 10; top_factors.append(f"从低点反弹{rebound:.0f}%(累积涨幅大)")
            elif rebound > 20:
                top_score += 5; top_factors.append(f"从低点反弹{rebound:.0f}%")

            # 底部：从高点回撤幅度大
            if drawdown > 30:
                bottom_score += 10; bottom_factors.append(f"从高点回撤{drawdown:.0f}%(累积跌幅大)")
            elif drawdown > 20:
                bottom_score += 5; bottom_factors.append(f"从高点回撤{drawdown:.0f}%")

    # ============================================================
    # 2. 当日涨跌幅度 + 连续涨跌
    # ============================================================
    if prev_close and prev_close > 0:
        pct_change = (close - prev_close) / prev_close * 100

        # 大阳线（顶部信号：暴涨后可能见顶）
        if pct_change > 7:
            top_score += 12; top_factors.append(f"大阳线(+{pct_change:.1f}%，短期过热)")
        elif pct_change > 5:
            top_score += 7; top_factors.append(f"中阳线(+{pct_change:.1f}%)")

        # 大阴线（底部信号：暴跌后可能见底）
        if pct_change < -7:
            bottom_score += 12; bottom_factors.append(f"大阴线({pct_change:.1f}%，恐慌抛售)")
        elif pct_change < -5:
            bottom_score += 7; bottom_factors.append(f"中阴线({pct_change:.1f}%)")

        # 小阴小阳（顶部滞涨/底部抗跌）
        if abs(pct_change) < 1:
            if i >= lookback:
                recent_high = df['high'].iloc[i - lookback:i].max()
                recent_low = df['low'].iloc[i - lookback:i].min()
                if close > recent_high * 0.95:
                    top_score += 6; top_factors.append(f"高位横盘(涨跌{pct_change:+.1f}%，滞涨)")
                if close < recent_low * 1.05:
                    bottom_score += 6; bottom_factors.append(f"低位横盘(涨跌{pct_change:+.1f}%，抗跌)")

    # 连续涨跌天数
    if i >= 5:
        up_days = sum(1 for j in range(i - 4, i + 1) if df.iloc[j]['close'] > df.iloc[j - 1]['close'])
        down_days = 5 - up_days
        if up_days >= 4:
            top_score += 8; top_factors.append(f"连续上涨({up_days}/5日，多头过热)")
        if down_days >= 4:
            bottom_score += 8; bottom_factors.append(f"连续下跌({down_days}/5日，空头衰竭)")

    # ============================================================
    # 3. 波动幅度
    # ============================================================
    if prev_close and prev_close > 0:
        amplitude = (high - low) / prev_close * 100  # 当日振幅
        if amplitude > 8:
            top_score += 6; top_factors.append(f"高振幅({amplitude:.1f}%，多空分歧大)")
            bottom_score += 4; bottom_factors.append(f"高振幅({amplitude:.1f}%，可能探底)")
        elif amplitude > 5:
            top_score += 3; top_factors.append(f"振幅较大({amplitude:.1f}%)")

    # ATR（平均真实波幅）
    atr = _safe_get_swing(df, 'ATR14', i)
    if pd.notna(atr) and prev_close and prev_close > 0:
        atr_pct = atr / prev_close * 100
        if atr_pct > 6:
            top_score += 4; top_factors.append(f"ATR高({atr_pct:.1f}%，波动率放大)")

    # ============================================================
    # 4. 量能
    # ============================================================
    if pd.notna(volume) and i >= 5:
        avg_vol5 = df['volume'].iloc[i - 5:i].mean()
        if avg_vol5 > 0:
            vol_ratio = volume / avg_vol5
            pct_change = (close - prev_close) / prev_close * 100 if prev_close and prev_close > 0 else 0

            # 放量滞涨（顶部）
            if vol_ratio > 2 and 0 <= pct_change < 2:
                top_score += 15; top_factors.append(f"放量滞涨(量比{vol_ratio:.1f}，涨{pct_change:.1f}%)")
            elif vol_ratio > 1.5 and abs(pct_change) < 1:
                if pct_change >= 0:
                    top_score += 10; top_factors.append(f"放量滞涨(量比{vol_ratio:.1f})")
                else:
                    bottom_score += 10; bottom_factors.append(f"放量抗跌(量比{vol_ratio:.1f})")

            # 放量大跌（底部：恐慌抛售）
            if vol_ratio > 1.5 and pct_change < -3:
                bottom_score += 8; bottom_factors.append(f"放量大跌(量比{vol_ratio:.1f}，跌{pct_change:.1f}%，恐慌盘)")

            # 缩量下跌（底部：抛压衰竭）
            if vol_ratio < 0.7 and pct_change < 0:
                bottom_score += 8; bottom_factors.append(f"缩量下跌(量比{vol_ratio:.1f}，抛压衰竭)")

            # 缩量上涨（顶部：买盘不足）
            if vol_ratio < 0.7 and pct_change > 0:
                top_score += 6; top_factors.append(f"缩量上涨(量比{vol_ratio:.1f}，买盘不足)")

    # ============================================================
    # 5. RSI
    # ============================================================
    rsi = _safe_get_swing(df, 'RSI14', i) or _safe_get_swing(df, 'RSI_14', i)
    if pd.notna(rsi):
        if rsi > 80:
            top_score += 18; top_factors.append(f"RSI严重超买({rsi:.1f})")
        elif rsi > 70:
            top_score += 12; top_factors.append(f"RSI超买({rsi:.1f})")
        elif rsi > 60:
            top_score += 5; top_factors.append(f"RSI偏高({rsi:.1f})")

        if rsi < 20:
            bottom_score += 18; bottom_factors.append(f"RSI严重超卖({rsi:.1f})")
        elif rsi < 30:
            bottom_score += 12; bottom_factors.append(f"RSI超卖({rsi:.1f})")
        elif rsi < 40:
            bottom_score += 5; bottom_factors.append(f"RSI偏低({rsi:.1f})")

    # ============================================================
    # 6. KDJ
    # ============================================================
    kdj_k = _safe_get_swing(df, 'K', i) or _safe_get_swing(df, 'KDJ_K', i)
    if pd.notna(kdj_k):
        if kdj_k > 90:
            top_score += 15; top_factors.append(f"KDJ严重超买(K={kdj_k:.1f})")
        elif kdj_k > 80:
            top_score += 10; top_factors.append(f"KDJ超买(K={kdj_k:.1f})")
        elif kdj_k > 70:
            top_score += 5; top_factors.append(f"KDJ偏高(K={kdj_k:.1f})")

        if kdj_k < 10:
            bottom_score += 15; bottom_factors.append(f"KDJ严重超卖(K={kdj_k:.1f})")
        elif kdj_k < 20:
            bottom_score += 10; bottom_factors.append(f"KDJ超卖(K={kdj_k:.1f})")
        elif kdj_k < 30:
            bottom_score += 5; bottom_factors.append(f"KDJ偏低(K={kdj_k:.1f})")

    # ============================================================
    # 7. MACD
    # ============================================================
    dif = _safe_get_swing(df, 'DIF', i) or _safe_get_swing(df, 'MACD_DIF', i)
    dea = _safe_get_swing(df, 'DEA', i) or _safe_get_swing(df, 'MACD_DEA', i)
    dif_prev = _safe_get_swing(df, 'DIF', i - 1) or _safe_get_swing(df, 'MACD_DIF', i - 1)
    if pd.notna(dif):
        # 零轴上方且走弱（顶部）
        if dif > 0 and pd.notna(dif_prev) and dif < dif_prev:
            top_score += 10; top_factors.append(f"MACD高位走弱(DIF={dif:.2f})")
        # 零轴下方且走强（底部）
        if dif < 0 and pd.notna(dif_prev) and dif > dif_prev:
            bottom_score += 10; bottom_factors.append(f"MACD低位走强(DIF={dif:.2f})")
        # 死叉（顶部）
        if pd.notna(dea) and pd.notna(dif_prev):
            dea_prev = _safe_get_swing(df, 'DEA', i - 1) or _safe_get_swing(df, 'MACD_DEA', i - 1)
            if pd.notna(dea_prev) and dif_prev >= dea_prev and dif < dea:
                top_score += 8; top_factors.append("MACD死叉")
            if pd.notna(dea_prev) and dif_prev <= dea_prev and dif > dea:
                bottom_score += 8; bottom_factors.append("MACD金叉")

    # ============================================================
    # 8. 布林带
    # ============================================================
    boll_up = _safe_get_swing(df, 'BOLL_UP', i) or _safe_get_swing(df, 'BOLL_UPPER', i)
    boll_low = _safe_get_swing(df, 'BOLL_LOW', i) or _safe_get_swing(df, 'BOLL_LOWER', i)
    boll_mid = _safe_get_swing(df, 'BOLL_MID', i)
    pctb = _safe_get_swing(df, 'BOLL_PCTB', i)

    if pd.notna(boll_up) and close >= boll_up * 0.98:
        top_score += 12; top_factors.append(f"触碰布林上轨(收盘{close:.2f})")
    if pd.notna(boll_low) and close <= boll_low * 1.02:
        bottom_score += 12; bottom_factors.append(f"触碰布林下轨(收盘{close:.2f})")

    # 布林%b位置
    if pd.notna(pctb):
        if pctb > 0.9:
            top_score += 8; top_factors.append(f"布林%b极高({pctb:.2f})")
        elif pctb > 0.8:
            top_score += 4; top_factors.append(f"布林%b偏高({pctb:.2f})")
        if pctb < 0.1:
            bottom_score += 8; bottom_factors.append(f"布林%b极低({pctb:.2f})")
        elif pctb < 0.2:
            bottom_score += 4; bottom_factors.append(f"布林%b偏低({pctb:.2f})")

    # ============================================================
    # 9. 均线系统
    # ============================================================
    ma5 = _safe_get_swing(df, 'MA5', i)
    ma10 = _safe_get_swing(df, 'MA10', i)
    ma20 = _safe_get_swing(df, 'MA20', i)
    ma60 = _safe_get_swing(df, 'MA60', i)

    # 多头排列（顶部风险）
    if all(pd.notna(x) for x in [ma5, ma10, ma20]) and ma5 > ma10 > ma20:
        if ma20 and ma20 > 0 and (close - ma20) / ma20 * 100 > 20:
            top_score += 10; top_factors.append(f"多头排列+偏离MA20过远(+{(close-ma20)/ma20*100:.1f}%)")
        elif ma20 and ma20 > 0 and (close - ma20) / ma20 * 100 > 10:
            top_score += 5; top_factors.append(f"多头排列+偏离MA20(+{(close-ma20)/ma20*100:.1f}%)")

    # 空头排列（底部机会）
    if all(pd.notna(x) for x in [ma5, ma10, ma20]) and ma5 < ma10 < ma20:
        if ma20 and ma20 > 0 and (ma20 - close) / ma20 * 100 > 20:
            bottom_score += 10; bottom_factors.append(f"空头排列+偏离MA20过远(-{(ma20-close)/ma20*100:.1f}%)")
        elif ma20 and ma20 > 0 and (ma20 - close) / ma20 * 100 > 10:
            bottom_score += 5; bottom_factors.append(f"空头排列+偏离MA20(-{(ma20-close)/ma20*100:.1f}%)")

    # 乖离率BIAS
    bias20 = _safe_get_swing(df, 'BIAS20', i)
    if pd.notna(bias20):
        if bias20 > 15:
            top_score += 8; top_factors.append(f"BIAS20过高({bias20:.1f}%)")
        elif bias20 > 10:
            top_score += 4; top_factors.append(f"BIAS20偏高({bias20:.1f}%)")
        if bias20 < -15:
            bottom_score += 8; bottom_factors.append(f"BIAS20过低({bias20:.1f}%)")
        elif bias20 < -10:
            bottom_score += 4; bottom_factors.append(f"BIAS20偏低({bias20:.1f}%)")

    # ============================================================
    # 10. CCI / WR / MFI 多指标确认
    # ============================================================
    cci = _safe_get_swing(df, 'CCI', i)
    if pd.notna(cci):
        if cci > 200:
            top_score += 10; top_factors.append(f"CCI严重超买({cci:.0f})")
        elif cci > 100:
            top_score += 6; top_factors.append(f"CCI超买({cci:.0f})")
        if cci < -200:
            bottom_score += 10; bottom_factors.append(f"CCI严重超卖({cci:.0f})")
        elif cci < -100:
            bottom_score += 6; bottom_factors.append(f"CCI超卖({cci:.0f})")

    wr = _safe_get_swing(df, 'WR', i)
    if pd.notna(wr):
        if wr < -80:
            bottom_score += 8; bottom_factors.append(f"WR超卖({wr:.0f})")
        elif wr < -60:
            bottom_score += 4; bottom_factors.append(f"WR偏低({wr:.0f})")
        if wr > -20:
            top_score += 8; top_factors.append(f"WR超买({wr:.0f})")
        elif wr > -40:
            top_score += 4; top_factors.append(f"WR偏高({wr:.0f})")

    mfi = _safe_get_swing(df, 'MFI14', i)
    if pd.notna(mfi):
        if mfi > 80:
            top_score += 10; top_factors.append(f"MFI超买({mfi:.1f}，资金过热)")
        elif mfi > 70:
            top_score += 5; top_factors.append(f"MFI偏高({mfi:.1f})")
        if mfi < 20:
            bottom_score += 10; bottom_factors.append(f"MFI超卖({mfi:.1f}，资金枯竭)")
        elif mfi < 30:
            bottom_score += 5; bottom_factors.append(f"MFI偏低({mfi:.1f})")

    # ============================================================
    # 11. PSY心理线
    # ============================================================
    psy = _safe_get_swing(df, 'PSY12', i)
    if pd.notna(psy):
        if psy > 75:
            top_score += 8; top_factors.append(f"PSY过高({psy:.0f}，市场情绪过热)")
        elif psy > 65:
            top_score += 4; top_factors.append(f"PSY偏高({psy:.0f})")
        if psy < 25:
            bottom_score += 8; bottom_factors.append(f"PSY过低({psy:.0f}，市场情绪冰点)")
        elif psy < 35:
            bottom_score += 4; bottom_factors.append(f"PSY偏低({psy:.0f})")

    # ============================================================
    # 12. K线形态
    # ============================================================
    body = abs(close - open_p)
    if body > 0:
        upper_shadow = (high - max(close, open_p)) / body
        lower_shadow = (min(close, open_p) - low) / body

        if upper_shadow > 3:
            top_score += 12; top_factors.append(f"极长上影线(上影/实体={upper_shadow:.1f}，抛压重)")
        elif upper_shadow > 2:
            top_score += 7; top_factors.append(f"长上影线(上影/实体={upper_shadow:.1f})")

        if lower_shadow > 3:
            bottom_score += 12; bottom_factors.append(f"极长下影线(下影/实体={lower_shadow:.1f}，承接强)")
        elif lower_shadow > 2:
            bottom_score += 7; bottom_factors.append(f"长下影线(下影/实体={lower_shadow:.1f})")

    # 十字星（高位十字星见顶，低位十字星见底）
    if body > 0 and body / close < 0.005:
        if i >= lookback:
            recent_high = df['high'].iloc[i - lookback:i].max()
            recent_low = df['low'].iloc[i - lookback:i].min()
            if close > recent_high * 0.9:
                top_score += 6; top_factors.append("高位十字星(变盘信号)")
            if close < recent_low * 1.1:
                bottom_score += 6; bottom_factors.append("低位十字星(变盘信号)")

    # ============================================================
    # 归一化与状态判断
    # ============================================================
    max_score = 180  # 理论最高分约180（12个维度）
    top_prob = min(top_score / max_score * 100, 100)
    bottom_prob = min(bottom_score / max_score * 100, 100)

    if top_prob >= 45 and top_prob > bottom_prob:
        state = '顶部区域（注意回调风险）'
    elif bottom_prob >= 45 and bottom_prob > top_prob:
        state = '底部区域（关注反弹机会）'
    elif top_prob > bottom_prob + 12:
        state = '偏顶部（谨慎追高）'
    elif bottom_prob > top_prob + 12:
        state = '偏底部（可考虑低吸）'
    else:
        if all(pd.notna(x) for x in [ma5, ma20]) and ma5 > ma20:
            state = '上升趋势中（持有为主）'
        elif all(pd.notna(x) for x in [ma5, ma20]) and ma5 < ma20:
            state = '下降趋势中（观望为主）'
        else:
            state = '震荡整理（方向不明）'

    return {
        'top_probability': round(top_prob, 1),
        'bottom_probability': round(bottom_prob, 1),
        'state': state,
        'top_factors': top_factors,
        'bottom_factors': bottom_factors,
        'top_score': round(top_score, 1),
        'bottom_score': round(bottom_score, 1),
        'current_price': round(close, 2),
        'current_date': str(df.iloc[i]['date'])[:10],
        'dimensions': {
            'price_position': '价格位置',
            'price_change': '涨跌幅度',
            'volatility': '波动幅度',
            'volume': '量能',
            'rsi': 'RSI',
            'kdj': 'KDJ',
            'macd': 'MACD',
            'bollinger': '布林带',
            'ma': '均线系统',
            'cci_wr_mfi': 'CCI/WR/MFI',
            'psy': 'PSY心理线',
            'candlestick': 'K线形态',
        }
    }


def _safe_get_swing(df, col, idx, default=np.nan):
    """安全获取DataFrame某列某行的值"""
    if col in df.columns and 0 <= idx < len(df):
        val = df.iloc[idx][col]
        return val if pd.notna(val) else default
    return default
