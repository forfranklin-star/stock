#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常信号捕捉模块
================
检测"应跌未跌"和"应涨未涨"的异常信号，结合量价关系和资金流数据，
判断是否有主力资金进入或离场。

信号类型：
  - 应跌未跌（看涨背离）：技术面看跌但价格抗跌+放量 → 主力吸筹信号
  - 应涨未涨（看跌背离）：技术面看涨但价格滞涨+放量 → 主力出货信号

检测维度：
  1. 技术指标背离（MACD/RSI/KDJ/均线/布林带）
  2. 量价关系（放量滞涨/缩量抗跌/长上影/长下影）
  3. 资金流验证（主力净流入/大单流向）
  4. 学习规律加权（基于历史学习参数调整信号强度）
"""
import numpy as np
import pandas as pd
from datetime import datetime


def _safe_get(df, col, idx, default=np.nan):
    """安全获取DataFrame某列某行的值"""
    if col in df.columns and 0 <= idx < len(df):
        val = df.iloc[idx][col]
        return val if pd.notna(val) else default
    return default


def detect_bearish_divergence(df, idx, lookback=5):
    """
    检测"应跌"信号：技术面显示应该下跌。
    返回: (触发数量, 触发详情列表)
    """
    triggers = []
    close = _safe_get(df, 'close', idx)
    if pd.isna(close):
        return 0, triggers

    # 1. MACD死叉或DIF向下
    macd_dif = _safe_get(df, 'MACD_DIF', idx)
    macd_dea = _safe_get(df, 'MACD_DEA', idx)
    macd_dif_prev = _safe_get(df, 'MACD_DIF', idx - 1)
    if pd.notna(macd_dif) and pd.notna(macd_dea) and pd.notna(macd_dif_prev):
        if macd_dif < macd_dea or (macd_dif < macd_dif_prev and macd_dif > 0):
            triggers.append(f"MACD走弱(DIF={macd_dif:.3f})")

    # 2. RSI超买回落
    rsi = _safe_get(df, 'RSI_14', idx)
    rsi_prev = _safe_get(df, 'RSI_14', idx - 1)
    if pd.notna(rsi) and pd.notna(rsi_prev):
        if rsi > 70 or (rsi_prev > 70 and rsi < rsi_prev):
            triggers.append(f"RSI超买/回落({rsi:.1f})")

    # 3. KDJ死叉
    kdj_k = _safe_get(df, 'KDJ_K', idx)
    kdj_d = _safe_get(df, 'KDJ_D', idx)
    kdj_k_prev = _safe_get(df, 'KDJ_K', idx - 1)
    kdj_d_prev = _safe_get(df, 'KDJ_D', idx - 1)
    if all(pd.notna(x) for x in [kdj_k, kdj_d, kdj_k_prev, kdj_d_prev]):
        if kdj_k < kdj_d or (kdj_k_prev > kdj_d_prev and kdj_k < kdj_d):
            triggers.append(f"KDJ死叉(K={kdj_k:.1f},D={kdj_d:.1f})")

    # 4. 跌破短期均线
    ma5 = _safe_get(df, 'MA5', idx)
    ma10 = _safe_get(df, 'MA10', idx)
    if pd.notna(ma5) and pd.notna(ma10):
        if close < ma5 or ma5 < ma10:
            triggers.append(f"均线走弱(收盘{close:.2f}<MA5{ma5:.2f})")

    # 5. 布林带中轨下方
    boll_mid = _safe_get(df, 'BOLL_MID', idx)
    if pd.notna(boll_mid) and close < boll_mid:
        triggers.append(f"布林中轨下方(收盘{close:.2f}<{boll_mid:.2f})")

    # 6. 近期高点回落
    if idx >= lookback:
        recent_high = df['close'].iloc[idx - lookback:idx + 1].max()
        if close < recent_high * 0.97:
            triggers.append(f"近期高点回落({(close/recent_high-1)*100:.1f}%)")

    return len(triggers), triggers


def detect_bullish_divergence(df, idx, lookback=5):
    """
    检测"应涨"信号：技术面显示应该上涨。
    返回: (触发数量, 触发详情列表)
    """
    triggers = []
    close = _safe_get(df, 'close', idx)
    if pd.isna(close):
        return 0, triggers

    # 1. MACD金叉或DIF向上
    macd_dif = _safe_get(df, 'MACD_DIF', idx)
    macd_dea = _safe_get(df, 'MACD_DEA', idx)
    macd_dif_prev = _safe_get(df, 'MACD_DIF', idx - 1)
    if pd.notna(macd_dif) and pd.notna(macd_dea) and pd.notna(macd_dif_prev):
        if macd_dif > macd_dea or (macd_dif > macd_dif_prev and macd_dif < 0):
            triggers.append(f"MACD走强(DIF={macd_dif:.3f})")

    # 2. RSI超卖反弹
    rsi = _safe_get(df, 'RSI_14', idx)
    rsi_prev = _safe_get(df, 'RSI_14', idx - 1)
    if pd.notna(rsi) and pd.notna(rsi_prev):
        if rsi < 30 or (rsi_prev < 30 and rsi > rsi_prev):
            triggers.append(f"RSI超卖/反弹({rsi:.1f})")

    # 3. KDJ金叉
    kdj_k = _safe_get(df, 'KDJ_K', idx)
    kdj_d = _safe_get(df, 'KDJ_D', idx)
    kdj_k_prev = _safe_get(df, 'KDJ_K', idx - 1)
    kdj_d_prev = _safe_get(df, 'KDJ_D', idx - 1)
    if all(pd.notna(x) for x in [kdj_k, kdj_d, kdj_k_prev, kdj_d_prev]):
        if kdj_k > kdj_d or (kdj_k_prev < kdj_d_prev and kdj_k > kdj_d):
            triggers.append(f"KDJ金叉(K={kdj_k:.1f},D={kdj_d:.1f})")

    # 4. 站上短期均线
    ma5 = _safe_get(df, 'MA5', idx)
    ma10 = _safe_get(df, 'MA10', idx)
    if pd.notna(ma5) and pd.notna(ma10):
        if close > ma5 or ma5 > ma10:
            triggers.append(f"均线走强(收盘{close:.2f}>MA5{ma5:.2f})")

    # 5. 布林带中轨上方
    boll_mid = _safe_get(df, 'BOLL_MID', idx)
    if pd.notna(boll_mid) and close > boll_mid:
        triggers.append(f"布林中轨上方(收盘{close:.2f}>{boll_mid:.2f})")

    # 6. 近期低点回升
    if idx >= lookback:
        recent_low = df['close'].iloc[idx - lookback:idx + 1].min()
        if close > recent_low * 1.03:
            triggers.append(f"近期低点回升({(close/recent_low-1)*100:.1f}%)")

    return len(triggers), triggers


def analyze_volume_price(df, idx):
    """
    分析量价关系，判断是否有主力资金异动。
    返回: dict (volume_ratio, price_change, upper_shadow_ratio, lower_shadow_ratio,
                 volume_signal, details)
    """
    result = {
        'volume_ratio': np.nan, 'price_change': np.nan,
        'upper_shadow_ratio': np.nan, 'lower_shadow_ratio': np.nan,
        'volume_signal': '', 'details': []
    }

    close = _safe_get(df, 'close', idx)
    open_ = _safe_get(df, 'open', idx)
    high = _safe_get(df, 'high', idx)
    low = _safe_get(df, 'low', idx)
    volume = _safe_get(df, 'volume', idx)

    if any(pd.isna(x) for x in [close, open_, high, low, volume]):
        return result

    # 量比：当日成交量 / 5日均量
    if idx >= 5:
        avg_vol5 = df['volume'].iloc[idx - 5:idx].mean()
        if avg_vol5 > 0:
            result['volume_ratio'] = round(volume / avg_vol5, 2)

    # 涨跌幅
    if idx > 0:
        prev_close = _safe_get(df, 'close', idx - 1)
        if pd.notna(prev_close) and prev_close > 0:
            result['price_change'] = round((close - prev_close) / prev_close * 100, 2)

    # 上影线/下影线比例
    body = abs(close - open_)
    if body > 0:
        result['upper_shadow_ratio'] = round((high - max(close, open_)) / body, 2)
        result['lower_shadow_ratio'] = round((min(close, open_) - low) / body, 2)
    else:
        result['upper_shadow_ratio'] = 0
        result['lower_shadow_ratio'] = 0

    # 量价信号判断
    vr = result['volume_ratio']
    pc = result['price_change']
    usr = result['upper_shadow_ratio']
    lsr = result['lower_shadow_ratio']

    if pd.notna(vr) and pd.notna(pc):
        # 放量震荡（量比>1.5但涨跌幅<1%）→ 主力可能在吸筹或出货，需结合趋势判断
        if vr > 1.5 and abs(pc) < 1.0:
            result['volume_signal'] = '放量震荡'
            result['details'].append(f"量比{vr:.1f}但涨跌幅仅{pc:.1f}%（放量但价格不动）")
        # 放量抗跌（量比>1.5且跌幅在1-2%之间）→ 有承接
        elif vr > 1.5 and pc > -2.0 and pc <= -1.0:
            result['volume_signal'] = '放量抗跌'
            result['details'].append(f"量比{vr:.1f}但仅跌{pc:.1f}%（有承接）")
        # 缩量抗跌（量比<0.7但跌幅<1%）→ 惜售
        elif vr < 0.7 and pc > -1.0 and pc < 0:
            result['volume_signal'] = '缩量抗跌'
            result['details'].append(f"量比{vr:.1f}但仅跌{pc:.1f}%（惜售）")
        # 缩量滞涨（量比<0.7但涨幅<1%）→ 买盘不足
        elif vr < 0.7 and pc > 0 and pc < 1.0:
            result['volume_signal'] = '缩量滞涨'
            result['details'].append(f"量比{vr:.1f}但仅涨{pc:.1f}%（买盘不足）")

    # 长上影线（>2倍实体）+ 放量 → 出货
    if pd.notna(usr) and usr > 2.0 and pd.notna(vr) and vr > 1.2:
        if '放量' not in result['volume_signal']:
            result['volume_signal'] = '长上影放量'
        result['details'].append(f"上影线/实体={usr:.1f}（抛压重）")

    # 长下影线（>2倍实体）+ 放量 → 吸筹
    if pd.notna(lsr) and lsr > 2.0 and pd.notna(vr) and vr > 1.2:
        if '放量' not in result['volume_signal']:
            result['volume_signal'] = '长下影放量'
        result['details'].append(f"下影线/实体={lsr:.1f}（承接强）")

    return result


def _detect_trend(df, idx, ma_period=20):
    """
    检测当前趋势状态。
    返回: ('uptrend'/'downtrend'/'sideways', trend_strength 0-1)
    """
    if idx < ma_period + 5:
        return 'sideways', 0.0

    ma_col = f'MA{ma_period}'
    if ma_col not in df.columns:
        return 'sideways', 0.0

    ma_now = _safe_get(df, ma_col, idx)
    ma_prev = _safe_get(df, ma_col, idx - 5)
    close = _safe_get(df, 'close', idx)

    if any(pd.isna(x) for x in [ma_now, ma_prev, close]):
        return 'sideways', 0.0

    ma_slope = (ma_now - ma_prev) / ma_prev * 100 if ma_prev > 0 else 0
    price_vs_ma = (close - ma_now) / ma_now * 100

    if ma_slope > 0.3 and price_vs_ma > -3:
        trend = 'uptrend'
        strength = min(abs(ma_slope) / 2, 1.0)
    elif ma_slope < -0.3 and price_vs_ma < 3:
        trend = 'downtrend'
        strength = min(abs(ma_slope) / 2, 1.0)
    else:
        trend = 'sideways'
        strength = 0.2

    return trend, round(strength, 2)


def detect_anomaly_signals(df, capital_flow_df=None, lookback=5, min_triggers=2):
    """
    主函数：检测异常信号（优化版v2）。

    核心改进：
      1. 趋势过滤：上涨趋势中只发"应跌未跌→看涨"，下跌趋势中只发"应涨未涨→看跌"
      2. 严格"未涨/未跌"定义：涨跌幅<0.5%且不创新高/新低
      3. 强制量价确认：必须有量价信号，不能仅靠技术指标数量绕过
      4. 趋势对齐加分：信号方向与大趋势一致时强度+20
      5. 去掉"缩量滞涨"作为看跌确认（上涨趋势中可能是蓄势）

    参数:
      df: 含技术指标的K线DataFrame
      capital_flow_df: 资金流DataFrame（可选）
      lookback: 回看天数
      min_triggers: 最少触发指标数（默认2）

    返回:
      list: 异常信号列表
    """
    signals = []
    if df is None or len(df) < lookback + 25:
        return signals

    df = df.reset_index(drop=True)

    for i in range(lookback + 20, len(df)):
        close = _safe_get(df, 'close', i)
        high = _safe_get(df, 'high', i)
        low = _safe_get(df, 'low', i)
        if any(pd.isna(x) for x in [close, high, low]):
            continue

        # 检测趋势
        trend, trend_strength = _detect_trend(df, i)

        # 检测"应跌"信号数量
        bearish_count, bearish_triggers = detect_bearish_divergence(df, i, lookback)
        # 检测"应涨"信号数量
        bullish_count, bullish_triggers = detect_bullish_divergence(df, i, lookback)

        # 量价分析
        vp = analyze_volume_price(df, i)
        price_change = vp['price_change']
        if pd.isna(price_change):
            continue

        # 检查是否创新高/新低（严格"未涨/未跌"定义）
        recent_highs = df['high'].iloc[i - lookback:i].max() if i >= lookback else high
        recent_lows = df['low'].iloc[i - lookback:i].min() if i >= lookback else low
        makes_new_high = high >= recent_highs * 0.995
        makes_new_low = low <= recent_lows * 1.005

        signal = None

        # === 核心逻辑：异常确认趋势 ===
        # 信号方向 = 大趋势方向（uptrend→看涨，downtrend→看跌，sideways→背离方向）
        # 异常类型（应跌未跌/应涨未涨）只是确认趋势的强度和持续性
        #
        # 逻辑依据：
        #   uptrend中的"应涨未涨" = 蓄势整理，后续看涨（如4月17日600118缩量整理后涨停）
        #   uptrend中的"应跌未跌" = 强势抗跌，后续看涨
        #   downtrend中的"应跌未跌" = 下跌中继，后续看跌
        #   downtrend中的"应涨未涨" = 弱势反弹，后续看跌

        # 检测"应跌未跌"：技术面看跌但价格抗跌
        not_falling = (price_change > -1.5) and (not makes_new_low)
        vp_confirm_not_falling = vp['volume_signal'] in [
            '放量震荡', '放量抗跌', '长下影放量',  # 放量确认（权重高）
            '缩量抗跌'  # 缩量惜售（权重低，uptrend中可能是蓄势）
        ]
        is_anomaly_bullish = (bearish_count >= min_triggers and not_falling
                               and vp_confirm_not_falling)

        # 检测"应涨未涨"：技术面看涨但价格滞涨
        not_rising = (price_change < 1.5) and (not makes_new_high)
        vp_confirm_not_rising = vp['volume_signal'] in [
            '放量震荡', '长上影放量',  # 放量确认（权重高）
            '缩量滞涨'  # 缩量整理（uptrend中可能是蓄势，downtrend中是买盘不足）
        ]
        is_anomaly_bearish = (bullish_count >= min_triggers and not_rising
                               and vp_confirm_not_rising)

        # 确定信号方向（基于趋势）
        if trend == 'uptrend':
            signal_direction = '看涨'
            # uptrend中两种异常都看涨：应跌未跌=强势，应涨未涨=蓄势
            anomaly_type = '应跌未跌' if is_anomaly_bullish else '应涨未涨'
            anomaly_count = bearish_count if is_anomaly_bullish else bullish_count
            anomaly_triggers = bearish_triggers if is_anomaly_bullish else bullish_triggers
            has_anomaly = is_anomaly_bullish or is_anomaly_bearish
            vp_signal = vp['volume_signal']
        elif trend == 'downtrend':
            signal_direction = '看跌'
            # downtrend中两种异常都看跌：应跌未跌=下跌中继，应涨未涨=弱势
            anomaly_type = '应跌未跌' if is_anomaly_bullish else '应涨未涨'
            anomaly_count = bearish_count if is_anomaly_bullish else bullish_count
            anomaly_triggers = bearish_triggers if is_anomaly_bullish else bullish_triggers
            has_anomaly = is_anomaly_bullish or is_anomaly_bearish
            vp_signal = vp['volume_signal']
        else:  # sideways
            # 震荡趋势中，信号质量较差，提高门槛：需要3项触发+放量确认
            if is_anomaly_bullish and bearish_count >= 3 and vp['volume_signal'] in [
                    '放量震荡', '放量抗跌', '长下影放量']:
                signal_direction = '看涨'
                anomaly_type = '应跌未跌'
                anomaly_count = bearish_count
                anomaly_triggers = bearish_triggers
                has_anomaly = True
            elif is_anomaly_bearish and bullish_count >= 3 and vp['volume_signal'] in [
                    '放量震荡', '长上影放量']:
                signal_direction = '看跌'
                anomaly_type = '应涨未涨'
                anomaly_count = bullish_count
                anomaly_triggers = bullish_triggers
                has_anomaly = True
            else:
                has_anomaly = False
            vp_signal = vp['volume_signal']

        if has_anomaly:
            # 资金流验证
            cf_dir = 'bullish' if signal_direction == '看涨' else 'bearish'
            cf_data, main_force = _check_capital_flow(capital_flow_df, df, i, cf_dir)

            # 信号强度计算
            base_strength = min(anomaly_count * 12, 45)  # 技术背离最高45
            # 量价确认：放量+20，缩量+10
            if vp_signal in ['放量震荡', '放量抗跌', '长下影放量', '长上影放量']:
                base_strength += 20
            else:
                base_strength += 10
            # 资金流确认
            if (signal_direction == '看涨' and main_force == '吸筹') or \
               (signal_direction == '看跌' and main_force == '出货'):
                base_strength += 20
            # 趋势强度加分
            base_strength += 15 * trend_strength
            # 异常类型加分：应跌未跌在uptrend中更强（强势抗跌）
            if trend == 'uptrend' and anomaly_type == '应跌未跌':
                base_strength += 5
            if trend == 'downtrend' and anomaly_type == '应涨未涨':
                base_strength += 5

            strength = min(base_strength, 100)
            confidence = '高' if strength >= 60 else ('中' if strength >= 40 else '低')

            # 解读文本
            if trend == 'uptrend':
                if anomaly_type == '应跌未跌':
                    interp = f"[上涨趋势]技术面{anomaly_count}项看跌但价格抗跌({price_change:+.1f}%)，{vp_signal}，强势确认，后续看涨"
                else:
                    interp = f"[上涨趋势]技术面{anomaly_count}项看涨但价格滞涨({price_change:+.1f}%)，{vp_signal}，蓄势整理，后续看涨"
            elif trend == 'downtrend':
                if anomaly_type == '应跌未跌':
                    interp = f"[下跌趋势]技术面{anomaly_count}项看跌但价格抗跌({price_change:+.1f}%)，{vp_signal}，下跌中继，后续看跌"
                else:
                    interp = f"[下跌趋势]技术面{anomaly_count}项看涨但价格滞涨({price_change:+.1f}%)，{vp_signal}，弱势反弹，后续看跌"
            else:
                if signal_direction == '看涨':
                    interp = f"[震荡趋势]技术面{anomaly_count}项看跌但价格抗跌({price_change:+.1f}%)，{vp_signal}，疑似吸筹"
                else:
                    interp = f"[震荡趋势]技术面{anomaly_count}项看涨但价格滞涨({price_change:+.1f}%)，{vp_signal}，疑似出货"

            if main_force in ['吸筹', '出货']:
                interp += f"，主力{main_force}"

            signal = {
                'date': str(df.iloc[i]['date']),
                'signal_type': anomaly_type,
                'direction': signal_direction,
                'strength': round(strength, 1),
                'confidence': confidence,
                'trend': trend,
                'anomaly_trigger_count': anomaly_count,
                'triggers': anomaly_triggers,
                'volume_price': {
                    'volume_ratio': vp['volume_ratio'],
                    'price_change': vp['price_change'],
                    'volume_signal': vp_signal,
                    'details': vp['details'],
                },
                'capital_flow': cf_data,
                'main_force_action': main_force,
                'close': round(close, 2),
                'interpretation': interp,
            }

        if signal:
            signals.append(signal)

    # 按强度排序
    signals.sort(key=lambda x: x['strength'], reverse=True)
    return signals


def _check_capital_flow(capital_flow_df, df, idx, signal_direction):
    """检查资金流数据，返回(cf_data, main_force_action)"""
    cf_data = None
    main_force = '观望'
    if capital_flow_df is not None and len(capital_flow_df) > 0:
        try:
            signal_date = pd.to_datetime(df.iloc[idx]['date'])
            cf_match = capital_flow_df[pd.to_datetime(capital_flow_df['date']) == signal_date]
            if len(cf_match) > 0:
                cf_row = cf_match.iloc[0]
                cf_data = {
                    'main_net_inflow': float(cf_row.get('main_net_inflow', 0)),
                    'main_net_pct': float(cf_row.get('main_net_pct', 0)),
                    'super_large_net': float(cf_row.get('super_large_net', 0)),
                    'large_net': float(cf_row.get('large_net', 0)),
                }
                if signal_direction == 'bullish':
                    if cf_data['main_net_inflow'] > 0:
                        main_force = '吸筹'
                    elif cf_data['main_net_inflow'] < -1e7:
                        main_force = '出货'
                else:
                    if cf_data['main_net_inflow'] < -1e7:
                        main_force = '出货'
                    elif cf_data['main_net_inflow'] > 0:
                        main_force = '吸筹'
        except Exception:
            pass
    return cf_data, main_force


def get_latest_anomaly_signal(signals):
    """获取最新的异常信号"""
    if not signals:
        return None
    # 按日期排序取最新
    sorted_by_date = sorted(signals, key=lambda x: x['date'], reverse=True)
    return sorted_by_date[0]


def get_signal_summary(signals):
    """获取信号统计摘要"""
    if not signals:
        return {'total': 0, 'bullish': 0, 'bearish': 0, 'high_confidence': 0,
                'latest': None, 'main_force_accumulation': 0, 'main_force_distribution': 0}

    bullish = sum(1 for s in signals if s['direction'] == '看涨')
    bearish = sum(1 for s in signals if s['direction'] == '看跌')
    high_conf = sum(1 for s in signals if s['confidence'] == '高')
    accumulation = sum(1 for s in signals if s['main_force_action'] == '吸筹')
    distribution = sum(1 for s in signals if s['main_force_action'] == '出货')
    latest = get_latest_anomaly_signal(signals)

    return {
        'total': len(signals),
        'bullish': bullish,
        'bearish': bearish,
        'high_confidence': high_conf,
        'main_force_accumulation': accumulation,
        'main_force_distribution': distribution,
        'latest': latest,
    }


# ============================================================
# 异常信号回测验证
# ============================================================

def backtest_anomaly_signals(signals, df, holding_periods=[1, 3, 5, 10, 20]):
    """
    对异常信号进行回测验证。

    参数:
      signals: 异常信号列表
      df: 含date和close列的K线DataFrame
      holding_periods: 持有期列表（交易日）

    返回:
      dict: 回测结果，包含:
        - overall: 整体统计
        - by_direction: 按方向分组（看涨/看跌）
        - by_confidence: 按置信度分组
        - by_signal_type: 按信号类型分组
        - by_strength_bucket: 按强度区间分组
        - signal_returns: 每个信号的详细收益
    """
    if not signals or df is None or len(df) == 0:
        return {'error': '无信号或无数据', 'overall': {}}

    df = df.reset_index(drop=True)
    # 确保date列可比较
    df['_date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    signal_returns = []
    for sig in signals:
        sig_date = sig['date'][:10]
        # 找到信号日在df中的位置
        match = df[df['_date_str'] == sig_date]
        if len(match) == 0:
            continue
        idx = match.index[0]
        entry_price = df.iloc[idx]['close']

        returns = {}
        for days in holding_periods:
            if idx + days < len(df):
                exit_price = df.iloc[idx + days]['close']
                raw_ret = (exit_price - entry_price) / entry_price * 100
                # 看涨信号：收益=raw_ret；看跌信号：收益=-raw_ret（做空视角）
                if sig['direction'] == '看涨':
                    returns[f'{days}d_ret'] = round(raw_ret, 2)
                    returns[f'{days}d_correct'] = raw_ret > 0
                else:
                    returns[f'{days}d_ret'] = round(-raw_ret, 2)
                    returns[f'{days}d_correct'] = raw_ret < 0
            else:
                returns[f'{days}d_ret'] = None
                returns[f'{days}d_correct'] = None

        signal_returns.append({
            'date': sig_date,
            'signal_type': sig['signal_type'],
            'direction': sig['direction'],
            'strength': sig['strength'],
            'confidence': sig['confidence'],
            'trend': sig.get('trend', 'unknown'),
            'main_force_action': sig.get('main_force_action', '未知'),
            'entry_price': round(entry_price, 2),
            **returns,
        })

    if not signal_returns:
        return {'error': '无有效信号可回测', 'overall': {}}

    # 整体统计
    overall = _calc_group_stats(signal_returns, holding_periods)

    # 按方向分组
    by_direction = {}
    for direction in ['看涨', '看跌']:
        group = [s for s in signal_returns if s['direction'] == direction]
        if group:
            by_direction[direction] = _calc_group_stats(group, holding_periods)

    # 按置信度分组
    by_confidence = {}
    for conf in ['高', '中', '低']:
        group = [s for s in signal_returns if s['confidence'] == conf]
        if group:
            by_confidence[conf] = _calc_group_stats(group, holding_periods)

    # 按信号类型分组
    by_signal_type = {}
    for stype in set(s['signal_type'] for s in signal_returns):
        group = [s for s in signal_returns if s['signal_type'] == stype]
        if group:
            by_signal_type[stype] = _calc_group_stats(group, holding_periods)

    # 按强度区间分组
    by_strength_bucket = {}
    buckets = [(80, 101, '80-100'), (60, 80, '60-80'), (40, 60, '40-60'), (0, 40, '0-40')]
    for low, high, label in buckets:
        group = [s for s in signal_returns if low <= s['strength'] < high]
        if group:
            by_strength_bucket[label] = _calc_group_stats(group, holding_periods)

    return {
        'overall': overall,
        'by_direction': by_direction,
        'by_confidence': by_confidence,
        'by_signal_type': by_signal_type,
        'by_strength_bucket': by_strength_bucket,
        'signal_returns': signal_returns,
        'holding_periods': holding_periods,
    }


def _calc_group_stats(signal_list, holding_periods):
    """计算一组信号的统计指标"""
    stats = {'count': len(signal_list)}

    for days in holding_periods:
        rets = [s[f'{days}d_ret'] for s in signal_list if s[f'{days}d_ret'] is not None]
        correct = [s[f'{days}d_correct'] for s in signal_list if s[f'{days}d_correct'] is not None]

        if rets:
            stats[f'{days}d_avg_ret'] = round(np.mean(rets), 2)
            stats[f'{days}d_median_ret'] = round(np.median(rets), 2)
            stats[f'{days}d_win_rate'] = round(sum(correct) / len(correct) * 100, 1) if correct else 0
            stats[f'{days}d_max_ret'] = round(max(rets), 2)
            stats[f'{days}d_min_ret'] = round(min(rets), 2)
            # 盈亏比
            wins = [r for r in rets if r > 0]
            losses = [abs(r) for r in rets if r < 0]
            if wins and losses:
                stats[f'{days}d_profit_loss_ratio'] = round(np.mean(wins) / np.mean(losses), 2)
            else:
                stats[f'{days}d_profit_loss_ratio'] = None
        else:
            stats[f'{days}d_avg_ret'] = None
            stats[f'{days}d_win_rate'] = None

    return stats
