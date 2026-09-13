#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股股票分析系统 - Flask 后端
================================
功能模块：
  1. 数据获取（akshare，前复权，带磁盘缓存）
  2. 技术指标计算（MACD/RSI/KDJ/布林带/MA/量能/OBV/WR/CCI）
  3. 最优买卖点计算（贪心峰谷法，多次交易，不可同时持仓）
  4. 技术指标相关性分析（点二列相关 + t检验显著性）
  5. 可解释数学模型（指标阈值加权打分系统）
  6. 历史回测（初始资金100万，全仓，手续费万2.5，印花税千1）
  7. 实时信号生成（最新交易日指标 + 模型概率）

运行：python app.py  然后浏览器打开 http://127.0.0.1:5000
"""

import os
import json
import time
import hashlib
import warnings
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
from scipy import stats
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 回测参数
INITIAL_CAPITAL = 1_000_000.0   # 初始资金 100万
COMMISSION_RATE = 0.00025        # 手续费 万2.5（买卖均收）
STAMP_TAX_RATE = 0.001           # 印花税 千1（仅卖出）
MIN_COMMISSION = 5.0              # 最低佣金5元

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')


# ============================================================
# 工具：磁盘缓存装饰器
# ============================================================
def cache_pickle(expire_seconds=3600):
    """将函数返回的 DataFrame 缓存为 pickle 文件，过期自动刷新。"""
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
                        pass  # 缓存损坏则重新获取
            result = func(*args, **kwargs)
            try:
                result.to_pickle(cache_file)
            except Exception:
                pass
            return result
        return wrapper
    return decorator


# ============================================================
# 1. 数据获取层
# ============================================================
@cache_pickle(expire_seconds=86400)  # 历史日线缓存1天
def fetch_stock_data(code, start_date, end_date):
    """
    获取A股前复权日线数据。
    参数：code='600519', start_date='2023-01-01', end_date='2023-12-31'
    返回：DataFrame [date, open, high, low, close, volume, amount, pct_change]
    """
    import akshare as ak
    s = start_date.replace('-', '')
    e = end_date.replace('-', '')
    # akshare 前复权接口
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=s, end_date=e, adjust="qfq"
    )
    if df is None or len(df) == 0:
        raise ValueError(f"未获取到股票 {code} 在 {start_date}~{end_date} 的数据，请检查代码或日期范围")
    # 统一列名
    col_map = {
        '日期': 'date', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume',
        '成交额': 'amount', '涨跌幅': 'pct_change', '换手率': 'turnover'
    }
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']
    df = df[[c for c in keep if c in df.columns]].sort_values('date').reset_index(drop=True)
    return df


def fetch_recent_data(code, days=150):
    """获取最近一段交易日数据（用于实时信号计算），缓存2小时。"""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days * 2)).strftime('%Y-%m-%d')
    # 用较短缓存的包装
    @cache_pickle(expire_seconds=7200)
    def _fetch(c, s, e):
        return fetch_stock_data(c, s, e)
    return _fetch(code, start, end)


# ============================================================
# 2. 技术指标计算
# ============================================================
def calc_indicators(df):
    """
    计算全部技术指标，返回追加指标列的 DataFrame。
    指标清单：
      - 均线：MA5, MA10, MA20, MA60
      - MACD：DIF, DEA, MACD柱
      - RSI：RSI6, RSI14
      - KDJ：K, D, J
      - 布林带：BOLL_MID, BOLL_UP, BOLL_LOW, BOLL_PCTB
      - 量能：VOL_MA5, VOL_RATE（成交量变化率%）
      - OBV（能量潮）
      - WR（威廉指标）
      - CCI（顺势指标）
      - 价格相对均线位置：CLOSE_MA20_RATIO
    """
    d = df.copy()
    c = d['close']
    h = d['high']
    l = d['low']
    v = d['volume']

    # --- 均线系统 ---
    for w in [5, 10, 20, 60]:
        d[f'MA{w}'] = c.rolling(w).mean()

    # --- MACD (12,26,9) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['DIF'] = ema12 - ema26
    d['DEA'] = d['DIF'].ewm(span=9, adjust=False).mean()
    d['MACD'] = 2 * (d['DIF'] - d['DEA'])

    # --- RSI ---
    for period in [6, 14]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        d[f'RSI{period}'] = 100 - 100 / (1 + rs)

    # --- KDJ (9,3,3) ---
    n = 9
    low_n = l.rolling(n).min()
    high_n = h.rolling(n).max()
    rsv = (c - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    d['K'] = rsv.ewm(com=2, adjust=False).mean()
    d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    d['J'] = 3 * d['K'] - 2 * d['D']

    # --- 布林带 (20, 2) ---
    d['BOLL_MID'] = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d['BOLL_UP'] = d['BOLL_MID'] + 2 * std20
    d['BOLL_LOW'] = d['BOLL_MID'] - 2 * std20
    d['BOLL_PCTB'] = (c - d['BOLL_LOW']) / (d['BOLL_UP'] - d['BOLL_LOW']).replace(0, np.nan)

    # --- 量能 ---
    d['VOL_MA5'] = v.rolling(5).mean()
    d['VOL_RATE'] = (v - v.shift(1)) / v.shift(1).replace(0, np.nan) * 100

    # --- OBV ---
    direction = np.sign(c.diff()).fillna(0)
    d['OBV'] = (direction * v).cumsum()

    # --- WR (威廉指标, 14日) ---
    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    d['WR'] = (hh14 - c) / (hh14 - ll14).replace(0, np.nan) * -100

    # --- CCI (14日) ---
    tp = (h + l + c) / 3
    ma_tp = tp.rolling(14).mean()
    md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d['CCI'] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))

    # --- 收盘价相对MA20的位置（归一化） ---
    d['CLOSE_MA20_RATIO'] = (c - d['MA20']) / d['MA20'].replace(0, np.nan) * 100

    # 去除前60行（MA60需要预热）
    d = d.iloc[60:].reset_index(drop=True)
    # 填充剩余NaN
    d = d.ffill().bfill()
    return d


# 参与相关性分析和建模的指标列
INDICATOR_COLS = [
    'MA5', 'MA10', 'MA20', 'MA60',
    'DIF', 'DEA', 'MACD',
    'RSI6', 'RSI14',
    'K', 'D', 'J',
    'BOLL_MID', 'BOLL_UP', 'BOLL_LOW', 'BOLL_PCTB',
    'VOL_MA5', 'VOL_RATE',
    'OBV', 'WR', 'CCI',
    'CLOSE_MA20_RATIO'
]


# ============================================================
# 3. 最优买卖点计算（贪心峰谷法）
# ============================================================
def find_optimal_trades(df, min_profit_pct=1.0):
    """
    在选定时间段内寻找最大收益的买卖操作序列。
    算法：贪心峰谷法——在每个局部谷底买入，下一个局部峰顶卖出。
          对于"允许多次交易、不可同时持仓"的无交易成本场景，
          该算法理论上可获取最大收益。
    参数：
      min_profit_pct: 过滤掉收益率低于此阈值的噪声交易（默认1%）
    返回：
      trades: 交易列表，每笔含 buy_date, buy_price, sell_date, sell_price, profit_rate
      total_return: 总收益率（复利累计）
      buy_hold_return: 同期买入持有收益率
    """
    closes = df['close'].values
    dates = df['date'].values
    n = len(closes)
    if n < 2:
        return [], 0.0, 0.0

    trades = []
    i = 0
    while i < n - 1:
        # 寻找局部谷底（连续下跌后的最低点）
        while i < n - 1 and closes[i + 1] <= closes[i]:
            i += 1
        if i >= n - 1:
            break
        buy_idx = i
        buy_price = closes[buy_idx]
        # 寻找局部峰顶（连续上涨后的最高点）
        while i < n - 1 and closes[i + 1] >= closes[i]:
            i += 1
        sell_idx = i
        sell_price = closes[sell_idx]
        profit = (sell_price - buy_price) / buy_price * 100
        if profit >= min_profit_pct:
            trades.append({
                'buy_date': str(dates[buy_idx]),
                'buy_price': round(float(buy_price), 2),
                'sell_date': str(dates[sell_idx]),
                'sell_price': round(float(sell_price), 2),
                'profit_rate': round(float(profit), 2),
                'holding_days': int(sell_idx - buy_idx)
            })

    # 计算复利总收益率
    total = 1.0
    for t in trades:
        total *= (1 + t['profit_rate'] / 100)
    total_return = round((total - 1) * 100, 2)

    # 买入持有收益率
    buy_hold = round((closes[-1] - closes[0]) / closes[0] * 100, 2)

    return trades, total_return, buy_hold


# ============================================================
# 4. 技术指标相关性分析
# ============================================================
def analyze_correlation(df, trades):
    """
    分析各技术指标在最优买卖点出现前后的数值特征，
    计算指标与买卖点的点二列相关系数及t检验显著性。
    返回：
      buy_corr: 买点相关性排序 [{indicator, corr, pvalue, mean_at, mean_other, direction}]
      sell_corr: 卖点相关性排序
      buy_profiles: 买点前后各指标均值变化（t-3~t+3）
      sell_profiles: 卖点前后各指标均值变化
    """
    n = len(df)
    buy_dates = set(t['buy_date'] for t in trades)
    sell_dates = set(t['sell_date'] for t in trades)

    buy_labels = np.array([1 if d in buy_dates else 0 for d in df['date']])
    sell_labels = np.array([1 if d in sell_dates else 0 for d in df['date']])

    def _corr_for_labels(labels):
        results = []
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        for col in INDICATOR_COLS:
            vals = df[col].values.astype(float)
            # 跳过全NaN或常数列
            if np.isnan(vals).all() or np.nanstd(vals) == 0:
                continue
            # 点二列相关 = Pearson(连续, 二元)
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue
            r, p = stats.pointbiserialr(labels[valid], vals[valid])
            if np.isnan(r) or np.isnan(p):
                continue
            mean_at = float(np.nanmean(vals[labels == 1])) if n_pos > 0 else 0
            mean_other = float(np.nanmean(vals[labels == 0])) if n_neg > 0 else 0
            # direction: 'high' 表示在事件点数值偏高, 'low' 表示偏低
            direction = 'high' if mean_at > mean_other else 'low'
            results.append({
                'indicator': col,
                'corr': round(float(r), 4),
                'abs_corr': round(abs(float(r)), 4),
                'pvalue': round(float(p), 6),
                'significant': bool(p < 0.05),
                'mean_at_event': round(mean_at, 4),
                'mean_other': round(mean_other, 4),
                'direction': direction
            })
        results.sort(key=lambda x: x['abs_corr'], reverse=True)
        return results

    buy_corr = _corr_for_labels(buy_labels)
    sell_corr = _corr_for_labels(sell_labels)

    # 买卖点前后指标均值画像（t-3 到 t+3）
    def _profile(event_dates):
        event_idx = [i for i, d in enumerate(df['date']) if d in event_dates]
        profile = {}
        offsets = list(range(-3, 4))
        for col in INDICATOR_COLS:
            vals = df[col].values.astype(float)
            series = []
            for off in offsets:
                gathered = []
                for ei in event_idx:
                    j = ei + off
                    if 0 <= j < n and not np.isnan(vals[j]):
                        gathered.append(vals[j])
                series.append(round(float(np.mean(gathered)), 4) if gathered else None)
            profile[col] = {'offsets': offsets, 'values': series}
        return profile

    buy_profiles = _profile(buy_dates)
    sell_profiles = _profile(sell_dates)

    return buy_corr, sell_corr, buy_profiles, sell_profiles


# ============================================================
# 5. 可解释数学模型（阈值加权打分系统）
# ============================================================
def build_model(df, buy_corr, sell_corr, top_n=6):
    """
    基于相关性分析结果，构建可解释的阈值加权打分模型。

    模型原理：
      1. 选取与买点/卖点相关性最高的 top_n 个指标
      2. 对每个指标，用事件点均值与非事件点均值的中点作为阈值
      3. 每个指标根据方向（high/low）和阈值判断是否发出买/卖信号
      4. 信号强度 = 归一化的指标偏离度 × 相关系数权重
      5. 综合得分 = 所有指标信号之和，映射为 [0,1] 概率

    返回模型规则字典，可直接用于前端可视化。
    """
    def _make_rules(corr_list, event_type):
        rules = []
        for item in corr_list[:top_n]:
            col = item['indicator']
            weight = item['abs_corr']
            direction = item['direction']
            mean_at = item['mean_at_event']
            mean_ot = item['mean_other']
            # 阈值 = 事件点与非事件点均值的中点
            threshold = (mean_at + mean_ot) / 2
            # 标准差用于归一化偏离度
            std_val = float(df[col].std())
            if std_val == 0 or np.isnan(std_val):
                std_val = 1.0
            rules.append({
                'indicator': col,
                'weight': round(weight, 4),
                'direction': direction,       # high=事件点偏高, low=事件点偏低
                'threshold': round(float(threshold), 4),
                'std': round(std_val, 4),
                'mean_at_event': mean_at,
                'mean_other': mean_ot,
                'event_type': event_type
            })
        return rules

    buy_rules = _make_rules(buy_corr, 'buy')
    sell_rules = _make_rules(sell_corr, 'sell')

    model = {
        'type': '阈值加权打分系统 (Threshold-Weighted Scoring)',
        'description': (
            '选取与最优买卖点相关性最高的指标，以事件点/非事件点均值中点为阈值，'
            '以指标偏离度（标准化）×相关系数为权重，综合打分映射为买卖概率。'
        ),
        'buy_rules': buy_rules,
        'sell_rules': sell_rules,
        'buy_threshold_prob': 0.60,   # 买点概率阈值
        'sell_threshold_prob': 0.60,  # 卖点概率阈值
        'formula': (
            'score_buy = Σ [ weight_i × sigmoid( (value_i - threshold_i) / std_i × sign_i ) ]\n'
            'prob_buy = score_buy / Σ weight_i\n'
            '其中 sign_i = +1 (direction=high), -1 (direction=low)'
        )
    }
    return model


def compute_model_signals(df, model):
    """
    对每一天计算模型的买点概率和卖点概率。
    返回：buy_probs (np.array), sell_probs (np.array), triggered (list of dict)
    """
    buy_rules = model['buy_rules']
    sell_rules = model['sell_rules']
    buy_w_sum = sum(r['weight'] for r in buy_rules) or 1
    sell_w_sum = sum(r['weight'] for r in sell_rules) or 1

    n = len(df)
    buy_probs = np.zeros(n)
    sell_probs = np.zeros(n)

    def _calc_prob(rules, w_sum):
        probs = np.zeros(n)
        for r in rules:
            col = r['indicator']
            vals = df[col].values.astype(float)
            sign = 1 if r['direction'] == 'high' else -1
            threshold = r['threshold']
            std = r['std'] or 1.0
            weight = r['weight']
            # 标准化偏离度 -> sigmoid 映射到 [0,1]
            z = sign * (vals - threshold) / std
            sig = 1 / (1 + np.exp(-np.clip(z, -10, 10)))
            probs += weight * sig
        return probs / w_sum

    buy_probs = _calc_prob(buy_rules, buy_w_sum)
    sell_probs = _calc_prob(sell_rules, sell_w_sum)

    return buy_probs, sell_probs


# ============================================================
# 6. 历史回测
# ============================================================
def backtest(df, model, buy_probs, sell_probs):
    """
    使用模型在历史数据上模拟交易。
    规则：
      - 初始资金 100万，全仓买卖
      - 信号在T日收盘后生成，T+1日开盘价执行
      - 买入手续费 万2.5（最低5元），卖出手续费 万2.5 + 印花税 千1
      - 买入条件：buy_prob >= 0.60 且 当前空仓
      - 卖出条件：sell_prob >= 0.60 且 当前持仓
    返回：equity_curve, trades, metrics
    """
    buy_th = model['buy_threshold_prob']
    sell_th = model['sell_threshold_prob']

    n = len(df)
    cash = INITIAL_CAPITAL
    shares = 0
    equity = np.zeros(n)
    trades = []
    position = None  # {'buy_date', 'buy_price', 'shares', 'buy_cost'}

    for i in range(n):
        # T日信号，T+1执行；最后一天无法执行
        if i < n - 1:
            exec_price = float(df['open'].iloc[i + 1])
            exec_date = str(df['date'].iloc[i + 1])

            # 买入信号
            if position is None and buy_probs[i] >= buy_th:
                # 计算可买股数（全仓，预留手续费），取整到100股
                max_cost = cash / (1 + COMMISSION_RATE)
                buy_shares = int(max_cost / exec_price / 100) * 100
                if buy_shares >= 100:
                    cost = buy_shares * exec_price
                    commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
                    total_cost = cost + commission
                    if total_cost <= cash:
                        cash -= total_cost
                        shares = buy_shares
                        position = {
                            'buy_date': exec_date,
                            'buy_price': round(exec_price, 2),
                            'shares': buy_shares,
                            'buy_cost': round(total_cost, 2),
                            'buy_prob': round(float(buy_probs[i]), 4)
                        }

            # 卖出信号
            elif position is not None and sell_probs[i] >= sell_th:
                revenue = shares * exec_price
                commission = max(revenue * COMMISSION_RATE, MIN_COMMISSION)
                stamp_tax = revenue * STAMP_TAX_RATE
                net_revenue = revenue - commission - stamp_tax
                cash += net_revenue
                profit = net_revenue - position['buy_cost']
                profit_rate = profit / position['buy_cost'] * 100
                trades.append({
                    'buy_date': position['buy_date'],
                    'buy_price': position['buy_price'],
                    'sell_date': exec_date,
                    'sell_price': round(exec_price, 2),
                    'shares': position['shares'],
                    'profit': round(profit, 2),
                    'profit_rate': round(profit_rate, 2),
                    'buy_prob': position['buy_prob'],
                    'sell_prob': round(float(sell_probs[i]), 4)
                })
                shares = 0
                position = None

        # 计算当日权益（按收盘价计）
        equity[i] = cash + shares * float(df['close'].iloc[i])

    # 期末强制平仓（按最后收盘价）
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
            'buy_date': position['buy_date'],
            'buy_price': position['buy_price'],
            'sell_date': str(df['date'].iloc[-1]) + '(期末平仓)',
            'sell_price': round(last_price, 2),
            'shares': position['shares'],
            'profit': round(profit, 2),
            'profit_rate': round(profit_rate, 2),
            'buy_prob': position['buy_prob'],
            'sell_prob': 1.0
        })
        equity[-1] = cash
        shares = 0
        position = None

    # ---- 计算回测指标 ----
    equity_series = pd.Series(equity)
    total_return = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 年化收益率
    days = len(df)
    years = days / 252
    annual_return = ((equity[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    # 最大回撤
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = float(drawdown.min())

    # 胜率
    win_trades = [t for t in trades if t['profit'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    # 盈亏比
    avg_win = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
    loss_trades = [t for t in trades if t['profit'] <= 0]
    avg_loss = abs(np.mean([t['profit'] for t in loss_trades])) if loss_trades else 0
    profit_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else float('inf')

    # 买入持有基准
    bh_return = (float(df['close'].iloc[-1]) - float(df['close'].iloc[0])) / float(df['close'].iloc[0]) * 100

    metrics = {
        'initial_capital': INITIAL_CAPITAL,
        'final_equity': round(float(equity[-1]), 2),
        'total_return': round(float(total_return), 2),
        'annual_return': round(float(annual_return), 2),
        'max_drawdown': round(float(max_drawdown), 2),
        'win_rate': round(float(win_rate), 2),
        'profit_loss_ratio': profit_loss_ratio if profit_loss_ratio != float('inf') else '∞',
        'total_trades': len(trades),
        'win_trades': len(win_trades),
        'loss_trades': len(loss_trades),
        'buy_hold_return': round(float(bh_return), 2),
        'excess_return': round(float(total_return - bh_return), 2)
    }

    # 资金曲线数据（采样，最多500个点）
    step = max(1, n // 500)
    equity_curve = [
        {'date': str(df['date'].iloc[i]), 'equity': round(float(equity[i]), 2)}
        for i in range(0, n, step)
    ]
    if equity_curve[-1]['date'] != str(df['date'].iloc[-1]):
        equity_curve.append({'date': str(df['date'].iloc[-1]), 'equity': round(float(equity[-1]), 2)})

    return equity_curve, trades, metrics


def evaluate_prediction_accuracy(df, buy_probs, sell_probs, horizon=5):
    """
    评估模型预测准确率：
      - 预测买入后 horizon 日内上涨概率
      - 预测卖出后 horizon 日内下跌概率
    """
    n = len(df)
    closes = df['close'].values
    buy_th = 0.60
    sell_th = 0.60

    buy_signals = []
    sell_signals = []

    for i in range(n - horizon):
        if buy_probs[i] >= buy_th:
            future_max = float(np.max(closes[i+1:i+1+horizon]))
            current = float(closes[i])
            rose = future_max > current * 1.01  # 未来5日内涨幅超1%
            buy_signals.append({
                'date': str(df['date'].iloc[i]),
                'prob': round(float(buy_probs[i]), 4),
                'future_return': round((future_max - current) / current * 100, 2),
                'correct': rose
            })
        if sell_probs[i] >= sell_th:
            future_min = float(np.min(closes[i+1:i+1+horizon]))
            current = float(closes[i])
            fell = future_min < current * 0.99  # 未来5日内跌幅超1%
            sell_signals.append({
                'date': str(df['date'].iloc[i]),
                'prob': round(float(sell_probs[i]), 4),
                'future_return': round((future_min - current) / current * 100, 2),
                'correct': fell
            })

    buy_acc = sum(1 for s in buy_signals if s['correct']) / len(buy_signals) * 100 if buy_signals else 0
    sell_acc = sum(1 for s in sell_signals if s['correct']) / len(sell_signals) * 100 if sell_signals else 0

    return {
        'horizon': horizon,
        'buy_signal_count': len(buy_signals),
        'buy_accuracy': round(buy_acc, 2),
        'sell_signal_count': len(sell_signals),
        'sell_accuracy': round(sell_acc, 2),
        'buy_signals_sample': buy_signals[:20],
        'sell_signals_sample': sell_signals[:20]
    }


# ============================================================
# 7. 实时信号生成
# ============================================================
def generate_realtime_signal(code):
    """
    获取最新数据，计算指标，用模型输出当前买卖信号。
    返回：signal, buy_prob, sell_prob, key_indicators, latest_data
    """
    # 获取最近150个交易日数据（确保MA60等指标有效）
    df_raw = fetch_recent_data(code, days=150)
    if len(df_raw) < 70:
        raise ValueError(f"股票 {code} 近期数据不足，无法计算实时信号")

    df = calc_indicators(df_raw)

    # 用全量数据做一次快速分析以获得模型
    trades, _, _ = find_optimal_trades(df, min_profit_pct=1.0)
    if len(trades) < 2:
        # 交易太少时降低阈值
        trades, _, _ = find_optimal_trades(df, min_profit_pct=0.5)

    buy_corr, sell_corr, _, _ = analyze_correlation(df, trades)
    model = build_model(df, buy_corr, sell_corr, top_n=6)
    buy_probs, sell_probs = compute_model_signals(df, model)

    # 最新一天的信号
    latest = df.iloc[-1]
    latest_buy_prob = float(buy_probs[-1])
    latest_sell_prob = float(sell_probs[-1])

    if latest_buy_prob >= model['buy_threshold_prob'] and latest_buy_prob > latest_sell_prob:
        signal = '买入'
        signal_strength = latest_buy_prob
    elif latest_sell_prob >= model['sell_threshold_prob'] and latest_sell_prob > latest_buy_prob:
        signal = '卖出'
        signal_strength = latest_sell_prob
    else:
        signal = '持有'
        signal_strength = max(latest_buy_prob, latest_sell_prob)

    # 触发信号的关键指标状态
    key_indicators = []
    active_rules = model['buy_rules'] if signal == '买入' else (model['sell_rules'] if signal == '卖出' else model['buy_rules'][:3] + model['sell_rules'][:3])
    for r in active_rules[:6]:
        col = r['indicator']
        val = float(latest[col])
        sign = 1 if r['direction'] == 'high' else -1
        triggered = sign * (val - r['threshold']) > 0
        key_indicators.append({
            'indicator': col,
            'value': round(val, 4),
            'threshold': r['threshold'],
            'direction': r['direction'],
            'weight': r['weight'],
            'triggered': bool(triggered)
        })

    return {
        'code': code,
        'date': str(latest['date']),
        'close': round(float(latest['close']), 2),
        'open': round(float(latest['open']), 2),
        'high': round(float(latest['high']), 2),
        'low': round(float(latest['low']), 2),
        'volume': int(latest['volume']),
        'signal': signal,
        'signal_strength': round(signal_strength, 4),
        'buy_probability': round(latest_buy_prob, 4),
        'sell_probability': round(latest_sell_prob, 4),
        'key_indicators': key_indicators,
        'model_threshold_buy': model['buy_threshold_prob'],
        'model_threshold_sell': model['sell_threshold_prob']
    }


# ============================================================
# 8. Flask 路由
# ============================================================
@app.route('/')
def index():
    """提供前端页面"""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    主分析接口。
    请求体 JSON: { code: '600519', start_date: '2023-01-01', end_date: '2023-12-31' }
    返回：完整分析结果 JSON
    """
    try:
        data = request.get_json(force=True)
        code = str(data.get('code', '')).strip()
        start_date = str(data.get('start_date', '')).strip()
        end_date = str(data.get('end_date', '')).strip()

        # 参数校验
        if not code or not code.isdigit() or len(code) != 6:
            return jsonify({'error': '股票代码格式错误，请输入6位数字代码（如600519）'}), 400
        if not start_date or not end_date:
            return jsonify({'error': '请选择起始日期和结束日期'}), 400
        try:
            sd = datetime.strptime(start_date, '%Y-%m-%d')
            ed = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': '日期格式错误，请使用 YYYY-MM-DD 格式'}), 400
        if sd >= ed:
            return jsonify({'error': '起始日期必须早于结束日期'}), 400

        # 为了MA60等指标有效，提前90天取数
        fetch_start = (sd - timedelta(days=120)).strftime('%Y-%m-%d')
        df_raw = fetch_stock_data(code, fetch_start, end_date)
        if len(df_raw) < 70:
            return jsonify({'error': f'数据量不足（仅{len(df_raw)}条），请扩大日期范围'}), 400

        # 计算指标
        df_full = calc_indicators(df_raw)
        # 截取用户选定时间段
        df = df_full[df_full['date'] >= start_date].reset_index(drop=True)
        if len(df) < 10:
            return jsonify({'error': f'选定时间段内有效数据不足（仅{len(df)}条），请扩大范围'}), 400

        # 3. 最优买卖点
        trades, total_return, buy_hold_return = find_optimal_trades(df, min_profit_pct=1.0)

        # 4. 相关性分析
        buy_corr, sell_corr, buy_profiles, sell_profiles = analyze_correlation(df, trades)

        # 5. 构建模型
        model = build_model(df, buy_corr, sell_corr, top_n=6)

        # 6. 计算模型信号
        buy_probs, sell_probs = compute_model_signals(df, model)

        # 7. 回测
        equity_curve, backtest_trades, metrics = backtest(df, model, buy_probs, sell_probs)

        # 8. 预测准确率
        accuracy = evaluate_prediction_accuracy(df, buy_probs, sell_probs, horizon=5)

        # K线数据（采样，最多800根）
        step = max(1, len(df) // 800)
        kline_data = []
        for i in range(0, len(df), step):
            row = df.iloc[i]
            kline_data.append([
                str(row['date']),
                round(float(row['open']), 2),
                round(float(row['close']), 2),
                round(float(row['low']), 2),
                round(float(row['high']), 2),
                int(row['volume'])
            ])

        # 买卖点标记坐标
        buy_markers = []
        sell_markers = []
        for t in trades:
            # 找到对应K线索引
            for i, kd in enumerate(kline_data):
                if kd[0] == t['buy_date']:
                    buy_markers.append({'index': i, 'date': t['buy_date'], 'price': t['buy_price']})
                    break
            for i, kd in enumerate(kline_data):
                if kd[0] == t['sell_date']:
                    sell_markers.append({'index': i, 'date': t['sell_date'], 'price': t['sell_price']})
                    break

        # 模型概率曲线（采样）
        prob_curve = []
        for i in range(0, len(df), step):
            prob_curve.append({
                'date': str(df['date'].iloc[i]),
                'buy_prob': round(float(buy_probs[i]), 4),
                'sell_prob': round(float(sell_probs[i]), 4)
            })

        result = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'data_points': len(df),
            # 最优买卖点
            'optimal_trades': trades,
            'optimal_total_return': total_return,
            'optimal_buy_hold_return': buy_hold_return,
            'trade_count': len(trades),
            # K线
            'kline_data': kline_data,
            'buy_markers': buy_markers,
            'sell_markers': sell_markers,
            # 相关性
            'buy_correlation': buy_corr,
            'sell_correlation': sell_corr,
            'buy_profiles': buy_profiles,
            'sell_profiles': sell_profiles,
            # 模型
            'model': model,
            'probability_curve': prob_curve,
            # 回测
            'equity_curve': equity_curve,
            'backtest_trades': backtest_trades,
            'metrics': metrics,
            'accuracy': accuracy
        }
        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'分析过程出错: {str(e)}'}), 500


@app.route('/api/signal', methods=['GET'])
def api_signal():
    """实时信号接口：?code=600519"""
    try:
        code = request.args.get('code', '').strip()
        if not code or not code.isdigit() or len(code) != 6:
            return jsonify({'error': '股票代码格式错误'}), 400
        signal = generate_realtime_signal(code)
        return jsonify(signal)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取实时信号失败: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    """健康检查"""
    return jsonify({'status': 'ok', 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  A股股票分析系统启动中...")
    print("  访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
