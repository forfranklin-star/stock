#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持续学习模块
============
存储和更新模型从历史数据中学到的参数：
  - 新闻敏感度：不同类型公告/新闻发布后的股价平均反应
  - 资金流模式：主力资金流入/流出后的价格延续概率
  - 指标权重更新：基于回测表现动态调整

学习数据以JSON文件存储在 learning/ 目录下，每次分析后自动更新。
"""
import os
import json
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.join(BASE_DIR, 'learning')
os.makedirs(LEARNING_DIR, exist_ok=True)


def _load(filename):
    """加载学习数据文件"""
    filepath = os.path.join(LEARNING_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save(filename, data):
    """保存学习数据文件"""
    filepath = os.path.join(LEARNING_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return filepath


# ============================================================
# 新闻敏感度学习
# ============================================================
def update_news_sensitivity(code, news_type, timing, price_reaction, holding_days=5):
    """
    更新某只股票的新闻敏感度参数。

    参数:
      code: 股票代码
      news_type: 新闻类型（业绩预告/增持/减持/重组/监管/合同/其他）
      timing: 'intraday'(盘中) 或 'after_hours'(盘后)
      price_reaction: 新闻后N日的涨跌幅(%)
      holding_days: 持有天数
    """
    data = _load('news_sensitivity.json') or {}
    key = f"{code}_{news_type}_{timing}"
    if key not in data:
        data[key] = {
            'code': code, 'news_type': news_type, 'timing': timing,
            'count': 0, 'sum_reaction': 0.0, 'sum_sq': 0.0,
            'positive_count': 0, 'negative_count': 0,
            'avg_reaction': 0.0, 'std_reaction': 0.0,
            'win_rate': 0.0, 'last_updated': '',
            'history': []
        }
    entry = data[key]
    entry['count'] += 1
    entry['sum_reaction'] += price_reaction
    entry['sum_sq'] += price_reaction ** 2
    if price_reaction > 0:
        entry['positive_count'] += 1
    else:
        entry['negative_count'] += 1
    entry['avg_reaction'] = round(entry['sum_reaction'] / entry['count'], 4)
    if entry['count'] > 1:
        variance = entry['sum_sq'] / entry['count'] - entry['avg_reaction'] ** 2
        entry['std_reaction'] = round(max(np.sqrt(max(variance, 0)), 0.01), 4)
    entry['win_rate'] = round(entry['positive_count'] / entry['count'] * 100, 2)
    entry['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 保留最近50条历史
    entry['history'].append({'date': datetime.now().strftime('%Y-%m-%d'),
                              'reaction': round(price_reaction, 4), 'holding_days': holding_days})
    entry['history'] = entry['history'][-50:]
    _save('news_sensitivity.json', data)
    return entry


def get_news_sensitivity(code, news_type=None, timing=None):
    """查询新闻敏感度学习结果"""
    data = _load('news_sensitivity.json') or {}
    results = []
    for key, entry in data.items():
        if entry['code'] != code:
            continue
        if news_type and entry['news_type'] != news_type:
            continue
        if timing and entry['timing'] != timing:
            continue
        results.append(entry)
    return results


def get_all_news_sensitivity():
    """获取全部新闻敏感度数据"""
    return _load('news_sensitivity.json') or {}


# ============================================================
# 资金流模式学习
# ============================================================
def update_capital_flow_pattern(code, flow_direction, flow_strength, price_followup, days=3):
    """
    更新资金流模式学习参数。

    参数:
      code: 股票代码
      flow_direction: 'inflow'(主力净流入) 或 'outflow'(主力净流出)
      flow_strength: 资金流强度（占流通市值比例%）
      price_followup: 后续N日涨跌幅(%)
      days: 跟踪天数
    """
    data = _load('capital_flow_patterns.json') or {}
    # 按强度分档：弱(<1%), 中(1-3%), 强(>3%)
    if abs(flow_strength) < 1:
        strength_bin = 'weak'
    elif abs(flow_strength) < 3:
        strength_bin = 'medium'
    else:
        strength_bin = 'strong'
    key = f"{code}_{flow_direction}_{strength_bin}"
    if key not in data:
        data[key] = {
            'code': code, 'flow_direction': flow_direction, 'strength_bin': strength_bin,
            'count': 0, 'sum_followup': 0.0, 'sum_sq': 0.0,
            'positive_count': 0, 'avg_followup': 0.0, 'std_followup': 0.0,
            'continuation_rate': 0.0, 'last_updated': '', 'history': []
        }
    entry = data[key]
    entry['count'] += 1
    entry['sum_followup'] += price_followup
    entry['sum_sq'] += price_followup ** 2
    # 延续率：资金流入后上涨 / 资金流出后下跌
    if (flow_direction == 'inflow' and price_followup > 0) or \
       (flow_direction == 'outflow' and price_followup < 0):
        entry['positive_count'] += 1
    entry['avg_followup'] = round(entry['sum_followup'] / entry['count'], 4)
    if entry['count'] > 1:
        variance = entry['sum_sq'] / entry['count'] - entry['avg_followup'] ** 2
        entry['std_followup'] = round(max(np.sqrt(max(variance, 0)), 0.01), 4)
    entry['continuation_rate'] = round(entry['positive_count'] / entry['count'] * 100, 2)
    entry['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry['history'].append({'date': datetime.now().strftime('%Y-%m-%d'),
                              'followup': round(price_followup, 4),
                              'flow_strength': round(flow_strength, 4)})
    entry['history'] = entry['history'][-50:]
    _save('capital_flow_patterns.json', data)
    return entry


def get_capital_flow_patterns(code):
    """查询某只股票的资金流模式学习结果"""
    data = _load('capital_flow_patterns.json') or {}
    return {k: v for k, v in data.items() if v['code'] == code}


# ============================================================
# 主力仓位学习（持续跟踪）
# ============================================================
def update_main_force_position(code, estimated_position, estimated_cost, date):
    """
    更新主力仓位估算（持续跟踪）。
    每次分析后更新，形成仓位变化轨迹。
    """
    data = _load('main_force_positions.json') or {}
    if code not in data:
        data[code] = {'code': code, 'positions': [], 'current_position': 0.0,
                       'current_cost': 0.0, 'last_updated': ''}
    entry = data[code]
    entry['positions'].append({'date': date, 'position': round(estimated_position, 4),
                                 'cost': round(estimated_cost, 4)})
    entry['positions'] = entry['positions'][-200:]  # 保留最近200条
    entry['current_position'] = round(estimated_position, 4)
    entry['current_cost'] = round(estimated_cost, 4)
    entry['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _save('main_force_positions.json', data)
    return entry


def get_main_force_position(code):
    """查询主力仓位学习结果"""
    data = _load('main_force_positions.json') or {}
    return data.get(code, None)


# ============================================================
# 学习状态总览
# ============================================================
def get_learning_summary():
    """获取学习状态总览"""
    news = _load('news_sensitivity.json') or {}
    flows = _load('capital_flow_patterns.json') or {}
    positions = _load('main_force_positions.json') or {}
    return {
        'news_sensitivity_entries': len(news),
        'news_codes_learned': len(set(v['code'] for v in news.values())),
        'capital_flow_entries': len(flows),
        'flow_codes_learned': len(set(v['code'] for v in flows.values())),
        'position_codes_tracked': len(positions),
        'learning_dir': LEARNING_DIR
    }


# ============================================================
# 批量导出/导入（支持跨部署迁移）
# ============================================================
def export_all_learning():
    """
    导出全部学习数据为一个JSON对象，支持下载到本地。
    返回: dict（包含所有学习文件内容和元信息）
    """
    export_data = {
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0',
        'files': {}
    }
    for filename in ['news_sensitivity.json', 'capital_flow_patterns.json', 'main_force_positions.json']:
        data = _load(filename)
        if data:
            export_data['files'][filename] = data
    # 统计
    export_data['summary'] = get_learning_summary()
    return export_data


def import_learning_data(import_data, merge=True):
    """
    从本地上传的JSON导入学习数据。

    参数:
      import_data: dict（export_all_learning导出的格式）
      merge: True=合并（追加历史记录），False=覆盖

    返回: dict（导入结果统计）
    """
    if not isinstance(import_data, dict) or 'files' not in import_data:
        return {'success': False, 'error': '格式不正确，缺少files字段'}

    imported = {}
    for filename, data in import_data['files'].items():
        if filename not in ['news_sensitivity.json', 'capital_flow_patterns.json', 'main_force_positions.json']:
            continue
        if merge:
            existing = _load(filename) or {}
            # 合并：对每个key，累加count和sum
            for key, entry in data.items():
                if key in existing and isinstance(existing[key], dict) and 'count' in existing[key]:
                    ex = existing[key]
                    ex['count'] += entry.get('count', 0)
                    ex['sum_reaction'] = ex.get('sum_reaction', 0) + entry.get('sum_reaction', 0)
                    ex['sum_sq'] = ex.get('sum_sq', 0) + entry.get('sum_sq', 0)
                    ex['positive_count'] = ex.get('positive_count', 0) + entry.get('positive_count', 0)
                    if ex['count'] > 0:
                        ex['avg_reaction'] = round(ex['sum_reaction'] / ex['count'], 4)
                        ex['win_rate'] = round(ex['positive_count'] / ex['count'] * 100, 2)
                    ex['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # 合并历史记录（去重）
                    if 'history' in entry and 'history' in ex:
                        seen = set(h.get('date', '') for h in ex['history'])
                        for h in entry['history']:
                            if h.get('date', '') not in seen:
                                ex['history'].append(h)
                        ex['history'] = ex['history'][-100:]
                else:
                    existing[key] = entry
            _save(filename, existing)
        else:
            _save(filename, data)
        imported[filename] = len(data) if isinstance(data, dict) else 0

    return {
        'success': True,
        'imported_files': imported,
        'merge_mode': merge,
        'new_summary': get_learning_summary()
    }


def export_model(model):
    """
    导出单个模型为JSON，支持下载到本地。
    """
    import copy
    export = copy.deepcopy(model)
    export['export_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    export['export_version'] = '1.0'
    return export


def import_model(import_data):
    """
    从本地上传的JSON导入模型。
    返回: model dict 或 None
    """
    if not isinstance(import_data, dict):
        return None
    # 验证必要字段（name 与 model_name 兼容）
    has_name = 'name' in import_data or 'model_name' in import_data
    required = ['buy_rules', 'sell_rules', 'buy_threshold_prob', 'sell_threshold_prob']
    if not has_name:
        return None
    for field in required:
        if field not in import_data:
            return None
    # 兼容 model_name → name
    if 'model_name' in import_data and 'name' not in import_data:
        import_data['name'] = import_data['model_name']
    # 移除导出时添加的元信息
    import_data.pop('export_time', None)
    import_data.pop('export_version', None)
    return import_data


# ============================================================
# 模型权重自更新（基于回测表现持续学习）
# ============================================================
def update_model_weights_from_backtest(model, backtest_trades, metrics):
    """
    基于回测结果自动调整模型指标权重。

    策略：
      - 盈利交易中触发的指标 → 权重增加
      - 亏损交易中触发的指标 → 权重减少
      - 胜率高的指标 → 权重增加
      - 调整幅度与回测交易数成正比（交易越多调整越可信）

    返回: 更新后的model
    """
    if not backtest_trades or len(backtest_trades) < 3:
        return model  # 交易太少，不调整

    import copy
    model = copy.deepcopy(model)

    # 统计每个指标在盈利/亏损交易中的触发次数
    indicator_stats = {}  # {indicator: {'win_trigger': 0, 'loss_trigger': 0, 'win_total': 0, 'loss_total': 0}}

    for trade in backtest_trades:
        is_win = trade.get('return_pct', 0) > 0
        # 买入时触发的指标
        buy_triggered = trade.get('buy_indicators_triggered', [])
        for ind in buy_triggered:
            if ind not in indicator_stats:
                indicator_stats[ind] = {'win_trigger': 0, 'loss_trigger': 0, 'win_total': 0, 'loss_total': 0}
            if is_win:
                indicator_stats[ind]['win_trigger'] += 1
            else:
                indicator_stats[ind]['loss_trigger'] += 1
        # 统计总盈利/亏损交易数
        if is_win:
            for ind in indicator_stats:
                indicator_stats[ind]['win_total'] += 1
        else:
            for ind in indicator_stats:
                indicator_stats[ind]['loss_total'] += 1

    # 调整权重
    learning_rate = min(0.1, 3.0 / len(backtest_trades))  # 学习率随交易数衰减
    adjustments = []

    for rules_key in ['buy_rules', 'sell_rules']:
        if rules_key not in model:
            continue
        for rule in model[rules_key]:
            ind = rule.get('indicator', '')
            if ind in indicator_stats:
                stats = indicator_stats[ind]
                total_trigger = stats['win_trigger'] + stats['loss_trigger']
                if total_trigger >= 2:
                    trigger_win_rate = stats['win_trigger'] / total_trigger
                    # 基准胜率（所有交易的胜率）
                    overall_win_rate = metrics.get('win_rate', 50) / 100
                    # 调整方向：指标触发胜率 > 整体胜率 → 增加权重
                    delta = (trigger_win_rate - overall_win_rate) * learning_rate
                    old_weight = rule.get('weight', 1.0)
                    new_weight = max(0.1, min(3.0, old_weight + delta))
                    rule['weight'] = round(new_weight, 4)
                    rule['last_weight_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    adjustments.append({
                        'indicator': ind,
                        'side': 'buy' if rules_key == 'buy_rules' else 'sell',
                        'old_weight': round(old_weight, 4),
                        'new_weight': round(new_weight, 4),
                        'trigger_win_rate': round(trigger_win_rate * 100, 2),
                        'delta': round(delta, 4)
                    })

    model['weight_learning'] = {
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backtest_trades_used': len(backtest_trades),
        'learning_rate': round(learning_rate, 4),
        'adjustments': adjustments,
        'total_adjustments': len(adjustments)
    }
    return model
