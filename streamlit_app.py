#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能量化分析系统 v2.0 - Streamlit 版本
============================================
增强功能：
  - 61个技术指标（交易员常用全覆盖）
  - 多周期：日线/60分钟/120分钟/30分钟/15分钟/周线
  - 大盘指数分析：上证/深证/创业板/沪深300/中证500等
  - 改进数学模型：全指标扫描→自动筛选→淘汰低相关→阈值加权
  - 模型灵敏度：保守/均衡/激进
  - 模型命名存储与加载
  - 实时信号（支持灵敏度和已保存模型）
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# plotly 容错导入：Streamlit Cloud 若未安装 plotly 不崩溃，给出友好提示
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # 创建占位对象，避免后续代码 AttributeError
    class _DummyFig:
        def __init__(self, *a, **kw): pass
        def add_trace(self, *a, **kw): return self
        def update_layout(self, *a, **kw): return self
        def update_yaxes(self, *a, **kw): return self
        def update_xaxes(self, *a, **kw): return self
        def show(self, *a, **kw): pass
    class _DummyGo:
        def __getattr__(self, name): return _DummyFig
    go = _DummyGo()
    make_subplots = lambda *a, **kw: _DummyFig()

import analysis


def _safe_plotly_chart(fig, **kwargs):
    """plotly 图表容错渲染：plotly 不可用时显示提示而非崩溃"""
    if PLOTLY_AVAILABLE:
        st.plotly_chart(fig, **kwargs)
    else:
        st.info("📊 图表未显示：plotly 库未安装，请按页面顶部提示安装依赖")

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="A股智能量化分析系统 v2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:26px; font-weight:700;
        background:linear-gradient(90deg,#1976d2,#388e3c); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; margin-bottom:2px; }
    .sub-header { font-size:12px; color:#666; margin-bottom:16px; }
    .signal-buy { background:linear-gradient(135deg,#ffebee,#ffcdd2); padding:20px;
        border-radius:12px; text-align:center; border:2px solid #ef5350; }
    .signal-sell { background:linear-gradient(135deg,#e0f2f1,#b2dfdb); padding:20px;
        border-radius:12px; text-align:center; border:2px solid #26a69a; }
    .signal-hold { background:linear-gradient(135deg,#fff8e1,#ffecb3); padding:20px;
        border-radius:12px; text-align:center; border:2px solid #ffa726; }
    .signal-text { font-size:36px; font-weight:800; letter-spacing:4px; }
    .formula-box { background:#263238; color:#a5d6a7; padding:14px 18px;
        border-radius:8px; font-family:'Courier New',monospace; font-size:12px;
        line-height:1.8; white-space:pre-wrap; }
    .rule-item { background:#fafafa; padding:10px 14px; border-radius:6px;
        margin-bottom:6px; border-left:3px solid #1976d2; }
    .eliminated-item { background:#f5f5f5; padding:6px 10px; border-radius:4px;
        margin-bottom:3px; font-size:12px; color:#999; }
</style>
""", unsafe_allow_html=True)

# 会话状态
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'signal_result' not in st.session_state:
    st.session_state.signal_result = None


# ============================================================
# 绘图函数
# ============================================================
def plot_kline(data, buy_markers, sell_markers):
    dates = [d['date'] for d in data]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=('K线与买卖点', '成交量'))
    fig.add_trace(go.Candlestick(
        x=dates, open=[d['open'] for d in data], high=[d['high'] for d in data],
        low=[d['low'] for d in data], close=[d['close'] for d in data],
        name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        increasing_fillcolor='#ef5350', decreasing_fillcolor='#26a69a'), row=1, col=1)
    if buy_markers:
        fig.add_trace(go.Scatter(
            x=[m['date'] for m in buy_markers], y=[m['price'] for m in buy_markers],
            mode='markers+text', marker=dict(symbol='triangle-up', size=14, color='#ef5350'),
            text=['▲']*len(buy_markers), textposition='bottom center',
            textfont=dict(color='#ef5350', size=14),
            name=f'买入点({len(buy_markers)})'), row=1, col=1)
    if sell_markers:
        fig.add_trace(go.Scatter(
            x=[m['date'] for m in sell_markers], y=[m['price'] for m in sell_markers],
            mode='markers+text', marker=dict(symbol='triangle-down', size=14, color='#26a69a'),
            text=['▼']*len(sell_markers), textposition='top center',
            textfont=dict(color='#26a69a', size=14),
            name=f'卖出点({len(sell_markers)})'), row=1, col=1)
    colors = ['#ef5350' if d['close'] >= d['open'] else '#26a69a' for d in data]
    fig.add_trace(go.Bar(x=dates, y=[d['volume'] for d in data],
                          marker_color=colors, name='成交量', showlegend=False), row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', paper_bgcolor='white')
    fig.update_xaxes(type='category', tickangle=-45, row=2, col=1)
    return fig


def plot_prob_curve(prob_curve, threshold):
    dates = [p['date'] for p in prob_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=[p['buy_prob'] for p in prob_curve],
                              mode='lines', name='买点概率', line=dict(color='#ef5350', width=2),
                              fill='tozeroy', fillcolor='rgba(239,83,80,0.1)'))
    fig.add_trace(go.Scatter(x=dates, y=[p['sell_prob'] for p in prob_curve],
                              mode='lines', name='卖点概率', line=dict(color='#26a69a', width=2),
                              fill='tozeroy', fillcolor='rgba(38,166,154,0.1)'))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#666",
                  annotation_text=f"信号阈值{threshold}", annotation_position="top right")
    fig.update_layout(height=340, hovermode='x unified', yaxis=dict(range=[0, 1], title='概率'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', paper_bgcolor='white')
    return fig


def plot_correlation(corr_data, title, top_n=20):
    top = corr_data[:top_n]
    fig = go.Figure(go.Bar(
        y=[c['indicator'] for c in reversed(top)],
        x=[c['corr'] for c in reversed(top)], orientation='h',
        marker_color=['#ef5350' if c['direction']=='high' else '#26a69a' for c in reversed(top)],
        text=[f"{c['corr']:.3f}{'★' if c['significant'] else ''}" for c in reversed(top)],
        textposition='outside'))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=420,
                      xaxis=dict(title='相关系数'), margin=dict(l=10, r=50, t=50, b=10),
                      plot_bgcolor='white', paper_bgcolor='white')
    return fig


def plot_equity(equity_curve):
    dates = [e['date'] for e in equity_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=[e['equity'] for e in equity_curve],
                              mode='lines', name='策略权益', line=dict(color='#1976d2', width=2),
                              fill='tozeroy', fillcolor='rgba(25,118,210,0.08)'))
    fig.add_trace(go.Scatter(x=dates, y=[analysis.INITIAL_CAPITAL]*len(dates),
                              mode='lines', name='初始资金', line=dict(color='#999', width=1, dash='dash')))
    fig.update_layout(height=400, hovermode='x unified',
                      yaxis=dict(title='权益(元)', tickprefix='¥'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', paper_bgcolor='white')
    return fig


# ============================================================
# 主标题
# ============================================================
st.markdown('<div class="main-header">📈 A股智能量化分析系统 v2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">61指标 · 多周期 · 大盘指数 · 自动筛选模型 · 灵敏度分级 · 模型存储 · 实时信号</div>', unsafe_allow_html=True)

# plotly 未安装时的醒目警告
if not PLOTLY_AVAILABLE:
    st.error("""
    ⚠️ **图表库 plotly 未安装**，所有图表将无法显示。

    **解决方法**：请确保 GitHub 仓库根目录下的 `requirements.txt` 包含以下内容：
    ```
    streamlit>=1.30.0
    plotly>=5.18.0
    pandas>=2.0.0
    numpy>=1.24.0
    akshare>=1.12.0
    yfinance>=0.2.30
    scipy>=1.10.0
    scikit-learn>=1.3.0
    ```
    上传/更新 `requirements.txt` 后，在 Streamlit Cloud 点击 **Manage app → ⋮ → Reboot app** 重新安装依赖。
    """)


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ 分析参数")

    analysis_type = st.radio("分析类型", ["个股", "大盘指数"], horizontal=True)

    if analysis_type == "大盘指数":
        index_options = {f"{v['name']}({k})": k for k, v in analysis.INDEX_MAP.items()}
        selected_idx = st.selectbox("选择指数", list(index_options.keys()))
        stock_code = index_options[selected_idx]
        is_index = True
    else:
        stock_code = st.text_input("股票代码", value="600118",
                                    help="6位代码，如 600118(中国卫星)、600519(茅台)、000001(平安银行)、600141(兴发集团)")
        is_index = False

    tf_options = {v['name']: k for k, v in analysis.TIMEFRAMES.items()}
    timeframe_name = st.selectbox("K线周期", list(tf_options.keys()), index=0)
    timeframe = tf_options[timeframe_name]

    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=730 if timeframe in ('daily','weekly') else 60)).strftime("%Y-%m-%d")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("起始日期", value=pd.to_datetime(default_start))
    with col_e:
        end_date = st.date_input("结束日期", value=pd.to_datetime(default_end))

    st.markdown("---")
    st.markdown("### 🧮 模型设置")

    sens_labels = {"conservative": "🛡️ 保守", "balanced": "⚖️ 均衡", "aggressive": "⚡ 激进"}
    sens_descs = {k: v['description'] for k, v in analysis.SENSITIVITY_CONFIG.items()}
    sensitivity = st.radio("模型灵敏度", list(sens_labels.keys()),
                            format_func=lambda x: sens_labels[x], horizontal=True)
    st.caption(sens_descs[sensitivity])

    min_profit = st.slider("最小交易收益率(%)", 0.5, 5.0, 1.0, 0.5)
    model_name = st.text_input("模型名称（可选）", value="",
                                help="输入名称后分析完成可保存该模型，用于后续实时信号调用")

    st.markdown("---")
    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    st.markdown("#### 📡 实时信号")
    saved_models_for_signal = analysis.list_saved_models()
    signal_model_options = ["自动构建模型（基于当前灵敏度）"]
    if st.session_state.current_model:
        signal_model_options.insert(0, f"📌 当前加载: {st.session_state.current_model['name']}")
    for m in saved_models_for_signal:
        signal_model_options.append(f"💾 {m['name']} ({m['sensitivity']})")
    selected_signal_model = st.selectbox("信号使用的模型", signal_model_options,
                                          help="选择已保存的模型生成信号，或自动构建")
    signal_btn = st.button("📡 获取实时信号", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 💾 已保存模型")
    saved_models = analysis.list_saved_models()
    if saved_models:
        for m in saved_models:
            with st.expander(f"{m['name']} ({m['sensitivity']})"):
                st.caption(f"创建: {m['created_at']}")
                st.caption(f"指标: 买{m['buy_indicators']}/卖{m['sell_indicators']}")
                if m.get('total_return') != '':
                    st.caption(f"回测: {m['total_return']}% | 胜率: {m['win_rate']}%")
                col_load, col_del = st.columns(2)
                if col_load.button("📂 加载", key=f"load_{m['name']}", use_container_width=True):
                    loaded_model, _, _ = analysis.load_model(m['name'])
                    st.session_state.current_model = loaded_model
                    st.success(f"已加载: {m['name']}")
                if col_del.button("🗑️ 删除", key=f"del_{m['name']}", use_container_width=True):
                    analysis.delete_model(m['name'])
                    st.rerun()
    else:
        st.caption("暂无保存的模型")

    if st.session_state.current_model:
        st.markdown(f"✅ 当前加载: **{st.session_state.current_model['name']}**")
        if st.button("清除加载模型", use_container_width=True):
            st.session_state.current_model = None
            st.rerun()

    # 持续学习数据管理
    st.markdown("### 🧠 持续学习数据")
    import learning as _learn
    learn_summary = _learn.get_learning_summary()
    st.caption(f"新闻敏感度: {learn_summary['news_sensitivity_entries']}条 | "
               f"资金流模式: {learn_summary['capital_flow_entries']}条 | "
               f"仓位跟踪: {learn_summary['position_codes_tracked']}只")
    le1, le2 = st.columns(2)
    with le1:
        import json
        learn_export = _learn.export_all_learning()
        learn_json = json.dumps(learn_export, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            label="⬇️ 导出学习数据",
            data=learn_json,
            file_name=f"learning_data_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )
    with le2:
        uploaded_learn = st.file_uploader("⬆️ 导入学习数据", type=['json'], key='learn_import', label_visibility="collapsed")
        if uploaded_learn is not None:
            try:
                imported = json.loads(uploaded_learn.read())
                result_imp = _learn.import_learning_data(imported, merge=True)
                if result_imp['success']:
                    st.success(f"✅ 学习数据导入成功！合并模式")
                    st.rerun()
                else:
                    st.error(f"❌ 导入失败: {result_imp.get('error', '未知错误')}")
            except Exception as e:
                st.error(f"❌ 导入失败: {e}")


# ============================================================
# 分析执行
# ============================================================
def run_analysis():
    code = stock_code.strip()
    sd = start_date.strftime("%Y-%m-%d")
    ed = end_date.strftime("%Y-%m-%d")
    if not code or not code.isdigit() or len(code) != 6:
        st.error("❌ 请输入6位数字代码"); return
    if start_date >= end_date:
        st.error("❌ 起始日期必须早于结束日期"); return
    with st.spinner(f"正在分析 {code} ({timeframe_name})..."):
        try:
            result = analysis.run_full_analysis(
                code=code, start_date=sd, end_date=ed,
                timeframe=timeframe, is_index=is_index,
                sensitivity=sensitivity, model_name=model_name or None,
                min_profit_pct=min_profit)
            st.session_state.analysis_result = result
            st.session_state.current_model = result['model']
            st.success(f"✅ 分析完成！{code} | {timeframe_name} | {result['data_points']}根K线 | "
                       f"{result['trade_count']}笔最优交易 | 模型选中{len(result['model']['buy_rules'])}买/{len(result['model']['sell_rules'])}卖指标")
        except Exception as e:
            st.error(f"❌ 分析失败: {str(e)}"); st.exception(e)


def run_signal():
    code = stock_code.strip()
    if not code or not code.isdigit() or len(code) != 6:
        st.error("❌ 请输入6位数字代码"); return

    # 解析选择的模型
    signal_model = None
    sel = selected_signal_model
    if sel.startswith("📌 当前加载:"):
        signal_model = st.session_state.current_model
    elif sel.startswith("💾"):
        # 从 "💾 模型名 (灵敏度)" 中提取模型名
        model_name = sel[2:].split(' (')[0]
        loaded, _, _ = analysis.load_model(model_name)
        signal_model = loaded
        st.session_state.current_model = loaded
    # else: 自动构建，model=None

    with st.spinner(f"正在获取 {code} 实时信号（模型: {sel}）..."):
        try:
            sig = analysis.generate_realtime_signal(
                code=code, timeframe=timeframe, is_index=is_index,
                model=signal_model, sensitivity=sensitivity)
            st.session_state.signal_result = sig
            st.success(f"✅ {sig['signal']} | 强度{sig['signal_strength']*100:.1f}% | "
                       f"买{sig['buy_probability']*100:.1f}%/卖{sig['sell_probability']*100:.1f}% | "
                       f"模型: {sig.get('model_name','自动')}")
        except Exception as e:
            st.error(f"❌ 获取信号失败: {str(e)}"); st.exception(e)


if analyze_btn:
    run_analysis()
if signal_btn:
    run_signal()


# ============================================================
# 结果展示
# ============================================================
result = st.session_state.analysis_result

if result:
    model = result['model']
    metrics = result['metrics']

    cols = st.columns(6)
    cols[0].metric("最优策略收益", f"{result['optimal_total_return']}%",
                    delta=f"vs持有{result['optimal_buy_hold_return']}%")
    cols[1].metric("模型回测收益", f"{metrics['total_return']}%",
                    delta=f"超额{metrics['excess_return']}%")
    cols[2].metric("年化收益", f"{metrics['annual_return']}%")
    cols[3].metric("最大回撤", f"{metrics['max_drawdown']}%")
    cols[4].metric("胜率", f"{metrics['win_rate']}%",
                    delta=f"{metrics['win_trades']}胜{metrics['loss_trades']}负")
    cols[5].metric(f"模型({model['sensitivity_name']})",
                    f"买{len(model['buy_rules'])}/卖{len(model['sell_rules'])}指标",
                    delta=f"淘汰{len(model['buy_eliminated'])}个")

    # 模型命名保存区域（分析完成后始终可见）
    st.markdown("#### 💾 保存当前模型")
    default_save_name = model.get('name', '') if model.get('name') and not model['name'].startswith('实时_') else ''
    save_name = st.text_input("模型名称", value=default_save_name, key="save_model_name",
                               placeholder="输入名称后点击保存，如：兴发集团_均衡_2024",
                               help="保存后可在侧边栏「已保存模型」中加载，或用于实时信号")
    col_save1, col_save2 = st.columns([1, 3])
    with col_save1:
        if st.button("💾 保存模型", type="primary", use_container_width=True):
            if not save_name.strip():
                st.error("❌ 请输入模型名称")
            else:
                analysis.save_model(model, metrics=metrics,
                                    backtest_info={'code': stock_code, 'timeframe': timeframe,
                                                   'start': result['start_date'], 'end': result['end_date']},
                                    custom_name=save_name.strip())
                st.success(f"✅ 模型 '{save_name.strip()}' 已保存！侧边栏可加载，实时信号可调用")
    with col_save2:
        st.caption(f"当前模型：{model.get('name','未命名')} | 灵敏度：{model['sensitivity_name']} | "
                   f"买点指标：{len(model['buy_rules'])}个 | 卖点指标：{len(model['sell_rules'])}个")

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 K线与买卖点", "📋 最优交易", "📊 指标相关性",
        "🧮 数学模型", "💰 回测结果", "🎯 预测准确率",
        "📰 公告新闻", "💹 资金流", "⚡ 异常信号", "📐 波段顶底"])

    with tab1:
        _safe_plotly_chart(plot_kline(result['kline_data'], result['buy_markers'], result['sell_markers']),
                        use_container_width=True)
        st.markdown("##### 模型买卖概率曲线")
        _safe_plotly_chart(plot_prob_curve(result['probability_curve'], model['buy_threshold_prob']),
                        use_container_width=True)

    with tab2:
        c1, c2, c3 = st.columns(3)
        c1.metric("最优交易笔数", result['trade_count'])
        c2.metric("复利总收益率", f"{result['optimal_total_return']}%")
        c3.metric("买入持有收益", f"{result['optimal_buy_hold_return']}%")
        if result['optimal_trades']:
            tdf = pd.DataFrame(result['optimal_trades'])
            tdf.index = range(1, len(tdf)+1); tdf.index.name = '序号'
            st.dataframe(tdf, use_container_width=True, height=400)
        else:
            st.info("无符合条件的交易")

    with tab3:
        cl, cr = st.columns(2)
        with cl:
            _safe_plotly_chart(plot_correlation(result['buy_correlation'],
                                              f"买点相关性Top20（共{len(result['buy_correlation'])}个指标）"),
                            use_container_width=True)
        with cr:
            _safe_plotly_chart(plot_correlation(result['sell_correlation'],
                                              f"卖点相关性Top20（共{len(result['sell_correlation'])}个指标）"),
                            use_container_width=True)
        with st.expander("📋 买点相关性完整数据"):
            st.dataframe(pd.DataFrame(result['buy_correlation']), use_container_width=True)
        with st.expander("📋 卖点相关性完整数据"):
            st.dataframe(pd.DataFrame(result['sell_correlation']), use_container_width=True)

    with tab4:
        st.markdown(f"**模型名称**: {model['name']}")
        st.markdown(f"**灵敏度**: {model['sensitivity_name']} ({sensitivity})")
        st.info(model['description'])
        cfg = model['sensitivity_config']
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("最低|r|", cfg['min_corr'])
        cc2.metric("最高p值", cfg['max_pvalue'])
        cc3.metric("指标范围", f"{cfg['min_indicators']}-{cfg['max_indicators']}")
        cc4.metric("概率阈值", cfg['prob_threshold'])
        st.markdown("##### 📐 模型公式")
        st.markdown(f'<div class="formula-box">{model["formula"]}</div>', unsafe_allow_html=True)

        rl1, rl2 = st.columns(2)
        with rl1:
            st.markdown(f"##### 🟢 买点规则（选中{len(model['buy_rules'])}个）")
            for r in model['buy_rules']:
                dt = "偏高买入" if r['direction']=='high' else "偏低买入"
                dc = "#ef5350" if r['direction']=='high' else "#26a69a"
                st.markdown(f"""
                <div class="rule-item">
                    <strong style="color:#1976d2;">{r['indicator']}</strong>
                    <span style="background:{dc};color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;margin-left:6px;">{dt}</span>
                    <span style="float:right;background:#fff3e0;color:#e65100;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:600;">权重{r['weight']}</span>
                    <br><span style="font-size:11px;color:#666;">
                        阈值:{r['threshold']:.4f} | r={r.get('corr',0):.3f} | p={r.get('pvalue',0):.4f}
                    </span>
                </div>""", unsafe_allow_html=True)
        with rl2:
            st.markdown(f"##### 🔴 卖点规则（选中{len(model['sell_rules'])}个）")
            for r in model['sell_rules']:
                dt = "偏高卖出" if r['direction']=='high' else "偏低卖出"
                dc = "#ef5350" if r['direction']=='high' else "#26a69a"
                st.markdown(f"""
                <div class="rule-item">
                    <strong style="color:#1976d2;">{r['indicator']}</strong>
                    <span style="background:{dc};color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;margin-left:6px;">{dt}</span>
                    <span style="float:right;background:#fff3e0;color:#e65100;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:600;">权重{r['weight']}</span>
                    <br><span style="font-size:11px;color:#666;">
                        阈值:{r['threshold']:.4f} | r={r.get('corr',0):.3f} | p={r.get('pvalue',0):.4f}
                    </span>
                </div>""", unsafe_allow_html=True)

        with st.expander(f"❌ 被淘汰指标（买点 {len(model['buy_eliminated'])} 个）"):
            for e in model['buy_eliminated'][:30]:
                st.markdown(f'<div class="eliminated-item">{e["indicator"]} | r={e["corr"]:.4f} | p={e["pvalue"]:.4f}</div>',
                            unsafe_allow_html=True)
        with st.expander(f"❌ 被淘汰指标（卖点 {len(model['sell_eliminated'])} 个）"):
            for e in model['sell_eliminated'][:30]:
                st.markdown(f'<div class="eliminated-item">{e["indicator"]} | r={e["corr"]:.4f} | p={e["pvalue"]:.4f}</div>',
                            unsafe_allow_html=True)

        # 权重自学习状态
        wl = result.get('weight_learning')
        if wl and wl.get('total_adjustments', 0) > 0:
            st.markdown("##### 🧠 模型权重自学习（本次更新）")
            st.info(f"基于{wl.get('backtest_trades_used', 0)}笔回测交易，学习率{wl.get('learning_rate', 0)}，自动调整了{wl.get('total_adjustments', 0)}个指标权重")
            adj_df = pd.DataFrame(wl.get('adjustments', []))
            if len(adj_df) > 0:
                st.dataframe(adj_df, use_container_width=True, hide_index=True)

        # 模型导出/导入
        st.markdown("##### 📦 模型导出 / 导入")
        me1, me2 = st.columns(2)
        with me1:
            import json, learning as _learn
            model_export = _learn.export_model(model)
            model_json = json.dumps(model_export, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="⬇️ 导出当前模型到本地",
                data=model_json,
                file_name=f"model_{model['name']}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        with me2:
            uploaded_model = st.file_uploader("⬆️ 从本地导入模型", type=['json'], key='model_import')
            if uploaded_model is not None:
                try:
                    imported = json.loads(uploaded_model.read())
                    imp_model = _learn.import_model(imported)
                    if imp_model:
                        st.success(f"✅ 模型「{imp_model['name']}」导入成功！可在下方保存后使用")
                        st.session_state['imported_model'] = imp_model
                    else:
                        st.error("❌ 模型文件格式不正确，缺少必要字段")
                except Exception as e:
                    st.error(f"❌ 导入失败: {e}")

    with tab5:
        st.warning("⚠️ **样本内回测说明**：本回测的模型规则（指标阈值、权重）是在同一时间段的数据上构建的，属于样本内测试（in-sample），结果可能高估模型实际表现。实盘使用前应进行样本外（out-of-sample）验证或滚动回测。")
        mc = st.columns(5)
        mc[0].metric("期末权益", f"¥{metrics['final_equity']:,.0f}")
        mc[1].metric("总收益率", f"{metrics['total_return']}%")
        mc[2].metric("年化收益率", f"{metrics['annual_return']}%")
        mc[3].metric("最大回撤", f"{metrics['max_drawdown']}%")
        mc[4].metric("盈亏比", metrics['profit_loss_ratio'])
        mc2 = st.columns(5)
        mc2[0].metric("交易次数", metrics['total_trades'])
        mc2[1].metric("盈利交易", metrics['win_trades'])
        mc2[2].metric("亏损交易", metrics['loss_trades'])
        mc2[3].metric("胜率", f"{metrics['win_rate']}%")
        mc2[4].metric("买入持有", f"{metrics['buy_hold_return']}%")
        _safe_plotly_chart(plot_equity(result['equity_curve']), use_container_width=True)
        if result['backtest_trades']:
            st.dataframe(pd.DataFrame(result['backtest_trades']), use_container_width=True, height=350)
        else:
            st.info("回测期间未产生交易")

    with tab6:
        acc = result['accuracy']
        ac1, ac2 = st.columns(2)
        ac1.metric("买入预测准确率", f"{acc['buy_accuracy']}%", delta=f"{acc['buy_signal_count']}个信号")
        ac2.metric("卖出预测准确率", f"{acc['sell_accuracy']}%", delta=f"{acc['sell_signal_count']}个信号")
        st.caption(f"预测窗口：未来{acc['horizon']}个周期 | 涨跌判定：±1%")
        as1, as2 = st.columns(2)
        with as1:
            st.markdown("**买入信号样本**")
            if acc['buy_signals_sample']:
                st.dataframe(pd.DataFrame(acc['buy_signals_sample']), use_container_width=True)
            else:
                st.info("无样本")
        with as2:
            st.markdown("**卖出信号样本**")
            if acc['sell_signals_sample']:
                st.dataframe(pd.DataFrame(acc['sell_signals_sample']), use_container_width=True)
            else:
                st.info("无样本")

    # ============================================================
    # tab7: 公告新闻事件研究
    # ============================================================
    with tab7:
        news = result.get('news_analysis')
        if not news or news.get('news_count', 0) == 0:
            if result.get('is_index') or result.get('timeframe') != 'daily':
                st.info("📰 新闻/公告分析仅支持日线级别个股分析")
            else:
                err = news.get('error', '未获取到新闻数据') if news else '新闻分析未执行'
                st.error(f"📰 {err}")
                # 显示详细数据源状态
                src_status = news.get('source_status', []) if news else []
                if src_status:
                    st.markdown("**🔍 数据源诊断：**")
                    for s in src_status:
                        icon = "✅" if s['success'] else "❌"
                        st.markdown(f"- {icon} **{s['source']}**: {s['detail']}")
                else:
                    st.info("💡 请检查网络连接，或稍后重试。东方财富直连API为首选数据源（不依赖akshare）。")
        else:
            st.markdown(f"#### 📰 公告与新闻（共{news['news_count']}条，持有{news['holding_days']}日事件研究）")
            if news.get('learning_updated'):
                st.success("🧠 已更新新闻敏感度学习参数")

            # 敏感度统计
            if news.get('sensitivity_by_type'):
                st.markdown("##### 📊 不同类型消息的股价敏感度")
                sens_df = pd.DataFrame(news['sensitivity_by_type'])
                st.dataframe(sens_df, use_container_width=True, hide_index=True)

            if news.get('sensitivity_by_timing'):
                st.markdown("##### ⏰ 盘中 vs 盘后消息影响对比")
                timing_df = pd.DataFrame(news['sensitivity_by_timing'])
                st.dataframe(timing_df, use_container_width=True, hide_index=True)

            # 事件研究结果
            if news.get('event_results'):
                st.markdown(f"##### 📋 事件研究详情（{len(news['event_results'])}条）")
                event_df = pd.DataFrame(news['event_results'])
                if 'news_date' in event_df.columns:
                    event_df['news_date'] = event_df['news_date'].astype(str).str[:19]
                st.dataframe(event_df, use_container_width=True, height=400, hide_index=True)

            # 新闻列表
            if news.get('news_list'):
                with st.expander(f"📰 新闻/公告列表（{len(news['news_list'])}条）"):
                    for n in news['news_list'][:50]:
                        timing_color = "#ef5350" if n['timing'] == '盘中' else "#1976d2"
                        cat = n.get('category', '资讯')
                        cat_color = "#6a1b9a" if cat == '公告' else "#00838f"
                        title_html = n['title']
                        if n.get('url'):
                            title_html = f'<a href="{n["url"]}" target="_blank" style="color:#333;text-decoration:none;">{n["title"]}</a>'
                        st.markdown(f"""
                        <div style="padding:6px 10px;margin-bottom:4px;background:#fafafa;border-radius:4px;border-left:3px solid {timing_color};">
                            <span style="font-size:11px;color:#888;">{n['date'][:16]}</span>
                            <span style="background:{cat_color};color:#fff;padding:1px 6px;border-radius:3px;font-size:10px;margin-left:6px;">{cat}</span>
                            <span style="background:{timing_color};color:#fff;padding:1px 6px;border-radius:3px;font-size:10px;margin-left:4px;">{n['timing']}</span>
                            <span style="background:#fff3e0;color:#e65100;padding:1px 6px;border-radius:3px;font-size:10px;margin-left:4px;">{n['news_type']}</span>
                            <br><span style="font-size:12px;">{title_html}</span>
                        </div>""", unsafe_allow_html=True)

    # ============================================================
    # tab8: 资金流分析
    # ============================================================
    with tab8:
        cf = result.get('capital_flow_analysis')
        if not cf or cf.get('flow_count', 0) == 0:
            if result.get('is_index') or result.get('timeframe') != 'daily':
                st.info("💹 资金流分析仅支持日线级别个股分析")
            else:
                err = cf.get('error', '未获取到资金流数据') if cf else '资金流分析未执行'
                st.error(f"💹 {err}")
                # 显示详细数据源状态
                src_status = cf.get('source_status', []) if cf else []
                if src_status:
                    st.markdown("**🔍 数据源诊断：**")
                    for s in src_status:
                        icon = "✅" if s['success'] else "❌"
                        st.markdown(f"- {icon} **{s['source']}**: {s['detail']}")
                else:
                    st.info("💡 资金流已按 东财→雪球(全球CDN)→akshare 三源自动降级。若全部失败，"
                            "通常是当前网络对国内财经接口访问受限，请稍后重试；K线、技术指标、模型等其他分析不受影响。")
        else:
            st.markdown(f"#### 💹 资金流分析（{cf['flow_count']}个交易日）")
            # 数据来源与口径（海外环境东财被封时自动切换雪球全球源）
            fsrc = cf.get('flow_source', '')
            if fsrc:
                if '雪球' in fsrc:
                    st.info(f"📡 数据来源：**{fsrc}**。海外服务器已自动切换至雪球全球节点；"
                            f"其主力口径(大单阈值)与东财5档不同，且仅提供近20个交易日，分档明细仅最新日可得。")
                else:
                    st.caption(f"📡 数据来源：{fsrc}")
            if cf.get('learning_updated'):
                st.success("🧠 已更新资金流模式学习参数")

            # 当前主力仓位概览
            cur = cf.get('current_position')
            if cur:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("估算主力仓位", f"{cur['position_pct']}%")
                c2.metric("估算平均成本", f"¥{cur['avg_cost']}")
                c3.metric("浮动盈亏", f"{cur['floating_pnl_pct']}%")
                c4.metric("当前行为", cur.get('behavior', '-'))

            # 行为统计
            if cf.get('behavior_stats'):
                st.markdown("##### 🎯 主力行为模式统计")
                bh_df = pd.DataFrame([{'行为模式': k, '天数': v} for k, v in cf['behavior_stats'].items()])
                st.dataframe(bh_df, use_container_width=True, hide_index=True)

            # 资金流-价格模式
            if cf.get('patterns'):
                st.markdown("##### 📈 资金流-价格延续模式（后续3日）")
                pat_df = pd.DataFrame(cf['patterns'])
                st.dataframe(pat_df, use_container_width=True, hide_index=True)

            # 仓位变化曲线数据
            if cf.get('position_data'):
                st.markdown("##### 📊 主力仓位与成本变化（最近60日）")
                pos_df = pd.DataFrame(cf['position_data'])
                st.dataframe(pos_df, use_container_width=True, height=350, hide_index=True)

            # 资金流明细
            if cf.get('flow_data'):
                with st.expander(f"💰 资金流明细（最近{len(cf['flow_data'])}日）"):
                    flow_df = pd.DataFrame(cf['flow_data'])
                    st.dataframe(flow_df, use_container_width=True, height=400, hide_index=True)

    # ============================================================
    # tab9: 异常信号捕捉（应跌未跌/应涨未涨）
    # ============================================================
    with tab9:
        anomaly = result.get('anomaly_signals')
        if not anomaly or anomaly.get('summary', {}).get('total', 0) == 0:
            st.info("⚡ 未检测到异常信号（需日线级别个股分析，且技术指标数据充足）")
        else:
            summary = anomaly['summary']
            signals = anomaly['signals']
            st.markdown(f"#### ⚡ 异常信号捕捉（共{summary['total']}个）")

            # 概览指标
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("应跌未跌(看涨)", summary.get('bullish', 0))
            c2.metric("应涨未涨(看跌)", summary.get('bearish', 0))
            c3.metric("高置信信号", summary.get('high_confidence', 0))
            c4.metric("主力吸筹信号", summary.get('main_force_accumulation', 0))
            c5.metric("主力出货信号", summary.get('main_force_distribution', 0))

            # 回测验证结果
            bt = anomaly.get('backtest')
            if bt and bt.get('overall') and bt['overall'].get('count', 0) > 0:
                st.markdown("##### 📊 异常信号历史回测验证")
                ov = bt['overall']
                bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                for idx, days in enumerate([1, 3, 5, 10, 20]):
                    avg = ov.get(f'{days}d_avg_ret')
                    wr = ov.get(f'{days}d_win_rate')
                    if avg is not None:
                        cols = [bc1, bc2, bc3, bc4, bc5]
                        cols[idx].metric(f"{days}日平均收益", f"{avg:+.2f}%", f"胜率{wr:.0f}%" if wr else "")

                # 按方向分组
                if bt.get('by_direction'):
                    st.markdown("###### 按信号方向（5日持有）")
                    dir_data = []
                    for d, s in bt['by_direction'].items():
                        dir_data.append({
                            '方向': d, '信号数': s['count'],
                            '5日平均收益': f"{s.get('5d_avg_ret', 0):+.2f}%",
                            '5日胜率': f"{s.get('5d_win_rate', 0):.1f}%",
                            '5日盈亏比': s.get('5d_profit_loss_ratio', '-'),
                        })
                    st.dataframe(pd.DataFrame(dir_data), use_container_width=True, hide_index=True)

                # 按趋势分组
                if bt.get('signal_returns'):
                    st.markdown("###### 按趋势环境（5日持有）")
                    trend_stats = {}
                    for sr in bt['signal_returns']:
                        t = sr.get('trend', 'unknown')
                        if t not in trend_stats:
                            trend_stats[t] = []
                        if sr.get('5d_ret') is not None:
                            trend_stats[t].append(sr['5d_ret'])
                    trend_data = []
                    for t, rets in trend_stats.items():
                        if rets:
                            import numpy as np
                            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
                            trend_data.append({
                                '趋势': t, '信号数': len(rets),
                                '5日平均收益': f"{np.mean(rets):+.2f}%",
                                '5日胜率': f"{wr:.1f}%",
                            })
                    if trend_data:
                        st.dataframe(pd.DataFrame(trend_data), use_container_width=True, hide_index=True)

                st.caption("💡 回测为样本内验证，仅供参考。信号方向=大趋势方向，异常类型确认趋势强度。")

            # 最新信号
            latest = summary.get('latest')
            if latest:
                direction_color = "#2e7d32" if latest['direction'] == '看涨' else "#c62828"
                st.markdown(f"""
                <div style="padding:16px;background:{'#e8f5e9' if latest['direction']=='看涨' else '#ffebee'};
                border-radius:10px;border-left:4px solid {direction_color};margin-bottom:16px;">
                    <div style="font-size:11px;color:#888;">最新异常信号 · {latest['date']}</div>
                    <div style="font-size:18px;font-weight:700;color:{direction_color};">
                        {latest['signal_type']} → {latest['direction']}（强度{latest['strength']}%，置信度{latest['confidence']}）
                    </div>
                    <div style="font-size:13px;color:#555;margin-top:6px;">{latest['interpretation']}</div>
                    <div style="font-size:11px;color:#888;margin-top:4px;">
                        收盘价: ¥{latest['close']} | 量比: {latest['volume_price'].get('volume_ratio','N/A')} |
                        量价信号: {latest['volume_price'].get('volume_signal','N/A')} |
                        主力行为: {latest.get('main_force_action','N/A')}
                    </div>
                </div>""", unsafe_allow_html=True)

            # 信号列表
            st.markdown("##### 📋 全部异常信号（按强度排序）")
            if signals:
                # 高置信信号优先展示
                high_signals = [s for s in signals if s['confidence'] == '高']
                other_signals = [s for s in signals if s['confidence'] != '高']
                display_signals = high_signals + other_signals

                for sig in display_signals[:30]:
                    d_color = "#2e7d32" if sig['direction'] == '看涨' else "#c62828"
                    bg = "#f1f8e9" if sig['direction'] == '看涨' else "#fff3e0"
                    triggers_str = " | ".join(sig['triggers'][:4])
                    vp = sig['volume_price']
                    cf = sig.get('capital_flow')

                    cf_info = ""
                    if cf:
                        main_inflow = cf.get('main_net_inflow', 0)
                        if abs(main_inflow) > 1e8:
                            cf_info = f"主力净流入{main_inflow/1e8:.2f}亿"
                        elif abs(main_inflow) > 1e4:
                            cf_info = f"主力净流入{main_inflow/1e4:.0f}万"
                        else:
                            cf_info = f"主力净流入{main_inflow:.0f}元"

                    st.markdown(f"""
                    <div style="padding:10px 14px;margin-bottom:8px;background:{bg};border-radius:8px;border-left:3px solid {d_color};">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-weight:600;color:{d_color};">{sig['signal_type']} → {sig['direction']}</span>
                            <span style="font-size:11px;color:#888;">{sig['date']} | 强度{sig['strength']}% | {sig['confidence']}置信</span>
                        </div>
                        <div style="font-size:12px;color:#555;margin-top:4px;">{sig['interpretation']}</div>
                        <div style="font-size:11px;color:#777;margin-top:3px;">
                            触发: {triggers_str} | 量比{vp.get('volume_ratio','N/A')} | {vp.get('volume_signal','N/A')}
                            {f' | {cf_info}' if cf_info else ''} | 主力:{sig.get('main_force_action','N/A')}
                        </div>
                    </div>""", unsafe_allow_html=True)

                if len(signals) > 30:
                    st.caption(f"仅显示强度最高的30个信号，共{len(signals)}个")

    # ============================================================
    # tab10: 波段顶/底识别
    # ============================================================
    with tab10:
        swing = result.get('swing_detection')
        if not swing or 'error' in swing:
            err = swing.get('error', '波段识别未执行') if swing else '波段识别未执行'
            st.error(f"📐 波段识别失败：{err}")
            st.info("💡 请检查数据是否充足（建议至少60根K线），或尝试更换股票代码/时间范围后重试。")
        else:
            st.markdown(f"#### 📐 波段顶/底识别（{swing.get('timeframe','日线')}，参数自动优化）")

            # 概览
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("波段顶部", swing.get('top_count', 0))
            c2.metric("波段底部", swing.get('bottom_count', 0))
            bt = swing.get('best_backtest', {})
            if '5d' in bt.get('tops', {}):
                c3.metric("顶部5日下跌率", f"{bt['tops']['5d']['decline_rate']}%")
                c4.metric("底部5日上涨率", f"{bt['bottoms']['5d']['rise_rate']}%")

            # 最优参数
            bp = swing.get('best_params', {})
            st.markdown("##### ⚙️ 最优参数（自动网格搜索）")
            pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
            pcol1.metric("摆动点左右", bp.get('swing_left', 3))
            pcol2.metric("ZigZag回撤", f"{bp.get('zigzag_pct', 0.08)*100:.0f}%")
            pcol3.metric("综合阈值", bp.get('top_threshold', 3))
            pcol4.metric("RSI阈值", f"{bp.get('rsi_ob',70)}/{bp.get('rsi_os',30)}")
            pcol5.metric("KDJ阈值", f"{bp.get('kdj_ob',80)}/{bp.get('kdj_os',20)}")

            # 回测验证
            st.markdown("##### 📊 回测验证（最优参数）")
            if bt:
                bt_data = []
                for days in ['3d', '5d', '10d', '20d']:
                    if days in bt.get('tops', {}) and days in bt.get('bottoms', {}):
                        t = bt['tops'][days]
                        b = bt['bottoms'][days]
                        bt_data.append({
                            '持有期': days.replace('d', '日'),
                            '顶部信号数': t['count'],
                            '顶部下跌率': f"{t['decline_rate']}%",
                            '顶部平均收益': f"{t['avg_return']:+.2f}%",
                            '底部信号数': b['count'],
                            '底部上涨率': f"{b['rise_rate']}%",
                            '底部平均收益': f"{b['avg_return']:+.2f}%",
                        })
                if bt_data:
                    st.dataframe(pd.DataFrame(bt_data), use_container_width=True, hide_index=True)

            # 各算法单独表现
            algo_perf = swing.get('algo_performance', {})
            if algo_perf:
                st.markdown("##### 🔬 各算法单独表现（5日持有）")
                algo_data = []
                for name, perf in algo_perf.items():
                    if 'error' not in perf:
                        algo_data.append({
                            '算法': name,
                            '顶部信号数': perf.get('top_count', 0),
                            '顶部下跌率': f"{perf.get('top_decline_rate', 0)}%",
                            '底部信号数': perf.get('bottom_count', 0),
                            '底部上涨率': f"{perf.get('bottom_rise_rate', 0)}%",
                        })
                if algo_data:
                    st.dataframe(pd.DataFrame(algo_data), use_container_width=True, hide_index=True)

            # 参数搜索Top5
            grid_top = swing.get('grid_search_top10', [])
            if grid_top:
                with st.expander("🔍 参数网格搜索Top10"):
                    st.dataframe(pd.DataFrame(grid_top), use_container_width=True, hide_index=True)

            # 波段点列表
            col_top, col_bottom = st.columns(2)
            with col_top:
                st.markdown("##### 🔴 波段顶部")
                tops = swing.get('top_points', [])
                if tops:
                    top_df = pd.DataFrame(tops)[['date', 'price', 'high', 'score']]
                    top_df.columns = ['日期', '收盘价', '最高价', '综合评分']
                    st.dataframe(top_df, use_container_width=True, hide_index=True, height=300)
                else:
                    st.info("无顶部信号")

            with col_bottom:
                st.markdown("##### 🟢 波段底部")
                bottoms = swing.get('bottom_points', [])
                if bottoms:
                    bot_df = pd.DataFrame(bottoms)[['date', 'price', 'low', 'score']]
                    bot_df.columns = ['日期', '收盘价', '最低价', '综合评分']
                    st.dataframe(bot_df, use_container_width=True, hide_index=True, height=300)
                else:
                    st.info("无底部信号")

            st.caption("💡 综合评分=摆动点+分形+ZigZag+MACD背离+RSI+KDJ+布林+量价确认的加权和。阈值越高信号越少但越可靠。")


# ============================================================
# 实时信号
# ============================================================
def _render_indicator_card(ind, side='buy'):
    """渲染单个指标触发状态卡片"""
    triggered = ind['triggered']
    stt = "✓ 触发" if triggered else "✗ 未触发"
    stc = "#2e7d32" if triggered else "#999"
    bg = "#e8f5e9" if triggered else "#fafafa"
    bd = "#4caf50" if triggered else "#e0e0e0"
    dt = "偏高" if ind['direction'] == 'high' else "偏低"
    side_color = "#ef5350" if side == 'buy' else "#26a69a"
    return f"""
    <div style="background:{bg};padding:8px 12px;border-radius:6px;margin-bottom:5px;border-left:3px solid {bd};">
        <strong style="color:{side_color};">{ind['indicator']}</strong>
        <span style="float:right;color:{stc};font-weight:600;font-size:12px;">{stt}</span>
        <br><span style="font-size:11px;color:#555;">
            当前值:<strong>{ind['value']:.4f}</strong> | 阈值:{ind['threshold']:.4f} |
            方向:{dt} | 权重:{ind['weight']} | r={ind.get('corr',0):.3f}
        </span>
    </div>"""


if st.session_state.signal_result:
    sig = st.session_state.signal_result
    st.markdown("---")
    st.markdown("### 📡 实时买卖信号")

    sc = 'signal-buy' if sig['signal'] == '买入' else ('signal-sell' if sig['signal'] == '卖出' else 'signal-hold')
    scolor = '#c62828' if sig['signal'] == '买入' else ('#00695c' if sig['signal'] == '卖出' else '#e65100')

    # 顶部：信号大卡片 + 操作建议说明
    top1, top2 = st.columns([1, 2])
    with top1:
        st.markdown(f"""
        <div class="{sc}">
            <div style="font-size:12px;color:#555;margin-bottom:6px;">当前操作建议</div>
            <div class="signal-text" style="color:{scolor};">{sig['signal']}</div>
            <div style="margin-top:10px;font-size:13px;">
                信号强度 <strong style="font-size:18px;color:{scolor};">{sig['signal_strength']*100:.1f}%</strong>
            </div>
            <div style="display:flex;justify-content:center;gap:16px;margin-top:12px;">
                <div><div style="font-size:10px;color:#888;">买点概率</div><div style="font-size:16px;font-weight:700;color:#ef5350;">{sig['buy_probability']*100:.1f}%</div></div>
                <div><div style="font-size:10px;color:#888;">卖点概率</div><div style="font-size:16px;font-weight:700;color:#26a69a;">{sig['sell_probability']*100:.1f}%</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with top2:
        st.markdown(f"""
        <div style="background:#f5f5f5;padding:14px 18px;border-radius:8px;border-left:4px solid {scolor};">
            <div style="font-size:13px;font-weight:600;color:#333;margin-bottom:6px;">📋 操作建议说明</div>
            <div style="font-size:13px;color:#444;line-height:1.7;">{sig.get('action_description', '')}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("")
        info1, info2, info3, info4 = st.columns(4)
        info1.caption(f"**数据时间**: {sig['date']}")
        info2.caption(f"**周期**: {sig.get('timeframe', '日线')}")
        info3.caption(f"**模型**: {sig.get('model_name', '自动构建')}")
        info4.caption(f"**收盘价**: ¥{sig['close']} | 量: {sig['volume']:,}")

    # 波段顶/底概率（无未来函数）
    swing = sig.get('swing_state')
    if swing:
        st.markdown("")
        st.markdown("##### 📐 波段顶/底概率（实时，无未来函数）")
        sw1, sw2, sw3 = st.columns([1, 1, 2])
        with sw1:
            top_pct = swing['top_probability']
            top_color = '#c62828' if top_pct > 50 else ('#ef6c00' if top_pct > 30 else '#666')
            st.markdown(f"""
            <div style="background:#fff3e0;padding:12px;border-radius:8px;text-align:center;border-left:4px solid {top_color};">
                <div style="font-size:11px;color:#888;">波段顶部概率</div>
                <div style="font-size:24px;font-weight:700;color:{top_color};">{top_pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with sw2:
            bot_pct = swing['bottom_probability']
            bot_color = '#2e7d32' if bot_pct > 50 else ('#558b2f' if bot_pct > 30 else '#666')
            st.markdown(f"""
            <div style="background:#e8f5e9;padding:12px;border-radius:8px;text-align:center;border-left:4px solid {bot_color};">
                <div style="font-size:11px;color:#888;">波段底部概率</div>
                <div style="font-size:24px;font-weight:700;color:{bot_color};">{bot_pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with sw3:
            st.markdown(f"""
            <div style="background:#f5f5f5;padding:12px 16px;border-radius:8px;">
                <div style="font-size:12px;color:#888;margin-bottom:4px;">当前波段状态</div>
                <div style="font-size:15px;font-weight:600;color:#333;">{swing['state']}</div>
                <div style="font-size:11px;color:#888;margin-top:4px;">
                    基于价格位置/RSI/KDJ/MACD/布林/量价/K线形态/均线8维度实时打分
                </div>
            </div>""", unsafe_allow_html=True)

        # 触发因素
        if swing.get('top_factors') or swing.get('bottom_factors'):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                if swing.get('top_factors'):
                    st.markdown("###### 🔴 顶部触发因素")
                    for f in swing['top_factors']:
                        st.markdown(f"- {f}")
            with fcol2:
                if swing.get('bottom_factors'):
                    st.markdown("###### 🟢 底部触发因素")
                    for f in swing['bottom_factors']:
                        st.markdown(f"- {f}")

    st.markdown("")

    # 底部：全部买点指标 + 全部卖点指标
    buy_inds = sig.get('buy_indicators', [])
    sell_inds = sig.get('sell_indicators', [])
    buy_trig = sig.get('buy_triggered_count', sum(1 for i in buy_inds if i['triggered']))
    sell_trig = sig.get('sell_triggered_count', sum(1 for i in sell_inds if i['triggered']))
    buy_total = sig.get('buy_total_count', len(buy_inds))
    sell_total = sig.get('sell_total_count', len(sell_inds))

    col_buy, col_sell = st.columns(2)

    with col_buy:
        st.markdown(f"#### 🟢 买点指标触发状态（{buy_trig}/{buy_total} 触发）")
        if buy_inds:
            for ind in buy_inds:
                st.markdown(_render_indicator_card(ind, 'buy'), unsafe_allow_html=True)
        else:
            st.info("模型无买点指标规则")

    with col_sell:
        st.markdown(f"#### 🔴 卖点指标触发状态（{sell_trig}/{sell_total} 触发）")
        if sell_inds:
            for ind in sell_inds:
                st.markdown(_render_indicator_card(ind, 'sell'), unsafe_allow_html=True)
        else:
            st.info("模型无卖点指标规则")


# 空状态
if not result and not st.session_state.signal_result:
    st.info("👈 左侧设置参数（个股/指数、周期、日期、灵敏度），点击「开始分析」或「获取实时信号」")
    st.markdown("""
    **v2.0 增强功能：**
    - 📊 **61个技术指标**：MACD/RSI/KDJ/布林/均线/ATR/ADX/SAR/BIAS/乖离/一目均衡/唐奇安/动量/资金流/MFI/CCI/WR/ROC/MOM/CMO/ULTOSC/PSY/VWAP/WMA/STD/DMA/TRIX/STOCHRSI 等全覆盖
    - ⏱️ **多周期**：日线/60分钟/120分钟/30分钟/15分钟/周线
    - 📈 **大盘指数**：上证/深证/创业板/沪深300/中证500/科创50/上证50/中证1000/中小板指
    - 🧮 **改进模型**：全指标相关性扫描→按灵敏度自动筛选→淘汰低相关指标→阈值加权打分
    - 🎚️ **灵敏度分级**：保守(|r|≥0.15,p≤0.01,阈值0.7)/均衡(|r|≥0.08,p≤0.05,阈值0.6)/激进(|r|≥0.04,p≤0.1,阈值0.5)
    - 💾 **模型存储**：命名保存模型，侧边栏管理，可加载用于实时信号
    - 📡 **实时信号**：基于最新行情+指定灵敏度/已保存模型计算买卖触发概率
    """)
