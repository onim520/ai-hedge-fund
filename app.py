import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
from datetime import datetime, timedelta
import ta
import os
from dotenv import load_dotenv

try:
    from src.tools import get_prices
    from src.advanced_tools import (
        get_real_time_quote,
        get_market_news,
        calculate_pivot_points,
        calculate_support_resistance_volume_profile,
    )
except ImportError:
    pass

load_dotenv()

default_end = datetime.now().strftime("%Y-%m-%d")
default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

custom_css = """
.report-text { font-size: 18px !important; line-height: 1.6 !important; color: #0f172a !important; }
.report-text a { color: #2563eb !important; text-decoration: underline !important; }
.action-buy { color: #166534 !important; background-color: #dcfce7 !important; font-weight: 700 !important; font-size: 24px !important; padding: 16px 24px !important; border-radius: 20px !important; display: inline-block !important; border: 2px solid #22c55e !important; }
.action-sell { color: #991b1b !important; background-color: #fee2e2 !important; font-weight: 700 !important; font-size: 24px !important; padding: 16px 24px !important; border-radius: 20px !important; display: inline-block !important; border: 2px solid #ef4444 !important; }
.action-hold { color: #854d0e !important; background-color: #fef9c3 !important; font-weight: 700 !important; font-size: 24px !important; padding: 16px 24px !important; border-radius: 20px !important; display: inline-block !important; border: 2px solid #eab308 !important; }
"""

def calculate_keltner_channel(df, period=20, scalar=2):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()
    middle = ta.trend.ema_indicator(typical_price, window=period)
    upper = middle + scalar * atr
    lower = middle - scalar * atr
    return upper, middle, lower

def calculate_parabolic_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
    high = df['high'].values
    low = df['low'].values
    n = len(df)
    sar = np.zeros(n)
    trend = np.ones(n)
    af = np.zeros(n)
    ep = np.zeros(n)
    trend[0] = 1
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = af_start
    for i in range(1, n):
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
        if trend[i-1] == 1:
            if sar[i] > low[i-1]: sar[i] = low[i-1]
            if i > 1 and sar[i] > low[i-2]: sar[i] = low[i-2]
        else:
            if sar[i] < high[i-1]: sar[i] = high[i-1]
            if i > 1 and sar[i] < high[i-2]: sar[i] = high[i-2]
        if trend[i-1] == 1:
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    return pd.Series(sar, index=df.index)

def create_interactive_chart(
    df, ticker, period_choice, theme='light',
    ma_selected=None, ema20=False, bb=False, kc=False, sar=False, vwap=False,
    volume=True, macd=False, rsi=False, kdj=False
):
    if df is None or df.empty:
        return go.Figure()

    bg_color = '#ffffff' if theme == 'light' else '#0d1117'
    grid_color = '#e5e7eb' if theme == 'light' else '#30363d'
    volume_up_color = '#26a69a'
    volume_down_color = '#ef5350'

    if ma_selected:
        if 'MA5' in ma_selected: df['MA5'] = ta.trend.sma_indicator(df['close'], window=5)
        if 'MA10' in ma_selected: df['MA10'] = ta.trend.sma_indicator(df['close'], window=10)
        if 'MA20' in ma_selected: df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
        if 'MA30' in ma_selected: df['MA30'] = ta.trend.sma_indicator(df['close'], window=30)
        if 'MA60' in ma_selected: df['MA60'] = ta.trend.sma_indicator(df['close'], window=60)
    if ema20: df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    if bb:
        bb_obj = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb_obj.bollinger_hband()
        df['BB_middle'] = bb_obj.bollinger_mavg()
        df['BB_lower'] = bb_obj.bollinger_lband()
    if kc:
        df['KC_upper'], df['KC_middle'], df['KC_lower'] = calculate_keltner_channel(df)
    if sar:
        df['SAR'] = calculate_parabolic_sar(df)
    if vwap:
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    if macd:
        macd_obj = ta.trend.MACD(df['close'])
        df['MACD'] = macd_obj.macd()
        df['MACD_signal'] = macd_obj.macd_signal()
        df['MACD_hist'] = macd_obj.macd_diff()
    if rsi:
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    if kdj:
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=9, smooth_window=3)
        df['K'] = stoch.stoch()
        df['D'] = stoch.stoch_signal()
        df['J'] = 3 * df['K'] - 2 * df['D']

    subplot_names = ["K线"]
    row_heights = [0.5]
    rows = 1
    if volume:
        subplot_names.append("成交量")
        row_heights.append(0.15)
        rows += 1
    if macd:
        subplot_names.append("MACD")
        row_heights.append(0.15)
        rows += 1
    if rsi:
        subplot_names.append("RSI (14)")
        row_heights.append(0.15)
        rows += 1
    if kdj:
        subplot_names.append("KDJ (9,3,3)")
        row_heights.append(0.15)
        rows += 1

    if rows == 1:
        subplot_names.append("成交量")
        row_heights.append(0.15)
        rows += 1

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=tuple(subplot_names)
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="K线", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    ma_colors = {'MA5': '#2962ff', 'MA10': '#ff6d00', 'MA20': '#e91e63', 'MA30': '#00bfa5', 'MA60': '#d50000'}
    if ma_selected:
        for ma in ma_selected:
            if ma in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines', name=ma, line=dict(color=ma_colors.get(ma, 'gray'), width=1)), row=1, col=1)
    if ema20 and 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20', line=dict(color='#651fff', width=1.5)), row=1, col=1)
    if bb:
        for band, color, dash in [('BB_upper', 'gray', 'dash'), ('BB_middle', 'gray', None), ('BB_lower', 'gray', 'dash')]:
            if band in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[band], mode='lines', name=band, line=dict(color=color, width=0.8, dash=dash)), row=1, col=1)
    if kc:
        for band in ['KC_upper', 'KC_lower']:
            if band in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[band], mode='lines', name=band, line=dict(color='orange', width=0.8)), row=1, col=1)
    if sar and 'SAR' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SAR'], mode='markers', name='SAR', marker=dict(color='purple', size=3)), row=1, col=1)
    if vwap and 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='#00bcd4', width=1.2)), row=1, col=1)

    current_row = 2

    if volume or rows == 2:
        colors = [volume_up_color if df['close'].iloc[i] >= df['open'].iloc[i] else volume_down_color for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="成交量", marker_color=colors, showlegend=False), row=current_row, col=1)
        current_row += 1

    if macd:
        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='#2962ff')), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='信号线', line=dict(color='#ff6d00')), row=current_row, col=1)
            colors_hist = ['#26a69a' if v >= 0 else '#ef5350' for v in df['MACD_hist'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='柱状图', marker_color=colors_hist, showlegend=False), row=current_row, col=1)
        current_row += 1

    if rsi:
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#9c27b0', width=1.5)), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        current_row += 1

    if kdj:
        if 'K' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['K'], mode='lines', name='K', line=dict(color='#2962ff', width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['D'], mode='lines', name='D', line=dict(color='#ff6d00', width=1.5)), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['J'], mode='lines', name='J', line=dict(color='#9c27b0', width=1.2, dash='dot')), row=current_row, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.3, row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.3, row=current_row, col=1)
        current_row += 1

    fig.update_layout(
        template='plotly_white' if theme == 'light' else 'plotly_dark',
        height=250 * rows, hovermode='x unified', showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50), xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(gridcolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color)
    return fig

def analyze_single(
    ticker, period_choice, theme,
    ma5, ma10, ma20, ma30, ma60, ema20, bb, kc, sar, vwap,
    volume, macd, rsi, kdj
):
    if not ticker:
        return None, "请输入股票代码", "", None
    ticker = ticker.strip().upper()
    try:
        if period_choice in ["1分钟", "3分钟", "5分钟"]:
            period_map = {"1分钟": ("5d", "1m"), "3分钟": ("5d", "2m"), "5分钟": ("7d", "5m")}
            yf_period, yf_interval = period_map[period_choice]
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=yf_period, interval=yf_interval)
        else:
            df = get_prices(ticker, default_start, default_end)

        if df is None or df.empty:
            return None, f"无法获取 {ticker} 的数据", "", None

        quote = get_real_time_quote(ticker)
        news = get_market_news(ticker, limit=3)
        pivot = calculate_pivot_points(df)
        vp = calculate_support_resistance_volume_profile(df)

        rsi_val = ta.momentum.rsi(df['close'], window=14).iloc[-1]
        if rsi_val < 30:
            action, cls = "📈 买入", "action-buy"
        elif rsi_val > 70:
            action, cls = "📉 卖出", "action-sell"
        else:
            action, cls = "⏸️ 持有", "action-hold"

        buy_price = pivot['S1']
        sell_price = pivot['R1']
        latest_close = df['close'].iloc[-1]
        if latest_close > sell_price: sell_price = pivot['R2']
        if latest_close < buy_price: buy_price = pivot['S2']

        quote_html = f'<div class="report-text"><b>📡 实时报价</b> 最新: ${quote["c"]:.2f} (高:{quote["h"]:.2f} 低:{quote["l"]:.2f})</div>' if quote else ""
        pivot_html = f'<div class="report-text"><b>🎯 枢轴点</b> R3:{pivot["R3"]:.2f} R2:{pivot["R2"]:.2f} R1:{pivot["R1"]:.2f} P:{pivot["P"]:.2f} S1:{pivot["S1"]:.2f} S2:{pivot["S2"]:.2f} S3:{pivot["S3"]:.2f}</div>'
        vp_html = f'<div class="report-text"><b>📊 成交量剖面</b> 1.${vp["Level_1"]["price"]:.2f} (量:{vp["Level_1"]["volume"]:,.0f}) 2.${vp["Level_2"]["price"]:.2f} (量:{vp["Level_2"]["volume"]:,.0f}) 3.${vp["Level_3"]["price"]:.2f} (量:{vp["Level_3"]["volume"]:,.0f})</div>'
        trade_html = f'<div class="report-text" style="background:#eef2ff; border-left-color:#6366f1;"><b>💡 建议买卖参考价</b><br>🟢 买入: ${buy_price:.2f} ｜ 🔴 卖出: ${sell_price:.2f}</div>'
        news_html = '<div class="report-text"><b>📰 新闻</b> ' + '; '.join([f'<a href="{n["url"]}">{n["headline"]}</a>' for n in news]) + '</div>' if news else ""
        final_html = f'<div class="{cls}">最终建议：{action}</div>'
        report_html = final_html + quote_html + pivot_html + vp_html + trade_html + news_html

        ma_selected = []
        if ma5: ma_selected.append('MA5')
        if ma10: ma_selected.append('MA10')
        if ma20: ma_selected.append('MA20')
        if ma30: ma_selected.append('MA30')
        if ma60: ma_selected.append('MA60')

        fig = create_interactive_chart(
            df, ticker, period_choice, theme,
            ma_selected=ma_selected, ema20=ema20, bb=bb, kc=kc, sar=sar, vwap=vwap,
            volume=volume, macd=macd, rsi=rsi, kdj=kdj
        )

        data_info = f"{ticker} | {period_choice} | {len(df)} 根K线"

        # 构建结构化数据字典（用于 Telegram 机器人）
        data_dict = {
            'ticker': ticker,
            'period': period_choice,
            'data_info': data_info,
            'latest_price': quote['c'] if quote else None,
            'action': action,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'news_count': len(news) if news else 0,
        }

        return fig, report_html, data_info, data_dict
    except Exception as e:
        return None, f"<pre>{traceback.format_exc()}</pre>", "", None

if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks(title="AI 股票分析助手", css=custom_css) as demo:
        gr.Markdown("# 📊 AI 股票技术分析助手 (即时更新版)")

        with gr.Row():
            with gr.Column(scale=1):
                ticker_single = gr.Textbox(label="股票代码", value="AAPL")
                period_choice = gr.Dropdown(choices=["1分钟", "3分钟", "5分钟", "日线", "周线", "月线"], label="K线周期", value="日线")
                theme_choice = gr.Dropdown(choices=["light", "dark"], label="主题", value="light")

                gr.Markdown("### 📊 主图指标")
                with gr.Row():
                    ma5 = gr.Checkbox(label="MA5", value=True)
                    ma10 = gr.Checkbox(label="MA10", value=True)
                    ma20 = gr.Checkbox(label="MA20", value=True)
                with gr.Row():
                    ma30 = gr.Checkbox(label="MA30", value=False)
                    ma60 = gr.Checkbox(label="MA60", value=False)
                    ema20 = gr.Checkbox(label="EMA20", value=False)
                with gr.Row():
                    bb = gr.Checkbox(label="布林带", value=True)
                    kc = gr.Checkbox(label="肯特纳通道", value=False)
                with gr.Row():
                    sar = gr.Checkbox(label="SAR", value=False)
                    vwap = gr.Checkbox(label="VWAP", value=False)

                gr.Markdown("### 📉 副图指标")
                volume = gr.Checkbox(label="成交量", value=True)
                macd = gr.Checkbox(label="MACD", value=True)
                rsi = gr.Checkbox(label="RSI", value=True)
                kdj = gr.Checkbox(label="KDJ", value=True)

                btn_single = gr.Button("刷新", variant="primary")
                data_single = gr.Textbox(label="数据摘要", interactive=False)

            with gr.Column(scale=2):
                report_single = gr.HTML(label="分析报告")
                plot_single = gr.Plot(label="技术图表")

        inputs_list = [ticker_single, period_choice, theme_choice,
                       ma5, ma10, ma20, ma30, ma60, ema20, bb, kc, sar, vwap,
                       volume, macd, rsi, kdj]
        outputs_list = [plot_single, report_single, data_single]

        btn_single.click(fn=analyze_single, inputs=inputs_list, outputs=outputs_list)

        for comp in inputs_list:
            if hasattr(comp, 'change'):
                comp.change(fn=analyze_single, inputs=inputs_list, outputs=outputs_list)

        gr.Markdown("---\n💡 **提示**：修改任何选项后图表将自动更新。支持缩放、悬停查看数值。")

    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))