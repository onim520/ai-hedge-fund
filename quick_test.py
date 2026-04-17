import os
import pandas as pd
from dotenv import load_dotenv
from src.tools import calculate_macd, calculate_rsi, calculate_bollinger_bands

load_dotenv()

print("📈 正在获取 AAPL 股票数据 (2024-01-01 至 2024-12-31)...")
from src.tools import get_prices
df = get_prices('AAPL', '2024-01-01', '2024-12-31')
print(f"✅ 成功获取 {len(df)} 天的数据")

print("📊 正在计算技术指标...")
macd, signal = calculate_macd(df)
rsi = calculate_rsi(df)
bb_upper, bb_lower = calculate_bollinger_bands(df)

latest_price = df['close'].iloc[-1]
macd_val = macd.iloc[-1]
signal_val = signal.iloc[-1]
rsi_val = rsi.iloc[-1]
bb_upper_val = bb_upper.iloc[-1]
bb_lower_val = bb_lower.iloc[-1]

# 信号判断
signals = []
if macd_val > signal_val:
    signals.append(("bullish", "MACD 金叉（看涨）"))
else:
    signals.append(("bearish", "MACD 死叉（看跌）"))

if rsi_val < 30:
    signals.append(("bullish", "RSI 超卖（看涨）"))
elif rsi_val > 70:
    signals.append(("bearish", "RSI 超买（看跌）"))
else:
    signals.append(("neutral", "RSI 中性"))

if latest_price < bb_lower_val:
    signals.append(("bullish", "价格跌破布林带下轨（看涨）"))
elif latest_price > bb_upper_val:
    signals.append(("bearish", "价格突破布林带上轨（看跌）"))
else:
    signals.append(("neutral", "价格位于布林带内（中性）"))

# 基于信号计算最终决策
bullish_count = sum(1 for s in signals if s[0] == "bullish")
bearish_count = sum(1 for s in signals if s[0] == "bearish")

if bullish_count > bearish_count:
    action = "买入"
    reason = "多项技术指标呈现看涨信号，市场动能向上，风险可控。"
elif bearish_count > bullish_count:
    action = "卖出"
    reason = "空头信号占据主导，技术面转弱，建议降低仓位。"
else:
    action = "持有"
    reason = "多空信号均衡，市场方向不明，保持现有仓位观望为宜。"

# 巴菲特风格建议
advice = f"""
作为价值投资者，我们关注的是企业的长期竞争优势和合理估值，而非短期市场波动。
不过，技术指标能为我们提供市场情绪的参考。

当前苹果公司 (AAPL) 的技术面呈现如下特征：
- 最新收盘价：${latest_price:.2f}
- MACD 线：{macd_val:.4f}，信号线：{signal_val:.4f}
- RSI (14日)：{rsi_val:.2f}
- 布林带上轨：${bb_upper_val:.2f}，下轨：${bb_lower_val:.2f}

信号汇总：
{chr(10).join([s[1] for s in signals])}

综合判断：{action}。
{reason}

（注：本建议由本地规则引擎生成，未调用外部 AI 模型。）
"""

print("\n" + "="*50)
print("📋 AI 投资委员会最终建议（本地规则引擎）")
print("="*50)
print(advice)
print("="*50)

for ticker in ['AAPL', 'MSFT', 'GOOGL']:
    df = get_prices(ticker, '2024-01-01', '2024-12-31')
    # ... 复制上面的指标计算和信号判断逻辑 ...
    print(f"\n{ticker} 最终建议：{action}")