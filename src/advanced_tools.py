# src/advanced_tools.py

import pandas as pd
import numpy as np
import ta
import finnhub
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Tuple

load_dotenv()


# --- 1. 实时报价与新闻功能 ---
def get_real_time_quote(ticker: str) -> dict | None:
    """使用 yfinance 获取实时报价（免费、无需 API Key）"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return None
        last_price = hist['Close'].iloc[-1]
        day_high = hist['High'].iloc[-1]
        day_low = hist['Low'].iloc[-1]
        day_open = hist['Open'].iloc[-1]
        return {
            'c': last_price,
            'h': day_high,
            'l': day_low,
            'o': day_open
        }
    except Exception as e:
        print(f"yfinance 获取报价失败: {e}")
        return None


def get_market_news(ticker: str = None, limit: int = 5) -> list[dict]:
    """获取新闻，优先 Finnhub，回退 yfinance（自动处理大小写）"""
    if not ticker:
        return []
    
    ticker = ticker.strip().upper()
    api_key = os.getenv("FINNHUB_API_KEY")
    
    # 1. 尝试 Finnhub
    if api_key:
        try:
            client = finnhub.Client(api_key=api_key)
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            news = client.company_news(ticker, _from=from_date, to=to_date)
            if news:
                print(f"✅ Finnhub 返回 {len(news)} 条新闻 for {ticker}")
                return news[:limit]
            else:
                print(f"⚠️ Finnhub 对 {ticker} 未返回新闻（可能不在免费支持范围内）")
        except Exception as e:
            print(f"❌ Finnhub 调用失败: {e}")

    # 2. 回退到 yfinance
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        yf_news = stock.news
        if yf_news:
            print(f"✅ yfinance 返回 {len(yf_news)} 条新闻 for {ticker}")
            converted = []
            for item in yf_news[:limit]:
                # 注意：yfinance 新闻字段为 'title' 和 'link'
                converted.append({
                    'headline': item.get('title', '无标题'),
                    'url': item.get('link', '#'),
                    'source': item.get('publisher', ''),
                    'datetime': item.get('providerPublishTime', '')
                })
            return converted
    except Exception as e:
        print(f"❌ yfinance 新闻获取失败: {e}")

    print(f"⚠️ 无任何新闻源返回数据 for {ticker}")
    return []


# --- 2. 支撑与压力位计算 ---
def calculate_pivot_points(df: pd.DataFrame) -> dict:
    """
    计算经典的“枢轴点”(Pivot Points)支撑/压力位 (Floor Pivot)。
    它基于上一个交易日的 High, Low, Close 来预测今日的关键点位。
    """
    last_row = df.iloc[-1]
    h, l, c = last_row['high'], last_row['low'], last_row['close']

    pivot_points = {}
    pivot_points['P'] = (h + l + c) / 3
    pivot_points['R1'] = (2 * pivot_points['P']) - l
    pivot_points['R2'] = pivot_points['P'] + (h - l)
    pivot_points['R3'] = h + 2 * (pivot_points['P'] - l)
    pivot_points['S1'] = (2 * pivot_points['P']) - h
    pivot_points['S2'] = pivot_points['P'] - (h - l)
    pivot_points['S3'] = l - 2 * (h - pivot_points['P'])

    return pivot_points


def calculate_support_resistance_volume_profile(df: pd.DataFrame, bins: int = 30) -> dict:
    """
    使用“成交量剖面”(Volume Profile)法计算支撑/压力位。
    它找出成交量最大的价格区间，这些区域通常代表强力的支撑或压力。
    """
    price_range = df['high'].max() - df['low'].min()
    bin_size = price_range / bins
    bins_edges = [df['low'].min() + i * bin_size for i in range(bins + 1)]

    volume_profile = {}
    for i in range(len(bins_edges) - 1):
        low_edge, high_edge = bins_edges[i], bins_edges[i+1]
        mask = (df['low'] <= high_edge) & (df['high'] >= low_edge)
        volume_profile[low_edge] = df.loc[mask, 'volume'].sum()

    sorted_volumes = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
    top_3_levels = sorted_volumes[:3]

    return {f"Level_{i+1}": {"price": price, "volume": vol} for i, (price, vol) in enumerate(top_3_levels)}


# --- 3. Supertrend 指标计算 ---
def calculate_supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.0):
    """
    计算 Supertrend 指标
    """
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    hl2 = (high + low) / 2
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = pd.Series(index=df.index, dtype=float)
    final_lb = pd.Series(index=df.index, dtype=float)
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(period, len(df)):
        if pd.isna(basic_ub.iloc[i]) or pd.isna(final_ub.iloc[i-1]):
            continue
        if basic_ub.iloc[i] < final_ub.iloc[i-1] or close.iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]

        if basic_lb.iloc[i] > final_lb.iloc[i-1] or close.iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]

        if pd.isna(st.iloc[i-1]) or pd.isna(final_ub.iloc[i-1]):
            continue
        if st.iloc[i-1] == final_ub.iloc[i-1]:
            if close.iloc[i] <= final_ub.iloc[i]:
                st.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = -1
            else:
                st.iloc[i] = final_lb.iloc[i]
                direction.iloc[i] = 1
        elif st.iloc[i-1] == final_lb.iloc[i-1]:
            if close.iloc[i] >= final_lb.iloc[i]:
                st.iloc[i] = final_lb.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i] = final_ub.iloc[i]
                direction.iloc[i] = -1

    df['SUPERT_7_3.0'] = st
    df['SUPERTd_7_3.0'] = direction
    return df