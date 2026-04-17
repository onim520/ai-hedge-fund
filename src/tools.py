import os
import warnings
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------- Alpaca 相关导入与初始化（可选） ----------
try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER_ENDPOINT = os.getenv('ALPACA_PAPER_ENDPOINT', "https://paper-api.alpaca.markets")
ALPACA_LIVE_ENDPOINT = os.getenv('ALPACA_LIVE_ENDPOINT', "https://api.alpaca.markets")

data_client = None
paper_trading_client = None
live_trading_client = None

if ALPACA_AVAILABLE and all([ALPACA_API_KEY, ALPACA_SECRET_KEY]):
    try:
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        paper_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        live_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=False)
    except Exception as e:
        warnings.warn(f"Alpaca clients initialization failed: {e}")
else:
    warnings.warn("Alpaca credentials missing or SDK not installed. Trading functions disabled.")

def get_trading_client(paper=True):
    if paper_trading_client is None and live_trading_client is None:
        raise RuntimeError("Alpaca trading clients are not available.")
    return paper_trading_client if paper else live_trading_client

# ---------- 数据获取核心函数 ----------
def get_prices(ticker, start_date, end_date):
    """
    获取股票价格数据，按优先级依次尝试：
    1. Financial Datasets（需配置 FINANCIAL_DATASETS_API_KEY）
    2. Alpaca（需配置 Alpaca 凭证）
    3. yfinance（需安装 yfinance）
    """
    import requests

    # 1. Financial Datasets
    financial_key = os.getenv("FINANCIAL_DATASETS_API_KEY")
    if financial_key:
        try:
            url = (
                f'https://api.financialdatasets.ai/prices/'
                f'?ticker={ticker}'
                f'&interval=day'
                f'&start_date={start_date}'
                f'&end_date={end_date}'
            )
            headers = {"X-API-KEY": financial_key}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            prices = data.get('prices')
            if prices:
                df = pd.DataFrame(prices)
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    return df
        except Exception as e:
            warnings.warn(f"Financial Datasets failed: {e}")

    # 2. Alpaca
    if data_client is not None:
        try:
            try:
                timeframe = TimeFrame.Day
            except AttributeError:
                timeframe = TimeFrame(1, TimeFrameUnit.Day)
            req = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=timeframe,
                start=datetime.strptime(start_date, "%Y-%m-%d"),
                end=datetime.strptime(end_date, "%Y-%m-%d")
            )
            bars = data_client.get_stock_bars(req)
            df = bars.df
            if len(df) > 0:
                return df
        except Exception as e:
            warnings.warn(f"Alpaca failed: {e}")

    # 3. yfinance (fallback)
    try:
        import yfinance as yf
        tick = yf.Ticker(ticker)
        df = tick.history(start=start_date, end=end_date)
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except ImportError:
        raise ImportError("yfinance not installed and other sources unavailable.")
    except Exception as e:
        raise RuntimeError(f"yfinance fetch failed: {e}")

    raise ValueError(f"Could not retrieve price data for {ticker} from any source.")

def prices_to_df(prices_df):
    df = prices_df.copy()
    df.index.name = "Date"
    df.columns = [c.lower() for c in df.columns]
    return df

def get_price_data(ticker, start_date, end_date):
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)

# ---------- 技术指标计算函数 ----------
def calculate_confidence_level(signals):
    sma_diff_prev = abs(signals['sma_5_prev'] - signals['sma_20_prev'])
    sma_diff_curr = abs(signals['sma_5_curr'] - signals['sma_20_curr'])
    diff_change = sma_diff_curr - sma_diff_prev
    confidence = min(max(diff_change / signals['current_price'], 0), 1)
    return confidence

def calculate_macd(prices_df):
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df, period=14):
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices_df, window=20):
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def calculate_obv(prices_df):
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']

# ---------- 交易执行函数 ----------
def execute_trade(ticker, action, quantity, paper=True, current_price=None):
    try:
        trading_client = get_trading_client(paper)
        if trading_client is None:
            return {"status": "error", "error": "Alpaca trading client not available"}
        order_side = OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)
        return {
            "status": "submitted",
            "order_id": order.id,
            "side": order_side.value,
            "quantity": quantity,
            "mode": "paper" if paper else "live"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}