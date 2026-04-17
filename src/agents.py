import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
import pandas as pd

from src.base_agent import BaseAgent
from src.tools import calculate_bollinger_bands, calculate_macd, calculate_obv, calculate_rsi, get_prices, prices_to_df
from src.llm_config import llm_config

logger = logging.getLogger(__name__)

# ---------- MarketDataAgent ----------
class MarketDataAgent(BaseAgent):
    def __init__(self, ticker="AAPL", start_date="2024-01-01", end_date="2024-12-31", user_name=None):
        super().__init__(name="Market Data Agent", user_name=user_name)
        self.last_update = 0
        self.update_interval = 0
        self.market_data = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date
        }

    async def initialize(self, user_name=None):
        await super().initialize(user_name)
        await self.broadcast_thought("Analyzing current market conditions and preparing data streams.", private=False)

    async def process(self):
        if time.time() - self.last_update < self.update_interval:
            return

        try:
            await self.broadcast_thought("Fetching market data...")
            ticker = self.market_data.get("ticker", "AAPL")
            start = self.market_data.get("start_date", "2024-01-01")
            end = self.market_data.get("end_date", "2024-12-31")
            
            self.logger.info(f"Fetching {ticker} from {start} to {end}")
            prices = get_prices(ticker, start, end)
            
            if prices is not None and not prices.empty:
                prices_dict = {
                    'index': [str(idx) for idx in prices.index],
                    'data': prices.to_dict('records')
                }
                await self.broadcast_message({
                    "prices": prices_dict,
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat()
                }, "market_data")
                self.last_update = time.time()
                await self.broadcast_thought(f"Market data fetched successfully for {ticker}")
            else:
                await self.broadcast_thought(f"No data returned for {ticker}")
        except Exception as e:
            logger.error(f"Error in MarketDataAgent: {e}")
            await self.broadcast_thought(f"Data fetch error: {str(e)}")

    async def handle_message(self, message: dict):
        if message["type"] == "user_message":
            content = message["content"]
            if isinstance(content, dict) and "ticker" in content:
                self.market_data["ticker"] = content["ticker"]
                self.last_update = 0

# ---------- QuantitativeAgent ----------
class QuantitativeAgent(BaseAgent):
    def __init__(self, user_name=None):
        super().__init__(name="Quantitative Agent", user_name=user_name)
        self.last_analysis = 0
        self.analysis_interval = 0

    async def initialize(self, user_name=None):
        await super().initialize(user_name)
        await self.broadcast_thought("Calibrating trading algorithms and preparing strategic analysis.", private=False)

    async def process(self):
        if time.time() - self.last_analysis < self.analysis_interval:
            return

        try:
            await self.broadcast_thought("Analyzing market data...")
            if "prices" in self.state:
                prices_data = self.state["prices"]
                df_data = prices_data['data']
                df = pd.DataFrame(df_data)
                df.index = pd.to_datetime(prices_data['index'])
                
                bb_upper, bb_lower = calculate_bollinger_bands(df)
                macd_line, signal_line = calculate_macd(df)
                rsi = calculate_rsi(df)
                obv = calculate_obv(df)
                
                signals = []
                macd_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
                signals.append("bullish" if macd_diff > 0 else "bearish")
                rsi_value = rsi.iloc[-1]
                signals.append("bullish" if rsi_value < 30 else "bearish" if rsi_value > 70 else "neutral")
                price = df["close"].iloc[-1]
                bb_position = (price - (bb_upper.iloc[-1] + bb_lower.iloc[-1])/2) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                signals.append("bullish" if bb_position < -1 else "bearish" if bb_position > 1 else "neutral")
                
                analysis = {
                    "signals": signals,
                    "indicators": {
                        "bollinger_bands": {"upper": bb_upper.to_dict(), "lower": bb_lower.to_dict()},
                        "macd": {"macd": macd_line.to_dict(), "signal": signal_line.to_dict()},
                        "rsi": rsi.to_dict(),
                        "obv": obv.to_dict(),
                    },
                    "timestamp": datetime.now().isoformat()
                }
                await self.broadcast_message(analysis, "technical_analysis")
                self.last_analysis = time.time()
                await self.broadcast_thought("Technical analysis completed")
            else:
                await self.broadcast_thought("No market data available for analysis")
        except Exception as e:
            logger.error(f"Error in QuantitativeAgent: {e}", exc_info=True)
            await self.broadcast_thought(f"Analysis error: {str(e)}")

    async def handle_message(self, message: dict):
        if message["type"] == "market_data":
            self.state["prices"] = message["content"]["prices"]
            self.last_analysis = 0

# ---------- RiskManagementAgent ----------
class RiskManagementAgent(BaseAgent):
    def __init__(self, user_name=None):
        super().__init__(name="Risk Management Agent", user_name=user_name)
        self.last_assessment = 0
        self.assessment_interval = 0

    async def initialize(self, user_name=None):
        await super().initialize(user_name)
        await self.broadcast_thought("Assessing portfolio risk levels and preparing mitigation strategies.", private=False)

    async def process(self):
        if time.time() - self.last_assessment < self.assessment_interval:
            return

        try:
            await self.broadcast_thought("Assessing portfolio risk...")
            if "technical_analysis" in self.state:
                signals = self.state["technical_analysis"]["signals"]
                bullish = signals.count("bullish")
                bearish = signals.count("bearish")
                if bearish > bullish:
                    risk_level = "high"
                    max_position = 0.05
                elif bullish > bearish:
                    risk_level = "low"
                    max_position = 0.15
                else:
                    risk_level = "medium"
                    max_position = 0.10
                assessment = {
                    "risk_level": risk_level,
                    "max_position_size": max_position,
                    "stop_loss": 0.02,
                    "timestamp": datetime.now().isoformat()
                }
                await self.broadcast_message(assessment, "risk_assessment")
                self.last_assessment = time.time()
                await self.broadcast_thought(f"Risk assessment completed: {risk_level} risk")
            else:
                await self.broadcast_thought("No technical analysis available for risk assessment")
        except Exception as e:
            logger.error(f"Error in RiskManagementAgent: {e}")
            await self.broadcast_thought(f"Risk error: {str(e)}")

    async def handle_message(self, message: dict):
        if message["type"] == "technical_analysis":
            self.state["technical_analysis"] = message["content"]
            self.last_assessment = 0

# ---------- PortfolioManagementAgent ----------
class PortfolioManagementAgent(BaseAgent):
    def __init__(self, user_name=None):
        super().__init__(name="Portfolio Management Agent", user_name=user_name)
        self.last_decision = 0
        self.decision_interval = 0

    async def initialize(self, user_name=None):
        await super().initialize(user_name)
        await self.broadcast_thought("Analyzing portfolio composition and preparing recommendations.", private=False)

    async def process(self):
        if time.time() - self.last_decision < self.decision_interval:
            return

        try:
            await self.broadcast_thought("Making trading decisions...")
            if "technical_analysis" in self.state and "risk_assessment" in self.state:
                analysis = self.state["technical_analysis"]
                risk = self.state["risk_assessment"]
                signals = analysis["signals"]
                bullish = signals.count("bullish")
                bearish = signals.count("bearish")
                if bullish > bearish and risk["risk_level"] != "high":
                    action = "buy"
                    reason = "Bullish signals with acceptable risk"
                elif bearish > bullish or risk["risk_level"] == "high":
                    action = "sell"
                    reason = "Bearish signals or high risk"
                else:
                    action = "hold"
                    reason = "Mixed signals or neutral risk"
                decision = {
                    "action": action,
                    "reason": reason,
                    "max_position_size": risk["max_position_size"],
                    "stop_loss": risk["stop_loss"],
                    "timestamp": datetime.now().isoformat()
                }
                await self.broadcast_message(decision, "trading_decision")
                self.last_decision = time.time()
                await self.broadcast_thought(f"Trading decision made: {action}")
            else:
                await self.broadcast_thought("Insufficient data to make trading decision")
        except Exception as e:
            logger.error(f"Error in PortfolioManagementAgent: {e}")
            await self.broadcast_thought(f"Decision error: {str(e)}")

    async def handle_message(self, message: dict):
        if message["type"] == "technical_analysis":
            self.state["technical_analysis"] = message["content"]
        elif message["type"] == "risk_assessment":
            self.state["risk_assessment"] = message["content"]
            self.last_decision = 0

# ---------- 主运行逻辑 ----------
async def async_main(args):
    market_agent = MarketDataAgent(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date
    )
    quant_agent = QuantitativeAgent()
    risk_agent = RiskManagementAgent()
    portfolio_agent = PortfolioManagementAgent()
    
    agents = [market_agent, quant_agent, risk_agent, portfolio_agent]
    
    for agent in agents:
        await agent.initialize()
    
    await market_agent.process()
    await asyncio.sleep(0.5)
    await quant_agent.process()
    await asyncio.sleep(0.5)
    await risk_agent.process()
    await asyncio.sleep(0.5)
    await portfolio_agent.process()
    await asyncio.sleep(0.5)
    
    final_result = {
        "market_data_agent": {
            "ticker": args.ticker,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "data_fetched": "prices" in quant_agent.state
        },
        "quant_agent": {
            "signals": quant_agent.state.get("technical_analysis", {}).get("signals", [])
        },
        "risk_management_agent": {
            "risk_level": risk_agent.state.get("risk_assessment", {}).get("risk_level", "unknown")
        },
        "portfolio_management_agent": {
            "action": portfolio_agent.state.get("trading_decision", {}).get("action", "hold"),
            "reason": portfolio_agent.state.get("trading_decision", {}).get("reason", "")
        }
    }
    
    for agent in agents:
        await agent.stop()
    
    return final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--show-reasoning', action='store_true', help='Show reasoning from each agent')
    args = parser.parse_args()
    
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")
    
    result = asyncio.run(async_main(args))
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    for agent_name, data in result.items():
        print(f"{agent_name}:")
        for k, v in data.items():
            print(f"  {k}: {v}")
    print("="*50)