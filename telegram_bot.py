import os
import re
import asyncio
import threading
from flask import Flask, request
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ 未找到 TELEGRAM_BOT_TOKEN，请在 .env 文件中配置")

# 导入分析函数
from web_app import analyze_single

# Flask 应用
app = Flask(__name__)

# Telegram Bot 应用
application = Application.builder().token(TOKEN).build()

# --- 创建一个独立的后台事件循环，用于运行所有异步操作 ---
loop = asyncio.new_event_loop()

def start_background_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

# 启动后台线程
threading.Thread(target=start_background_loop, daemon=True).start()

# 同步等待初始化完成，确保接收请求前 application 已就绪
init_future = asyncio.run_coroutine_threadsafe(application.initialize(), loop)
try:
    init_future.result(timeout=10)  # 等待最多10秒
    print("✅ Telegram Application initialized")
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    raise e

# --- 辅助函数：从 HTML 报告中提取摘要 ---
def extract_summary_from_html(html: str) -> dict:
    summary = {
        "action": "未知",
        "latest_price": "N/A",
        "buy_price": "N/A",
        "sell_price": "N/A",
        "news_count": 0
    }
    action_match = re.search(r'最终建议：([📈📉⏸️\s]+[买卖持有]+)', html)
    if action_match:
        summary["action"] = action_match.group(1).strip()
    price_match = re.search(r'最新:\s*\$?([\d.]+)', html)
    if price_match:
        summary["latest_price"] = price_match.group(1)
    buy_match = re.search(r'买入:\s*\$?([\d.]+)', html)
    sell_match = re.search(r'卖出:\s*\$?([\d.]+)', html)
    if buy_match:
        summary["buy_price"] = buy_match.group(1)
    if sell_match:
        summary["sell_price"] = sell_match.group(1)
    news_matches = re.findall(r'href="[^"]+"', html)
    summary["news_count"] = len(news_matches)
    return summary

# --- 指令处理（异步）---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"👋 你好 {user.first_name}！我是你的 AI 股票分析助手。\n\n"
        f"发送 /a <股票代码> 即可获取技术分析摘要。\n"
        f"例如：/a AAPL"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
**可用指令：**
/a <股票代码> - 分析指定股票（例如 /a AAPL）
/start - 开始使用
/help - 显示此帮助信息
"""
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("请在 /a 后输入股票代码，例如：/a AAPL")
        return

    ticker = context.args[0].upper().strip()
    processing_msg = await update.message.reply_text(f"🔍 正在分析 {ticker}，请稍候...")

    try:
        fig, report_html, data_info = analyze_single(
            ticker=ticker,
            period_choice="日线",
            theme="light",
            ma5=True, ma10=True, ma20=True, ma30=False, ma60=False,
            ema20=False, bb=True, kc=False, sar=False, vwap=False,
            volume=True, macd=True, rsi=True, kdj=True
        )

        summary = extract_summary_from_html(report_html)

        reply = (
            f"**{ticker} 技术分析摘要**\n\n"
            f"📊 {data_info}\n"
            f"💰 最新价: ${summary['latest_price']}\n"
            f"📈 最终建议: {summary['action']}\n"
            f"💡 参考买入价: ${summary['buy_price']} | 卖出价: ${summary['sell_price']}\n"
            f"📰 相关新闻: {summary['news_count']} 条"
        )

        await processing_msg.edit_text(reply, parse_mode="Markdown")

    except Exception as e:
        await processing_msg.edit_text(f"❌ 分析 {ticker} 时出错：{str(e)}")

# 注册指令
application.add_handler(CommandHandler('start', start_command))
application.add_handler(CommandHandler('help', help_command))
application.add_handler(CommandHandler('a', analyze_command))

# --- Webhook 路由（同步视图，安全提交到后台事件循环）---
@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        update = Update.de_json(request.get_json(force=True), application.bot)
        future = asyncio.run_coroutine_threadsafe(application.process_update(update), loop)
        try:
            future.result(timeout=30)
        except Exception as e:
            print(f"处理 webhook 时出错: {e}")
    return 'ok'

@app.route('/')
def home():
    return '🤖 AI 股票分析机器人正在运行中...'

# --- 启动 Flask ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)