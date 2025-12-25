# Telegram Bot for MetaQuant

import logging
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from config import Config

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for MetaQuant alerts and commands."""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = None
        self.bot = None
        
    async def start(self):
        """Initialize and start the bot."""
        self.config.validate()
        
        # Build application
        self.app = Application.builder().token(self.config.telegram_bot_token).build()
        self.bot = self.app.bot
        
        # Register command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("pathway", self.cmd_pathway))
        self.app.add_handler(CommandHandler("flow", self.cmd_flow))
        self.app.add_handler(CommandHandler("alerts", self.cmd_alerts))
        self.app.add_handler(CommandHandler("watchlist", self.cmd_watchlist))
        self.app.add_handler(CommandHandler("summary", self.cmd_summary))
        self.app.add_handler(CommandHandler("top5", self.cmd_top5))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("test", self.cmd_test))  # Test all jobs
        
        # Start polling
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
        logger.info("Telegram bot started")
        
    async def stop(self):
        """Stop the bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        logger.info("Telegram bot stopped")
    
    async def send_alert(self, message: str, parse_mode: str = 'HTML'):
        """Send an alert to the configured chat."""
        if self.bot:
            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message,
                parse_mode=parse_mode
            )
    
    # ========== COMMAND HANDLERS ==========
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome = """
🏦 <b>MetaQuant Nigeria</b>
━━━━━━━━━━━━━━━━━━━━

Welcome to your NGX trading intelligence bot!

<b>Commands:</b>
/pathway SYMBOL - Price pathway prediction
/flow SYMBOL - Order flow analysis
/alerts on|off - Toggle alerts
/watchlist - Manage watchlist
/summary - Market synthesis
/top5 - Top signals
/status - Bot status

<b>Market Hours:</b> 10:00 - 14:30 WAT
        """
        await update.message.reply_html(welcome.strip())
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await self.cmd_start(update, context)
    
    async def cmd_pathway(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pathway SYMBOL command."""
        if not context.args:
            await update.message.reply_text("Usage: /pathway SYMBOL\nExample: /pathway MTNN")
            return
        
        symbol = context.args[0].upper()
        await update.message.reply_text(f"🔮 Generating pathway for {symbol}...")
        
        # Import analyzer
        from analyzers.pathway import generate_pathway_alert
        result = await generate_pathway_alert(self.config, symbol)
        await update.message.reply_html(result)
    
    async def cmd_flow(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /flow SYMBOL command."""
        if not context.args:
            await update.message.reply_text("Usage: /flow SYMBOL\nExample: /flow DANGCEM")
            return
        
        symbol = context.args[0].upper()
        await update.message.reply_text(f"📊 Analyzing flow for {symbol}...")
        
        from analyzers.flow import generate_flow_alert
        result = await generate_flow_alert(self.config, symbol)
        await update.message.reply_html(result)
    
    async def cmd_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alerts on|off command."""
        if not context.args:
            await update.message.reply_text("Usage: /alerts on|off")
            return
        
        state = context.args[0].lower()
        if state == 'on':
            await update.message.reply_text("🔔 Alerts enabled")
        elif state == 'off':
            await update.message.reply_text("🔕 Alerts disabled")
        else:
            await update.message.reply_text("Usage: /alerts on|off")
    
    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /watchlist command."""
        watchlist = self.config.default_watchlist
        formatted = "\n".join([f"• {s}" for s in watchlist])
        await update.message.reply_html(f"<b>📋 Watchlist</b>\n\n{formatted}")
    
    async def cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summary command."""
        await update.message.reply_text("📊 Generating market summary...")
        
        from analyzers.market_intel import generate_market_summary
        result = await generate_market_summary(self.config)
        await update.message.reply_html(result)
    
    async def cmd_top5(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /top5 command."""
        await update.message.reply_text("🎯 Scanning for top signals...")
        
        from analyzers.signals import get_top_signals
        result = await get_top_signals(self.config, limit=5)
        await update.message.reply_html(result)
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        from datetime import datetime
        import pytz
        
        ngx_tz = pytz.timezone('Africa/Lagos')
        now = datetime.now(ngx_tz)
        
        status = f"""
<b>🤖 Bot Status</b>
━━━━━━━━━━━━━━━
Time: {now.strftime('%H:%M:%S WAT')}
Date: {now.strftime('%Y-%m-%d')}
Watchlist: {len(self.config.default_watchlist)} symbols
Status: ✅ Running
        """
        await update.message.reply_html(status.strip())
    
    async def cmd_test(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /test command - Run all scheduled jobs for testing."""
        await update.message.reply_text("🧪 Running ALL scheduled jobs for testing...")
        
        from scheduler.jobs import ScheduledJobs
        jobs = ScheduledJobs(self.config, self)
        
        try:
            await update.message.reply_text("1️⃣ Running overnight processing...")
            await jobs.overnight_processing()
            
            await update.message.reply_text("2️⃣ Running pre-market briefing...")
            await jobs.pre_market_brief()
            
            await update.message.reply_text("3️⃣ Running market open...")
            await jobs.market_open()
            
            await update.message.reply_text("4️⃣ Running intraday scan...")
            await jobs.intraday_scan()
            
            await update.message.reply_text("5️⃣ Running midday synthesis...")
            await jobs.midday_synthesis()
            
            await update.message.reply_text("6️⃣ Running pre-close positioning...")
            await jobs.pre_close()
            
            await update.message.reply_text("7️⃣ Running market close...")
            await jobs.market_close()
            
            await update.message.reply_text("8️⃣ Running evening digest...")
            await jobs.evening_digest()
            
            await update.message.reply_text("✅ All 8 jobs completed successfully!")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Test failed: {e}")

