# MetaQuant Telegram Daemon

Headless daemon for NGX market analysis with Telegram alerts.

## Quick Start

1. **Create Telegram Bot**
   ```
   1. Open Telegram, search @BotFather
   2. Send: /newbot
   3. Name: MetaQuant Nigeria
   4. Username: metaquant_ngx_bot (or your choice)
   5. Copy the BOT TOKEN
   ```

2. **Get Your Chat ID**
   ```
   1. Start chat with your new bot
   2. Send any message
   3. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
   4. Find "chat":{"id": YOUR_CHAT_ID}
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your tokens
   ```

4. **Run Locally**
   ```bash
   pip install -r requirements.txt
   python main.py
   ```

5. **Deploy to Render**
   - Push to GitHub
   - New > Worker from render.yaml
   - Set environment variables in dashboard

## Commands

| Command | Description |
|---------|-------------|
| `/pathway SYMBOL` | Get price pathway prediction |
| `/flow SYMBOL` | Order flow analysis |
| `/alerts on/off` | Toggle alerts |
| `/watchlist` | View watchlist |
| `/summary` | Market synthesis |
| `/top5` | Top bull/bear signals |
| `/status` | Bot status |

## Schedule (NGX Hours: 10:00-14:30 WAT)

| Time | Job |
|------|-----|
| 08:00 | Overnight ML training, disclosure scrape |
| 09:30 | Pre-market briefing |
| 10:00 | Market open alert |
| */15 | Intraday flow scan |
| 12:00 | Midday pathway synthesis |
| 14:00 | Pre-close signals |
| 14:30 | Market close summary |
| 16:00 | Evening digest |
