```bash
# Manual update
python scripts/data_worker.py --update

# Run scheduler (updates at 3 PM daily after market close)
python scripts/data_worker.py --scheduler --time 15:00

# Full refresh with history (when TradingView is available)
python scripts/data_worker.py --seed --history 365
```