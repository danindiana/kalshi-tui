# Kalshi BTC Auto-Trader — Session Retrospective
**Date:** 2026-04-12 / 2026-04-13  
**Outcome:** $15.84 lost. System now in DRY_RUN=1 with rebuild in progress.

---

## What We Built (the full arc)

We started with a working Kalshi TUI in Rust, then bolted on a Python ML trading system during this session. By the end we had:

- A live auto-trader with RSA-PSS Kalshi auth, Kelly sizing, SQLite trade log
- A GPU-accelerated LSTM inference pipeline (dual RTX 3080/3060)
- A historical candle pipeline from Coinbase Exchange (523k rows, 1 year)
- A running L2 order book collector (SQLite, every 60s)
- A rebuilt trainer: delta-model, chronological split, backtest gate
- Systemd services for both the auto-trader and order book collector
- A comprehensive post-mortem of what broke and why

---

## What We Got Right

### Authentication and API Integration
The Kalshi RSA-PSS auth (`kalshi_auth.py`) worked on the first real attempt. The signing scheme — `{timestamp_ms}{METHOD}{path}` via SHA-256/MGF1 PSS — was implemented correctly. No auth errors in production.

### Kelly Criterion Sizing
The Kelly formula was correctly implemented with fractional Kelly (0.25×). The sizing was conservative: never risked more than 25% of fractional Kelly stake. In isolation this was fine.

### GPU Acceleration (Eventually)
After significant troubleshooting, we got both GPUs working with TF 2.21.0. The key fix chain:
- `pip install nvidia-cuda-nvcc-cu12` → correct ptxas + libdevice.10.bc for CUDA 12.5.1
- `XLA_FLAGS=--xla_gpu_cuda_data_dir=<venv>/nvidia/cuda_nvcc`
- `pip install nvidia-nccl-cu12` for MirroredStrategy
- Inference dropped from ~20s (CPU) to ~4.7s (dual GPU)

### Historical Data Pipeline
Switched from Binance (geo-blocked, HTTP 451) → CryptoCompare (7-day limit) → Coinbase Exchange public API. The Coinbase paginated fetcher works reliably: 300 bars/chunk, 0.12s sleep, 1 year = 523k rows.

### Delta Model Architecture
The rebuilt trainer (`train_delta_model.py`) is architecturally correct:
- Predicts `price[t+1] - price[t]` — stationary distribution, no regime sensitivity
- All features are relative (price_rel_ma, momentum, volatility, RSI, ma_slope) — no absolute price
- Chronological split, never shuffled
- Backtest gate before deployment
- Bias dropped to +$0.02/min (vs +$5-15/min systematic bias in the old model)

### Safety Rails (After the Fact)
Once we understood the failure modes, we implemented:
- `DRY_RUN=1` in the service file — halts all live trading
- `MAX_HORIZON_MINUTES=20` — LSTM has no signal at 30+ min horizons
- Candidate file pattern — model never overwrites production unless MAE improves AND backtest passes
- Daily loss limit (already had this; midnight reset was the gap)

---

## What Went Wrong (Detailed Post-Mortem)

### Mistake 1: Shuffled Train/Test Split on Time Series

**What happened:** The original model was trained with `sklearn.train_test_split(random_state=42)` on a single day of 1440 candles. This shuffles the data before splitting, meaning the model was trained on bars from 2pm to predict bars from 9am — impossible in production, trivial during training.

**Reported MAE:** $118.95 (on shuffled test set)  
**Actual forward-looking MAE:** ~$894 (discovered after rebuilding with chronological split)

**Lesson:** For any time-series ML, the test set must be a contiguous future slice. `train_test_split()` must never be used on temporal data. The correct code is always:
```python
split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
```

### Mistake 2: Absolute Price Prediction Creates Systematic Bias

**What happened:** The LSTM learned to predict absolute BTC price. On the training data (mostly a rising market), it learned "add a small positive drift." This appeared as edge in shuffled testing. In production, during flat/declining markets, this bias meant every YES bet was placed on a wrong direction.

**Evidence:** The model would output $75,998 when BTC was at $71,200 — a $4,800 upward error not from the LSTM "thinking," but from a baked-in regime assumption.

**Lesson:** Never predict absolute price levels. Predict changes (deltas) or normalized ratios. The distribution of 1-min deltas is ~N(-0.15, 55.83) — stationary across all price regimes. A scaler trained on this generalizes across any BTC price level.

### Mistake 3: Model Deployed Regardless of Quality

**What happened:** The original trainer called `model.save()` unconditionally after every training run. When we retrained on 90 days with a broken XLA config, it saved a model with MAE=$2,691 over the previous MAE=$118 model. The auto-trader loaded it and immediately started placing trades with a wildly wrong model.

**Compounding factor:** The `wait $PID` logic used in the background training pipeline didn't work correctly — the subshell returned immediately, so training ran on the old 1441-row CSV before the fetch completed.

**Lesson:** Always gate deployment. The correct pattern:
1. Save to `*_candidate.keras` first
2. Evaluate against previous production metrics  
3. Only `shutil.move()` to production if new model is better
4. For a trading system, add a backtest gate: don't deploy unless simulated bets show positive EV

### Mistake 4: No Maximum Horizon — Trading Near-Certainty Strikes

**What happened:** `CUTOFF_MINUTES=10` was supposed to prevent trading too close to settlement. But there was no *maximum* horizon. The system would happily trade when 45+ minutes remained until settlement. A 1-minute LSTM has literally zero predictive signal at 45-minute horizon. The market price at those horizons is already an efficient aggregation of all available information.

**Fix:** Added `MAX_HORIZON_MINUTES=20`. Only trade when 10–20 minutes remain — close enough that the LSTM's 1-min prediction is relevant, far enough to avoid settlement edge collapse.

### Mistake 5: Daily Loss Limit Reset at Midnight

**What happened:** `daily_loss()` counted only trades from "today" (`datetime.now().date()`). After the first day's losses, the still-broken 90-day model kept running. At 00:30 AM CDT, the daily loss counter reset to $0, and the model deployed another $8.46 in trades in a single 30-minute timer fire.

**Lesson:** A "daily" loss limit is dangerous when models are unreliable. The limit should be `DRY_RUN=1` until the model is verified, not a dollar-amount safety net.

### Mistake 6: Going Live Too Fast

**What happened:** We confirmed the system "worked" (auth, orders, mock tests passed), verified GPU acceleration, and went live — all within the same session. We had zero paper trading history. We had no baseline for whether the model's predicted edges were real.

**The critical gap:** The model was never validated on out-of-sample chronological data before live trading. "Works" ≠ "has edge."

**Lesson:** The go-live gate should be:
1. Chronological backtest shows positive EV on holdout data
2. Paper trade for at least 48 hours, verify DRY_RUN predictions would have been correct
3. *Then* DRY_RUN=0

---

## Why the Delta Model Also Has No Edge (Yet)

The delta model (`train_delta_model.py`) trained correctly but showed:
- MAE = $25.40/min  
- Naive baseline (predict zero every time) = $25.42/min
- Backtest bets simulated = 0 (no edge found, no bets placed)

This is *not a bug* — it's the efficient market hypothesis in practice. Public 1-minute OHLCV data on BTC is among the most widely-studied price series in existence. Every hedge fund, quant shop, and ML researcher has run LSTM variants on it. There is no exploitable edge in that feature set.

The path forward is **order book features**: bid/ask imbalance, spread, top-of-book depth. These reflect real-time order flow that isn't priced into the OHLCV bars. Literature (Cont et al. 2014, Zhang et al. 2019) shows L2 book features have 50-100 basis points of edge at 1-minute horizons, decaying to zero at 5+ minutes.

We have the order book collector running. We need weeks of data before retraining with OB features.

---

## The Path Forward

### Phase 1: Data Accumulation (Now → ~4 weeks)
- `orderbook-collector.service` is running, capturing L2 snapshots every 60s
- Need ~2,000 rows minimum for training; 10,000+ for robust features
- Target: 30 days of data = ~43,200 snapshots
- Monitor: `sqlite3 ~/.local/share/kalshi-tui/orderbook.db "SELECT count(*) FROM snapshots"`

### Phase 2: Retrain with Order Book Features
- Run `train_delta_model.py` modified to include: `bid_imbalance`, `spread`, `depth_ratio`
- Already supported by `train_production_model_v2.py` (OB feature merging implemented)
- Deployment gate: `avg_ev > 0 AND bets >= 50` in backtest
- If backtest passes, paper trade for 48 hours

### Phase 3: Go Live (Only After Gate Passes)
1. Delta model with OB features passes backtest
2. 48h paper trading with positive simulated PnL
3. Edit `/etc/systemd/system/kalshi-auto-trader.service`: set `DRY_RUN=0`
4. `sudo systemctl daemon-reload && sudo systemctl restart kalshi-auto-trader.timer`
5. Monitor first 48h closely

### Phase 4: Bankroll Management
- Current balance: ~$4.16 (of $20 deposited)
- Don't add more capital until the model has demonstrated positive EV in paper trading
- With $4-5 bankroll, `MAX_STAKE_DOLLARS=0.50`, `DAILY_LOSS_LIMIT=2.00`
- Kelly sizing with 0.25× fraction naturally caps position sizes

---

## Lessons for Future ML Trading Systems

1. **Backtest before anything.** Chronological. Never shuffled. Gate deployment on positive out-of-sample EV.
2. **Predict changes, not levels.** Absolute prices are non-stationary. Deltas are stationary.
3. **Paper trade first.** At least 48 hours. Real market, no real money.
4. **Understand your baseline.** If your LSTM can't beat "predict zero every time," it has learned nothing.
5. **Loss limits aren't a safety net for broken models.** If the model is broken, `DRY_RUN=1`.
6. **Maximum horizon matters as much as minimum.** A 1-min model has zero signal at 30+ min.
7. **Candidate files + atomic swap.** Never overwrite production without gating on quality.
8. **Watch for systematic bias.** Plot `mean(predictions - actuals)`. Near zero is correct. Large positive → your model has learned to always bet up.
9. **Order book > OHLCV for short-horizon.** Public price data is efficiently priced. Order flow at the bid/ask is not.
10. **Read the error messages.** `MAE=$2691` is not a normal number. Stop and investigate before anything goes live.

---

## Current System State

| Component | Status |
|-----------|--------|
| Auto-trader service | Running, `DRY_RUN=1` |
| Order book collector | Running, accumulating data |
| Delta model | Trained, NOT deployed (no backtest edge) |
| Production model | Old absolute model (MAE=$894 chrono) — will be replaced |
| Kalshi balance | ~$4.16 |
| Go-live gate | Backtest + 48h paper trade required |

---

*Generated 2026-04-13.*
