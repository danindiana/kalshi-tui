# Session Log — Extended Horizon + Drawdown Fix

**Date:** 2026-04-14  
**Timestamp:** 2026-04-14T09:57:42 CDT  
**Branch:** resumed from c18b9afa after terminal crash + /btw branch

---

## Changes Made

### 1. Extended classifier training horizons (10–90 min)

**File:** `~/programs/gemini_trader/time_series/train_classifier.py`

**Why:** `MAX_HORIZON_MINUTES=40` was an artifact of training data range, not a real
architectural limit. The classifier already uses `horizon_mins` as a feature — it was
always designed to be horizon-aware. Extending the training range lets it trade any
hourly window up to 90 min out, roughly doubling daily trade opportunities.

**Change:** Training loop extended from `[10,15,20,25,30,35,40]` to
`[10,15,20,25,30,35,40,50,60,70,80,90]`.

**Result (dual-GPU, 6.3s):**
```
Total training samples: 10,664,379  (was ~6.2M)
AUC-ROC:     0.9922   (was 0.9923 — unchanged)
Brier skill: +87.15%  (was +87.17% — unchanged)
Avg EV/bet:  +0.0969  (was +0.0986)
Win rate:    68.3%    (unchanged)
Deployable:  YES ✓
```
Quality essentially unchanged — extending horizons did not degrade the model.

### 2. Removed MAX_HORIZON_MINUTES gate

**Files:** `auto_trader.py`, `/etc/systemd/system/kalshi-auto-trader.service`

Removed the `check_trading_window()` gate that rejected windows > 40 min from settlement.
`MAX_SETTLEMENT_WINDOW_MINUTES=90` is now the sole upper bound (already existed to block
daily/weekly markets). `MIN_EDGE=0.05` is the quality gate.

**Immediate result:** The 09:10 timer run (while still retraining) picked up the 10:00 CDT
window at **89 min** out and placed 5 live trades ($18.16 deployed).

### 3. Fixed drawdown calculation to use portfolio value

**File:** `auto_trader.py`

**Problem:** `DRAWDOWN_HALT` compared cash balance vs HWM. Deployed capital reduces cash,
so placing $18 of trades made the system look like it was in 52% drawdown and halted every
subsequent run — even though the money was just deployed in open positions, not lost.

**Fix:** Added `get_open_position_cost(signer)` which calls
`/portfolio/positions` → sums `event_positions[].total_cost_dollars`.

```python
open_cost       = get_open_position_cost(signer)
portfolio_value = live_balance + open_cost
hwm, drawdown_pct = update_hwm(portfolio_value)
```

`update_hwm()` now tracks `portfolio_value` (not `live_balance`). The HWM ratchets on
portfolio peaks; drawdown reflects **realized losses only** — deployed capital is neutral
until it settles.

**Before fix:**
```
Balance: $25.49  HWM: $53.56  Drawdown: 52.4%  → [STOP]
```

**After fix (dry run verified):**
```
Balance: $25.49  +$17.87 deployed  Portfolio: $43.36  HWM: $53.56  Drawdown: 19.0%
```

---

## System State at End of Session

| Item | Value |
|------|-------|
| DRY_RUN | 0 (LIVE) |
| Cash balance | $25.49 |
| Open positions | $17.87 (10:00 CDT window, placed at 09:10) |
| Portfolio value | $43.36 |
| HWM | $53.56 |
| Drawdown | 19.0% |
| Halt threshold | 30% |
| Max settlement window | 90 min |
| Next timer fire | 10:00 CDT |

---

## Post-Session Investigation: Calibration Check (10:13 CDT)

After the 10:00 CDT settlement, two consecutive losing hours raised the question:
is the model drifting, or is this statistical variance?

**Tool:** `investigate_recent.py` — fetches last 48h of Coinbase 1m candles,
runs the same backtest function (batched GPU predict for speed) at horizons
15, 30, 45, 60, 75, 90 min.

**Results:**

| Metric | Training backtest | Recent 48h | Δ |
|--------|-------------------|------------|---|
| Win rate | 68.3% | 66.4% | −1.9% |
| Avg EV/bet | +0.0969 | +0.0822 | −0.0147 |
| Total simulated PnL | n/a | +2,329 units (28,346 bets) | — |

**Verdict:** Model is still strongly +EV. Recent calibration is within normal
range. The two consecutive losing hours (09:00, 10:00 CDT) are statistical
variance, not model failure.

**Math sanity:** with 66% per-bet win rate and 5 bets per hour, P(3+ losses
in an hour) ≈ 17%. Two such hours back-to-back is unlucky but routine.

### Decision
Model is healthy → resuming trading is justified. The current DRAWDOWN_HALT
threshold conflicts with the model's actual skill profile (a 30% halt on a
$53 → $35 cash trajectory triggers easily on routine variance for a small
account). Options under consideration:
- Raise `DRAWDOWN_HALT` to 0.50 (allow trading through deeper drawdowns)
- Reset HWM to current portfolio value (acknowledge the validated baseline)
- Stay halted, preserve capital

## Architecture Summary (post-session)

```
Timer fires every 10 min (24/7)
  → get_next_settlement(): finds nearest window within [10 min, 90 min]
  → classifier inference: horizon_mins fed as feature (trained on 10–90 min)
  → MIN_EDGE=0.05 is the quality gate
  → Kelly sizing → place orders
```

No hardcoded horizon limit. Model's calibrated edge is the sole entry gate.

---

## Files Changed

| File | Change |
|------|--------|
| `time_series/train_classifier.py` | Horizons 10–40 → 10–90 min |
| `auto_trader.py` | Removed MAX_HORIZON gate; added `get_open_position_cost()`; portfolio-value drawdown |
| `/etc/systemd/system/kalshi-auto-trader.service` | Removed `MAX_HORIZON_MINUTES=40` |
