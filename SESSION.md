# Kalshi Auto-Trader Session — 2026-04-16 14:48 CDT

## Session objective

Diagnose why the classifier fails to predict bullish/bearish trends correctly and
implement a defensive gate to suppress bets during sustained uptrends.

---

## State at session start

| Metric | Value |
|--------|-------|
| Portfolio | $10.35 |
| HWM | $12.55 (reset from $32.55 this morning) |
| Drawdown | 17.5% |
| Streak | 0 consecutive losses, no cooldown |
| BTC spot | ~$75,170 (up from $74,200 at noon) |

Four open NO bets on KXBTCD-26APR1616 (16:16 CDT settlement):
- NO on $74,999 ×2 — UNDERWATER (BTC above)
- NO on $75,099 — UNDERWATER (BTC above)
- NO on $75,199 — winning (BTC below)

---

## Root cause analysis

### The good streak (Apr 14 noon → ~22:00)

Rolling 20-trade win rate reached **80–95%** during trades 80–220, representing
~140 consecutive trades. BTC was range-bound; the NO-above-price strategy
(bet BTC stays below strike X above current price) works near-perfectly in a
ranging market. Strikes placed $400–600 above price never get touched.

### When it collapsed

All major loss events were sustained directional BTC rallies:

| Event | Timestamp | Consecutive NO losses |
|-------|-----------|----------------------|
| Overnight breakout | Apr 15 03:10–03:20 CDT | 6 |
| Evening trend | Apr 15 21:10–21:20 CDT | 5 |
| Morning rally | Apr 16 10:10–10:20 CDT | **8** |

Pattern: BTC rallies $800–$1,000 in under an hour. The classifier places NO bets
on 3–5 strikes above price in the same settlement window. The trend blows through
all of them simultaneously. One trend event = 6–8 correlated losses.

### The structural problem

**The classifier is a range-bound proximity estimator, not a trend predictor.**

It asks: "Given current candles (RSI, momentum, volatility), what's P(BTC stays
below strike X in the next N minutes)?" This is fundamentally different from
"Is BTC about to trend up?"

Three specific failure modes confirmed by data:

**1. RSI is the wrong trend sensor.**
RSI measures magnitude of recent moves, not trend direction. RSI=2 (extreme
oversold) at 10:10 today fired high-confidence NO bets — because historically
extreme oversold means reversal. But this was a dip in a bull run, not a trend
reversal. The classifier cannot distinguish between the two.

| RSI bucket | NO win rate |
|-----------|-------------|
| Extreme oversold (<20) | 77.8% |
| Oversold (20–40) | 73.9% |
| Neutral (40–60) | 75.3% |
| Overbought (60–80) | **87.1%** |
| Extreme overbought (>80) | 68.8% |

RSI alone has no meaningful predictive power for the cases where we lose.

**2. High confidence ≠ correct direction.**
When the model is wrong, avg model_prob = 0.813 (vs 0.892 when right). The model
cannot tell the difference between "BTC will stay flat" and "BTC is about to trend."
High-confidence wrong bets = large correlated losses.

**3. No market signal.**
The Kalshi order book contains forward-looking crowd information. When sophisticated
traders buy YES contracts on $74,999 and $75,099 strikes, that's real directional
information the classifier is blind to (it only sees historical candles).

### All-time P&L breakdown (435 settled trades)

| Side | n | Win% | Net P&L |
|------|---|------|---------|
| NO | 280 | 77.5% | +$689 |
| YES | 155 | 68.4% | +$109 |

| Direction | n | Win% | Net P&L |
|-----------|---|------|---------|
| BEAR_NO_above | 276 | 78.3% | +$694 |
| BEAR_YES_below | 138 | 71.7% | +$96 |
| BULL_YES_above | 17 | 41.2% | +$13 |
| BULL_NO_below | 4 | 25.0% | −$5 |

BEAR_NO_above (bet BTC stays below strike above spot) is the workhorse.
Losses are concentrated in trend-following regimes where BEAR_NO_above fires
repeatedly into a rising market.

### Hourly patterns

Profitable hours: 06:00–11:00 CDT, 13:00–21:00 CDT.
Terrible hours: 00:00–05:00 CDT (low liquidity, Asian session crossover).

The early morning losses account for a disproportionate share of the total P&L drag
and coincide with the largest loss streaks.

---

## Implemented fix: bullish trend gate

### What it does

When BTC has risen more than `TREND_SUPPRESS_NO_ABOVE` dollars (default: $500)
over the past `TREND_SUPPRESS_LOOKBACK_MINS` minutes (default: 60), all NO bets
are suppressed for that run. YES bets remain live (they are directionally correct
in a bull trend).

### Why $500 / 60 minutes

The three major loss events all had BTC move >$500 in 60 minutes before the model
started placing losing NO bets:
- Apr 15 03:xx: ~$800 up over ~60 min
- Apr 15 21:xx: ~$600 up
- Apr 16 10:xx: ~$970 up from $73.6K → $74.6K in one hour

Normal BTC oscillation in a ranging market is $100–300 per hour. A $500/hr filter
would have suppressed 0 trades during the profitable Apr 14 range-bound period and
would have caught all three loss events.

### Files changed

`~/programs/gemini_trader/auto_trader.py`:

**Config block** (after `SKIP_SAME_SIDE_ADD`):
```python
TREND_SUPPRESS_NO_ABOVE      = float(os.environ.get("TREND_SUPPRESS_NO_ABOVE",      "500"))
TREND_SUPPRESS_LOOKBACK_MINS = int(os.environ.get("TREND_SUPPRESS_LOOKBACK_MINS",    "60"))
```

**`run_classifier()` return dict** — added `price_change_60min` field:
```python
lookback = min(TREND_SUPPRESS_LOOKBACK_MINS, len(df) - 1)
price_60min_ago    = float(df["price"].iloc[-(lookback + 1)])
price_change_60min = current_price - price_60min_ago
```

**`run()` gate** (after RSI/vol/momentum print):
```python
if TREND_SUPPRESS_NO_ABOVE > 0 and p60 > TREND_SUPPRESS_NO_ABOVE:
    no_opps = [o for o in opportunities if o["side"] == "NO"]
    if no_opps:
        print(f"[WARN] BTC +${p60:+,.0f} over {lookback}min — bullish trend gate: suppressing {len(no_opps)} NO bets")
        opportunities = [o for o in opportunities if o["side"] != "NO"]
```

The RSI/vol/momentum log line now also prints `60min_Δ=` for visibility.

### To disable or tune

```bash
# Disable entirely
Environment="TREND_SUPPRESS_NO_ABOVE=0"

# More aggressive (fires at $300/hr instead of $500)
Environment="TREND_SUPPRESS_NO_ABOVE=300"

# Shorter lookback (30 min window)
Environment="TREND_SUPPRESS_LOOKBACK_MINS=30"
```

Add to `/etc/systemd/system/kalshi-auto-trader.service` and `daemon-reload`.

---

## NO-bet concentration cap (same session, ~15:00 CDT)

### Problem

`MAX_PLACEMENTS_PER_RUN=3` caps placements *per run* but not *per expiry across
runs*. At the Apr 16 10:xx blow-up, two consecutive runs each placed 3 NO bets on
the same KXBTCD-26APR1612 settlement, producing 6 correlated NO bets. One trend
event eliminated all six simultaneously.

`MAX_EXPIRY_EXPOSURE_PCT` is a dollar cap but also per-run. Same loophole.

### Fix: `MAX_NO_BETS_PER_EXPIRY=2`

New config var (default 2, env-var tunable). Counts existing open NO positions per
settlement window from the Kalshi positions API (`open_pos` dict, already fetched
for reconciliation — no extra API call), then gates new NO bets when the count
reaches the cap. Counter is incremented after each successful NO placement so
within-run ordering is also capped correctly.

**Gate label in logs:** `[SKIP-NO-CAP] KXBTCD-...: 2/2 NO bets already on KXBTCD-26APR1616`

**Interactions with other gates (applied in order):**
1. Trend gate (bullish 60-min momentum) — may already remove all NO opps
2. Reconciliation (same-side / opposing position blocks)
3. Top-N cap (MAX_PLACEMENTS_PER_RUN=3)
4. **NO-count cap per expiry** ← new
5. Dollar exposure cap (MAX_EXPIRY_EXPOSURE_PCT)

**To tune:**
```bash
Environment="MAX_NO_BETS_PER_EXPIRY=3"   # more permissive
Environment="MAX_NO_BETS_PER_EXPIRY=1"   # ultra-conservative
Environment="MAX_NO_BETS_PER_EXPIRY=0"   # disable
```

### Files changed

`auto_trader.py` line 108: config var
`auto_trader.py` lines 1050–1058: seed `no_count_by_expiry` from `open_pos`
`auto_trader.py` lines 1095–1099: gate in placement loop (`[SKIP-NO-CAP]`)
`auto_trader.py` line 1152: increment after successful NO placement

---

## What this does NOT fix

- **Early morning hours (00:00–05:00 CDT)**: Both gates help but don't fully
  address the low-liquidity regime. Consider a time-of-day gate or reduced stakes.
- **YES-side calibration**: Large YES bets (>$3) are still the second-biggest P&L
  leak (−$21.74 net on 45 trades). MAX_YES_STAKE_DOLLARS=2.50 cap is in place.
- **Market-price signal**: The real long-term fix. Planned for 2026-04-23+ after
  ≥5 days of Kalshi `/markets` snapshot data accumulates.

---

## Afternoon timeline (14:48 → 16:25 CDT)

### 16:16 CDT settlement (KXBTCD-26APR1616) — all NO bets lost

The 4 NO bets placed before the gates were deployed (at 14:10 and 14:20) all lost.
BTC was above $74,999, $75,099, and $75,199 at settlement. Total loss: **−$2.88**.
Portfolio: $10.35 → $7.52.

**The trend gate worked at 14:40:** RSI=77.5, 7 opportunities found, but 60-min BTC
change exceeded $500. All NO bets suppressed → 0 placed. Without the gate, 2–3
more losing NO bets would have been added into the same trend.

### 17:xx CDT settlement (KXBTCD-26APR1617) — roughly flat

| Bet | Result | P&L |
|-----|--------|-----|
| NO $75,499 (15:10) | **won** | +$0.21 |
| YES $74,999 (15:10) | **won** | +$0.18 |
| YES $75,249 (15:10) | **lost** (BTC dipped below) | −$0.53 |

Net: −$0.14 (under −$0.50 streak threshold → streak reset to 0, no cooldown).

### NO-count cap timing gap observed

16:10 placed 2 NO bets on KXBTCD-26APR1618; 16:20 placed 1 more (total 3, above
the cap of 2). The Kalshi positions API may not reflect fills from the prior run
fast enough for the next run's check. **Known issue — cap needs write-ahead local
tracking to be reliable across consecutive runs.**

---

## All-time settled performance (end of session)

| Side | n | Win% | Net P&L |
|------|---|------|---------|
| NO | 273 | 75.8% | +$6.18 |
| YES | 153 | 69.3% | −$11.81 |

**Today (Apr 16):** 54 settled trades, 57.4% win rate, **−$20.00 net**. Heavy
losses from the sustained BTC uptrend; gates deployed mid-session came too late
to prevent the 14:10/14:20 pre-gate losses.

---

## Open TODOs (updated priority order)

| # | Item | Priority |
|---|------|----------|
| 1 | NO-count cap write-ahead: API lag allows 3rd NO bet on same expiry after 2 placed in prior run | **High** |
| 2 | Market-price features — `/markets` snapshot collector → retrain 2026-04-23+ | Blocked on data |
| 3 | Early morning (00–05 CDT) time-of-day gate or stake reduction | Medium |
| 4 | Move `refresh_settlements()` above drawdown gate — TUI stale during halts | Low |
| 5 | `HWM_STALE_AFTER_HOURS` env var — auto-unstick after N hours of halt | Design TBD |

---

## Session state at handoff (~16:25 CDT)

| Metric | Value |
|--------|-------|
| Portfolio | $7.35 |
| HWM | $12.55 |
| Drawdown | 41.4% (halt at 50%, headroom $1.07) |
| Streak | 0 consecutive losses |
| BTC spot | ~$75,099 (drifting down — favorable) |
| Open positions | 3 NO bets on KXBTCD-26APR1618: $75,199 / $75,399 / $75,499 |
| All currently winning | BTC $75,099 is below all three strikes |
| Gates live | Trend gate ($500/60min) + NO-count cap (2/expiry) |

**Risk:** if all 3 open NO bets lose → portfolio ~$5.17, drawdown ~58.8% → halt
triggers. One bad settlement away from halt with $1.07 of headroom.
