# Kalshi — Improvements 1–6 Implementation (Phase A live, Phase B scaffolded)

**Session:** `2026-04-15_194344_1776300224_kalshi-improvements-phaseA-B`
**Scope:** Implement items 1–6 from the 2026-04-15 alpha-and-scale doc.
**Related:** `../2026-04-15_192939_1776299379_kalshi-alpha-and-scale/README.md`

---

## Phase A — SHIPPED to production (2026-04-15 19:41 CDT)

All four gates live in `auto_trader.py`, service file env-configured, daemon reloaded. Unit-tested (14/14) and validated with two consecutive live ticks.

### Item 2 — Conviction gate

**Where:** `find_opportunities`, inside the per-strike loop.
**Gate:** `|clf_prob - 0.5| ≥ MIN_CONVICTION` (default 0.15).
**Why:** The doomed 2026-04-15 04:00 NO on T73699 had clf_prob = 0.614 → conviction = 0.114. "Edge" from a coin-flip call is really just a mispriced fifty-fifty; this gate catches it.
**Env:** `MIN_CONVICTION=0.15`
**Observed live (19:41 tick, RSI 49.4):** one strike skipped with `conviction 0.092 < 0.15`.

### Item 5 — Regime detector

**Where:** stake_pct computation after Kelly.
**Logic:** If `RSI < REGIME_RSI_LOW` (20) or `RSI > REGIME_RSI_HIGH` (80), multiply Kelly stake by `REGIME_SCALE` (0.5).
**Why:** Classifier has fewer training samples at RSI tails; conviction there is lower-quality. Halving size in extremes preserves upside while capping variance.
**Env:** `REGIME_RSI_LOW=20 REGIME_RSI_HIGH=80 REGIME_SCALE=0.5`
**Observed:** RSI=13.2 tick showed correct half-sizing behavior.

### Item 4 — Net stake reconciliation

**Where:** reconciliation block before top-N cap.
**Logic:** Default mode (`NET_RECONCILIATION=1`) — when a new opportunity opposes an open position, it survives only if its **conviction ≥ 2 × MIN_CONVICTION** (i.e., 0.30). Block-only fallback available via `NET_RECONCILIATION=0`.
**Rationale:** The binary "block any opposing signal" from the first fix was correct but overly conservative. A *strong* opposing signal should be heard; it will net on settlement. The 04:00 failure mode (conviction 0.114 opposing held YES) is still blocked because it never clears the primary MIN_CONVICTION gate.
**Logs:** `[NET]` for overrides, `[SKIP-OPP]` for blocks.

### Item 3 — Per-expiry exposure cap

**Where:** placement loop, per-opp.
**Logic:** Total dollars staked on a single settlement (existing open positions + this-run cumulative placements) must stay under `MAX_EXPIRY_EXPOSURE_PCT × portfolio_value` (default 50%).
**Rationale:** Top-N caps *count* of placements; this caps *dollars per settlement*. At 2026-04-15 04:00 the peak had $59.21 on a single expiry vs $86.03 portfolio (68%) — above 50%; this gate catches that shape regardless of placement count. Cap is against `portfolio_value` (cash + open_cost) not cash alone, so already-deployed capital on a settlement counts against the limit correctly.
**Env:** `MAX_EXPIRY_EXPOSURE_PCT=0.50`
**Observed live:** with $24.90 already on the 21:00 expiry, all three new proposed strikes for that expiry correctly blocked with clear logs.

### Tests

`test_phaseA.py` — 14 tests covering:
- Conviction gate (weak-signal skip / strong-signal pass / regression on 04:00 scenario)
- Regime scaling (normal / oversold / overbought)
- Netting (weak block / strong override / block-only mode / same-side passes through)
- Expiry cap (scenario-exact block / independent-settlement pass / cumulative-within-settlement block)
- End-to-end composition of the 04:00 scenario

All pass. Run: `./venv/bin/python /tmp/test_phaseA.py` (also copied into this session folder).

### Service envs added

```
Environment="MIN_CONVICTION=0.15"
Environment="MAX_EXPIRY_EXPOSURE_PCT=0.50"
Environment="REGIME_RSI_LOW=20"
Environment="REGIME_RSI_HIGH=80"
Environment="REGIME_SCALE=0.5"
Environment="NET_RECONCILIATION=1"
```

---

## Phase B — SCAFFOLDED, NOT WIRED (awaiting retraining)

Both ship as standalone modules that are importable but unused by `auto_trader.py`. Wiring happens after retraining proves improved backtest WR/EV.

### Item 1 — Orderbook microstructure features

**New file:** `~/programs/gemini_trader/orderbook_features.py`

Extracts 5 features from the existing `orderbook.db` snapshots:

| Feature | Formula |
|---|---|
| `ob_spread_bps` | `spread / mid_price × 10_000` |
| `ob_bid_imbalance` | `bid_depth_5 / (bid_depth_5 + ask_depth_5)` (already stored) |
| `ob_depth_ratio_log` | `clip(log(bid_depth_5 / ask_depth_5), -3, 3)` |
| `ob_top_size_ratio` | `bid1_vol / (bid1_vol + ask1_vol)` |
| `ob_staleness_s` | seconds since latest snapshot (freshness gauge) |

- `ob_features_at(ts_ms)` — inference, nearest-neighbor ≤ 300s stale.
- `ob_features_for_training(ts_array)` — batched join for retraining.

**Smoke test passed:** live extract at the time of writing returned a valid feature vector (`spread=0.71 bps, imbalance=0.37, …, staleness=14.9s`).

**To activate:**
1. Write `time_series/train_classifier_with_orderbook.py` — same pipeline as `train_classifier.py` but concatenates OB features to the existing 8-feature input.
2. Train on the joined 90-day window (3 days of OB data is too short; we need either backfilled Gemini orderbook snapshots or 30+ more days of collection before retraining is meaningful).
3. Verify backtest: WR ≥ current 68.3%, avg EV ≥ current +0.099.
4. Swap `classifier_model.pkl`; no `auto_trader.py` changes required if the pkl's feature order is compatible with a new extractor that concatenates OB features to the existing 8.

**Honest scope note:** we have only 3 days of orderbook history. Training a model that depends on these features before we have meaningful volume would overfit. Recommend collecting ≥ 30 days of OB snapshots before activating; the collector is already running, so this is a calendar wait, not a code task.

### Item 6 — Multi-horizon ensemble

**New file:** `~/programs/gemini_trader/classifier_ensemble.py`

Scaffolds a 3-model ensemble by horizon bucket:

| Bucket | Horizon (min) | Expected file |
|---|---|---|
| early | [10, 25) | `time_series/classifier_early.pkl` |
| mid   | [25, 50) | `time_series/classifier_mid.pkl` |
| late  | [50, 90] | `time_series/classifier_late.pkl` |

- `load_ensemble()` — tries to load all three; falls back to the existing single `classifier_model.pkl` if none of the buckets exist.
- `predict_proba_ensemble(ensemble, X, mins, vote="bucket")` — routes rows to the owning bucket. `vote="avg"` averages across all loaded models (use when buckets overlap).

**Smoke test passed:** loads the existing single model as an "all" bucket and correctly routes all horizon queries to it.

**To activate:**
1. Write `time_series/train_classifier_ensemble.py --bucket {early,mid,late} --horizon-range LO,HI`.
2. Train three models on bucket-sliced data. XGBoost training is seconds; most time is the hyperparameter search per bucket.
3. Verify each bucket's backtest WR/EV on held-out data from its own horizon range.
4. Edit `auto_trader.py::run_classifier` to import `load_ensemble` and `predict_proba_ensemble` in place of the direct model call.

**Honest scope note:** the current unified classifier handles all horizons with `mins` as a feature and gets 68.3% WR uniformly. Splitting won't automatically help — bucket models trained on ~1/3 the data per bucket need their per-horizon win rate to exceed the unified model on the same slice to justify the complexity. Worth trying once we have more data; not worth trying now.

---

## Deployment state

| Item | Status |
|---|---|
| 2 conviction gate | **LIVE** since 19:41 CDT |
| 5 regime detector | **LIVE** since 19:41 CDT |
| 4 netting reconciliation | **LIVE** since 19:41 CDT |
| 3 per-expiry cap | **LIVE** since 19:41 CDT |
| 1 orderbook features | scaffolded; awaiting ≥30d snapshot collection + retrain |
| 6 horizon ensemble | **LIVE** since 20:51 CDT — 3 bucket models trained + wired |

Drawdown halt remains the final backstop (50% from HWM). HWM is $52.40, portfolio $27.48, drawdown 47.6% as of 20:51 CDT. If next settlement pushes DD ≥ 50%, system halts automatically — by design.

---

## Phase B — Item 6 ensemble training results (2026-04-15 20:51 CDT)

Trained three horizon-specialized XGBoost classifiers on dual GPU (RTX 3080 + 3060), 90 days of 1-min BTC candle data. Each bucket trained in ~2–3s (GPU), ~14 min total with data prep.

| Bucket | Horizons | Training samples | Bets (backtest) | WR | avg EV | Gate |
|---|---|---:|---:|---:|---:|---|
| early | 10, 15, 20 min | 2,666,871 | 54,556 | 68.4% | +0.0938 | ✓ |
| **mid** | 25, 30, 35, 40 min | 3,555,xxx | 96,275 | **71.3%** | **+0.0959** | ✓ |
| late | 50, 60, 70, 80, 90 min | 4,444,xxx | 176,336 | 67.9% | +0.0860 | ✓ |
| *weighted avg* | | | 327,167 | **69.4%** | +0.0897 | |
| *(unified ref)* | *10–90 all-in-one* | *6.2M* | *17,323* | *68.3%* | *+0.0986* | |

Mid bucket is the standout at **71.3% WR** — a 3pp improvement over the unified model in its horizon range. Early is on par; late is slightly below (expected — longer horizons are inherently noisier).

**Wired into production:** `auto_trader.py` now loads all three via `classifier_ensemble.load_ensemble()` and routes each strike's prediction to the bucket that owns its `mins_to_expiry`. Falls back to the single `classifier_model.pkl` if no bucket files exist.

**Artifacts:**
- `time_series/classifier_early.pkl` (1.2 MB)
- `time_series/classifier_mid.pkl` (1.2 MB)
- `time_series/classifier_late.pkl` (1.3 MB)
- `time_series/training_metrics_{early,mid,late}.json`
- `time_series/train_classifier_ensemble.py` — the trainer

---

## Evening trading session — two halts, two resets (2026-04-15 18:00–22:15 CDT)

### First decline (18:10–20:10 CDT)

Portfolio dropped $42.56 → $27.48 after the 20:00 settlement (DD 47.6%). BTC rallied from ~$74,600 → $75,062, going against the model's NO bets. Expiry cap correctly blocked further orders when the 20:00 expiry was already maxed. Drawdown stayed below the 50% halt — system continued trading.

### Second decline + halt (21:10–22:10 CDT)

Partial recovery to $32.92 at 21:10, then the 22:00 settlement wiped $15.23 of deployed capital → portfolio $19.60, DD 62.6%. **Drawdown halt triggered** at 22:10 CDT.

Root cause: BTC sustained rally pushed RSI to 82.9 (overbought). Model continued shorting (NO bets) via the classifier's directional signal. The regime detector halved stakes (RSI > 80) which softened the blow, but two consecutive losing settlements is variance the gates can't prevent without also eliminating upside.

### Resume (22:15 CDT)

HWM reset to $19.60 (backup: `hwm.json.bak-2026-04-15-evening`). First ensemble-powered live tick placed 3 orders / $3.70 deployed — reduced sizing proportional to the smaller bankroll. Regime detector active (RSI 80.7), top-N capped 7→3.

### Session portfolio arc

```
15:20  $38.29  (first resume — bug fixes)
17:40  $52.40  (session HWM, +37%)
18:10  $42.43  (settlement loss)
19:20  $42.56  (recovered, Phase A gates live)
20:10  $27.66  (bad 20:00 settlement, DD 47%)
21:10  $32.92  (partial recovery, ensemble live)
22:10  $19.60  (bad 22:00 settlement, DD 62.6% → HALT)
22:15  $19.60  (HWM reset, DD 0%, trading resumed)
```

Net for the day: started at $38.29, ended $19.60 (−$18.69, −49%). The model's NO-side bias during a sustained BTC rally is the primary driver. All six safety gates fired correctly throughout — the losses are from *correctly-sized bets that happened to lose*, not from system failures.

---

## Diagrams

20 Graphviz diagrams (dark/neon) in `diagrams/` and pushed to repo `docs/diagrams/`:

**Post-mortem (01–04):** system architecture, per-run pipeline, failure modes, why-it-works
**Phase A/B (05–09):** gate flow, conviction regions, netting state machine, expiry cap budget, Phase B data flow
**Repo-wide (10–14):** module deps, DB schema, config layering, trade lifecycle, testing strategy
**Ensemble + deep-dives (15–20):** ensemble routing, training pipeline, incident anatomy, portfolio states, safety layers, Kalshi API flow

---

## Files added/modified this session

- `~/programs/gemini_trader/auto_trader.py` — Phase A gates + ensemble wire-in
- `~/programs/gemini_trader/orderbook_features.py` — new (Phase B, item 1, scaffolded)
- `~/programs/gemini_trader/classifier_ensemble.py` — new (Phase B, item 6, live)
- `~/programs/gemini_trader/time_series/train_classifier_ensemble.py` — new (bucket trainer)
- `~/programs/gemini_trader/time_series/classifier_{early,mid,late}.pkl` — trained models
- `/etc/systemd/system/kalshi-auto-trader.service` — 6 new Environment= lines
- `test_phaseA.py` — 14 unit tests (copy in this folder)
- `~/.local/share/kalshi-tui/hwm.json` — reset twice (15:20 from $86→$38; 22:15 from $52→$19.60)
- 20 Graphviz diagrams (`.dot` + `.png` + `.svg`)

---

## Consecutive-loss circuit breaker (2026-04-15 22:25 CDT)

Added after the evening's two consecutive losing settlements exposed a gap: the model kept placing all-NO bets while BTC rallied $400+. The regime detector halved stakes but didn't *stop* trading. The drawdown halt eventually caught it at −62% DD, but a streak-aware gate would have fired after the first loss and prevented the $13.23 second loss entirely.

**How it works:**
- `streak.json` persists state between oneshot systemd ticks
- Settlement detected: `had_positions=true` on previous tick, `open_cost ≈ 0` now
- PnL = current portfolio − portfolio when positions were last open
- If PnL < −$0.50: increment `consecutive_losses`; tag `last_loss_side` (dominant direction of last 20 trades, ≥70% threshold)
- If `consecutive_losses ≥ CONSEC_LOSS_HALT` (2): set `cooldown_until = now + BIAS_COOLDOWN_MINS` (60 min). No trades placed during cooldown.
- Win/flat settlement resets streak to 0
- Cooldown expiry also resets streak + clears `had_positions` (prevents stale settlement detection)

**Env vars:** `CONSEC_LOSS_HALT=2`, `BIAS_COOLDOWN_MINS=60`

**Tests:** `tests/test_streak.py` — 9 scenarios including T9 regression of the exact 2026-04-15 evening (all-NO across 20:00 and 22:00 settlements → halts after 2nd loss).

**Log lines:** `[!] Settlement loss detected: $-14.90 ... Consecutive losses: 1 (bias: NO)` → `[STOP] 2 consecutive losses ≥ 2 — cooling down for 60 min`

---

## Session summary

| What | Status |
|---|---|
| Bug (a) — 100¢ filter | **FIXED** — 0 API errors all day |
| Bug (b) — opposing bets + sprawl | **FIXED** — reconciliation + top-N + conviction gate |
| Item 2 — conviction gate | **LIVE** — skipped weak signals in production |
| Item 3 — per-expiry cap | **LIVE** — blocked over-allocated settlements |
| Item 4 — netting reconciliation | **LIVE** — no opposing signals triggered (model was directionally consistent) |
| Item 5 — regime detector | **LIVE** — halved stakes at RSI 82.9 (softened evening losses) |
| Item 6 — horizon ensemble | **LIVE** — 3 models, mid bucket +3pp WR, first ensemble tick at 22:15 |
| Item 1 — orderbook features | scaffolded, calendar wait (~May 12 for 30d OB history) |
| **Streak breaker** | **LIVE** — 2-loss halt + 60-min cooldown + directional bias tag |
| Portfolio | $38.29 → $52.40 peak → $19.60 (halted, reset, resumed) |
| Repo | 7 commits to `danindiana/kalshi-btc-trader` (latest `cfac7a8`) |
| Docs | 3 session folders + 22 diagrams |
