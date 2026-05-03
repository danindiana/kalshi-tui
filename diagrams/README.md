# Diagrams — Phase A/B update + repo-wide architecture

Dark background, neon palette (magenta = new/Phase B, green = operational, orange = external/state, cyan = data/portfolio, red = failures). PNG + SVG + `.dot` source for every figure; PNGs render cleanly on GitHub.

Diagrams 01–04 from the overnight post-mortem session are in `../../2026-04-15_150954_1776283794_kalshi-overnight-postmortem/diagrams/`. This folder picks up at 05.

## Phase A/B update

| # | File | What it shows |
|---:|---|---|
| 05 | `05_phaseA_gate_flow.png` | Full ordered flow of the gates inside `find_opportunities` and the placement loop. Magenta nodes = new this session (conviction, regime, netting, expiry cap). |
| 06 | `06_conviction_regions.png` | Partition of `clf_prob ∈ [0,1]` into STRONG NO / MOD NO / COIN-FLIP / MOD YES / STRONG YES regions with the 0.15 and 0.30 thresholds marked; annotates the 04:00 regression. |
| 07 | `07_netting_state_machine.png` | State machine for `(held_side × new_side × conviction) → action`. Three outcomes: `[PASS]`, `[NET]`, `[SKIP-OPP]`. Shows why `NET_RECONCILIATION=0` collapses to the earlier block-only behavior. |
| 08 | `08_expiry_cap_budget.png` | Per-settlement dollar budget — how `expiry_cost` is seeded from `/portfolio/positions` and accumulated per placement. Annotated with live tick data and the 04:00 failure shape. |
| 09 | `09_phaseB_data_flow.png` | Phase B scaffolds — `orderbook_features.py` and `classifier_ensemble.py` — their inputs, planned training path, and the calendar-wait for sufficient OB history. |

## Repo-wide architecture

| # | File | What it shows |
|---:|---|---|
| 10 | `10_module_dependencies.png` | Python import graph across all modules in `~/programs/gemini_trader/`. Grouped by role: entrypoints, runtime support, Phase B scaffolds, training. |
| 11 | `11_db_schema.png` | ER-style layout of the three SQLite databases (`auto_trader.db`, `orderbook.db`, `predictions.db`) + `hwm.json`, with FKs and which module writes which table. |
| 12 | `12_config_layering.png` | How a single env var (`MIN_CONVICTION`) resolves through systemd → process env → code default → runtime → decision site. Side panel shows complete env inventory with new vars highlighted. |
| 13 | `13_trade_lifecycle.png` | Single trade timeline from T−90 (market listed) through decision (T−10), order, settlement, P&L realization, and feedback into the next run's sizing. |
| 14 | `14_testing_strategy.png` | Five validation layers: unit → DRY_RUN integration → backtest gate → production canary → live monitoring. Shows which layer each change type must clear. |
