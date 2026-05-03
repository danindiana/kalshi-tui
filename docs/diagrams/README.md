# Diagrams

Graphviz diagrams for kalshi-tui. All rendered on a near-black background
(`#0a0e14`) with a neon palette (green `#39ff14`, cyan `#00ffff`, magenta
`#ff00ff`, amber `#ffaa00`) and JetBrains Mono throughout.

| # | File | What it shows |
|---|---|---|
| 01 | [`01_module_architecture.dot`](01_module_architecture.dot) | Rust module wiring — every `.rs` under `src/` and how they connect to external sources. |
| 02 | [`02_data_flow.dot`](02_data_flow.dot) | Each data stream from source → transform → the tab it lands on. Solid = push, dashed = pull. |
| 03 | [`03_status_tab_layout.dot`](03_status_tab_layout.dot) | Physical layout of the five rows in the Status tab, including the new Performance-by-Side and Size × Side panels. |
| 04 | [`04_trade_outcomes_pipeline.dot`](04_trade_outcomes_pipeline.dot) | `trade_outcomes` SQL view — from `runs` / `trades` / `settlements_raw` through the CASE joins to the two perf panels. |
| 05 | [`05_event_loop.dot`](05_event_loop.dot) | `tokio::select!` branches per tick: input, timers, command channel, fetch results. |

## Rebuild

```sh
cd docs/diagrams
for f in *.dot; do dot -Tpng "$f" -o "${f%.dot}.png"; done
```

Requires `graphviz` (`apt install graphviz` on Debian/Ubuntu).

## PNGs

PNGs are checked in alongside the sources so the repo renders on GitHub
without a build step. If you edit a `.dot`, rebuild and commit both.
