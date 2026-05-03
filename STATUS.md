# kalshi-tui — Status

**Created:** 2026-04-12  
**Goal:** Ratatui TUI for tracking Kalshi crypto prediction-market prices and personal predictions.

## v0.2 — current

### Source tree
```
src/
  main.rs    — terminal init, tokio runtime, key event loop, clap CLI args
  api.rs     — Kalshi REST fetch (KXBTCD / KXETHUSD) + Gemini spot BTC price; configurable interval
  app.rs     — App state, tab/scroll/input, prediction CRUD, edge calc, WIN/LOSS outcome marking
  db.rs      — SQLite persistence via rusqlite (bundled); ~/.local/share/kalshi-tui/predictions.db
  ui.rs      — Three-tab Ratatui TUI with spot price overlay and outcome column
```

### Features
- **Markets tab** — two-column BTC | ETH table; YES¢/NO¢ color-coded; spot price in title + ▲/▼ column markers
- **My Predictions tab** — full prediction log with entry price, live current price, edge P&L, and WIN/LOSS result
  - `j`/`k` — select row
  - `w` — mark selected WIN
  - `l` — mark selected LOSS
- **Edge Analysis tab** — horizontal bar chart of YES% per strike; SPOT marker on BTC panel
- **Input overlay** — `p` opens dialog; shows current spot as hint; Enter saves, Esc cancels
- **SQLite persistence** — predictions survive restarts; loaded on startup from `~/.local/share/kalshi-tui/predictions.db`
- **Gemini spot price** — polled every ⌊refresh/3⌋s (min 10s); shown in tab bar header and market tables
- **Configurable refresh** — `kalshi-tui --refresh 60` (default 30s)
- **Immediate fetch** — fires on startup without waiting for first interval tick
- **Installed** at `/usr/local/bin/kalshi-tui`

### Kalshi API fields used
| Field | Meaning |
|-------|---------|
| `floor_strike` | Strike price in USD |
| `yes_ask` | YES contract ask price in **cents** (÷100 → probability) |
| `no_ask` | NO contract ask price in cents |
| `volume_24h` | Rolling 24-hour volume |
| `open_interest` | Open interest |

### Key bindings
| Key | Action |
|-----|--------|
| Tab / BackTab | Switch tabs |
| j / k | Scroll / select row |
| p | New prediction (enter target price) |
| w | Mark selected prediction WIN |
| l | Mark selected prediction LOSS |
| r | Force immediate market refresh |
| q / Ctrl-C | Quit |

## One-time setup for Portfolio tab
```bash
# 1. Find your API Key ID in Kalshi portal → Settings → API Keys
# 2. Edit the config:
nano ~/.config/kalshi-tui/config.toml
# Set:  key_id = "your-key-id-here"
# 3. Run: kalshi-tui
#    → Tab to "Portfolio" — balance, positions, and resting orders appear live
```

## Auth scheme (for reference)
- Algorithm: RSA-PSS / SHA-256 / MGF1(SHA-256) / salt_length=DIGEST_LENGTH
- Message signed: `{timestamp_ms}{METHOD}{/trade-api/v2/path}`
- Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-SIGNATURE` (base64), `KALSHI-ACCESS-TIMESTAMP`
- Private key: `~/.config/kalshi-tui/private_key.pem` (mode 600)
- Config: `~/.config/kalshi-tui/config.toml`

## Automation (v0.3 — 2026-04-12)

BTC-only automated trading pipeline. Runs via systemd timer every 30 min, 9 AM – 4:30 PM CT.

### New files in `~/programs/gemini_trader/`

| File | Purpose |
|------|---------|
| `live_candles.py` | Fetch 1440 1m candles from Gemini production API (replaces stale sandbox CSV) |
| `kalshi_auth.py` | Python RSA-PSS signing for Kalshi v2 authenticated endpoints |
| `auto_trader.py` | Full orchestration: model → edge analysis → Kelly sizing → order placement |
| `trader_log.py` | Review trade history from `~/.local/share/kalshi-tui/auto_trader.db` |

`predict_json.py` updated to use `live_candles.py` (TUI model inference now uses live data too).

### Systemd units
```
/etc/systemd/system/kalshi-auto-trader.service
/etc/systemd/system/kalshi-auto-trader.timer    ← enabled, fires every 30 min 14:00–21:30 UTC
```

### Go-live checklist
1. **Fill in `key_id`** in `~/.config/kalshi-tui/config.toml` (from Kalshi portal → Settings → API Keys)
2. **Set `BANKROLL`** in the `.service` file to your actual capital
3. **Disable dry-run**: change `DRY_RUN=1` → `DRY_RUN=0` in the service file, then `sudo systemctl daemon-reload`
4. Monitor logs: `tail -f ~/.local/share/kalshi-tui/auto_trader.log`
5. Review trades: `python ~/programs/gemini_trader/trader_log.py`

### Safety rails (configurable via service Environment= lines)
| Variable | Default | Meaning |
|----------|---------|---------|
| `BANKROLL` | $2,000 | Total capital for Kelly sizing |
| `DAILY_LOSS_LIMIT` | $500 | Stop trading for the day if deployed this much |
| `MAX_STAKE_DOLLARS` | $200 | Cap per single trade |
| `MAX_RUN_DOLLARS` | $1,000 | Cap total deployed in one run |
| `MIN_EDGE` | 0.05 | Minimum probability edge (5%) required to trade |
| `DRY_RUN` | 1 | Set to 0 to enable live order placement |

### Logs & DB
```
~/.local/share/kalshi-tui/auto_trader.log   — stdout/stderr from every run
~/.local/share/kalshi-tui/auto_trader.db    — SQLite: runs + trades tables
```

## Next steps (v0.4 ideas)
- [ ] Outcome tracking: auto-check settled market results and update trade P&L in DB
- [ ] Daily P&L summary email/notification
- [ ] Sparkline of YES% history per strike (ring buffer in TUI)
- [ ] `kalshi-tui --series KXETHUSD` to filter to a single series
- [ ] Retrain classifier on Coinbase/Gemini live candles weekly (LSTM fallback removed 2026-04-14 — classifier is the sole trading model; see `~/Documents/claude_creations/2026-04-14_111934_lstm-fallback-disable-and-retrain/SESSION_LOG.md`)

## Related code
- `~/programs/gemini_trader/kalshi_client.py` — Kalshi public market fetch
- `~/programs/gemini_trader/kalshi_simulation.py` — legacy LSTM + edge analysis prototype (pre-classifier design; reference only)
- `~/programs/gemini_trader/risk_manager.py` — Kelly criterion recommendation engine

## Build & run (TUI)
```bash
cargo build --release
kalshi-tui                 # default 30s refresh
kalshi-tui --refresh 60    # 60s refresh
```

## Run automation manually
```bash
# Dry run (no orders placed)
DRY_RUN=1 python ~/programs/gemini_trader/auto_trader.py

# Live (requires key_id set and DRY_RUN=0)
DRY_RUN=0 python ~/programs/gemini_trader/auto_trader.py

# Review logs
python ~/programs/gemini_trader/trader_log.py
python ~/programs/gemini_trader/trader_log.py --all
```
