#!/usr/bin/env python3
"""
auto_trader.py — Automated BTC prediction-market trading on Kalshi.

Pipeline (runs every 10 minutes via systemd timer):
  1. Fetch live 1m BTC candles from Gemini
  2. Run GBT classifier → per-strike win probabilities
  3. Query Kalshi KXBTCD markets → strikes near spot price
  4. Apply Kelly criterion → find trades with real edge (≥ MIN_EDGE)
  5. Place limit orders via Kalshi authenticated API
  6. Log everything to SQLite

Ratchet sizing — all limits scale with live balance:
  - Daily deployment cap:  DAILY_LOSS_PCT  (50%) × bankroll, ≤ DAILY_LOSS_LIMIT  ($200 abs)
  - Per-run budget:        MAX_RUN_PCT     (40%) × bankroll, ≤ MAX_RUN_DOLLARS   ($100 abs)
  - Per-trade stake:       MAX_STAKE_PCT   (12%) × bankroll, ≤ MAX_STAKE_DOLLARS ($25 abs)
  - Bankroll base:         live balance,   ≤ MAX_BANKROLL ($500 abs)

Stop-loss:
  - Halt if balance < MIN_BALANCE ($5) — prevents wipe on bad days
  - Halt if drawdown from HWM ≥ DRAWDOWN_HALT (30%) on portfolio value (cash + open positions)
  - Halt if balance API unavailable (can't determine safe limits)

Contract gates:
  - Skip contract if price < MIN_CONTRACT_CENTS (10¢)
  - Skip contract if model_prob / market_price > MAX_PROB_RATIO (3×)
  - Only trade within MAX_SETTLEMENT_WINDOW_MINUTES (90 min) of settlement; model edge is the quality gate
  - DEFENSIVE TREND GATE: Uses MACD to suppress NO-bets during sustained uptrends (and YES-bets in downtrends)

LSTM retired 2026-04-14 — see time_series/_retired_2026-04-14_lstm_full/README.md.
"""

import io
import json
import math
import os
import pickle
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BANKROLL          = float(os.environ.get("BANKROLL",          "2000"))   # fallback if live balance unavailable
MAX_BANKROLL      = float(os.environ.get("MAX_BANKROLL",      "500"))    # hard cap on balance used as sizing base

# Proportional limits — scale with live bankroll (ratchet up as account grows)
DAILY_LOSS_PCT    = float(os.environ.get("DAILY_LOSS_PCT",    "0.50"))   # 50% of bankroll per day
MAX_STAKE_PCT     = float(os.environ.get("MAX_STAKE_PCT",     "0.12"))   # 12% of bankroll per trade
MAX_RUN_PCT       = float(os.environ.get("MAX_RUN_PCT",       "0.40"))   # 40% of bankroll per run

# Absolute ceilings — proportional limits can't exceed these (safety at large scale)
DAILY_LOSS_LIMIT  = float(os.environ.get("DAILY_LOSS_LIMIT",  "200"))    # absolute daily ceiling
MAX_STAKE_DOLLARS = float(os.environ.get("MAX_STAKE_DOLLARS", "25"))     # absolute per-trade ceiling
MAX_RUN_DOLLARS   = float(os.environ.get("MAX_RUN_DOLLARS",   "100"))    # absolute per-run ceiling

# Stop-loss
MIN_BALANCE       = float(os.environ.get("MIN_BALANCE",       "5.00"))   # halt if balance < this (wipe prevention)
DRAWDOWN_HALT     = float(os.environ.get("DRAWDOWN_HALT",     "0.30"))   # halt if 30% drawdown from HWM

MIN_EDGE          = float(os.environ.get("MIN_EDGE",          "0.05"))   # minimum probability edge to trade
FRACTIONAL_KELLY  = float(os.environ.get("FRACTIONAL_KELLY",  "0.25"))   # Kelly fraction (1.0 = full Kelly)
CUTOFF_MINUTES    = 10                                                   # stop trading N min before settlement

# ── Safety gates ──────────────────────────────────────────────────────────────
MIN_CONTRACT_CENTS= float(os.environ.get("MIN_CONTRACT_CENTS", "10"))    # reject contracts priced below N cents
MAX_CONTRACT_CENTS= float(os.environ.get("MAX_CONTRACT_CENTS", "95"))    # reject contracts priced at/above N cents (Kalshi rejects 100¢)
MAX_PROB_RATIO    = float(os.environ.get("MAX_PROB_RATIO",     "3.0"))   # reject if model_prob / market_price > N×
MAX_SETTLEMENT_WINDOW_MINUTES = float(os.environ.get("MAX_SETTLEMENT_WINDOW_MINUTES", "90"))  # ignore daily/weekly markets beyond this
MACD_TREND_THRESHOLD = float(os.environ.get("MACD_TREND_THRESHOLD", "50.0")) # threshold for sustained trend activation
YES_IMBALANCE_THRESHOLD = float(os.environ.get("YES_IMBALANCE_THRESHOLD", "0.65")) # threshold for orderbook imbalance gate
NO_PROXIMITY_BUFFER     = float(os.environ.get("NO_PROXIMITY_BUFFER",     "200"))  # min $ gap required between spot and a NO-bet strike
METRICS_PATH      = Path("/home/jeb/programs/gemini_trader/time_series/training_metrics.json")
DRY_RUN           = os.environ.get("DRY_RUN", "0") != "0"                # set DRY_RUN=1 to simulate

DB_PATH           = Path.home() / ".local" / "share" / "kalshi-tui" / "auto_trader.db"
HWM_PATH          = Path.home() / ".local" / "share" / "kalshi-tui" / "hwm.json"
VENV_PYTHON       = "/home/jeb/programs/gemini_trader/venv/bin/python"
GEMINI_TRADER_DIR = "/home/jeb/programs/gemini_trader"

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

import numpy as np
import pandas as pd

sys.path.insert(0, GEMINI_TRADER_DIR)
import live_candles
import risk_manager
from kalshi_auth import KalshiSigner

# Pin workspace orderbook_features by absolute path so sys.path ordering never
# resolves to legacy gemini_trader/orderbook_features.py (which lacks
# fetch_orderbook/imbalance_allows_bet).  Must register in sys.modules before
# exec_module so @dataclass can resolve cls.__module__ during class creation.
import importlib.util as _ilu
_ob_spec = _ilu.spec_from_file_location(
    "workspace_orderbook_features",
    Path(__file__).parent / "orderbook_features.py",
)
_ob_mod = _ilu.module_from_spec(_ob_spec)
sys.modules["workspace_orderbook_features"] = _ob_mod
_ob_spec.loader.exec_module(_ob_mod)
del _ilu, _ob_spec

# ── Database ──────────────────────────────────────────────────────────────────

def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at            TEXT NOT NULL,
            current_price     REAL,
            model_prediction  REAL,
            mae_current       REAL,
            mins_to_expiry    REAL,
            rsi               REAL,
            volatility        REAL,
            momentum          REAL,
            opportunities     INTEGER,
            trades_placed     INTEGER,
            total_staked      REAL,
            dry_run           INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER REFERENCES runs(id),
            placed_at       TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            strike          REAL NOT NULL,
            side            TEXT NOT NULL,
            count           INTEGER NOT NULL,
            price_cents     INTEGER NOT NULL,
            cost_dollars    REAL NOT NULL,
            model_prob      REAL,
            market_price    REAL,
            edge            REAL,
            stake_pct       REAL,
            order_id        TEXT,
            status          TEXT NOT NULL,
            error           TEXT
        );
    """)
    con.commit()
    return con


def daily_loss(con: sqlite3.Connection) -> float:
    """Return total cost of contracts placed today (all placed capital treated as at-risk).
    Acts as a daily deployment cap — resets each calendar day.
    Note: tracks cost of placed orders, not realised P&L."""
    today = datetime.now().strftime("%Y-%m-%d")
    row = con.execute(
        "SELECT COALESCE(SUM(cost_dollars), 0) FROM trades WHERE placed_at LIKE ? AND status='placed'",
        (f"{today}%",),
    ).fetchone()
    return row[0] if row else 0.0


# ── Settlement discovery ──────────────────────────────────────────────────────

def get_next_settlement(signer) -> tuple[datetime | None, float, list[dict]]:
    """
    Query live Kalshi markets to discover the next upcoming settlement.
    BTC trades 24/7 and Kalshi has multiple daily windows (e.g. 8 PM, 4 PM CT).

    Returns: (settlement_utc, mins_to_settlement, markets_for_that_window)
    Returns (None, 0, []) if no tradeable window found.
    """
    url = f"{KALSHI_BASE}/markets?status=open&series_ticker=KXBTCD&limit=200"
    r = signer.get(url)
    r.raise_for_status()
    raw_markets = r.json().get("markets", [])

    # Group markets by close_time, find the nearest future one
    from collections import defaultdict
    by_close: dict[datetime, list] = defaultdict(list)
    now_utc = datetime.now(timezone.utc)

    for m in raw_markets:
        ct_str = m.get("close_time")
        if not ct_str:
            continue
        try:
            ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        mins_away = (ct - now_utc).total_seconds() / 60
        if mins_away > CUTOFF_MINUTES and mins_away <= MAX_SETTLEMENT_WINDOW_MINUTES:  # hourly window only; skip daily/weekly
            try:
                by_close[ct].append({
                    "ticker":    m["ticker"],
                    "strike":    float(m.get("floor_strike", 0)),
                    "yes_price": float(m.get("yes_ask_dollars", 0) or 0),
                    "no_price":  float(m.get("no_ask_dollars", 0) or 0),
                    "close_time": ct,
                })
            except (ValueError, TypeError, KeyError):
                continue

    if not by_close:
        return None, 0.0, []

    # Pick the NEAREST upcoming settlement
    next_close = min(by_close.keys())
    mins_to    = (next_close - now_utc).total_seconds() / 60
    markets    = sorted(by_close[next_close], key=lambda x: x["strike"])
    return next_close, mins_to, markets


# ── Time guards ───────────────────────────────────────────────────────────────

def check_trading_window(signer) -> tuple[bool, str, datetime | None, float, list]:
    """
    Return (can_trade, reason, settlement_utc, mins_to_settlement, markets).
    Discovers settlement windows dynamically from live Kalshi API — works for
    any settlement time (8 PM, 4 PM, overnight, etc.).
    """
    settlement_utc, mins, markets = get_next_settlement(signer)

    if settlement_utc is None:
        return False, "No open markets with sufficient time remaining", None, 0, []

    local_str = settlement_utc.astimezone().strftime("%H:%M %Z")
    return True, f"{mins:.0f} min to {local_str} settlement ({len(markets)} strikes)", settlement_utc, mins, markets


# ── Model inference ───────────────────────────────────────────────────────────

CLF_PATH = Path(GEMINI_TRADER_DIR) / "time_series" / "classifier_model.pkl"


def run_classifier(signer, settlement_utc: datetime, mins_to_settlement: float, markets: list) -> dict | None:
    """
    Run the directional GBT classifier to produce per-strike win probabilities.
    Returns a dict with 'opportunities' already computed, or None if unavailable.

    The classifier directly outputs P(price > strike) for each market, eliminating
    the Gaussian approximation used by the LSTM path.
    """
    try:
        # GPU is treated as a contended resource. If no GPU has ≥ 1500 MiB free
        # (e.g. Ollama or a training job is resident), skip this run entirely.
        # CPU fallback is intentionally disabled — the system should be in
        # continuous GPU use, and stale CPU-only inference is not a safe substitute.
        from classifier_device import pick_device
        device_used = pick_device()

        if device_used == "cpu":
            print("[SKIP] GPU unavailable (< 1500 MiB free) — standing down (GPU contention model)")
            return None

        # CalibratedXGB was pickled when train_classifier.py ran as __main__, so
        # pickle stores it as "__main__.CalibratedXGB". Use a custom Unpickler to
        # redirect that lookup to the actual module without polluting __main__.
        import pickle as _pkl
        sys.path.insert(0, str(Path(GEMINI_TRADER_DIR) / "time_series"))

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

        class _Unpickler(_pkl.Unpickler):
            def find_class(self, module, name):
                if name == "CalibratedXGB":
                    from train_classifier import CalibratedXGB as _C
                    return _C
                return super().find_class(module, name)

        with open(CLF_PATH, "rb") as f:
            clf = _Unpickler(f).load()

        # Detect if model was trained with OB features (13 features) or plain (8).
        n_expected = getattr(getattr(clf, 'clf', None), 'n_features_in_', 8)
        n_ob = n_expected - 8  # 5 for OB model, 0 for plain

        print(f"[OK]   Classifier device: {device_used}  features={n_expected}")
    except FileNotFoundError:
        return None

    df = live_candles.fetch_live_candles()
    w  = 5
    df["price_ma"]         = df["price"].rolling(w).mean()
    df["price_rel_ma"]     = df["price"] - df["price_ma"]
    df["price_volatility"] = df["price"].rolling(w).std()
    df["price_momentum"]   = df["price"].diff(w)
    df["ma_slope"]         = df["price_ma"].diff()
    
    # RSI — 14-period Wilder's EWM, matching train_classifier*.py (updated 2026-05-02).
    delta = df["price"].diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    
    # MACD Calculation for the Defensive Trend Gate
    df["macd_short"]  = df["price"].ewm(span=12, adjust=False).mean()
    df["macd_long"]   = df["price"].ewm(span=26, adjust=False).mean()
    df["macd_line"]   = df["macd_short"] - df["macd_long"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]
    
    df = df.dropna()

    current_price = float(df["price"].iloc[-1])
    mins          = max(1.0, mins_to_settlement)

    # Fetch Gemini L2 OB features if the deployed model was trained with them.
    ob_feats = None
    if n_ob > 0:
        import time as _time
        from orderbook_features import ob_features_at as _ob_at
        ob_feats = _ob_at(int(_time.time() * 1000))
        if ob_feats is None:
            raise RuntimeError("OB features unavailable (orderbook.db stale/missing)")
        print(f"[OK]   OB features: staleness={ob_feats[4]:.0f}s  imbalance={ob_feats[1]:.3f}  spread={ob_feats[0]:.1f}bps")

    # Capture current MACD values for the gate
    current_macd = float(df["macd_line"].iloc[-1])
    current_signal = float(df["macd_signal"].iloc[-1])
    current_hist = float(df["macd_hist"].iloc[-1])

    # Build base feature vector from latest bar
    last = df.iloc[-1]
    base_feats = np.array([
        last["price_rel_ma"],
        last["price_volatility"],
        last["price_momentum"],
        last["rsi"],
        last["ma_slope"],
        last["amount"],
    ], dtype="float32")

    price_lo = current_price * 0.985
    price_hi = current_price * 1.015

    opps = []
    for mk in markets:
        strike = mk["strike"]
        if not (price_lo <= strike <= price_hi):
            continue

        dist = (strike - current_price) / current_price
        if ob_feats is not None:
            row = np.concatenate([base_feats, [dist, mins], ob_feats]).reshape(1, -1)
        else:
            row = np.concatenate([base_feats, [dist, mins]]).reshape(1, -1)
        clf_prob = float(clf.predict_proba(row)[0][1])   # P(price > strike)

        yes_price = mk["yes_price"]
        if clf_prob > 0.5:
            side         = "YES"
            model_prob   = clf_prob
            market_price = yes_price
        else:
            side         = "NO"
            model_prob   = 1.0 - clf_prob
            market_price = 1.0 - yes_price

        edge = model_prob - market_price
        if edge < MIN_EDGE:
            continue

        # Gates 2 & 3
        contract_cents = round(market_price * 100)
        if contract_cents < MIN_CONTRACT_CENTS:
            print(f"  [SKIP] Strike ${strike:,.0f}: {contract_cents}¢ < {MIN_CONTRACT_CENTS:.0f}¢ min")
            continue
        if contract_cents >= MAX_CONTRACT_CENTS:
            print(f"  [SKIP] Strike ${strike:,.0f}: {contract_cents}¢ >= {MAX_CONTRACT_CENTS:.0f}¢ max (Kalshi rejects at-limit contracts)")
            continue
        if market_price > 0 and model_prob / market_price > MAX_PROB_RATIO:
            ratio = model_prob / market_price
            print(f"  [SKIP] Strike ${strike:,.0f}: ratio {ratio:.1f}× > {MAX_PROB_RATIO:.1f}×")
            continue

        # --- NEW DEFENSIVE TREND GATE ---
        # Detect sustained regimes to suppress fading (NO-side bias) against strong momentum
        is_sustained_uptrend = (current_macd > current_signal) and (current_hist > MACD_TREND_THRESHOLD)
        is_sustained_downtrend = (current_macd < current_signal) and (current_hist < -MACD_TREND_THRESHOLD)

        if is_sustained_uptrend and side == "NO":
            print(f"  [SKIP] Strike ${strike:,.0f}: DEFENSIVE GATE TRIGGERED. Suppressing 'NO' bet during sustained uptrend (MACD Hist: {current_hist:.2f})")
            continue
            
        if is_sustained_downtrend and side == "YES":
            print(f"  [SKIP] Strike ${strike:,.0f}: DEFENSIVE GATE TRIGGERED. Suppressing 'YES' bet during sustained downtrend (MACD Hist: {current_hist:.2f})")
            continue
        # --------------------------------

        # NO proximity guard — reject if spot is within NO_PROXIMITY_BUFFER of the
        # NO-bet strike.  Placing a NO with BTC already at or above the strike is
        # an immediate losing position; even a small adverse move causes full loss.
        if side == "NO" and current_price >= strike - NO_PROXIMITY_BUFFER:
            print(
                f"  [SKIP] Strike ${strike:,.0f}: NO-proximity gate "
                f"(spot ${current_price:,.0f}, need ≥${NO_PROXIMITY_BUFFER:.0f} buffer)"
            )
            continue

        # --- ORDERBOOK CONVICTION GATE ---
        try:
            ob_snap = _ob_mod.fetch_orderbook(signer, mk["ticker"])
            ob_allowed, ob_reason = _ob_mod.imbalance_allows_bet(ob_snap, side, YES_IMBALANCE_THRESHOLD)
            if not ob_allowed:
                print(f"  [SKIP-OB] Strike ${strike:,.0f}: {ob_reason}")
                continue
        except Exception as e:
            print(f"  [WARN] Orderbook gate error for {mk['ticker']}: {e}")

        from risk_manager import kelly_criterion
        stake_pct = kelly_criterion(model_prob, market_price, fractional_kelly=FRACTIONAL_KELLY)

        opps.append({
            **mk,
            "side":       side,
            "model_prob": round(model_prob, 4),
            "edge":       round(edge, 4),
            "stake_pct":  stake_pct,
        })

    opps.sort(key=lambda x: x["edge"], reverse=True)

    return {
        "current_price":    current_price,
        "model_prediction": current_price,   # classifier has no single point prediction
        "mae_current":      0.0,             # N/A for classifier
        "mins_to_expiry":   mins,
        "model_type":       "classifier",
        "rsi":              float(df["rsi"].iloc[-1]),
        "volatility":       float(df["price_volatility"].iloc[-1]),
        "momentum":         float(df["price_momentum"].iloc[-1]),
        "macd_hist":        current_hist,    # Logged for potential future debugging
        "opportunities":    opps,
    }


# ── Kalshi market fetch (markets already fetched by check_trading_window) ─────
# Markets are returned by get_next_settlement() — no redundant fetch needed.


# ── Order placement ───────────────────────────────────────────────────────────

def place_order(
    signer: KalshiSigner,
    ticker: str,
    side: str,
    count: int,
    price_cents: int,
) -> dict:
    """Place a limit order on Kalshi. Returns response dict."""
    url  = f"{KALSHI_BASE}/portfolio/orders"
    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          "buy",
        "side":            side.lower(),
        "count":           count,
    }
    if side.lower() == "yes":
        body["yes_price"] = price_cents
    else:
        body["no_price"] = price_cents

    r = signer.post(url, json=body)
    r.raise_for_status()
    return r.json()


# ── Main trading logic ────────────────────────────────────────────────────────

def get_live_balance(signer) -> float:
    """Fetch live Kalshi cash balance in dollars. Returns -1.0 if API unavailable."""
    try:
        r = signer.get(f"{KALSHI_BASE}/portfolio/balance")
        r.raise_for_status()
        cents = r.json().get("balance", 0)
        return cents / 100.0
    except Exception:
        return -1.0


def get_open_position_cost(signer) -> float:
    """
    Fetch total cost basis of currently open Kalshi positions in dollars.
    Uses event_positions[].total_cost_dollars — the aggregate amount deployed
    per settlement event that hasn't yet resolved.
    Returns 0.0 on any error (safe: we just won't add it to portfolio value).
    """
    try:
        r = signer.get(f"{KALSHI_BASE}/portfolio/positions")
        r.raise_for_status()
        events = r.json().get("event_positions", [])
        return sum(float(e.get("total_cost_dollars", 0)) for e in events)
    except Exception:
        return 0.0


def update_hwm(portfolio_value: float) -> tuple[float, float]:
    """
    Track peak (high-water mark) portfolio value across runs.
    Portfolio value = cash balance + cost basis of open positions, so deployed
    capital doesn't inflate the drawdown percentage while positions are live.
    Ratschets up on new highs; never drops.
    Returns (hwm_dollars, drawdown_pct).
    """
    try:
        with open(HWM_PATH) as f:
            data = json.load(f)
        hwm = float(data.get("hwm", portfolio_value))
    except (FileNotFoundError, json.JSONDecodeError):
        hwm = portfolio_value

    hwm = max(hwm, portfolio_value)       # ratchet up on new high
    drawdown_pct = (hwm - portfolio_value) / hwm if hwm > 0 else 0.0

    HWM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HWM_PATH, "w") as f:
        json.dump({
            "hwm":             round(hwm, 4),
            "portfolio_value": round(portfolio_value, 4),
            "drawdown_pct":    round(drawdown_pct, 4),
            "updated_at":      datetime.now().isoformat(),
        }, f, indent=2)

    return hwm, drawdown_pct


def run():
    print(f"\n{'='*60}")
    print(f"  Kalshi Auto-Trader  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print(f"{'='*60}")

    # ── Auth (needed before market discovery) ───────────────────────────────
    try:
        signer = KalshiSigner.from_config()
        print(f"[OK]   Auth loaded: key_id={signer.key_id[:8]}…")
    except Exception as e:
        print(f"[FAIL] Auth error: {e}")
        print("        Set key_id in ~/.config/kalshi-tui/config.toml")
        return

    # ── Live balance, HWM, and dynamic limits ───────────────────────────────
    live_balance = get_live_balance(signer)
    if live_balance < 0:
        # Can't determine safe sizing limits without live balance — skip this run.
        print("[STOP] Balance API unavailable — cannot compute safe limits, skipping run")
        return
    else:
        bankroll = min(live_balance, MAX_BANKROLL)

        if live_balance < MIN_BALANCE:
            print(f"[STOP] Balance ${live_balance:.2f} below floor ${MIN_BALANCE:.2f} — halting to prevent wipe")
            return

        # Portfolio value = cash + cost basis of open positions.
        # Deployed capital is not a loss until it settles; using portfolio_value
        # for the drawdown calc prevents false halts while positions are live.
        open_cost = get_open_position_cost(signer)
        portfolio_value = live_balance + open_cost

        hwm, drawdown_pct = update_hwm(portfolio_value)
        deployed_note = f"  +${open_cost:.2f} deployed" if open_cost > 0.01 else ""
        print(
            f"[OK]   Balance: ${live_balance:.2f}{deployed_note}  "
            f"Portfolio: ${portfolio_value:.2f}  "
            f"HWM: ${hwm:.2f}  "
            f"Drawdown: {drawdown_pct:.1%}"
            + (f"  [sizing on ${bankroll:.0f} cap]" if bankroll < live_balance else "")
        )

        if drawdown_pct >= DRAWDOWN_HALT:
            print(
                f"[STOP] Drawdown {drawdown_pct:.1%} ≥ {DRAWDOWN_HALT:.0%} halt threshold "
                f"(portfolio ${portfolio_value:.2f} vs HWM ${hwm:.2f}) — standing down"
            )
            return

    # Dynamic limits — proportional to bankroll, capped by absolute ceilings
    max_stake = min(bankroll * MAX_STAKE_PCT, MAX_STAKE_DOLLARS)
    max_run   = min(bankroll * MAX_RUN_PCT,   MAX_RUN_DOLLARS)
    daily_cap = min(bankroll * DAILY_LOSS_PCT, DAILY_LOSS_LIMIT)
    print(
        f"[OK]   Limits — daily: ${daily_cap:.2f}  "
        f"run: ${max_run:.2f}  "
        f"stake: ${max_stake:.2f}"
    )

    # ── Discover next settlement from live API ───────────────────────────────
    # BTC trades 24/7 — Kalshi may have evening, overnight, and afternoon windows.
    # We find whichever real settlement is nearest and still tradeable (>15 min away).
    can_trade, reason, settlement_utc, mins_to_settlement, markets = check_trading_window(signer)
    if not can_trade:
        print(f"[SKIP] {reason}")
        return

    local_settle = settlement_utc.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    print(f"[OK]   Next settlement: {local_settle} ({mins_to_settlement:.0f} min away)")
    print(f"[OK]   {len(markets)} strikes in this window")

    # ── Database ────────────────────────────────────────────────────────────
    con = open_db()
    spent_today = daily_loss(con)
    if spent_today >= daily_cap:
        print(f"[STOP] Daily cap reached: ${spent_today:.2f} ≥ ${daily_cap:.2f} ({DAILY_LOSS_PCT:.0%} of ${bankroll:.2f})")
        con.close()
        return
    remaining_daily = daily_cap - spent_today
    print(f"[OK]   Daily budget: ${daily_cap:.2f} — ${remaining_daily:.2f} remaining")

    # ── Model inference ─────────────────────────────────────────────────────
    # Classifier-only. LSTM stack was retired 2026-04-14 (see
    # time_series/_retired_2026-04-14_lstm_full/README.md). A silent skip on
    # classifier unavailability is strictly safer than any LSTM-based fallback
    # — past evidence showed the LSTM never produced real edge regardless of
    # which trainer produced it.
    if not CLF_PATH.exists():
        print(f"[SKIP] Classifier missing at {CLF_PATH} — no trades this run")
        con.close()
        return

    try:
        print("[...] Running classifier inference…")
        clf_result = run_classifier(signer, settlement_utc, mins_to_settlement, markets)
    except Exception as e:
        print(f"[SKIP] Classifier inference failed ({e}) — no trades this run")
        con.close()
        return

    if clf_result is None:
        print("[SKIP] Classifier returned no result — no trades this run")
        con.close()
        return

    m             = clf_result
    opportunities = clf_result["opportunities"]
    macd_hist_str = f"macd_hist={m.get('macd_hist', 0):+.1f}"
    
    print(f"[OK]   Current price: ${m['current_price']:,.2f}  (classifier model)")
    print(f"[OK]   RSI={m['rsi']:.1f}  vol=${m['volatility']:,.0f}  momentum=${m['momentum']:+,.0f}  {macd_hist_str}")
    print(f"[OK]   {len(opportunities)} opportunities with edge ≥ {MIN_EDGE*100:.0f}%")

    # ── Log this run ─────────────────────────────────────────────────────────
    cur = con.execute(
        """INSERT INTO runs
           (run_at, current_price, model_prediction, mae_current, mins_to_expiry,
            rsi, volatility, momentum, opportunities, trades_placed, total_staked, dry_run)
           VALUES (?,?,?,?,?,?,?,?,?,0,0,?)""",
        (
            datetime.now().isoformat(),
            m["current_price"], m["model_prediction"], m["mae_current"], m["mins_to_expiry"],
            m["rsi"], m["volatility"], m["momentum"],
            len(opportunities),
            1 if DRY_RUN else 0,
        ),
    )
    run_id = cur.lastrowid
    con.commit()

    if not opportunities:
        print("[OK]   No tradeable edges found. Done.")
        con.close()
        return

    # ── Place orders ─────────────────────────────────────────────────────────
    run_budget = min(max_run, remaining_daily)
    trades_placed = 0
    total_staked  = 0.0

    for opp in opportunities:
        if total_staked >= run_budget:
            print(f"[STOP] Run budget exhausted (${total_staked:.2f})")
            break

        # Dollar stake from Kelly (sized against live bankroll)
        raw_stake   = bankroll * opp["stake_pct"]
        stake       = min(raw_stake, max_stake, run_budget - total_staked)
        stake       = max(stake, 1.0)  # minimum $1

        # Number of contracts (Kalshi prices each contract at price_cents/100 dollars)
        contract_price = opp["yes_price"] if opp["side"] == "YES" else opp["no_price"]
        if contract_price <= 0:
            continue
        count       = max(1, int(stake / contract_price))
        cost        = count * contract_price
        price_cents = round(contract_price * 100)

        side_str = opp["side"].lower()
        print(
            f"\n  [{opp['side']}] Strike ${opp['strike']:,.0f}  "
            f"edge={opp['edge']*100:+.1f}%  "
            f"{count} contracts @ {price_cents}¢  "
            f"cost=${cost:.2f}"
        )

        if DRY_RUN:
            print("  [DRY_RUN] Order not placed.")
            status   = "dry_run"
            order_id = None
            err      = None
        else:
            try:
                resp     = place_order(signer, opp["ticker"], opp["side"], count, price_cents)
                order_id = resp.get("order", {}).get("order_id") or resp.get("order_id")
                status   = "placed"
                err      = None
                print(f"  [PLACED] order_id={order_id}")
            except Exception as e:
                order_id = None
                status   = "error"
                err      = str(e)
                print(f"  [ERROR]  {e}")

        con.execute(
            """INSERT INTO trades
               (run_id, placed_at, ticker, strike, side, count, price_cents, cost_dollars,
                model_prob, market_price, edge, stake_pct, order_id, status, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                datetime.now().isoformat(),
                opp["ticker"], opp["strike"], opp["side"], count, price_cents, cost,
                opp["model_prob"], contract_price, opp["edge"], opp["stake_pct"],
                order_id, status, err,
            ),
        )
        con.commit()

        if status == "placed":
            trades_placed += 1
            total_staked  += cost

    # ── Update run summary ───────────────────────────────────────────────────
    con.execute(
        "UPDATE runs SET trades_placed=?, total_staked=? WHERE id=?",
        (trades_placed, total_staked, run_id),
    )
    con.commit()
    con.close()

    print(f"\n{'─'*60}")
    print(f"  Run complete: {trades_placed} orders placed, ${total_staked:.2f} deployed")
    print(f"  DB: {DB_PATH}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    run()
