#!/usr/bin/env python3
"""
investigate_recent.py — One-shot diagnostic: is the classifier's recent out-of-sample
performance consistent with its training backtest?

Fetches 1-min candles for the past N hours, builds features, runs the same backtest
function used at training time, and prints a side-by-side comparison.
"""
import json
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "time_series"))
from train_classifier import CalibratedXGB  # noqa: F401

STRIKE_STEP  = 250
STRIKES_EACH = 4
MIN_EDGE     = 0.05


def fast_backtest(clf, df_test: pd.DataFrame, horizon: int) -> dict:
    """Batched backtest — all (i, k) tuples in one predict_proba call."""
    prices   = df_test["price"].values
    hist_vol = float(df_test["price_volatility"].mean())
    feats    = df_test[["price_rel_ma","price_volatility","price_momentum",
                         "rsi","ma_slope","amount"]].values

    n = len(df_test) - horizon
    if n <= 0:
        return {"bets": 0, "wins": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_ev": 0.0}

    # Build all rows in one go
    rows = []
    meta = []  # (i, strike)
    for i in range(n):
        cur  = prices[i]
        base = round(cur / STRIKE_STEP) * STRIKE_STEP
        for k in range(-STRIKES_EACH, STRIKES_EACH + 1):
            strike = base + k * STRIKE_STEP
            dist   = (strike - cur) / cur
            rows.append(np.concatenate([feats[i], [dist, float(horizon)]]))
            meta.append((i, strike))

    X = np.array(rows, dtype="float32")
    probs = clf.predict_proba(X)[:, 1]   # one batched GPU call

    market_std = hist_vol * np.sqrt(horizon) * 1.25
    bets, wins, pnl = 0, 0, 0.0

    for (i, strike), clf_prob in zip(meta, probs):
        cur    = prices[i]
        future = prices[i + horizon]
        market_prob = float(1 - norm.cdf((strike - cur) / max(market_std, 1)))
        market_prob = max(0.02, min(0.98, market_prob))

        edge = float(clf_prob) - market_prob
        if abs(edge) <= MIN_EDGE: continue
        if round(market_prob * 100) < 10: continue
        if market_prob > 0 and clf_prob / market_prob > 3.0: continue

        cost = market_prob if edge > 0 else 1.0 - market_prob
        won  = (future > strike) if edge > 0 else (future <= strike)
        pnl += (1.0 - cost) if won else -cost
        bets += 1
        wins += int(won)

    win_rate = wins / bets if bets else 0
    avg_ev   = pnl  / bets if bets else 0
    return {"bets": bets, "wins": wins, "win_rate": round(win_rate, 4),
            "total_pnl": round(pnl, 4), "avg_ev": round(avg_ev, 4)}


backtest = fast_backtest  # shadow the slow per-row version

COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
MODEL_PATH   = SCRIPT_DIR / "time_series" / "classifier_model.pkl"
METRICS_PATH = SCRIPT_DIR / "time_series" / "training_metrics.json"

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 48


def fetch_coinbase(start_s: int, end_s: int) -> list:
    bars = []
    chunk = 300 * 60  # 5 hr per chunk
    s = start_s
    while s < end_s:
        e = min(s + chunk, end_s)
        r = requests.get(COINBASE_URL, params={"granularity": 60, "start": s, "end": e}, timeout=10)
        r.raise_for_status()
        bars.extend(r.json())
        s = e
    return bars


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    w = 5
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["price_ma"]         = df["price"].rolling(w).mean()
    df["price_rel_ma"]     = df["price"] - df["price_ma"]
    df["price_volatility"] = df["price"].rolling(w).std()
    df["price_momentum"]   = df["price"].diff(w)
    df["ma_slope"]         = df["price_ma"].diff()
    delta = df["price"].diff()
    gain  = delta.where(delta > 0, 0).rolling(w).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(w).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / loss)))
    return df.dropna().reset_index(drop=True)


def main():
    now_s = int(datetime.now(timezone.utc).timestamp())
    start_s = now_s - HOURS * 3600

    print(f"Fetching last {HOURS} hours of 1m BTC candles from Coinbase…")
    raw = fetch_coinbase(start_s, now_s)
    # Coinbase returns: [time, low, high, open, close, volume]
    df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["price"]  = df["close"]
    df["amount"] = df["volume"]
    df = build_features(df)
    print(f"  {len(df):,} bars after feature build  "
          f"({datetime.fromtimestamp(df['timestamp'].iloc[0], tz=timezone.utc)} → "
          f"{datetime.fromtimestamp(df['timestamp'].iloc[-1], tz=timezone.utc)})")

    price_range = f"${df['price'].min():,.0f} – ${df['price'].max():,.0f}"
    print(f"  Price range: {price_range}")

    print(f"\nLoading classifier from {MODEL_PATH}…")
    import pickle as _pkl
    class _Unpickler(_pkl.Unpickler):
        def find_class(self, module, name):
            if name == "CalibratedXGB":
                from train_classifier import CalibratedXGB as _C
                return _C
            return super().find_class(module, name)
    with open(MODEL_PATH, "rb") as f:
        clf = _Unpickler(f).load()

    with open(METRICS_PATH) as f:
        trained = json.load(f)

    print(f"\nRunning backtest on recent data at multiple horizons…")
    print(f"{'horizon':>8}  {'bets':>6}  {'win_rate':>10}  {'avg_ev':>10}  {'total_pnl':>10}")
    print("-" * 55)
    horizons = [15, 30, 45, 60, 75, 90]
    recent_stats = []
    for h in horizons:
        bt = backtest(clf, df, h)
        recent_stats.append((h, bt))
        print(f"{h:>6}min  {bt['bets']:>6,}  {bt['win_rate']:>9.1%}  {bt['avg_ev']:>+10.4f}  {bt['total_pnl']:>+10.3f}", flush=True)

    # Aggregate
    total_bets = sum(bt["bets"] for _, bt in recent_stats)
    total_wins = sum(bt.get("wins", int(bt["win_rate"] * bt["bets"])) for _, bt in recent_stats)
    total_pnl  = sum(bt["total_pnl"] for _, bt in recent_stats)
    agg_win_rate = total_wins / total_bets if total_bets else 0
    agg_avg_ev   = total_pnl  / total_bets if total_bets else 0

    print("-" * 55)
    print(f"{'ALL':>8}  {total_bets:>6,}  {agg_win_rate:>9.1%}  {agg_avg_ev:>+10.4f}  {total_pnl:>+10.3f}")

    print("\n" + "=" * 60)
    print("COMPARISON vs training backtest")
    print("=" * 60)
    trained_bt = trained.get("backtest", {})
    print(f"  Training (test split, h=15min):")
    print(f"    bets:      {trained_bt.get('bets'):>8,}")
    print(f"    win_rate:  {trained_bt.get('win_rate'):>8.1%}")
    print(f"    avg_ev:    {trained_bt.get('avg_ev'):>+8.4f}")
    print(f"  Recent ({HOURS}h out-of-sample, all horizons combined):")
    print(f"    bets:      {total_bets:>8,}")
    print(f"    win_rate:  {agg_win_rate:>8.1%}")
    print(f"    avg_ev:    {agg_avg_ev:>+8.4f}")

    delta_wr = agg_win_rate - trained_bt.get("win_rate", 0)
    delta_ev = agg_avg_ev - trained_bt.get("avg_ev", 0)
    print(f"\n  Δ win_rate: {delta_wr:+.1%}")
    print(f"  Δ avg_ev:   {delta_ev:+.4f}")

    if agg_avg_ev < 0:
        print("\n  ⚠ Recent EV is NEGATIVE — model is losing money in current regime.")
    elif delta_ev < -0.03:
        print("\n  ⚠ Recent EV significantly below training — possible drift.")
    else:
        print("\n  ✓ Recent performance roughly consistent with training.")


if __name__ == "__main__":
    main()
