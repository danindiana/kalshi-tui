#!/usr/bin/env python3
"""
train_classifier.py — Directional XGBoost classifier for Kalshi BTC settlement prediction.

Directly outputs P(BTC_price_at_settlement > strike) for any (strike, horizon),
replacing the delta-LSTM + Gaussian approximation in auto_trader.py.

Model: XGBoost (GPU-accelerated, multi-GPU via NCCL) with isotonic calibration.
  - device='cuda' → uses both GPUs automatically when NCCL is available
  - Calibration: sklearn CalibratedClassifierCV with isotonic regression
  - Output: calibrated P(win) directly usable by Kelly criterion

Feature set (per bar × per strike):
  price_rel_ma      — mean reversion signal
  price_volatility  — regime signal
  price_momentum    — trend signal
  rsi               — overbought/oversold
  ma_slope          — trend acceleration
  amount            — volume / participation
  strike_dist_pct   — (strike - spot) / spot  (strike-aware)
  horizon_mins      — time to settlement (horizon-aware, one model for all windows)

Training: chronological split (last 15% = test, never shuffled).
Deployment gate: avg_ev > 0 and bets >= 50, OR --force.

Usage:
  python train_classifier.py [--days 90] [--horizon 15] [--force]
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# ── GPU setup ─────────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HIST = True
except ImportError:
    HAS_HIST = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
CANDLE_CSV   = SCRIPT_DIR / "historical_candle_data_btcusd_full.csv"
CLF_OUT      = SCRIPT_DIR / "classifier_model.pkl"
METRICS_OUT  = SCRIPT_DIR / "training_metrics.json"
CAND_METRICS = SCRIPT_DIR / "training_metrics_candidate.json"

WIN          = 5
STRIKE_STEP  = 250
STRIKES_EACH = 4
MIN_EDGE     = 0.05

FEATURE_COLS = [
    "price_rel_ma",
    "price_volatility",
    "price_momentum",
    "rsi",
    "ma_slope",
    "amount",
    "strike_dist_pct",
    "horizon_mins",
]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_engineer(path: Path, days: int) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("timestampms").reset_index(drop=True)
    if "price"  not in df.columns: df["price"]  = df["close"]
    if "amount" not in df.columns: df["amount"] = df["volume"]

    if days > 0:
        cutoff = df["timestampms"].max() - days * 86400 * 1000
        df = df[df["timestampms"] >= cutoff].copy()

    df["price_ma"]         = df["price"].rolling(WIN).mean()
    df["price_rel_ma"]     = df["price"] - df["price_ma"]
    df["price_volatility"] = df["price"].rolling(WIN).std()
    df["price_momentum"]   = df["price"].diff(WIN)
    df["ma_slope"]         = df["price_ma"].diff()

    delta = df["price"].diff()
    gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    return df.dropna().reset_index(drop=True)


def build_samples(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    prices = df["price"].values
    n      = len(df)
    feats  = df[["price_rel_ma","price_volatility","price_momentum",
                  "rsi","ma_slope","amount"]].values

    X_rows, y_rows = [], []
    for i in range(n - horizon):
        cur    = prices[i]
        future = prices[i + horizon]
        base   = round(cur / STRIKE_STEP) * STRIKE_STEP

        for k in range(-STRIKES_EACH, STRIKES_EACH + 1):
            strike = base + k * STRIKE_STEP
            dist   = (strike - cur) / cur
            row    = np.concatenate([feats[i], [dist, float(horizon)]])
            X_rows.append(row)
            y_rows.append(1 if future > strike else 0)

    return np.array(X_rows, dtype="float32"), np.array(y_rows, dtype="int8")


# ── Model ─────────────────────────────────────────────────────────────────────

class CalibratedXGB:
    """XGBoost + isotonic calibration wrapper. Must be module-level for pickle."""
    def __init__(self, clf, iso):
        self.clf = clf
        self.iso = iso

    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])


def build_and_train(X_train, y_train, X_val, y_val) -> object:
    """Train XGBoost on GPU (falls back to HistGBT if XGBoost unavailable)."""

    if HAS_XGB:
        print("  Backend: XGBoost GPU (both GPUs via NCCL)")

        # Detect GPU count
        try:
            import subprocess
            n_gpu = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True).strip().count("\n")) + 1
        except Exception:
            n_gpu = 1
        print(f"  GPUs detected: {n_gpu}")

        params = dict(
            device           = "cuda",        # uses all GPUs with NCCL when n_gpu > 1
            tree_method      = "hist",
            n_estimators     = 1000,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 50,
            reg_lambda       = 1.0,
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            early_stopping_rounds = 30,
            random_state     = 42,
            verbosity        = 1,
            n_jobs           = -1,
        )
        base = xgb.XGBClassifier(**params)

        print("  Fitting XGBoost (GPU) with early stopping on validation set…")
        base.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        print(f"  Best iteration: {base.best_iteration}  "
              f"val_logloss={base.best_score:.4f}")

        # Isotonic calibration on the validation set (separate from CV to avoid data leakage)
        print("  Applying isotonic calibration on validation set…")
        from sklearn.isotonic import IsotonicRegression
        raw_probs = base.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs, y_val)

        # Wrap into module-level class so pickle works
        return CalibratedXGB(base, iso)

    elif HAS_HIST:
        print("  Backend: HistGradientBoosting (CPU fallback — install xgboost for GPU)")
        from sklearn.ensemble import HistGradientBoostingClassifier
        base = HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, min_samples_leaf=50,
            learning_rate=0.05, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42, verbose=1,
        )
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
        clf.fit(X_train, y_train)
        return clf
    else:
        raise RuntimeError("No supported classifier backend found (install xgboost or scikit-learn)")


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(clf, df_test: pd.DataFrame, horizon: int) -> dict:
    prices   = df_test["price"].values
    hist_vol = float(df_test["price_volatility"].mean())
    feats    = df_test[["price_rel_ma","price_volatility","price_momentum",
                         "rsi","ma_slope","amount"]].values

    bets, wins, pnl = 0, 0, 0.0

    for i in range(len(df_test) - horizon):
        cur    = prices[i]
        future = prices[i + horizon]
        base   = round(cur / STRIKE_STEP) * STRIKE_STEP

        for k in range(-STRIKES_EACH, STRIKES_EACH + 1):
            strike = base + k * STRIKE_STEP
            dist   = (strike - cur) / cur
            row    = np.concatenate([feats[i], [dist, float(horizon)]]).reshape(1, -1)

            clf_prob    = float(clf.predict_proba(row)[0][1])
            market_std  = hist_vol * np.sqrt(horizon) * 1.25
            market_prob = float(1 - norm.cdf((strike - cur) / max(market_std, 1)))
            market_prob = max(0.02, min(0.98, market_prob))

            edge = clf_prob - market_prob
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
    return {
        "bets":       bets,
        "wins":       wins,
        "win_rate":   round(win_rate, 4),
        "total_pnl":  round(pnl, 4),
        "avg_ev":     round(avg_ev, 4),
        "deployable": avg_ev > 0 and bets >= 50,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int, default=90)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    if not CANDLE_CSV.exists():
        print(f"ERROR: {CANDLE_CSV} not found")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.days}-day candle dataset…")
    df = load_and_engineer(CANDLE_CSV, args.days)
    print(f"  {len(df):,} rows  "
          f"({datetime.fromtimestamp(df['timestampms'].iloc[0]/1000).date()} → "
          f"{datetime.fromtimestamp(df['timestampms'].iloc[-1]/1000).date()})")
    print(f"  Price range: ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")

    # ── Split ─────────────────────────────────────────────────────────────────
    split    = int(len(df) * 0.85)
    df_train = df.iloc[:split].copy().reset_index(drop=True)
    df_test  = df.iloc[split:].copy().reset_index(drop=True)

    # Use last 10% of train as validation for early stopping
    val_split = int(len(df_train) * 0.90)
    df_tr  = df_train.iloc[:val_split].reset_index(drop=True)
    df_val = df_train.iloc[val_split:].reset_index(drop=True)

    print(f"  Train: {len(df_tr):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # ── Samples — train on multiple horizons ──────────────────────────────────
    print(f"\nBuilding training samples (horizons 10–90 min)…")
    X_all, y_all = [], []
    for h in [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]:
        Xh, yh = build_samples(df_tr, h)
        X_all.append(Xh)
        y_all.append(yh)
        print(f"  h={h:2d}min  {len(Xh):,} samples  pos_rate={yh.mean():.3f}")
    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)
    print(f"  Total: {len(X_train):,} samples")

    print(f"\nBuilding validation samples (h={args.horizon}min for early stopping)…")
    X_val, y_val = build_samples(df_val, args.horizon)
    print(f"  {len(X_val):,} validation samples")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining classifier…")
    t0  = datetime.now()
    clf = build_and_train(X_train, y_train, X_val, y_val)
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Training time: {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nEvaluating on test set (h={args.horizon}min)…")
    X_test, y_test = build_samples(df_test, args.horizon)
    print(f"  {len(X_test):,} test samples")

    y_prob = clf.predict_proba(X_test)[:, 1]
    brier      = brier_score_loss(y_test, y_prob)
    ll         = log_loss(y_test, y_prob)
    auc        = roc_auc_score(y_test, y_prob)
    base_rate  = y_test.mean()
    brier_base = brier_score_loss(y_test, np.full_like(y_prob, base_rate))
    brier_skill = (1 - brier / brier_base) * 100

    print(f"\n  Test metrics (h={args.horizon}min):")
    print(f"    AUC-ROC:       {auc:.4f}  (0.5=random, 1.0=perfect)")
    print(f"    Brier score:   {brier:.4f}  (baseline={brier_base:.4f})")
    print(f"    Brier skill:   {brier_skill:+.2f}%  (>0 = beats baseline)")
    print(f"    Log-loss:      {ll:.4f}")

    # ── Backtest ──────────────────────────────────────────────────────────────
    print(f"\nRunning Kalshi backtest on test set…")
    bt = backtest(clf, df_test, args.horizon)
    print(f"  Bets simulated:  {bt['bets']:,}")
    print(f"  Win rate:        {bt['win_rate']:.1%}")
    print(f"  Total P&L:       {bt['total_pnl']:+.3f} units")
    print(f"  Avg EV/bet:      {bt['avg_ev']:+.4f}")
    print(f"  Deployable:      {'YES ✓' if bt['deployable'] else 'NO'}")

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics = {
        "trained_at":      datetime.now().isoformat(),
        "model_type":      "classifier",
        "feature_cols":    FEATURE_COLS,
        "horizon_default": args.horizon,
        "backend":         "xgboost_gpu" if HAS_XGB else "histgbt_cpu",
        "auc":             round(auc, 4),
        "brier":           round(brier, 4),
        "brier_base":      round(brier_base, 4),
        "brier_skill":     round(brier_skill, 2),
        "training_secs":   round(elapsed, 1),
        "backtest":        bt,
        "chronological_split": True,
    }

    if bt["deployable"] or args.force:
        with open(CLF_OUT, "wb") as f:
            pickle.dump(clf, f)
        with open(METRICS_OUT, "w") as f:
            json.dump(metrics, f, indent=2)
        reason = "backtest passed" if bt["deployable"] else "--force"
        print(f"\n✓ Deployed to production ({reason})")
        print(f"  {CLF_OUT}")
        print(f"  AUC={auc:.4f}  Brier skill={brier_skill:+.2f}%  "
              f"avg_ev={bt['avg_ev']:+.4f}")
    else:
        with open(CAND_METRICS, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✗ NOT deployed — backtest failed (avg_ev={bt['avg_ev']:+.4f})")
        print("  Use --force to deploy anyway")


if __name__ == "__main__":
    main()
