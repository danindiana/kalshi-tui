#!/usr/bin/env python3
"""
train_and_eval_v2feats.py — Add synthetic market-implied features to XGBoost
training and evaluate per-side band metrics vs the live (v1-feature) models.

New features (on top of the existing 8):
  strike_z           = (strike - cur) / max(price_volatility * sqrt(horizon), 1)
  market_implied_prob= 1 - norm.cdf(strike_z * 1.25)

Both are deterministic from the v1 features, but giving XGBoost the composites
directly lets it use them as single splits instead of learning the interaction.

This script:
  1. Builds the same chronological train/val/test splits as the live trainer
  2. Rebuilds samples with 10 features (build_samples_v2) for each bucket
  3. Trains a new XGBoost + isotonic calibration
  4. Saves as classifier_{bucket}_v2feat.pkl (NOT deployed to live inference)
  5. Reports per-side band-region calibration vs live models

Does NOT modify live pkls. Does NOT modify auto_trader.py. Safe to rerun.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

GEMINI = Path.home() / "programs" / "gemini_trader"
TS     = GEMINI / "time_series"
sys.path.insert(0, str(GEMINI))
sys.path.insert(0, str(TS))

from train_classifier import load_and_engineer, CANDLE_CSV, STRIKE_STEP, STRIKES_EACH
from train_classifier_ensemble import BUCKETS, BUCKET_REP_HORIZON


FEATURE_COLS_V2 = [
    "price_rel_ma",
    "price_volatility",
    "price_momentum",
    "rsi",
    "ma_slope",
    "amount",
    "strike_dist_pct",
    "horizon_mins",
    "strike_z",             # NEW
    "market_implied_prob",  # NEW
]


def build_samples_v2(df, horizon: int):
    """Vectorized: all bars × all strikes in one numpy broadcast.
    Same output shape/order as train_classifier.build_samples + 2 new cols."""
    prices = df["price"].values.astype(np.float64)
    vols   = df["price_volatility"].values.astype(np.float64)
    feats6 = df[["price_rel_ma", "price_volatility", "price_momentum",
                 "rsi", "ma_slope", "amount"]].values.astype(np.float64)

    n_eff = len(df) - horizon
    if n_eff <= 0:
        return np.empty((0, 10), dtype="float32"), np.empty((0,), dtype="int8")

    cur    = prices[:n_eff]                        # (N,)
    future = prices[horizon:horizon + n_eff]       # (N,)
    vol    = np.maximum(vols[:n_eff], 1.0)         # (N,)

    base = np.round(cur / STRIKE_STEP) * STRIKE_STEP
    k_offsets = np.arange(-STRIKES_EACH, STRIKES_EACH + 1, dtype=np.float64)  # (K,)
    strikes = base[:, None] + k_offsets[None, :] * STRIKE_STEP                # (N, K)

    N, K = strikes.shape
    dist      = (strikes - cur[:, None]) / cur[:, None]
    strike_z  = (strikes - cur[:, None]) / (vol[:, None] * np.sqrt(horizon))
    m_prob    = 1.0 - norm.cdf(strike_z * 1.25)
    h_col     = np.full((N, K), float(horizon))

    feats_rep = np.broadcast_to(feats6[:n_eff][:, None, :], (N, K, 6))  # (N, K, 6)
    extras    = np.stack([dist, h_col, strike_z, m_prob], axis=-1)      # (N, K, 4)
    X = np.concatenate([feats_rep, extras], axis=-1).reshape(N * K, 10).astype("float32")
    y = (future[:, None] > strikes).reshape(-1).astype("int8")
    return X, y


class CalibratedXGBv2Feat:
    """Mirror of CalibratedXGB but marks 10-feature shape. Module-level for pickle."""
    def __init__(self, clf, iso):
        self.clf = clf
        self.iso = iso

    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])


def split_same_as_training(days: int):
    df = load_and_engineer(CANDLE_CSV, days=days)
    ts = int(len(df) * 0.85)
    df_train_all = df.iloc[:ts].reset_index(drop=True)
    df_test      = df.iloc[ts:].reset_index(drop=True)
    vs = int(len(df_train_all) * 0.90)
    df_tr  = df_train_all.iloc[:vs].reset_index(drop=True)
    df_val = df_train_all.iloc[vs:].reset_index(drop=True)
    return df_tr, df_val, df_test


def train_v2feat(bucket: str, days: int = 90):
    print(f"\n{'='*70}\n  Training v2-features bucket: {bucket}\n{'='*70}")
    horizons, _ = BUCKETS[bucket]
    rep_h = BUCKET_REP_HORIZON[bucket]

    df_tr, df_val, df_test = split_same_as_training(days)

    # Build training set across bucket horizons
    print(f"  Building training samples for horizons {horizons}…")
    X_list, y_list = [], []
    for h in horizons:
        Xh, yh = build_samples_v2(df_tr, h)
        X_list.append(Xh); y_list.append(yh)
    X_tr = np.vstack(X_list); y_tr = np.concatenate(y_list)
    print(f"  X_tr: {X_tr.shape}  pos_rate: {y_tr.mean():.3f}")

    X_val, y_val = build_samples_v2(df_val, rep_h)
    print(f"  X_val (h={rep_h}): {X_val.shape}")

    # Train XGBoost GPU
    params = dict(
        device="cuda", tree_method="hist",
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
        reg_lambda=1.0, objective="binary:logistic",
        eval_metric="logloss", early_stopping_rounds=30,
        random_state=42, verbosity=0,
    )
    base = xgb.XGBClassifier(**params)
    t0 = time.time()
    base.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Trained in {time.time()-t0:.1f}s  best_iter={base.best_iteration}  val_logloss={base.best_score:.4f}")

    # Isotonic calibration on val
    raw_val = base.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    new_clf = CalibratedXGBv2Feat(base, iso)

    # Feature importance
    importance = dict(zip(FEATURE_COLS_V2, base.feature_importances_))
    print(f"\n  Feature importance (sorted):")
    for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
        tag = " ← NEW" if name in ("strike_z", "market_implied_prob") else ""
        print(f"    {name:<22} {imp:.4f}{tag}")

    return new_clf, df_test, rep_h, importance


def per_side_band_metrics(y_true, y_prob, band=(0.55, 0.90)):
    out = {}
    is_yes = y_prob > 0.5
    is_no  = ~is_yes
    for side, mask in [("YES", is_yes), ("NO", is_no)]:
        if mask.sum() == 0:
            out[side] = {"n": 0}; continue
        yt = 1 - y_true[mask] if side == "NO" else y_true[mask]
        yp = 1 - y_prob[mask] if side == "NO" else y_prob[mask]
        in_band = (yp >= band[0]) & (yp <= band[1])
        nb = int(in_band.sum())
        o = {"n": int(mask.sum()), "band_n": nb}
        if nb >= 50:
            o.update({
                "band_claimed":   round(float(yp[in_band].mean()), 4),
                "band_actual":    round(float(yt[in_band].mean()), 4),
                "band_over":      round(float(yp[in_band].mean() - yt[in_band].mean()), 4),
                "band_brier":     round(float(brier_score_loss(yt[in_band], yp[in_band])), 4),
                "band_log_loss":  round(float(log_loss(yt[in_band], yp[in_band], labels=[0,1])), 4),
            })
        out[side] = o
    return out


def compare_bucket(bucket: str, days: int = 90, save: bool = True):
    new_clf, df_test, rep_h, importance = train_v2feat(bucket, days)

    # Evaluate on test set — new model uses v2 features, old model uses v1
    from train_classifier import build_samples as build_samples_v1
    X_test_v1, y_test = build_samples_v1(df_test, rep_h)
    X_test_v2, y_test2 = build_samples_v2(df_test, rep_h)
    assert np.array_equal(y_test, y_test2), "label mismatch between v1 and v2 builders"

    pkl_old = TS / BUCKETS[bucket][1]
    with open(pkl_old, "rb") as f:
        old_clf = pickle.load(f)
    old_prob = old_clf.predict_proba(X_test_v1)[:, 1]
    new_prob = new_clf.predict_proba(X_test_v2)[:, 1]

    m_before = per_side_band_metrics(y_test, old_prob)
    m_after  = per_side_band_metrics(y_test, new_prob)

    print(f"\n  Bet-relevant band [0.55,0.90] metrics on test set:")
    print(f"  {'stage':<18} {'side':<5} {'band_n':>8} {'claimed':>10} {'actual':>10} {'over':>10} {'brier':>10} {'log_loss':>10}")
    for stage, d in [("BEFORE (live v1)", m_before), ("AFTER  (v2feats)", m_after)]:
        for side in ("YES", "NO"):
            s = d.get(side, {})
            if s.get("band_n", 0) < 50:
                continue
            print(f"  {stage:<18} {side:<5} {s['band_n']:>8,} "
                  f"{s['band_claimed']:>10.4f} {s['band_actual']:>10.4f} "
                  f"{s['band_over']:>+10.4f} "
                  f"{s['band_brier']:>10.4f} {s['band_log_loss']:>10.4f}")

    # Delta report
    print(f"\n  Band-|overconfidence| delta (|before| - |after|):")
    for side in ("YES", "NO"):
        b = m_before.get(side, {}); a = m_after.get(side, {})
        if "band_over" not in b or "band_over" not in a:
            continue
        d = abs(b["band_over"]) - abs(a["band_over"])
        tag = "✓" if d > 0.005 else ("·" if d > 0 else "✗")
        print(f"    {tag} {side}: |{b['band_over']:+.4f}| → |{a['band_over']:+.4f}|  Δ={d:+.4f}")

    if save:
        pkl_out = TS / f"classifier_{bucket}_v2feat.pkl"
        with open(pkl_out, "wb") as f:
            pickle.dump(new_clf, f)
        print(f"\n  ✓ Saved {pkl_out.name}")

    return {
        "bucket":  bucket,
        "before":  m_before,
        "after":   m_after,
        "importance": {k: round(float(v), 4) for k, v in importance.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="all", choices=["all","early","mid","late"])
    p.add_argument("--days",   type=int, default=90)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    buckets = ["early","mid","late"] if args.bucket == "all" else [args.bucket]
    results = {}
    for b in buckets:
        try:
            results[b] = compare_bucket(b, args.days, save=not args.no_save)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[b] = {"error": str(e)}

    out_path = Path(__file__).parent / "v2feat_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
