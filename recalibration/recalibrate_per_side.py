#!/usr/bin/env python3
"""
recalibrate_per_side.py — Refit per-side isotonic calibration on existing models.

For each bucket (early/mid/late):
  1. Load the live CalibratedXGB pkl
  2. Pull raw XGB predictions on the training val split (same split trainer used)
  3. Fit two IsotonicRegressions:
       iso_high on val samples where raw_prob > 0.5
       iso_low  on val samples where raw_prob <= 0.5
  4. Wrap into PerSideCalibratedXGB and save as classifier_{bucket}_v2.pkl
  5. Report per-side calibration before/after

Does NOT overwrite the live pkls — always writes to _v2.pkl.
Deployment is a separate manual step (cp v2 over after inspection).

Usage:
    ~/programs/gemini_trader/venv/bin/python recalibrate_per_side.py [--bucket all]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

GEMINI = Path.home() / "programs" / "gemini_trader"
TS     = GEMINI / "time_series"
sys.path.insert(0, str(GEMINI))
sys.path.insert(0, str(TS))

from train_classifier import (
    load_and_engineer, build_samples, CANDLE_CSV,
    CalibratedXGB, PerSideCalibratedXGB,  # noqa: F401
)
from train_classifier_ensemble import BUCKETS, BUCKET_REP_HORIZON


def split_same_as_training(days: int):
    """Reproduce the exact train/val/test split train_classifier_ensemble.py uses."""
    df = load_and_engineer(CANDLE_CSV, days=days)
    test_split   = int(len(df) * 0.85)
    df_train_all = df.iloc[:test_split].reset_index(drop=True)
    df_test      = df.iloc[test_split:].reset_index(drop=True)
    val_split    = int(len(df_train_all) * 0.90)
    df_val       = df_train_all.iloc[val_split:].reset_index(drop=True)
    return df_val, df_test


def fit_per_side_iso(raw_val: np.ndarray, y_val: np.ndarray):
    """Fit two isotonics on disjoint regions of raw prob."""
    mask_high = raw_val > 0.5
    if mask_high.sum() < 200 or (~mask_high).sum() < 200:
        raise RuntimeError(f"insufficient samples per side "
                           f"(high={mask_high.sum()}, low={(~mask_high).sum()})")

    iso_high = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_high.fit(raw_val[mask_high], y_val[mask_high])

    iso_low = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_low.fit(raw_val[~mask_high], y_val[~mask_high])

    return iso_low, iso_high


def per_side_metrics(y_true, y_prob, name):
    """Compact per-side metrics for comparison reporting."""
    out = {"name": name}
    is_yes = y_prob > 0.5
    is_no  = ~is_yes
    for side, mask in [("YES", is_yes), ("NO", is_no)]:
        n = int(mask.sum())
        if n == 0:
            out[side] = {"n": 0}
            continue
        yt = y_true[mask]; yp = y_prob[mask]
        if side == "NO":
            yt, yp = 1 - yt, 1 - yp
        out[side] = {
            "n":             n,
            "mean_claimed":  round(float(yp.mean()), 4),
            "mean_actual":   round(float(yt.mean()), 4),
            "overconfidence":round(float(yp.mean() - yt.mean()), 4),
            "brier":         round(float(brier_score_loss(yt, yp)), 4),
            "log_loss":      round(float(log_loss(yt, yp, labels=[0,1])), 4),
        }
    return out


def recalibrate_bucket(bucket: str, days: int = 90) -> dict:
    print(f"\n{'='*70}\n  Recalibrating bucket: {bucket}\n{'='*70}")
    pkl_in = TS / BUCKETS[bucket][1]
    pkl_out = TS / f"classifier_{bucket}_v2.pkl"

    with open(pkl_in, "rb") as f:
        old = pickle.load(f)
    print(f"  Loaded: {pkl_in.name}")

    df_val, df_test = split_same_as_training(days)
    rep_h = BUCKET_REP_HORIZON[bucket]
    print(f"  Val bars: {len(df_val):,}  Test bars: {len(df_test):,}  rep_h={rep_h}")

    # Pull raw (uncalibrated) XGB predictions on val
    Xv, yv = build_samples(df_val, rep_h)
    raw_val = old.clf.predict_proba(Xv)[:, 1]
    print(f"  Val samples: {len(Xv):,}  "
          f"raw_prob>0.5 = {int((raw_val>0.5).sum()):,}  "
          f"raw_prob<=0.5 = {int((raw_val<=0.5).sum()):,}")

    # Fit per-side isotonics
    iso_low, iso_high = fit_per_side_iso(raw_val, yv)
    new_clf = PerSideCalibratedXGB(old.clf, iso_low, iso_high)

    # Evaluate on held-out test set
    Xt, yt = build_samples(df_test, rep_h)
    old_prob = old.predict_proba(Xt)[:, 1]
    new_prob = new_clf.predict_proba(Xt)[:, 1]

    before = per_side_metrics(yt, old_prob, "before")
    after  = per_side_metrics(yt, new_prob, "after")

    # Print comparison
    print(f"\n  Per-side metrics (test set, n={len(Xt):,}):")
    print(f"  {'':<20} {'side':<5} {'n':>10} {'claimed':>10} {'actual':>10} {'over±':>10} {'brier':>10} {'log_loss':>10}")
    for stage, d in [("BEFORE (live)", before), ("AFTER  (v2)",   after)]:
        for side in ("YES", "NO"):
            s = d.get(side, {"n": 0})
            if s["n"] == 0:
                continue
            print(f"  {stage:<20} {side:<5} {s['n']:>10,} "
                  f"{s['mean_claimed']:>10.4f} {s['mean_actual']:>10.4f} "
                  f"{s['overconfidence']:>+10.4f} "
                  f"{s['brier']:>10.4f} {s['log_loss']:>10.4f}")

    # Delta (abs reduction in |overconfidence|)
    print(f"\n  Calibration deltas (|overconfidence| reduced by):")
    for side in ("YES", "NO"):
        b = before.get(side, {})
        a = after.get(side, {})
        if not b or not a:
            continue
        delta = abs(b["overconfidence"]) - abs(a["overconfidence"])
        tag = "✓" if delta > 0 else "✗"
        print(f"    {tag} {side}: |{b['overconfidence']:+.4f}| → |{a['overconfidence']:+.4f}| "
              f"  Δ={delta:+.4f}")

    # Save
    with open(pkl_out, "wb") as f:
        pickle.dump(new_clf, f)
    print(f"\n  ✓ Wrote {pkl_out.name}")

    return {
        "bucket":  bucket,
        "pkl_in":  str(pkl_in),
        "pkl_out": str(pkl_out),
        "before":  before,
        "after":   after,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="all", choices=["all", "early", "mid", "late"])
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--out", type=str, default="recalibration_results.json")
    args = p.parse_args()

    buckets = ["early", "mid", "late"] if args.bucket == "all" else [args.bucket]
    results = {}
    for b in buckets:
        try:
            results[b] = recalibrate_bucket(b, args.days)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[b] = {"error": str(e)}

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
