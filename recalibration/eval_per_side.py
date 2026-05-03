#!/usr/bin/env python3
"""
eval_per_side.py — Diagnose per-side calibration of the live bucket classifiers.

For each bucket (early/mid/late), load the live pkl, rebuild the *same* test
split that training used, run predictions, then split test points by whether
clf predicted side YES (calibrated_prob > 0.5) or side NO (prob <= 0.5) and
report per-side calibration error + reliability bins.

Reads but does not modify any pkl. Safe to run anytime.

Usage:
    ~/programs/gemini_trader/venv/bin/python eval_per_side.py [--bucket early|mid|late|all]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

GEMINI = Path.home() / "programs" / "gemini_trader"
sys.path.insert(0, str(GEMINI))
sys.path.insert(0, str(GEMINI / "time_series"))

# Bring in the training helpers so the test split matches what was trained on.
from train_classifier import load_and_engineer, build_samples, CANDLE_CSV, CalibratedXGB  # noqa: F401
from train_classifier_ensemble import BUCKETS, BUCKET_REP_HORIZON


def load_bucket_clf(bucket: str):
    pkl = GEMINI / "time_series" / BUCKETS[bucket][1]
    if not pkl.exists():
        raise FileNotFoundError(pkl)
    with open(pkl, "rb") as f:
        return pickle.load(f), pkl


def per_side_metrics(y_true: np.ndarray, y_prob: np.ndarray, label: str):
    """Split predictions by predicted side, report per-side calibration."""
    n_total = len(y_prob)
    is_yes  = y_prob > 0.5          # model predicts price > strike
    is_no   = ~is_yes

    out = {"label": label, "n_total": int(n_total)}

    for side_name, mask in [("YES", is_yes), ("NO", is_no)]:
        k = int(mask.sum())
        if k == 0:
            out[side_name] = {"n": 0}
            continue
        yt = y_true[mask]
        yp = y_prob[mask]

        # When side=NO, the model is predicting "price <= strike" (outcome=0).
        # For calibration quality, we look at how well the model's claimed
        # probability matches reality — flip prob and label for NO so we're
        # always scoring the claim "this will happen".
        if side_name == "NO":
            yt_sc = 1 - yt
            yp_sc = 1 - yp
        else:
            yt_sc = yt
            yp_sc = yp

        side_metrics = {
            "n":          k,
            "pct":        round(100.0 * k / n_total, 2),
            "mean_prob":  round(float(yp_sc.mean()), 4),
            "mean_actual":round(float(yt_sc.mean()), 4),
            # Signed gap: positive = overconfident (claimed > actual)
            "overconfidence": round(float(yp_sc.mean() - yt_sc.mean()), 4),
            "brier":      round(float(brier_score_loss(yt_sc, yp_sc)), 4),
            "log_loss":   round(float(log_loss(yt_sc, yp_sc, labels=[0, 1])), 4),
        }

        # Reliability bins: decile-width buckets on the claimed probability.
        bins = np.linspace(0.5, 1.0, 6)  # 5 bins: 0.5-0.6, 0.6-0.7, ...
        rel = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            bmask = (yp_sc >= lo) & (yp_sc < hi if i < len(bins) - 2 else yp_sc <= hi)
            bn = int(bmask.sum())
            if bn < 20:
                continue
            rel.append({
                "bin":       f"{lo:.2f}-{hi:.2f}",
                "n":         bn,
                "claimed":   round(float(yp_sc[bmask].mean()), 4),
                "actual":    round(float(yt_sc[bmask].mean()), 4),
                "gap":       round(float(yp_sc[bmask].mean() - yt_sc[bmask].mean()), 4),
            })
        side_metrics["reliability"] = rel
        out[side_name] = side_metrics

    return out


def eval_bucket(bucket: str, days: int = 90) -> dict:
    print(f"\n{'='*70}\n  Evaluating bucket: {bucket}\n{'='*70}")
    clf, pkl_path = load_bucket_clf(bucket)
    print(f"  Loaded: {pkl_path.name}")

    # Rebuild the same test split that training used.
    df = load_and_engineer(CANDLE_CSV, days=days)
    test_split = int(len(df) * 0.85)
    df_test = df.iloc[test_split:].reset_index(drop=True)

    rep_h = BUCKET_REP_HORIZON[bucket]
    X_test, y_test = build_samples(df_test, rep_h)
    print(f"  Test samples (h={rep_h}): {len(X_test):,}")

    y_prob = clf.predict_proba(X_test)[:, 1]
    print(f"  Overall: Brier={brier_score_loss(y_test, y_prob):.4f}  "
          f"log_loss={log_loss(y_test, y_prob):.4f}")

    result = per_side_metrics(y_test, y_prob, label=f"{bucket} h={rep_h}")
    result["bucket"] = bucket
    result["horizon"] = rep_h
    result["pkl"] = str(pkl_path)

    # Print human-readable
    for side in ("YES", "NO"):
        s = result[side]
        if s.get("n", 0) == 0:
            continue
        tag = "⚠" if abs(s["overconfidence"]) > 0.02 else " "
        print(f"\n  {tag} {side}-side (predicted {side}): n={s['n']:,} ({s['pct']}%)")
        print(f"     mean claimed prob : {s['mean_prob']:.4f}")
        print(f"     mean actual freq  : {s['mean_actual']:.4f}")
        print(f"     overconfidence    : {s['overconfidence']:+.4f}  "
              f"({'OVERCONFIDENT' if s['overconfidence']>0 else 'under'})")
        print(f"     Brier / log_loss  : {s['brier']:.4f} / {s['log_loss']:.4f}")
        if s["reliability"]:
            print(f"     Reliability bins (claimed vs actual):")
            for r in s["reliability"]:
                tag = " ⚠" if abs(r["gap"]) > 0.03 else ""
                print(f"       {r['bin']}  n={r['n']:5d}  "
                      f"claimed={r['claimed']:.3f}  actual={r['actual']:.3f}  "
                      f"gap={r['gap']:+.4f}{tag}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="all", choices=["all", "early", "mid", "late"])
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--out", type=str, default="eval_per_side_results.json")
    args = p.parse_args()

    buckets = ["early", "mid", "late"] if args.bucket == "all" else [args.bucket]
    results = {}
    for b in buckets:
        try:
            results[b] = eval_bucket(b, days=args.days)
        except Exception as e:
            print(f"  !! {b}: {e}")
            results[b] = {"error": str(e)}

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
