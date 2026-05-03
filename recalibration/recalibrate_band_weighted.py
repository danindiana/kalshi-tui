#!/usr/bin/env python3
"""
recalibrate_band_weighted.py — Per-side isotonic with bet-relevance weighting.

The unweighted per-side isotonic (recalibrate_per_side.py) barely moved metrics
because the val set is dominated by easy extreme-prob samples (~80% fall in
0.90-1.00). Bets happen in the 0.55-0.85 band where market vs model disagree.

This variant uses sklearn IsotonicRegression(sample_weight=...), weighting
band-region samples 10× higher so the fit tracks the reliability curve where
it matters for production.

Output: classifier_{bucket}_v3.pkl (wraps PerSideCalibratedXGB).
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


# ── Config ──────────────────────────────────────────────────────────────────
BAND_LO    = 0.55      # claimed-prob band lower edge (where we bet)
BAND_HI    = 0.90      # band upper edge
BAND_BOOST = 10.0      # weight multiplier inside the band


def split_same_as_training(days: int):
    df = load_and_engineer(CANDLE_CSV, days=days)
    ts = int(len(df) * 0.85)
    df_train = df.iloc[:ts].reset_index(drop=True)
    df_test  = df.iloc[ts:].reset_index(drop=True)
    vs = int(len(df_train) * 0.90)
    df_val = df_train.iloc[vs:].reset_index(drop=True)
    return df_val, df_test


def band_weighted_iso(raw: np.ndarray, y: np.ndarray):
    """Fit isotonic with 10× weight inside [BAND_LO, BAND_HI], and also on the
    mirror band [1-BAND_HI, 1-BAND_LO] for NO-side samples — symmetric boost."""
    # For a raw_prob p, the claimed prob from the side model bets is
    # max(p, 1-p) — the model's stronger confidence side. Band-weight uses
    # that.
    claimed = np.maximum(raw, 1 - raw)
    weights = np.ones_like(raw)
    in_band = (claimed >= BAND_LO) & (claimed <= BAND_HI)
    weights[in_band] = BAND_BOOST
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y, sample_weight=weights)
    return iso, int(in_band.sum()), int((~in_band).sum())


def fit_per_side_band(raw_val: np.ndarray, y_val: np.ndarray):
    """Fit two band-weighted isotonics on raw>0.5 and raw<=0.5 regions."""
    mask_high = raw_val > 0.5
    iso_high, nh_band, nh_out = band_weighted_iso(raw_val[mask_high], y_val[mask_high])
    iso_low,  nl_band, nl_out = band_weighted_iso(raw_val[~mask_high], y_val[~mask_high])
    stats = {
        "high_band_n": nh_band, "high_out_n": nh_out,
        "low_band_n":  nl_band, "low_out_n":  nl_out,
    }
    return iso_low, iso_high, stats


def per_band_metrics(y_true, y_prob, name, band_lo=BAND_LO, band_hi=BAND_HI):
    """Metrics restricted to the bet-relevant band + per side."""
    out = {"name": name, "n_total": int(len(y_prob))}
    for side_name, mask_side in [("YES", y_prob > 0.5), ("NO", y_prob <= 0.5)]:
        if side_name == "NO":
            yt = 1 - y_true[mask_side]; yp = 1 - y_prob[mask_side]
        else:
            yt = y_true[mask_side]; yp = y_prob[mask_side]
        if len(yp) == 0:
            out[side_name] = {"n": 0}; continue
        in_band = (yp >= band_lo) & (yp <= band_hi)
        nb = int(in_band.sum())
        side_out = {
            "n":         int(len(yp)),
            "band_n":    nb,
            "band_pct":  round(100.0 * nb / max(1, len(yp)), 2),
        }
        if nb >= 50:
            yt_b = yt[in_band]; yp_b = yp[in_band]
            side_out["band_claimed"]   = round(float(yp_b.mean()), 4)
            side_out["band_actual"]    = round(float(yt_b.mean()), 4)
            side_out["band_over"]      = round(float(yp_b.mean() - yt_b.mean()), 4)
            side_out["band_brier"]     = round(float(brier_score_loss(yt_b, yp_b)), 4)
            side_out["band_log_loss"]  = round(float(log_loss(yt_b, yp_b, labels=[0,1])), 4)
        out[side_name] = side_out
    return out


def recalibrate(bucket: str, days: int = 90) -> dict:
    print(f"\n{'='*70}\n  Band-weighted per-side recalibration: {bucket}\n{'='*70}")
    pkl_in  = TS / BUCKETS[bucket][1]
    pkl_out = TS / f"classifier_{bucket}_v3.pkl"
    with open(pkl_in, "rb") as f:
        old = pickle.load(f)

    df_val, df_test = split_same_as_training(days)
    rep_h = BUCKET_REP_HORIZON[bucket]

    Xv, yv = build_samples(df_val, rep_h)
    raw_val = old.clf.predict_proba(Xv)[:, 1]
    iso_low, iso_high, stats = fit_per_side_band(raw_val, yv)
    print(f"  Band-weighted fit (boost={BAND_BOOST}x in [{BAND_LO},{BAND_HI}]): {stats}")

    new_clf = PerSideCalibratedXGB(old.clf, iso_low, iso_high)

    Xt, yt = build_samples(df_test, rep_h)
    old_prob = old.predict_proba(Xt)[:, 1]
    new_prob = new_clf.predict_proba(Xt)[:, 1]

    before = per_band_metrics(yt, old_prob, "before")
    after  = per_band_metrics(yt, new_prob, "after")

    # Focus report on the band metrics — that's where bets live
    print(f"\n  Bet-relevant band [{BAND_LO},{BAND_HI}] metrics:")
    print(f"  {'stage':<16} {'side':<5} {'band_n':>8} {'claimed':>10} {'actual':>10} {'over':>10} {'brier':>10} {'log_loss':>10}")
    for stage, d in [("BEFORE (live)", before), ("AFTER  (v3)",   after)]:
        for side in ("YES", "NO"):
            s = d.get(side, {})
            if s.get("band_n", 0) < 50:
                continue
            print(f"  {stage:<16} {side:<5} {s['band_n']:>8,} "
                  f"{s['band_claimed']:>10.4f} {s['band_actual']:>10.4f} "
                  f"{s['band_over']:>+10.4f} "
                  f"{s['band_brier']:>10.4f} {s['band_log_loss']:>10.4f}")

    print(f"\n  Band-|overconfidence| delta:")
    for side in ("YES", "NO"):
        b = before.get(side, {}); a = after.get(side, {})
        if "band_over" not in b or "band_over" not in a:
            continue
        d = abs(b["band_over"]) - abs(a["band_over"])
        tag = "✓" if d > 0.003 else ("·" if d > 0 else "✗")
        print(f"    {tag} {side}: |{b['band_over']:+.4f}| → |{a['band_over']:+.4f}|  Δ={d:+.4f}")

    with open(pkl_out, "wb") as f:
        pickle.dump(new_clf, f)
    print(f"\n  ✓ Wrote {pkl_out.name}")

    return {"bucket": bucket, "pkl_in": str(pkl_in), "pkl_out": str(pkl_out),
            "fit_stats": stats, "before": before, "after": after}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="all", choices=["all", "early", "mid", "late"])
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--out", type=str, default="band_weighted_results.json")
    args = p.parse_args()

    buckets = ["early", "mid", "late"] if args.bucket == "all" else [args.bucket]
    results = {}
    for b in buckets:
        try:
            results[b] = recalibrate(b, args.days)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[b] = {"error": str(e)}

    out_path = Path(__file__).parent / args.out
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
