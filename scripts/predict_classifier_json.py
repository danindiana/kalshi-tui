"""
predict_classifier_json.py — emit classifier-driven prediction JSON for the
wizard and kalshi-tui UI.

Matches the JSON shape of the retired predict_json.py so consumers (wizard.py
§4 panel, kalshi-tui src/api.rs PREDICT_SCRIPT subprocess) keep working after
the LSTM retirement 2026-04-14. No TensorFlow; no scaler pickles.

Output fields:
  current_price, model_prediction (= current_price for classifier),
  mae_current (= 0 for classifier), mins_to_expiry, indicators{rsi,
  volatility, momentum}, opportunities[{strike, market_yes, model_prob,
  edge, stake_pct, side}], feature_cols, model_type="classifier".
"""
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

GEMINI_DIR = Path("/home/jeb/programs/gemini_trader")
sys.path.append(str(GEMINI_DIR))

import kalshi_client
import live_candles
from classifier_device import pick_device
from risk_manager import kelly_criterion

# Pick CUDA vs CPU BEFORE the classifier pickle (and xgboost) load. See
# classifier_device.py for the why. Module-level call: this script runs fresh
# per invocation, no stale xgboost import to worry about.
_DEVICE = pick_device()

CLF_PATH = GEMINI_DIR / "time_series" / "classifier_model.pkl"
FEATURE_COLS = ["price_rel_ma", "price_volatility", "price_momentum",
                "rsi", "ma_slope", "amount"]


class _Unpickler(pickle.Unpickler):
    """Redirect CalibratedXGB lookup from __main__ to train_classifier."""
    def find_class(self, module, name):
        if name == "CalibratedXGB":
            sys.path.insert(0, str(GEMINI_DIR / "time_series"))
            from train_classifier import CalibratedXGB as _C
            return _C
        return super().find_class(module, name)


def _load_classifier():
    with open(CLF_PATH, "rb") as f:
        clf = _Unpickler(f).load()
    if _DEVICE == "cpu":
        try:
            clf.clf.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass
    return clf


def _build_features(df):
    w = 5
    df["price_ma"]         = df["price"].rolling(w).mean()
    df["price_rel_ma"]     = df["price"] - df["price_ma"]
    df["price_volatility"] = df["price"].rolling(w).std()
    df["price_momentum"]   = df["price"].diff(w)
    df["ma_slope"]         = df["price_ma"].diff()
    d    = df["price"].diff()
    gain = d.where(d > 0, 0).rolling(w).mean()
    loss = (-d.where(d < 0, 0)).rolling(w).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-12))))
    return df.dropna()


def get_data():
    df = live_candles.fetch_live_candles()
    df = _build_features(df)
    current_price = float(df["price"].iloc[-1])
    last = df.iloc[-1]

    # Approximate next hourly settlement (same convention predict_json.py used)
    now = datetime.now()
    target = now.replace(minute=0, second=0, microsecond=0)
    if target <= now:
        target = target.replace(hour=target.hour + 1)
    mins = max(1.0, (target - now).total_seconds() / 60)

    clf = _load_classifier()

    base_feats = np.array([
        last["price_rel_ma"],
        last["price_volatility"],
        last["price_momentum"],
        last["rsi"],
        last["ma_slope"],
        last["amount"],
    ], dtype="float32")

    sent = kalshi_client.analyze_sentiment("KXBTCD")
    opportunities = []
    if sent is not None and not sent.empty:
        lo = current_price * 0.98
        hi = current_price * 1.02
        for _, row in sent.iterrows():
            strike = float(row["strike"])
            if not (lo <= strike <= hi):
                continue
            dist     = (strike - current_price) / current_price
            feat_vec = np.concatenate([base_feats, [dist, mins]]).reshape(1, -1)
            p_above  = float(clf.predict_proba(feat_vec)[0][1])

            yes_price = float(row["yes_price"])
            if p_above > 0.5:
                side, model_prob, market_price = "YES", p_above, yes_price
            else:
                side, model_prob, market_price = "NO", 1.0 - p_above, 1.0 - yes_price

            edge      = model_prob - market_price
            stake_pct = kelly_criterion(model_prob, market_price, fractional_kelly=0.25)
            opportunities.append({
                "strike":     strike,
                "market_yes": yes_price,
                "model_prob": round(model_prob, 4),
                "edge":       round(edge, 4),
                "stake_pct":  float(stake_pct),
                "side":       side,
            })
    opportunities.sort(key=lambda o: o["edge"], reverse=True)

    return {
        "timestamp":        datetime.now().isoformat(),
        "current_price":    current_price,
        "model_prediction": current_price,  # classifier has no scalar price forecast
        "mae_current":      0.0,            # classifier probabilities, not Gaussian
        "mins_to_expiry":   mins,
        "model_type":       "classifier",
        "device":           _DEVICE,
        "indicators": {
            "rsi":        float(last["rsi"]),
            "volatility": float(last["price_volatility"]),
            "momentum":   float(last["price_momentum"]),
        },
        "opportunities": opportunities,
        "feature_cols":  FEATURE_COLS + ["strike_dist_pct", "horizon_mins"],
    }


if __name__ == "__main__":
    try:
        print(json.dumps(get_data()))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
