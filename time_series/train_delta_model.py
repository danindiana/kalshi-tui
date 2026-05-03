#!/usr/bin/env python3
"""
train_delta_model.py — Train an LSTM that predicts 1-minute BTC price *change*.

Why delta instead of absolute price:
  - 1-min delta is stationary (~$0 mean, ~$50 std) — no regime shifts
  - Scaler trained once generalises across different price levels
  - Model bias is directly observable (systematic over/under prediction)
  - Inference: predicted_price = current_price + predicted_delta

Features (all relative, no absolute price):
  price_rel_ma   = price - 5-min MA         (mean-reversion signal)
  amount         = volume                    (participation signal)
  price_volatility = 5-min rolling std      (regime signal)
  price_momentum = price.diff(5)            (trend signal)
  rsi            = 5-period RSI             (momentum oscillator)
  ma_slope       = 5-min MA first diff      (trend acceleration)

Target: price[t+1] - price[t]  (next-minute price change)

Train/test: CHRONOLOGICAL — last 15% of rows as test, never shuffled.

Deployment gate: model only replaces production if backtest shows
  positive expected value (EV > 0) on the test set's Kalshi simulation.

Usage:
  python train_delta_model.py [--days 90] [--epochs 50] [--force]
"""

import os, sys, json, pickle, argparse, shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir="
    "/home/jeb/programs/gemini_trader/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc"
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D, LayerNormalization
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

SCRIPT_DIR   = Path(__file__).parent
CANDLE_CSV   = SCRIPT_DIR / "historical_candle_data_btcusd_full.csv"
MODEL_OUT    = SCRIPT_DIR / "production_lstm_model.keras"
SCALER_X_OUT = SCRIPT_DIR / "scaler_x.pkl"
SCALER_Y_OUT = SCRIPT_DIR / "scaler_y.pkl"
METRICS_OUT  = SCRIPT_DIR / "training_metrics.json"

SEQ_LEN   = 30   # 30 minutes of context (vs 15 before)
WIN       = 5    # rolling window for indicators
MIN_EDGE  = 0.05

FEATURE_COLS = [
    "price_rel_ma",       # price distance from 5-min MA
    "amount",             # volume
    "price_volatility",   # 5-min rolling std
    "price_momentum",     # price.diff(5)
    "rsi",                # 5-period RSI
    "ma_slope",           # 5-min MA slope (diff)
]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_engineer(path: Path, days: int) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("timestampms").reset_index(drop=True)
    if "price" not in df.columns:
        df["price"] = df["close"]
    if "amount" not in df.columns:
        df["amount"] = df["volume"]

    if days > 0:
        cutoff = df["timestampms"].max() - days * 86400 * 1000
        df = df[df["timestampms"] >= cutoff].copy()

    # Features
    df["price_ma"]       = df["price"].rolling(WIN).mean()
    df["price_rel_ma"]   = df["price"] - df["price_ma"]
    df["price_volatility"] = df["price"].rolling(WIN).std()
    df["price_momentum"] = df["price"].diff(WIN)
    df["ma_slope"]       = df["price_ma"].diff()

    delta = df["price"].diff()
    gain  = delta.where(delta > 0, 0).rolling(WIN).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(WIN).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # Target: next-minute price change
    df["target_delta"] = df["price"].diff().shift(-1)   # delta at t+1

    df = df.dropna(subset=FEATURE_COLS + ["target_delta"]).reset_index(drop=True)
    return df


def make_sequences(df: pd.DataFrame, scaler_x, scaler_y, fit: bool):
    feats  = df[FEATURE_COLS].values
    target = df["target_delta"].values.reshape(-1, 1)

    if fit:
        feats_s  = scaler_x.fit_transform(feats)
        target_s = scaler_y.fit_transform(target)
    else:
        feats_s  = scaler_x.transform(feats)
        target_s = scaler_y.transform(target)

    X, y = [], []
    for i in range(len(feats_s) - SEQ_LEN):
        X.append(feats_s[i : i + SEQ_LEN])
        y.append(target_s[i + SEQ_LEN][0])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_features: int) -> Model:
    inp  = Input(shape=(SEQ_LEN, n_features))
    x    = LSTM(256, return_sequences=True, dropout=0.2)(inp)
    x    = LayerNormalization()(x)
    x    = LSTM(128, return_sequences=True, dropout=0.2)(x)
    attn = Attention()([x, x])
    x    = GlobalAveragePooling1D()(attn)
    x    = Dense(128, activation="relu")(x)
    x    = Dropout(0.2)(x)
    x    = Dense(64, activation="relu")(x)
    out  = Dense(1)(x)
    m    = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    return m


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(df_test: pd.DataFrame, delta_preds: np.ndarray, mae_delta: float) -> dict:
    """
    Simulate Kalshi YES/NO bets on the test set.

    For each bar we:
      1. Compute predicted next price = current + predicted_delta
      2. For strikes near current price, compute model_prob and market_fair_price
         (market_fair_price uses normal distribution with 1-min historical vol)
      3. If edge > MIN_EDGE: simulate bet, record win/loss
    """
    prices  = df_test["price"].values[SEQ_LEN:]   # current prices aligned with preds
    actuals = df_test["price"].values[SEQ_LEN:] + \
              df_test["target_delta"].values[SEQ_LEN:]   # actual next price

    hist_vol = float(df_test["price_volatility"].mean())  # market's 1-min vol estimate

    bets, wins, pnl = 0, 0, 0.0
    model_probs, actual_wins = [], []

    for i, (cur, pred_delta, actual_next) in enumerate(zip(prices, delta_preds, actuals)):
        predicted_next = cur + pred_delta

        # Strikes ±1% of current price, in $250 increments (Kalshi style)
        base = round(cur / 250) * 250
        strikes = [base + k * 250 for k in range(-4, 5)]

        for strike in strikes:
            # Model probability: normal dist centered at predicted_next, std from MAE
            std_model  = mae_delta * 1.25
            model_prob = float(1 - norm.cdf(strike, predicted_next, std_model))

            # Market fair price: normal dist centered at current price (no drift), std from hist vol
            market_prob = float(1 - norm.cdf(strike, cur, hist_vol * 1.25))

            # Clamp to avoid 0/1
            market_prob = max(0.02, min(0.98, market_prob))
            model_prob  = max(0.02, min(0.98, model_prob))

            edge = model_prob - market_prob
            if abs(edge) <= MIN_EDGE:
                continue

            if edge > 0:   # bet YES
                cost   = market_prob
                win    = actual_next > strike
                profit = (1.0 - cost) if win else -cost
            else:           # bet NO
                cost   = 1.0 - market_prob
                win    = actual_next <= strike
                profit = (1.0 - cost) if win else -cost

            bets    += 1
            wins    += int(win)
            pnl     += profit
            model_probs.append(model_prob if edge > 0 else 1 - model_prob)
            actual_wins.append(int(win))

    win_rate = wins / bets if bets else 0
    avg_ev   = pnl / bets  if bets else 0

    # Calibration: how well does model_prob predict actual win rate?
    calib_mae = float(np.mean(np.abs(np.array(model_probs) - np.array(actual_wins)))) if bets else 1.0

    return {
        "bets":        bets,
        "wins":        wins,
        "win_rate":    round(win_rate, 4),
        "total_pnl":   round(pnl, 4),
        "avg_ev":      round(avg_ev, 4),   # positive = good
        "calib_mae":   round(calib_mae, 4),
        "deployable":  avg_ev > 0 and bets >= 50,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=90)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--force",  action="store_true", help="Deploy even if backtest fails")
    args = parser.parse_args()

    if not CANDLE_CSV.exists():
        print(f"ERROR: {CANDLE_CSV} not found — run fetch_historical_candles.py first")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {args.days}-day candle dataset...")
    df = load_and_engineer(CANDLE_CSV, args.days)
    print(f"  {len(df):,} rows  "
          f"({datetime.fromtimestamp(df['timestampms'].iloc[0]/1000).date()} → "
          f"{datetime.fromtimestamp(df['timestampms'].iloc[-1]/1000).date()})")
    print(f"  1-min delta: mean={df['target_delta'].mean():.2f}  std={df['target_delta'].std():.2f}")
    print(f"  Price range: ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")

    # ── Chronological split ───────────────────────────────────────────────────
    split = int(len(df) * 0.85)
    df_train = df.iloc[:split].copy()
    df_test  = df.iloc[split:].copy()
    print(f"  Train: {len(df_train):,}  Test: {len(df_test):,}  (chronological)")

    # ── Build sequences ───────────────────────────────────────────────────────
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train, y_train = make_sequences(df_train, scaler_x, scaler_y, fit=True)
    X_test,  y_test  = make_sequences(df_test,  scaler_x, scaler_y, fit=False)
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    # ── GPU setup ─────────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    strategy = tf.distribute.get_strategy()
    if len(gpus) > 1:
        try:
            s = tf.distribute.MirroredStrategy()
            with s.scope():
                tf.Variable(1.0)
            strategy = s
        except Exception as e:
            print(f"  MirroredStrategy failed ({e}), single GPU")
    print(f"  GPUs: {len(gpus)}  Strategy: {strategy.__class__.__name__}")

    # ── Train ─────────────────────────────────────────────────────────────────
    with strategy.scope():
        model = build_model(len(FEATURE_COLS))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\nTraining {args.epochs} epochs on {len(X_train):,} sequences...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    preds_s = model.predict(X_test, verbose=0)
    preds   = scaler_y.inverse_transform(preds_s).flatten()
    y_true  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = float(mean_absolute_error(y_true, preds))
    bias = float(np.mean(preds - y_true))
    print(f"\nTest MAE (delta):  ${mae:.2f}/min")
    print(f"Prediction bias:   ${bias:+.2f}/min  (near 0 = good)")
    print(f"Naive baseline:    ${df_test['target_delta'].abs().mean():.2f}/min  (mean-abs-delta)")

    # ── Backtest ──────────────────────────────────────────────────────────────
    print("\nRunning Kalshi backtest on test set...")
    bt = backtest(df_test, preds, mae)
    print(f"  Bets simulated:  {bt['bets']:,}")
    print(f"  Win rate:        {bt['win_rate']:.1%}")
    print(f"  Total P&L:       {bt['total_pnl']:+.2f} units")
    print(f"  Avg EV per bet:  {bt['avg_ev']:+.4f}")
    print(f"  Calibration MAE: {bt['calib_mae']:.4f}")
    print(f"  Deployable:      {'YES' if bt['deployable'] else 'NO'}")

    # ── Save (only if backtest passes or --force) ─────────────────────────────
    prev_mae = float("inf")
    if METRICS_OUT.exists():
        try:
            prev_mae = json.load(open(METRICS_OUT)).get("mae", float("inf"))
        except Exception:
            pass

    metrics = {
        "trained_at":      datetime.now().isoformat(),
        "model_type":      "delta",
        "mae":             mae,
        "bias":            bias,
        "seq_len":         SEQ_LEN,
        "feature_cols":    FEATURE_COLS,
        "days_of_data":    args.days,
        "epochs_run":      len(history.history["loss"]),
        "backtest":        bt,
        "chronological_split": True,
    }

    if bt["deployable"] or args.force:
        # Save candidate then atomically move
        cand_model = SCRIPT_DIR / "production_lstm_model_candidate.keras"
        cand_sx    = SCRIPT_DIR / "scaler_x_candidate.pkl"
        cand_sy    = SCRIPT_DIR / "scaler_y_candidate.pkl"

        model.save(str(cand_model))
        with open(cand_sx, "wb") as f: pickle.dump(scaler_x, f)
        with open(cand_sy, "wb") as f: pickle.dump(scaler_y, f)

        shutil.move(str(cand_model), str(MODEL_OUT))
        shutil.move(str(cand_sx),    str(SCALER_X_OUT))
        shutil.move(str(cand_sy),    str(SCALER_Y_OUT))
        with open(METRICS_OUT, "w") as f:
            json.dump(metrics, f, indent=2)

        reason = "backtest passed" if bt["deployable"] else "--force flag"
        print(f"\n✓ Deployed to production ({reason})")
        print(f"  Model: {MODEL_OUT}")
        print(f"  MAE: ${mae:.2f}/min  Bias: ${bias:+.2f}/min")
    else:
        with open(SCRIPT_DIR / "training_metrics_candidate.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✗ NOT deployed — backtest failed (avg_ev={bt['avg_ev']:+.4f})")
        print(f"  Candidate metrics saved to training_metrics_candidate.json")
        print(f"  Use --force to deploy anyway")


if __name__ == "__main__":
    main()
