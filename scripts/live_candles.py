"""
live_candles.py — Fetch 1440 1-minute BTC/USD candles from Gemini production API.

Returns a DataFrame with columns matching the model's expected input:
  timestamp (ms), open, high, low, close, volume, price (=close), amount (=volume)
"""

import requests
import pandas as pd
import numpy as np


GEMINI_CANDLES_URL = "https://api.gemini.com/v2/candles/btcusd/1m"


def fetch_live_candles() -> pd.DataFrame:
    """Fetch 1440 1m candles from Gemini and return a model-ready DataFrame."""
    resp = requests.get(GEMINI_CANDLES_URL, timeout=15)
    resp.raise_for_status()
    raw = resp.json()  # list of [ts_ms, open, high, low, close, volume]

    df = pd.DataFrame(raw, columns=["timestampms", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestampms").reset_index(drop=True)

    # Aliases expected by predict_json.py model code
    df["timestamp"] = df["timestampms"]
    df["price"] = df["close"]
    df["amount"] = df["volume"]

    return df


if __name__ == "__main__":
    df = fetch_live_candles()
    print(f"Fetched {len(df)} candles")
    print(f"Newest:  ts={df['timestampms'].iloc[-1]}  price=${df['price'].iloc[-1]:,.2f}")
    print(f"Oldest:  ts={df['timestampms'].iloc[0]}   price=${df['price'].iloc[0]:,.2f}")
