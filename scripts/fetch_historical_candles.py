#!/usr/bin/env python3
"""
fetch_historical_candles.py — Build a large 1-minute OHLCV training dataset.

Strategy:
  1. Binance public klines API (BTCUSDT, 1m) — paginated, no auth, ~same price as Gemini
  2. Append live Gemini candles (most recent 24h, authoritative for inference scaler)
  3. Deduplicate & save to time_series/historical_candle_data_btcusd_full.csv

Usage:
  python fetch_historical_candles.py [--days N]   (default: 365)
"""

import argparse
import time
import sys
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GEMINI_CANDLES_URL   = "https://api.gemini.com/v2/candles/btcusd/1m"
OUT_FILE = Path(__file__).parent / "time_series" / "historical_candle_data_btcusd_full.csv"
LIMIT = 300   # Coinbase max per request (1m granularity)


def fetch_coinbase_chunk(start_s: int, end_s: int) -> list:
    """Fetch up to LIMIT 1m bars from Coinbase Exchange [start_s, end_s]."""
    params = {"granularity": 60, "start": start_s, "end": end_s}
    headers = {"Accept": "application/json"}
    for attempt in range(5):
        try:
            r = requests.get(COINBASE_CANDLES_URL, params=params,
                             headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "message" in data:
                raise ValueError(f"API error: {data['message']}")
            return data
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise


def coinbase_to_df(raw: list) -> pd.DataFrame:
    """Convert Coinbase candle list to DataFrame with model schema.

    Coinbase row: [time_seconds, low, high, open, close, volume]
    Returned in DESC order — we sort ascending.
    """
    if not raw:
        return pd.DataFrame()
    rows = []
    for k in raw:
        ts_s = int(k[0])
        rows.append({
            "timestampms": ts_s * 1000,
            "open":   float(k[3]),
            "high":   float(k[2]),
            "low":    float(k[1]),
            "close":  float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(rows).sort_values("timestampms").reset_index(drop=True)
    df["timestamp"] = df["timestampms"]
    df["price"]     = df["close"]
    df["amount"]    = df["volume"]
    return df


def fetch_coinbase_range(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Paginate Coinbase candles API over the full date range."""
    start_s = int(start_dt.timestamp())
    end_s   = int(end_dt.timestamp())
    chunk_s = LIMIT * 60   # 300 minutes per chunk

    all_frames = []
    cursor = start_s
    total  = 0

    print(f"Fetching Coinbase 1m candles: {start_dt.date()} → {end_dt.date()}")
    while cursor < end_s:
        chunk_end = min(cursor + chunk_s, end_s)
        raw = fetch_coinbase_chunk(cursor, chunk_end)
        if not raw:
            cursor = chunk_end + 60
            continue
        df_chunk = coinbase_to_df(raw)
        if df_chunk.empty:
            cursor = chunk_end + 60
            continue
        all_frames.append(df_chunk)
        total += len(df_chunk)
        last_ts_s = df_chunk["timestampms"].iloc[-1] // 1000
        cursor = chunk_end + 60

        pct = 100 * (cursor - start_s) / (end_s - start_s)
        last_dt = datetime.fromtimestamp(last_ts_s).strftime("%Y-%m-%d %H:%M")
        print(f"\r  {pct:5.1f}%  {total:,} bars  last={last_dt}",
              end="", flush=True)

        # Coinbase rate limit: 10 req/s — be conservative
        time.sleep(0.12)

    print()
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def fetch_gemini_recent() -> pd.DataFrame:
    """Fetch last 24h of 1m candles from Gemini (authoritative source)."""
    print("Fetching Gemini live 1m candles (last 24h)...")
    r = requests.get(GEMINI_CANDLES_URL, timeout=15)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=["timestampms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = df["timestampms"]
    df["price"] = df["close"]
    df["amount"] = df["volume"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Build large 1m BTC training dataset")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default 365)")
    parser.add_argument("--out", type=str, default=str(OUT_FILE), help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)

    # Fetch Coinbase historical data
    df_coinbase = fetch_coinbase_range(start, now - timedelta(hours=1))

    # Fetch Gemini for the last 24h (most authoritative for scaler alignment)
    df_gemini = fetch_gemini_recent()

    # Combine, deduplicate by timestamp (prefer Gemini for recent overlap)
    df_all = pd.concat([df_coinbase, df_gemini], ignore_index=True)
    df_all = df_all.sort_values("timestampms")

    # Drop duplicates: keep last (Gemini) for any overlapping minute
    df_all["minute_bucket"] = df_all["timestampms"] // 60_000
    df_all = df_all.drop_duplicates(subset="minute_bucket", keep="last")
    df_all = df_all.drop(columns=["minute_bucket"]).reset_index(drop=True)

    # Save
    cols = ["timestampms", "open", "high", "low", "close", "volume",
            "timestamp", "price", "amount"]
    df_all[cols].to_csv(out_path, index=False)

    print(f"\nSaved {len(df_all):,} 1m bars to {out_path}")
    print(f"  Oldest: {datetime.fromtimestamp(df_all['timestampms'].iloc[0]/1000)}")
    print(f"  Newest: {datetime.fromtimestamp(df_all['timestampms'].iloc[-1]/1000)}")
    print(f"  Span:   {(df_all['timestampms'].iloc[-1] - df_all['timestampms'].iloc[0]) / 86400_000:.1f} days")
    print(f"  Price range: ${df_all['price'].min():,.0f} – ${df_all['price'].max():,.0f}")


if __name__ == "__main__":
    main()
