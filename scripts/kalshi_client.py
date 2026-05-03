import requests
import pandas as pd
from datetime import datetime

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def get_crypto_markets(series_ticker="KXBTCD"):
    """Fetch open crypto markets from Kalshi."""
    url = f"{BASE_URL}/markets?status=open&series_ticker={series_ticker}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('markets', [])
    except Exception as e:
        print(f"Error fetching Kalshi markets: {e}")
        return []

def analyze_sentiment(series_ticker="KXBTCD"):
    """Analyze market sentiment based on Yes/No prices across strikes."""
    markets = get_crypto_markets(series_ticker)
    if not markets:
        return None
    
    analysis = []
    for m in markets:
        try:
            analysis.append({
                'ticker': m.get('ticker'),
                'strike': float(m.get('floor_strike', 0)),
                'yes_price': float(m.get('yes_ask_dollars', 0)),
                'no_price': float(m.get('no_ask_dollars', 0)),
                'volume_24h': float(m.get('volume_24h_fp', 0)),
                'title': m.get('title')
            })
        except (ValueError, TypeError):
            continue
    
    df = pd.DataFrame(analysis)
    if not df.empty:
        df = df.sort_values('strike').reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Fetching Kalshi Bitcoin Market Sentiment...")
    df = analyze_sentiment("KXBTCD")
    if df is not None and not df.empty:
        print("\n--- Current Kalshi BTC Sentiment (Strike Prices) ---")
        print(df[['strike', 'yes_price', 'no_price', 'volume_24h']].head(10))
    else:
        print("No active markets found.")
