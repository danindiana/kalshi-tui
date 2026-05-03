"""
orderbook_features.py — Live Kalshi orderbook microstructure features.

Queries Kalshi's REST API for a single market ticker's resting-order book and
returns bid-ask imbalance + spread metrics for use as a conviction gate in
auto_trader.py.

Kalshi orderbook schema  (/trade-api/v2/markets/{ticker}/orderbook):
  yes: [[price_cents, size], ...]  — resting bids to buy YES (highest first)
  no:  [[price_cents, size], ...]  — resting bids to buy NO  (highest first)

Interpretation:
  - Each YES bid says "I'll buy YES at X¢" — demand for BTC above strike
  - Each NO bid says "I'll buy NO at X¢" — demand for BTC below strike
  - yes_imbalance > 0.5: crowd leans bullish at this strike
  - yes_imbalance < 0.5: crowd leans bearish at this strike

Gate semantics (see imbalance_allows_bet):
  Fail-open on any data unavailability — if we can't fetch the book we don't
  suppress the bet, we just trade on the classifier alone (status quo).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional

_GEMINI_TRADER_DIR = "/home/jeb/programs/gemini_trader"
if _GEMINI_TRADER_DIR not in sys.path:
    sys.path.insert(0, _GEMINI_TRADER_DIR)

from kalshi_auth import KalshiSigner

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class OrderbookSnapshot:
    ticker: str
    yes_volume: int       # total resting YES contracts across all price levels
    no_volume: int        # total resting NO contracts across all price levels
    yes_imbalance: float  # yes_volume / (yes_volume + no_volume); 0.5 = balanced
    best_yes_bid: int     # highest YES bid in cents (0 if book empty)
    best_no_bid: int      # highest NO bid in cents (0 if book empty)
    market_spread: int    # 100 - best_yes_bid - best_no_bid cents (lower = tighter)
    fetched_at: float     # unix timestamp of fetch


def fetch_orderbook(
    signer: KalshiSigner,
    ticker: str,
    timeout: float = 5.0,
) -> Optional[OrderbookSnapshot]:
    """
    Fetch live orderbook for a single Kalshi market ticker.

    Returns None on any API error, timeout, or empty book.
    Callers must treat None as "data unavailable" and fail-open.
    """
    url = f"{KALSHI_BASE}/markets/{ticker}/orderbook"
    try:
        r = signer.get(url, timeout=timeout)
        r.raise_for_status()
        ob = r.json().get("orderbook", {})
    except Exception:
        return None

    yes_bids: list = ob.get("yes", [])
    no_bids: list  = ob.get("no",  [])

    if not yes_bids and not no_bids:
        return None

    yes_volume = sum(int(size) for _, size in yes_bids)
    no_volume  = sum(int(size) for _, size in no_bids)
    total = yes_volume + no_volume
    if total == 0:
        return None

    yes_imbalance = yes_volume / total

    # Kalshi sorts bids highest-price-first, so index 0 is the best bid.
    best_yes_bid = int(yes_bids[0][0]) if yes_bids else 0
    best_no_bid  = int(no_bids[0][0])  if no_bids  else 0

    # In a perfectly liquid book, best_yes_bid + best_no_bid ≈ 99¢
    # (Kalshi rejects 100¢ orders, so the tightest possible spread is 1¢).
    # A wide spread (>30¢) signals a thin, unreliable book.
    market_spread = 100 - best_yes_bid - best_no_bid

    return OrderbookSnapshot(
        ticker=ticker,
        yes_volume=yes_volume,
        no_volume=no_volume,
        yes_imbalance=yes_imbalance,
        best_yes_bid=best_yes_bid,
        best_no_bid=best_no_bid,
        market_spread=market_spread,
        fetched_at=time.time(),
    )


# Default gate threshold — tunable via YES_IMBALANCE_THRESHOLD env var in service.
DEFAULT_IMBALANCE_THRESHOLD = 0.65
THIN_BOOK_SPREAD_CENTS = 30  # spread wider than this → skip gate (noisy signal)


def imbalance_allows_bet(
    snap: Optional[OrderbookSnapshot],
    side: str,
    threshold: float = DEFAULT_IMBALANCE_THRESHOLD,
) -> tuple[bool, str]:
    """
    Returns (allowed: bool, reason: str).

    Rule: suppress a bet when the crowd is *heavily positioned against* it.
      - NO bet suppressed when YES-heavy (imbalance > threshold):
        sophisticated traders are buying YES at this strike (expect BTC above) —
        do not fade them with a NO bet.
      - YES bet suppressed when NO-heavy (imbalance < 1 - threshold):
        crowd is buying NO at this strike (expect BTC below) — do not fade.

    Fail-open conditions (always return allowed=True):
      - snap is None (fetch failed / timeout)
      - market_spread > THIN_BOOK_SPREAD_CENTS (too illiquid to trust)
    """
    if snap is None:
        return True, "ob_unavailable (fail-open)"

    if snap.market_spread > THIN_BOOK_SPREAD_CENTS:
        return True, (
            f"ob_thin: spread={snap.market_spread}¢ > {THIN_BOOK_SPREAD_CENTS}¢ "
            f"(book illiquid, gate bypassed)"
        )

    side_up = side.upper()

    if side_up == "NO" and snap.yes_imbalance > threshold:
        no_imb = 1.0 - snap.yes_imbalance
        return False, (
            f"ob_gate: YES-heavy (YES={snap.yes_imbalance:.0%} NO={no_imb:.0%}) "
            f"> {threshold:.0%} — crowd buying YES, suppressing NO bet "
            f"[yes_vol={snap.yes_volume} no_vol={snap.no_volume}]"
        )

    if side_up == "YES" and snap.yes_imbalance < (1.0 - threshold):
        no_imb = 1.0 - snap.yes_imbalance
        return False, (
            f"ob_gate: NO-heavy (NO={no_imb:.0%} YES={snap.yes_imbalance:.0%}) "
            f"> {threshold:.0%} — crowd buying NO, suppressing YES bet "
            f"[yes_vol={snap.yes_volume} no_vol={snap.no_volume}]"
        )

    return True, (
        f"ob_ok (imbalance={snap.yes_imbalance:.0%}, spread={snap.market_spread}¢, "
        f"yes_vol={snap.yes_volume}, no_vol={snap.no_volume})"
    )


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch live Kalshi orderbook features for one or more tickers"
    )
    parser.add_argument(
        "ticker", nargs="?",
        help="Specific ticker, e.g. KXBTCD-26APR2917-T75999.99. "
             "Omit to auto-discover the nearest BTC settlement window."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_IMBALANCE_THRESHOLD,
        help=f"Imbalance gate threshold (default {DEFAULT_IMBALANCE_THRESHOLD})"
    )
    args = parser.parse_args()

    signer = KalshiSigner.from_config()

    if args.ticker:
        tickers = [args.ticker]
    else:
        r = signer.get(f"{KALSHI_BASE}/markets?status=open&series_ticker=KXBTCD&limit=10")
        r.raise_for_status()
        tickers = [m["ticker"] for m in r.json().get("markets", [])][:5]
        if not tickers:
            print("No open KXBTCD markets found.")
            sys.exit(1)
        print(f"Auto-discovered {len(tickers)} tickers from nearest settlement window.\n")

    for ticker in tickers:
        snap = fetch_orderbook(signer, ticker)
        if snap is None:
            print(f"{ticker}: no data (empty book or API error)")
            continue

        no_allowed, no_reason = imbalance_allows_bet(snap, "NO", args.threshold)
        yes_allowed, yes_reason = imbalance_allows_bet(snap, "YES", args.threshold)

        print(
            f"{ticker}\n"
            f"  YES vol={snap.yes_volume:5d}  NO vol={snap.no_volume:5d}  "
            f"imbalance={snap.yes_imbalance:.1%}  "
            f"spread={snap.market_spread}¢  "
            f"best_yes={snap.best_yes_bid}¢  best_no={snap.best_no_bid}¢\n"
            f"  NO  bet → {'ALLOW  ' if no_allowed  else 'SUPPRESS'} — {no_reason}\n"
            f"  YES bet → {'ALLOW  ' if yes_allowed else 'SUPPRESS'} — {yes_reason}\n"
        )
