"""
Fetch Kalshi settlements + fills, join to auto_trader.db trades, compute per-trade PnL.
Writes results into a new `trade_outcomes` table and prints summary stats by side and direction.
"""

import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.home() / "programs" / "gemini_trader"))
from kalshi_auth import KalshiSigner

BASE = "https://api.elections.kalshi.com/trade-api/v2"
DB   = Path.home() / ".local" / "share" / "kalshi-tui" / "auto_trader.db"


def fetch_paginated(signer, path, params=None, key=None):
    """Generic paginated fetch for Kalshi v2 endpoints that use `cursor`."""
    items = []
    cursor = None
    params = dict(params or {})
    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
        qs = "&".join(f"{k}={v}" for k, v in q.items())
        url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
        r = signer.get(url)
        if r.status_code != 200:
            print(f"  ! {path} -> {r.status_code}: {r.text[:200]}", file=sys.stderr)
            break
        data = r.json()
        batch = data.get(key, [])
        items.extend(batch)
        cursor = data.get("cursor") or None
        if not cursor or not batch:
            break
    return items


def main():
    signer = KalshiSigner.from_config()

    print("Fetching settlements (paginated)...")
    settlements = fetch_paginated(signer, "/portfolio/settlements",
                                  params={"limit": 1000}, key="settlements")
    print(f"  got {len(settlements)} settlement rows")

    print("Fetching fills (paginated)...")
    fills = fetch_paginated(signer, "/portfolio/fills",
                            params={"limit": 1000}, key="fills")
    print(f"  got {len(fills)} fill rows")

    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS settlements_raw (
            ticker TEXT PRIMARY KEY,
            market_result TEXT,
            yes_total_cost REAL,
            no_total_cost REAL,
            revenue REAL,
            settled_time TEXT,
            raw_json TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fills_raw (
            trade_id TEXT PRIMARY KEY,
            order_id TEXT,
            ticker TEXT,
            side TEXT,
            action TEXT,
            count INTEGER,
            yes_price INTEGER,
            no_price INTEGER,
            is_taker INTEGER,
            created_time TEXT,
            raw_json TEXT
        )
    """)

    for s in settlements:
        con.execute("INSERT OR REPLACE INTO settlements_raw VALUES (?,?,?,?,?,?,?)", (
            s.get("ticker"),
            s.get("market_result"),
            s.get("yes_total_cost"),
            s.get("no_total_cost"),
            s.get("revenue"),
            s.get("settled_time"),
            json.dumps(s),
        ))
    for f in fills:
        con.execute("INSERT OR REPLACE INTO fills_raw VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            f.get("trade_id"),
            f.get("order_id"),
            f.get("ticker"),
            f.get("side"),
            f.get("action"),
            f.get("count"),
            f.get("yes_price"),
            f.get("no_price"),
            int(bool(f.get("is_taker"))),
            f.get("created_time"),
            json.dumps(f),
        ))
    con.commit()

    # Build per-trade outcome view:
    # A YES contract pays $1 if market_result == "yes", $0 otherwise.
    # A NO contract pays $1 if market_result == "no", $0 otherwise.
    # PnL (dollars) = payout - cost_dollars.
    con.execute("DROP TABLE IF EXISTS trade_outcomes")
    con.execute("""
        CREATE TABLE trade_outcomes AS
        SELECT
            t.id,
            t.run_id,
            t.placed_at,
            t.ticker,
            t.strike,
            t.side,
            t.count,
            t.price_cents,
            t.cost_dollars,
            t.edge,
            r.current_price,
            (t.strike - r.current_price) AS strike_offset,
            s.market_result,
            CASE
              WHEN s.market_result IS NULL THEN NULL
              WHEN t.side='YES' AND s.market_result='yes' THEN t.count * 1.0
              WHEN t.side='NO'  AND s.market_result='no'  THEN t.count * 1.0
              ELSE 0.0
            END AS payout,
            CASE
              WHEN s.market_result IS NULL THEN NULL
              WHEN t.side='YES' AND s.market_result='yes' THEN (t.count * 1.0) - t.cost_dollars
              WHEN t.side='NO'  AND s.market_result='no'  THEN (t.count * 1.0) - t.cost_dollars
              ELSE -t.cost_dollars
            END AS pnl,
            CASE
              WHEN s.market_result IS NULL THEN 'unsettled'
              WHEN (t.side='YES' AND s.market_result='yes') OR (t.side='NO' AND s.market_result='no') THEN 'win'
              ELSE 'loss'
            END AS outcome
        FROM trades t
        JOIN runs r ON t.run_id = r.id
        LEFT JOIN settlements_raw s ON s.ticker = t.ticker
        WHERE t.status='placed'
    """)
    con.commit()

    def q(sql):
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.execute(sql).description]
        return cols, rows

    def render(title, cols, rows):
        print(f"\n### {title}")
        widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c))
                  for i, c in enumerate(cols)]
        print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))
        print("-+-".join("-"*w for w in widths))
        for r in rows:
            print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))

    print("\n" + "=" * 70)
    print(" PER-TRADE PnL ANALYSIS")
    print("=" * 70)

    cols, rows = q("""
        SELECT outcome, COUNT(*) n, ROUND(SUM(cost_dollars),2) cost,
               ROUND(SUM(pnl),2) pnl, ROUND(AVG(pnl),3) avg_pnl
        FROM trade_outcomes GROUP BY outcome ORDER BY outcome
    """)
    render("Settled vs unsettled totals", cols, rows)

    cols, rows = q("""
        SELECT side, outcome, COUNT(*) n,
               ROUND(AVG(cost_dollars),2) avg_cost,
               ROUND(SUM(cost_dollars),2) total_cost,
               ROUND(SUM(pnl),2) pnl,
               ROUND(AVG(pnl),3) avg_pnl
        FROM trade_outcomes
        WHERE outcome != 'unsettled'
        GROUP BY side, outcome
        ORDER BY side, outcome
    """)
    render("By side x outcome", cols, rows)

    cols, rows = q("""
        SELECT side,
               COUNT(*) n,
               SUM(CASE WHEN outcome='win'  THEN 1 ELSE 0 END) wins,
               SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) losses,
               ROUND(100.0*SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END)/COUNT(*),1) win_pct,
               ROUND(AVG(cost_dollars),2) avg_cost_placed,
               ROUND(AVG(CASE WHEN outcome='win'  THEN cost_dollars END),2) avg_cost_when_win,
               ROUND(AVG(CASE WHEN outcome='loss' THEN cost_dollars END),2) avg_cost_when_loss,
               ROUND(SUM(pnl),2) total_pnl
        FROM trade_outcomes
        WHERE outcome != 'unsettled'
        GROUP BY side
    """)
    render("Side summary — do we bet bigger on losing side?", cols, rows)

    cols, rows = q("""
        WITH d AS (
          SELECT *,
            CASE
              WHEN side='YES' AND strike >  current_price THEN 'BULLISH_YES_above'
              WHEN side='YES' AND strike <= current_price THEN 'BEARISH_YES_below'
              WHEN side='NO'  AND strike >  current_price THEN 'BEARISH_NO_above'
              WHEN side='NO'  AND strike <= current_price THEN 'BULLISH_NO_below'
            END AS direction
          FROM trade_outcomes WHERE outcome != 'unsettled'
        )
        SELECT direction,
               COUNT(*) n,
               SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) wins,
               ROUND(100.0*SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END)/COUNT(*),1) win_pct,
               ROUND(AVG(cost_dollars),2) avg_cost,
               ROUND(SUM(pnl),2) total_pnl,
               ROUND(AVG(pnl),3) avg_pnl
        FROM d GROUP BY direction ORDER BY direction
    """)
    render("By directional bias (bullish vs bearish)", cols, rows)

    cols, rows = q("""
        SELECT
          CASE
            WHEN cost_dollars < 1.50           THEN 'small <$1.50'
            WHEN cost_dollars BETWEEN 1.50 AND 3.00 THEN 'mid $1.50-3'
            ELSE 'large >$3'
          END AS bucket,
          side,
          COUNT(*) n,
          ROUND(100.0*SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END)/COUNT(*),1) win_pct,
          ROUND(SUM(pnl),2) total_pnl
        FROM trade_outcomes WHERE outcome != 'unsettled'
        GROUP BY bucket, side
        ORDER BY side, bucket
    """)
    render("Size bucket x side — bigger bets, worse outcomes?", cols, rows)


if __name__ == "__main__":
    main()
