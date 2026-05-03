#!/usr/bin/env python3
"""
paper_trader.py — Paper trading simulation for the Kalshi BTC auto-trader.

Runs the classifier pipeline against live market data with virtual dollars.
No real orders are placed. Settles up at the end using actual Kalshi results.

Usage:
  paper_trader.py --balance 100 --max-cycles 6 --max-trades 20
"""
import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPTS_DIR       = Path(__file__).parent
WORKSPACE_DIR     = SCRIPTS_DIR.parent
GEMINI_TRADER_DIR = "/home/jeb/programs/gemini_trader"
DB_PATH           = Path.home() / ".local" / "share" / "kalshi-tui" / "auto_trader.db"
KALSHI_BASE       = "https://api.elections.kalshi.com/trade-api/v2"

sys.path.insert(0, GEMINI_TRADER_DIR)

# Load the workspace auto_trader.py by absolute path so we always get the
# version with the 4-arg run_classifier(signer, ...) and OB features, even
# when the older 3-arg gemini_trader copy is earlier on sys.path.
import importlib.util as _ilu
_at_spec = _ilu.spec_from_file_location("auto_trader", str(SCRIPTS_DIR / "auto_trader.py"))
at = _ilu.module_from_spec(_at_spec)
sys.modules["auto_trader"] = at
_at_spec.loader.exec_module(at)

# ── DB schema additions ───────────────────────────────────────────────────────

_PAPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    initial_bal  REAL NOT NULL,
    max_cycles   INTEGER,
    max_trades   INTEGER,
    completed    INTEGER DEFAULT 0,
    final_bal    REAL,
    total_trades INTEGER DEFAULT 0,
    wins         INTEGER DEFAULT 0,
    losses       INTEGER DEFAULT 0,
    pending      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES paper_sessions(id),
    cycle_num    INTEGER,
    placed_at    TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    strike       REAL NOT NULL,
    side         TEXT NOT NULL,
    count        INTEGER NOT NULL,
    price_cents  INTEGER NOT NULL,
    cost_dollars REAL NOT NULL,
    edge         REAL,
    status       TEXT DEFAULT 'pending',
    pnl          REAL
);
"""


def _open_paper_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(_PAPER_SCHEMA)
    con.commit()
    return con


def _resolve_settlements(con: sqlite3.Connection, session_id: int, signer,
                         paper_trades: list) -> tuple[int, int, int]:
    """
    Check Kalshi market results for each paper trade ticker.
    Updates paper_trades rows with win/loss/pending and pnl.
    Returns (wins, losses, pending).
    """
    wins = losses = pending = 0
    resolved_tickers: dict[str, str] = {}  # ticker → "yes" | "no" | ""

    for trade in paper_trades:
        ticker      = trade["ticker"]
        side        = trade["side"].lower()
        count       = trade["count"]
        cost        = trade["cost_dollars"]
        trade_id    = trade["id"]

        if ticker not in resolved_tickers:
            try:
                r = signer.get(f"{KALSHI_BASE}/markets/{ticker}")
                r.raise_for_status()
                mkt = r.json().get("market", {})
                resolved_tickers[ticker] = (mkt.get("result") or "").lower()
            except Exception as exc:
                print(f"  [WARN] Could not fetch {ticker}: {exc}")
                resolved_tickers[ticker] = ""

        result = resolved_tickers[ticker]

        if result in ("yes", "no"):
            if result == side:
                pnl    = round(count * 1.0 - cost, 4)   # $1/contract payout
                status = "win"
                wins  += 1
            else:
                pnl    = round(-cost, 4)
                status = "loss"
                losses += 1
            print(f"  [{status.upper():4s}] {ticker}  {side.upper()} → result={result.upper()}  P&L=${pnl:+.2f}")
        else:
            # Not yet settled; mark-to-market via current yes_ask
            try:
                mkt      = signer.get(f"{KALSHI_BASE}/markets/{ticker}").json().get("market", {})
                yes_ask  = float(mkt.get("yes_ask", 0) or 0)
                no_ask   = float(mkt.get("no_ask",  0) or 0)
                cur_price = yes_ask if side == "yes" else no_ask
                pnl       = round(count * cur_price - cost, 4)
            except Exception:
                pnl = None
            status = "pending"
            pending += 1
            print(f"  [PEND] {ticker}  {side.upper()} not yet settled  mtm=${pnl:+.2f}" if pnl is not None else
                  f"  [PEND] {ticker}  {side.upper()} not yet settled")

        con.execute(
            "UPDATE paper_trades SET status=?, pnl=? WHERE id=?",
            (status, pnl, trade_id),
        )

    con.commit()
    return wins, losses, pending


def _mtm_snapshot(paper_trades: list, signer) -> tuple[float, float, float]:
    """
    Fetch current market prices for all unique tickers in paper_trades and
    compute aggregate (total_cost, total_mtm, total_unrealized).
    Used for the settlement wait-loop heartbeat.
    """
    price_cache: dict[str, dict] = {}
    total_cost = total_mtm = 0.0

    for pos in paper_trades:
        if pos.get("status") not in (None, "pending"):
            continue   # already settled, skip
        ticker = pos["ticker"]
        side   = pos["side"].lower()
        count  = pos["count"]
        cost   = pos["cost_dollars"]
        total_cost += cost

        if ticker not in price_cache:
            try:
                r   = signer.get(f"{KALSHI_BASE}/markets/{ticker}")
                r.raise_for_status()
                mkt = r.json().get("market", {})
                price_cache[ticker] = {
                    "yes": float(mkt.get("yes_ask", 0) or 0),
                    "no":  float(mkt.get("no_ask",  0) or 0),
                }
            except Exception:
                price_cache[ticker] = {"yes": 0.0, "no": 0.0}

        cur  = price_cache[ticker]["yes"] if side == "yes" else price_cache[ticker]["no"]
        total_mtm += count * cur

    return total_cost, total_mtm, total_mtm - total_cost


def _print_portfolio_snapshot(
    open_positions: list,
    markets: list,
    mins: float,
    virtual_balance: float,
    initial_balance: float,
    cycle_num: int,
) -> None:
    """Print a live portfolio snapshot after each cycle."""
    if not open_positions:
        return

    # Build ticker → current prices lookup from the markets already fetched
    price_map: dict[str, dict] = {}
    for m in markets:
        price_map[m["ticker"]] = {
            "yes": m["yes_price"],
            "no":  m["no_price"],
        }

    total_cost   = 0.0
    total_mtm    = 0.0
    total_exp_pnl = 0.0

    rows = []
    for pos in open_positions:
        ticker      = pos["ticker"]
        side        = pos["side"].upper()
        count       = pos["count"]
        cost        = pos["cost_dollars"]
        model_prob  = pos.get("model_prob", None)
        entry_price = pos["price_cents"] / 100.0

        prices = price_map.get(ticker)
        if prices:
            cur_price = prices["yes"] if side == "YES" else prices["no"]
            mtm_value = count * cur_price
        else:
            cur_price = entry_price
            mtm_value = cost   # can't price it, show flat

        unrlzd = mtm_value - cost

        # Model's expected profit at entry: (model_prob × payout) - cost
        if model_prob is not None:
            exp_payout = count * model_prob
            exp_pnl    = exp_payout - cost
        else:
            exp_pnl = (pos.get("edge", 0) or 0) * count   # rough fallback

        total_cost    += cost
        total_mtm     += mtm_value
        total_exp_pnl += exp_pnl

        rows.append((side, pos["strike"], count, pos["price_cents"],
                     cur_price, cost, mtm_value, unrlzd, exp_pnl))

    total_unrlzd = total_mtm - total_cost
    settle_str   = f"{mins:.0f} min" if mins > 1 else "< 1 min"

    print(f"\n  ┌─ Portfolio Snapshot — cycle {cycle_num}  ({settle_str} to settlement) {'─'*10}")
    for side, strike, count, entry_c, cur_price, cost, mtm, unrlzd, exp_pnl in rows:
        cur_c   = round(cur_price * 100)
        sign    = "+" if unrlzd >= 0 else ""
        exp_s   = f"{sign}{exp_pnl:+.2f}" if unrlzd >= 0 else f"{exp_pnl:+.2f}"
        row_str = (
            f"  │  {side:3s} ${strike:>8,.0f}  "
            f"{count:2d}ct  entry={entry_c}¢ → cur={cur_c}¢  "
            f"cost=${cost:.2f}  mtm=${mtm:.2f}  "
            f"unrlzd=${unrlzd:+.2f}  expctd=${exp_pnl:+.2f}"
        )
        print(row_str)
    print(
        f"  │  {'─'*56}\n"
        f"  │  TOTAL  cost=${total_cost:.2f}  mtm=${total_mtm:.2f}  "
        f"unrlzd=${total_unrlzd:+.2f}  expctd=${total_exp_pnl:+.2f}\n"
        f"  │  Cash: ${virtual_balance:.2f}  "
        f"Portfolio: ${virtual_balance + total_cost:.2f}  "
        f"(started ${initial_balance:.2f})\n"
        f"  └{'─'*57}"
    )


def _write_session_doc(path: Path, ts: str, initial_bal: float, max_cycles: int,
                       max_trades: int, total_trades: int, wins: int, losses: int,
                       pending: int, net_pnl: float, final_bal: float,
                       trade_rows: list) -> None:
    pct = (net_pnl / initial_bal * 100) if initial_bal > 0 else 0.0
    lines = [
        f"# Paper Trading Session — {ts.replace('_', ' ')}",
        "",
        "## Configuration",
        f"- Starting balance: ${initial_bal:.2f}",
        f"- Max predict cycles: {max_cycles if max_cycles else 'unlimited'}",
        f"- Max trades: {max_trades if max_trades else 'unlimited'}",
        "",
        "## Results",
        f"- Trades placed: {total_trades}",
        f"- Settled: {wins + losses}  ({wins} wins / {losses} losses)",
        f"- Pending (awaiting settlement): {pending}",
        f"- Net P&L (settled only): ${net_pnl:+.2f}  ({pct:+.1f}%)",
        f"- Final balance: ${final_bal:.2f}",
        "",
        "## Trade Log",
        "",
        "| Cycle | Side | Strike | Count | Price | Cost | Edge | Status | P&L |",
        "|-------|------|--------|-------|-------|------|------|--------|-----|",
    ]
    for r in trade_rows:
        cycle, side, strike, count, price_c, cost, edge, status, pnl = r
        edge_s = f"{edge*100:+.1f}%" if edge is not None else "—"
        pnl_s  = f"${pnl:+.2f}"     if pnl  is not None else "—"
        lines.append(
            f"| {cycle} | {side} | ${strike:,.0f} | {count} | {price_c}¢ | "
            f"${cost:.2f} | {edge_s} | {status} | {pnl_s} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_paper_session(balance: float, max_cycles: int, max_trades: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Kalshi Paper Trader  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Starting balance: ${balance:.2f}  "
          f"Cycles: {max_cycles or '∞'}  "
          f"Max trades: {max_trades or '∞'}")
    print(f"{'='*60}")

    # Auth
    try:
        signer = at.KalshiSigner.from_config()
        print(f"[OK]   Auth: key_id={signer.key_id[:8]}…")
    except Exception as exc:
        print(f"[FAIL] Auth: {exc}")
        return

    # DB + paper schema
    con = _open_paper_db()
    cur = con.execute(
        "INSERT INTO paper_sessions (started_at, initial_bal, max_cycles, max_trades) "
        "VALUES (?,?,?,?)",
        (datetime.now().isoformat(), balance, max_cycles or None, max_trades or None),
    )
    session_id = cur.lastrowid
    con.commit()

    # Sizing mirrors auto_trader's ratchet, applied to the paper bankroll
    virtual_balance      = balance
    total_trades         = 0
    paper_trades         = []   # list of dicts for settlement resolution
    open_positions       = []   # same list, kept for live portfolio snapshots
    last_settlement_utc  = None # most recent settlement window observed

    cycle_num = 0
    while True:
        cycle_num += 1
        if max_cycles > 0 and cycle_num > max_cycles:
            print(f"\n[OK]   Max cycles reached ({max_cycles}).")
            break
        if max_trades > 0 and total_trades >= max_trades:
            print(f"\n[OK]   Max trades reached ({max_trades}).")
            break
        if virtual_balance < 0.50:
            print(f"\n[STOP] Virtual balance exhausted (${virtual_balance:.2f}).")
            break

        print(f"\n{'─'*60}")
        print(f"  Cycle {cycle_num}"
              + (f"/{max_cycles}" if max_cycles else "")
              + f"  |  Paper balance: ${virtual_balance:.2f}")
        print(f"{'─'*60}")

        can_trade, reason, settlement_utc, mins, markets = at.check_trading_window(signer)
        if not can_trade:
            print(f"[SKIP] {reason}")
            if cycle_num == 1:
                print("[FAIL] No trading window available on first cycle — exiting.")
                break
            continue

        last_settlement_utc = settlement_utc
        local_settle = settlement_utc.astimezone().strftime("%H:%M %Z")
        print(f"[OK]   Settlement: {local_settle} ({mins:.0f} min)  {len(markets)} strikes")

        try:
            print("[...] Running classifier inference…")
            clf = at.run_classifier(signer, settlement_utc, mins, markets)
        except Exception as exc:
            print(f"[SKIP] Classifier: {exc}")
            continue

        if clf is None:
            print("[SKIP] No classifier result.")
            continue

        opps = clf["opportunities"]
        print(
            f"[OK]   ${clf['current_price']:,.2f}  "
            f"RSI={clf['rsi']:.1f}  "
            f"vol=${clf['volatility']:,.0f}  "
            f"momentum=${clf['momentum']:+,.0f}  "
            f"{len(opps)} edge(s)"
        )

        if not opps:
            print("[OK]   No tradeable edges this cycle.")
            continue

        # Ratchet sizing against paper bankroll
        bankroll   = min(virtual_balance, at.MAX_BANKROLL)
        max_stake  = min(bankroll * at.MAX_STAKE_PCT, at.MAX_STAKE_DOLLARS)
        run_budget = min(bankroll * at.MAX_RUN_PCT,   at.MAX_RUN_DOLLARS)
        run_spent  = 0.0
        cycle_trades = 0

        for opp in opps:
            if max_trades > 0 and total_trades >= max_trades:
                break
            if run_spent >= run_budget:
                break
            if virtual_balance < 0.50:
                break

            contract_price = opp["yes_price"] if opp["side"] == "YES" else opp["no_price"]
            if contract_price <= 0:
                continue

            raw_stake = bankroll * opp["stake_pct"]
            stake     = min(raw_stake, max_stake, run_budget - run_spent, virtual_balance)
            stake     = max(stake, 0.50)

            count       = max(1, int(stake / contract_price))
            cost        = count * contract_price
            price_cents = round(contract_price * 100)

            if cost > virtual_balance:
                continue

            virtual_balance -= cost
            run_spent       += cost
            total_trades    += 1
            cycle_trades    += 1

            print(
                f"  [PAPER] {opp['side']:3s}  "
                f"${opp['strike']:,.0f}  "
                f"edge={opp['edge']*100:+.1f}%  "
                f"{count}ct @ {price_cents}¢  "
                f"${cost:.2f}  "
                f"(bal ${virtual_balance:.2f})"
            )

            cur = con.execute(
                """INSERT INTO paper_trades
                   (session_id, cycle_num, placed_at, ticker, strike, side,
                    count, price_cents, cost_dollars, edge)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, cycle_num, datetime.now().isoformat(),
                    opp["ticker"], opp["strike"], opp["side"],
                    count, price_cents, round(cost, 4), opp["edge"],
                ),
            )
            entry = {
                "id":           cur.lastrowid,
                "ticker":       opp["ticker"],
                "strike":       opp["strike"],
                "side":         opp["side"],
                "count":        count,
                "cost_dollars": round(cost, 4),
                "price_cents":  price_cents,
                "model_prob":   opp.get("model_prob"),
                "edge":         opp.get("edge"),
            }
            paper_trades.append(entry)
            open_positions.append(entry)
            con.commit()

        print(
            f"[OK]   Cycle {cycle_num} done: "
            f"{cycle_trades} paper trade(s), ${run_spent:.2f} deployed"
        )

        # Live portfolio snapshot after every cycle
        _print_portfolio_snapshot(
            open_positions, markets, mins,
            virtual_balance, balance, cycle_num,
        )

    # ── Wait for settlement if window hasn't closed yet ───────────────────────
    if paper_trades and last_settlement_utc is not None:
        now = datetime.now(timezone.utc)
        # Wait until 90 s after the settlement window closes (Kalshi finalization lag)
        settle_deadline = last_settlement_utc + timedelta(seconds=90)
        secs_until = (settle_deadline - now).total_seconds()

        if secs_until > 0:
            local_s = last_settlement_utc.astimezone().strftime("%H:%M %Z")
            print(f"\n{'='*60}")
            print(f"  Waiting for settlement at {local_s}")
            print(f"  Monitoring {len(paper_trades)} open position(s) — press Ctrl+C to exit early")
            print(f"{'='*60}")

            POLL_SECS = 30
            while True:
                now        = datetime.now(timezone.utc)
                remaining  = (last_settlement_utc - now).total_seconds()
                since_close = (now - last_settlement_utc).total_seconds()

                if remaining > 0:
                    # Pre-settlement: print mark-to-market heartbeat
                    try:
                        cost, mtm, unrlzd = _mtm_snapshot(paper_trades, signer)
                        sign = "+" if unrlzd >= 0 else ""
                        print(
                            f"[MTM]  cost=${cost:.2f}  mtm=${mtm:.2f}  "
                            f"unrlzd=${unrlzd:+.2f}  "
                            f"({remaining/60:.1f} min to settlement)"
                        )
                    except Exception as exc:
                        print(f"[MTM]  (fetch error: {exc})  ({remaining/60:.1f} min)")
                    sleep_secs = min(POLL_SECS, remaining)
                    time.sleep(max(sleep_secs, 1))
                else:
                    # Post-settlement grace period — try to resolve
                    if since_close >= 90:
                        print(f"[OK]   Settlement window closed {since_close/60:.1f} min ago — resolving…")
                        break
                    else:
                        print(f"[WAIT] Settlement closed — waiting for Kalshi to finalize… ({90 - since_close:.0f}s)")
                        time.sleep(min(POLL_SECS, 90 - since_close + 5))

                # Safety: don't wait more than 12 min past settlement
                if (datetime.now(timezone.utc) - last_settlement_utc).total_seconds() > 720:
                    print("[WARN] Still unresolved 12 min post-settlement — proceeding anyway.")
                    break

    # ── Settlement resolution ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Resolving {len(paper_trades)} paper trade(s)…")
    print(f"{'='*60}")

    if paper_trades:
        wins, losses, pending = _resolve_settlements(con, session_id, signer, paper_trades)
    else:
        wins = losses = pending = 0

    # Settled P&L only (don't include mark-to-market of pending bets)
    settled_rows = con.execute(
        "SELECT pnl FROM paper_trades WHERE session_id=? AND status IN ('win','loss')",
        (session_id,),
    ).fetchall()
    settled_pnl = sum(r[0] for r in settled_rows if r[0] is not None)
    final_balance = round(balance + settled_pnl, 2)

    con.execute(
        """UPDATE paper_sessions
           SET completed=1, final_bal=?, total_trades=?, wins=?, losses=?, pending=?
           WHERE id=?""",
        (final_balance, total_trades, wins, losses, pending, session_id),
    )
    con.commit()

    # Collect trade rows for doc before closing
    trade_rows = con.execute(
        """SELECT cycle_num, side, strike, count, price_cents, cost_dollars,
                  edge, status, pnl
           FROM paper_trades WHERE session_id=? ORDER BY id""",
        (session_id,),
    ).fetchall()
    con.close()

    net_pnl    = final_balance - balance
    pct_return = (net_pnl / balance * 100) if balance > 0 else 0.0

    print(f"\n  Starting balance:  ${balance:.2f}")
    print(f"  Final balance:     ${final_balance:.2f}")
    print(f"  Net P&L:           ${net_pnl:+.2f}  ({pct_return:+.1f}%)")
    print(f"  Trades placed:     {total_trades}")
    print(f"  Settled:           {wins + losses}  ({wins}W / {losses}L)")
    print(f"  Pending:           {pending}")

    # ── Session doc ───────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    doc_path = WORKSPACE_DIR / f"SESSION_{ts}.md"
    _write_session_doc(
        doc_path, ts, balance, max_cycles, max_trades,
        total_trades, wins, losses, pending, net_pnl, final_balance,
        trade_rows,
    )
    print(f"[OK]   Session doc: {doc_path}")

    # ── Final JSON for TUI (must be last stdout line, prefixed JSON:) ─────────
    result = {
        "status":          "done",
        "initial_balance": balance,
        "final_balance":   final_balance,
        "pnl":             round(net_pnl, 2),
        "pct_return":      round(pct_return, 1),
        "total_trades":    total_trades,
        "wins":            wins,
        "losses":          losses,
        "pending":         pending,
        "session_doc":     str(doc_path),
    }
    print(f"\nJSON:{json.dumps(result)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi paper trading simulation")
    parser.add_argument("--balance",    type=float, default=100.0,
                        help="Starting mock balance in dollars (default: 100)")
    parser.add_argument("--max-cycles", type=int,   default=0,
                        help="Max predict cycles (0 = unlimited)")
    parser.add_argument("--max-trades", type=int,   default=0,
                        help="Max paper trades total (0 = unlimited)")
    args = parser.parse_args()
    run_paper_session(args.balance, args.max_cycles, args.max_trades)
