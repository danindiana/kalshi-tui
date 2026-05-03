#!/usr/bin/env python3
"""
wizard.py — Kalshi BTC Auto-Trader Operator Wizard

Interactive operator dashboard that:
  1. Checks all system prerequisites
  2. Shows live ratchet sizing (balance → daily/run/stake limits)
  3. Shows classifier stats (sole trading model; LSTM retired 2026-04-14)
  4. Shows live classifier prediction + opportunities (predict_classifier_json.py)
  5. Shows recent run history and today's deployed capital
  6. Optionally runs an Ollama AI briefing / strategy review
  7. Gets operator approval, then runs auto_trader.py once or in a loop

Usage:
  python wizard.py            # full interactive wizard
  python wizard.py --check    # health check + ratchet status only, then exit
  python wizard.py --brief    # include deepseek strategy review (slower)
  python wizard.py --no-ai    # skip Ollama steps
  python wizard.py --loop [--interval N]  # run every N minutes until Ctrl+C
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tomllib
from datetime import datetime, date, timezone
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

console = Console()

# ── Paths & constants ─────────────────────────────────────────────────────────
GEMINI_DIR   = Path("/home/jeb/programs/gemini_trader")
VENV_PYTHON  = GEMINI_DIR / "venv" / "bin" / "python"
CONFIG_PATH  = Path.home() / ".config" / "kalshi-tui" / "config.toml"
KEY_PATH     = Path.home() / ".config" / "kalshi-tui" / "private_key.pem"
DB_PATH      = Path.home() / ".local" / "share" / "kalshi-tui" / "auto_trader.db"
HWM_PATH     = Path.home() / ".local" / "share" / "kalshi-tui" / "hwm.json"
LOG_PATH     = Path.home() / ".local" / "share" / "kalshi-tui" / "auto_trader.log"
METRICS_PATH = GEMINI_DIR / "time_series" / "training_metrics.json"
CLF_PATH     = GEMINI_DIR / "time_series" / "classifier_model.pkl"
OLLAMA_URL   = "http://localhost:11434"
KALSHI_BASE  = "https://api.elections.kalshi.com/trade-api/v2"

# Ratchet constants (must match service file defaults)
MAX_BANKROLL      = float(os.environ.get("MAX_BANKROLL",      "500"))
DAILY_LOSS_PCT    = float(os.environ.get("DAILY_LOSS_PCT",    "0.50"))
MAX_RUN_PCT       = float(os.environ.get("MAX_RUN_PCT",       "0.40"))
MAX_STAKE_PCT     = float(os.environ.get("MAX_STAKE_PCT",     "0.12"))
DAILY_LOSS_LIMIT  = float(os.environ.get("DAILY_LOSS_LIMIT",  "200"))
MAX_RUN_DOLLARS   = float(os.environ.get("MAX_RUN_DOLLARS",   "100"))
MAX_STAKE_DOLLARS = float(os.environ.get("MAX_STAKE_DOLLARS", "25"))
DRAWDOWN_HALT     = float(os.environ.get("DRAWDOWN_HALT",     "0.30"))
MIN_BALANCE       = float(os.environ.get("MIN_BALANCE",       "5.00"))

# Local model assignments
BRIEFING_MODEL  = "ministral-3:8b"
REASONING_MODEL = "deepseek-r1:14b"


# ═══════════════════════════════════════════════════════════════════════════════
# § 1 — System Health Checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_prerequisites() -> dict:
    """Run all prerequisite checks. Returns {label: (ok, detail)}."""
    results = {}

    # Kalshi config
    try:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        key_id = cfg.get("key_id", "").strip()
        if key_id:
            results["Kalshi key_id"] = (True, f"{key_id[:8]}…")
        else:
            results["Kalshi key_id"] = (False, f"Empty — set in {CONFIG_PATH}")
    except Exception as e:
        results["Kalshi key_id"] = (False, str(e))

    # Private key
    if KEY_PATH.exists():
        perms = oct(KEY_PATH.stat().st_mode)[-3:]
        ok    = perms in ("600", "400")
        results["Private key"] = (ok, f"perms={perms} ← fix: chmod 600" if not ok else str(KEY_PATH))
    else:
        results["Private key"] = (False, f"Missing: {KEY_PATH}")

    # Python venv
    results["Python venv"] = (VENV_PYTHON.exists(), str(VENV_PYTHON))

    # Classifier model (sole live trading signal; LSTM retired 2026-04-14)
    clf_ok = CLF_PATH.exists()
    results["Classifier model (required)"] = (clf_ok, "OK" if clf_ok else f"MISSING — retrain: python time_series/train_classifier.py")

    # Systemd timer
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "kalshi-auto-trader.timer"],
            capture_output=True, text=True, timeout=5
        )
        active = r.stdout.strip() == "active"
        results["Systemd timer (10 min)"] = (active, r.stdout.strip())
    except Exception as e:
        results["Systemd timer (10 min)"] = (False, str(e))

    # Ollama
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        count = len(data.get("models", []))
        results["Ollama"] = (True, f"{count} models available")
    except Exception:
        results["Ollama"] = (False, "Not responding on :11434")

    # Gemini spot price
    try:
        import urllib.request
        with urllib.request.urlopen("https://api.gemini.com/v1/pubticker/btcusd", timeout=5) as resp:
            d = json.loads(resp.read())
        price = float(d["last"])
        results["Gemini API"] = (True, f"BTC spot ${price:,.2f}")
    except Exception as e:
        results["Gemini API"] = (False, str(e))

    # Kalshi account balance (live API check)
    try:
        with open(CONFIG_PATH, "rb") as f:
            _cfg = tomllib.load(f)
        _kid = _cfg.get("key_id", "").strip()
        if _kid:
            sys.path.insert(0, str(GEMINI_DIR))
            from kalshi_auth import KalshiSigner as _Signer
            _signer = _Signer.from_config()
            _r = _signer.get(f"{KALSHI_BASE}/portfolio/balance")
            _bal = _r.json().get("balance", 0)
            _dollars = _bal / 100.0
            results["Kalshi balance"] = (True, f"${_dollars:.2f} available")
        else:
            results["Kalshi balance"] = (False, "key_id not set")
    except Exception as e:
        results["Kalshi balance"] = (False, f"Auth error: {str(e)[:50]}")

    # DRY_RUN flag in service file
    try:
        svc = Path("/etc/systemd/system/kalshi-auto-trader.service").read_text()
        dry = "DRY_RUN=1" in svc
        results["Live trading"] = (not dry, "LIVE (DRY_RUN=0)" if not dry else "Simulation (DRY_RUN=1)")
    except Exception:
        results["Live trading"] = (False, "Service file unreadable")

    return results


def print_health_table(checks: dict) -> bool:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Check",  style="white", min_width=30)
    table.add_column("Status", min_width=6, justify="center")
    table.add_column("Detail", style="dim")

    for label, (ok, detail) in checks.items():
        icon = "[bold green]✓[/]" if ok else "[bold red]✗[/]"
        table.add_row(label, icon, detail)

    console.print(table)
    passed = sum(1 for ok, _ in checks.values() if ok)
    total  = len(checks)
    color  = "green" if passed == total else "yellow"
    console.print(f"  [{color}]{passed}/{total} checks passed[/]\n")
    return passed == total


# ═══════════════════════════════════════════════════════════════════════════════
# § 2 — Ratchet Sizing Status
# ═══════════════════════════════════════════════════════════════════════════════

def get_live_balance() -> float:
    """Fetch live Kalshi balance in dollars. Returns -1.0 on failure."""
    try:
        sys.path.insert(0, str(GEMINI_DIR))
        from kalshi_auth import KalshiSigner
        signer = KalshiSigner.from_config()
        r = signer.get(f"{KALSHI_BASE}/portfolio/balance")
        r.raise_for_status()
        return r.json().get("balance", 0) / 100.0
    except Exception:
        return -1.0


def get_daily_deployed() -> float:
    """Return total capital deployed today from the trades DB."""
    if not DB_PATH.exists():
        return 0.0
    try:
        con = sqlite3.connect(str(DB_PATH))
        today = date.today().isoformat()
        row = con.execute(
            "SELECT COALESCE(SUM(cost_dollars),0) FROM trades WHERE placed_at LIKE ? AND status='placed'",
            (f"{today}%",),
        ).fetchone()
        con.close()
        return row[0] if row else 0.0
    except Exception:
        return 0.0


def print_ratchet_panel():
    """Display live balance, HWM, drawdown, and all computed limits."""
    live = get_live_balance()
    if live < 0:
        console.print("  [yellow]Balance API unavailable — cannot display ratchet status.[/]\n")
        return

    bankroll  = min(live, MAX_BANKROLL)
    daily_cap = min(bankroll * DAILY_LOSS_PCT, DAILY_LOSS_LIMIT)
    run_cap   = min(bankroll * MAX_RUN_PCT,    MAX_RUN_DOLLARS)
    stake_cap = min(bankroll * MAX_STAKE_PCT,  MAX_STAKE_DOLLARS)
    deployed  = get_daily_deployed()
    remaining = max(0.0, daily_cap - deployed)

    # HWM
    hwm = live
    drawdown = 0.0
    try:
        with open(HWM_PATH) as f:
            h = json.load(f)
        hwm      = float(h.get("hwm", live))
        drawdown = float(h.get("drawdown_pct", 0.0))
    except Exception:
        pass

    # Drawdown color
    if drawdown >= DRAWDOWN_HALT:
        dd_color = "bold red"
    elif drawdown >= DRAWDOWN_HALT * 0.7:
        dd_color = "yellow"
    else:
        dd_color = "green"

    capped_note = f"  [dim](capped at ${MAX_BANKROLL:.0f})[/dim]" if bankroll < live else ""
    lines = [
        f"[white]Live balance:[/]    [bold]${live:.2f}[/]{capped_note}",
        f"[white]HWM:[/]             ${hwm:.2f}   "
        f"Drawdown: [{dd_color}]{drawdown:.1%}[/]"
        + (f"  [bold red]← HALT THRESHOLD {DRAWDOWN_HALT:.0%}[/]" if drawdown >= DRAWDOWN_HALT else ""),
        "",
        f"[white]Daily cap (50%):[/] ${daily_cap:.2f}   "
        f"deployed today: [{'red' if deployed >= daily_cap else 'white'}]${deployed:.2f}[/]   "
        f"remaining: [bold]${remaining:.2f}[/]",
        f"[white]Per-run cap (40%):[/] ${run_cap:.2f}",
        f"[white]Per-trade cap (12%):[/] ${stake_cap:.2f}",
        "",
        f"[dim]Absolute ceilings: daily ${DAILY_LOSS_LIMIT:.0f} / run ${MAX_RUN_DOLLARS:.0f} / stake ${MAX_STAKE_DOLLARS:.0f}[/dim]",
    ]
    style = "bold red" if drawdown >= DRAWDOWN_HALT or deployed >= daily_cap else "green"
    console.print(Panel("\n".join(lines), title="[bold cyan]Ratchet Sizing[/]", box=box.ROUNDED, border_style=style))
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# § 3 — Active Model Status
# ═══════════════════════════════════════════════════════════════════════════════

def print_model_status():
    """Show classifier stats from training_metrics.json."""
    if not METRICS_PATH.exists():
        console.print("  [dim]No training_metrics.json found.[/dim]\n")
        return

    try:
        with open(METRICS_PATH) as f:
            m = json.load(f)
    except Exception as e:
        console.print(f"  [red]Could not read training_metrics.json:[/] {e}\n")
        return

    model_type = m.get("model_type", "unknown")
    trained_at = m.get("trained_at", "unknown")
    try:
        trained_at = datetime.fromisoformat(trained_at).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    if model_type == "classifier":
        bt = m.get("backtest", {})
        lines = [
            f"[white]Active model:[/]    [bold green]GBT Classifier[/]",
            f"[white]Trained at:[/]      {trained_at}",
            f"[white]Backend:[/]         {m.get('backend', 'unknown')}",
            f"[white]AUC:[/]             {m.get('auc', 0):.4f}",
            f"[white]Brier skill:[/]     {m.get('brier_skill', 0):.1f}%  [dim](0=baseline, 100=perfect)[/dim]",
            "",
            f"[white]Backtest (OOS):[/]  {bt.get('bets', 0):,} bets   "
            f"win rate {bt.get('win_rate', 0):.1%}   "
            f"avg EV {bt.get('avg_ev', 0)*100:.1f}%",
            f"[white]Simulated P&L:[/]   ${bt.get('total_pnl', 0):+,.2f}   "
            f"[dim]chronological split: {m.get('chronological_split', False)}[/dim]",
        ]
        deployable = bt.get("deployable", False)
        style = "green" if deployable else "yellow"
        console.print(Panel("\n".join(lines), title="[bold cyan]Model Status[/]", box=box.ROUNDED, border_style=style))
    else:
        lines = [
            f"[bold red]⚠ training_metrics.json does not describe the classifier.[/]",
            f"[white]Trained at:[/]      {trained_at}",
            f"[white]Model type:[/]      {model_type}  [dim](expected: 'classifier')[/dim]",
            "",
            f"[yellow]Live trader uses only classifier_model.pkl (LSTM retired 2026-04-14).",
            f"[yellow]To refresh metrics: python time_series/train_classifier.py",
        ]
        console.print(Panel("\n".join(lines), title="[bold red]Model Status — stale metrics[/]", box=box.ROUNDED, border_style="red"))

    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
# § 4 — Live Classifier Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def run_prediction() -> dict | None:
    console.print("  [dim]Running classifier (≈3 s)…[/dim]")
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(GEMINI_DIR / "predict_classifier_json.py")],
            capture_output=True, text=True, timeout=60,
            cwd=str(GEMINI_DIR)
        )
        data = json.loads(result.stdout.strip())
        if "error" in data:
            console.print(f"  [red]Classifier error:[/] {data['error']}")
            return None
        return data
    except Exception as e:
        console.print(f"  [red]Prediction failed:[/] {e}")
        return None


def print_prediction_panel(pred: dict):
    ts = datetime.fromisoformat(pred["timestamp"]).strftime("%H:%M:%S")
    opps = pred.get("opportunities", [])
    tradeable = [o for o in opps if o["edge"] >= 0.05]

    lines = [
        f"[white]Time:[/]              {ts}",
        f"[white]Current price:[/]     [bold]${pred['current_price']:,.2f}[/]",
        f"[white]Mins to settlement:[/] {pred['mins_to_expiry']:.0f}",
        f"[white]Tradeable edges:[/]   [bold cyan]{len(tradeable)}[/]  [dim]of {len(opps)} evaluated[/dim]",
        "",
        f"[white]RSI:[/]               {pred['indicators']['rsi']:.1f}",
        f"[white]Volatility:[/]        ${pred['indicators']['volatility']:,.0f}",
        f"[white]Momentum:[/]          ${pred['indicators']['momentum']:+,.0f}",
    ]
    console.print(Panel("\n".join(lines), title="[bold cyan]Classifier Prediction[/]", box=box.ROUNDED))


def print_opportunities_table(pred: dict):
    opps = pred.get("opportunities", [])
    if not opps:
        console.print("  [yellow]No opportunities found.[/]\n")
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Strike",    justify="right",  style="white")
    table.add_column("Side",      justify="center")
    table.add_column("Mkt YES",   justify="right")
    table.add_column("Mdl Prob",  justify="right")
    table.add_column("Edge",      justify="right")
    table.add_column("Stake %",   justify="right")

    for opp in sorted(opps, key=lambda x: abs(x["edge"]), reverse=True):
        edge     = opp["edge"]
        side_str = f"[green]{opp['side']}[/]" if opp["side"] == "YES" else f"[red]{opp['side']}[/]"
        edge_str = f"[green]+{edge*100:.1f}%[/]" if edge >= 0.05 else f"[dim]{edge*100:+.1f}%[/]"
        table.add_row(
            f"${opp['strike']:,.0f}",
            side_str,
            f"{opp['market_yes']*100:.0f}¢",
            f"{opp['model_prob']*100:.0f}%",
            edge_str,
            f"{opp['stake_pct']*100:.1f}%",
        )

    console.print(table)
    tradeable = [o for o in opps if abs(o["edge"]) >= 0.05]
    console.print(f"  [bold]{len(tradeable)} trades with edge ≥ 5%[/] out of {len(opps)} candidates\n")


# ═══════════════════════════════════════════════════════════════════════════════
# § 5 — Recent Performance
# ═══════════════════════════════════════════════════════════════════════════════

def print_recent_performance():
    if not DB_PATH.exists():
        console.print("  [dim]No trade history yet.[/dim]\n")
        return

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    runs = con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 8").fetchall()
    if not runs:
        con.close()
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Run time",   style="dim", min_width=19)
    table.add_column("Spot",       justify="right")
    table.add_column("Opps",      justify="right")
    table.add_column("Placed",    justify="right")
    table.add_column("Deployed",  justify="right")
    table.add_column("Mode",      justify="center")

    for r in reversed(runs):
        mode = "[yellow]DRY[/]" if r["dry_run"] else "[green]LIVE[/]"
        table.add_row(
            r["run_at"][:19],
            f"${r['current_price']:,.0f}",
            str(r["opportunities"]),
            str(r["trades_placed"]),
            f"${r['total_staked']:.2f}",
            mode,
        )
    console.print(table)

    # Daily summary for the last 7 days
    rows = con.execute(
        """SELECT date(run_at) as day, count(*) as runs, sum(trades_placed) as trades,
                  round(sum(total_staked),2) as deployed
           FROM runs GROUP BY date(run_at) ORDER BY day DESC LIMIT 7"""
    ).fetchall()
    if rows:
        console.print()
        dtable = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", title="Daily Summary")
        dtable.add_column("Date",     style="dim")
        dtable.add_column("Runs",     justify="right")
        dtable.add_column("Trades",   justify="right")
        dtable.add_column("Deployed", justify="right")
        for row in rows:
            dtable.add_row(row["day"], str(row["runs"]), str(row["trades"] or 0), f"${row['deployed'] or 0:.2f}")
        console.print(dtable)

    today = date.today().isoformat()
    row = con.execute(
        "SELECT COALESCE(SUM(cost_dollars),0) FROM trades WHERE placed_at LIKE ? AND status='placed'",
        (f"{today}%",),
    ).fetchone()
    console.print(f"\n  Today's live capital deployed: [bold]${row[0]:.2f}[/]\n")
    con.close()


# ═══════════════════════════════════════════════════════════════════════════════
# § 6 — Ollama AI Briefing
# ═══════════════════════════════════════════════════════════════════════════════

def ollama_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def ollama_generate(model: str, prompt: str, max_tokens: int = 400) -> str:
    import urllib.request
    body = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


def generate_morning_briefing(pred: dict) -> str:
    """ministral-3:8b — fast plain-language summary of trading conditions."""
    opps = pred.get("opportunities", [])
    tradeable = [o for o in opps if abs(o["edge"]) >= 0.05]
    top = sorted(tradeable, key=lambda x: abs(x["edge"]), reverse=True)[:5]
    top_json = json.dumps([
        {"strike": o["strike"], "side": o["side"], "edge": f"{o['edge']*100:.1f}%"}
        for o in top
    ])

    prompt = (
        "You are a quant trading analyst writing a concise briefing "
        "for a Bitcoin prediction-market trader on Kalshi (hourly KXBTCD settlements, 24/7).\n\n"
        f"Current BTC: ${pred['current_price']:,.0f}\n"
        f"RSI: {pred['indicators']['rsi']:.1f} | Volatility: ${pred['indicators']['volatility']:.0f}"
        f" | Momentum: ${pred['indicators']['momentum']:+.0f}\n"
        f"Mins to next settlement: {pred['mins_to_expiry']:.0f}\n"
        f"Tradeable opportunities from classifier (P(price>strike) ≥ 5% above market): {len(tradeable)}\n"
        f"Top trades: {top_json}\n\n"
        "Write 4-6 sentences: current market read, confidence level, top trade summary, "
        "plain-language recommendation (PROCEED / CAUTIOUS / SKIP). Be direct."
    )
    return ollama_generate(BRIEFING_MODEL, prompt, max_tokens=300)


def generate_strategy_review(pred: dict) -> str:
    """deepseek-r1:14b — reasoning review of the proposed trade plan."""
    opps = pred.get("opportunities", [])
    tradeable = [o for o in opps if abs(o["edge"]) >= 0.05]

    prompt = (
        f"Review this proposed set of Kalshi BTC hourly prediction-market trades.\n\n"
        f"BTC now: ${pred['current_price']:,.0f} | mins to settlement: {pred['mins_to_expiry']:.0f}\n"
        f"RSI: {pred['indicators']['rsi']:.1f} | Vol: ${pred['indicators']['volatility']:.0f} | Momentum: ${pred['indicators']['momentum']:+.0f}\n\n"
        f"Signal source: calibrated GBT classifier outputs P(price > strike at settlement). Edge = model_prob − market_price.\n\n"
        f"Proposed trades:\n"
        + "\n".join(
            f"  {o['side']} ${o['strike']:,.0f} — edge {o['edge']*100:.1f}%, stake {o['stake_pct']*100:.1f}%"
            for o in tradeable
        )
        + "\n\nIn 3-5 sentences: flag any red flags, comment on edge concentration and MAE risk. "
        "End with exactly one of: PROCEED / REDUCE_SIZE / SKIP"
    )
    return ollama_generate(REASONING_MODEL, prompt, max_tokens=280)


# ═══════════════════════════════════════════════════════════════════════════════
# § 7 — Execute auto_trader
# ═══════════════════════════════════════════════════════════════════════════════

def run_auto_trader(dry_run: bool):
    env = os.environ.copy()
    env["DRY_RUN"] = "1" if dry_run else "0"
    label = "[yellow]DRY RUN[/]" if dry_run else "[bold red]LIVE[/]"
    console.print(f"\n  Launching auto_trader.py — {label}\n")
    subprocess.run(
        [str(VENV_PYTHON), str(GEMINI_DIR / "auto_trader.py")],
        env=env, cwd=str(GEMINI_DIR),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# § 8 — Loop mode (--loop)
# ═══════════════════════════════════════════════════════════════════════════════

def run_loop(dry_run: bool, interval: int):
    """
    After operator approval, run auto_trader every `interval` minutes until Ctrl+C.
    Shows a live countdown between cycles. Designed to run inside a tmux session.
    Note: the systemd timer already runs every 10 min — use --loop only for manual
    sessions where you want an interactive countdown and per-cycle output.
    """
    import time

    label = "[yellow]DRY RUN[/]" if dry_run else "[bold red]LIVE[/]"
    console.print(Panel(
        f"Mode:      {label}\n"
        f"Interval:  every {interval} minutes\n"
        f"Logs:      {LOG_PATH}\n"
        f"DB:        {DB_PATH}\n\n"
        f"[dim]Ctrl+C at any time to stop cleanly.[/dim]\n"
        f"[dim]Note: systemd timer also fires every 10 min independently.[/dim]",
        title="[bold cyan]Loop Mode Active[/]",
        box=box.ROUNDED,
    ))

    cycle = 0
    try:
        while True:
            cycle += 1
            now_str = datetime.now().strftime("%H:%M:%S")
            console.print(Rule(f"[bold cyan]Cycle {cycle}  —  {now_str}[/]"))
            try:
                run_auto_trader(dry_run=dry_run)
            except Exception as exc:
                console.print(f"  [red]auto_trader error (will retry next cycle):[/] {exc}")

            deadline = time.monotonic() + interval * 60
            while True:
                left = int(deadline - time.monotonic())
                if left <= 0:
                    break
                m, s = divmod(left, 60)
                sys.stdout.write(f"\r  Next cycle in {m:02d}:{s:02d}  (Ctrl+C to stop)  ")
                sys.stdout.flush()
                time.sleep(0.5)
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

    except KeyboardInterrupt:
        console.print("\n")
        console.print(Rule("[bold yellow]Loop stopped — session ended[/]"))


# ═══════════════════════════════════════════════════════════════════════════════
# § Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Kalshi BTC Auto-Trader Operator Wizard")
    ap.add_argument("--check",    action="store_true", help="Health check + ratchet status only, then exit")
    ap.add_argument("--brief",    action="store_true", help="Add deepseek strategy review (slower)")
    ap.add_argument("--no-ai",    action="store_true", help="Skip all Ollama AI steps")
    ap.add_argument("--no-pred",  action="store_true", help="Skip classifier prediction step (faster startup)")
    ap.add_argument("--loop",     action="store_true", help="Run continuously — trade every --interval minutes until Ctrl+C")
    ap.add_argument("--interval", type=int, default=10, metavar="MIN",
                    help="Minutes between cycles in --loop mode (default: 10, matches systemd timer)")
    args = ap.parse_args()

    console.print(Rule("[bold cyan]Kalshi BTC Auto-Trader — Operator Wizard[/]"))
    console.print(f"  [dim]{datetime.now().strftime('%A %B %d %Y  %H:%M:%S')}[/]\n")

    # 1 — Health
    console.print(Rule("[bold]1 · System Health[/]", style="dim"))
    checks = check_prerequisites()
    print_health_table(checks)
    key_ok    = checks.get("Kalshi key_id",  (False,))[0]
    live_mode = checks.get("Live trading",   (False,))[0]

    # 2 — Ratchet sizing
    console.print(Rule("[bold]2 · Ratchet Sizing & Limits[/]", style="dim"))
    print_ratchet_panel()

    # 3 — Model status
    console.print(Rule("[bold]3 · Active Model Status[/]", style="dim"))
    print_model_status()

    if args.check:
        sys.exit(0)

    # 4 — Classifier Prediction
    if args.no_pred:
        pred = None
    else:
        console.print(Rule("[bold]4 · Classifier Prediction[/]", style="dim"))
        pred = run_prediction()
        if pred is not None:
            print_prediction_panel(pred)
            console.print(Rule("[bold]4b · Classifier Opportunities[/]", style="dim"))
            print_opportunities_table(pred)

    # 5 — History
    console.print(Rule("[bold]5 · Recent Run History[/]", style="dim"))
    print_recent_performance()

    # 6 — Ollama AI (free, local, optional)
    use_ai = not args.no_ai and ollama_available() and pred is not None
    if use_ai:
        console.print(Rule("[bold]6 · AI Briefing  (local Ollama — no API cost)[/]", style="dim"))
        console.print(f"  [dim]{BRIEFING_MODEL} — briefing…[/dim]")
        try:
            briefing = generate_morning_briefing(pred)
            console.print(Panel(briefing, title=f"[bold]{BRIEFING_MODEL}[/] Briefing", box=box.ROUNDED, style="cyan"))
        except Exception as e:
            console.print(f"  [yellow]Briefing unavailable:[/] {e}")

        if args.brief:
            console.print(f"\n  [dim]{REASONING_MODEL} — strategy review…[/dim]")
            try:
                review = generate_strategy_review(pred)
                console.print(Panel(review, title=f"[bold]{REASONING_MODEL}[/] Strategy Review", box=box.ROUNDED, style="magenta"))
            except Exception as e:
                console.print(f"  [yellow]Review unavailable:[/] {e}")
    elif not args.no_ai and pred is not None:
        console.print(Rule("[bold]6 · AI Briefing[/]", style="dim"))
        console.print("  [dim]Ollama not running — skipping AI briefing. Start with: systemctl start ollama[/dim]\n")

    # 7 — Decision
    console.print(Rule("[bold]7 · Operator Decision[/]", style="dim"))

    if not key_ok:
        console.print(Panel(
            f"[yellow]key_id is not configured.[/]\n\n"
            f"1. Kalshi portal → Settings → API Keys\n"
            f"2. [cyan]nano {CONFIG_PATH}[/]\n"
            f"   key_id = \"your-key-here\"\n"
            f"3. Re-run wizard",
            title="⚠  Auth not set up", box=box.ROUNDED
        ))
        sys.exit(1)

    mode_str = "[green]LIVE trading[/]" if live_mode else "[yellow]DRY RUN mode[/]"
    console.print(f"  Service configured for: {mode_str}")
    if not live_mode:
        console.print(
            "\n  To enable live trading:\n"
            "    [dim]sudo nano /etc/systemd/system/kalshi-auto-trader.service[/dim]\n"
            "    [dim]  Environment=\"DRY_RUN=0\"[/dim]\n"
            "    [dim]sudo systemctl daemon-reload[/dim]\n"
        )

    console.print(
        "\n  [dim]Note: systemd timer fires every 10 min automatically.[/dim]\n"
        "  [dim]Running wizard manually executes one additional cycle.[/dim]\n"
    )
    console.print()
    choice = Prompt.ask(
        "  Run now?",
        choices=["live", "dry", "skip"],
        default="dry" if not live_mode else "live",
    )

    if choice == "skip":
        console.print("  [dim]Skipped.[/dim]")
        console.print(Rule("[bold cyan]Wizard complete[/]"))
    elif args.loop:
        run_loop(dry_run=(choice == "dry"), interval=args.interval)
    else:
        run_auto_trader(dry_run=(choice == "dry"))
        console.print(f"\n  Logs:     [cyan]{LOG_PATH}[/]")
        console.print(f"  DB:       [cyan]{DB_PATH}[/]")
        console.print(f"  Review:   python {GEMINI_DIR}/trader_log.py")
        console.print(Rule("[bold cyan]Wizard complete[/]"))


if __name__ == "__main__":
    main()
