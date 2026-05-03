"""Phase A unit tests: conviction gate, netting reconciliation, expiry cap, regime scale."""
import sys
sys.path.insert(0, '/home/jeb/programs/gemini_trader')

# Helpers mirroring auto_trader.py logic
MIN_CONVICTION = 0.15
MIN_EDGE = 0.05
MAX_CONTRACT_CENTS = 95
MIN_CONTRACT_CENTS = 10
MAX_PLACEMENTS_PER_RUN = 3
MAX_EXPIRY_EXPOSURE_PCT = 0.50
REGIME_RSI_LOW, REGIME_RSI_HIGH, REGIME_SCALE = 20.0, 80.0, 0.5


def conviction_gate_filter(opps_with_prob):
    """opps_with_prob: list of (clf_prob, side, edge). Returns passes."""
    out = []
    for clf_prob, side, edge in opps_with_prob:
        if edge < MIN_EDGE:
            continue
        conv = abs(clf_prob - 0.5)
        if conv < MIN_CONVICTION:
            continue
        out.append((clf_prob, side, edge, conv))
    return out


def regime_scaled_stake(base_stake_pct, rsi):
    extreme = rsi < REGIME_RSI_LOW or rsi > REGIME_RSI_HIGH
    return base_stake_pct * (REGIME_SCALE if extreme else 1.0), extreme


def netting_reconcile(opps, open_pos, net_mode=True):
    """opps: dicts with ticker, side, conviction. open_pos: {ticker: signed_int}."""
    override_threshold = MIN_CONVICTION * 2
    out = []
    for opp in opps:
        held = open_pos.get(opp["ticker"], 0)
        wants_yes = opp["side"] == "YES"
        opposes = (held > 0 and not wants_yes) or (held < 0 and wants_yes)
        if not opposes:
            out.append(opp)
            continue
        if not net_mode:
            continue  # block-only
        if opp.get("conviction", 0) >= override_threshold:
            out.append({**opp, "_netted": True})
    return out


def expiry_ticker(t):
    return t.rsplit("-T", 1)[0] if "-T" in t else t


def expiry_cap_filter(opps_with_cost, expiry_cost_seed, bankroll):
    """opps_with_cost: [(ticker, cost)]. expiry_cost_seed: {event: $ already on it}."""
    cap = bankroll * MAX_EXPIRY_EXPOSURE_PCT
    out, cost = [], dict(expiry_cost_seed)
    for ticker, c in opps_with_cost:
        ev = expiry_ticker(ticker)
        if cost.get(ev, 0) + c > cap:
            continue
        out.append((ticker, c))
        cost[ev] = cost.get(ev, 0) + c
    return out


# ═════════════════════ TESTS ════════════════════════════════════════════════

# ── Conviction gate ─────────────────────────────────────────────────────────
print("=== T1: conviction gate — 04:00 regression (clf=0.614 NO should skip) ===")
# From the postmortem: NO on T73699 had clf_prob=0.614 ⇒ model_prob=0.386? No —
# wait, NO side: model_prob = 1-clf_prob. But conviction is |clf_prob - 0.5|,
# which is symmetric for both sides. conviction = |0.614 - 0.5| = 0.114 < 0.15.
res = conviction_gate_filter([(0.614, "NO", 0.094)])
assert len(res) == 0, f"expected 0 (conviction fail), got {res}"
print("  PASS — weak-conviction NO correctly skipped\n")

print("=== T2: strong-conviction bet passes ===")
res = conviction_gate_filter([(0.85, "YES", 0.20)])
assert len(res) == 1
print(f"  PASS — strong signal retained, conviction={res[0][3]:.3f}\n")

print("=== T3: strong YES + weak NO — only YES survives ===")
res = conviction_gate_filter([(0.85, "YES", 0.20), (0.55, "NO", 0.10)])
assert len(res) == 1 and res[0][1] == "YES"
print("  PASS\n")

# ── Regime scaling ──────────────────────────────────────────────────────────
print("=== T4: regime scale — normal RSI, no scaling ===")
s, ext = regime_scaled_stake(0.10, rsi=50)
assert s == 0.10 and not ext
print(f"  PASS — stake_pct unchanged at {s}\n")

print("=== T5: regime scale — RSI extreme oversold (< 20) ===")
s, ext = regime_scaled_stake(0.10, rsi=15)
assert abs(s - 0.05) < 1e-9 and ext
print(f"  PASS — stake halved to {s}\n")

print("=== T6: regime scale — RSI extreme overbought (> 80) ===")
s, ext = regime_scaled_stake(0.10, rsi=85)
assert abs(s - 0.05) < 1e-9 and ext
print(f"  PASS — stake halved to {s}\n")

# ── Netting reconciliation ──────────────────────────────────────────────────
print("=== T7: netting — weak opposing signal blocked (04:00 regression) ===")
opps = [{"ticker": "K-1504-T73699", "side": "NO", "conviction": 0.114}]
open_pos = {"K-1504-T73699": 9}  # held 9 YES
res = netting_reconcile(opps, open_pos, net_mode=True)
assert len(res) == 0, f"weak opposing should be blocked; got {res}"
print("  PASS — weak NO correctly blocked by held YES\n")

print("=== T8: netting — strong opposing signal overrides ===")
opps = [{"ticker": "K-1504-T73699", "side": "NO", "conviction": 0.35}]
open_pos = {"K-1504-T73699": 9}
res = netting_reconcile(opps, open_pos, net_mode=True)
assert len(res) == 1 and res[0].get("_netted")
print("  PASS — strong NO (conv 0.35) overrides held YES\n")

print("=== T9: block-only mode — even strong opposing blocked ===")
opps = [{"ticker": "K-1504-T73699", "side": "NO", "conviction": 0.35}]
open_pos = {"K-1504-T73699": 9}
res = netting_reconcile(opps, open_pos, net_mode=False)
assert len(res) == 0
print("  PASS — block-only ignores strength\n")

print("=== T10: same-side signal always passes ===")
opps = [{"ticker": "K-1504-T73699", "side": "YES", "conviction": 0.20}]
open_pos = {"K-1504-T73699": 9}
res = netting_reconcile(opps, open_pos, net_mode=True)
assert len(res) == 1
print("  PASS\n")

# ── Per-expiry exposure cap ─────────────────────────────────────────────────
print("=== T11: expiry cap — 04:00 scenario rejected ===")
# $86 bankroll, 50% cap = $43 max per expiry. Already $40 on expiry, $10 new = $50 > $43.
opps = [("KXBTCD-26APR1504-T73699.99", 10)]
res = expiry_cap_filter(opps, {"KXBTCD-26APR1504": 40.0}, bankroll=86.0)
assert len(res) == 0
print("  PASS — bets blocked once expiry exposure exceeds cap\n")

print("=== T12: expiry cap — different expiries independent ===")
opps = [("KXBTCD-26APR1504-T73699.99", 10),
        ("KXBTCD-26APR1505-T73799.99", 10),
        ("KXBTCD-26APR1506-T73899.99", 10)]
res = expiry_cap_filter(opps, {}, bankroll=100.0)
assert len(res) == 3
print(f"  PASS — 3 different expiries all under individual 50% caps\n")

print("=== T13: expiry cap — accumulates within single expiry ===")
opps = [("KXBTCD-26APR1504-T73699.99", 20),   # 20 ≤ 50 ✓
        ("KXBTCD-26APR1504-T73799.99", 20),   # 40 ≤ 50 ✓
        ("KXBTCD-26APR1504-T73899.99", 20)]   # 60 > 50 ✗
res = expiry_cap_filter(opps, {}, bankroll=100.0)
assert len(res) == 2, f"expected 2, got {len(res)}"
print("  PASS — third strike on same expiry blocked when cumulative exceeds cap\n")

# ── End-to-end composition test ─────────────────────────────────────────────
print("=== T14: end-to-end — the 04:00 scenario should now stop ===")
# Simulate: classifier signals NO on T73699 with clf_prob=0.614 (conv=0.114).
# We hold 9 YES from an earlier run. Three gates must reject this:
#   (i) conviction < 0.15 → skipped in find_opportunities
#   (ii) if it survived, netting blocks weak opposing (conv < 0.30)
#   (iii) if somehow both failed, expiry cap would still bound total exposure
# Primary gate (conviction):
res = conviction_gate_filter([(0.614, "NO", 0.094)])
assert len(res) == 0, "Conviction should block"
print("  PASS — primary conviction gate stops it\n")

print("ALL PHASE A TESTS PASSED ✓")
