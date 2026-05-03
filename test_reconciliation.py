"""Unit test for the position-reconciliation + top-N cap logic."""
import sys
sys.path.insert(0, '/home/jeb/programs/gemini_trader')

# Simulate the reconciliation + cap block from auto_trader.py against synthetic inputs.
MAX_PLACEMENTS_PER_RUN = 3

def reconcile_and_cap(opportunities, open_pos, verbose=True):
    """Extract of the production logic for isolated testing."""
    if open_pos:
        before = len(opportunities)
        filtered = []
        for opp in opportunities:
            held = open_pos.get(opp["ticker"], 0)
            wants_yes = opp["side"] == "YES"
            if held > 0 and not wants_yes:
                if verbose: print(f"  [SKIP-OPP] {opp['ticker']}: holding {held} YES, new NO blocked")
                continue
            if held < 0 and wants_yes:
                if verbose: print(f"  [SKIP-OPP] {opp['ticker']}: holding {-held} NO, new YES blocked")
                continue
            filtered.append(opp)
        opportunities = filtered
        dropped = before - len(opportunities)
        if dropped and verbose:
            print(f"  Reconciliation: {dropped} blocked; {len(opportunities)} remain")
    if len(opportunities) > MAX_PLACEMENTS_PER_RUN:
        dropped = len(opportunities) - MAX_PLACEMENTS_PER_RUN
        if verbose: print(f"  Top-N: dropping {dropped}, keeping top {MAX_PLACEMENTS_PER_RUN}")
        opportunities = opportunities[:MAX_PLACEMENTS_PER_RUN]
    return opportunities


def opp(ticker, side, edge):
    return {"ticker": ticker, "side": side, "edge": edge}


# ── Test 1: No open positions, 5 opps → top-3 by edge ──────────────────────
print("=== Test 1: top-N cap only (no open positions) ===")
opps = [opp("A", "YES", 0.20), opp("B", "NO", 0.15), opp("C", "YES", 0.10),
        opp("D", "YES", 0.08), opp("E", "NO", 0.06)]
res = reconcile_and_cap(opps, {})
assert len(res) == 3, f"expected 3, got {len(res)}"
assert [x["ticker"] for x in res] == ["A", "B", "C"], res
print(f"  PASS — kept {[x['ticker'] for x in res]}\n")


# ── Test 2: Opposing open position blocks new signal ───────────────────────
print("=== Test 2: opposing-signal block ===")
opps = [opp("A", "NO", 0.20), opp("B", "YES", 0.15), opp("C", "NO", 0.10)]
# We hold 5 YES on A, 3 NO on B
open_pos = {"A": 5, "B": -3}
res = reconcile_and_cap(opps, open_pos)
assert len(res) == 1, f"expected 1 (only C), got {len(res)}: {res}"
assert res[0]["ticker"] == "C"
print(f"  PASS — only {[x['ticker'] for x in res]} survived\n")


# ── Test 3: Same-side signal as held position passes through ───────────────
print("=== Test 3: same-side held, new signal OK ===")
opps = [opp("A", "YES", 0.20), opp("B", "NO", 0.15)]
open_pos = {"A": 5, "B": -3}  # YES on A matches signal; NO on B matches signal
res = reconcile_and_cap(opps, open_pos)
assert len(res) == 2, f"expected 2, got {len(res)}"
print(f"  PASS — kept {[x['ticker'] for x in res]}\n")


# ── Test 4: Reconciliation + cap combined ──────────────────────────────────
print("=== Test 4: block 2, then cap remaining 4 to 3 ===")
opps = [opp("A", "NO", 0.30),   # blocked (hold YES)
        opp("B", "YES", 0.25),
        opp("C", "NO", 0.20),   # blocked (hold YES)
        opp("D", "YES", 0.15),
        opp("E", "NO", 0.10),
        opp("F", "YES", 0.08)]
open_pos = {"A": 2, "C": 4}
res = reconcile_and_cap(opps, open_pos)
assert len(res) == 3, f"expected 3, got {len(res)}"
assert [x["ticker"] for x in res] == ["B", "D", "E"], res
print(f"  PASS — kept {[x['ticker'] for x in res]}\n")


# ── Test 5: The actual bug (b) scenario: conflicting YES+NO on same strike ──
# The 02:10 run placed YES on 73699; the 02:40 run would have placed NO on
# the same strike. With reconciliation, the 02:40 NO must be blocked.
print("=== Test 5: regression — 2026-04-15 04:00 scenario ===")
ticker_73699 = "KXBTCD-26APR1504-T73699.99"
# State after 02:10 run: hold 9 YES on the 73699 strike
open_pos = {ticker_73699: 9}
# 02:40 run proposes NO on the same strike (what actually happened)
opps = [opp(ticker_73699, "NO", 0.094)]
res = reconcile_and_cap(opps, open_pos)
assert len(res) == 0, f"BUG: opposing bet placed! {res}"
print(f"  PASS — opposing NO correctly blocked on held YES\n")


# ── Test 6: No open-position API data (empty dict) → no reconciliation ─────
print("=== Test 6: empty open_pos (API failure) — no filtering ===")
opps = [opp("A", "YES", 0.20), opp("B", "NO", 0.15)]
res = reconcile_and_cap(opps, {})
assert len(res) == 2
print(f"  PASS — {len(res)} opportunities pass through\n")


print("ALL TESTS PASSED ✓")
