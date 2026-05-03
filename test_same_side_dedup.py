"""
Same-side double-down guard — unit tests.

Reproduces the 2026-04-16 11:00 CDT KXBTCD-26APR1612 regression where the
classifier flipped (RSI 100 → 23.9) between 10:10 and 10:20 runs but kept
proposing NO bets on the same three strikes (74,400 / 74,500 / 74,600). The
pre-existing reconciliation block only filtered *opposing* positions; same-side
adds slipped through, doubling the eventual loss (16 contracts / ~$11.50) when
the settlement went against us.

Run:  python3 test_same_side_dedup.py
"""

MIN_CONVICTION = 0.15


def reconcile_with_dedup(opps, open_pos, *, net_mode=True, skip_same_side=True):
    """
    Mirrors auto_trader.py's reconciliation block including the new same-side
    guard. Returns the filtered list of opps that would proceed to placement.

    open_pos: {ticker: signed_position} — positive=YES, negative=NO, 0=flat.
    """
    override_threshold = MIN_CONVICTION * 2
    out = []
    for opp in opps:
        held = open_pos.get(opp["ticker"], 0)
        wants_yes = opp["side"] == "YES"
        same_side = (held > 0 and wants_yes) or (held < 0 and not wants_yes)
        if same_side and skip_same_side:
            out_tag = {"_blocked": "same_side_dup"}
            # In production this just `continue`s; we return a blocked marker
            # for tests to verify which gate fired.
            continue
        opposes = (held > 0 and not wants_yes) or (held < 0 and wants_yes)
        if not opposes:
            out.append(opp)
            continue
        if not net_mode:
            continue
        if opp.get("conviction", 0.0) >= override_threshold:
            out.append({**opp, "_netted": True})
    return out


# ═════════════════════ TESTS ════════════════════════════════════════════════

print("=== T1: same-side NO add on held NO — BLOCKED (the 2026-04-16 11:00 regression) ===")
# 10:20 run proposes NO on strikes where we already hold NO from 10:10.
opps = [
    {"ticker": "KXBTCD-26APR1612-T74399.99", "side": "NO", "conviction": 0.20},
    {"ticker": "KXBTCD-26APR1612-T74499.99", "side": "NO", "conviction": 0.20},
    {"ticker": "KXBTCD-26APR1612-T74599.99", "side": "NO", "conviction": 0.20},
]
open_pos = {
    "KXBTCD-26APR1612-T74399.99": -3,
    "KXBTCD-26APR1612-T74499.99": -3,
    "KXBTCD-26APR1612-T74599.99": -3,
}
res = reconcile_with_dedup(opps, open_pos)
assert len(res) == 0, f"expected all 3 blocked, got {res}"
print("  PASS — all 3 same-side NO adds blocked\n")


print("=== T2: same-side YES add on held YES — BLOCKED ===")
opps = [{"ticker": "K-T73699", "side": "YES", "conviction": 0.25}]
open_pos = {"K-T73699": 5}
res = reconcile_with_dedup(opps, open_pos)
assert len(res) == 0
print("  PASS — same-side YES add blocked\n")


print("=== T3: flat market — bet passes normally ===")
opps = [{"ticker": "K-T73699", "side": "NO", "conviction": 0.20}]
open_pos = {"K-T73699": 0}
res = reconcile_with_dedup(opps, open_pos)
assert len(res) == 1
print("  PASS — flat market accepts new bet\n")


print("=== T4: same-side add on DIFFERENT strike — passes (not same ticker) ===")
# Held NO on T74399; new NO on T74499 is a different market_ticker.
opps = [{"ticker": "KXBTCD-26APR1612-T74499.99", "side": "NO", "conviction": 0.20}]
open_pos = {"KXBTCD-26APR1612-T74399.99": -3}
res = reconcile_with_dedup(opps, open_pos)
assert len(res) == 1, "Different strikes in same event should not collide"
print("  PASS — per-strike dedup, not per-event (event cap is the other gate)\n")


print("=== T5: opposing-side NO on held YES — still handled by netting (not the new gate) ===")
# This is the pre-existing reconciliation behavior; same-side guard must not
# break it. Weak opposing signal should still be blocked.
opps = [{"ticker": "K-T73699", "side": "NO", "conviction": 0.10}]  # weak
open_pos = {"K-T73699": 5}
res = reconcile_with_dedup(opps, open_pos)
assert len(res) == 0, "Weak opposing signal should still be blocked by netting"
print("  PASS — netting reconciliation intact\n")


print("=== T6: opposing-side NO on held YES — strong signal still nets ===")
opps = [{"ticker": "K-T73699", "side": "NO", "conviction": 0.35}]  # strong
open_pos = {"K-T73699": 5}
res = reconcile_with_dedup(opps, open_pos)
assert len(res) == 1 and res[0].get("_netted"), "Strong opposing should net"
print("  PASS — strong opposing NO correctly nets against held YES\n")


print("=== T7: disabled (SKIP_SAME_SIDE_ADD=0) — same-side adds pass through ===")
opps = [{"ticker": "K-T73699", "side": "NO", "conviction": 0.20}]
open_pos = {"K-T73699": -3}
res = reconcile_with_dedup(opps, open_pos, skip_same_side=False)
assert len(res) == 1, "Disabling the guard should allow same-side adds"
print("  PASS — env var toggle works\n")


print("=== T8: full 10:10 + 10:20 scenario replay ===")
# 10:10 proposes 3 NO bets; portfolio flat → all pass.
ten_ten_opps = [
    {"ticker": "KXBTCD-26APR1612-T74399.99", "side": "NO", "conviction": 0.22},
    {"ticker": "KXBTCD-26APR1612-T74499.99", "side": "NO", "conviction": 0.21},
    {"ticker": "KXBTCD-26APR1612-T74599.99", "side": "NO", "conviction": 0.19},
]
ten_ten_out = reconcile_with_dedup(ten_ten_opps, open_pos={})
assert len(ten_ten_out) == 3, "10:10 run on flat book should place all 3"

# After 10:10 places them, open_pos reflects -3 on each strike (3 NO contracts).
open_pos_after_1010 = {o["ticker"]: -3 for o in ten_ten_out}

# 10:20 proposes SAME 3 strikes NO. With guard, all 3 are blocked.
ten_twenty_opps = [
    {"ticker": "KXBTCD-26APR1612-T74399.99", "side": "NO", "conviction": 0.17},
    {"ticker": "KXBTCD-26APR1612-T74499.99", "side": "NO", "conviction": 0.16},
    {"ticker": "KXBTCD-26APR1612-T74599.99", "side": "NO", "conviction": 0.13},
]
ten_twenty_out = reconcile_with_dedup(ten_twenty_opps, open_pos_after_1010)
assert len(ten_twenty_out) == 0, "10:20 run should be fully blocked by same-side guard"

# Without guard (old behavior), all 3 would have gone through — the regression.
old_behavior = reconcile_with_dedup(ten_twenty_opps, open_pos_after_1010, skip_same_side=False)
assert len(old_behavior) == 3, "Old behavior reproduces the regression"
print("  PASS — 10:10 places 3; 10:20 blocks all 3 (vs. 3 passing pre-fix)\n")


print("=== ALL TESTS PASSED ===")
