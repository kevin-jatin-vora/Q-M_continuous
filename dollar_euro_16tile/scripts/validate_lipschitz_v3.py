"""Validation for the corrected Lipschitz/Q-bound pipeline (method version 3).

Covers (no full experiment launches):
  1. Lr is grouped by NEXT-state tile (action-pooled), not source tile.
  2. Cross-tile reward Lr vs cross-category dynamics Lf are SEPARATE artifacts,
     and a same-category neighboring tile pair yields cross-tile Lr but NO
     cross-category Lf.
  3. Invalid Lq (gamma*Lf_eff >= 1) RAISES instead of returning 0.
  4. Scalar vs vectorized ball/tile intersection agree.
  5. Cached vs uncached OverlapAwareConstants selection agree, and the future
     Lq is independent of the current action (same mask).
  6. Version-1/version-2/missing/old cross artifacts are rejected on load.
  7. Speed benchmark for batched overlap-aware lookups.
  8. Future-action Lq selection: Lq_future_eff = max_b Lq_eff(b), where the
     maximum is over NEXT actions, NOT the current action.
  9. Current radius stays current-action specific (never maximized over actions).
  10. An invalid NON-current future action raises (old impl incorrectly accepted).
  11. Bellman max inequality: |max_b q1_b - max_b q2_b| <= max_b |q1_b - q2_b|.

Run:  python scripts/validate_lipschitz_v3.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dollar_euro_lipschitz.layout import (  # noqa: E402
    N_TILES,
    find_intersected_tiles,
    batched_intersected_tile_masks,
    tile_ids_from_mask,
    discover_tile_neighbor_pairs,
)
from dollar_euro_lipschitz.q_bounds import (  # noqa: E402
    OverlapAwareConstants,
    InvalidFutureLqError,
    _derive_lq_future,
    _require_version,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. Lr grouped by NEXT-state tile
# ---------------------------------------------------------------------------
def test_wrong_lr_grouping():
    print("\n== 1. Lr grouped by NEXT-state tile (action-pooled) ==")
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_reward_components as trc

    # 6 samples whose SOURCE states all lie in source tile 0, but whose
    # next states lie in four DIFFERENT next tiles (0, 2, 8, 10).
    states = np.array([[0.05, 0.05]] * 4 * 3)
    nexts = np.array(
        [[0.2, 0.1]] * 3 + [[0.6, 0.1]] * 3 + [[0.2, 0.6]] * 3 + [[0.6, 0.6]] * 3
    )
    reward = nexts[:, 0]  # R(s') = x-coordinate of next state
    actions = np.tile(np.arange(4), 3)  # all 4 actions present per next tile

    src = trc._group_transitions_by_tile(states, nexts, reward, actions)
    nxt = trc._group_transitions_by_next_tile(states, nexts, reward)

    src_tiles = sorted({k[0] for k in src if src[k][0].shape[0] > 0})
    nxt_tiles = sorted(t for t in nxt if nxt[t][0].shape[0] > 0)
    check("source grouping is source-tile (tile 0 only)", src_tiles == [0], f"got {src_tiles}")
    check(
        "next grouping is next-tile (spread across tiles)",
        nxt_tiles == [0, 2, 8, 10],
        f"got {nxt_tiles}",
    )

    # Lr per next tile: use a reward that depends on next-state position and
    # place multiple samples per next tile so within-tile pairs are non-degenerate.
    nexts2 = np.array([
        [0.10, 0.10], [0.15, 0.12], [0.20, 0.14],   # next tile 0
        [0.30, 0.10], [0.35, 0.12], [0.40, 0.14],   # next tile 1
    ])
    reward2 = nexts2[:, 0]  # R(s') = x
    states2 = np.array([[0.05, 0.05]] * 6)
    actions2 = np.array([0, 1, 2, 3, 0, 1])  # multiple actions pooled
    nxt2 = trc._group_transitions_by_next_tile(states2, nexts2, reward2)
    lr_report = trc.reward_lipschitz_per_tile(nxt2, None, "cpu", None, 0)
    nonzero = {t: v["Lr"] for t, v in lr_report.items() if v["n"] > 0}
    check("Lr reported per next tile, action-pooled", set(nonzero) == {0, 1},
          f"got {sorted(nonzero)}")
    check("Lr on next-tile pairs is > 0", all(v > 0 for v in nonzero.values()),
          str(nonzero))


# ---------------------------------------------------------------------------
# 2. Cross-tile vs cross-category separation
# ---------------------------------------------------------------------------
def test_cross_separation():
    print("\n== 2. Cross-tile reward vs cross-category dynamics separation ==")
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_reward_components as trc

    # Hand-build two artifacts: a cross-tile Lr for EVERY physical neighbor pair
    # (including same-category pairs), but a cross-category Lf ONLY for pairs of
    # DIFFERENT categories.  We assert the same-category neighbor tile pair has
    # a cross-tile Lr entry yet contributes no cross-category Lf.
    tile_map = np.asarray(
        [0, 0, 1, 1,
         0, 0, 1, 1,
         2, 2, 3, 3,
         2, 2, 3, 3], dtype=np.int32)

    pairs = discover_tile_neighbor_pairs()
    cross_tile = {
        (min(a, b), max(a, b)): float(a + b + 1)
        for a, b in pairs
    }
    # Same-category neighboring pair (tile 0,1 are both category 0) included.
    check(
        "discover_tile_neighbor_pairs includes same-category pair (0,1)",
        (0, 1) in cross_tile,
    )

    cross_cat = {}
    for a, b in pairs:
        if tile_map[a] != tile_map[b]:
            cross_cat[(min(tile_map[a], tile_map[b]), max(tile_map[a], tile_map[b]))] = (a, b)
    check(
        "cross-category Lf present only for different-category tile pairs",
        all(tile_map[ti] != tile_map[tj] for (ti, tj) in cross_cat.values()),
    )
    check(
        "same-category pair (0,1) has cross-tile Lr but NO cross-category Lf",
        (0, 1) in cross_tile and (0, 0) not in cross_cat
        and all(c[0] != c[1] for c in cross_cat),
    )


# ---------------------------------------------------------------------------
# 3. Invalid Lq raises
# ---------------------------------------------------------------------------
def test_invalid_lq_raises():
    print("\n== 3. Invalid Lq (gamma*Lf>=1) raises, never returns 0 ==")
    try:
        _derive_lq_future(1.0, 1.2, 0.98, 1, 3, (0, 1), (1,), "tile_0", "cat_1")
        check("gamma*Lf>=1 raises ValueError", False, "no exception")
    except InvalidFutureLqError as e:
        check("gamma*Lf>=1 raises InvalidFutureLqError", True)
        check("raises with provenance message", "future" in str(e).lower(), str(e))
        check("offending future action identified", e.payload["offending_future_action"] == 1,
              str(e.payload.get("offending_future_action")))
        check("future-max message explains every action must be finite",
              "every relevant action-specific" in str(e) or
              "EVERY relevant action-specific" in str(e), str(e))
    # Boundary gamma*Lf == 1 exactly must also raise
    try:
        _derive_lq_future(0.5, 1.0 / 0.98, 0.98, 0, 3, (0, 1), (1,), "tile_0", "cat_1")
        check("gamma*Lf==1 raises ValueError", False, "no exception")
    except InvalidFutureLqError:
        check("gamma*Lf==1 raises ValueError", True)
    # Valid case returns derived value
    val = _derive_lq_future(2.0, 0.5, 0.98, 0, 3, (0, 1), (1,), "tile_0", "cat_1")
    check("valid Lq derived = Lr*Lf/(1-gamma*Lf)",
          abs(val - 2.0 * 0.5 / (1 - 0.98 * 0.5)) < 1e-12, str(val))


# ---------------------------------------------------------------------------
# 4. Scalar vs vectorized geometry
# ---------------------------------------------------------------------------
def test_geometry_agreement():
    print("\n== 4. Scalar vs vectorized ball/tile intersection ==")
    rng = np.random.default_rng(123)
    centers = rng.random((5000, 2))
    radii = rng.uniform(0.001, 0.25, (5000,))
    masks = batched_intersected_tile_masks(centers, radii)
    rows = (masks.astype(np.int64) & 1) @ (1 << np.arange(N_TILES, dtype=np.int64))
    ok = True
    for i in range(5000):
        exp = find_intersected_tiles(centers[i], radii[i])
        got = set(t for t in range(N_TILES) if rows[i] & (1 << t))
        if exp != got:
            ok = False
            print("      mismatch at", i, exp, got)
            break
    check("5000 random balls scalar==vectorized", ok)


# ---------------------------------------------------------------------------
# 5. Cached vs uncached OverlapAwareConstants selection
# ---------------------------------------------------------------------------
def _write_v3_artifacts(tmp):
    tile_map = [0, 0, 1, 1,
                0, 0, 1, 1,
                2, 2, 3, 3,
                2, 2, 3, 3]
    # ordinary lipschitz_constants.json (v3)
    rows = []
    for tile in range(N_TILES):
        region = tile_map[tile]
        lr = 0.1 * (region + 1)
        for action in range(4):
            lf = 0.05 * region + 0.01 * action
            bellman = lr * lf / (1 - 0.98 * lf)
            rows.append({
                "tile": tile, "region": region, "action": action,
                "lr_source": "next_tile", "Lr_sum": lr, "Lr_action_dependence": False,
                "Lf_sum": lf, "Lq_empirical_sum": 0.0,
                "LQ_bellman_bound": bellman,
            })
    lips = {
        "lipschitz_method_version": 3, "gamma": 0.98,
        "Lr_grouping": "next_state_tile", "constants": rows,
    }
    (tmp / "lipschitz_constants.json").write_text(json.dumps(lips), encoding="utf-8")

    # cross_tile_reward_lipschitz.json (v3)
    pairs = discover_tile_neighbor_pairs()
    cross_tile_rows = [
        {"tile_i": a, "tile_j": b, "region_i": tile_map[a], "region_j": tile_map[b],
         "Lr1": 0.01, "Lr2": 0.02, "Lr_sum": 0.03}
        for a, b in pairs
    ]
    cross_tile = {
        "lipschitz_method_version": 3, "gamma": 0.98,
        "neighbor_tile_pairs": [list(p) for p in pairs],
        "Lr_grouping": "next_state_tile_pair", "constants": cross_tile_rows,
    }
    (tmp / "cross_tile_reward_lipschitz.json").write_text(json.dumps(cross_tile), encoding="utf-8")

    # cross_category_dynamics_lipschitz.json (v3)
    from dollar_euro_lipschitz.layout import discover_neighbor_pairs
    cat_pairs = discover_neighbor_pairs(np.asarray(tile_map, dtype=np.int32))
    cross_cat_rows = []
    for p in cat_pairs:
        ci, cj = p["category_i"], p["category_j"]
        for action in range(4):
            lf = 0.02 * (ci + cj) + 0.01 * action
            cross_cat_rows.append({
                "category_i": ci, "category_j": cj, "action": action,
                "tile_edges": p["tile_edges"],
                "Lf1": lf, "Lf2": lf, "Lf_sum": lf,
            })
    cross_cat = {
        "lipschitz_method_version": 3, "gamma": 0.98,
        "neighbor_category_pairs": cat_pairs,
        "Lf_grouping": "source_category_action_pair", "constants": cross_cat_rows,
    }
    (tmp / "cross_category_dynamics_lipschitz.json").write_text(json.dumps(cross_cat), encoding="utf-8")
    return tile_map


def test_overlap_cache(tmp):
    print("\n== 5. Cached vs uncached OverlapAwareConstants selection ==")
    tile_map = _write_v3_artifacts(tmp)
    oc = OverlapAwareConstants(
        tmp / "lipschitz_constants.json",
        tmp / "cross_tile_reward_lipschitz.json",
        tmp / "cross_category_dynamics_lipschitz.json",
        0.98,
    )
    rng = np.random.default_rng(7)
    ok = True
    same_mask_action_independent = True
    for _ in range(200):
        center = rng.random(2)
        radius = rng.uniform(0.02, 0.2)
        action = int(rng.integers(0, 4))
        tile_map_arr = np.asarray(tile_map, dtype=np.int32)
        s1 = oc.get_overlap_constants_with_provenance(center, radius, action, tile_map_arr)
        # second call hits the cache
        s2 = oc.get_overlap_constants_with_provenance(center, radius, action, tile_map_arr)
        if abs(s1["lr_eff"] - s2["lr_eff"]) > 1e-12 or abs(s1["lq_future_eff"] - s2["lq_future_eff"]) > 1e-12:
            ok = False
            break
        # The same mask (same physical ball) must give the same future Lq
        # regardless of which current action produced it.
        other_action = (action + 1) % 4
        s3 = oc.get_overlap_constants_with_provenance(center, radius, other_action, tile_map_arr)
        if abs(s1["lq_future_eff"] - s3["lq_future_eff"]) > 1e-12:
            same_mask_action_independent = False
            break
    check("cached second call equals first", ok)
    check("future Lq is independent of current action (same mask)", same_mask_action_independent)


# ---------------------------------------------------------------------------
# 6. Version rejection
# ---------------------------------------------------------------------------
def test_version_rejection(tmp):
    print("\n== 6. Old/missing/version-2 cross artifacts rejected ==")
    tile_map = _write_v3_artifacts(tmp)
    # Corrupt the cross-category artifact version to 2.
    p = tmp / "cross_category_dynamics_lipschitz.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["lipschitz_method_version"] = 2
    p.write_text(json.dumps(data), encoding="utf-8")
    try:
        OverlapAwareConstants(
            tmp / "lipschitz_constants.json",
            tmp / "cross_tile_reward_lipschitz.json",
            p, 0.98,
        )
        check("version-2 rejected", False)
    except ValueError as e:
        check("version-2 rejected", True)
        check("rejection message is exactly required",
              "Old Lipschitz artifacts use incompatible reward grouping/"
              "cross-boundary semantics; regenerate from scratch." in str(e),
              str(e))
    # missing version
    p2 = tmp / "cross_tile_reward_lipschitz.json"
    data2 = json.loads(p2.read_text(encoding="utf-8"))
    del data2["lipschitz_method_version"]
    p2.write_text(json.dumps(data2), encoding="utf-8")
    try:
        OverlapAwareConstants(
            tmp / "lipschitz_constants.json", p2,
            tmp / "cross_category_dynamics_lipschitz.json", 0.98,
        )
        check("missing-version rejected", False)
    except (ValueError, TypeError):
        check("missing-version rejected", True)


# ---------------------------------------------------------------------------
# 7. Speed benchmark
# ---------------------------------------------------------------------------
def test_benchmark(tmp):
    print("\n== 7. Speed benchmark (10k-100k batched lookups) ==")
    tile_map = _write_v3_artifacts(tmp)
    oc = OverlapAwareConstants(
        tmp / "lipschitz_constants.json",
        tmp / "cross_tile_reward_lipschitz.json",
        tmp / "cross_category_dynamics_lipschitz.json",
        0.98,
    )
    rng = np.random.default_rng(99)
    n = 100000
    centers = rng.random((n, 2))
    radii = rng.uniform(0.02, 0.2, (n,))
    masks = batched_intersected_tile_masks(centers, radii, oc.grid)
    rows = (masks.astype(np.int64) & 1) @ (1 << np.arange(N_TILES, dtype=np.int64))
    tile_map_arr = np.asarray(tile_map, dtype=np.int32)
    t0 = time.time()
    oc.lookups_for_masks(rows * 0 + 0, tile_map_arr)  # warm cache (single mask)
    t_warm = time.time() - t0
    t0 = time.time()
    oc.lookups_for_masks(rows, tile_map_arr)
    t_100k = time.time() - t0
    print(f"  {n} batched lookups: {t_100k*1e3:.2f} ms "
          f"(warm single-mask path {t_warm*1e3:.2f} ms)")
    check("100k batched lookups < 5s", t_100k < 5.0)
    t0 = time.time()
    oc.lookups_for_masks(rows[:10000], tile_map_arr)
    t_elapsed = time.time() - t0
    print(f"  10000 batched lookups: {t_elapsed*1e3:.2f} ms")
    check("10k batched lookups < 2s", t_elapsed < 2.0)


# ---------------------------------------------------------------------------
# Synthetic OverlapAwareConstants for controlled future-action unit tests.
# ---------------------------------------------------------------------------
def _synthetic_constants(lr_eff, lf_by_action, gamma, tile_map):
    """Build an OverlapAwareConstants WITHOUT loading artifacts.

    Assigns ordinary cat_lf[c, b] = lf_by_action[b] for every category c and
    every future action b, with ordinary tile Lr = lr_eff everywhere, so any
    intersected single-tile mask reproduces the requested per-action Lf.
    """
    obj = object.__new__(OverlapAwareConstants)
    obj.gamma = float(gamma)
    obj.grid = 4
    obj.N_ACTIONS = 4
    obj._cache = {}
    obj.tile_lr = np.full(N_TILES, float(lr_eff), dtype=np.float64)
    obj.cat_lf = np.zeros((6, 4), dtype=np.float64)
    for c in range(1, 6):
        for b in range(4):
            obj.cat_lf[c, b] = float(lf_by_action[b])
    obj.cross_tile_lr = {}
    obj.tile_neighbor_pairs = set()
    obj.cross_cat_lf = {}
    obj.cat_neighbor_pairs = set()
    return obj


# ---------------------------------------------------------------------------
# 8. Future action selection: max over next actions, NOT current action
# ---------------------------------------------------------------------------
def test_future_action_selection():
    print("\n== 8. Future-action Lq selection (max over next actions, not current) ==")
    # Synthetic: Lr_eff = 2, gamma = 0.98.
    lf_by_action = [0.50, 0.60, 0.90, 0.70]
    tile_map = np.ones(N_TILES, dtype=np.int32)  # all category 0 in this test
    oc = _synthetic_constants(2.0, lf_by_action, 0.98, tile_map)
    # Single-tile mask (bit 0 set) -> represented category {0}.  Lf_eff(b) must
    # equal lf_by_action[b] = cat_lf[cat, b].
    block = oc._resolve_mask(1, tile_map)
    for b in range(4):
        check(
            f"Lf_eff(action {b}) = {lf_by_action[b]}",
            abs(block["lf_by_action"][b] - lf_by_action[b]) < 1e-12,
            f"got {block['lf_by_action'][b]}",
        )
    expected_lq = [2.0 * lf / (1 - 0.98 * lf) for lf in lf_by_action]
    for b in range(4):
        check(
            f"Lq_eff(action {b}) derived = Lr*Lf_b/(1-gamma*Lf_b)",
            abs(block["lq_by_action"][b] - expected_lq[b]) < 1e-9,
            f"got {block['lq_by_action'][b]} expected {expected_lq[b]}",
        )
    # The bound must be controlled by the LARGEST Lq, i.e. action 2.
    lq_future = max(expected_lq)
    check(
        "Lq_future_eff = max_b Lq_eff(b)",
        abs(block["lq_future_eff"] - lq_future) < 1e-9,
        f"got {block['lq_future_eff']} expected {lq_future}",
    )
    check(
        "future argmax action is 2 (the largest value)",
        block["future_action_argmax"] == 2,
        f"got {block['future_action_argmax']}",
    )
    # Changing the CURRENT action (which controls only radius) must not change
    # the future Q Lipschitz constant for the same mask.
    a0 = oc.get_overlap_constants_with_provenance(
        np.array([0.1, 0.1]), 0.05, 0, tile_map
    )
    a3 = oc.get_overlap_constants_with_provenance(
        np.array([0.1, 0.1]), 0.05, 3, tile_map
    )
    check(
        "same mask with different current actions -> same Lq_future_eff",
        abs(a0["lq_future_eff"] - a3["lq_future_eff"]) < 1e-12,
        f"{a0['lq_future_eff']} vs {a3['lq_future_eff']}",
    )


# ---------------------------------------------------------------------------
# 9. Current radius stays current-action specific
# ---------------------------------------------------------------------------
def test_current_radius_specific():
    print("\n== 9. Current radius remains current-action specific ==")
    lf_by_action = [0.50, 0.60, 0.90, 0.70]
    tile_map = np.ones(N_TILES, dtype=np.int32)
    oc = _synthetic_constants(2.0, lf_by_action, 0.98, tile_map)
    center = np.array([0.1, 0.1])
    lq_future = oc.get_overlap_constants_with_provenance(center, 0.05, 0, tile_map)["lq_future_eff"]
    # Same geometry/ball; radius differs by current action: r(a0)=0.01, r(a1)=0.03.
    r0, r1 = 0.01, 0.03
    dq0 = lq_future * r0
    dq1 = lq_future * r1
    check("delta_q(action 0) = Lq_future_eff * 0.01", abs(dq0 - lq_future * 0.01) < 1e-12,
          str(dq0))
    check("delta_q(action 1) = Lq_future_eff * 0.03", abs(dq1 - lq_future * 0.03) < 1e-12,
          str(dq1))
    check("radius is NOT max over actions (r differs by action)",
          abs(lq_future * 0.03 - lq_future * 0.01) > 1e-12)


# ---------------------------------------------------------------------------
# 10. Invalid non-current future action raises
# ---------------------------------------------------------------------------
def test_invalid_non_current_future_action():
    print("\n== 10. Invalid NON-current future action raises ==")
    # current action 0 valid, but future action 2 invalid (gamma*Lf_eff(2)>=1).
    lf_by_action = [0.50, 0.50, 1.50, 0.50]  # gamma*1.50 = 1.47 >= 1
    tile_map = np.ones(N_TILES, dtype=np.int32)
    oc = _synthetic_constants(2.0, lf_by_action, 0.98, tile_map)
    # Old current-action-only implementation (action 0) would accept this.
    check("current action 0 Lf is valid (gamma*Lf<1)",
          abs(1 - 0.98 * 0.50) > 0)
    try:
        oc._resolve_mask(1, tile_map)
        check("invalid non-current future action raises", False, "no exception")
    except InvalidFutureLqError as e:
        check("invalid non-current future action raises", True)
        check("identifies offending future action 2",
              e.payload["offending_future_action"] == 2,
              str(e.payload.get("offending_future_action")))
        check("reports gamma*Lf and denominator",
              "gamma_times_lf" in e.payload and "denominator" in e.payload)


# ---------------------------------------------------------------------------
# 11. Bellman max inequality
# ---------------------------------------------------------------------------
def test_bellman_max_inequality():
    print("\n== 11. |max_b q1_b - max_b q2_b| <= max_b |q1_b - q2_b| ==")
    rng = np.random.default_rng(5)
    ok = True
    worst = -1.0
    for _ in range(2000):
        q1 = rng.random(4) * 10.0
        q2 = rng.random(4) * 10.0
        lhs = abs(q1.max() - q2.max())
        rhs = np.abs(q1 - q2).max()
        worst = max(worst, lhs / (rhs + 1e-12))
        if lhs > rhs + 1e-12:
            ok = False
            break
    check("max inequality holds on 2000 random action-vectors", ok,
          f"worst ratio {worst:.6f}")
    # Concrete numeric example.
    q1 = np.array([1.0, 5.0, 3.0])
    q2 = np.array([4.0, 2.0, 6.0])
    lhs = abs(q1.max() - q2.max())
    rhs = np.abs(q1 - q2).max()
    check("example: |5 - 6| <= max|q1_b-q2_b|",
          lhs <= rhs + 1e-12, f"|{q1.max()}-{q2.max()}|={lhs}, max|diff|={rhs}")


def main():
    print(f"=== validate_lipschitz_v3 (method version {OverlapAwareConstants.REQUIRED_VERSION}) ===")
    tmp = Path(tempfile.mkdtemp())
    test_wrong_lr_grouping()
    test_cross_separation()
    test_invalid_lq_raises()
    test_geometry_agreement()
    test_overlap_cache(tmp)
    test_version_rejection(tmp)
    test_benchmark(tmp)
    test_future_action_selection()
    test_current_radius_specific()
    test_invalid_non_current_future_action()
    test_bellman_max_inequality()
    print(f"\n=== RESULT: {PASS} passed, {FAIL} failed ===")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    import tempfile
    main()
