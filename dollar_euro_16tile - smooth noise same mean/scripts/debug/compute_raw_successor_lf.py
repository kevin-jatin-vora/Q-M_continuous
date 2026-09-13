"""Compare local/global raw sampled-successor Lipschitz maxima.

This intentionally evaluates the older sample-pair quantity

    ||s1' - s2'|| / ||s1 - s2||

and does not use the fitted Wasserstein transition model. It is a diagnostic
comparison only: independently sampled successor noise remains in the ratio.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dollar_euro_lipschitz.layout import discover_neighbor_pairs, layout_summary
import train_reward_components as trc


def json_dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def gap_tag(value):
    text = format(float(value), ".10g").replace("-", "m").replace(".", "p")
    return f"raw_successor_lf_gap_{text}"


def empty_set():
    return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))


def load_records(npz, behavior):
    records = {}
    for category in range(1, 6):
        for action in range(4):
            prefix = f"r{behavior}_c{category}_a{action}"
            records[(category, action)] = (
                npz[f"{prefix}_states"],
                npz[f"{prefix}_next"],
                npz[f"{prefix}_rewards"],
            )
    return records


def raw_successor_max(
    set_a, set_b, max_pairs, seed, min_pair_distance, same_set=False,
):
    states_a = np.asarray(set_a[0], dtype=np.float64)
    states_b = np.asarray(set_b[0], dtype=np.float64)
    next_a = np.asarray(set_a[1], dtype=np.float64)
    next_b = np.asarray(set_b[1], dtype=np.float64)
    n_a, n_b = int(states_a.shape[0]), int(states_b.shape[0])
    empty = {
        "n_a": n_a,
        "n_b": n_b,
        "pairs_considered": 0,
        "pairs_rejected_below_radius": 0,
        "pairs_used": 0,
        "Lf": 0.0,
        "witness": None,
    }
    if n_a == 0 or n_b == 0:
        return empty

    rng = np.random.default_rng(seed + 13 * n_a + 29 * n_b)
    if same_set:
        idx_a, idx_b = trc.unique_unordered_pairs(n_a, max_pairs, rng)
    else:
        total = n_a * n_b
        if max_pairs is None or total <= int(max_pairs):
            idx_a = np.repeat(np.arange(n_a, dtype=np.int64), n_b)
            idx_b = np.tile(np.arange(n_b, dtype=np.int64), n_a)
        else:
            flat = rng.choice(total, size=int(max_pairs), replace=False)
            idx_a, idx_b = flat // n_b, flat % n_b

    source_distance = np.linalg.norm(states_a[idx_a] - states_b[idx_b], axis=1)
    pairs_considered = int(source_distance.size)
    valid = trc.pair_distance_mask(source_distance, min_pair_distance)
    pairs_rejected = int(pairs_considered - np.count_nonzero(valid))
    idx_a, idx_b = idx_a[valid], idx_b[valid]
    source_distance = source_distance[valid]
    if source_distance.size == 0:
        return {
            **empty,
            "pairs_considered": pairs_considered,
            "pairs_rejected_below_radius": pairs_rejected,
        }

    next_distance = np.linalg.norm(next_a[idx_a] - next_b[idx_b], axis=1)
    ratios = next_distance / source_distance
    pos = int(np.argmax(ratios))
    left, right = int(idx_a[pos]), int(idx_b[pos])
    return {
        "n_a": n_a,
        "n_b": n_b,
        "pairs_considered": pairs_considered,
        "pairs_rejected_below_radius": pairs_rejected,
        "pairs_used": int(source_distance.size),
        "Lf": float(ratios[pos]),
        "witness": {
            "index_a": left,
            "index_b": right,
            "source_state_a": states_a[left].tolist(),
            "source_state_b": states_b[right].tolist(),
            "next_state_a": next_a[left].tolist(),
            "next_state_b": next_b[right].tolist(),
            "source_distance": float(source_distance[pos]),
            "next_distance": float(next_distance[pos]),
            "pair_ratio": float(ratios[pos]),
        },
    }

def combine_results(first, second):
    chosen_behavior, chosen = ("R1", first) if first["Lf"] >= second["Lf"] else ("R2", second)
    return {
        "Lf1": first["Lf"],
        "Lf2": second["Lf"],
        "Lf": max(first["Lf"], second["Lf"]),
        "pairs_r1": first["pairs_used"],
        "pairs_r2": second["pairs_used"],
        "witness_behavior": chosen_behavior,
        "witness": chosen["witness"],
    }


def pooled_action_set(records, action):
    chunks = [records.get((category, action), empty_set()) for category in range(1, 6)]
    return tuple(np.concatenate([chunk[index] for chunk in chunks], axis=0) for index in range(3))


def print_maximum(label, row, gamma):
    print(
        f"{label}: Lf={row['Lf']:.12g} gamma*Lf={gamma * row['Lf']:.12g} "
        f"behavior={row['witness_behavior']}"
    )
    witness = row.get("witness")
    if witness:
        print(f"  source A/B: {witness['source_state_a']}  {witness['source_state_b']}")
        print(f"  next A/B:   {witness['next_state_a']}  {witness['next_state_b']}")
        print(
            f"  distances: source={witness['source_distance']:.12g} "
            f"next={witness['next_distance']:.12g}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    shared_dir = run_dir / "shared"
    required = (
        shared_dir / "raw_transitions.npz",
        shared_dir / "transition_bounds.json",
        shared_dir / "provenance.json",
    )
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing shared diagnostic data: " + ", ".join(missing))

    bounds = json.loads(required[1].read_text(encoding="utf-8"))
    provenance = json.loads(required[2].read_text(encoding="utf-8"))
    min_distance = trc.minimum_represented_pruning_radius(bounds)
    gamma = float(provenance["gamma"])
    tile_map = np.asarray(
        layout_summary(float(provenance["determinism"]))["tile_map_row_major"],
        dtype=np.int32,
    )

    with np.load(required[0], allow_pickle=True) as npz:
        records1, records2 = load_records(npz, 1), load_records(npz, 2)

    local_rows = []
    for category in range(1, 6):
        for action in range(4):
            set1 = records1[(category, action)]
            set2 = records2[(category, action)]
            result1 = raw_successor_max(
                set1, set1, args.max_pairs, args.seed + 101 * category + action,
                min_distance,
                same_set=True,
            )
            result2 = raw_successor_max(
                set2, set2, args.max_pairs, args.seed + 10_000 + 101 * category + action,
                min_distance,
                same_set=True,
            )
            local_rows.append({
                "category": category,
                "action": action,
                **combine_results(result1, result2),
            })

    neighbor_rows = []
    for pair in discover_neighbor_pairs(tile_map):
        category_i, category_j = pair["category_i"], pair["category_j"]
        for action in range(4):
            result1 = raw_successor_max(
                records1[(category_i, action)], records1[(category_j, action)],
                args.max_pairs, args.seed + 20_000 + 101 * category_i + 211 * category_j + action,
                min_distance,
            )
            result2 = raw_successor_max(
                records2[(category_i, action)], records2[(category_j, action)],
                args.max_pairs, args.seed + 30_000 + 101 * category_i + 211 * category_j + action,
                min_distance,
            )
            neighbor_rows.append({
                "category_i": category_i,
                "category_j": category_j,
                "action": action,
                "tile_edges": pair["tile_edges"],
                **combine_results(result1, result2),
            })

    global_rows = []
    for action in range(4):
        pooled1, pooled2 = pooled_action_set(records1, action), pooled_action_set(records2, action)
        result1 = raw_successor_max(
            pooled1, pooled1, args.max_pairs, args.seed + 40_000 + action,
            min_distance,
            same_set=True,
        )
        result2 = raw_successor_max(
            pooled2, pooled2, args.max_pairs, args.seed + 50_000 + action,
            min_distance,
            same_set=True,
        )
        global_rows.append({"action": action, **combine_results(result1, result2)})

    local_max = max(local_rows, key=lambda row: row["Lf"])
    neighbor_max = max(neighbor_rows, key=lambda row: row["Lf"]) if neighbor_rows else None
    global_max = max(global_rows, key=lambda row: row["Lf"])
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / gap_tag(min_distance)
    payload = {
        "definition": "max ||s1' - s2'|| / ||s1 - s2|| over raw sampled successor pairs",
        "warning": "Independent successor noise is included; this is not a Wasserstein-kernel estimate.",
        "gamma": gamma,
        "min_pair_distance": min_distance,
        "min_pair_distance_source": "minimum represented pruning radius",
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "local_within_category_action": local_rows,
        "local_neighbor_category_action": neighbor_rows,
        "global_pooled_by_action": global_rows,
        "summary": {
            "local_within_max": local_max,
            "local_neighbor_max": neighbor_max,
            "global_max": global_max,
        },
    }
    json_dump(payload, output_dir / "raw_successor_lf_max.json")

    print(f"Pair-distance threshold: {min_distance:.12g}")
    print_maximum("Local within-category max", local_max, gamma)
    if neighbor_max is not None:
        print_maximum("Local neighboring-category max", neighbor_max, gamma)
    print_maximum("Global pooled max", global_max, gamma)
    print(f"Saved {(output_dir / 'raw_successor_lf_max.json').resolve()}")


if __name__ == "__main__":
    main()
