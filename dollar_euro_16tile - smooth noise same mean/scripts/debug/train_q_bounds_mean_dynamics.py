"""Debug-local entry point for the unchanged production Q-bound trainer."""

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pop_debug_method(argv):
    method = "debug_diagnostic"
    if "--debug-method" in argv:
        pos = argv.index("--debug-method")
        method = argv[pos + 1]
        del argv[pos:pos + 2]
    return method


if __name__ == "__main__":
    method = pop_debug_method(sys.argv)
    runpy.run_path(str(ROOT / "scripts" / "train_q_bounds.py"), run_name="__main__")
    out_ub = Path(sys.argv[sys.argv.index("--out-ub") + 1])
    if "--manifest" in sys.argv:
        manifest = Path(sys.argv[sys.argv.index("--manifest") + 1])
    else:
        lq_source = sys.argv[sys.argv.index("--lq-source") + 1] if "--lq-source" in sys.argv else "theoretical"
        manifest = out_ub.parent / f"q_bounds_{lq_source}_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["debug_diagnostic_method"] = method
        payload["debug_entry_point"] = "scripts/debug/train_q_bounds_mean_dynamics.py"
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
