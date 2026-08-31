from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.json"

STATE_DIM = 4
ACTION_DIM = 2
STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_file():
        candidate = ROOT / str(config_path).replace("\\", "/")
        if candidate.is_file():
            config_path = candidate
        else:
            raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolved_training_values(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gamma": float(config.get("gamma", 0.97)),
        "lr": float(config.get("learning_rate", 3e-4)),
        "batch_size": int(config.get("batch_size", 128)),
        "buffer_size": int(config.get("buffer_size", 100_000)),
        "tau": float(config.get("target_tau", config.get("tau", 5e-3))),
        "update_every": int(config.get("update_every", 4)),
        "eps_start": float(config.get("epsilon_start", 1.0)),
        "eps_end": float(config.get("epsilon_end", 0.01)),
        "eps_decay": float(config.get("epsilon_decay", 0.994)),
        "max_t": int(config.get("max_steps_per_episode", 500)),
    }


def apply_training_defaults(args, config: Dict[str, Any]) -> None:
    values = resolved_training_values(config)
    for key, value in values.items():
        attr = {
            "lr": "lr",
            "tau": "tau",
            "eps_start": "eps_start",
            "eps_end": "eps_end",
            "eps_decay": "eps_decay",
            "max_t": "max_t",
            "gamma": "gamma",
            "batch_size": "batch_size",
            "buffer_size": "buffer_size",
            "update_every": "update_every",
        }[key]
        if not hasattr(args, attr) or getattr(args, attr) is None:
            setattr(args, attr, value)
        # Prefer config when argparse used library defaults that match placeholders.
        current = getattr(args, attr, None)
        if current is None:
            setattr(args, attr, value)


def format_training_args(args) -> str:
    return (
        f"gamma={args.gamma} lr={args.lr} batch={args.batch_size} "
        f"buffer={args.buffer_size} tau={args.tau} update_every={args.update_every} "
        f"eps={args.eps_start}->{args.eps_end} decay={args.eps_decay} max_t={args.max_t}"
    )


def add_training_arguments(parser) -> None:
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--update-every", dest="update_every", type=int, default=None)
    parser.add_argument("--eps-start", dest="eps_start", type=float, default=None)
    parser.add_argument("--eps-end", dest="eps_end", type=float, default=None)
    parser.add_argument("--eps-decay", dest="eps_decay", type=float, default=None)
    parser.add_argument("--max-t", dest="max_t", type=int, default=None)
