import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Base Gaussian transition standard deviation (overrides config.environment.sigma).",
    )
    parser.add_argument(
        "--stochasticity-scale",
        type=float,
        default=None,
        help="Multiplier on config environment.sigma; covariance scales by its square. Do not combine with --sigma.",
    )
    parser.add_argument(
        "--allow-radius-sigma-mismatch",
        action="store_true",
        help="Allow fixed bounds estimated at another sigma (scientifically unsafe unless justified).",
    )


def resolve_sigma(
    config: Dict[str, Any],
    sigma: Optional[float],
    scale: Optional[float] = None,
) -> float:
    if sigma is not None and scale is not None:
        raise ValueError("Pass only one of --sigma or --stochasticity-scale.")
    configured = float(config["environment"]["sigma"])
    if sigma is not None:
        resolved = float(sigma)
    else:
        resolved = configured * float(1.0 if scale is None else scale)
    if resolved < 0.0:
        raise ValueError("Dynamics sigma must be nonnegative.")
    return resolved


def apply_resolved_sigma(config: Dict[str, Any], sigma: float) -> Dict[str, Any]:
    """Write the resolved sigma back so later readers cannot use a stale config value."""
    environment = dict(config.get("environment") or {})
    environment["sigma"] = float(sigma)
    config["environment"] = environment
    return config


def env_kwargs(config: Dict[str, Any], sigma: float) -> Dict[str, Any]:
    environment = config["environment"]
    return {
        "sigma": float(sigma),
        "step_size": float(environment.get("step_size", 0.04)),
        "tau": float(environment.get("tau", 2.4)),
        "horizon": int(environment.get("horizon", 200)),
    }


def bounds_reference_sigma(bounds_path) -> Optional[float]:
    with Path(bounds_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get("metadata", {}).get("dynamics_sigma")
    return None if value is None else float(value)


def lipschitz_reference_sigma(lipschitz_path) -> Optional[float]:
    with Path(lipschitz_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get("dynamics_sigma")
    return None if value is None else float(value)


def validate_pruning_provenance(bounds_path, lipschitz_path, runtime_sigma: float) -> None:
    """Require all data-derived pruning inputs to have matching sigma provenance."""
    bounds_sigma = bounds_reference_sigma(bounds_path)
    lipschitz_sigma = lipschitz_reference_sigma(lipschitz_path)
    if bounds_sigma is None:
        raise ValueError(f"{bounds_path} does not declare metadata.dynamics_sigma.")
    if lipschitz_sigma is None:
        raise ValueError(f"{lipschitz_path} does not declare dynamics_sigma.")
    tolerance = max(1e-15, abs(runtime_sigma) * 1e-9)
    if abs(bounds_sigma - runtime_sigma) > tolerance or abs(lipschitz_sigma - runtime_sigma) > tolerance:
        raise ValueError(
            "Pruning provenance mismatch: "
            f"runtime sigma={runtime_sigma:.10g}, bounds sigma={bounds_sigma:.10g}, "
            f"Lipschitz sigma={lipschitz_sigma:.10g}. Regenerate matching inputs."
        )


def validate_bounds_sigma(
    bounds_path,
    runtime_sigma: float,
    *,
    allow_mismatch: bool = False,
    context: str = "bounds-based analysis",
) -> Optional[float]:
    reference = bounds_reference_sigma(bounds_path)
    if reference is None:
        raise ValueError(
            f"{bounds_path} does not declare metadata.dynamics_sigma; "
            f"cannot validate {context}."
        )
    if abs(float(runtime_sigma) - reference) > max(1e-15, abs(reference) * 1e-9):
        message = (
            f"{context} requested sigma={runtime_sigma:.10g}, but fixed uncertainty "
            f"radii in {bounds_path} were estimated at sigma={reference:.10g}. "
            "Changing runtime noise does not recompute those radii."
        )
        if not allow_mismatch:
            raise ValueError(message + " Use matching bounds or explicitly pass --allow-radius-sigma-mismatch.")
        print(f"WARNING: {message}")
    return reference


# CLI dest -> (config key, fallback). `tau` is DQN Polyak, not environment.tau.
TRAINING_ARG_DEFAULTS = {
    "gamma": ("gamma", 0.99),
    "lr": ("learning_rate", 1e-3),
    "batch_size": ("batch_size", 256),
    "buffer_size": ("buffer_size", 100_000),
    "tau": ("tau", 5e-4),
    "update_every": ("update_every", 4),
    "eps_start": ("epsilon_start", 1.0),
    "eps_end": ("epsilon_end", 0.05),
    "eps_decay": ("epsilon_decay", 0.995),
    "max_t": ("max_steps_per_episode", 100),
}

_TRAINING_FLAGS = {
    "gamma": ("--gamma", float, "Discount factor (config gamma if omitted)"),
    "lr": ("--lr", float, "Adam learning rate (config learning_rate if omitted)"),
    "batch_size": ("--batch-size", int, "Minibatch size (config batch_size if omitted)"),
    "buffer_size": ("--buffer-size", int, "Replay capacity (config buffer_size if omitted)"),
    "tau": ("--tau", float, "DQN target Polyak tau; not environment living penalty"),
    "update_every": ("--update-every", int, "Learn every N env steps (config update_every if omitted)"),
    "eps_start": ("--eps-start", float, "Starting epsilon (config epsilon_start if omitted)"),
    "eps_end": ("--eps-end", float, "Floor epsilon (config epsilon_end if omitted)"),
    "eps_decay": ("--eps-decay", float, "Per-episode epsilon multiplier (config epsilon_decay if omitted)"),
    "max_t": ("--max-t", int, "Episode cutoff (config max_steps_per_episode if omitted)"),
}


def add_training_arguments(parser: argparse.ArgumentParser, *names: str) -> None:
    selected = names or tuple(_TRAINING_FLAGS)
    for name in selected:
        flag, typ, help_text = _TRAINING_FLAGS[name]
        parser.add_argument(flag, type=typ, default=None, help=help_text)


def artifact_sigma_tag(sigma: float) -> str:
    return format(float(sigma), ".0e").replace("+", "")


def _training_value(config: Dict[str, Any], attr: str):
    key, fallback = TRAINING_ARG_DEFAULTS[attr]
    if attr == "tau":
        return config.get("target_tau", config.get("tau", fallback))
    return config.get(key, fallback)


def apply_training_defaults(args, config: Dict[str, Any], *, role: str = "agent"):
    """Fill unset CLI training fields from config so JSON values are actually used.

    ``role="component"`` uses the original two-reward collection cutoffs
    (200-step episodes, epsilon decay 0.998) unless the flags were passed.
    """
    pending = {
        attr: getattr(args, attr) is None
        for attr in TRAINING_ARG_DEFAULTS
        if hasattr(args, attr)
    }
    for attr in TRAINING_ARG_DEFAULTS:
        if not hasattr(args, attr):
            continue
        if getattr(args, attr) is None:
            setattr(args, attr, _training_value(config, attr))
    if role == "component":
        if pending.get("max_t"):
            args.max_t = int(config.get("component_max_steps_per_episode", 200))
        if pending.get("eps_decay"):
            args.eps_decay = float(config.get("component_epsilon_decay", 0.998))
    policy = str(config.get("bounds_sigma_policy", "error")).lower()
    if hasattr(args, "allow_radius_sigma_mismatch") and not args.allow_radius_sigma_mismatch:
        if policy in {"allow", "warn"}:
            args.allow_radius_sigma_mismatch = True
    return args


def resolved_training_values(config: Dict[str, Any]) -> Dict[str, Any]:
    return {attr: _training_value(config, attr) for attr in TRAINING_ARG_DEFAULTS}


def format_training_args(args) -> str:
    parts = []
    for attr in TRAINING_ARG_DEFAULTS:
        if hasattr(args, attr):
            parts.append(f"{attr}={getattr(args, attr)}")
    return " ".join(parts)
