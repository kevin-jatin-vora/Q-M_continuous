import csv
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .bounds import (
    REGION_CENTER,
    allowed_mask,
    build_margin_table,
    require_finite_bellman_lq,
)
from .models import QNet


TABLE1_COLUMNS = [
    "Tile", "Reg", "Act", "Lr1", "Lr2", "Lr", "Lf1", "Lf2", "Lf",
    "Lq_empirical", "Lq_theoretical", "lr_source", "stochasticity",
]
TABLE2_COLUMNS = [
    "Reg", "Act", "n", "n_r1", "n_r2",
    "delta_mean", "delta_std", "student_t_half_width", "pruning_radius",
    "confidence_level", "stochasticity",
]


def _json_vector(values) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _atomic_write(path: Path, writer: Callable[[Path], None], attempts: int = 8) -> Path:
    """Write via a temp file, then replace. Retries Windows locks (Excel, Explorer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real suffix so matplotlib/csv consumers infer format correctly.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part{path.suffix}")
    last_error = None
    for attempt in range(attempts):
        try:
            writer(tmp)
            os.replace(tmp, path)
            return path
        except PermissionError as exc:
            last_error = exc
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(0.4 * (attempt + 1))
    fallback = path.with_name(f"{path.stem}_unlocked{path.suffix}")
    try:
        writer(fallback)
        print(f"warning: {path} is locked; wrote {fallback} instead")
        return fallback
    except Exception:
        if last_error is not None:
            raise last_error
        raise


def write_table1(lipschitz_path, output_path) -> Path:
    with Path(lipschitz_path).open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    sigma = source.get("dynamics_sigma")
    required = ["tile", "region", "Lr1", "Lr2", "Lr_sum", "Lf1", "Lf2", "Lf_sum", "Lq_empirical_sum", "LQ_bellman_bound"]
    if sigma is None:
        raise ValueError(f"{lipschitz_path} is missing authoritative dynamics_sigma.")
    gamma = float(source.get("gamma", 0.99))
    require_finite_bellman_lq(source["constants"], gamma, path=str(lipschitz_path))
    rows = []
    for entry in source["constants"]:
        missing = [name for name in required if name not in entry]
        if missing:
            raise ValueError(
                f"Cannot generate Table 1: tile={entry.get('tile')} region={entry.get('region')} "
                f"action={entry.get('action')} is missing {missing}."
            )
        rows.append({
            "Tile": int(entry["tile"]),
            "Reg": int(entry["region"]),
            "Act": int(entry["action"]),
            "Lr1": float(entry["Lr1"]),
            "Lr2": float(entry["Lr2"]),
            "Lr": float(entry["Lr_sum"]),
            "Lf1": float(entry["Lf1"]),
            "Lf2": float(entry["Lf2"]),
            "Lf": float(entry["Lf_sum"]),
            "Lq_empirical": float(entry["Lq_empirical_sum"]),
            "Lq_theoretical": float(entry["LQ_bellman_bound"]),
            "lr_source": str(entry.get("lr_source", "tile")),
            "stochasticity": float(sigma),
        })
    def _write(tmp: Path):
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE1_COLUMNS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["Tile"], row["Act"])))

    return _atomic_write(output_path, _write)


def write_table2(bounds_path, output_path) -> Path:
    with Path(bounds_path).open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    metadata = source.get("metadata", {})
    config = source.get("config", {})
    sigma = metadata.get("dynamics_sigma")
    if sigma is None:
        raise ValueError(f"{bounds_path} is missing metadata.dynamics_sigma.")
    confidence_level = float(config.get("confidence_level", 0.95))
    rows = []
    for region, action_map in source["by_region_action"].items():
        for action, entry in action_map.items():
            rows.append({
                "Reg": int(region),
                "Act": int(action),
                "n": int(entry["n_total"]),
                "n_r1": int(entry.get("n_q1_behavior", 0)),
                "n_r2": int(entry.get("n_q2_behavior", 0)),
                "delta_mean": _json_vector(entry["delta_mean"]),
                "delta_std": _json_vector(entry["delta_std"]),
                "student_t_half_width": _json_vector(entry["student_t_half_width"]),
                "pruning_radius": float(entry["pruning_radius"]),
                "confidence_level": confidence_level,
                "stochasticity": float(sigma),
            })
    def _write(tmp: Path):
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE2_COLUMNS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["Reg"], row["Act"])))

    return _atomic_write(output_path, _write)


def _margin_table(bounds_path, lipschitz_path, lq_source, gamma, legacy_margin, determinism=None):
    table, _tile_map = build_margin_table(
        bounds_path,
        lipschitz_path,
        lq_source,
        gamma,
        legacy_margin,
        determinism=determinism,
    )
    return table


def generate_pruning_heatmap(
    q_path,
    bounds_path,
    lipschitz_path,
    output_prefix,
    *,
    lq_source: str = "empirical",
    gamma: float = 0.99,
    tol: float = 1e-5,
    grid_size: int = 201,
    legacy_margin: bool = False,
    clamp: bool = True,
    determinism: Optional[float] = None,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
) -> List[Path]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNet().to(device)
    model.load_state_dict(torch.load(q_path, map_location=device))
    model.eval()

    coordinates = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    states_np = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    states = torch.as_tensor(states_np, device=device)
    with torch.inference_mode():
        q_values = model(states)

    from dollar_euro_lipschitz.layout import tile_ids_from_states
    from dollar_euro_lipschitz.bounds import _DEFAULT_DETERMINISM

    det = _DEFAULT_DETERMINISM if determinism is None else float(determinism)
    tile_indices = torch.as_tensor(tile_ids_from_states(states_np), device=device, dtype=torch.long)
    margins = torch.as_tensor(
        _margin_table(bounds_path, lipschitz_path, lq_source, gamma, legacy_margin, determinism=det),
        dtype=torch.float32,
        device=device,
    )[tile_indices]
    remaining = (
        allowed_mask(q_values, margins, tol=tol, clamp=clamp)
        .sum(dim=1)
        .cpu()
        .numpy()
        .reshape(grid_size, grid_size)
    )

    figure, axis = plt.subplots(figsize=(6.7, 5.7))
    image = axis.imshow(
        remaining,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="viridis",
        vmin=0,
        vmax=4,
        interpolation="nearest",
        aspect="equal",
    )
    # 4x4 tile grid
    for edge in (0.25, 0.5, 0.75):
        axis.axvline(edge, color="white", linestyle="--", linewidth=0.6, alpha=0.8)
        axis.axhline(edge, color="white", linestyle="--", linewidth=0.6, alpha=0.8)
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3, 4])
    colorbar.set_label("Actions remaining after pruning")
    axis.set(xlabel="State x", ylabel="State y")
    axis.set_title(f"RA-DQN pruning map ({lq_source} $L_q$)")
    figure.tight_layout()

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for image_format in formats:
        image_format = image_format.lower().lstrip(".")
        if image_format not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported heatmap format: {image_format}")
        output = prefix.with_suffix(f".{image_format}")

        def _save(tmp: Path):
            figure.savefig(tmp, dpi=dpi, bbox_inches="tight", format=image_format)

        outputs.append(_atomic_write(output, _save))
    plt.close(figure)
    return outputs
