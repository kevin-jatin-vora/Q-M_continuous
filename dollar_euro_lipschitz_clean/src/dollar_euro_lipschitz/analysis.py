import csv
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .bounds import (
    LipschitzConstants,
    REGION_CENTER,
    RegionActionBounds,
    action_margin,
    allowed_mask,
    require_finite_bellman_lq,
)
from .models import QNet


TABLE1_COLUMNS = [
    "Reg", "Act", "Lr1", "Lr2", "Lr", "Lf1", "Lf2", "Lf",
    "Lq_theoretical", "stochasticity",
]
TABLE2_COLUMNS = [
    "Reg", "Act", "n", "n_r1", "n_r2", "n_fit", "n_calibration",
    "mean_delta", "mean_conf_radius_vec", "mean_conf_radius",
    "predictive_radius_vec", "predictive_radius", "pruning_radius_semantics",
    "stochasticity",
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
    required = ["Lr1", "Lr2", "Lr_sum", "Lf1", "Lf2", "Lf_sum", "LQ_bellman_bound"]
    if sigma is None:
        raise ValueError(f"{lipschitz_path} is missing authoritative dynamics_sigma.")
    gamma = float(source.get("gamma", 0.99))
    require_finite_bellman_lq(source["constants"], gamma, path=str(lipschitz_path))
    rows = []
    for entry in source["constants"]:
        missing = [name for name in required if name not in entry]
        if missing:
            raise ValueError(
                f"Cannot generate Table 1: region={entry.get('region')} action={entry.get('action')} "
                f"is missing {missing}."
            )
        rows.append({
            "Reg": int(entry["region"]),
            "Act": int(entry["action"]),
            "Lr1": float(entry["Lr1"]),
            "Lr2": float(entry["Lr2"]),
            "Lr": float(entry["Lr_sum"]),
            "Lf1": float(entry["Lf1"]),
            "Lf2": float(entry["Lf2"]),
            "Lf": float(entry["Lf_sum"]),
            "Lq_theoretical": float(entry["LQ_bellman_bound"]),
            "stochasticity": float(sigma),
        })
    def _write(tmp: Path):
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE1_COLUMNS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["Reg"], row["Act"])))

    return _atomic_write(output_path, _write)


def write_table2(bounds_path, output_path) -> Path:
    with Path(bounds_path).open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    metadata = source.get("metadata", {})
    sigma = metadata.get("dynamics_sigma")
    if sigma is None:
        raise ValueError(f"{bounds_path} is missing metadata.dynamics_sigma.")
    semantics = metadata.get("radius_scalar_semantics", "unspecified")
    rows = []
    for region, action_map in source["by_region_action"].items():
        for action, entry in action_map.items():
            rows.append({
                "Reg": int(region),
                "Act": int(action),
                "n": int(entry["n_total"]),
                "n_r1": int(entry.get("n_q1_behavior", 0)),
                "n_r2": int(entry.get("n_q2_behavior", 0)),
                "n_fit": int(entry.get("n_fit", entry["n_total"])),
                "n_calibration": int(entry.get("n_calibration", 0)),
                "mean_delta": _json_vector(entry["mean_delta"]),
                "mean_conf_radius_vec": _json_vector(entry["student_t_conf_radius_mean_delta"]),
                "mean_conf_radius": float(entry.get(
                    "mean_radius_scalar",
                    np.linalg.norm(entry["student_t_conf_radius_mean_delta"]),
                )),
                "predictive_radius_vec": _json_vector(entry.get(
                    "predictive_radius_delta",
                    entry["student_t_conf_radius_mean_delta"],
                )),
                "predictive_radius": float(entry.get("predictive_radius_scalar", entry["radius_scalar"])),
                "pruning_radius_semantics": semantics,
                "stochasticity": float(sigma),
            })
    def _write(tmp: Path):
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE2_COLUMNS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["Reg"], row["Act"])))

    return _atomic_write(output_path, _write)


def _margin_table(bounds_path, lipschitz_path, lq_source, gamma, legacy_margin):
    bounds = RegionActionBounds(bounds_path)
    constants = LipschitzConstants(lipschitz_path, source=lq_source)
    table = np.empty((4, 4), dtype=np.float32)
    for region in range(1, 5):
        for action in range(4):
            radius = float(bounds.get(region, action)["radius_scalar"])
            lr_value, lq_value = constants.get(region, action)
            table[region - 1, action] = action_margin(
                lr_value, lq_value, radius, gamma, legacy_margin
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

    right = states[:, 0] >= REGION_CENTER[0]
    top = states[:, 1] >= REGION_CENTER[1]
    regions = torch.where(top, torch.where(right, 0, 1), torch.where(right, 3, 2))
    margins = torch.as_tensor(
        _margin_table(bounds_path, lipschitz_path, lq_source, gamma, legacy_margin),
        dtype=torch.float32,
        device=device,
    )[regions]
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
    axis.axvline(REGION_CENTER[0], color="white", linestyle="--", linewidth=0.8)
    axis.axhline(REGION_CENTER[1], color="white", linestyle="--", linewidth=0.8)
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
