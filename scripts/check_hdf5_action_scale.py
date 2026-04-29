#!/usr/bin/env python3
from __future__ import annotations
"""Check action/state numeric scale in HDF5 episodes for RDT finetuning.

RDT in this repo uses prediction_type=sample, so target is the clean action
itself and MSE scales quadratically with action magnitude. This script helps
spot unit/scale issues early (deg vs rad, mm vs m, encoder ticks, timestamps).
"""

import argparse
import fnmatch
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
try:
    import h5py
except ModuleNotFoundError:
    h5py = None


DEFAULT_THRESHOLDS = {
    "abs_very_large": 1e6,
    "abs_large": 1e4,
    "abs_medium": 1e3,
    "abs_expected_soft": 10.0,
}


def resolve_default_hdf5_dir(repo_root: str) -> str:
    """Mirror data/hdf5_vla_dataset.py path priority."""
    default_dir = os.path.normpath(os.path.join(repo_root, "..", "traindata"))
    hdf5_dir = os.environ.get("RDT_HDF5_DIR", default_dir)
    if not os.path.isdir(hdf5_dir):
        hdf5_dir = os.path.join(repo_root, "data", "datasets", "my_cool_dataset", "rdt_data")
    return hdf5_dir


def discover_hdf5_files(hdf5_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(hdf5_dir):
        for name in fnmatch.filter(names, "*.hdf5"):
            files.append(os.path.join(root, name))
    files.sort()
    return files


def _read_path(h5f: h5py.File, candidate_paths: Iterable[str]) -> Optional[np.ndarray]:
    for path in candidate_paths:
        if "/" in path:
            if path in h5f:
                return h5f[path][:]
        else:
            if path in h5f:
                return h5f[path][:]
    return None


@dataclass
class StreamStats:
    dim: int

    def __post_init__(self) -> None:
        self.count = np.zeros(self.dim, dtype=np.int64)
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)
        self.abs_max = np.zeros(self.dim, dtype=np.float64)
        self.nan_count = np.zeros(self.dim, dtype=np.int64)
        self.inf_count = np.zeros(self.dim, dtype=np.int64)
        self.gt_1e3 = np.zeros(self.dim, dtype=np.int64)
        self.gt_1e4 = np.zeros(self.dim, dtype=np.int64)
        self.gt_1e6 = np.zeros(self.dim, dtype=np.int64)

    def update(self, x: np.ndarray) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected shape [N, {self.dim}], got {x.shape}")
        x = x.astype(np.float64, copy=False)
        finite = np.isfinite(x)
        self.nan_count += np.isnan(x).sum(axis=0)
        self.inf_count += np.isinf(x).sum(axis=0)

        x_safe = np.where(finite, x, 0.0)
        finite_count = finite.sum(axis=0).astype(np.int64)
        self.count += finite_count
        self.sum += x_safe.sum(axis=0)
        self.sumsq += (x_safe * x_safe).sum(axis=0)

        if np.any(finite):
            pos_inf = np.full_like(x, np.inf)
            neg_inf = np.full_like(x, -np.inf)
            self.min = np.minimum(self.min, np.where(finite, x, pos_inf).min(axis=0))
            self.max = np.maximum(self.max, np.where(finite, x, neg_inf).max(axis=0))
            self.abs_max = np.maximum(self.abs_max, np.where(finite, np.abs(x), 0.0).max(axis=0))

        abs_x = np.abs(np.where(finite, x, 0.0))
        self.gt_1e3 += (abs_x > 1e3).sum(axis=0)
        self.gt_1e4 += (abs_x > 1e4).sum(axis=0)
        self.gt_1e6 += (abs_x > 1e6).sum(axis=0)

    def mean(self) -> np.ndarray:
        denom = np.maximum(self.count, 1)
        return self.sum / denom

    def std(self) -> np.ndarray:
        denom = np.maximum(self.count, 1)
        mean = self.sum / denom
        var = np.maximum(self.sumsq / denom - mean * mean, 0.0)
        return np.sqrt(var)


def sample_rows(arr: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if arr.shape[0] <= max_rows:
        return arr
    idx = rng.choice(arr.shape[0], size=max_rows, replace=False)
    return arr[idx]


def pct_summary(samples: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    if samples is None or samples.shape[0] == 0:
        return {}
    valid = np.where(np.isfinite(samples), samples, np.nan)
    return {
        "p01": np.nanpercentile(valid, 1, axis=0),
        "p50": np.nanpercentile(valid, 50, axis=0),
        "p99": np.nanpercentile(valid, 99, axis=0),
    }


def infer_scale_alerts(dim_stats: StreamStats, pct: Dict[str, np.ndarray]) -> List[str]:
    alerts: List[str] = []
    mean = dim_stats.mean()
    std = dim_stats.std()
    p50 = pct.get("p50")
    abs_max = dim_stats.abs_max

    if np.any(dim_stats.nan_count > 0):
        alerts.append("NaN values detected in some dimensions.")
    if np.any(dim_stats.inf_count > 0):
        alerts.append("Inf values detected in some dimensions.")
    if np.any(abs_max > DEFAULT_THRESHOLDS["abs_very_large"]):
        alerts.append("Abs max > 1e6 detected: extremely likely wrong units/columns.")
    elif np.any(abs_max > DEFAULT_THRESHOLDS["abs_large"]):
        alerts.append("Abs max > 1e4 detected: likely ticks/timestamps or unscaled signals.")
    elif np.any(abs_max > DEFAULT_THRESHOLDS["abs_medium"]):
        alerts.append("Abs max > 1e3 detected: suspicious for RDT sample-prediction training.")

    if p50 is not None:
        # Heuristic: many dims around ~180 suggest degrees rather than radians.
        deg_like = np.where((np.abs(p50) > 20.0) & (np.abs(p50) < 400.0) & (abs_max <= 400.0))[0]
        if deg_like.size > 0:
            alerts.append(
                "Some dimensions look degree-like (median in [20,400] and bounded around <400). "
                "Joint angles should usually be radians."
            )
        # Heuristic: medians in tens to thousands may indicate mm-like scales.
        mm_like = np.where((np.abs(p50) > 2.0) & (np.abs(p50) < 5000.0) & (std > 0.1))[0]
        if mm_like.size > dim_stats.dim // 3:
            alerts.append(
                "Many dimensions have meter-unfriendly magnitudes (median > 2). "
                "Check mm->m conversion and whether non-action columns were mixed in."
            )

    if np.any(np.abs(mean) > DEFAULT_THRESHOLDS["abs_expected_soft"]):
        alerts.append("Mean magnitude > 10 in some dimensions; verify expected action/state range.")
    return alerts


def format_dim_row(
    i: int,
    stats: StreamStats,
    pct: Dict[str, np.ndarray],
) -> str:
    mean = stats.mean()[i]
    std = stats.std()[i]
    p01 = pct.get("p01", np.full(stats.dim, np.nan))[i]
    p50 = pct.get("p50", np.full(stats.dim, np.nan))[i]
    p99 = pct.get("p99", np.full(stats.dim, np.nan))[i]
    return (
        f"{i:>3d} | n={stats.count[i]:>9d} | min={stats.min[i]:>12.4g} | p01={p01:>10.4g} "
        f"| p50={p50:>10.4g} | p99={p99:>10.4g} | max={stats.max[i]:>12.4g} "
        f"| mean={mean:>10.4g} | std={std:>10.4g} | abs_max={stats.abs_max[i]:>10.4g} "
        f"| NaN={stats.nan_count[i]:>6d} | Inf={stats.inf_count[i]:>6d}"
    )


def scan_dataset(
    hdf5_files: List[str],
    max_files: int,
    sample_rows_per_file: int,
    seed: int,
) -> Tuple[StreamStats, StreamStats, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scanned = 0
    qpos_stats: Optional[StreamStats] = None
    action_stats: Optional[StreamStats] = None
    qpos_samples: List[np.ndarray] = []
    action_samples: List[np.ndarray] = []

    for file_path in hdf5_files[:max_files]:
        with h5py.File(file_path, "r") as f:
            qpos = _read_path(f, ["observations/qpos"])
            if qpos is None and "observations" in f and "qpos" in f["observations"]:
                qpos = f["observations"]["qpos"][:]
            action = _read_path(f, ["action", "actions"])

        if qpos is None or action is None:
            print(f"[WARN] Skip {file_path}: missing qpos/action dataset.")
            continue
        if qpos.ndim != 2 or action.ndim != 2:
            print(f"[WARN] Skip {file_path}: qpos/action not rank-2, got {qpos.shape}, {action.shape}")
            continue
        if qpos.shape[1] != action.shape[1]:
            print(f"[WARN] {file_path}: qpos dim {qpos.shape[1]} != action dim {action.shape[1]}")

        if qpos_stats is None:
            qpos_stats = StreamStats(qpos.shape[1])
        if action_stats is None:
            action_stats = StreamStats(action.shape[1])

        if qpos.shape[1] != qpos_stats.dim or action.shape[1] != action_stats.dim:
            print(
                f"[WARN] Skip {file_path}: inconsistent dim. "
                f"expected qpos/action {qpos_stats.dim}/{action_stats.dim}, got {qpos.shape[1]}/{action.shape[1]}"
            )
            continue

        qpos = qpos.astype(np.float32, copy=False)
        action = action.astype(np.float32, copy=False)
        qpos_stats.update(qpos)
        action_stats.update(action)
        qpos_samples.append(sample_rows(qpos, sample_rows_per_file, rng))
        action_samples.append(sample_rows(action, sample_rows_per_file, rng))
        scanned += 1

    if scanned == 0 or qpos_stats is None or action_stats is None:
        raise RuntimeError("No valid HDF5 episodes scanned. Check --hdf5-dir and dataset keys.")

    return (
        qpos_stats,
        action_stats,
        np.concatenate(qpos_samples, axis=0) if qpos_samples else np.zeros((0, qpos_stats.dim), dtype=np.float32),
        np.concatenate(action_samples, axis=0) if action_samples else np.zeros((0, action_stats.dim), dtype=np.float32),
    )


def print_summary(name: str, stats: StreamStats, sampled: np.ndarray, show_dims: Optional[List[int]]) -> None:
    pct = pct_summary(sampled)
    print(f"\n=== {name} summary ===")
    print(f"dim={stats.dim}, finite_count_total={int(stats.count.sum())}")
    print("alerts:")
    alerts = infer_scale_alerts(stats, pct)
    if alerts:
        for msg in alerts:
            print(f"  - {msg}")
    else:
        print("  - no obvious global scale anomalies detected")

    header = (
        "dim | count      | min          | p01        | p50        | p99        | max          "
        "| mean       | std        | abs_max    | NaN    | Inf"
    )
    print("\n" + header)
    print("-" * len(header))

    if show_dims is None:
        dims = range(stats.dim)
    else:
        dims = [d for d in show_dims if 0 <= d < stats.dim]
    for i in dims:
        print(format_dim_row(i, stats, pct))

    bad_dims = np.where(
        (stats.nan_count > 0)
        | (stats.inf_count > 0)
        | (stats.abs_max > DEFAULT_THRESHOLDS["abs_large"])
    )[0]
    if bad_dims.size > 0:
        print(f"\n{name} suspicious dims (NaN/Inf/abs_max>1e4): {bad_dims.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 action/state value scales for RDT finetuning."
    )
    parser.add_argument(
        "--hdf5-dir",
        type=str,
        default=None,
        help=(
            "HDF5 dataset directory. Default follows loader logic: "
            "RDT_HDF5_DIR -> ../traindata -> data/datasets/my_cool_dataset/rdt_data."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Max number of hdf5 files to scan.",
    )
    parser.add_argument(
        "--sample-rows-per-file",
        type=int,
        default=5000,
        help="Rows sampled per file for percentile estimation.",
    )
    parser.add_argument(
        "--show-dims",
        type=str,
        default="all",
        help='Comma-separated dimensions to print (e.g. "0,1,2,14,19"), or "all".',
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if h5py is None:
        raise ModuleNotFoundError(
            "h5py is required to scan HDF5 files. Install it with `pip install h5py`."
        )
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    hdf5_dir = args.hdf5_dir if args.hdf5_dir else resolve_default_hdf5_dir(repo_root)
    hdf5_files = discover_hdf5_files(hdf5_dir)
    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found under {hdf5_dir}")

    show_dims: Optional[List[int]]
    if args.show_dims.strip().lower() == "all":
        show_dims = None
    else:
        show_dims = [int(x.strip()) for x in args.show_dims.split(",") if x.strip()]

    print(f"HDF5 dir: {hdf5_dir}")
    print(f"Found files: {len(hdf5_files)}")
    print(f"Scanning up to {min(len(hdf5_files), args.max_files)} files ...")
    qpos_stats, action_stats, qpos_sampled, action_sampled = scan_dataset(
        hdf5_files=hdf5_files,
        max_files=args.max_files,
        sample_rows_per_file=args.sample_rows_per_file,
        seed=args.seed,
    )

    print_summary("qpos", qpos_stats, qpos_sampled, show_dims)
    print_summary("action", action_stats, action_sampled, show_dims)

    print("\nInterpretation guide:")
    print("- For prediction_type=sample, loss is MSE(pred, clean_action).")
    print("- If abs values are huge, loss explodes quadratically.")
    print("- Joint angles should typically be radians (often within about [-pi, pi]).")
    print("- Positions should usually be meters (not mm).")
    print("- Gripper dimensions are expected to be bounded, often normalized to [0,1].")


if __name__ == "__main__":
    main()
