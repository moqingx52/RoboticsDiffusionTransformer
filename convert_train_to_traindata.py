#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


def _read_csv_matrix(path: Path, expected_dim: int) -> np.ndarray:
    data = np.loadtxt(str(path), delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 1 + expected_dim:
        raise ValueError(f"Bad shape for {path}: {getattr(data, 'shape', None)}")
    return data[:, 1 : 1 + expected_dim].astype(np.float32, copy=False)


def _safe_traj_id(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name.strip())


def _episode_output_path(out_dir: Path, traj_id: str, use_task_layout: bool, task_name: str) -> Path:
    if use_task_layout:
        return out_dir / "rdt_data" / task_name / f"episode_{traj_id}.hdf5"
    return out_dir / f"episode_{traj_id}.hdf5"


def _maybe_write_instruction_json(out_hdf5_path: Path, instruction: str, use_task_layout: bool):
    if not use_task_layout:
        return
    task_dir = out_hdf5_path.parent
    json_path = task_dir / "expanded_instruction_gpt-4-turbo.json"
    if json_path.exists():
        return
    data = [{
        "instruction": instruction
    }]
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_video_rgb_frames(video_path: Path, frame_width: int, frame_height: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frame decoded from video: {video_path}")
    return np.asarray(frames, dtype=np.uint8)


def convert_one(
    traj_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    frame_width: int = 640,
    frame_height: int = 480,
    require_video: bool = False,
    use_task_layout: bool = False,
    task_name: str = "task_001",
) -> Path:
    instr_path = traj_dir / "instruction.txt"
    joint_path = traj_dir / "joint.txt"
    action_path = traj_dir / "action.txt"
    video_path = traj_dir / "video.mp4"
    if not (instr_path.is_file() and joint_path.is_file() and action_path.is_file()):
        raise FileNotFoundError(f"Missing required files in {traj_dir}")

    instruction = instr_path.read_text(encoding="utf-8").strip()
    qpos = _read_csv_matrix(joint_path, expected_dim=26)
    action = _read_csv_matrix(action_path, expected_dim=26)

    frames = None
    if video_path.is_file():
        frames = _read_video_rgb_frames(video_path, frame_width=frame_width, frame_height=frame_height)
    elif require_video:
        raise FileNotFoundError(f"Missing required video.mp4 in {traj_dir}")

    if frames is None:
        t = int(min(qpos.shape[0], action.shape[0]))
        frames = np.zeros((t, frame_height, frame_width, 3), dtype=np.uint8)
    else:
        t = int(min(len(frames), qpos.shape[0], action.shape[0]))
    if t <= 0:
        raise ValueError(f"Empty trajectory: {traj_dir}")

    frames = frames[:t]
    qpos = qpos[:t]
    action = action[:t]
    zero_cam = np.zeros_like(frames, dtype=np.uint8)

    traj_id = _safe_traj_id(traj_dir.name)
    out_path = _episode_output_path(out_dir=out_dir, traj_id=traj_id, use_task_layout=use_task_layout, task_name=task_name)
    if out_path.exists() and not overwrite:
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos, compression="gzip")
        imgs = obs.create_group("images")
        imgs.create_dataset("cam_high", data=frames, compression="gzip")
        imgs.create_dataset("cam_left_wrist", data=zero_cam, compression="gzip")
        imgs.create_dataset("cam_right_wrist", data=zero_cam, compression="gzip")
        f.create_dataset("action", data=action, compression="gzip")
        f.create_dataset("instruction", data=np.array(instruction, dtype=h5py.string_dtype("utf-8")))
        f.create_dataset("traj_id", data=np.array(traj_id, dtype=h5py.string_dtype("utf-8")))
    _maybe_write_instruction_json(out_path, instruction, use_task_layout)
    return out_path


def iter_traj_dirs(train_dir: Path):
    for p in sorted(train_dir.iterdir()):
        if p.is_dir():
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default="train")
    ap.add_argument("--out_dir", type=str, default="traindata")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--frame_width", type=int, default=640)
    ap.add_argument("--frame_height", type=int, default=480)
    ap.add_argument("--require_video", action="store_true")
    ap.add_argument("--use_task_layout", action="store_true", help="Write to out_dir/rdt_data/<task_name>/episode_*.hdf5 and create expanded_instruction json.")
    ap.add_argument("--task_name", type=str, default="task_001")
    args = ap.parse_args()

    root = Path(os.getcwd())
    train_dir = (root / args.train_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    converted, skipped, failed = 0, 0, 0
    traj_dirs = list(iter_traj_dirs(train_dir))
    total = len(traj_dirs)
    print(f"Found {total} trajectories under {train_dir}")
    for idx, traj_dir in enumerate(traj_dirs, start=1):
        progress = (idx / total * 100.0) if total > 0 else 100.0
        print(f"[{idx}/{total}] ({progress:6.2f}%) Processing {traj_dir.name}")
        try:
            out_path = _episode_output_path(
                out_dir=out_dir,
                traj_id=_safe_traj_id(traj_dir.name),
                use_task_layout=args.use_task_layout,
                task_name=args.task_name,
            )
            if out_path.exists() and not args.overwrite:
                skipped += 1
                print(f"[SKIP] {traj_dir.name}")
                continue
            convert_one(
                traj_dir=traj_dir,
                out_dir=out_dir,
                overwrite=args.overwrite,
                frame_width=args.frame_width,
                frame_height=args.frame_height,
                require_video=args.require_video,
                use_task_layout=args.use_task_layout,
                task_name=args.task_name,
            )
            converted += 1
            print(f"[OK]   {traj_dir.name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {traj_dir}: {e}")

    print(f"Done. converted={converted} skipped={skipped} failed={failed} out_dir={out_dir}")


if __name__ == "__main__":
    main()

