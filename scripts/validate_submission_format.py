#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def read_csv_header_and_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows[0], rows[1:]


def normalize_instruction(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def validate_indices(path: Path, rows: List[List[str]], expected_steps: int) -> None:
    if len(rows) != expected_steps:
        raise ValueError(f"{path}: expected {expected_steps} data rows, got {len(rows)}")
    idx = []
    for r in rows:
        if not r:
            raise ValueError(f"{path}: found empty data row")
        idx.append(int(float(r[0])))
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            raise ValueError(f"{path}: index not continuous at row {i + 2}: {idx[i - 1]} -> {idx[i]}")


def pick_instruction_file(dir_path: Path) -> Path | None:
    for name in ("instruction.txt", "instructions.txt"):
        p = dir_path / name
        if p.is_file():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_test_dir", type=str, required=True)
    parser.add_argument("--submission_dir", type=str, required=True)
    parser.add_argument("--predict_steps", type=int, default=50)
    parser.add_argument(
        "--action_steps",
        type=int,
        default=None,
        help="Expected action/joint rows. Default is predict_steps + 1 (51 when predict_steps=50).",
    )
    args = parser.parse_args()
    action_steps = args.action_steps if args.action_steps is not None else (args.predict_steps + 1)

    test_dir = Path(args.input_test_dir)
    submission_dir = Path(args.submission_dir)

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Input test directory not found: {test_dir}")
    if not submission_dir.is_dir():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")

    traj_dirs = sorted([p for p in test_dir.iterdir() if p.is_dir()])
    if not traj_dirs:
        raise FileNotFoundError(f"No trajectory folders found under {test_dir}")

    for traj_dir in traj_dirs:
        traj_name = traj_dir.name
        out_dir = submission_dir / traj_name
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Missing output folder: {out_dir}")

        test_action = traj_dir / "action.txt"
        test_joint = traj_dir / "joint.txt"
        test_instr = pick_instruction_file(traj_dir)
        if test_instr is None:
            raise FileNotFoundError(f"Missing instruction file under input trajectory: {traj_dir}")

        out_action = out_dir / "action.txt"
        out_joint = out_dir / "joint.txt"
        out_instr = out_dir / "instruction.txt"
        out_video = out_dir / "video.mp4"

        for p in (out_action, out_joint, out_instr, out_video):
            if not p.exists():
                raise FileNotFoundError(f"{traj_name}: missing required output file: {p.name}")

        in_action_header, _ = read_csv_header_and_rows(test_action)
        in_joint_header, _ = read_csv_header_and_rows(test_joint)
        out_action_header, out_action_rows = read_csv_header_and_rows(out_action)
        out_joint_header, out_joint_rows = read_csv_header_and_rows(out_joint)

        if out_action_header != in_action_header:
            raise ValueError(f"{traj_name}: action header mismatch")
        if out_joint_header != in_joint_header:
            raise ValueError(f"{traj_name}: joint header mismatch")

        validate_indices(out_action, out_action_rows, action_steps)
        validate_indices(out_joint, out_joint_rows, action_steps)

        if normalize_instruction(out_instr) != normalize_instruction(test_instr):
            raise ValueError(f"{traj_name}: instruction text mismatch")

    print(
        f"[OK] Format validation passed for {len(traj_dirs)} trajectories: {submission_dir}. "
        f"Checked action/joint rows={action_steps}, expected video frames={args.predict_steps}."
    )


if __name__ == "__main__":
    main()
