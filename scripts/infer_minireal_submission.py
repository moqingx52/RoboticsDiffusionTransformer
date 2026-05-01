#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.t5_encoder import T5Embedder
from scripts.agilex_model import create_model


LEFT_ARM = [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)]
RIGHT_ARM = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]
LEFT_HAND_5 = [STATE_VEC_IDX_MAPPING[f"left_gripper_joint_{i}_pos"] for i in range(5)]
RIGHT_HAND_5 = [STATE_VEC_IDX_MAPPING[f"right_gripper_joint_{i}_pos"] for i in range(5)]
LEFT_HAND_AUX = STATE_VEC_IDX_MAPPING["left_arm_joint_7_pos"]
RIGHT_HAND_AUX = STATE_VEC_IDX_MAPPING["right_arm_joint_7_pos"]
USED_STATE_INDICES = (
    LEFT_ARM
    + RIGHT_ARM
    + LEFT_HAND_5
    + [LEFT_HAND_AUX]
    + RIGHT_HAND_5
    + [RIGHT_HAND_AUX]
)


def raw26_to_rdt128(x: np.ndarray, state_dim: int) -> np.ndarray:
    y = np.zeros(x.shape[:-1] + (state_dim,), dtype=np.float32)
    y[..., LEFT_ARM] = x[..., 0:7]
    y[..., RIGHT_ARM] = x[..., 7:14]
    y[..., LEFT_HAND_5] = x[..., 14:19]
    y[..., LEFT_HAND_AUX] = x[..., 19]
    y[..., RIGHT_HAND_5] = x[..., 20:25]
    y[..., RIGHT_HAND_AUX] = x[..., 25]
    return y


def rdt128_to_raw26(y: np.ndarray) -> np.ndarray:
    x = np.zeros(y.shape[:-1] + (26,), dtype=np.float32)
    x[..., 0:7] = y[..., LEFT_ARM]
    x[..., 7:14] = y[..., RIGHT_ARM]
    x[..., 14:19] = y[..., LEFT_HAND_5]
    x[..., 19] = y[..., LEFT_HAND_AUX]
    x[..., 20:25] = y[..., RIGHT_HAND_5]
    x[..., 25] = y[..., RIGHT_HAND_AUX]
    return x


def read_csv_matrix(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    header = rows[0]
    data_rows = [r for r in rows[1:] if r]
    if not data_rows:
        raise ValueError(f"CSV has no data rows: {path}")
    data = np.asarray(data_rows, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 27:
        raise ValueError(f"Expected >=27 columns in {path}, got shape={data.shape}")
    return header, data


def write_csv_matrix(path: Path, header: Sequence[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in data:
            row_out = [int(round(row[0]))]
            row_out.extend([f"{float(v):.10f}" for v in row[1:]])
            writer.writerow(row_out)


def load_video_frames(video_path: Path) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frame decoded from video: {video_path}")
    return frames


def write_video(path: Path, frames_rgb: Sequence[np.ndarray], fps: int) -> None:
    import cv2

    if not frames_rgb:
        raise ValueError("No frames for video writing.")
    h, w = frames_rgb[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create writer: {path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def preprocess_images_for_policy(model, images: Sequence[Image.Image | None], device: torch.device, dtype: torch.dtype):
    image_processor = model.image_processor
    background_color = np.array([int(x * 255) for x in image_processor.image_mean], dtype=np.uint8).reshape(1, 1, 3)
    background_image = np.ones((image_processor.size["height"], image_processor.size["width"], 3), dtype=np.uint8) * background_color

    image_tensor_list = []
    for image in images:
        if image is None:
            image = Image.fromarray(background_image)

        if model.args["dataset"].get("auto_adjust_image_brightness", False):
            pixel_values = list(image.getdata())
            average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
            if average_brightness <= 0.15:
                image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

        if model.args["dataset"].get("image_aspect_ratio", "pad") == "pad":
            width, height = image.size
            if width != height:
                side = max(width, height)
                canvas = Image.new(image.mode, (side, side), tuple(int(x * 255) for x in image_processor.image_mean))
                canvas.paste(image, ((side - width) // 2, (side - height) // 2))
                image = canvas

        tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor_list.append(tensor)

    image_tensor = torch.stack(image_tensor_list, dim=0).to(device=device, dtype=dtype)
    image_embeds = model.vision_model(image_tensor).detach()
    return image_embeds.reshape(-1, model.vision_model.hidden_size).unsqueeze(0)


def encode_instruction(
    instruction: str,
    text_tokenizer,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    key = instruction.strip()
    if key in cache:
        return cache[key]

    tokens = text_tokenizer(key, return_tensors="pt", padding="longest", truncation=True)["input_ids"].to(device)
    with torch.no_grad():
        embeds = text_encoder(tokens).last_hidden_state.detach().to(dtype=dtype)
    cache[key] = embeds
    return embeds


@torch.no_grad()
def predict_actions_50(
    model,
    text_embeds: torch.Tensor,
    state26: np.ndarray,
    frame_tm1: np.ndarray,
    frame_t: np.ndarray,
    ctrl_freq: int,
    state_dim: int,
    predict_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    images = [
        Image.fromarray(frame_tm1),
        None,
        None,
        Image.fromarray(frame_t),
        None,
        None,
    ]
    image_tokens = preprocess_images_for_policy(model, images, device=device, dtype=dtype)

    state128 = raw26_to_rdt128(state26.reshape(1, 1, 26), state_dim=state_dim)
    state_tokens = torch.from_numpy(state128).to(device=device, dtype=dtype)
    state_mask = torch.zeros((1, state_dim), device=device, dtype=dtype)
    state_mask[:, USED_STATE_INDICES] = 1.0
    ctrl_freqs = torch.tensor([ctrl_freq], device=device)

    trajectory = model.policy.predict_action(
        lang_tokens=text_embeds,
        lang_attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=device),
        img_tokens=image_tokens,
        state_tokens=state_tokens[:, -1:, :],
        action_mask=state_mask.unsqueeze(1),
        ctrl_freqs=ctrl_freqs,
    ).to(torch.float32)

    pred64_state128 = trajectory.cpu().numpy()[0]
    pred64_raw26 = rdt128_to_raw26(pred64_state128)
    return pred64_raw26[:predict_steps]


def pick_video_path(traj_dir: Path) -> Path | None:
    candidates = [
        traj_dir / "video.mp4",
        traj_dir / "rgb.mp4",
        traj_dir / "cam_high.mp4",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def pick_instruction_path(traj_dir: Path) -> Path | None:
    candidates = [
        traj_dir / "instruction.txt",
        traj_dir / "instructions.txt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def resolve_checkpoint_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_file():
        return str(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Checkpoint path must be file or directory: {path}")

    candidates = [
        path / "pytorch_model" / "mp_rank_00_model_states.pt",
        path / "mp_rank_00_model_states.pt",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    pt_files = sorted(path.glob("**/*.pt"))
    if len(pt_files) == 1:
        return str(pt_files[0])
    if len(pt_files) > 1:
        preview = ", ".join(str(p) for p in pt_files[:5])
        raise ValueError(
            f"Found multiple .pt files under {path}. Please pass an explicit file path. "
            f"Examples: {preview}"
        )
    raise FileNotFoundError(f"No .pt file found under checkpoint directory: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="/workspace/test")
    parser.add_argument("--output_dir", type=str, default="/workspace/sample_result_rdt")
    parser.add_argument("--config_path", type=str, default="configs/base.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, default="/workspace/google/siglip-so400m-patch14-384")
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, default="/workspace/google/t5-v1_1-xxl")
    parser.add_argument("--ctrl_freq", type=int, default=25)
    parser.add_argument("--history_steps", type=int, default=16)
    parser.add_argument("--predict_steps", type=int, default=50)
    parser.add_argument("--video_fps", type=int, default=25)
    parser.add_argument("--fallback_width", type=int, default=640)
    parser.add_argument("--fallback_height", type=int, default=480)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--allow_missing_video", action="store_true")
    parser.add_argument(
        "--traj_name",
        type=str,
        default=None,
        help="Optional trajectory folder name for single-sample smoke test, e.g. 1_1",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    infer_dtype = dtype_map[args.dtype]
    if device.type == "cpu" and infer_dtype != torch.float32:
        infer_dtype = torch.float32

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    state_dim = int(config["common"]["state_dim"])

    checkpoint_path = resolve_checkpoint_path(args.pretrained_model_name_or_path)
    print(f"Using checkpoint: {checkpoint_path}")

    model = create_model(
        args=config,
        dtype=infer_dtype,
        pretrained=checkpoint_path,
        pretrained_vision_encoder_name_or_path=args.pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )
    model.device = str(device)
    model.dtype = infer_dtype
    model.reset()

    text_embedder = T5Embedder(
        from_pretrained=args.pretrained_text_encoder_name_or_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
    )
    text_tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    text_encoder.eval()
    text_encoder = text_encoder.to(device=device, dtype=infer_dtype)

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_dirs = sorted([p for p in test_dir.iterdir() if p.is_dir()])
    if args.traj_name is not None:
        traj_dirs = [p for p in traj_dirs if p.name == args.traj_name]
        if not traj_dirs:
            raise FileNotFoundError(f"No trajectory named '{args.traj_name}' under {test_dir}")
    if not traj_dirs:
        raise FileNotFoundError(f"No trajectory directories found under {test_dir}")
    text_cache: Dict[str, torch.Tensor] = {}

    for idx, traj_dir in enumerate(traj_dirs, start=1):
        print(f"[{idx}/{len(traj_dirs)}] Processing {traj_dir.name}")
        instruction_path = pick_instruction_path(traj_dir)
        joint_path = traj_dir / "joint.txt"
        action_path = traj_dir / "action.txt"
        if instruction_path is None:
            raise FileNotFoundError(f"Missing instruction.txt or instructions.txt under {traj_dir}")
        if not joint_path.is_file():
            raise FileNotFoundError(f"Missing joint.txt under {traj_dir}")
        if not action_path.is_file():
            raise FileNotFoundError(f"Missing action.txt under {traj_dir}")

        instruction = instruction_path.read_text(encoding="utf-8").strip()
        joint_header, joint_data = read_csv_matrix(joint_path)
        action_header, action_data = read_csv_matrix(action_path)
        if joint_data.shape[0] < args.history_steps:
            raise ValueError(f"{traj_dir}: expected >= {args.history_steps} rows, got {joint_data.shape[0]}")

        video_path = pick_video_path(traj_dir)
        if video_path is not None:
            frames = load_video_frames(video_path)
        elif args.allow_missing_video:
            blank = np.zeros((args.fallback_height, args.fallback_width, 3), dtype=np.uint8)
            frames = [blank.copy() for _ in range(args.history_steps)]
            print(f"  [WARN] video missing in {traj_dir}, using blank context frames.")
        else:
            raise FileNotFoundError(f"No video file found in {traj_dir}")

        if len(frames) < args.history_steps:
            raise ValueError(f"{traj_dir}: video has {len(frames)} frames, needs >= {args.history_steps}")
        frame_tm1 = frames[args.history_steps - 2]
        frame_t = frames[args.history_steps - 1]

        text_embeds = encode_instruction(
            instruction=instruction,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            device=device,
            dtype=infer_dtype,
            cache=text_cache,
        )
        state26 = joint_data[args.history_steps - 1, 1:27].astype(np.float32, copy=False)
        pred_action26 = predict_actions_50(
            model=model,
            text_embeds=text_embeds,
            state26=state26,
            frame_tm1=frame_tm1,
            frame_t=frame_t,
            ctrl_freq=args.ctrl_freq,
            state_dim=state_dim,
            predict_steps=args.predict_steps,
            device=device,
            dtype=infer_dtype,
        )

        start_idx = int(round(max(joint_data[:, 0].max(), action_data[:, 0].max()))) + 1
        time_idx = np.arange(start_idx, start_idx + args.predict_steps, dtype=np.float32).reshape(-1, 1)
        action_out = np.concatenate([time_idx, pred_action26], axis=1)

        # For this benchmark format, we output joint predictions aligned to the same 26D trajectory.
        joint_out = action_out.copy()

        out_traj = output_dir / traj_dir.name
        out_traj.mkdir(parents=True, exist_ok=True)
        write_csv_matrix(out_traj / "action.txt", action_header, action_out)
        write_csv_matrix(out_traj / "joint.txt", joint_header, joint_out)
        (out_traj / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
        pred_video = [frame_t.copy() for _ in range(args.predict_steps)]
        write_video(out_traj / "video.mp4", pred_video, fps=args.video_fps)

    print(f"Done. Wrote predictions to: {output_dir}")


if __name__ == "__main__":
    main()
