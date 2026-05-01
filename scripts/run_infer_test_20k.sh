#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/RoboticsDiffusionTransformer"
TEST_DIR="/workspace/test"
OUTPUT_DIR="/workspace/sample_result_rdt"
CKPT_DIR="/workspace/RoboticsDiffusionTransformer/checkpoints/rdt-170m-my-cool-dataset-smoke/checkpoint-20000"
VISION_ENCODER_PATH="${VISION_ENCODER_PATH:-/workspace/google/siglip-so400m-patch14-384}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-/workspace/google/t5-v1_1-xxl}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
CTRL_FREQ="${CTRL_FREQ:-25}"
PREDICT_STEPS="${PREDICT_STEPS:-50}"
ACTION_STEPS="${ACTION_STEPS:-$((PREDICT_STEPS + 1))}"
VIDEO_FPS="${VIDEO_FPS:-25}"
TRAJ_NAME="${TRAJ_NAME:-}"

cd "${REPO_DIR}"

if [[ ! -d "${VISION_ENCODER_PATH}" && -d "${REPO_DIR}/google/siglip-so400m-patch14-384" ]]; then
  VISION_ENCODER_PATH="${REPO_DIR}/google/siglip-so400m-patch14-384"
fi
if [[ ! -d "${TEXT_ENCODER_PATH}" && -d "${REPO_DIR}/google/t5-v1_1-xxl" ]]; then
  TEXT_ENCODER_PATH="${REPO_DIR}/google/t5-v1_1-xxl"
fi

echo "[INFO] repo: ${REPO_DIR}"
echo "[INFO] test_dir: ${TEST_DIR}"
echo "[INFO] output_dir: ${OUTPUT_DIR}"
echo "[INFO] ckpt_dir_or_file: ${CKPT_DIR}"
echo "[INFO] vision_encoder: ${VISION_ENCODER_PATH}"
echo "[INFO] text_encoder: ${TEXT_ENCODER_PATH}"
echo "[INFO] predict_steps(video): ${PREDICT_STEPS}"
echo "[INFO] action_steps(csv): ${ACTION_STEPS}"

ARGS=(
  -m scripts.infer_minireal_submission
  --test_dir "${TEST_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --pretrained_model_name_or_path "${CKPT_DIR}"
  --pretrained_vision_encoder_name_or_path "${VISION_ENCODER_PATH}"
  --pretrained_text_encoder_name_or_path "${TEXT_ENCODER_PATH}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --ctrl_freq "${CTRL_FREQ}"
  --predict_steps "${PREDICT_STEPS}"
  --action_steps "${ACTION_STEPS}"
  --video_fps "${VIDEO_FPS}"
)

if [[ -n "${TRAJ_NAME}" ]]; then
  ARGS+=(--traj_name "${TRAJ_NAME}")
  echo "[INFO] single trajectory smoke mode: ${TRAJ_NAME}"
fi

python "${ARGS[@]}"

python -m scripts.validate_submission_format \
  --input_test_dir "${TEST_DIR}" \
  --submission_dir "${OUTPUT_DIR}" \
  --predict_steps "${PREDICT_STEPS}" \
  --action_steps "${ACTION_STEPS}"

echo "[DONE] Inference + format validation completed."
