#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root no matter where this script is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Docker layout:
# /workspace/
#   ├── RoboticsDiffusionTransformer
#   ├── google
#   └── pretrained
export TEXT_ENCODER_NAME="/workspace/google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="/workspace/google/siglip-so400m-patch14-384"
export PRETRAINED_RDT="/workspace/pretrained/rdt-170m"
export OUTPUT_DIR="$SCRIPT_DIR/checkpoints/rdt-170m-my-cool-dataset-smoke"

accelerate launch --num_processes 1 main.py \
  --deepspeed="./configs/zero2.json" \
  --pretrained_model_name_or_path="$PRETRAINED_RDT" \
  --pretrained_text_encoder_name_or_path="$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path="$VISION_ENCODER_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=16 \
  --max_train_steps=500 \
  --checkpointing_period=250 \
  --sample_period=250 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" \
  --learning_rate=5e-5 \
  --mixed_precision="bf16" \
  --dataloader_num_workers=0 \
  --dataset_type="finetune" \
  --state_noise_snr=40 \
  --load_from_hdf5 \
  --image_aug \
  --use_8bit_adam \
  --report_to=tensorboard
