#!/usr/bin/env bash
set -euo pipefail

export TEXT_ENCODER_NAME="./google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="./google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-170m-my-cool-dataset"
export NUM_GPUS="${NUM_GPUS:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"

# Use all 8 GPUs by default unless user already specifies CUDA_VISIBLE_DEVICES.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi

accelerate launch --num_processes "$NUM_GPUS" main.py \
  --deepspeed="./configs/zero2.json" \
  --pretrained_model_name_or_path="./pretrained/rdt-170m" \
  --pretrained_text_encoder_name_or_path="$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path="$VISION_ENCODER_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=16 \
  --max_train_steps=20000 \
  --checkpointing_period=1000 \
  --sample_period=1000 \
  --checkpoints_total_limit=5 \
  --lr_scheduler="constant" \
  --learning_rate=5e-5 \
  --mixed_precision="bf16" \
  --dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  --dataset_type="finetune" \
  --state_noise_snr=40 \
  --load_from_hdf5 \
  --image_aug \
  --use_8bit_adam \
  --report_to=tensorboard
