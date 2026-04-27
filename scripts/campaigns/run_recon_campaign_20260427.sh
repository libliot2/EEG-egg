#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaoh/DeepLearning/EEG

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/data/xiaoh/DeepLearning_storage/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/xiaoh/DeepLearning_storage/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/data/xiaoh/DeepLearning_storage/.cache/huggingface/hub
export TORCH_HOME=/data/xiaoh/DeepLearning_storage/.cache/torch
export HF_HUB_OFFLINE=1

PY=/home/xiaoh/DeepLearning/.conda-envs/project1-eeg/bin/python
ROOT=${RECON_CAMPAIGN_ROOT:-outputs/reconstruction_campaigns/20260427_longrun}
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

DATA=/home/xiaoh/DeepLearning/image-eeg-data
SPLIT=outputs/subsets/val64_seed0.json
INIT_BEST=outputs/reconstruction_lowlevel_init_blend/val64_low85_posterior15/images
RECON_V2=outputs/experiments/reconstruction_clip_embed_v2_loss_imgsoft_local/seed_0/best.pt
ADAPTER_V1=outputs/experiments/clip_to_kandinsky_adapter_v1/seed_0/best.pt
LOSSIMG=outputs_local/experiments/retrieval_adapter_atm_large_loss_imgsoft_dir_e40/seed_0/best.pt
POSTCP=outputs/experiments/retrieval_channel_posteriorcp28_atm_base_ides_v1_local/seed_0/best.pt

COMMON_PRED_ARGS=(
  --conditioning-bank outputs/cache/clip_train.pt
  --embedding-bank outputs/cache/kandinsky_train.pt
  --data-dir "$DATA"
  --split train
  --split-file "$SPLIT"
  --image-id-source val_ids
  --embedding-source predicted
  --decoder-steps 20
  --decoder-height 512
  --decoder-width 512
  --batch-size 4
  --num-workers 4
  --local-files-only
  --evaluate
  --device cuda
)

log_run() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[$(date -Is)] START gpu=${gpu} ${name}" | tee -a "$ROOT/campaign.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "HOST=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    printf 'CMD:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  ) >"$LOGS/${name}.log" 2>&1
  echo "[$(date -Is)] END gpu=${gpu} ${name}" | tee -a "$ROOT/campaign.log"
}

summarize_now() {
  "$PY" scripts/summarize_recon_campaign.py "$ROOT" --output "$ROOT/summary.md" --top-k 40 >/dev/null || true
}

predict_clip_adapter() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  local adapter="$4"
  local init_dir="$5"
  local strength="$6"
  local candidates="$7"
  local guidance="$8"
  shift 8
  log_run "$gpu" "$name" "$PY" scripts/predict_reconstruction_embed.py \
    "${COMMON_PRED_ARGS[@]}" \
    --reconstruction-checkpoint "$checkpoint" \
    --conditioning-adapter "$adapter" \
    --init-image-dir "$init_dir" \
    --img2img-strength "$strength" \
    --num-candidates "$candidates" \
    --decoder-guidance-scale "$guidance" \
    --output-dir "$ROOT/$name" \
    "$@"
  summarize_now
}

train_adapter_variant() {
  local gpu="$1"
  local name="$2"
  local hidden="$3"
  local dropout="$4"
  local mse="$5"
  local cosine="$6"
  local contrastive="$7"
  log_run "$gpu" "train_${name}" "$PY" scripts/train_embedding_adapter.py \
    --source-bank outputs/cache/clip_train.pt \
    --target-bank outputs/cache/kandinsky_train.pt \
    --output-dir "$ROOT/adapter_${name}" \
    --epochs 80 \
    --batch-size 256 \
    --hidden-dim "$hidden" \
    --dropout "$dropout" \
    --mse-weight "$mse" \
    --cosine-weight "$cosine" \
    --contrastive-weight "$contrastive" \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --device cuda
  summarize_now
}

train_clip_predictor_variant() {
  local gpu="$1"
  local name="$2"
  local retrieval_ckpt="$3"
  local hidden="$4"
  local dropout="$5"
  log_run "$gpu" "train_${name}" "$PY" scripts/train_reconstruction_embed.py \
    --retrieval-checkpoint "$retrieval_ckpt" \
    --embedding-bank outputs/cache/clip_train.pt \
    --target-space clip \
    --output-dir "$ROOT/${name}" \
    --epochs 80 \
    --batch-size 96 \
    --num-workers 4 \
    --head-hidden-dim "$hidden" \
    --dropout "$dropout" \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --mse-weight 1.0 \
    --cosine-weight 0.75 \
    --contrastive-weight 0.2 \
    --selection-metric val_subset_top1_then_top5 \
    --embedding-eval-every 1 \
    --image-eval-limit 0 \
    --device cuda
  summarize_now
}

train_lowlevel_variant() {
  local gpu="$1"
  local name="$2"
  local retrieval_ckpt="$3"
  local l1="$4"
  local mse="$5"
  local ssim="$6"
  local lpips_w="$7"
  local clip_w="$8"
  log_run "$gpu" "train_${name}" "$PY" scripts/train_lowlevel_image_prior.py \
    --retrieval-checkpoint "$retrieval_ckpt" \
    --data-dir "$DATA" \
    --output-dir "$ROOT/${name}" \
    --epochs 80 \
    --batch-size 32 \
    --num-workers 4 \
    --learning-rate 2e-4 \
    --weight-decay 1e-4 \
    --base-channels 256 \
    --image-size 256 \
    --l1-weight "$l1" \
    --mse-weight "$mse" \
    --ssim-weight "$ssim" \
    --lpips-weight "$lpips_w" \
    --clip-weight "$clip_w" \
    --eval-split-file "$SPLIT" \
    --eval-image-id-source val_ids \
    --eval-output-name eval_val64 \
    --device cuda
  summarize_now
}

echo "[$(date -Is)] campaign root: $ROOT" | tee "$ROOT/campaign.log"

(
  for guidance in 3.5 4.0 4.5; do
    for strength in 0.24 0.26 0.28 0.30 0.32; do
      tag="a1_c4_g${guidance/./p}_s${strength/./p}"
      predict_clip_adapter 0 "$tag" "$RECON_V2" "$ADAPTER_V1" "$INIT_BEST" "$strength" 4 "$guidance"
    done
  done
) &
pid_a1_c4=$!

(
  for guidance in 3.5 4.0 4.5; do
    for strength in 0.24 0.26 0.28 0.30 0.32; do
      tag="a1_c8_g${guidance/./p}_s${strength/./p}"
      predict_clip_adapter 1 "$tag" "$RECON_V2" "$ADAPTER_V1" "$INIT_BEST" "$strength" 8 "$guidance"
    done
  done
) &
pid_a1_c8=$!

(
  for guidance in 3.5 4.0 4.5; do
    for strength in 0.24 0.26 0.28 0.30 0.32; do
      tag="a1_c16_g${guidance/./p}_s${strength/./p}"
      predict_clip_adapter 2 "$tag" "$RECON_V2" "$ADAPTER_V1" "$INIT_BEST" "$strength" 16 "$guidance"
    done
  done
) &
pid_a1_c16=$!

(
  for metric in pixel_cosine neg_mse; do
    for weight in 0.10 0.20 0.35 0.50; do
      tag="a2_c8_semantic_lowlevel_${metric}_w${weight/./p}"
      predict_clip_adapter 3 "$tag" "$RECON_V2" "$ADAPTER_V1" "$INIT_BEST" 0.30 8 4.0 \
        --candidate-selection-mode semantic_lowlevel \
        --candidate-lowlevel-metric "$metric" \
        --candidate-lowlevel-weight "$weight" \
        --candidate-score-normalization per_query_minmax
    done
  done

  train_adapter_variant 3 "c2_h2048_d005_cosheavy" 2048 0.05 0.5 1.0 0.2
  predict_clip_adapter 3 "c2_h2048_d005_cosheavy_val64" "$RECON_V2" "$ROOT/adapter_c2_h2048_d005_cosheavy/seed_0/best.pt" "$INIT_BEST" 0.30 8 4.0
  train_adapter_variant 3 "c2_h4096_d010_contrast" 4096 0.10 0.25 1.5 0.3
  predict_clip_adapter 3 "c2_h4096_d010_contrast_val64" "$RECON_V2" "$ROOT/adapter_c2_h4096_d010_contrast/seed_0/best.pt" "$INIT_BEST" 0.30 8 4.0
) &
pid_a2_c2=$!

(
  train_lowlevel_variant 4 "b2_lowlevel_rgb_lossimg_structure" "$LOSSIMG" 1.0 0.25 0.75 0.25 0.05
  predict_clip_adapter 4 "b2_lowlevel_rgb_lossimg_structure_str025" "$RECON_V2" "$ADAPTER_V1" "$ROOT/b2_lowlevel_rgb_lossimg_structure/seed_0/eval_val64/images" 0.25 8 4.0
  predict_clip_adapter 4 "b2_lowlevel_rgb_lossimg_structure_str030" "$RECON_V2" "$ADAPTER_V1" "$ROOT/b2_lowlevel_rgb_lossimg_structure/seed_0/eval_val64/images" 0.30 8 4.0

  train_lowlevel_variant 4 "b2_lowlevel_rgb_lossimg_perceptual" "$LOSSIMG" 0.75 0.10 0.50 1.00 0.10
  predict_clip_adapter 4 "b2_lowlevel_rgb_lossimg_perceptual_str025" "$RECON_V2" "$ADAPTER_V1" "$ROOT/b2_lowlevel_rgb_lossimg_perceptual/seed_0/eval_val64/images" 0.25 8 4.0
  predict_clip_adapter 4 "b2_lowlevel_rgb_lossimg_perceptual_str030" "$RECON_V2" "$ADAPTER_V1" "$ROOT/b2_lowlevel_rgb_lossimg_perceptual/seed_0/eval_val64/images" 0.30 8 4.0

  train_lowlevel_variant 4 "b2_lowlevel_rgb_postcp_structure" "$POSTCP" 1.0 0.25 0.75 0.25 0.05
  predict_clip_adapter 4 "b2_lowlevel_rgb_postcp_structure_str025" "$RECON_V2" "$ADAPTER_V1" "$ROOT/b2_lowlevel_rgb_postcp_structure/seed_0/eval_val64/images" 0.25 8 4.0
  predict_clip_adapter 4 "b2_lowlevel_rgb_postcp_structure_str030" "$RECON_V2" "$ADAPTER_V1" "$ROOT/b2_lowlevel_rgb_postcp_structure/seed_0/eval_val64/images" 0.30 8 4.0
) &
pid_b2=$!

(
  train_clip_predictor_variant 5 "c1_clip_lossimg_h2048_d005" "$LOSSIMG" 2048 0.05
  predict_clip_adapter 5 "c1_clip_lossimg_h2048_d005_val64" "$ROOT/c1_clip_lossimg_h2048_d005/seed_0/best.pt" "$ADAPTER_V1" "$INIT_BEST" 0.30 8 4.0

  train_clip_predictor_variant 5 "c1_clip_lossimg_h4096_d010" "$LOSSIMG" 4096 0.10
  predict_clip_adapter 5 "c1_clip_lossimg_h4096_d010_val64" "$ROOT/c1_clip_lossimg_h4096_d010/seed_0/best.pt" "$ADAPTER_V1" "$INIT_BEST" 0.30 8 4.0

  train_clip_predictor_variant 5 "c1_clip_postcp_h2048_d005" "$POSTCP" 2048 0.05
  predict_clip_adapter 5 "c1_clip_postcp_h2048_d005_val64" "$ROOT/c1_clip_postcp_h2048_d005/seed_0/best.pt" "$ADAPTER_V1" "$INIT_BEST" 0.30 8 4.0
) &
pid_c1=$!

failures=0
for pid in "$pid_a1_c4" "$pid_a1_c8" "$pid_a1_c16" "$pid_a2_c2" "$pid_b2" "$pid_c1"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

summarize_now
echo "[$(date -Is)] campaign complete failures=${failures}" | tee -a "$ROOT/campaign.log"
exit "$failures"
