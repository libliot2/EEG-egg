#!/bin/bash
#SBATCH --job-name=p1_rp2_ssim
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/slurm_logs/%x_%j.out

set -euo pipefail

cd /hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TORCH_HOME=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/.cache/torch
export MPLCONFIGDIR=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/.cache/matplotlib
export XDG_CACHE_HOME=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/.cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

PY=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/.conda-envs/project1-eeg/bin/python
ROOT=outputs/reconstruction_campaigns/20260427_phase2_ssim
LOG=$ROOT/campaign.log
DATA=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/image-eeg-data
SPLIT=outputs/subsets/val64_seed0.json
RECON=outputs/experiments/reconstruction_clip_embed_v2_loss_imgsoft_local/seed_0/best.pt
ADAPTER=outputs/experiments/clip_to_kandinsky_adapter_v1/seed_0/best.pt
LOSSIMG=outputs_local/experiments/retrieval_adapter_atm_large_loss_imgsoft_dir_e40/seed_0/best.pt
POSTCP=outputs/experiments/retrieval_channel_posteriorcp28_atm_base_ides_v1_local/seed_0/best.pt

mkdir -p "$ROOT" outputs/slurm_logs

log_msg() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

train_lowlevel() {
  local name="$1"
  local retrieval="$2"
  local l1="$3"
  local mse="$4"
  local ssim="$5"
  local lpips="$6"
  local clip="$7"
  if [ -f "$ROOT/$name/seed_0/best.pt" ]; then
    log_msg "SKIP train $name existing best.pt"
    return
  fi
  log_msg "START train $name host=$(hostname)"
  "$PY" -u scripts/train_lowlevel_image_prior.py \
    --retrieval-checkpoint "$retrieval" \
    --data-dir "$DATA" \
    --output-dir "$ROOT/$name" \
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
    --lpips-weight "$lpips" \
    --clip-weight "$clip" \
    --eval-split-file "$SPLIT" \
    --eval-image-id-source val_ids \
    --eval-output-name eval_val64 \
    --device cuda
  log_msg "END train $name"
}

predict_from_init() {
  local name="$1"
  local init_dir="$2"
  local strength="$3"
  local guidance="$4"
  local low_weight="$5"
  log_msg "START predict $name strength=$strength guidance=$guidance low_weight=$low_weight"
  "$PY" -u scripts/predict_reconstruction_embed.py \
    --reconstruction-checkpoint "$RECON" \
    --conditioning-bank outputs/cache/clip_train.pt \
    --embedding-bank outputs/cache/kandinsky_train.pt \
    --conditioning-adapter "$ADAPTER" \
    --data-dir "$DATA" \
    --split train \
    --split-file "$SPLIT" \
    --image-id-source val_ids \
    --embedding-source predicted \
    --decoder-steps 20 \
    --decoder-height 512 \
    --decoder-width 512 \
    --batch-size 4 \
    --num-workers 4 \
    --local-files-only \
    --evaluate \
    --device cuda \
    --init-image-dir "$init_dir" \
    --img2img-strength "$strength" \
    --num-candidates 8 \
    --decoder-guidance-scale "$guidance" \
    --candidate-selection-mode semantic_lowlevel \
    --candidate-lowlevel-metric neg_mse \
    --candidate-lowlevel-weight "$low_weight" \
    --candidate-score-normalization per_query_minmax \
    --output-dir "$ROOT/$name"
  "$PY" scripts/summarize_recon_campaign.py "$ROOT" --output "$ROOT/summary.md" --top-k 80 >/dev/null || true
  log_msg "END predict $name"
}

log_msg "START phase2 lowlevel SSIM job=$SLURM_JOB_ID host=$(hostname)"

train_lowlevel "lowlevel_lossimg_structure" "$LOSSIMG" 1.0 0.25 0.75 0.25 0.05
train_lowlevel "lowlevel_lossimg_perceptual" "$LOSSIMG" 0.75 0.10 0.50 1.00 0.10
train_lowlevel "lowlevel_postcp_structure" "$POSTCP" 1.0 0.25 0.75 0.25 0.05

for init_name in lowlevel_lossimg_structure lowlevel_lossimg_perceptual lowlevel_postcp_structure; do
  init_dir="$ROOT/$init_name/seed_0/eval_val64/images"
  for strength in 0.25 0.28 0.30; do
    for guidance in 4.0 4.5; do
      predict_from_init "${init_name}_str${strength/./p}_g${guidance/./p}_w020" "$init_dir" "$strength" "$guidance" 0.20
    done
  done
  predict_from_init "${init_name}_str025_g400_w010" "$init_dir" 0.25 4.0 0.10
  predict_from_init "${init_name}_str025_g400_w030" "$init_dir" 0.25 4.0 0.30
done

"$PY" scripts/summarize_recon_campaign.py "$ROOT" --output "$ROOT/summary.md" --top-k 100 >/dev/null || true
log_msg "END phase2 lowlevel SSIM job=$SLURM_JOB_ID"
