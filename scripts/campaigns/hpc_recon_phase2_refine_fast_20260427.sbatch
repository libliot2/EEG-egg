#!/bin/bash
#SBATCH --job-name=p1_rp2_fast
#SBATCH --partition=i64m1tga800u
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-45%1
#SBATCH --output=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/slurm_logs/%x_%A_%a.out

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
ROOT=outputs/reconstruction_campaigns/20260427_phase2_refine
LOG=$ROOT/campaign.log
DATA=/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/image-eeg-data
SPLIT=outputs/subsets/val64_seed0.json
INIT=outputs/reconstruction_lowlevel_init_blend/val64_low85_posterior15/images
RECON=outputs/experiments/reconstruction_clip_embed_v2_loss_imgsoft_local/seed_0/best.pt
ADAPTER=outputs/experiments/clip_to_kandinsky_adapter_v1/seed_0/best.pt

mkdir -p "$ROOT" outputs/slurm_logs

CONFIGS=(
  # balanced local refinement around a2 neg_mse.
  "balanced_negmse_w025_s028_c8_g400_o0|0.28|8|4.0|neg_mse|0.25|0"
  "balanced_negmse_w025_s030_c8_g400_o0|0.30|8|4.0|neg_mse|0.25|0"
  "balanced_negmse_w025_s032_c8_g400_o0|0.32|8|4.0|neg_mse|0.25|0"
  "balanced_negmse_w030_s028_c8_g400_o0|0.28|8|4.0|neg_mse|0.30|0"
  "balanced_negmse_w030_s030_c8_g400_o0|0.30|8|4.0|neg_mse|0.30|0"
  "balanced_negmse_w030_s032_c8_g400_o0|0.32|8|4.0|neg_mse|0.30|0"
  "balanced_negmse_w035_s028_c8_g400_o0|0.28|8|4.0|neg_mse|0.35|0"
  "balanced_negmse_w035_s030_c8_g400_o0|0.30|8|4.0|neg_mse|0.35|0"
  "balanced_negmse_w035_s032_c8_g400_o0|0.32|8|4.0|neg_mse|0.35|0"
  "balanced_negmse_w040_s028_c8_g400_o0|0.28|8|4.0|neg_mse|0.40|0"
  "balanced_negmse_w040_s030_c8_g400_o0|0.30|8|4.0|neg_mse|0.40|0"
  "balanced_negmse_w040_s032_c8_g400_o0|0.32|8|4.0|neg_mse|0.40|0"
  "balanced_negmse_w045_s028_c8_g400_o0|0.28|8|4.0|neg_mse|0.45|0"
  "balanced_negmse_w045_s030_c8_g400_o0|0.30|8|4.0|neg_mse|0.45|0"
  "balanced_negmse_w045_s032_c8_g400_o0|0.32|8|4.0|neg_mse|0.45|0"
  "balanced_negmse_w035_s030_c12_g400_o0|0.30|12|4.0|neg_mse|0.35|0"
  "balanced_negmse_w035_s032_c12_g400_o0|0.32|12|4.0|neg_mse|0.35|0"
  "balanced_negmse_w040_s030_c12_g400_o0|0.30|12|4.0|neg_mse|0.40|0"
  "balanced_negmse_w040_s032_c12_g400_o0|0.32|12|4.0|neg_mse|0.40|0"
  "balanced_negmse_w035_s030_c8_g425_o0|0.30|8|4.25|neg_mse|0.35|0"
  "balanced_negmse_w040_s030_c8_g425_o0|0.30|8|4.25|neg_mse|0.40|0"
  "balanced_negmse_w035_s030_c8_g400_o1000|0.30|8|4.0|neg_mse|0.35|1000"
  "balanced_negmse_w035_s030_c8_g400_o2000|0.30|8|4.0|neg_mse|0.35|2000"
  # CLIP-heavy branch: high guidance with light low-level regularization.
  "clipheavy_negmse_w000_s028_c8_g450_o0|0.28|8|4.5|neg_mse|0.00|0"
  "clipheavy_negmse_w000_s030_c8_g450_o0|0.30|8|4.5|neg_mse|0.00|0"
  "clipheavy_negmse_w005_s028_c8_g450_o0|0.28|8|4.5|neg_mse|0.05|0"
  "clipheavy_negmse_w005_s030_c8_g450_o0|0.30|8|4.5|neg_mse|0.05|0"
  "clipheavy_negmse_w010_s028_c8_g450_o0|0.28|8|4.5|neg_mse|0.10|0"
  "clipheavy_negmse_w010_s030_c8_g450_o0|0.30|8|4.5|neg_mse|0.10|0"
  "clipheavy_negmse_w015_s028_c8_g450_o0|0.28|8|4.5|neg_mse|0.15|0"
  "clipheavy_negmse_w015_s030_c8_g450_o0|0.30|8|4.5|neg_mse|0.15|0"
  "clipheavy_negmse_w020_s028_c8_g450_o0|0.28|8|4.5|neg_mse|0.20|0"
  "clipheavy_negmse_w020_s030_c8_g450_o0|0.30|8|4.5|neg_mse|0.20|0"
  "clipheavy_negmse_w000_s028_c16_g450_o0|0.28|16|4.5|neg_mse|0.00|0"
  "clipheavy_negmse_w000_s030_c16_g450_o0|0.30|16|4.5|neg_mse|0.00|0"
  "clipheavy_negmse_w005_s028_c16_g450_o0|0.28|16|4.5|neg_mse|0.05|0"
  "clipheavy_negmse_w005_s030_c16_g450_o0|0.30|16|4.5|neg_mse|0.05|0"
  "clipheavy_negmse_w010_s028_c16_g450_o0|0.28|16|4.5|neg_mse|0.10|0"
  "clipheavy_negmse_w010_s030_c16_g450_o0|0.30|16|4.5|neg_mse|0.10|0"
  "clipheavy_negmse_w015_s028_c16_g450_o0|0.28|16|4.5|neg_mse|0.15|0"
  "clipheavy_negmse_w015_s030_c16_g450_o0|0.30|16|4.5|neg_mse|0.15|0"
  "clipheavy_negmse_w020_s028_c16_g450_o0|0.28|16|4.5|neg_mse|0.20|0"
  "clipheavy_negmse_w020_s030_c16_g450_o0|0.30|16|4.5|neg_mse|0.20|0"
  "clipheavy_negmse_w000_s030_c8_g450_o1000|0.30|8|4.5|neg_mse|0.00|1000"
  "clipheavy_negmse_w000_s030_c8_g450_o2000|0.30|8|4.5|neg_mse|0.00|2000"
  "clipheavy_negmse_w010_s030_c8_g450_o1000|0.30|8|4.5|neg_mse|0.10|1000"
)

config="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r NAME STRENGTH CANDIDATES GUIDANCE LOW_METRIC LOW_WEIGHT SEED_OFFSET <<< "$config"

echo "[$(date -Is)] START job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) name=$NAME" | tee -a "$LOG"

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
  --init-image-dir "$INIT" \
  --img2img-strength "$STRENGTH" \
  --num-candidates "$CANDIDATES" \
  --decoder-guidance-scale "$GUIDANCE" \
  --candidate-seed-offset "$SEED_OFFSET" \
  --candidate-selection-mode semantic_lowlevel \
  --candidate-lowlevel-metric "$LOW_METRIC" \
  --candidate-lowlevel-weight "$LOW_WEIGHT" \
  --candidate-score-normalization per_query_minmax \
  --output-dir "$ROOT/$NAME"

"$PY" scripts/summarize_recon_campaign.py "$ROOT" --output "$ROOT/summary.md" --top-k 60 >/dev/null || true

echo "[$(date -Is)] END job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID name=$NAME" | tee -a "$LOG"
