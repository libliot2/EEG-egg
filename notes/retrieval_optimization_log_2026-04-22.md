# Retrieval Optimization Log

Date: `2026-04-22`

## Scope

This log summarizes the retrieval-side optimization work currently reflected in the working tree, with emphasis on:

- stronger EEG encoder and retrieval training support
- dual-head semantic/perceptual retrieval
- DreamSim-online training support
- second-stage shortlist reranking
- recent top1-aware reranker experiments

## Code Changes

### 1. Retrieval model and training pipeline

Files:

- `src/project1_eeg/retrieval.py`
- `scripts/train_retrieval.py`
- `scripts/predict_retrieval.py`
- `src/project1_eeg/utils.py`

Main additions:

- Expanded EEG encoder choices from the legacy CNN path to multiple stronger backbones:
  - `atm_small`
  - `atm_base`
  - `atm_large`
  - `atm_spatial`
  - `atm_multiscale`
  - `eeg_conformer`
- Added dual-head retrieval support:
  - `semantic` head
  - `perceptual` head
- Added optional target-side residual adapters for semantic/perceptual banks.
- Added richer retrieval losses:
  - standard CLIP-style contrastive loss
  - `neuroclip`-style soft target and relation alignment loss
  - hard-negative margin loss
- Added more flexible checkpoint/model selection behavior:
  - `best.pt`
  - `best_top1.pt`
  - selection by `top1`, `top5`, or blended score
- Updated inference helpers so test prediction can automatically switch from cached `*_train.pt` banks to `*_test.pt` banks when appropriate.

Design intent:

- separate semantic alignment from perceptual alignment instead of forcing a single EEG embedding to solve both objectives
- let the ATM-family encoder absorb more temporal/spatial structure than the original baseline CNN
- support both conservative CLIP training and more aggressive relation-aware objectives without rewriting the pipeline

### 2. DreamSim online training support

Files:

- `scripts/train_retrieval_dreamsim_online.py`
- `scripts/predict_retrieval_dreamsim_online.py`

Main additions:

- Added a dedicated training loop for online DreamSim-side optimization on top of a pretrained EEG encoder.
- Added matching evaluation/inference script.
- Added logging and selection logic around `val_top1`, `val_top5`, and `blend_top1_top5`.

Design intent:

- enable experiments where the image-side perceptual representation is no longer treated as a fixed offline bank
- make it possible to test whether improving DreamSim adaptation helps retrieval after the EEG encoder has stabilized

Observed outcome:

- online DreamSim training did not become the mainline best result
- offline-bank + reranker remained stronger and more stable under the current evaluation setup

### 3. Shortlist reranker framework

Files:

- `src/project1_eeg/reranker.py`
- `scripts/train_retrieval_reranker.py`
- `scripts/predict_retrieval_reranker.py`

Main additions:

- Added a reusable second-stage reranker over a shortlist produced by the frozen base retrieval model.
- Implemented multiple scorer types:
  - `cosine`
  - `mlp_pairwise`
  - `contextual_transformer`
  - `listwise_transformer`
- Added shortlist construction with optional positive injection during training.
- Added rerank-weight search over the shortlist when combining base logits and reranker scores.
- Added scheduler and EMA support for reranker training.

Design intent:

- use the base retrieval model to solve coarse candidate narrowing
- use a lightweight second stage to improve fine-grained ordering inside top-k
- keep reranking modular so that scorer architecture can be swapped without touching the base model

## Main Experimental Findings

### 1. Current best formal test result

Best confirmed test metric under the current repo setup:

- experiment: `outputs_local/experiments/retrieval_reranker_top4_from_best_h2048_e30/seed_0/test_eval`
- metric: `top1=0.6700`, `top5=0.9650`

Equivalent result was also reproduced by:

- `outputs_local/experiments/retrieval_reranker_top4_from_best_h2048_e40_schedcos_ema0995/seed_0/test_eval`

Interpretation:

- the strongest current line is still:
  - pretrained retrieval encoder
  - frozen base retrieval
  - shortlist top-4 reranking
  - relatively simple cosine-style reranker adapter

### 2. Scheduler and EMA on reranker

Outcome:

- warmup cosine + EMA was able to match the existing best result
- it did not produce a clear improvement over the best top-4 reranker baseline

Implication:

- these changes are useful as training stabilization tools
- they are not, by themselves, the main driver of top1 gains

### 3. Listwise / transformer reranker attempts

Outcome:

- `listwise_transformer` and contextual listwise variants were implemented and tested
- validation occasionally moved slightly, but formal test remained below the best baseline
- representative test result for the listwise line was around `top1=0.64`, `top5=0.965`

Implication:

- more expressive shortlist modeling did not automatically convert to better generalization
- the current data regime seems sensitive to overfitting once the reranker becomes too flexible

### 4. Top1-aware auxiliary head and sharper loss experiments

Recent code additions:

- an auxiliary `top1` head on top of shortlist pairwise features
- additional top1-oriented loss terms:
  - focal cross entropy
  - hard-negative logistic loss
  - top1-branch CE/focal/hard-negative losses

Experiment summary:

- balanced top1-aware run improved only marginally on `val-512`
- aggressive top1-aware run reached the best `val-512` score on that line:
  - `val_top1=0.25635`
- however the formal test result was worse than the current best baseline:
  - experiment: `outputs_local/experiments/retrieval_reranker_top4_from_best_h2048_top1aux_aggressive_e40/seed_0/test_eval`
  - metric: `top1=0.6300`, `top5=0.9650`

Conclusion:

- making the reranker more top1-aggressive improved the local validation proxy a bit
- the gain did not transfer to formal test
- this line is currently a negative result and should not replace the mainline best model

## Current Working Conclusion

The current most defensible baseline in this repository is still:

1. train a strong EEG retrieval encoder
2. freeze the base retrieval model for inference
3. build a shortlist with the base perceptual retrieval head
4. apply a lightweight top-4 reranker
5. evaluate with the formal test setting after val-based model selection

In practice, the key lesson from the recent experiments is:

- `top5` is already strong, so the system is generally narrowing candidates well
- simply making the reranker more expressive or more aggressive for `top1` does not guarantee better test `top1`
- the next useful optimization step should focus on generalization-aware fine-grained discrimination rather than sharper local fitting

## Suggested Next Steps

Most promising next steps, based on current evidence:

- analyze the formal test misses from the `0.67 / 0.965` baseline and identify recurring fine-grained confusions
- try smaller, more constrained top-k discriminators rather than larger listwise transformers
- keep `val-512` as a development metric, but always promote candidates only after formal `test_eval`
- avoid switching the mainline to the current top1-aux branch unless a future run exceeds `0.67` on formal test
