# Experiment Log

This file is the project-wide experiment ledger. Use `scripts/log_experiment.py` to append new attempts.

## Current Best Retrieval
<!-- BEST_RETRIEVAL_START -->
- Attempt ID: `EXP-20260418-183838-retrieval-dreamsim-only-fixed-test`
- Scope: `test`
- Output Dir: `/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/test_eval`
- Metrics: `top1_acc=0.3450`, `top5_acc=0.6250`, `alpha=0.0000`
- Goal: Evaluate the fixed DreamSim-only retrieval checkpoint on the local 200-way test set.
<!-- BEST_RETRIEVAL_END -->

## Current Best Reconstruction
<!-- BEST_RECONSTRUCTION_START -->
- Attempt ID: `EXP-20260421-reconstruction-kandinsky-img2img-test`
- Scope: `test`
- Output Dir: `/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/eval_compare/test_seed0/hpc_img2img_v4_s20_c4_g4p0_str035`
- Metrics: `eval_clip=0.7513`, `eval_ssim=0.3767`, `eval_pixcorr=0.1567`, `eval_alex5=0.8489`
- Goal: Evaluate the current Kandinsky img2img mainline that uses retrieval prototype init plus predicted Kandinsky embedding conditioning.
- Note: this is the current best overall reconstruction mainline by combined semantic quality and structural similarity, even though the old prototype-only baseline still had higher raw SSIM.
<!-- BEST_RECONSTRUCTION_END -->

## Open Issues
<!-- OPEN_ISSUES_START -->
- Prototype-based reconstruction is qualitatively failing on test EEG because the prototype bank comes from `training_images`, while train/test concept overlap is zero.
- Closed-set retrieval test accuracy and reconstruction prototype selection are different tasks; retrieval `test_acc` must not be used as a proxy for reconstruction quality.
- The current `train prototype + residual VAE` reconstruction path is not a promising mainline and should only be kept as a baseline.
<!-- OPEN_ISSUES_END -->

## Current Running Experiments

- `9701725` `p1_r_atm_b`: queued on HKUST-GZ HPC (`i64m1tga800ue`), training `retrieval_dreamsim_only_atm_base_v1` with robust checkpoint selection (`blend_top1_top5`) and last-5 checkpoint retention.
- `9701726` `p1_r_ides`: queued on HKUST-GZ HPC (`i64m1tga800ue`), training `retrieval_dreamsim_only_atm_base_ides_v1` with the same ATM-base backbone plus random trial averaging (`k=2..4`).

## Experiment Entries
<!-- LOG_ENTRIES_START -->
### EXP-20260421-125412-retrieval-atm-base-v1 [started]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260421-125412-retrieval-atm-base-v1","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"sbatch retrieval_atm_base_v1.sbatch (job 9701725)","goal":"Train DreamSim-only ATM-base retrieval baseline with robust checkpoint selection.","key_inputs":["encoder_type=atm_base","hidden_dim=384, embedding_dim=1024","transformer_layers=4, transformer_heads=8","selection_metric=blend_top1_top5","keep_last_k=5"],"kind":"train","metric_scope":"val","metric_source":null,"metrics":{},"next_action":"Wait for the queued HPC job to start, then compare its best-by-blend checkpoint against retrieval_dreamsim_only_atm_small_fixed.","observations":["The first submission used stale remote code paths, so it was cancelled and resubmitted as job 9701725 after rsyncing the updated retrieval/data modules to the correct remote paths.","Current HPC scheduler state: PENDING (Priority)."],"output_dir":"/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_base_v1","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":"blend_top1_top5","status":"started","timestamp":"2026-04-21T12:54:12+08:00"} -->

- Timestamp: 2026-04-21T12:54:12+08:00
- Area: retrieval
- Kind: train
- Goal: Train DreamSim-only ATM-base retrieval baseline with robust checkpoint selection.
- Metric Scope: val
- Metric Source: None
- Output Dir: /hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_base_v1
- Backfilled: no

#### Command
```bash
sbatch retrieval_atm_base_v1.sbatch (job 9701725)
```

#### Key Inputs
- encoder_type=atm_base
- hidden_dim=384, embedding_dim=1024
- transformer_layers=4, transformer_heads=8
- selection_metric=blend_top1_top5
- keep_last_k=5

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- The first submission used stale remote code paths, so it was cancelled and resubmitted as job `9701725` after rsyncing the updated retrieval/data modules to the correct remote paths.
- Current HPC scheduler state: `PENDING (Priority)`.

#### Next Action
Wait for the queued HPC job to start, then compare its best-by-blend checkpoint against `retrieval_dreamsim_only_atm_small_fixed`.

### EXP-20260421-125412-retrieval-atm-base-ides-v1 [started]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260421-125412-retrieval-atm-base-ides-v1","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"sbatch retrieval_atm_base_ides_v1.sbatch (job 9701726)","goal":"Train ATM-base retrieval with random trial averaging (IDES-style) on DreamSim target.","key_inputs":["encoder_type=atm_base","trial_sampling=random_avg, k=2..4","selection_metric=blend_top1_top5","keep_last_k=5"],"kind":"train","metric_scope":"val","metric_source":null,"metrics":{},"next_action":"Wait for the queued HPC job to start, then compare whether trial-sampling beats the plain ATM-base retrieval run on the same blend selection rule.","observations":["The first submission used stale remote code paths, so it was cancelled and resubmitted as job 9701726 after rsyncing the updated retrieval/data modules to the correct remote paths.","Current HPC scheduler state: PENDING (Priority)."],"output_dir":"/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_base_ides_v1","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":"blend_top1_top5","status":"started","timestamp":"2026-04-21T12:54:12+08:00"} -->

- Timestamp: 2026-04-21T12:54:12+08:00
- Area: retrieval
- Kind: train
- Goal: Train ATM-base retrieval with random trial averaging (IDES-style) on DreamSim target.
- Metric Scope: val
- Metric Source: None
- Output Dir: /hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_base_ides_v1
- Backfilled: no

#### Command
```bash
sbatch retrieval_atm_base_ides_v1.sbatch (job 9701726)
```

#### Key Inputs
- encoder_type=atm_base
- trial_sampling=random_avg, k=2..4
- selection_metric=blend_top1_top5
- keep_last_k=5

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- The first submission used stale remote code paths, so it was cancelled and resubmitted as job `9701726` after rsyncing the updated retrieval/data modules to the correct remote paths.
- Current HPC scheduler state: `PENDING (Priority)`.

#### Next Action
Wait for the queued HPC job to start, then compare whether trial-sampling beats the plain ATM-base retrieval run on the same blend selection rule.

### EXP-20260420-163059-reconstruction [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-163059-reconstruction","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"sbatch hpc_img2img_val64_strength_long (array 0-2%2)","goal":"Queue val64 img2img strength sweep on HKUST-GZ HPC","key_inputs":[],"kind":"ablation","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":null,"selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-20T16:30:59+08:00"} -->

- Timestamp: 2026-04-20T16:30:59+08:00
- Area: reconstruction
- Kind: ablation
- Goal: Queue val64 img2img strength sweep on HKUST-GZ HPC
- Metric Scope: unknown
- Metric Source: None
- Output Dir: None
- Backfilled: no

#### Command
```bash
sbatch hpc_img2img_val64_strength_long (array 0-2%2)
```

#### Key Inputs
- None

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
None

### EXP-20260420-163055-reconstruction [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-163055-reconstruction","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"sbatch hpc_val8_prototype_init_long -> hpc_img2img_smoke_long","goal":"Queue val8 prototype init + img2img smoke on HKUST-GZ HPC","key_inputs":[],"kind":"smoke","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":null,"selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-20T16:30:55+08:00"} -->

- Timestamp: 2026-04-20T16:30:55+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Queue val8 prototype init + img2img smoke on HKUST-GZ HPC
- Metric Scope: unknown
- Metric Source: None
- Output Dir: None
- Backfilled: no

#### Command
```bash
sbatch hpc_val8_prototype_init_long -> hpc_img2img_smoke_long
```

#### Key Inputs
- None

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
None

### EXP-20260420-142948-reconstruction-d0-sweep [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-142948-reconstruction-d0-sweep","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"sbatch array D0 decoder sweep (6 configs, %2 concurrency)","goal":"HPC D0 decoder micro-sweep for v4_proxyselect on val64 with 2xA800","key_inputs":[],"kind":"ablation","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":null,"selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-20T14:29:48+08:00"} -->

- Timestamp: 2026-04-20T14:29:48+08:00
- Area: reconstruction
- Kind: ablation
- Goal: HPC D0 decoder micro-sweep for v4_proxyselect on val64 with 2xA800
- Metric Scope: unknown
- Metric Source: None
- Output Dir: None
- Backfilled: no

#### Command
```bash
sbatch array D0 decoder sweep (6 configs, %2 concurrency)
```

#### Key Inputs
- None

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
None

### EXP-20260420-142948-reconstruction-hpc-smoke [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-142948-reconstruction-hpc-smoke","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"sbatch hpc warmup+smoke v4 val8 on A800","goal":"HPC warmup + v4 smoke decode on val8","key_inputs":[],"kind":"smoke","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":null,"selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-20T14:29:48+08:00"} -->

- Timestamp: 2026-04-20T14:29:48+08:00
- Area: reconstruction
- Kind: smoke
- Goal: HPC warmup + v4 smoke decode on val8
- Metric Scope: unknown
- Metric Source: None
- Output Dir: None
- Backfilled: no

#### Command
```bash
sbatch hpc warmup+smoke v4 val8 on A800
```

#### Key Inputs
- None

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
None

### EXP-20260420-113124-reconstruction [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-113124-reconstruction","backfilled":true,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"/hpc2hdd/home/dsaa2012_054/jhspoolers/DeepLearning/.conda-envs/project1-eeg/bin/python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --pretrain-checkpoint outputs/experiments/eeg_mask_pretrain_v1/seed_0/encoder_best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800 --epochs 30 --batch-size 64 --num-workers 8 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 0 --image-eval-limit 0 --staged-regression-finetune --freeze-encoder-epochs 5 --encoder-learning-rate 3e-5 --head-learning-rate 3e-4 --device cuda","goal":"Fine-tune Kandinsky embedding regression with staged head-first optimization from a masked-pretrained EEG encoder on HKUST-GZ HPC A800.","key_inputs":["staged_regression_finetune=true","freeze_encoder_epochs=5","encoder_lr=3e-5 head_lr=3e-4"],"kind":"train","metric_scope":"val","metric_source":null,"metrics":{},"next_action":"Do not promote masked-pretrain initialization to the reconstruction mainline; return to stronger representation ideas or a different auxiliary objective.","observations":["Backfilled from the April 20 A800 HPC run.","Staged finetuning was better than direct finetuning but still far below the v4_proxyselect baseline, so masked-pretrain initialization remains a negative result."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-20T11:54:41+08:00"} -->

- Timestamp: 2026-04-20T11:54:41+08:00
- Area: reconstruction
- Kind: train
- Goal: Fine-tune Kandinsky embedding regression with staged head-first optimization from a masked-pretrained EEG encoder on HKUST-GZ HPC A800.
- Metric Scope: val
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800
- Backfilled: yes

#### Command
```bash
/hpc2hdd/home/dsaa2012_054/jhspoolers/DeepLearning/.conda-envs/project1-eeg/bin/python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --pretrain-checkpoint outputs/experiments/eeg_mask_pretrain_v1/seed_0/encoder_best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v6_mask_pretrain_staged_a800 --epochs 30 --batch-size 64 --num-workers 8 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 0 --image-eval-limit 0 --staged-regression-finetune --freeze-encoder-epochs 5 --encoder-learning-rate 3e-5 --head-learning-rate 3e-4 --device cuda
```

#### Key Inputs
- staged_regression_finetune=true
- freeze_encoder_epochs=5
- encoder_lr=3e-5 head_lr=3e-4

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- Backfilled from the April 20 A800 HPC run.
- Staged finetuning was better than direct finetuning but still far below the v4_proxyselect baseline, so masked-pretrain initialization remains a negative result.

#### Next Action
Do not promote masked-pretrain initialization to the reconstruction mainline; return to stronger representation ideas or a different auxiliary objective.

### EXP-20260420-105954-reconstruction [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-105954-reconstruction","backfilled":true,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"/hpc2hdd/home/dsaa2012_054/jhspoolers/DeepLearning/.conda-envs/project1-eeg/bin/python scripts/train_eeg_pretrain.py --output-dir outputs/experiments/eeg_mask_pretrain_v1 --epochs 20 --batch-size 128 --num-workers 8 --device cuda","goal":"Masked EEG pretrain for encoder initialization before Kandinsky embedding regression on HKUST-GZ HPC A800.","key_inputs":["partition=i64m1tga800ue","gpu=A800 x1","cpus=8 mem=64G"],"kind":"train","metric_scope":"val","metric_source":null,"metrics":{},"next_action":"Compare encoder initialization against the current Kandinsky embedding regression baseline.","observations":["Backfilled from the April 20 A800 HPC run.","Masked EEG pretraining converged cleanly but its value must be judged by downstream reconstruction transfer."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/eeg_mask_pretrain_v1","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-20T11:54:41+08:00"} -->

- Timestamp: 2026-04-20T11:54:41+08:00
- Area: reconstruction
- Kind: train
- Goal: Masked EEG pretrain for encoder initialization before Kandinsky embedding regression on HKUST-GZ HPC A800.
- Metric Scope: val
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/eeg_mask_pretrain_v1
- Backfilled: yes

#### Command
```bash
/hpc2hdd/home/dsaa2012_054/jhspoolers/DeepLearning/.conda-envs/project1-eeg/bin/python scripts/train_eeg_pretrain.py --output-dir outputs/experiments/eeg_mask_pretrain_v1 --epochs 20 --batch-size 128 --num-workers 8 --device cuda
```

#### Key Inputs
- partition=i64m1tga800ue
- gpu=A800 x1
- cpus=8 mem=64G

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- Backfilled from the April 20 A800 HPC run.
- Masked EEG pretraining converged cleanly but its value must be judged by downstream reconstruction transfer.

#### Next Action
Compare encoder initialization against the current Kandinsky embedding regression baseline.

### EXP-20260420-112007-reconstruction [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-112007-reconstruction","backfilled":true,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"/hpc2hdd/home/dsaa2012_054/jhspoolers/DeepLearning/.conda-envs/project1-eeg/bin/python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --pretrain-checkpoint outputs/experiments/eeg_mask_pretrain_v1/seed_0/encoder_best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v5_mask_pretrain_a800 --epochs 30 --batch-size 64 --num-workers 8 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 0 --image-eval-limit 0 --device cuda","goal":"Fine-tune Kandinsky embedding regression with a masked-pretrained EEG encoder on HKUST-GZ HPC A800.","key_inputs":["retrieval_ckpt=retrieval_dreamsim_only_atm_small_fixed","pretrain_ckpt=eeg_mask_pretrain_v1/seed_0/encoder_best.pt","selection_metric=val_subset_top1_then_top5"],"kind":"train","metric_scope":"val","metric_source":null,"metrics":{},"next_action":"Try a head-first staged finetuning schedule or retire masked-pretrain initialization if that also fails.","observations":["Backfilled from the April 20 A800 HPC run.","This direct finetune was substantially worse than reconstruction_kandinsky_embed_v4_proxyselect despite loading the pretrained encoder with zero missing keys."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v5_mask_pretrain_a800","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-20T11:54:41+08:00"} -->

- Timestamp: 2026-04-20T11:54:41+08:00
- Area: reconstruction
- Kind: train
- Goal: Fine-tune Kandinsky embedding regression with a masked-pretrained EEG encoder on HKUST-GZ HPC A800.
- Metric Scope: val
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v5_mask_pretrain_a800
- Backfilled: yes

#### Command
```bash
/hpc2hdd/home/dsaa2012_054/jhspoolers/DeepLearning/.conda-envs/project1-eeg/bin/python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --pretrain-checkpoint outputs/experiments/eeg_mask_pretrain_v1/seed_0/encoder_best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v5_mask_pretrain_a800 --epochs 30 --batch-size 64 --num-workers 8 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 0 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- retrieval_ckpt=retrieval_dreamsim_only_atm_small_fixed
- pretrain_ckpt=eeg_mask_pretrain_v1/seed_0/encoder_best.pt
- selection_metric=val_subset_top1_then_top5

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- Backfilled from the April 20 A800 HPC run.
- This direct finetune was substantially worse than reconstruction_kandinsky_embed_v4_proxyselect despite loading the pretrained encoder with zero missing keys.

#### Next Action
Try a head-first staged finetuning schedule or retire masked-pretrain initialization if that also fails.

### EXP-20260419-rag-residual-v1-train [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-v1-train","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_v1 --epochs 30 --freeze-encoder-epochs 5 --batch-size 64 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 10 --image-eval-limit 16 --num-candidates 4 --decoder-steps 20 --decoder-guidance-scale 4.0 --device cuda:0 --local-files-only","goal":"Train the first full rag_residual reconstruction model without text or pretraining.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/kandinsky_train.pt","outputs/cache/dreamsim_train.pt"],"kind":"train","metric_scope":"val","metric_source":null,"metrics":{},"next_action":"Pivot to a revised residual design or a stronger text/pretraining variant instead of promoting this run to val64/full-val.","observations":["Training completed all 30 epochs.","Best checkpoint was selected at epoch 22 by proxy retrieval metrics.","The retrieval residual gate collapsed close to zero, so the model behaved mostly like a direct regressor."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_v1","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T11:13:25+08:00"} -->

- Timestamp: 2026-04-19T11:13:25+08:00
- Area: reconstruction
- Kind: train
- Goal: Train the first full rag_residual reconstruction model without text or pretraining.
- Metric Scope: val
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_v1
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_v1 --epochs 30 --freeze-encoder-epochs 5 --batch-size 64 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 10 --image-eval-limit 16 --num-candidates 4 --decoder-steps 20 --decoder-guidance-scale 4.0 --device cuda:0 --local-files-only
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/kandinsky_train.pt
- outputs/cache/dreamsim_train.pt

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- Training completed all 30 epochs.
- Best checkpoint was selected at epoch 22 by proxy retrieval metrics.
- The retrieval residual gate collapsed close to zero, so the model behaved mostly like a direct regressor.

#### Next Action
Pivot to a revised residual design or a stronger text/pretraining variant instead of promoting this run to val64/full-val.

### EXP-20260419-rag-residual-smoke-predict-retry [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-smoke-predict-retry","backfilled":false,"best_decoder_eval":{"eval_alex5":0.75,"eval_clip":0.6964285714285714,"eval_inception":0.75,"eval_pixcorr":0.07218751235577439,"eval_ssim":0.007025868884483819},"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only","goal":"Retry rag_residual prediction smoke after fixing local decoder fallback.","key_inputs":["outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt","outputs/subsets/val8_smoke_seed0.json"],"kind":"smoke","metric_scope":"smoke","metric_source":"reconstruction_metrics.json","metrics":{"decoder_guidance_scale":4.0,"decoder_steps":50.0,"eval_alex2":0.6964285714285714,"eval_alex5":0.75,"eval_clip":0.6964285714285714,"eval_effnet":0.898733377456665,"eval_inception":0.75,"eval_pixcorr":0.07218751235577439,"eval_ssim":0.007025868884483819,"eval_swav":0.8953134417533875,"num_candidates":4.0},"next_action":"Launch the first full rag_residual training run.","observations":["Prediction smoke succeeded after adding local decoder path fallback."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T10:46:53+08:00"} -->

- Timestamp: 2026-04-19T10:46:53+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Retry rag_residual prediction smoke after fixing local decoder fallback.
- Metric Scope: smoke
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only
```

#### Key Inputs
- outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt
- outputs/subsets/val8_smoke_seed0.json

#### Metrics
- `decoder_guidance_scale` = 4.0000
- `decoder_steps` = 50.0000
- `eval_alex2` = 0.6964
- `eval_alex5` = 0.7500
- `eval_clip` = 0.6964
- `eval_effnet` = 0.8987
- `eval_inception` = 0.7500
- `eval_pixcorr` = 0.0722
- `eval_ssim` = 0.0070
- `eval_swav` = 0.8953
- `num_candidates` = 4.0000

#### Selection Summary
- Best Decoder Eval: `eval_alex5=0.7500`, `eval_clip=0.6964`, `eval_inception=0.7500`, `eval_pixcorr=0.0722`, `eval_ssim=0.0070`

#### Observations
- Prediction smoke succeeded after adding local decoder path fallback.

#### Next Action
Launch the first full rag_residual training run.

### EXP-20260419-rag-residual-v1-train [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-v1-train","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_v1 --epochs 30 --freeze-encoder-epochs 5 --batch-size 64 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 10 --image-eval-limit 16 --num-candidates 4 --decoder-steps 20 --decoder-guidance-scale 4.0 --device cuda:0 --local-files-only","goal":"Train the first full rag_residual reconstruction model without text or pretraining.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/kandinsky_train.pt","outputs/cache/dreamsim_train.pt"],"kind":"train","metric_scope":"val","metric_source":null,"metrics":{},"next_action":"Compare val64/full-val against kandinsky_predicted_v4_fast once training completes.","observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_v1","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-19T10:46:53+08:00"} -->

- Timestamp: 2026-04-19T10:46:53+08:00
- Area: reconstruction
- Kind: train
- Goal: Train the first full rag_residual reconstruction model without text or pretraining.
- Metric Scope: val
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_v1
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_v1 --epochs 30 --freeze-encoder-epochs 5 --batch-size 64 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-every 10 --image-eval-limit 16 --num-candidates 4 --decoder-steps 20 --decoder-guidance-scale 4.0 --device cuda:0 --local-files-only
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/kandinsky_train.pt
- outputs/cache/dreamsim_train.pt

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
Compare val64/full-val against kandinsky_predicted_v4_fast once training completes.

### EXP-20260419-rag-residual-smoke-train [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-smoke-train","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=2 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_smoke --epochs 1 --freeze-encoder-epochs 1 --batch-size 32 --embedding-eval-every 1 --image-eval-limit 0 --device cuda:0","goal":"Smoke-test rag_residual training path for Kandinsky embedding reconstruction.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/kandinsky_train.pt","outputs/cache/dreamsim_train.pt"],"kind":"smoke","metric_scope":"smoke","metric_source":null,"metrics":{},"next_action":"Run a val8 prediction smoke and then launch the first full training run.","observations":["The rag_residual training path completed 1 epoch and produced a usable checkpoint."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_smoke","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T10:46:53+08:00"} -->

- Timestamp: 2026-04-19T10:46:53+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Smoke-test rag_residual training path for Kandinsky embedding reconstruction.
- Metric Scope: smoke
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_smoke
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_smoke --epochs 1 --freeze-encoder-epochs 1 --batch-size 32 --embedding-eval-every 1 --image-eval-limit 0 --device cuda:0
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/kandinsky_train.pt
- outputs/cache/dreamsim_train.pt

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- The rag_residual training path completed 1 epoch and produced a usable checkpoint.

#### Next Action
Run a val8 prediction smoke and then launch the first full training run.

### EXP-20260419-rag-residual-smoke-predict [failed]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-smoke-predict","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only","goal":"Smoke-test rag_residual prediction path and image evaluation on val8.","key_inputs":["outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt","outputs/subsets/val8_smoke_seed0.json"],"kind":"smoke","metric_scope":"smoke","metric_source":null,"metrics":{},"next_action":"Retry after adding local decoder path fallback in kandinsky.py.","observations":["Prediction failed before generation because local_files_only could not resolve the decoder repo id to the locally cached decoder directory."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"failed","timestamp":"2026-04-19T10:44:27+08:00"} -->

- Timestamp: 2026-04-19T10:44:27+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Smoke-test rag_residual prediction path and image evaluation on val8.
- Metric Scope: smoke
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only
```

#### Key Inputs
- outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt
- outputs/subsets/val8_smoke_seed0.json

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- Prediction failed before generation because local_files_only could not resolve the decoder repo id to the locally cached decoder directory.

#### Next Action
Retry after adding local decoder path fallback in kandinsky.py.

### EXP-20260419-rag-residual-smoke-predict-retry [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-smoke-predict-retry","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only","goal":"Retry rag_residual prediction smoke after fixing local decoder fallback.","key_inputs":["outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt","outputs/subsets/val8_smoke_seed0.json"],"kind":"smoke","metric_scope":"smoke","metric_source":null,"metrics":{},"next_action":"If retry succeeds, launch the first full rag_residual training run.","observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-19T10:44:27+08:00"} -->

- Timestamp: 2026-04-19T10:44:27+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Retry rag_residual prediction smoke after fixing local decoder fallback.
- Metric Scope: smoke
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only
```

#### Key Inputs
- outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt
- outputs/subsets/val8_smoke_seed0.json

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
If retry succeeds, launch the first full rag_residual training run.

### EXP-20260419-rag-residual-smoke-predict [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-smoke-predict","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only","goal":"Smoke-test rag_residual prediction path and image evaluation on val8.","key_inputs":["outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt","outputs/subsets/val8_smoke_seed0.json"],"kind":"smoke","metric_scope":"smoke","metric_source":null,"metrics":{},"next_action":"If prediction smoke succeeds, launch the first full rag_residual training run.","observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-19T10:43:19+08:00"} -->

- Timestamp: 2026-04-19T10:43:19+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Smoke-test rag_residual prediction path and image evaluation on val8.
- Metric Scope: smoke
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --reconstruction-checkpoint outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt --split train --split-file outputs/subsets/val8_smoke_seed0.json --image-id-source val_ids --embedding-source predicted --evaluate --output-dir outputs/eval_compare/val8_smoke_seed0/kandinsky_rag_residual_smoke --device cuda:0 --local-files-only
```

#### Key Inputs
- outputs/experiments/reconstruction_kandinsky_rag_residual_smoke/seed_0/best.pt
- outputs/subsets/val8_smoke_seed0.json

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
If prediction smoke succeeds, launch the first full rag_residual training run.

### EXP-20260419-rag-residual-smoke-train [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-rag-residual-smoke-train","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=2 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_smoke --epochs 1 --freeze-encoder-epochs 1 --batch-size 32 --embedding-eval-every 1 --image-eval-limit 0 --device cuda:0","goal":"Smoke-test rag_residual training path for Kandinsky embedding reconstruction.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/kandinsky_train.pt","outputs/cache/dreamsim_train.pt"],"kind":"smoke","metric_scope":"smoke","metric_source":null,"metrics":{},"next_action":"Run a val8 prediction smoke if training completes successfully.","observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_smoke","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-19T10:41:56+08:00"} -->

- Timestamp: 2026-04-19T10:41:56+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Smoke-test rag_residual training path for Kandinsky embedding reconstruction.
- Metric Scope: smoke
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_rag_residual_smoke
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_reconstruction_embed.py --model-type rag_residual --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --retrieval-bank outputs/cache/dreamsim_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_rag_residual_smoke --epochs 1 --freeze-encoder-epochs 1 --batch-size 32 --embedding-eval-every 1 --image-eval-limit 0 --device cuda:0
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/kandinsky_train.pt
- outputs/cache/dreamsim_train.pt

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
Run a val8 prediction smoke if training completes successfully.

### EXP-20260419-004420-reconstruction-kandinsky-predicted-v4-quality-val64 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-004420-reconstruction-kandinsky-predicted-v4-quality-val64","backfilled":false,"best_decoder_eval":{"eval_alex5":0.7839781746031746,"eval_clip":0.699156746031746,"eval_inception":0.6994047619047619,"eval_pixcorr":0.058450545700960255,"eval_ssim":0.0497572123390514},"best_embedding_proxy":null,"command":null,"goal":"val64 predicted-v4 quality decoder config sweep point","key_inputs":[],"kind":"ablation","metric_scope":"val","metric_source":"reconstruction_metrics.json","metrics":{"decoder_guidance_scale":6.0,"decoder_steps":60.0,"eval_alex2":0.7058531746031746,"eval_alex5":0.7839781746031746,"eval_clip":0.699156746031746,"eval_effnet":0.9161391258239746,"eval_inception":0.6994047619047619,"eval_pixcorr":0.058450545700960255,"eval_ssim":0.0497572123390514,"eval_swav":0.8999993205070496,"num_candidates":8.0},"next_action":null,"observations":["Quality config is slower and does not beat fast config on eval_clip.","Fast config remains the best predicted decoder point on val64."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_quality","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T00:35:56+08:00"} -->

- Timestamp: 2026-04-19T00:35:56+08:00
- Area: reconstruction
- Kind: ablation
- Goal: val64 predicted-v4 quality decoder config sweep point
- Metric Scope: val
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_quality
- Backfilled: no

#### Command
```bash
# none recorded
```

#### Key Inputs
- None

#### Metrics
- `decoder_guidance_scale` = 6.0000
- `decoder_steps` = 60.0000
- `eval_alex2` = 0.7059
- `eval_alex5` = 0.7840
- `eval_clip` = 0.6992
- `eval_effnet` = 0.9161
- `eval_inception` = 0.6994
- `eval_pixcorr` = 0.0585
- `eval_ssim` = 0.0498
- `eval_swav` = 0.9000
- `num_candidates` = 8.0000

#### Selection Summary
- Best Decoder Eval: `eval_alex5=0.7840`, `eval_clip=0.6992`, `eval_inception=0.6994`, `eval_pixcorr=0.0585`, `eval_ssim=0.0498`

#### Observations
- Quality config is slower and does not beat fast config on eval_clip.
- Fast config remains the best predicted decoder point on val64.

#### Next Action
None

### EXP-20260419-003340-reconstruction-kandinsky-predicted-v4-fast-val64 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-003340-reconstruction-kandinsky-predicted-v4-fast-val64","backfilled":false,"best_decoder_eval":{"eval_alex5":0.8204365079365079,"eval_clip":0.7078373015873016,"eval_inception":0.7021329365079365,"eval_pixcorr":0.08734651446054051,"eval_ssim":0.05086171048745882},"best_embedding_proxy":null,"command":null,"goal":"val64 predicted-v4 fast decoder config sweep point","key_inputs":[],"kind":"ablation","metric_scope":"val","metric_source":"reconstruction_metrics.json","metrics":{"decoder_guidance_scale":4.0,"decoder_steps":20.0,"eval_alex2":0.7264384920634921,"eval_alex5":0.8204365079365079,"eval_clip":0.7078373015873016,"eval_effnet":0.9222778081893921,"eval_inception":0.7021329365079365,"eval_pixcorr":0.08734651446054051,"eval_ssim":0.05086171048745882,"eval_swav":0.8967015743255615,"num_candidates":4.0},"next_action":null,"observations":["20-step fast config outperforms current 40-step balanced config on eval_clip.","Fast config is the current best predicted decoder point on val64."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_fast","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T00:25:57+08:00"} -->

- Timestamp: 2026-04-19T00:25:57+08:00
- Area: reconstruction
- Kind: ablation
- Goal: val64 predicted-v4 fast decoder config sweep point
- Metric Scope: val
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_fast
- Backfilled: no

#### Command
```bash
# none recorded
```

#### Key Inputs
- None

#### Metrics
- `decoder_guidance_scale` = 4.0000
- `decoder_steps` = 20.0000
- `eval_alex2` = 0.7264
- `eval_alex5` = 0.8204
- `eval_clip` = 0.7078
- `eval_effnet` = 0.9223
- `eval_inception` = 0.7021
- `eval_pixcorr` = 0.0873
- `eval_ssim` = 0.0509
- `eval_swav` = 0.8967
- `num_candidates` = 4.0000

#### Selection Summary
- Best Decoder Eval: `eval_alex5=0.8204`, `eval_clip=0.7078`, `eval_inception=0.7021`, `eval_pixcorr=0.0873`, `eval_ssim=0.0509`

#### Observations
- 20-step fast config outperforms current 40-step balanced config on eval_clip.
- Fast config is the current best predicted decoder point on val64.

#### Next Action
None

### EXP-20260419-002640-reconstruction-kandinsky-retrieval-top1-val64 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-002640-reconstruction-kandinsky-retrieval-top1-val64","backfilled":false,"best_decoder_eval":{"eval_alex5":0.7981150793650794,"eval_clip":0.6612103174603174,"eval_inception":0.6478174603174603,"eval_pixcorr":0.0793713516260108,"eval_ssim":0.021650344735536585},"best_embedding_proxy":null,"command":null,"goal":"val64 compare kandinsky retrieval_top1 with local decoder","key_inputs":[],"kind":"eval","metric_scope":"val","metric_source":"reconstruction_metrics.json","metrics":{"decoder_guidance_scale":4.0,"decoder_steps":40.0,"eval_alex2":0.7294146825396826,"eval_alex5":0.7981150793650794,"eval_clip":0.6612103174603174,"eval_effnet":0.9380069971084595,"eval_inception":0.6478174603174603,"eval_pixcorr":0.0793713516260108,"eval_ssim":0.021650344735536585,"eval_swav":0.8960443735122681,"num_candidates":4.0},"next_action":null,"observations":["Retrieval-top1 conditioning is below predicted-v4 on val64.","Discrete retrieval embedding is not the current best path."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_retrieval_top1_local","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T00:21:21+08:00"} -->

- Timestamp: 2026-04-19T00:21:21+08:00
- Area: reconstruction
- Kind: eval
- Goal: val64 compare kandinsky retrieval_top1 with local decoder
- Metric Scope: val
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_retrieval_top1_local
- Backfilled: no

#### Command
```bash
# none recorded
```

#### Key Inputs
- None

#### Metrics
- `decoder_guidance_scale` = 4.0000
- `decoder_steps` = 40.0000
- `eval_alex2` = 0.7294
- `eval_alex5` = 0.7981
- `eval_clip` = 0.6612
- `eval_effnet` = 0.9380
- `eval_inception` = 0.6478
- `eval_pixcorr` = 0.0794
- `eval_ssim` = 0.0217
- `eval_swav` = 0.8960
- `num_candidates` = 4.0000

#### Selection Summary
- Best Decoder Eval: `eval_alex5=0.7981`, `eval_clip=0.6612`, `eval_inception=0.6478`, `eval_pixcorr=0.0794`, `eval_ssim=0.0217`

#### Observations
- Retrieval-top1 conditioning is below predicted-v4 on val64.
- Discrete retrieval embedding is not the current best path.

#### Next Action
None

### EXP-20260419-002210-reconstruction-kandinsky-predicted-v4-val64 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-002210-reconstruction-kandinsky-predicted-v4-val64","backfilled":false,"best_decoder_eval":{"eval_alex5":0.7946428571428571,"eval_clip":0.6917162698412699,"eval_inception":0.7006448412698413,"eval_pixcorr":0.06006052700162377,"eval_ssim":0.04703260567812412},"best_embedding_proxy":null,"command":null,"goal":"val64 compare kandinsky predicted v4 proxyselect with local decoder","key_inputs":[],"kind":"eval","metric_scope":"val","metric_source":"reconstruction_metrics.json","metrics":{"decoder_guidance_scale":4.0,"decoder_steps":40.0,"eval_alex2":0.7291666666666666,"eval_alex5":0.7946428571428571,"eval_clip":0.6917162698412699,"eval_effnet":0.9330172538757324,"eval_inception":0.7006448412698413,"eval_pixcorr":0.06006052700162377,"eval_ssim":0.04703260567812412,"eval_swav":0.8978103399276733,"num_candidates":4.0},"next_action":null,"observations":["Predicted Kandinsky embedding outperforms residual-VAE baseline on val64.","Mainline checkpoint is reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_local","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T00:15:59+08:00"} -->

- Timestamp: 2026-04-19T00:15:59+08:00
- Area: reconstruction
- Kind: eval
- Goal: val64 compare kandinsky predicted v4 proxyselect with local decoder
- Metric Scope: val
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_predicted_v4_local
- Backfilled: no

#### Command
```bash
# none recorded
```

#### Key Inputs
- None

#### Metrics
- `decoder_guidance_scale` = 4.0000
- `decoder_steps` = 40.0000
- `eval_alex2` = 0.7292
- `eval_alex5` = 0.7946
- `eval_clip` = 0.6917
- `eval_effnet` = 0.9330
- `eval_inception` = 0.7006
- `eval_pixcorr` = 0.0601
- `eval_ssim` = 0.0470
- `eval_swav` = 0.8978
- `num_candidates` = 4.0000

#### Selection Summary
- Best Decoder Eval: `eval_alex5=0.7946`, `eval_clip=0.6917`, `eval_inception=0.7006`, `eval_pixcorr=0.0601`, `eval_ssim=0.0470`

#### Observations
- Predicted Kandinsky embedding outperforms residual-VAE baseline on val64.
- Mainline checkpoint is reconstruction_kandinsky_embed_v4_proxyselect/seed_0/best.pt.

#### Next Action
None

### EXP-20260419-001920-reconstruction-kandinsky-groundtruth-val64 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-001920-reconstruction-kandinsky-groundtruth-val64","backfilled":false,"best_decoder_eval":{"eval_alex5":0.982390873015873,"eval_clip":0.9972718253968254,"eval_inception":0.9995039682539683,"eval_pixcorr":0.22889926130349297,"eval_ssim":0.03639610297717804},"best_embedding_proxy":null,"command":null,"goal":"val64 compare kandinsky ground_truth with local decoder","key_inputs":[],"kind":"eval","metric_scope":"val","metric_source":"reconstruction_metrics.json","metrics":{"decoder_guidance_scale":4.0,"decoder_steps":40.0,"eval_alex2":0.9419642857142857,"eval_alex5":0.982390873015873,"eval_clip":0.9972718253968254,"eval_effnet":0.6930497884750366,"eval_inception":0.9995039682539683,"eval_pixcorr":0.22889926130349297,"eval_ssim":0.03639610297717804,"eval_swav":0.8948718309402466,"num_candidates":4.0},"next_action":null,"observations":["Ground-truth Kandinsky embedding establishes decoder upper bound on val64.","Offline local decoder path is working end-to-end."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T00:15:32+08:00"} -->

- Timestamp: 2026-04-19T00:15:32+08:00
- Area: reconstruction
- Kind: eval
- Goal: val64 compare kandinsky ground_truth with local decoder
- Metric Scope: val
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local
- Backfilled: no

#### Command
```bash
# none recorded
```

#### Key Inputs
- None

#### Metrics
- `decoder_guidance_scale` = 4.0000
- `decoder_steps` = 40.0000
- `eval_alex2` = 0.9420
- `eval_alex5` = 0.9824
- `eval_clip` = 0.9973
- `eval_effnet` = 0.6930
- `eval_inception` = 0.9995
- `eval_pixcorr` = 0.2289
- `eval_ssim` = 0.0364
- `eval_swav` = 0.8949
- `num_candidates` = 4.0000

#### Selection Summary
- Best Decoder Eval: `eval_alex5=0.9824`, `eval_clip=0.9973`, `eval_inception=0.9995`, `eval_pixcorr=0.2289`, `eval_ssim=0.0364`

#### Observations
- Ground-truth Kandinsky embedding establishes decoder upper bound on val64.
- Offline local decoder path is working end-to-end.

#### Next Action
None

### EXP-20260419-001500-reconstruction-baseline-topk4-val64 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-001500-reconstruction-baseline-topk4-val64","backfilled":false,"best_decoder_eval":{"eval_alex5":0.5607638888888888,"eval_clip":0.5270337301587301,"eval_inception":0.5171130952380952,"eval_pixcorr":0.16903136752488254,"eval_ssim":0.41072381474820563},"best_embedding_proxy":null,"command":null,"goal":"val64 compare baseline_topk4 on held-out val subset","key_inputs":[],"kind":"eval","metric_scope":"val","metric_source":"reconstruction_metrics.json","metrics":{"alpha":0.0,"eval_alex2":0.5915178571428571,"eval_alex5":0.5607638888888888,"eval_clip":0.5270337301587301,"eval_effnet":0.9704685211181641,"eval_inception":0.5171130952380952,"eval_pixcorr":0.16903136752488254,"eval_ssim":0.41072381474820563,"eval_swav":0.7194178104400635,"prototype_topk":4.0},"next_action":null,"observations":["Same val64 subset as Kandinsky comparison.","Residual-VAE baseline finished successfully."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/baseline_topk4","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"success","timestamp":"2026-04-19T00:10:45+08:00"} -->

- Timestamp: 2026-04-19T00:10:45+08:00
- Area: reconstruction
- Kind: eval
- Goal: val64 compare baseline_topk4 on held-out val subset
- Metric Scope: val
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/baseline_topk4
- Backfilled: no

#### Command
```bash
# none recorded
```

#### Key Inputs
- None

#### Metrics
- `alpha` = 0.0000
- `eval_alex2` = 0.5915
- `eval_alex5` = 0.5608
- `eval_clip` = 0.5270
- `eval_effnet` = 0.9705
- `eval_inception` = 0.5171
- `eval_pixcorr` = 0.1690
- `eval_ssim` = 0.4107
- `eval_swav` = 0.7194
- `prototype_topk` = 4.0000

#### Selection Summary
- Best Decoder Eval: `eval_alex5=0.5608`, `eval_clip=0.5270`, `eval_inception=0.5171`, `eval_pixcorr=0.1690`, `eval_ssim=0.4107`

#### Observations
- Same val64 subset as Kandinsky comparison.
- Residual-VAE baseline finished successfully.

#### Next Action
None

### EXP-20260419-000928-reconstruction [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-000928-reconstruction","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --embedding-bank outputs/cache/kandinsky_train.pt --data-dir ../image-eeg-data --output-dir outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local --split train --split-file outputs/subsets/val64_seed0.json --image-id-source val_ids --embedding-source ground_truth --prior-model kandinsky-community/kandinsky-2-2-prior --decoder-model /data/xiaoh/DeepLearning_storage/hf_models/kandinsky-2-2-decoder --num-candidates 4 --decoder-steps 40 --decoder-guidance-scale 4.0 --decoder-height 512 --decoder-width 512 --batch-size 4 --num-workers 2 --device cuda --evaluate --local-files-only","goal":"val64 compare kandinsky ground_truth with local decoder","key_inputs":[],"kind":"eval","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-19T00:09:28+08:00"} -->

- Timestamp: 2026-04-19T00:09:28+08:00
- Area: reconstruction
- Kind: eval
- Goal: val64 compare kandinsky ground_truth with local decoder
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/predict_reconstruction_embed.py --embedding-bank outputs/cache/kandinsky_train.pt --data-dir ../image-eeg-data --output-dir outputs/eval_compare/val64_seed0/kandinsky_groundtruth_local --split train --split-file outputs/subsets/val64_seed0.json --image-id-source val_ids --embedding-source ground_truth --prior-model kandinsky-community/kandinsky-2-2-prior --decoder-model /data/xiaoh/DeepLearning_storage/hf_models/kandinsky-2-2-decoder --num-candidates 4 --decoder-steps 40 --decoder-guidance-scale 4.0 --decoder-height 512 --decoder-width 512 --batch-size 4 --num-workers 2 --device cuda --evaluate --local-files-only
```

#### Key Inputs
- None

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
None

### EXP-20260419-000928-reconstruction [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260419-000928-reconstruction","backfilled":false,"best_decoder_eval":null,"best_embedding_proxy":null,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/predict_reconstruction.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --reconstruction-checkpoint outputs/experiments/reconstruction_dreamsim_topk4/seed_0/best.pt --split train --split-file outputs/subsets/val64_seed0.json --image-id-source val_ids --output-dir outputs/eval_compare/val64_seed0/baseline_topk4 --evaluate --batch-size 16 --num-workers 2 --device cuda","goal":"val64 compare baseline_topk4 on held-out val subset","key_inputs":[],"kind":"eval","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/baseline_topk4","selected_decoder_eval":null,"selected_embedding_proxy":null,"selected_epoch":null,"selection_metric":null,"status":"started","timestamp":"2026-04-19T00:09:28+08:00"} -->

- Timestamp: 2026-04-19T00:09:28+08:00
- Area: reconstruction
- Kind: eval
- Goal: val64 compare baseline_topk4 on held-out val subset
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/eval_compare/val64_seed0/baseline_topk4
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/predict_reconstruction.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --reconstruction-checkpoint outputs/experiments/reconstruction_dreamsim_topk4/seed_0/best.pt --split train --split-file outputs/subsets/val64_seed0.json --image-id-source val_ids --output-dir outputs/eval_compare/val64_seed0/baseline_topk4 --evaluate --batch-size 16 --num-workers 2 --device cuda
```

#### Key Inputs
- None

#### Metrics
- None

#### Selection Summary
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-223100-reconstruction-kandinsky-embed-train-v4-proxyselect [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-223100-reconstruction-kandinsky-embed-train-v4-proxyselect","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect --epochs 30 --batch-size 64 --num-workers 4 --learning-rate 1e-4 --weight-decay 1e-2 --dropout 0.1 --mse-weight 1.0 --cosine-weight 0.5 --contrastive-weight 0.1 --temperature 0.07 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-limit 0 --device cuda","goal":"Run the baseline Kandinsky embedding regressor with checkpoint selection driven by validation proxy retrieval instead of weighted loss.","key_inputs":["outputs/cache/kandinsky_train.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","scripts/train_reconstruction_embed.py"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"avg_target_cosine":0.6329413056373596,"epoch":7.0,"full_train_top1_acc":0.0024183797650039196,"full_train_top5_acc":0.008464328944683075,"lr":8.715724127386972e-05,"train_contrastive_loss":3.0909201587218584,"train_cosine_loss":0.34463756892814146,"train_mse_loss":0.8366229096707356,"train_total_loss":1.318033723360479,"val_contrastive_loss":3.440598386984605,"val_cosine_loss":0.3670063935793363,"val_mse_loss":0.8807002856181219,"val_subset_top1_acc":0.01874244213104248,"val_subset_top5_acc":0.07073760777711868,"val_total_loss":1.4082633394461412},"next_action":"Use the v4 proxy-selected checkpoint as the new mainline and decide whether to download the Kandinsky decoder for a generation smoke test.","observations":["Adding per-epoch validation proxy retrieval and selecting checkpoints by val_subset_top1 then top5 fixed the mismatch between weighted loss and the actual alignment objective.","The best checkpoint came from epoch 19 with val_subset_top1 about 0.0272 and val_subset_top5 about 0.0846, both stronger than every previous Kandinsky embedding run.","This run also improved full-train proxy retrieval to top1 about 0.0024 and top5 about 0.0115."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect/seed_0","status":"success","timestamp":"2026-04-18T22:40:17+08:00"} -->

- Timestamp: 2026-04-18T22:40:17+08:00
- Area: reconstruction
- Kind: train
- Goal: Run the baseline Kandinsky embedding regressor with checkpoint selection driven by validation proxy retrieval instead of weighted loss.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect/seed_0
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect --epochs 30 --batch-size 64 --num-workers 4 --learning-rate 1e-4 --weight-decay 1e-2 --dropout 0.1 --mse-weight 1.0 --cosine-weight 0.5 --contrastive-weight 0.1 --temperature 0.07 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- scripts/train_reconstruction_embed.py

#### Metrics
- `avg_target_cosine` = 0.6329
- `epoch` = 7.0000
- `full_train_top1_acc` = 0.0024
- `full_train_top5_acc` = 0.0085
- `lr` = 0.0001
- `train_contrastive_loss` = 3.0909
- `train_cosine_loss` = 0.3446
- `train_mse_loss` = 0.8366
- `train_total_loss` = 1.3180
- `val_contrastive_loss` = 3.4406
- `val_cosine_loss` = 0.3670
- `val_mse_loss` = 0.8807
- `val_subset_top1_acc` = 0.0187
- `val_subset_top5_acc` = 0.0707
- `val_total_loss` = 1.4083

#### Observations
- Adding per-epoch validation proxy retrieval and selecting checkpoints by val_subset_top1 then top5 fixed the mismatch between weighted loss and the actual alignment objective.
- The best checkpoint came from epoch 19 with val_subset_top1 about 0.0272 and val_subset_top5 about 0.0846, both stronger than every previous Kandinsky embedding run.
- This run also improved full-train proxy retrieval to top1 about 0.0024 and top5 about 0.0115.

#### Next Action
Use the v4 proxy-selected checkpoint as the new mainline and decide whether to download the Kandinsky decoder for a generation smoke test.

### EXP-20260418-222500-reconstruction-kandinsky-embed-train-v3-smallhead [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-222500-reconstruction-kandinsky-embed-train-v3-smallhead","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=5 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --head-hidden-dim 512 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda","goal":"Run a short regularized Kandinsky embedding ablation with a smaller head for stronger capacity control.","key_inputs":["outputs/cache/kandinsky_train.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt"],"kind":"ablation","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":8.0,"lr":0.0,"train_contrastive_loss":3.607567821812426,"train_cosine_loss":0.38418596473514527,"train_mse_loss":0.9147532587377434,"train_total_loss":1.5630761711006491,"val_contrastive_loss":3.9769979623647838,"val_cosine_loss":0.38781384321359486,"val_mse_loss":0.9204636491261996,"val_total_loss":1.6434452808820283},"next_action":"Drop the small-head path and continue from the stronger proxy-aware training direction.","observations":["The smaller-head regularization variant trained cleanly but underperformed the v2 metric-regularized run on every proxy retrieval metric.","Validation-subset proxy retrieval reached about top1 0.0127 and top5 0.0586, so the extra capacity reduction was too aggressive for this stage."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead/seed_0","status":"success","timestamp":"2026-04-18T22:32:08+08:00"} -->

- Timestamp: 2026-04-18T22:32:08+08:00
- Area: reconstruction
- Kind: ablation
- Goal: Run a short regularized Kandinsky embedding ablation with a smaller head for stronger capacity control.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead/seed_0
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --head-hidden-dim 512 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt

#### Metrics
- `epoch` = 8.0000
- `lr` = 0.0000
- `train_contrastive_loss` = 3.6076
- `train_cosine_loss` = 0.3842
- `train_mse_loss` = 0.9148
- `train_total_loss` = 1.5631
- `val_contrastive_loss` = 3.9770
- `val_cosine_loss` = 0.3878
- `val_mse_loss` = 0.9205
- `val_total_loss` = 1.6434

#### Observations
- The smaller-head regularization variant trained cleanly but underperformed the v2 metric-regularized run on every proxy retrieval metric.
- Validation-subset proxy retrieval reached about top1 0.0127 and top5 0.0586, so the extra capacity reduction was too aggressive for this stage.

#### Next Action
Drop the small-head path and continue from the stronger proxy-aware training direction.

### EXP-20260418-222500-reconstruction-kandinsky-embed-train-v2-metricreg [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-222500-reconstruction-kandinsky-embed-train-v2-metricreg","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda","goal":"Run a short regularized Kandinsky embedding ablation with stronger metric alignment losses.","key_inputs":["outputs/cache/kandinsky_train.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt"],"kind":"ablation","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":6.0,"lr":1.4644660940672627e-05,"train_contrastive_loss":3.4139482302543445,"train_cosine_loss":0.38033163904124856,"train_mse_loss":0.9065581616173443,"train_total_loss":1.5164003810311995,"val_contrastive_loss":3.8310767320486216,"val_cosine_loss":0.39055453813993013,"val_mse_loss":0.9241246168430035,"val_total_loss":1.6188322030580962},"next_action":"Pivot from hyperparameter-only tuning to proxy-driven checkpoint selection during training.","observations":["Compared with the original best.pt, this metric-regularized run improved validation-subset proxy retrieval to top1 about 0.0230 and top5 about 0.0756.","Its last checkpoint slightly improved top5 further to about 0.0774, showing that weighted loss and proxy retrieval are not perfectly aligned.","Despite the proxy gains over the original best.pt, it still did not beat the original run's last checkpoint on validation-subset top1/top5."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg/seed_0","status":"success","timestamp":"2026-04-18T22:32:06+08:00"} -->

- Timestamp: 2026-04-18T22:32:06+08:00
- Area: reconstruction
- Kind: ablation
- Goal: Run a short regularized Kandinsky embedding ablation with stronger metric alignment losses.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg/seed_0
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt

#### Metrics
- `epoch` = 6.0000
- `lr` = 0.0000
- `train_contrastive_loss` = 3.4139
- `train_cosine_loss` = 0.3803
- `train_mse_loss` = 0.9066
- `train_total_loss` = 1.5164
- `val_contrastive_loss` = 3.8311
- `val_cosine_loss` = 0.3906
- `val_mse_loss` = 0.9241
- `val_total_loss` = 1.6188

#### Observations
- Compared with the original best.pt, this metric-regularized run improved validation-subset proxy retrieval to top1 about 0.0230 and top5 about 0.0756.
- Its last checkpoint slightly improved top5 further to about 0.0774, showing that weighted loss and proxy retrieval are not perfectly aligned.
- Despite the proxy gains over the original best.pt, it still did not beat the original run's last checkpoint on validation-subset top1/top5.

#### Next Action
Pivot from hyperparameter-only tuning to proxy-driven checkpoint selection during training.

### EXP-20260418-223100-reconstruction-kandinsky-embed-train-v4-proxyselect [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-223100-reconstruction-kandinsky-embed-train-v4-proxyselect","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect --epochs 30 --batch-size 64 --num-workers 4 --learning-rate 1e-4 --weight-decay 1e-2 --dropout 0.1 --mse-weight 1.0 --cosine-weight 0.5 --contrastive-weight 0.1 --temperature 0.07 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-limit 0 --device cuda","goal":"Run the baseline Kandinsky embedding regressor with checkpoint selection driven by validation proxy retrieval instead of weighted loss.","key_inputs":["outputs/cache/kandinsky_train.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","scripts/train_reconstruction_embed.py"],"kind":"train","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect","status":"started","timestamp":"2026-04-18T22:31:33+08:00"} -->

- Timestamp: 2026-04-18T22:31:33+08:00
- Area: reconstruction
- Kind: train
- Goal: Run the baseline Kandinsky embedding regressor with checkpoint selection driven by validation proxy retrieval instead of weighted loss.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v4_proxyselect --epochs 30 --batch-size 64 --num-workers 4 --learning-rate 1e-4 --weight-decay 1e-2 --dropout 0.1 --mse-weight 1.0 --cosine-weight 0.5 --contrastive-weight 0.1 --temperature 0.07 --embedding-eval-every 1 --selection-metric val_subset_top1_then_top5 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- scripts/train_reconstruction_embed.py

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-222500-reconstruction-kandinsky-embed-train-v2-metricreg [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-222500-reconstruction-kandinsky-embed-train-v2-metricreg","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda","goal":"Run a short regularized Kandinsky embedding ablation with stronger metric alignment losses.","key_inputs":["outputs/cache/kandinsky_train.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt"],"kind":"ablation","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg","status":"started","timestamp":"2026-04-18T22:24:48+08:00"} -->

- Timestamp: 2026-04-18T22:24:48+08:00
- Area: reconstruction
- Kind: ablation
- Goal: Run a short regularized Kandinsky embedding ablation with stronger metric alignment losses.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v2_metricreg --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-222500-reconstruction-kandinsky-embed-train-v3-smallhead [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-222500-reconstruction-kandinsky-embed-train-v3-smallhead","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=5 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --head-hidden-dim 512 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda","goal":"Run a short regularized Kandinsky embedding ablation with a smaller head for stronger capacity control.","key_inputs":["outputs/cache/kandinsky_train.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt"],"kind":"ablation","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead","status":"started","timestamp":"2026-04-18T22:24:48+08:00"} -->

- Timestamp: 2026-04-18T22:24:48+08:00
- Area: reconstruction
- Kind: ablation
- Goal: Run a short regularized Kandinsky embedding ablation with a smaller head for stronger capacity control.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v3_smallhead --epochs 8 --batch-size 128 --num-workers 4 --learning-rate 1e-4 --weight-decay 5e-2 --head-hidden-dim 512 --dropout 0.2 --mse-weight 0.5 --cosine-weight 1.0 --contrastive-weight 0.2 --temperature 0.05 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-220400-reconstruction-kandinsky-embed-val-retrieval-proxy [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-220400-reconstruction-kandinsky-embed-val-retrieval-proxy","backfilled":false,"command":"python ad-hoc eval: load best.pt, predict val embeddings, score against kandinsky_train bank and val subset","goal":"Measure how well the trained EEG-to-Kandinsky embedding regressor retrieves the correct validation image in embedding space.","key_inputs":["outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/best.pt","outputs/cache/kandinsky_train.pt","outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/split.json"],"kind":"eval","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Try stronger regularization or shorter schedules, then prepare a decoder-backed image-generation smoke test from the epoch-5 checkpoint.","observations":["On the held-out validation queries, the regressor reached avg_target_cosine about 0.6357 against the matched Kandinsky image embedding.","Retrieval against the full train bank was weak but above random: top1 about 0.0018 and top5 about 0.0085.","Within the smaller validation subset candidate pool, top1 was about 0.0151 and top5 about 0.0556, indicating the embedding head learned some alignment but remains far from production quality."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/embedding_eval","status":"success","timestamp":"2026-04-18T22:05:57+08:00"} -->

- Timestamp: 2026-04-18T22:05:57+08:00
- Area: reconstruction
- Kind: eval
- Goal: Measure how well the trained EEG-to-Kandinsky embedding regressor retrieves the correct validation image in embedding space.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/embedding_eval
- Backfilled: no

#### Command
```bash
python ad-hoc eval: load best.pt, predict val embeddings, score against kandinsky_train bank and val subset
```

#### Key Inputs
- outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/best.pt
- outputs/cache/kandinsky_train.pt
- outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/split.json

#### Metrics
- None

#### Observations
- On the held-out validation queries, the regressor reached avg_target_cosine about 0.6357 against the matched Kandinsky image embedding.
- Retrieval against the full train bank was weak but above random: top1 about 0.0018 and top5 about 0.0085.
- Within the smaller validation subset candidate pool, top1 was about 0.0151 and top5 about 0.0556, indicating the embedding head learned some alignment but remains far from production quality.

#### Next Action
Try stronger regularization or shorter schedules, then prepare a decoder-backed image-generation smoke test from the epoch-5 checkpoint.

### EXP-20260418-220400-reconstruction-kandinsky-embed-val-retrieval-proxy [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-220400-reconstruction-kandinsky-embed-val-retrieval-proxy","backfilled":false,"command":"python ad-hoc eval: load best.pt, predict val embeddings, score against kandinsky_train bank and val subset","goal":"Measure how well the trained EEG-to-Kandinsky embedding regressor retrieves the correct validation image in embedding space.","key_inputs":["outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/best.pt","outputs/cache/kandinsky_train.pt","outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/split.json"],"kind":"eval","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/embedding_eval","status":"started","timestamp":"2026-04-18T22:04:00+08:00"} -->

- Timestamp: 2026-04-18T22:04:00+08:00
- Area: reconstruction
- Kind: eval
- Goal: Measure how well the trained EEG-to-Kandinsky embedding regressor retrieves the correct validation image in embedding space.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/embedding_eval
- Backfilled: no

#### Command
```bash
python ad-hoc eval: load best.pt, predict val embeddings, score against kandinsky_train bank and val subset
```

#### Key Inputs
- outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/best.pt
- outputs/cache/kandinsky_train.pt
- outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0/split.json

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-215800-reconstruction-kandinsky-embed-train-v1 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-215800-reconstruction-kandinsky-embed-train-v1","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v1 --epochs 20 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda","goal":"Run the first formal EEG-to-Kandinsky image-embedding regression experiment.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/kandinsky_train.pt"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":5.0,"lr":8.535533905932738e-05,"train_contrastive_loss":3.2446967614055193,"train_cosine_loss":0.35047918839516046,"train_mse_loss":0.848442128799504,"train_total_loss":1.348151411109728,"val_contrastive_loss":3.486764852817242,"val_cosine_loss":0.36426920615709746,"val_mse_loss":0.8755357632270226,"val_total_loss":1.4063468621327326},"next_action":"Use the best checkpoint for a small image-generation smoke test, or add stronger regularization / shorter schedules before the next training run.","observations":["The first formal EEG-to-Kandinsky embedding regression run completed 20 epochs without runtime issues.","Validation loss improved early and the best checkpoint came from epoch 5 with val_total_loss around 1.4063.","Later epochs continued reducing training loss but validation loss drifted upward, indicating early overfitting."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0","status":"success","timestamp":"2026-04-18T22:03:09+08:00"} -->

- Timestamp: 2026-04-18T22:03:09+08:00
- Area: reconstruction
- Kind: train
- Goal: Run the first formal EEG-to-Kandinsky image-embedding regression experiment.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1/seed_0
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v1 --epochs 20 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/kandinsky_train.pt

#### Metrics
- `epoch` = 5.0000
- `lr` = 0.0001
- `train_contrastive_loss` = 3.2447
- `train_cosine_loss` = 0.3505
- `train_mse_loss` = 0.8484
- `train_total_loss` = 1.3482
- `val_contrastive_loss` = 3.4868
- `val_cosine_loss` = 0.3643
- `val_mse_loss` = 0.8755
- `val_total_loss` = 1.4063

#### Observations
- The first formal EEG-to-Kandinsky embedding regression run completed 20 epochs without runtime issues.
- Validation loss improved early and the best checkpoint came from epoch 5 with val_total_loss around 1.4063.
- Later epochs continued reducing training loss but validation loss drifted upward, indicating early overfitting.

#### Next Action
Use the best checkpoint for a small image-generation smoke test, or add stronger regularization / shorter schedules before the next training run.

### EXP-20260418-215800-reconstruction-kandinsky-embed-train-v1 [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-215800-reconstruction-kandinsky-embed-train-v1","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v1 --epochs 20 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda","goal":"Run the first formal EEG-to-Kandinsky image-embedding regression experiment.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/kandinsky_train.pt"],"kind":"train","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1","status":"started","timestamp":"2026-04-18T21:57:39+08:00"} -->

- Timestamp: 2026-04-18T21:57:39+08:00
- Area: reconstruction
- Kind: train
- Goal: Run the first formal EEG-to-Kandinsky image-embedding regression experiment.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_kandinsky_embed_v1
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/kandinsky_train.pt --output-dir outputs/experiments/reconstruction_kandinsky_embed_v1 --epochs 20 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/kandinsky_train.pt

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-215400-reconstruction-kandinsky-train-bank-build [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-215400-reconstruction-kandinsky-train-bank-build","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Build the Kandinsky train image-embedding bank now that the image encoder weights are locally cached.","key_inputs":["outputs/cache/kandinsky_train.pt","kandinsky-community/kandinsky-2-2-prior:image_encoder local-cache-ready"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Launch the first formal EEG-to-Kandinsky embedding training run with image-level evaluation disabled for speed.","observations":["The Kandinsky train bank was built successfully after the image encoder weights were made available locally.","The final cached bank was written to outputs/cache/kandinsky_train.pt using the lightweight image-encoder-only path."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"success","timestamp":"2026-04-18T21:57:22+08:00"} -->

- Timestamp: 2026-04-18T21:57:22+08:00
- Area: reconstruction
- Kind: cache
- Goal: Build the Kandinsky train image-embedding bank now that the image encoder weights are locally cached.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- kandinsky-community/kandinsky-2-2-prior:image_encoder local-cache-ready

#### Metrics
- None

#### Observations
- The Kandinsky train bank was built successfully after the image encoder weights were made available locally.
- The final cached bank was written to outputs/cache/kandinsky_train.pt using the lightweight image-encoder-only path.

#### Next Action
Launch the first formal EEG-to-Kandinsky embedding training run with image-level evaluation disabled for speed.

### EXP-20260418-215400-reconstruction-kandinsky-train-bank-build [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-215400-reconstruction-kandinsky-train-bank-build","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Build the Kandinsky train image-embedding bank now that the image encoder weights are locally cached.","key_inputs":["outputs/cache/kandinsky_train.pt","kandinsky-community/kandinsky-2-2-prior:image_encoder local-cache-ready"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"started","timestamp":"2026-04-18T21:53:58+08:00"} -->

- Timestamp: 2026-04-18T21:53:58+08:00
- Area: reconstruction
- Kind: cache
- Goal: Build the Kandinsky train image-embedding bank now that the image encoder weights are locally cached.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- outputs/cache/kandinsky_train.pt
- kandinsky-community/kandinsky-2-2-prior:image_encoder local-cache-ready

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-214250-reconstruction-kandinsky-image-encoder-aria2 [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-214250-reconstruction-kandinsky-image-encoder-aria2","backfilled":false,"command":"aria2c -x 16 -s 16 -k 1M -c --allow-overwrite=true --auto-file-renaming=false -d <hf-cache-blobs-dir> -o 657723...incomplete https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors","goal":"Download the missing Kandinsky image_encoder weight blob efficiently with aria2 so bank construction can proceed.","key_inputs":["/data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior/blobs/657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d.incomplete","https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Re-run scripts/cache_image_bank.py to build kandinsky_train.pt now that the image encoder weights are locally available.","observations":["Resumed the partially downloaded image encoder blob with aria2 at roughly 5-6 MiB/s and completed the 3.69GB file.","Promoted the finished blob into the local HF cache snapshot and verified local_files_only loading for CLIPVisionModelWithProjection."],"output_dir":"/data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior","status":"success","timestamp":"2026-04-18T21:53:32+08:00"} -->

- Timestamp: 2026-04-18T21:53:32+08:00
- Area: reconstruction
- Kind: cache
- Goal: Download the missing Kandinsky image_encoder weight blob efficiently with aria2 so bank construction can proceed.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior
- Backfilled: no

#### Command
```bash
aria2c -x 16 -s 16 -k 1M -c --allow-overwrite=true --auto-file-renaming=false -d <hf-cache-blobs-dir> -o 657723...incomplete https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors
```

#### Key Inputs
- /data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior/blobs/657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d.incomplete
- https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors

#### Metrics
- None

#### Observations
- Resumed the partially downloaded image encoder blob with aria2 at roughly 5-6 MiB/s and completed the 3.69GB file.
- Promoted the finished blob into the local HF cache snapshot and verified local_files_only loading for CLIPVisionModelWithProjection.

#### Next Action
Re-run scripts/cache_image_bank.py to build kandinsky_train.pt now that the image encoder weights are locally available.

### EXP-20260418-214250-reconstruction-kandinsky-image-encoder-aria2 [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-214250-reconstruction-kandinsky-image-encoder-aria2","backfilled":false,"command":"aria2c -x 16 -s 16 -k 1M -c --allow-overwrite=true --auto-file-renaming=false -d <hf-cache-blobs-dir> -o 657723...incomplete https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors","goal":"Download the missing Kandinsky image_encoder weight blob efficiently with aria2 so bank construction can proceed.","key_inputs":["/data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior/blobs/657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d.incomplete","https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior","status":"started","timestamp":"2026-04-18T21:42:43+08:00"} -->

- Timestamp: 2026-04-18T21:42:43+08:00
- Area: reconstruction
- Kind: cache
- Goal: Download the missing Kandinsky image_encoder weight blob efficiently with aria2 so bank construction can proceed.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior
- Backfilled: no

#### Command
```bash
aria2c -x 16 -s 16 -k 1M -c --allow-overwrite=true --auto-file-renaming=false -d <hf-cache-blobs-dir> -o 657723...incomplete https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors
```

#### Key Inputs
- /data/xiaoh/DeepLearning_storage/.cache/huggingface/hub/models--kandinsky-community--kandinsky-2-2-prior/blobs/657723e09f46a7c3957df651601029f66b1748afb12b419816330f16ed45d64d.incomplete
- https://huggingface.co/kandinsky-community/kandinsky-2-2-prior/resolve/main/image_encoder/model.safetensors

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-214000-reconstruction-kandinsky-train-cache-http [aborted]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-214000-reconstruction-kandinsky-train-cache-http","backfilled":false,"command":"HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Cache the Kandinsky train image-embedding bank with Xet disabled so the image encoder downloads via standard HTTP/LFS.","key_inputs":["kandinsky-community/kandinsky-2-2-prior:image_encoder","image-eeg-data/training_images"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Resume the image encoder blob with aria2, wire it into the local HF cache snapshot, then continue bank construction.","observations":["Disabling Xet unblocked the stalled download, but the standard HTTP path only reached roughly 0.1-0.3 MB/s in steady state.","A direct aria2 resume test on the same blob reached roughly 5-6 MiB/s, making the hub client path no longer time-efficient."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"aborted","timestamp":"2026-04-18T21:42:25+08:00"} -->

- Timestamp: 2026-04-18T21:42:25+08:00
- Area: reconstruction
- Kind: cache
- Goal: Cache the Kandinsky train image-embedding bank with Xet disabled so the image encoder downloads via standard HTTP/LFS.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- kandinsky-community/kandinsky-2-2-prior:image_encoder
- image-eeg-data/training_images

#### Metrics
- None

#### Observations
- Disabling Xet unblocked the stalled download, but the standard HTTP path only reached roughly 0.1-0.3 MB/s in steady state.
- A direct aria2 resume test on the same blob reached roughly 5-6 MiB/s, making the hub client path no longer time-efficient.

#### Next Action
Resume the image encoder blob with aria2, wire it into the local HF cache snapshot, then continue bank construction.

### EXP-20260418-214000-reconstruction-kandinsky-train-cache-http [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-214000-reconstruction-kandinsky-train-cache-http","backfilled":false,"command":"HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Cache the Kandinsky train image-embedding bank with Xet disabled so the image encoder downloads via standard HTTP/LFS.","key_inputs":["kandinsky-community/kandinsky-2-2-prior:image_encoder","image-eeg-data/training_images"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"started","timestamp":"2026-04-18T21:31:26+08:00"} -->

- Timestamp: 2026-04-18T21:31:26+08:00
- Area: reconstruction
- Kind: cache
- Goal: Cache the Kandinsky train image-embedding bank with Xet disabled so the image encoder downloads via standard HTTP/LFS.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- kandinsky-community/kandinsky-2-2-prior:image_encoder
- image-eeg-data/training_images

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-213400-reconstruction-kandinsky-train-cache-lite [failed]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-213400-reconstruction-kandinsky-train-cache-lite","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Cache the Kandinsky train image-embedding bank using the lightweight image-encoder-only implementation.","key_inputs":["image-eeg-data/training_images","kandinsky-community/kandinsky-2-2-prior:image_encoder","kandinsky-community/kandinsky-2-2-prior:image_processor"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Re-run the cache with HF_HUB_DISABLE_XET=1 and longer HF Hub timeouts to force standard HTTP/LFS download.","observations":["The lighter image-encoder-only implementation correctly avoided the full prior snapshot, but the actual weight download stalled at 0 bytes.","Traceback showed the request stuck inside huggingface_hub.file_download.xet_get while fetching image_encoder/model.safetensors."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"failed","timestamp":"2026-04-18T21:31:07+08:00"} -->

- Timestamp: 2026-04-18T21:31:07+08:00
- Area: reconstruction
- Kind: cache
- Goal: Cache the Kandinsky train image-embedding bank using the lightweight image-encoder-only implementation.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- image-eeg-data/training_images
- kandinsky-community/kandinsky-2-2-prior:image_encoder
- kandinsky-community/kandinsky-2-2-prior:image_processor

#### Metrics
- None

#### Observations
- The lighter image-encoder-only implementation correctly avoided the full prior snapshot, but the actual weight download stalled at 0 bytes.
- Traceback showed the request stuck inside huggingface_hub.file_download.xet_get while fetching image_encoder/model.safetensors.

#### Next Action
Re-run the cache with HF_HUB_DISABLE_XET=1 and longer HF Hub timeouts to force standard HTTP/LFS download.

### EXP-20260418-213400-reconstruction-kandinsky-train-cache-lite [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-213400-reconstruction-kandinsky-train-cache-lite","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Cache the Kandinsky train image-embedding bank using the lightweight image-encoder-only implementation.","key_inputs":["image-eeg-data/training_images","kandinsky-community/kandinsky-2-2-prior:image_encoder","kandinsky-community/kandinsky-2-2-prior:image_processor"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"started","timestamp":"2026-04-18T21:28:05+08:00"} -->

- Timestamp: 2026-04-18T21:28:05+08:00
- Area: reconstruction
- Kind: cache
- Goal: Cache the Kandinsky train image-embedding bank using the lightweight image-encoder-only implementation.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- image-eeg-data/training_images
- kandinsky-community/kandinsky-2-2-prior:image_encoder
- kandinsky-community/kandinsky-2-2-prior:image_processor

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-212300-reconstruction-kandinsky-train-cache [aborted]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-212300-reconstruction-kandinsky-train-cache","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Cache the Kandinsky train image-embedding bank needed for EEG-to-embedding reconstruction training.","key_inputs":["image-eeg-data/training_images","kandinsky-community/kandinsky-2-2-prior"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Re-run the train bank cache with the lighter image-encoder-only implementation.","observations":["The first cache attempt used the full Kandinsky prior pipeline and began downloading roughly 7.8GB of weights.","Interrupted the attempt after confirming from local diffusers source that only the image encoder and image processor were actually needed for bank construction."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"aborted","timestamp":"2026-04-18T21:27:45+08:00"} -->

- Timestamp: 2026-04-18T21:27:45+08:00
- Area: reconstruction
- Kind: cache
- Goal: Cache the Kandinsky train image-embedding bank needed for EEG-to-embedding reconstruction training.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- image-eeg-data/training_images
- kandinsky-community/kandinsky-2-2-prior

#### Metrics
- None

#### Observations
- The first cache attempt used the full Kandinsky prior pipeline and began downloading roughly 7.8GB of weights.
- Interrupted the attempt after confirming from local diffusers source that only the image encoder and image processor were actually needed for bank construction.

#### Next Action
Re-run the train bank cache with the lighter image-encoder-only implementation.

### EXP-20260418-212600-debug-train-reconstruction-embed-loop-smoke [success]
<!-- log-meta: {"area":"debug","attempt_id":"EXP-20260418-212600-debug-train-reconstruction-embed-loop-smoke","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/clip_train.pt --output-dir outputs/experiments/reconstruction_embed_smoke_clip_proxy --epochs 1 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda","goal":"Validate the new reconstruction embedding training loop with a proxy embedding bank while Kandinsky downloads.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/clip_train.pt"],"kind":"smoke","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":1.0,"lr":0.0,"train_contrastive_loss":3.609386004091844,"train_cosine_loss":0.3363700214885335,"train_mse_loss":0.0048065480322461145,"train_total_loss":0.533930165829065,"val_contrastive_loss":3.5048540830612183,"val_cosine_loss":0.2924326933347262,"val_mse_loss":0.003511829015154105,"val_total_loss":0.500213582928364},"next_action":"Wait for the Kandinsky train bank to finish downloading/building, then launch the first real EEG-to-Kandinsky embedding run.","observations":["The embedding-regression training loop ran for one epoch on a proxy CLIP bank without runtime errors.","Checkpoint writing, split/config serialization, and history.json updates all succeeded."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_embed_smoke_clip_proxy/seed_0","status":"success","timestamp":"2026-04-18T21:24:16+08:00"} -->

- Timestamp: 2026-04-18T21:24:16+08:00
- Area: debug
- Kind: smoke
- Goal: Validate the new reconstruction embedding training loop with a proxy embedding bank while Kandinsky downloads.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_embed_smoke_clip_proxy/seed_0
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/clip_train.pt --output-dir outputs/experiments/reconstruction_embed_smoke_clip_proxy --epochs 1 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/clip_train.pt

#### Metrics
- `epoch` = 1.0000
- `lr` = 0.0000
- `train_contrastive_loss` = 3.6094
- `train_cosine_loss` = 0.3364
- `train_mse_loss` = 0.0048
- `train_total_loss` = 0.5339
- `val_contrastive_loss` = 3.5049
- `val_cosine_loss` = 0.2924
- `val_mse_loss` = 0.0035
- `val_total_loss` = 0.5002

#### Observations
- The embedding-regression training loop ran for one epoch on a proxy CLIP bank without runtime errors.
- Checkpoint writing, split/config serialization, and history.json updates all succeeded.

#### Next Action
Wait for the Kandinsky train bank to finish downloading/building, then launch the first real EEG-to-Kandinsky embedding run.

### EXP-20260418-212600-debug-train-reconstruction-embed-loop-smoke [started]
<!-- log-meta: {"area":"debug","attempt_id":"EXP-20260418-212600-debug-train-reconstruction-embed-loop-smoke","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/clip_train.pt --output-dir outputs/experiments/reconstruction_embed_smoke_clip_proxy --epochs 1 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda","goal":"Validate the new reconstruction embedding training loop with a proxy embedding bank while Kandinsky downloads.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/clip_train.pt"],"kind":"smoke","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_embed_smoke_clip_proxy","status":"started","timestamp":"2026-04-18T21:23:18+08:00"} -->

- Timestamp: 2026-04-18T21:23:18+08:00
- Area: debug
- Kind: smoke
- Goal: Validate the new reconstruction embedding training loop with a proxy embedding bank while Kandinsky downloads.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_embed_smoke_clip_proxy
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_reconstruction_embed.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --embedding-bank outputs/cache/clip_train.pt --output-dir outputs/experiments/reconstruction_embed_smoke_clip_proxy --epochs 1 --batch-size 64 --num-workers 4 --image-eval-limit 0 --device cuda
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/clip_train.pt

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-212300-reconstruction-kandinsky-train-cache [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-212300-reconstruction-kandinsky-train-cache","backfilled":false,"command":"CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt","goal":"Cache the Kandinsky train image-embedding bank needed for EEG-to-embedding reconstruction training.","key_inputs":["image-eeg-data/training_images","kandinsky-community/kandinsky-2-2-prior"],"kind":"cache","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache","status":"started","timestamp":"2026-04-18T21:20:12+08:00"} -->

- Timestamp: 2026-04-18T21:20:12+08:00
- Area: reconstruction
- Kind: cache
- Goal: Cache the Kandinsky train image-embedding bank needed for EEG-to-embedding reconstruction training.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/cache
- Backfilled: no

#### Command
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/cache_image_bank.py --bank-type kandinsky --split train --batch-size 8 --device cuda --output outputs/cache/kandinsky_train.pt
```

#### Key Inputs
- image-eeg-data/training_images
- kandinsky-community/kandinsky-2-2-prior

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260418-212100-debug-kandinsky-runtime-smoke [success]
<!-- log-meta: {"area":"debug","attempt_id":"EXP-20260418-212100-debug-kandinsky-runtime-smoke","backfilled":false,"command":"python -m py_compile ... && python scripts/train_reconstruction_embed.py --help && python scripts/predict_reconstruction_embed.py --help && runtime import/load smoke","goal":"Validate the new Kandinsky reconstruction code path before launching longer experiments.","key_inputs":["src/project1_eeg/kandinsky.py","scripts/train_reconstruction_embed.py","scripts/predict_reconstruction_embed.py","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt"],"kind":"smoke","metric_scope":"smoke","metric_source":null,"metrics":{},"next_action":"Cache the Kandinsky train/test image banks on a free GPU, then run a short training smoke test.","observations":["Py-compile and CLI help both succeeded for all new scripts.","The current project env already has torch 2.6.0+cu124 plus diffusers/transformers/accelerate.","Retrieval checkpoint loading works and the Kandinsky diffusers pipeline classes are available."],"output_dir":null,"status":"success","timestamp":"2026-04-18T21:19:34+08:00"} -->

- Timestamp: 2026-04-18T21:19:34+08:00
- Area: debug
- Kind: smoke
- Goal: Validate the new Kandinsky reconstruction code path before launching longer experiments.
- Metric Scope: smoke
- Metric Source: None
- Output Dir: None
- Backfilled: no

#### Command
```bash
python -m py_compile ... && python scripts/train_reconstruction_embed.py --help && python scripts/predict_reconstruction_embed.py --help && runtime import/load smoke
```

#### Key Inputs
- src/project1_eeg/kandinsky.py
- scripts/train_reconstruction_embed.py
- scripts/predict_reconstruction_embed.py
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt

#### Metrics
- None

#### Observations
- Py-compile and CLI help both succeeded for all new scripts.
- The current project env already has torch 2.6.0+cu124 plus diffusers/transformers/accelerate.
- Retrieval checkpoint loading works and the Kandinsky diffusers pipeline classes are available.

#### Next Action
Cache the Kandinsky train/test image banks on a free GPU, then run a short training smoke test.

### EXP-20260418-210000-infra-log-system [success]
<!-- log-meta: {"area":"infra","attempt_id":"EXP-20260418-210000-infra-log-system","backfilled":false,"command":"python scripts/log_experiment.py ...","goal":"Implement the project-wide experiment ledger and backfill the existing runs.","key_inputs":["EXPERIMENT_LOG.md","scripts/log_experiment.py","README.md"],"kind":"debug","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Use the logger before and after every new training, evaluation, or debugging attempt.","observations":["Added `scripts/log_experiment.py`, created `EXPERIMENT_LOG.md`, documented usage in `README.md`, and backfilled the main historical runs."],"output_dir":null,"status":"success","timestamp":"2026-04-18T21:05:30+08:00"} -->

- Timestamp: 2026-04-18T21:05:30+08:00
- Area: infra
- Kind: debug
- Goal: Implement the project-wide experiment ledger and backfill the existing runs.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: None
- Backfilled: no

#### Command
```bash
python scripts/log_experiment.py ...
```

#### Key Inputs
- EXPERIMENT_LOG.md
- scripts/log_experiment.py
- README.md

#### Metrics
- None

#### Observations
- Added `scripts/log_experiment.py`, created `EXPERIMENT_LOG.md`, documented usage in `README.md`, and backfilled the main historical runs.

#### Next Action
Use the logger before and after every new training, evaluation, or debugging attempt.

### EXP-20260418-204900-debug-reconstruction-diagnosis [success]
<!-- log-meta: {"area":"debug","attempt_id":"EXP-20260418-204900-debug-reconstruction-diagnosis","backfilled":true,"command":"manual analysis from reconstruction metadata, sampled outputs, and train/test concept directories","goal":"Diagnose why the prototype-based reconstruction outputs remain visually meaningless on the test set.","key_inputs":["reconstruction_metadata.json","test_predictions/images","image-eeg-data/training_images","image-eeg-data/test_images"],"kind":"debug","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":"Redesign reconstruction so it conditions a generator on EEG-derived embeddings instead of relying on train-bank nearest neighbors.","observations":["Confirmed that train/test concept overlap is zero, so `test EEG -> training_images` prototype retrieval is inherently cross-category.","Confirmed that local retrieval test accuracy was measured on `test EEG -> test_images`, which is not the same task as reconstruction prototype lookup."],"output_dir":null,"status":"success","timestamp":"2026-04-18T21:04:58+08:00"} -->

- Timestamp: 2026-04-18T21:04:58+08:00
- Area: debug
- Kind: debug
- Goal: Diagnose why the prototype-based reconstruction outputs remain visually meaningless on the test set.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: None
- Backfilled: yes

#### Command
```bash
manual analysis from reconstruction metadata, sampled outputs, and train/test concept directories
```

#### Key Inputs
- reconstruction_metadata.json
- test_predictions/images
- image-eeg-data/training_images
- image-eeg-data/test_images

#### Metrics
- None

#### Observations
- Confirmed that train/test concept overlap is zero, so `test EEG -> training_images` prototype retrieval is inherently cross-category.
- Confirmed that local retrieval test accuracy was measured on `test EEG -> test_images`, which is not the same task as reconstruction prototype lookup.

#### Next Action
Redesign reconstruction so it conditions a generator on EEG-derived embeddings instead of relying on train-bank nearest neighbors.

### EXP-20260418-204523-reconstruction-topk4-test [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-204523-reconstruction-topk4-test","backfilled":true,"command":"python scripts/predict_reconstruction.py --reconstruction-checkpoint outputs/experiments/reconstruction_dreamsim_topk4/seed_0/best.pt --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --split test --evaluate --output-dir outputs/experiments/reconstruction_dreamsim_topk4/seed_0/test_predictions","goal":"Generate and evaluate test reconstructions for the score-weighted top-k=4 residual model.","key_inputs":["outputs/experiments/reconstruction_dreamsim_topk4/seed_0/best.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt"],"kind":"predict","metric_scope":"test","metric_source":"reconstruction_metrics.json","metrics":{"alpha":0.0,"eval_alex2":0.6434422110552763,"eval_alex5":0.5920603015075376,"eval_clip":0.5385929648241207,"eval_effnet":0.977794349193573,"eval_inception":0.5331658291457286,"eval_pixcorr":0.21359461859861129,"eval_ssim":0.43264139896007464,"eval_swav":0.7227188944816589,"prototype_topk":4.0},"next_action":"Treat the prototype-residual VAE path as a baseline only and pivot reconstruction to a non-prototype generator.","observations":["Backfilled from existing artifacts.","Even with the best quantitative metrics, the generated images collapsed to low-information textures and were not semantically recognizable."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0/test_predictions","status":"success","timestamp":"2026-04-18T20:31:04+08:00"} -->

- Timestamp: 2026-04-18T20:31:04+08:00
- Area: reconstruction
- Kind: predict
- Goal: Generate and evaluate test reconstructions for the score-weighted top-k=4 residual model.
- Metric Scope: test
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0/test_predictions
- Backfilled: yes

#### Command
```bash
python scripts/predict_reconstruction.py --reconstruction-checkpoint outputs/experiments/reconstruction_dreamsim_topk4/seed_0/best.pt --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --split test --evaluate --output-dir outputs/experiments/reconstruction_dreamsim_topk4/seed_0/test_predictions
```

#### Key Inputs
- outputs/experiments/reconstruction_dreamsim_topk4/seed_0/best.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt

#### Metrics
- `alpha` = 0.0000
- `eval_alex2` = 0.6434
- `eval_alex5` = 0.5921
- `eval_clip` = 0.5386
- `eval_effnet` = 0.9778
- `eval_inception` = 0.5332
- `eval_pixcorr` = 0.2136
- `eval_ssim` = 0.4326
- `eval_swav` = 0.7227
- `prototype_topk` = 4.0000

#### Observations
- Backfilled from existing artifacts.
- Even with the best quantitative metrics, the generated images collapsed to low-information textures and were not semantically recognizable.

#### Next Action
Treat the prototype-residual VAE path as a baseline only and pivot reconstruction to a non-prototype generator.

### EXP-20260418-194826-reconstruction-topk4-train [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-194826-reconstruction-topk4-train","backfilled":true,"command":"python scripts/train_reconstruction.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --perceptual-bank outputs/cache/dreamsim_train.pt --latent-bank outputs/cache/vae_train.pt --output-dir outputs/experiments/reconstruction_dreamsim_topk4 --prototype-mode score_weighted_topk --prototype-topk 4 --epochs 10 --batch-size 16 --official-eval-limit 200 --official-eval-every 10 --device cuda","goal":"Train the residual VAE reconstruction baseline with score-weighted DreamSim top-k=4 prototypes.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/dreamsim_train.pt","outputs/cache/vae_train.pt","prototype_mode=score_weighted_topk","prototype_topk=4"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"alpha":0.0,"epoch":10.0,"eval_clip":0.5290954773869347,"eval_pixcorr":0.21651263307797614,"eval_ssim":0.42825098790633054,"lr":0.0,"train_total_loss":1.2168013036699223,"val_clip_loss":0.3511493389423077,"val_image_l1":0.21070005930960178,"val_latent_l1":0.6494535356760025,"val_lpips":0.5726395845413208,"val_total_loss":1.2342607321647496},"next_action":"Generate test reconstructions and inspect whether the qualitative quality matches the metric gains.","observations":["Backfilled from existing artifacts.","This was the strongest quantitative reconstruction run among the residual VAE baselines."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0","status":"success","timestamp":"2026-04-18T20:30:34+08:00"} -->

- Timestamp: 2026-04-18T20:30:34+08:00
- Area: reconstruction
- Kind: train
- Goal: Train the residual VAE reconstruction baseline with score-weighted DreamSim top-k=4 prototypes.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_topk4/seed_0
- Backfilled: yes

#### Command
```bash
python scripts/train_reconstruction.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --perceptual-bank outputs/cache/dreamsim_train.pt --latent-bank outputs/cache/vae_train.pt --output-dir outputs/experiments/reconstruction_dreamsim_topk4 --prototype-mode score_weighted_topk --prototype-topk 4 --epochs 10 --batch-size 16 --official-eval-limit 200 --official-eval-every 10 --device cuda
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/dreamsim_train.pt
- outputs/cache/vae_train.pt
- prototype_mode=score_weighted_topk
- prototype_topk=4

#### Metrics
- `alpha` = 0.0000
- `epoch` = 10.0000
- `eval_clip` = 0.5291
- `eval_pixcorr` = 0.2165
- `eval_ssim` = 0.4283
- `lr` = 0.0000
- `train_total_loss` = 1.2168
- `val_clip_loss` = 0.3511
- `val_image_l1` = 0.2107
- `val_latent_l1` = 0.6495
- `val_lpips` = 0.5726
- `val_total_loss` = 1.2343

#### Observations
- Backfilled from existing artifacts.
- This was the strongest quantitative reconstruction run among the residual VAE baselines.

#### Next Action
Generate test reconstructions and inspect whether the qualitative quality matches the metric gains.

### EXP-20260418-204517-reconstruction-top1-test [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-204517-reconstruction-top1-test","backfilled":true,"command":"python scripts/predict_reconstruction.py --reconstruction-checkpoint outputs/experiments/reconstruction_dreamsim_top1/seed_0/best.pt --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --split test --evaluate --output-dir outputs/experiments/reconstruction_dreamsim_top1/seed_0/test_predictions","goal":"Generate and evaluate test reconstructions for the top-1 prototype residual model.","key_inputs":["outputs/experiments/reconstruction_dreamsim_top1/seed_0/best.pt","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt"],"kind":"predict","metric_scope":"test","metric_source":"reconstruction_metrics.json","metrics":{"alpha":0.0,"eval_alex2":0.6253517587939699,"eval_alex5":0.5882160804020101,"eval_clip":0.5383165829145728,"eval_effnet":0.9781257510185242,"eval_inception":0.5455276381909547,"eval_pixcorr":0.1891786140474589,"eval_ssim":0.43126818536386624,"eval_swav":0.7107250690460205,"prototype_topk":1.0},"next_action":"Compare against the score-weighted top-k reconstruction variant.","observations":["Backfilled from existing artifacts.","Quantitative image metrics looked acceptable, but the generated images were not visually recognizable."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_top1/seed_0/test_predictions","status":"success","timestamp":"2026-04-18T20:31:04+08:00"} -->

- Timestamp: 2026-04-18T20:31:04+08:00
- Area: reconstruction
- Kind: predict
- Goal: Generate and evaluate test reconstructions for the top-1 prototype residual model.
- Metric Scope: test
- Metric Source: reconstruction_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_top1/seed_0/test_predictions
- Backfilled: yes

#### Command
```bash
python scripts/predict_reconstruction.py --reconstruction-checkpoint outputs/experiments/reconstruction_dreamsim_top1/seed_0/best.pt --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --split test --evaluate --output-dir outputs/experiments/reconstruction_dreamsim_top1/seed_0/test_predictions
```

#### Key Inputs
- outputs/experiments/reconstruction_dreamsim_top1/seed_0/best.pt
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt

#### Metrics
- `alpha` = 0.0000
- `eval_alex2` = 0.6254
- `eval_alex5` = 0.5882
- `eval_clip` = 0.5383
- `eval_effnet` = 0.9781
- `eval_inception` = 0.5455
- `eval_pixcorr` = 0.1892
- `eval_ssim` = 0.4313
- `eval_swav` = 0.7107
- `prototype_topk` = 1.0000

#### Observations
- Backfilled from existing artifacts.
- Quantitative image metrics looked acceptable, but the generated images were not visually recognizable.

#### Next Action
Compare against the score-weighted top-k reconstruction variant.

### EXP-20260418-194745-reconstruction-top1-train [success]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260418-194745-reconstruction-top1-train","backfilled":true,"command":"python scripts/train_reconstruction.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --perceptual-bank outputs/cache/dreamsim_train.pt --latent-bank outputs/cache/vae_train.pt --output-dir outputs/experiments/reconstruction_dreamsim_top1 --prototype-mode top1 --prototype-topk 1 --epochs 10 --batch-size 16 --official-eval-limit 200 --official-eval-every 10 --device cuda","goal":"Train the residual VAE reconstruction baseline with a single DreamSim top-1 prototype.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/dreamsim_train.pt","outputs/cache/vae_train.pt","prototype_mode=top1"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"alpha":0.0,"epoch":10.0,"eval_clip":0.5313819095477387,"eval_pixcorr":0.20538317636206257,"eval_ssim":0.42511333044980515,"lr":0.0,"train_total_loss":1.2349699419371936,"val_clip_loss":0.35846886268028844,"val_image_l1":0.21142482141462657,"val_latent_l1":0.6558630168437958,"val_lpips":0.5834595440671995,"val_total_loss":1.2486348266784961},"next_action":"Generate test reconstructions and inspect the qualitative outputs.","observations":["Backfilled from existing artifacts.","The residual baseline trained cleanly, but it still depended on train-bank prototypes that do not match test concepts."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_top1/seed_0","status":"success","timestamp":"2026-04-18T20:30:36+08:00"} -->

- Timestamp: 2026-04-18T20:30:36+08:00
- Area: reconstruction
- Kind: train
- Goal: Train the residual VAE reconstruction baseline with a single DreamSim top-1 prototype.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/reconstruction_dreamsim_top1/seed_0
- Backfilled: yes

#### Command
```bash
python scripts/train_reconstruction.py --retrieval-checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --perceptual-bank outputs/cache/dreamsim_train.pt --latent-bank outputs/cache/vae_train.pt --output-dir outputs/experiments/reconstruction_dreamsim_top1 --prototype-mode top1 --prototype-topk 1 --epochs 10 --batch-size 16 --official-eval-limit 200 --official-eval-every 10 --device cuda
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/dreamsim_train.pt
- outputs/cache/vae_train.pt
- prototype_mode=top1

#### Metrics
- `alpha` = 0.0000
- `epoch` = 10.0000
- `eval_clip` = 0.5314
- `eval_pixcorr` = 0.2054
- `eval_ssim` = 0.4251
- `lr` = 0.0000
- `train_total_loss` = 1.2350
- `val_clip_loss` = 0.3585
- `val_image_l1` = 0.2114
- `val_latent_l1` = 0.6559
- `val_lpips` = 0.5835
- `val_total_loss` = 1.2486

#### Observations
- Backfilled from existing artifacts.
- The residual baseline trained cleanly, but it still depended on train-bank prototypes that do not match test concepts.

#### Next Action
Generate test reconstructions and inspect the qualitative outputs.

### EXP-20260418-183838-retrieval-dreamsim-only-fixed-test [success]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260418-183838-retrieval-dreamsim-only-fixed-test","backfilled":true,"command":"python scripts/predict_retrieval.py --checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --perceptual-bank outputs/cache/dreamsim_test.pt --split test --evaluate --output-dir outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/test_eval","goal":"Evaluate the fixed DreamSim-only retrieval checkpoint on the local 200-way test set.","key_inputs":["outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/dreamsim_test.pt"],"kind":"eval","metric_scope":"test","metric_source":"retrieval_metrics.json","metrics":{"alpha":0.0,"top1_acc":0.3449999988079071,"top5_acc":0.625},"next_action":"Use this checkpoint as the retrieval backbone for subsequent reconstruction baselines.","observations":["Backfilled from existing artifacts.","This run raised local retrieval top1 from the starter baseline 0.155 to 0.345."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/test_eval","status":"success","timestamp":"2026-04-18T18:38:37+08:00"} -->

- Timestamp: 2026-04-18T18:38:37+08:00
- Area: retrieval
- Kind: eval
- Goal: Evaluate the fixed DreamSim-only retrieval checkpoint on the local 200-way test set.
- Metric Scope: test
- Metric Source: retrieval_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/test_eval
- Backfilled: yes

#### Command
```bash
python scripts/predict_retrieval.py --checkpoint outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt --perceptual-bank outputs/cache/dreamsim_test.pt --split test --evaluate --output-dir outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/test_eval
```

#### Key Inputs
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/dreamsim_test.pt

#### Metrics
- `alpha` = 0.0000
- `top1_acc` = 0.3450
- `top5_acc` = 0.6250

#### Observations
- Backfilled from existing artifacts.
- This run raised local retrieval top1 from the starter baseline 0.155 to 0.345.

#### Next Action
Use this checkpoint as the retrieval backbone for subsequent reconstruction baselines.

### EXP-20260418-183837-retrieval-dreamsim-only-fixed-train [success]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260418-183837-retrieval-dreamsim-only-fixed-train","backfilled":true,"command":"python scripts/train_retrieval.py --perceptual-bank outputs/cache/dreamsim_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_dreamsim_only_atm_small_fixed --epochs 20 --seed 0","goal":"Retrain the DreamSim-only atm_small retrieval model after removing the implicit CLIP bank.","key_inputs":["outputs/cache/dreamsim_train.pt","encoder_type=atm_small","semantic_bank=None"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":17.0,"lr":1.6349021371744834e-05,"train_perceptual_loss":0.8231934375517358,"train_total_loss":0.5762353966676115,"val_selected_alpha":0.0,"val_top1":0.13482466340065002,"val_top5":0.30652961134910583},"next_action":"Run local test retrieval evaluation on the fixed checkpoint.","observations":["Backfilled from existing artifacts.","This became the strongest retrieval training run on validation and selected alpha=0.0."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0","status":"success","timestamp":"2026-04-18T18:38:37+08:00"} -->

- Timestamp: 2026-04-18T18:38:37+08:00
- Area: retrieval
- Kind: train
- Goal: Retrain the DreamSim-only atm_small retrieval model after removing the implicit CLIP bank.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0
- Backfilled: yes

#### Command
```bash
python scripts/train_retrieval.py --perceptual-bank outputs/cache/dreamsim_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_dreamsim_only_atm_small_fixed --epochs 20 --seed 0
```

#### Key Inputs
- outputs/cache/dreamsim_train.pt
- encoder_type=atm_small
- semantic_bank=None

#### Metrics
- `epoch` = 17.0000
- `lr` = 0.0000
- `train_perceptual_loss` = 0.8232
- `train_total_loss` = 0.5762
- `val_selected_alpha` = 0.0000
- `val_top1` = 0.1348
- `val_top5` = 0.3065

#### Observations
- Backfilled from existing artifacts.
- This became the strongest retrieval training run on validation and selected alpha=0.0.

#### Next Action
Run local test retrieval evaluation on the fixed checkpoint.

### EXP-20260418-180924-retrieval-dreamsim-only-invalid [aborted]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260418-180924-retrieval-dreamsim-only-invalid","backfilled":true,"command":"python scripts/train_retrieval.py --perceptual-bank outputs/cache/dreamsim_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_dreamsim_only_atm_small --epochs 20 --seed 0","goal":"Train a DreamSim-only atm_small retrieval model.","key_inputs":["outputs/cache/dreamsim_train.pt","encoder_type=atm_small"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":7.0,"lr":0.00021809857496093204,"train_perceptual_loss":2.3341441799131073,"train_semantic_loss":2.9158424297627463,"train_total_loss":4.549743311599601,"val_selected_alpha":0.2,"val_top1":0.061064086854457855,"val_top5":0.16324062645435333},"next_action":"Patch the script so `semantic_bank` defaults to `None`, then rerun the DreamSim-only experiment from scratch.","observations":["Backfilled from existing artifacts.","This run was invalid because `scripts/train_retrieval.py` still defaulted `semantic_bank` to `clip_train.pt`, so the supposed DreamSim-only setup actually mixed CLIP back in."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small/seed_0","status":"aborted","timestamp":"2026-04-18T18:09:24+08:00"} -->

- Timestamp: 2026-04-18T18:09:24+08:00
- Area: retrieval
- Kind: train
- Goal: Train a DreamSim-only atm_small retrieval model.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_dreamsim_only_atm_small/seed_0
- Backfilled: yes

#### Command
```bash
python scripts/train_retrieval.py --perceptual-bank outputs/cache/dreamsim_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_dreamsim_only_atm_small --epochs 20 --seed 0
```

#### Key Inputs
- outputs/cache/dreamsim_train.pt
- encoder_type=atm_small

#### Metrics
- `epoch` = 7.0000
- `lr` = 0.0002
- `train_perceptual_loss` = 2.3341
- `train_semantic_loss` = 2.9158
- `train_total_loss` = 4.5497
- `val_selected_alpha` = 0.2000
- `val_top1` = 0.0611
- `val_top5` = 0.1632

#### Observations
- Backfilled from existing artifacts.
- This run was invalid because `scripts/train_retrieval.py` still defaulted `semantic_bank` to `clip_train.pt`, so the supposed DreamSim-only setup actually mixed CLIP back in.

#### Next Action
Patch the script so `semantic_bank` defaults to `None`, then rerun the DreamSim-only experiment from scratch.

### EXP-20260418-180926-retrieval-clip-dreamsim-atm-small [success]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260418-180926-retrieval-clip-dreamsim-atm-small","backfilled":true,"command":"python scripts/train_retrieval.py --semantic-bank outputs/cache/clip_train.pt --perceptual-bank outputs/cache/dreamsim_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_clip_dreamsim_atm_small --epochs 20 --seed 0","goal":"Train an atm_small retrieval model with both CLIP and DreamSim targets.","key_inputs":["outputs/cache/clip_train.pt","outputs/cache/dreamsim_train.pt","encoder_type=atm_small"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":18.0,"lr":7.341522555726972e-06,"train_perceptual_loss":0.9252826863092414,"train_semantic_loss":1.5726140051952247,"train_total_loss":2.220311881646578,"val_selected_alpha":0.0,"val_top1":0.12575574219226837,"val_top5":0.29081016778945923},"next_action":"Test whether a pure DreamSim objective works even better.","observations":["Backfilled from existing artifacts.","Validation selected alpha=0.0, indicating that DreamSim dominated CLIP in the fused model."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_clip_dreamsim_atm_small/seed_0","status":"success","timestamp":"2026-04-18T18:09:26+08:00"} -->

- Timestamp: 2026-04-18T18:09:26+08:00
- Area: retrieval
- Kind: train
- Goal: Train an atm_small retrieval model with both CLIP and DreamSim targets.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_clip_dreamsim_atm_small/seed_0
- Backfilled: yes

#### Command
```bash
python scripts/train_retrieval.py --semantic-bank outputs/cache/clip_train.pt --perceptual-bank outputs/cache/dreamsim_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_clip_dreamsim_atm_small --epochs 20 --seed 0
```

#### Key Inputs
- outputs/cache/clip_train.pt
- outputs/cache/dreamsim_train.pt
- encoder_type=atm_small

#### Metrics
- `epoch` = 18.0000
- `lr` = 0.0000
- `train_perceptual_loss` = 0.9253
- `train_semantic_loss` = 1.5726
- `train_total_loss` = 2.2203
- `val_selected_alpha` = 0.0000
- `val_top1` = 0.1258
- `val_top5` = 0.2908

#### Observations
- Backfilled from existing artifacts.
- Validation selected alpha=0.0, indicating that DreamSim dominated CLIP in the fused model.

#### Next Action
Test whether a pure DreamSim objective works even better.

### EXP-20260418-180301-retrieval-clip-only-atm-small [success]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260418-180301-retrieval-clip-only-atm-small","backfilled":true,"command":"python scripts/train_retrieval.py --semantic-bank outputs/cache/clip_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_clip_only_atm_small --epochs 20 --seed 0","goal":"Train an atm_small retrieval model against CLIP features only.","key_inputs":["outputs/cache/clip_train.pt","encoder_type=atm_small"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":20.0,"lr":0.0,"train_semantic_loss":1.6401433463771968,"train_total_loss":1.6401433463771968,"val_selected_alpha":1.0,"val_top1":0.03325271978974342,"val_top5":0.11789601296186447},"next_action":"Compare against CLIP+DreamSim fusion and DreamSim-only training.","observations":["Backfilled from existing artifacts.","Validation improved modestly over the starter baseline but remained far behind the DreamSim-based variants."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_clip_only_atm_small/seed_0","status":"success","timestamp":"2026-04-18T18:03:01+08:00"} -->

- Timestamp: 2026-04-18T18:03:01+08:00
- Area: retrieval
- Kind: train
- Goal: Train an atm_small retrieval model against CLIP features only.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/experiments/retrieval_clip_only_atm_small/seed_0
- Backfilled: yes

#### Command
```bash
python scripts/train_retrieval.py --semantic-bank outputs/cache/clip_train.pt --encoder-type atm_small --output-dir outputs/experiments/retrieval_clip_only_atm_small --epochs 20 --seed 0
```

#### Key Inputs
- outputs/cache/clip_train.pt
- encoder_type=atm_small

#### Metrics
- `epoch` = 20.0000
- `lr` = 0.0000
- `train_semantic_loss` = 1.6401
- `train_total_loss` = 1.6401
- `val_selected_alpha` = 1.0000
- `val_top1` = 0.0333
- `val_top5` = 0.1179

#### Observations
- Backfilled from existing artifacts.
- Validation improved modestly over the starter baseline but remained far behind the DreamSim-based variants.

#### Next Action
Compare against CLIP+DreamSim fusion and DreamSim-only training.

### EXP-20260418-111348-retrieval-baseline-test [success]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260418-111348-retrieval-baseline-test","backfilled":true,"command":"python scripts/predict_retrieval.py --checkpoint outputs/retrieval/seed_0/best.pt --semantic-bank outputs/cache/clip_test.pt --split test --evaluate --output-dir outputs/retrieval_predictions/seed_0_test_eval","goal":"Evaluate the initial CLIP-only retrieval baseline on the local 200-way test set.","key_inputs":["outputs/retrieval/seed_0/best.pt","outputs/cache/clip_test.pt"],"kind":"eval","metric_scope":"test","metric_source":"retrieval_metrics.json","metrics":{"top1_acc":0.1550000011920929,"top5_acc":0.4000000059604645},"next_action":"Use this number as the retrieval baseline when comparing new models.","observations":["Backfilled from existing artifacts.","This established the initial local test reference point at top1\u22480.155."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/retrieval_predictions/seed_0_test_eval","status":"success","timestamp":"2026-04-18T11:19:19+08:00"} -->

- Timestamp: 2026-04-18T11:19:19+08:00
- Area: retrieval
- Kind: eval
- Goal: Evaluate the initial CLIP-only retrieval baseline on the local 200-way test set.
- Metric Scope: test
- Metric Source: retrieval_metrics.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/retrieval_predictions/seed_0_test_eval
- Backfilled: yes

#### Command
```bash
python scripts/predict_retrieval.py --checkpoint outputs/retrieval/seed_0/best.pt --semantic-bank outputs/cache/clip_test.pt --split test --evaluate --output-dir outputs/retrieval_predictions/seed_0_test_eval
```

#### Key Inputs
- outputs/retrieval/seed_0/best.pt
- outputs/cache/clip_test.pt

#### Metrics
- `top1_acc` = 0.1550
- `top5_acc` = 0.4000

#### Observations
- Backfilled from existing artifacts.
- This established the initial local test reference point at top1≈0.155.

#### Next Action
Use this number as the retrieval baseline when comparing new models.

### EXP-20260418-111347-retrieval-baseline-train [success]
<!-- log-meta: {"area":"retrieval","attempt_id":"EXP-20260418-111347-retrieval-baseline-train","backfilled":true,"command":"python scripts/train_retrieval.py --epochs 20 --seed 0","goal":"Train the initial CLIP-only retrieval baseline from the starter workflow.","key_inputs":["outputs/cache/clip_train.pt","legacy baseline encoder"],"kind":"train","metric_scope":"val","metric_source":"history.json","metrics":{"epoch":10.0,"lr":0.00015000000000000001,"train_loss":2.7975948203323235,"val_top1":0.027811367064714432,"val_top5":0.08827085793018341},"next_action":"Run local test retrieval evaluation on the saved checkpoint.","observations":["Backfilled from existing artifacts.","Best validation top1 remained low, but this run established the reproducible starter baseline."],"output_dir":"/data/xiaoh/DeepLearning_storage/project1_eeg/outputs/retrieval/seed_0","status":"success","timestamp":"2026-04-18T11:13:47+08:00"} -->

- Timestamp: 2026-04-18T11:13:47+08:00
- Area: retrieval
- Kind: train
- Goal: Train the initial CLIP-only retrieval baseline from the starter workflow.
- Metric Scope: val
- Metric Source: history.json
- Output Dir: /data/xiaoh/DeepLearning_storage/project1_eeg/outputs/retrieval/seed_0
- Backfilled: yes

#### Command
```bash
python scripts/train_retrieval.py --epochs 20 --seed 0
```

#### Key Inputs
- outputs/cache/clip_train.pt
- legacy baseline encoder

#### Metrics
- `epoch` = 10.0000
- `lr` = 0.0002
- `train_loss` = 2.7976
- `val_top1` = 0.0278
- `val_top5` = 0.0883

#### Observations
- Backfilled from existing artifacts.
- Best validation top1 remained low, but this run established the reproducible starter baseline.

#### Next Action
Run local test retrieval evaluation on the saved checkpoint.

### EXP-20260418-210000-infra-log-system [started]
<!-- log-meta: {"area":"infra","attempt_id":"EXP-20260418-210000-infra-log-system","backfilled":false,"command":"python scripts/log_experiment.py ...","goal":"Implement the project-wide experiment ledger and backfill the existing runs.","key_inputs":["EXPERIMENT_LOG.md","scripts/log_experiment.py","README.md"],"kind":"debug","metric_scope":"unknown","metric_source":null,"metrics":{},"next_action":null,"observations":[],"output_dir":null,"status":"started","timestamp":"2026-04-18T21:04:55+08:00"} -->

- Timestamp: 2026-04-18T21:04:55+08:00
- Area: infra
- Kind: debug
- Goal: Implement the project-wide experiment ledger and backfill the existing runs.
- Metric Scope: unknown
- Metric Source: None
- Output Dir: None
- Backfilled: no

#### Command
```bash
python scripts/log_experiment.py ...
```

#### Key Inputs
- EXPERIMENT_LOG.md
- scripts/log_experiment.py
- README.md

#### Metrics
- None

#### Observations
- None

#### Next Action
None

### EXP-20260420-193152-reconstruction-sdxl-feasibility-smoke [started]
<!-- log-meta: {"area":"reconstruction","attempt_id":"EXP-20260420-193152-reconstruction-sdxl-feasibility-smoke","backfilled":false,"command":"sbatch outputs/tmp/hpc_sdxl_val8_smoke_long.sbatch","goal":"Queue an SDXL img2img feasibility smoke run that uses retrieval prototypes as init images and prototype text as prompts.","key_inputs":["scripts/predict_reconstruction_sdxl_img2img.py","outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt","outputs/cache/dreamsim_train.pt","outputs/subsets/val8_smoke_seed0.json"],"kind":"smoke","metric_scope":"val8","metric_source":"pending","metrics":{},"next_action":"Wait for the HPC smoke job to finish, then inspect download/runtime cost and the first SDXL feasibility metrics.","observations":["Added a dedicated SDXL img2img feasibility script instead of modifying the existing Kandinsky inference path.","Submitted HPC job 9700621 on long_gpu with stabilityai/sdxl-turbo, prototype_text prompts, 4 denoising steps, and strength 0.4."],"output_dir":"/hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/eval_compare/val8_smoke_seed0/sdxl_turbo_proto_text_s4_g0p0_str040","status":"started","timestamp":"2026-04-20T19:31:52+08:00"} -->

- Timestamp: 2026-04-20T19:31:52+08:00
- Area: reconstruction
- Kind: smoke
- Goal: Queue an SDXL img2img feasibility smoke run that uses retrieval prototypes as init images and prototype text as prompts.
- Metric Scope: val8
- Metric Source: pending
- Output Dir: /hpc2ssd/JH_DATA/spooler/dsaa2012_054/DeepLearning/project1_eeg/outputs/eval_compare/val8_smoke_seed0/sdxl_turbo_proto_text_s4_g0p0_str040
- Backfilled: no

#### Command
```bash
sbatch outputs/tmp/hpc_sdxl_val8_smoke_long.sbatch
```

#### Key Inputs
- scripts/predict_reconstruction_sdxl_img2img.py
- outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_0/best.pt
- outputs/cache/dreamsim_train.pt
- outputs/subsets/val8_smoke_seed0.json

#### Metrics
- None

#### Observations
- Added a dedicated SDXL img2img feasibility script instead of modifying the existing Kandinsky inference path.
- Submitted HPC job `9700621` on `long_gpu` with `stabilityai/sdxl-turbo`, `prototype_text` prompts, `4` denoising steps, and `strength=0.4`.

#### Next Action
Wait for the HPC smoke job to finish, then inspect download/runtime cost and the first SDXL feasibility metrics.

<!-- LOG_ENTRIES_END -->
