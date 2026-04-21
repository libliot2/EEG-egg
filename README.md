# Project1 EEG-to-Image System

This repository contains the code and documentation for an EEG-to-image project
with two tasks:

- `retrieval`: given an EEG sample and a candidate set of images, rank the
  correct image highly
- `reconstruction`: given an EEG sample, predict an image representation and
  decode it into an image

The codebase includes both the older prototype / residual-VAE baselines and the
newer Kandinsky-embedding reconstruction pipeline.

## What Is Included

- training / inference scripts under `scripts/`
- installable source package under `src/project1_eeg/`
- experiment documentation:
  - `PROJECT_EXPERIMENT_REPORT_ZH.md`
  - `EXPERIMENT_LOG.md`

## What Is Not Included

This repository is set up to exclude large local artifacts from version control:

- dataset folders such as `image-eeg-data/`
- generated experiment outputs under `outputs/`
- downloaded model weights and checkpoints under `models/`
- local environment directories, caches, and logs

If you want to run the project, place the released dataset next to the repo in a
directory named `image-eeg-data/`, so the default paths resolve as:

- `../image-eeg-data/train.pt`
- `../image-eeg-data/test.pt`
- `../image-eeg-data/training_images/`
- `../image-eeg-data/test_images/`

## Environment

Create a fresh environment:

```bash
conda env create -f environment.yml
conda activate project1-eeg
python -m pip install -r requirements.lock.txt
python -m pip install -e .
```

Notes:

- `environment.yml` only creates the base Python environment.
- `requirements.lock.txt` pins the Python packages used in the project.
- If you plan to train on GPU, you may want to replace the CPU-only `torch` /
  `torchvision` entries in `requirements.lock.txt` with CUDA builds that match
  your machine.

## Main Documents

- `PROJECT_EXPERIMENT_REPORT_ZH.md`
  - high-level project narrative, major findings, and current conclusions
- `EXPERIMENT_LOG.md`
  - canonical experiment ledger
- `EXPERIMENT_LOG_hpc.md`
  - optional HPC mirror / archive, not the primary source of truth

## Recommended Entry Points

Useful scripts:

- `scripts/train_retrieval.py`
- `scripts/predict_retrieval.py`
- `scripts/train_reconstruction_embed.py`
- `scripts/predict_reconstruction_embed.py`
- `scripts/predict_reconstruction_sdxl_img2img.py`
- `scripts/train_eeg_pretrain.py`
- `scripts/log_experiment.py`

Current historically strong runs discussed in the report:

- retrieval backbone: `retrieval_dreamsim_only_atm_small_fixed`
- reconstruction training checkpoint: `reconstruction_kandinsky_embed_v4_proxyselect`
- best locally decoded configuration: `kandinsky_predicted_v4_fast`

The corresponding artifacts are not committed to the repository by default.

## Experiment Logging

Use `EXPERIMENT_LOG.md` to record important runs.

Start an attempt:

```bash
python scripts/log_experiment.py start \
  --area retrieval \
  --kind train \
  --goal "train dreamsim retrieval seed 1" \
  --command "python scripts/train_retrieval.py ..."
```

Finish the same attempt:

```bash
python scripts/log_experiment.py finish \
  --attempt-id EXP-YYYYMMDD-HHMMSS-retrieval \
  --status success \
  --output-dir outputs/experiments/retrieval_dreamsim_only_atm_small_fixed/seed_1 \
  --observation "improved validation top1 over seed 0"
```

## Minimal Workflow

1. Cache the required image banks.
2. Train a retrieval model with `scripts/train_retrieval.py`.
3. Train a reconstruction model with `scripts/train_reconstruction_embed.py`.
4. Evaluate or decode predictions with the corresponding `predict_*.py` script.

## Notes

- Validation splits are created from the released training set.
- Local reconstruction evaluation is useful for comparison, but it is not an
  official benchmark server.
- Some reconstruction metrics are only proxies; see the report for the exact
  caveats when interpreting them.
