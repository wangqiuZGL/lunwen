# Single GPU fused REC experiment

Code: `/root/lunwen/oneref-ptp`, branch `setup/single-gpu-fused`.
Environment: `conda activate oneref` (Python 3.8.10, torch 2.0.0+cu118).
Assets: `/root/lunwen-data` is a real directory on the root disk (restored 2026-11-18).
Keep the OSS archives as an independent backup. Root-disk migration persistence
depends on the platform.
Outputs: `/hy-tmp/lunwen-runs`; back up before the instance storage expires.

The launcher samples 20% of RefCOCO+ training image IDs with seed 42 and keeps
all expressions for those images. It retains the full official validation and
test files. The manifest is copied into every run directory. Validation/test
image overlap is rejected. The original annotation files are never modified.

## Start

```bash
cd /root/lunwen/oneref-ptp
tmux new -s fused-rec
bash scripts/run_fused_single_gpu.sh train
```

Detach using Ctrl+B then D. Reattach using `tmux attach -t fused-rec`.
The console prints RUN_DIR. Read its `console.log` and stage `log.txt` files.
Each run includes a source snapshot (including new uncommitted scripts), the
base commit, code diff, environment versions and subset manifest.
The script enables SADB (lambda 0.1) and TCGS (16x16, weight 0.1).
It uses AMP, warmup 2 epochs (frozen backbone, microbatch 8, accumulation 4),
then finetuning 8 epochs (microbatch 8, accumulation 4). These are exploratory
settings, not a reproduction of the paper schedule and not final ablation evidence.
Keep the same data, initialization, budget, and model-selection rule for ablations.
No test-set evaluation is used during tuning. Do not start concurrent training runs.

## Smoke check

```bash
bash scripts/run_fused_single_gpu.sh smoke
```

This uses 8 training expressions and 16 validation expressions only, verifies
syntax on/off and zero-bias equivalence, observes head gradients, and reloads
the saved checkpoint strictly. Its accuracy is not an experimental result.

## Overrides

`BATCH_SIZE`, `UPDATE_FREQ`, `WARMUP_BATCH`, `WARMUP_EPOCHS`,
`FINETUNE_EPOCHS`, `NUM_WORKERS`, `ASSETS`, `RUN_ROOT`, `RUN_DIR` are configurable.
`SMOKE_SAMPLES` changes the training sample cap for smoke checks only.
For example `BATCH_SIZE=4 UPDATE_FREQ=8 bash scripts/run_fused_single_gpu.sh train`
retains effective batch 32; measure memory before changing these defaults.
Activation checkpointing and MRefM pretraining are not part of this launcher.

## Resume after interruption

Activate the environment and use the arguments printed in the stage log;
replace `--finetune` with `--resume /path/to/stage/checkpoint.pth` and keep the
same output directory, total epochs, module switches and data. Checkpoints
include optimizer, scheduler and AMP scaler state. Resume from epoch boundaries,
not from the middle of an accumulation window.

Only source/config/docs are intended for Git. Archives, weights, annotation data
and run outputs are outside the repository and ignored. Review staged paths
before every commit. The legacy `ptp_*` names refer to TCGS here, not full PTP
pretraining. Code on a server root disk is not guaranteed permanent storage.
