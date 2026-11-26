# SADB + local competition, long schedule pilot

Authorized 2026-11-24: test insufficient-training hypothesis on the fixed 20% RefCOCO+ subset.
Official reference: https://github.com/linhuixiao/OneRef/blob/master/train_and_eval_script/train_rec_single_dataset_finetuning_base.sh
Official single-dataset Base recipe: frozen10, lr2.5e-4; finetune20, lr3e-5; cosine,384 input.
This pilot follows those epoch counts and initial LRs, but uses single3090, AMP and effective batch32, rather than official eight-GPU global512/192. It is not a strict reproduction or equal-update-budget comparison to full-data results. More epochs alone do not establish undertraining; optimizer steps, data passes and convergence also matter.

Start from BEiT3 again; shared frozen10 with SADB and competition off. The previous frozen2 checkpoint is NOT renamed or resumed to claim frozen10, since its cosine schedule was shorter.
Compare paired-sampling control vs SADB(lambda0.1)+raw local competition(weight0.1,temp0.1), each finetune20 from identical frozen10. Shared augmentation/sampler/batch/seed/subset. Coverage filtering and score calibration OFF, TCGS OFF.
This two-arm screen estimates the combined effect, not separate SADB/local effects or synergy. If promising, add syntax-only and local-only matched20 arms from the same frozen checkpoint; do not use old 8-epoch results as those ablations.
Standalone SADB's prior +0.0372pp is not evidence of reliable benefit. The local competitor's +0.5484pp remains a single-seed exploratory observation.
Validation each epoch; test unused. Export per-sample predictions and OSS best/latest checkpoints each epoch. Check that both syntax bias and pair loss really execute before long training. Existing results are preserved.
Budget estimate from previous throughput: roughly8-10h for frozen10 + two20 arms, possibly longer due syntax parsing; measure actual first-epoch duration. Dense segmentation/category annotations are not used in training.
Outputs: `/root/lunwen-long-runs/syntax_local_20pct_20261124`; OSS prefix `oss://datasets/experiments/syntax_local_20pct_20261124`.

Deployment: `/root/lunwen-syntaxlong`, tmux `syntax-long`. Local/server38 tests pass. Real GPU smoke passed with 32 valid comparisons, nonzero SADB observed on8 forward calls, both head gradients checked, save/strict reload passed; peak15.61GiB. No new accuracy result yet. Previous pending ablation/seed43 queue superseded; no formal old run interrupted. Calibrated-only preflight finished and its weights are diagnostic only.
