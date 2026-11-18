# Single GPU fused experiment preparation

Goal: Prepare SADB + TCGS on a fixed 20% RefCOCO+ image subset and full validation set.

Approved scope: User requested server preparation and a command to begin experiments; do not start a full training run.

1. Keep assets outside the repository and ignore weights, archives and outputs.
2. Reuse working CUDA 11.8 PyTorch in a separate environment; pin dependencies.
3. Regression-test accumulation against equivalent large batches, including a final partial window; add opt-in AMP and save its state.
4. Prepare deterministic image-level training subset, unchanged validation/test splits, and an external manifest.
5. Add single-GPU fused launcher with configurable paths, warmup and finetuning, separate run directories and logs.
6. Verify CPU regression tests, syntax on/off behavior, labels, actual GPU forward/backward, save/reload and short validation.
7. Record environment and launch instructions. Do not claim accuracy improvements from smoke tests.
