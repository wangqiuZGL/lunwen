# Local query competition implementation and launch plan

Goal: implement the user-approved patch-level competition hypothesis, verify it, then launch a matched sampling-only control and treatment on the single 3090.

Spec: the approved chat proposal of 2026-11-19. For two descriptions of different annotated targets in the same image, supervise their relative scores in exclusive, aligned target regions. This is an exploratory hypothesis, not a proven novel method or accuracy improvement.

Constraints: preserve the published region-ranking default and model parameter structure; no SADB/TCGS, extra annotations or extra sample budget. Pair geometry is only a candidate identity check. Exclude overlapping/coarse ambiguous/padded patches; balance target areas. Temperature 0.1 and auxiliary weight 0.1 are fixed exploratory settings. No test-based tuning.

- [x] Add failing tests for local errors hidden by region means, gradient signs, region-size balance, overlap/padding/empty regions, temperature validation and zero-weight integration.
- [x] Add query_competition loss and parser/engine dispatch; preserve ranking behavior.
- [x] Add a fixed-pair overfit probe, a bounded real-GPU smoke, and a sequential matched experiment launcher. Warmup 2 epochs once (batch8/accum4/lr2.5e-4); control then treatment each finetune8 epochs (batch16/accum2/lr3e-5), full val, seed42, same subset. Both finetunes start independently from the identical warmup weights and RNG seed.
- [x] Run local/server tests and review. Deploy an isolated server checkout with exact source provenance.
- [x] Require finite gradients, valid comparisons, tiny-pair overfit and checkpoint reload before formal training. Do not interpret probe accuracy as experiment results.
- [x] Launch the sequential experiment in tmux, observe its live progress, report paths/configuration. Save initialization, metadata, sampling schedules and best/last checkpoints. Retain no more than necessary duplicate probe checkpoints on the 50GB data disk.

Execution: inline using writing-plans, test-driven-development and verification-before-completion skills. A final independent review is required by executing-plans. User already authorized work and training; no extra approval gate.

Verification ledger: 27 initial tests passed locally and on Python3.8 server; 2 additional backup failure/overwrite tests passed locally. Independent review found OSS overwrite-prompt risk; fixed with -f, DEVNULL, timeout, and verified two real uploads to the same key. Real tiny-pair probe: CE0.68599 ->0.09181, accuracy0.56193 ->0.95049 in150 steps. Bounded real-CUDA smoke saw32 valid comparisons, strict checkpoint reload and successful OSS archive upload. Temporary downloaded archives removed only after extraction size checks; OSS originals retained. No claimed novel-method accuracy result.
