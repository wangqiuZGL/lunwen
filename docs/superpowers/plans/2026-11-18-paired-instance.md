# Paired instance supervision implementation plan

> Execute inline with test-driven development and a final independent code review.

Goal: add optional same-image different-instance ranking to downstream REC without changing model parameters or default data flow.

Approved spec: the research route in this chat (2026-11-18). User authorized implementation, README experiment history, commit and GitHub push. Training is not requested in this turn.

Constraints: retain all existing local training changes; no weights/data/credentials in Git; no fabricated results; defaults off; B/C share sampling and augmentation; fixed sample budget; no test-based tuning. Initial support is single-process RefCOCO family downstream REC only, fail fast on unsupported combinations. Different boxes alone do not establish same category or instance identity: exact same boxes and high-overlap boxes are excluded, pairs remain candidates for audit.

## Tasks

- [x] Test then implement independent pair index, deterministic no-replacement batch sampler, loss, and audit CLI. Validate duplicates, overlap, budget, seed, gradient, padding and empty regions.
- [x] Test then integrate paired dataset wrapper/collation and optional multi-box transforms; preserve old paths. Reject crops that remove more than half of either target.
- [x] Integrate flags, metadata, epoch schedule manifests, diagnostics and optional validation predictions; add A/B/C launcher and result collector.
- [x] Record historical metrics with provenance and limitations in README, preserving upstream OneRef README. Add machine-readable experiment history and pending result template.
- [x] Run tests, syntax checks, source diff/secret scan, independent review.
- [ ] Push feature branch and verify remote SHA: blocked by absent Git write credentials; retain local commit.

## Progress / rulings

- Started from feat/oneref-1, 40a95a4, with existing uncommitted single-GPU runtime changes. Working on new branch in the same checkout preserves these changes.
- Local default Python lacks torch; use bundled Python with existing user-site torch and install only missing test dependencies.
- SSH public-key login to previously used hosts failed. Historical records come from prior chat and local experiment notes; original server logs are not claimed to be downloaded.
- Full checkpoint/CUDA verification requires actual server assets; CPU tests exercise real data transforms and optimizer integration, not a simulated full OneRef result.
- Independent review found no blocking defect in the normal A/B/C path. Corrected documentation: inherited RefCOCO pull_item truncates cached box coordinates before float transforms; no false claim of preserving cache decimals.
- Tests use bundled Python 3.12 with available torch/torchvision; default system Python 3.9 has no torch. Installed missing test dependencies locally, retaining server requirements separately.
- GitHub connectors both report push=false; local HTTPS push dry-run fails because no username credential is available. Do not claim remote publication. Local commit will be retained for a later authenticated push.
- Final verification: 18 unittest tests passed; compileall succeeded; all four Bash launchers passed bash -n using Git Bash; secret-pattern scan found no matches; no files under models/ changed. Historical note whitespace normalized without changing metrics.
