# 局部竞争的候选扩展与最小实验（2026-11-24）

状态：本地实现候选；未验证真实 GPU 训练或精度，不是已成立的论文创新。

## 目标与现有证据

Question: 在保持 OneRef 推理结构不变时，几何覆盖筛选和跨描述分数校准能否改善局部竞争监督？
Type: exploratory。
当前证据 ER-20261124-querycomp-01：归档 seed42，固定20% RefCOCO+，对照73.3408%，原局部竞争73.8892%。单次提升不能证明以下两个扩展有效。
主线：先验证机制和重复性，再用全量数据与近期同级方法比较。不能将20%成绩与全量论文数字直接排名。

## 三个候选及选择

### A 几何覆盖筛选（已实现）

Hypothesis: 边缘低覆盖 patch 的混合内容可能污染竞争监督。
Implementation: 先沿用重叠/padding排除，再保留独占框区域覆盖率 >= rho 的patch；两个目标各自归一化，任一目标不足 min_patches 时跳过整对。
初始探索 rho=0.5，并非最优值。默认rho=0精确保留旧行为。
Missing evidence: 实际低覆盖比例、小目标损失率、真实目标选择收益。
Support: 相同采样条件下改善准确率/成对目标切换，且覆盖率可接受。
Falsify: 大量小目标被丢弃，或收益来自删除难例而无总体改善。
Limits: 这是框区域几何覆盖，不是分割前景概率，也不核实实例身份或语义互斥。禁止称为“已验证语义可靠监督”。

### B 目标均衡的跨描述分数校准（已实现）

Hypothesis: 不同描述分数图的整体偏置可能影响逐位置的跨描述比较。
Let wa,wb be normalized non-overlapping target weights. Define mu=(wa+wb)/2.
z_q(p)=s_q(p)/tau; c_q=log(sum_p mu(p)*exp(z_q(p))); calibrated logits=z_q-c_q.
Use the existing two-query logistic loss on calibrated logits.
Both normalizers remain in the gradient graph. There is no detach or additional model parameter.
Default calibration=none preserves raw competition.
Analytical support: adding a spatially constant offset to either query leaves calibrated loss unchanged.
Missing evidence: whether such offsets matter on real score maps; whether removing them hurts useful absolute evidence.
Support: improve held-out localization and pair discrimination, not merely a synthetic invariance check.
Falsify: shifts are insignificant, or calibration suppresses useful scores and harms held-out performance.
Limits: removes additive offsets only, NOT multiplicative scales; calibration alone is not claimed novel. Under calibration, logged query accuracy uses calibrated logits, while score_gap remains raw region contrast. Do not compare those diagnostic metrics as if identical definitions.

### C 主体/关系引导的竞争配对（暂缓）

Hypothesis: parse subject/modifier/reference roles to flag semantically ambiguous pairs, combined with verified instance identity.
Missing evidence: relation-error prevalence, parser reliability and nearest prior art. Syntax cannot alone establish exclusivity.
Reason deferred: old SADB shows no clear gain; simply restoring it adds a variable without supporting evidence. If errors warrant, design this as a separate experiment.

Decision: implement A/B independently as training-only switches, then test a 2x2 ablation. No new backbone or inference code.

## 文献证据与新颖性边界

- ER-20261124-synonyms-02 — full paper: [Chen et al., IJCV 2022](https://link.springer.com/article/10.1007/s11263-022-01647-z), Sec3.1 response-map triplet and Sec3.2 negative mining. Already compares same-image expressions and response maps. Supports the need for a close loss-level baseline; forbids claiming first cross-expression/map contrast.
- ER-20261124-propvg-03 — full paper: [PropVG, ICCV2025](https://arxiv.org/html/2509.04833v1), Sec3.2 sentence/word-level proposal refer scoring. Supports recent discriminative grounding comparison; does not establish superiority of patch competition.
- ER-20261124-prove-04 — abstract-only, to-read: [PROVE, 2026 preprint](https://arxiv.org/abs/2610.32440), patch evidence with soft coverage targets and uncertainty. Indicates overlap in terminology; not evidence of novelty or effectiveness for A/B. Full method comparison still required.
- ER-20261124-refadv-05 — abstract-only, to-read: [Ref-Adv, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac37a79cc303ca982703f14caa033d8b-Abstract-Conference.html), distractor-rich reasoning evaluation. Candidate external diagnostic; not a replacement for standard REC.

Search scope: targeted 2024-2026 plus earlier direct antecedents, publisher/arXiv web retrieval; dedicated academic-search MCP unavailable. Not exhaustive. No claim of first/unique/SOTA or predicted gain.

## 实验顺序与停止条件

1. Local math/regression/CPU training-loop tests; show archived default preserved. This does not validate full model training.
2. On data, audit retained pairs/patches by object size before long runs; run 150-step tiny fit and 4-batch real-GPU save/load checks per new option. Fail/low coverage -> inspect, do not blindly train.
3. Same subset, shared warmup hash, same schedule: local / coverage / calibrated / combined. Keep seed42 for first screen; do not tune against test splits.
4. Also retain sampling-only and region-ranking baselines using the existing training entry point. A/B affect the loss, not the sampler.
5. Repeat promising effects with paired seeds, fixed subset; distinguish training-seed variation from data sampling. Report corrected/regressed predictions, category/size strata and image-clustered uncertainty. Set strata before examining gains.
6. Once supported: full training and RefCOCO/+/g; compare OneRef, C3VG, Latent-VG, PropVG, SSVG with pretraining/data/architecture settings clearly labeled. Do not promise beating them.

Launcher: `oneref-ptp/scripts/run_reliable_competition.sh` takes preflight/train and local/coverage/calibrated/combined. Explicit WARMUP and RUN_ROOT are required; formal train also requires OSS_PREFIX. Default executes bounded preflight only. Each call handles one variant; seed may change while subset remains seed42. No automatic long-training chain.

Allowed claim now: two optional hypotheses are implemented and mathematical properties can be tested.
Forbidden: validated improvements, semantic reliability, stable gains or completed GPU checks.

## 本地验证记录

- 2026-11-24: unittest discover 全部37项通过，包含梯度、空区域、过滤、校准偏置不变性、配对顺序对称性和CPU训练循环/零权重行为。
- 与135325c归档损失进行100组含有效独占目标区域的随机回归比较：默认损失、梯度、原有指标逐项精确相等。
- Python编译检查、Bash启动脚本语法检查和git diff --check通过。
- 未在真实数据上检查patch保留率；未运行本轮真实GPU检查或正式训练；没有新准确率。

## 2026-11-24 服务器执行更新

用户授权恢复数据并直接运行完整组合。代码部署于 `/root/lunwen-reliablecomp`；服务器37项测试通过。
恢复82,783张图片，重建子集manifest与归档完全相等（3,399图/24,125表达）。warmup归档所有manifest文件SHA256通过。
128对训练样本抽查：原有效比较246，筛选后242；patch从14,564到12,561（86.25%）。不代表语义可靠性或所有目标尺度均无影响。
combined的150步固定样本检查：loss0.884297->0.071038，accuracy45.337%->97.412%；probe权重丢弃。
真实CUDA smoke保存/严格重载通过，32次有效目标比较，峰值约13.76GiB。
正式运行：8轮微调，batch16/累积2，seed42，384输入，lr3e-5，温度0.1，损失权重0.1，覆盖阈值0.5，balanced_logmeanexp校准。复用归档共同warmup，不重新冻结。
运行目录 `/hy-tmp/lunwen-runs/reliablecomp_combined_20261124/combined/train/finetune`；OSS前缀 `oss://datasets/experiments/reliablecomp_combined_20261124/combined/finetune`。
tmux `reliablecomp-train`；总流程日志 `/root/reliablecomp-pipeline-20261124.log`。
这是联合候选实验，尚无验证集最终成绩；句法未开启。该更新替代上节“未执行真实GPU”的时间点状态，不替代新颖性/精度证据边界。

## 2026-11-25 复盘后更新：停止长训练，转向快速读出诊断

Question: OneRef最后的余弦分数softmax聚合是否保留过多padding/背景权重，且是否能在冻结表征下快速验证读出改进？
Type: exploratory; decision: diagnose before training.
Source: `oneref-ptp/models/OneRef_model.py` forward: normalized language/vision dot product -> softmax over patches -> weighted projected vision pooling -> bbox MLP. Official formulation also uses cosine+softmax: https://papers.nips.cc/paper_files/paper/2024/file/fcd812a51b8f8d05cfea22e3c9c4b369-Paper-Conference.pdf
Hypothesis only: diffuse pooling or padded-token mass may affect localization. No real score-map distribution has been measured yet; do not call this a verified bug or novel method. Temperature and padding masks themselves are conventional tools.

Measured loss accounting: downloaded runtime logs for Nov24 report16.0578h of training-stage wall time (includes stage validation and tiny smoke runs). Earlier Nov18/19 experiments, setup, transfers, upload-only waiting and paid idle time excluded. Not an invoice or actual GPU-busy-hour measurement. Currency loss cannot be inferred without billing.
Long recipe: plain75.2649%, paired74.8094%, fusion75.0418%. Single seed; same frozen initialization. Fusion corrects764/regresses788 vs plain, net-24. Paired vs plain net-49. Image-cluster bootstrap delta intervals include zero. Do not claim established general inferiority, reliable gains, or demonstrated same-category improvement.
Process failures: original ordinary-sampling long baseline came late; tiny overfit and finite gradients were promoted into motivation for long runs; candidate extensions lacked isolated evidence; experiments were advanced without a predefined budget/stop gate. Necessary baseline computation is reusable but does not justify all follow-up spending.

Minimal next action (requires GPU briefly; server currently has none): from saved best plain checkpoint, cache joint-query-conditioned projected patch features/scores or directly cache the pooled vectors for four prespecified candidates. Use ordinary sampling, train-only1024-2048 queries and separate validation queries; no image overlap; official test untouched. Report entropy, padding mass, target-region mass (GT only for audit). Cache depends on image AND query, not image alone.
Candidates: unchanged pool, valid-patch-only pool, valid pool with T=.5, valid pool with T=.25. No new backbone, generated labels or synthetic negative text.
Comparison: unchanged pretrained readout plus equal-budget refitted baseline head; every candidate head starts from the same saved head and sees identical samples/updates. Cache-screen metrics are not official full-validation results. Modifying the encoder/SADB/LoRA invalidates this cache.
Budget: target <=45 minutes total for the first screening round, with a hard15-minute cap per candidate rather than a promise of measured speed. Start with a small timing batch; abort/resize if extraction or fitting exceeds budget. No automatic full-network training after the screen.
Support: non-negligible measured padding/diffuseness and held-out localization improvement beyond equal-budget unchanged readout; pick on one fixed validation image subset, check once on another. Reproduce original pooled vector/predictions before interpreting effects. If no signal, stop pooling direction; do not scan many temperatures.
Falsify: real pooling already concentrates adequately; candidate changes merely increase map sharpness, damage localization, or fail the held-out check. Small differences require paired uncertainty/repetition, not a large training run by default.
Implementation status: `tools/rapid_pooling_probe.py` and five math/isolation tests exist in workspace; tests pass. These do NOT include a completed real-GPU cache extraction or establish accuracy gains. A complete cache-export/readout-fit run remains the next concrete action when a GPU is available.
