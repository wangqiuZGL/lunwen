# 同图实例成对辨别实验

状态：实现与 CPU 验证阶段。没有新方法的真实训练指标，也没有做完整 OneRef 权重/CUDA 兼容性验证。

## 研究问题和边界

原有 `use_box_mask_constraints` 已经用 focal + Dice 监督目标 patch。因此本方案研究的是：显式比较目标与同图特定干扰实例，能否在原监督之上减少实例混淆。加一个排序损失本身不等于已经成立的创新点。

使用现有可信 RefCOCO/+/g 训练缓存，不生成描述、不增加图片，不访问验证/测试标注构造训练配对。初版只支持单进程下游 REC，禁止同时启用句法偏置、PTP、MRefM、全局图文对比或分割，以免混入其他改动。

缓存没有实例ID和类别ID：使用同图、不同框、不同规范化描述、框 IoU 不超过0.5产生**候选配对**。相同框和近重复框排除；这些几何规则不能保证严格实例身份或同类别，必须先看审计预览。后续严格同类挖掘需要关联原始实例/类别标注。

## 数据流程

`PairIndex → PairBatchSampler → PairedDataset → paired_collate → train_one_epoch`

- 每轮先打乱索引，再从尚未使用的同图样本中贪心选配对。每条描述最多出现一次；最后不足一批丢弃。每轮训练样本数为 `floor(N/batch_size)*batch_size`，与原流程相同。
- `pair_fraction=0.5` 是配对样本槽位上限，不保证实际达到50%。batch16最多4对，其余为普通样本。配对覆盖率与实际采样数单独记录。
- 成对目标在一次增强调用中共享图像、缩放、裁剪和padding。裁剪至少保留每个框50%面积；失败时重试，达到上限则保留未裁剪图。两个描述的方向词都会影响原有裁剪保护逻辑。
- B/C与A的差异包括采样和成对增强。B/C相同种子时使用相同索引序列和增强流程；A/B的差异不能归因于排序损失。
- 框坐标沿用基线 `pull_item` 的处理：RefCOCO缓存坐标先取整，再转float进行几何变换；配对分支不额外改变这一行为，也不声称保留缓存原始小数。分数使用框掩码。两个框共享的像素、同时包含两个实例的粗patch，以及接触padding的patch不参与辅助比较。有效patch不足则跳过。

## 损失

对描述 A/B 各做一次比较：

`L_pair = mean(relu(margin - s(target, text) + s(distractor, text)))`

`s` 是现有原始余弦相似度按有效区域覆盖率归一化加权的平均值；总损失为原 OneRef 损失加 `pair_loss_weight * L_pair`。默认margin0.1，C组权重0.1只是探索起点，并非已经优化的参数。B组权重0仍计算诊断但不施加排序梯度。

无有效比较返回带计算图的零值。记录 `pair_loss_raw`、`loss_pair`、`pair_candidate_pairs`、`pair_valid_comparisons`、`pair_score_gap`（按batch平均）。逐轮另存候选对总数 `pair_candidate_pairs_total`、有效比较总数 `pair_valid_comparisons_total`、有效比较占比 `pair_valid_fraction`，以及只对有效比较加权的分数差 `pair_score_gap_valid_mean`；分析优先使用这些逐轮汇总字段。

## 开始前审计

在服务器 `conda activate oneref`，进入 `oneref-ptp`：

```bash
python scripts/audit_pairs.py \
  --train-cache /root/lunwen-data/subsets/refcocoplus20_seed42/single_dataset/unc+/unc+_train.pth \
  --image-root /root/lunwen-data/image_data/other/images/mscoco/images/train2014 \
  --output /hy-tmp/pair-audit
```

输出 `data_audit.json`、候选索引JSONL、seed42首轮采样清单，以及最多50对图片预览和对应描述JSON。先检查是否存在同目标误配、歧义描述和低覆盖率，再决定是否花训练预算。预览按索引顺序取样，不能代替随机抽查或完整人工审计。

## A/B/C 对照

| 组 | 冻结2轮 | 微调8轮 | 目的 |
|---|---|---|---|
| A | 原始数据流程 | 原始数据流程 | 同配置基线 |
| B | 原始数据流程 | 配对采样/增强，λ=0 | 数据流程对照 |
| C | 原始数据流程 | 与B一致，λ=0.1 | 排序监督增量 |

默认使用20%训练图片子集（保留这些图片的全部描述），子集seed固定42；训练seed可变。全验证集，不用test选参数。初始化、阶段轮数、学习率与原单卡基线保持一致。冻结阶段batch8×累积4，微调batch16×累积2；图像384、文本64、AMP。各组串行运行，不与正在运行的SADB争用GPU。

```bash
bash scripts/run_pair_ablation_single_gpu.sh A
bash scripts/run_pair_ablation_single_gpu.sh B
bash scripts/run_pair_ablation_single_gpu.sh C
```

以上命令会真实启动训练；本次开发没有执行。运行目录必须为空，脚本拒绝覆盖已有实验。需要恢复时使用原命令参数手工 `--resume`，不要重跑整个两阶段脚本。

覆盖选项：`ASSETS RUN_ROOT RUN_DIR SEED BATCH_SIZE UPDATE_FREQ WARMUP_BATCH WARMUP_EPOCHS FINETUNE_EPOCHS NUM_WORKERS PAIR_WEIGHT PAIR_MARGIN PAIR_FRACTION PAIR_MAX_IOU PAIR_MIN_PATCHES`。每次修改记入实验表；B/C保持除权重外完全一致。探索有效后再跑训练seed42/43/44；所有seed仍使用同一数据子集。仅在稳定收益后考虑全量训练。

## 记录与选择

启动脚本保存源码ZIP、Git SHA/diff、环境、启动脚本、子集清单和审计图。训练阶段保存 `config.json`、逐轮 `log.txt`、最后/最佳checkpoint、AMP scaler、每轮配对采样清单和逐样本验证预测。

验证预测字段：样本索引、图片ID、描述、预测/真实框、IoU、Acc@0.5命中。框坐标为**padding后输入上的归一化cxcywh**，不应直接叠加到原始图片；绘图需要先还原缩放和padding。IoU与现有验证函数一致，预测xyxy裁剪到[0,1]。

```bash
python scripts/collect_results.py /hy-tmp/lunwen-runs/pair_A_seed42_实际时间 \
  /hy-tmp/lunwen-runs/pair_B_seed42_实际时间 \
  /hy-tmp/lunwen-runs/pair_C_seed42_实际时间 --output /hy-tmp/pair-comparison
```

生成CSV/JSON：同一个最佳**已记录验证准确率轮次**的Acc和mIoU，不独立挑选最大mIoU。日志不完整或最佳权重缺失会显示，不能当作完整结果。旧训练器还会把初始化验证纳入best门槛：如果初始化/恢复前最佳优于本次全部轮次，汇总的最佳日志轮次可能不是实际最佳checkpoint，必须核对权重与日志；工具不会替代这一核查。

最终填写 [结果模板](../docs/experiments/pair-results-template.csv)。报告C−B与C−A（百分点）、3个种子均值/标准差、总验证集及预先定义的实例混淆子集，不只展示成功样例。子集定义要在看新结果前冻结；错误类型需人工抽查。按图片聚类的bootstrap和细分类错误审计作为后续分析步骤，当前工具没有声称自动完成。

## 测试和兼容性

```bash
python -m unittest discover -s tests -v
python -m compileall -q engine.py train_oneref.py utils scripts tests datasets/transforms.py
```

CPU测试包含真实数据增强、DataLoader整理、训练循环与优化器梯度、零权重、空/重叠/padding区域、逐样本预测及汇总选择。未加载真实BEiT3/OneRef权重；完整CUDA smoke、严格旧checkpoint加载、真实数据覆盖率与精度需要服务器资产。本改动不修改 `models/`，不新增模型参数；已有单卡AMP/梯度累积改动也在本次归档。
