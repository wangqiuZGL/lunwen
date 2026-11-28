# OneRef 推理改进与蒸馏实验交接

更新：2026-11-29。状态：已有结果归档；4096条成对训练流水线已启动：先teacher_train18生成教师，再pair_smoke19检查梯度与坐标，随后paired_distill19执行两组×两种子。

## 已确认与未确认的结论

| 方法 | 划分 | 原模型 Acc@0.5 | 方法 Acc@0.5 | 解释 |
|---|---|---:|---:|---|
| 双视图验证选框 | 完整val，10758条 | 75.2742% | 75.8226% | +0.5484个百分点 |
| 双视图验证选框 | testA，5726条 | 79.5669% | 80.3004% | +0.7335个百分点 |
| 双视图验证选框 | testB，4889条 | 67.4371% | 67.9894% | +0.5523个百分点 |
| 单次前向编码器锚定蒸馏 | 开发val，2048条 | 77.5879% | 两种子均77.6855% | 每次净2条，区间跨零，尚未确认提升 |

不同划分的分数不能混排。前三行是同一检查点、同一FP32精度的确定性推理比较，需两次前向，没有新增训练参数。第四行训练最后两层及定位头，不是全模型训练；不能将前三行的收益称为蒸馏学生收益。所有方法的新颖性均未认证，不声称SOTA。

双视图规则与0.02阈值在最终test之前冻结。testA/testB不再用于下一轮选参数。此前开发验证集反复用于探索，应明确其开发性质。历史val75.2649%与本轮匹配FP32基线差一条，不将差值计入方法收益。

## 本目录内容

- [推理结果与方法](INFERENCE_REPORT.md)、[机器可读汇总](INFERENCE_SUMMARY.json)。
- [编码器训练配置与结果](ENCODER_DISTILLATION_REPORT.md)、[编码器计划](ENCODER_PLAN.md)。
- [完整搜索和失败历史](SEARCH_HISTORY.md)、[蒸馏初筛](DISTILLATION_PILOTS.md)。
- [独立结果复核](RESULT_TO_CLAIM.md)：same-family/provisional；逐样本真值、官方指标和冻结规则已复算。另起experiment-audit未初始化，不声称独立审计PASS，见[AUDIT_AVAILABILITY.json](AUDIT_AVAILABILITY.json)。
- `results/`保存16个run的现有小型数字结果及协议，[ARTIFACT_INDEX.json](ARTIFACT_INDEX.json)给出SHA256。没有的历史协议不伪造补齐。
- [下一轮成对训练计划](NEXT_PAIR_PLAN.md)、[配置](NEXT_PAIR_CONFIG.json)：基线与候选均实际训练，每组两个种子，固定4096训练/2048开发验证、1024步、batch16。

## 代码

推理模块：[`utils/rec_refinement.py`](../../../oneref-ptp/utils/rec_refinement.py)，`eval.py`可选`--rec_refinement verified_flip`，默认`none`。学生单前向评估必须保持`none`。方向表达跳过翻转融合，但仍保留有效图像区域裁剪；不改变原分割输出。

研究入口在[`oneref-ptp/scripts/consensus_research/`](../../../oneref-ptp/scripts/consensus_research/)。这些是当前工作版本；实际历史run源码快照保存在对应OSS记录包中。脚本`--root`指向恢复后的实验根目录，不指向脚本目录。需要原始OneRef代码、兼容环境、检查点、分词器和数据；不能仅凭脚本目录重现训练。

`train_consensus_distillation.py`对应冻结骨干初筛；`train_encoder_distillation.py`对应512条教师缓存的最后两层训练。**后者目前不能直接扩为4096条，不能用其`full_replay`参数启动扩大实验**：教师manifest强制匹配512条。下一轮必须先实现可参数化教师数据入口，生成4096教师预测，并检查样本、坐标和图像隔离后再启动。

## 开机后的执行顺序

1. 验证GPU空闲、复用已确认环境和原检查点，不覆盖旧结果。
2. 固定4096官方train样本及共享manifest，排除验证图像；生成冻结教师预测。
3. 实现并检查4096教师数据入口，断言教师/学生顺序、坐标一致，复现原模型，检查有限梯度。
4. 基线和候选从相同原权重开始，同样数据、顺序、训练范围、学习率、步数、种子42/43；差异仅蒸馏项。两组都训练，不用旧模型分数替代新基线。
5. 固定最终步，保存各组权重、逐样本预测与配对统计，备份OSS。先实测100步吞吐再修正30–60分钟估算；数据环境恢复另计。

这里的基线是同预算继续训练，不是重新复现OneRef整篇论文。用户授权本轮两组训练；无需重复请求行动确认，可用SSH入口到位即可继续。

## 大文件与凭据

Git只存代码、协议、数字摘要和来源索引；不存SSH密码、OSS凭据、数据集、原始图片、权重或大缓存。原权重/707图恢复包为`layerwise_512_256_bootstrap.tar`，只适用原小划分，不能充作4096完整数据。

OSS根目录：`oss://datasets/experiments/layerwise_20261127/`。关键记录包：`verified_refinement_final_records.tar.gz`、`verified_refinement_final_review.tar.gz`、`distill15_records.tar.gz`、`distill_anchor16_records.tar.gz`、`encoder_distill17_records.tar.gz`。这些路径是先前已确认上传记录，本次归档未重新联网校验OSS对象。


## 2026-11-29实际启动补充

新增`prepare_teacher_4096.py --train-manifest`与`train_paired_distill_4096.py --teacher-dir`，已实现4096教师manifest独立入口；上文关于旧512脚本不能直接扩展的限制仍成立。新的训练入口强制4096条训练、教师manifest一致、每批教师GT/学生GT精确一致，原模型2048预测先复现。先运行2步anchored_quality真实GPU smoke（128条开发val），通过才执行正式基线与候选。

运行根目录`/hy-tmp/layerwise-20261127`；状态`pair19.status`，日志`teacher_train18.log`、`pair_smoke19.log`、`paired_distill19.log`。普通native与anchored_quality各42/43种子，1024步、batch16，训练最后两层/定位头；无筛选蒸馏不是本轮必跑组，旧结果仍保留。原始teacher/学生GT只来自官方train，最终test不参与本轮。

当前记录仅证明流水线已启动，不等于4次训练已完成。实际训练状态应以GPU进程、日志更新和results.json为准。归档最终自动上传`paired_distill19_records.tar.gz`到既有OSS实验根目录。SSH凭据未保存到仓库。
