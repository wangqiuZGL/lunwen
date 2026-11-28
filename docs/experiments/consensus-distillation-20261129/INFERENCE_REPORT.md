# 已验证的OneRef推理改进

实验标记20261127；报告日期2026-11-28（北京时间）。最终方案在完整val前冻结，在testA/testB后未修改。

## 结果

| 划分 | 描述数 | 原模型Acc@0.5 | 改进后 | 提升(pp) | 图像聚类95%区间(pp) |
|---|---:|---:|---:|---:|---|
| val | 10758 | 75.2742% | 75.8226% | +0.5484 | [+0.2714, +0.8184] |
| testA | 5726 | 79.5669% | 80.3004% | +0.7335 | [+0.3418, +1.1264] |
| testB | 4889 | 67.4371% | 67.9894% | +0.5523 | [+0.1036, +1.0035] |

val纠正123/破坏64；testA纠正72/破坏30；testB纠正65/破坏38。所有比较使用同一原检查点、同一FP32精度及对应完整划分。历史val记录75.2649%与本轮75.2742%差1条；该差异不计入方法收益，也未确定其来源。

## 冻结算法

1. 原图和水平翻转图各执行一次原模型推理；翻转框与patch匹配图映回原坐标。
2. 两框IoU≥0.5时取框均值；不一致时，仅当两个对齐的原生图文匹配图各自都支持翻转候选且平均cosine优势>0.02时选择翻转框，否则保留原框。
3. 每个候选在同一张图上按框内有效patch中心的平均分比较；不是跨视图比较未校准的绝对分数。
4. 含左右/罗盘/clock子串的描述跳过翻转选框及融合，仍保留有效图像边界裁剪。
5. 框裁剪至输入图像有效区域。全程不读取真实框或标签；真实标注只用于校准和评价。

0.02规则在官方train中的512条图像隔离校准描述上确定；完整val后保持冻结。val此前参与过模型选择与多轮探索，因此val区间不是独立测试结论。testA/testB用于本轮冻结方案最终确认；其他margin仅诊断报告，没有按测试最优选择。

## 代码与复现

主模块：code/lunwen-syntaxlong/oneref-ptp/utils/rec_refinement.py。eval.py增加可选参数 --rec_refinement verified_flip，默认none保留原行为；engine.py在evaluate/validate中接入。沿用原有正常评估启动方式和配置，使用相同FP32基线比较。
函数接口：predict_refined(model.eval(), image_tensor, padding_mask, texts, mode="verified_flip")，返回原OneRef推理元组，仅改变第一个框输出。分割输出保持原始预测，不声称改善分割。
完整实验入口：tools/flip_consistency_screen.py。冻结规则对照命令：python flip_consistency_screen.py --root /hy-tmp/layerwise-20261127 --output <新目录> --full --verify --split testA --integration-smoke。testB同理。不要用最终测试集继续选择阈值。
几何/方位回退单元检查通过；testA/testB首批真实样本验证独立推理模块与实验算法输出一致，最大误差见FINAL_SUMMARY.json。独立审阅者已用原始预测和仓库官方评估函数重建val真值并复算。

## 范围与成本

这是现有检查点的确定性推理改进，不需要新训练或新增学习参数；代价是两次骨干前向。本轮GPU当前已空闲，未自动关闭租赁实例。两个训练种子的失败实验不能作为该确定性方法的重复种子；证据来自固定输入复现、完整划分及配对图像统计。
该改进尚未通过系统新颖性认证，不等于已找到足够论文创新；没有声称SOTA或跨模型/跨数据集通用收益。框验证和精修已有相关工作，例如OneRef官方论文与VLTVG，需后续查新。

## 记录

保留此前分层、局部修正、完整微调、尺度/翻转等失败结果，未用小子集分数替换完整划分分数。原始预测、manifest、协议、统计与独立审阅报告随最终归档保存。

独立证据复核：fresh-context result-to-claim审阅者已重建val/testA/testB官方真值、复算官方指标并由缓存分数重建冻结规则，误差均0，窄范围claim_supported=yes，same-family/provisional。另起的experiment-audit会话未完成初始化，已记录REVIEW_UNAVAILABLE，不声称单独审计PASS。

OSS主归档：oss://datasets/experiments/layerwise_20261127/verified_refinement_final_records.tar.gz；补充复核记录另行归档。
