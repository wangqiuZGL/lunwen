# lunwen


## 最新归档：2026-11-29

[双视图推理改进、蒸馏实验和下一轮成对训练交接](docs/experiments/consensus-distillation-20261129/README.md)。完整val/testA/testB的推理收益已经验证；单前向蒸馏尚未确认提升。下一轮基线与候选均训练，4096条计划待服务器启动及教师入口适配，尚未开跑。

## 研究记录（更新于 2026-11-26）

研究任务：基于 OneRef 的视觉定位／指代表达理解（REC），输入图像和描述，输出目标框。

**当前结论：20% RefCOCO+、冻结10轮＋微调20轮、单seed实验中，SADB＋局部竞争未超过普通OneRef基线。暂停追加该组合的长训练，先验证末端定位读出。新方案尚未开始GPU训练。**

| 实验 | val Acc@0.5 | 同一最佳轮次的框平均IoU |
|---|---:|---:|
| 普通OneRef | **75.2649%** | 64.9632% |
| 配对采样对照 | 74.8094% | 64.6447% |
| SADB＋局部竞争 | 75.0418% | **65.1024%** |

全部使用同一冻结阶段初始化。此表是缩减数据、单随机种子的探索结果，不是作者全数据指标，也不能证明独立模块贡献。

- [继续研究前先读：当前状态与交接记录](docs/experiments/CURRENT_STATE.md)
- [代码、近三年文献和下一轮最小实验方案](docs/experiments/visual-grounding-route-review-20261125.md)
- [长训练精确结果与OSS备份校验记录](docs/experiments/long-comparison-20261125/verified_summary.json)
- [定位瓶颈离线诊断](docs/experiments/long-comparison-20261125/localization_bottlenecks.json)
- [旧实验和数据索引](docs/experiments/DATA_INDEX.md)

记录内早期的“运行中”“待验证”保留其历史时间点，以当前状态文件为准。权重、完整数据、账号凭据不进入这次记录提交。训练实现仍需依据各记录标注的源码版本或OSS源码快照恢复，不能把记录同步等同于新方法已实现。
