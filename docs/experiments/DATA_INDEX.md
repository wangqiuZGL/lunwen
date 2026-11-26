# 数据、实验与OSS索引

整理日期：2026-11-22。GitHub仅保存代码、说明、配置和小体积实验记录；不保存数据集、tokenizer、模型权重、凭据或大型归档。

## 当前服务器目录

| 目录 | 内容 |
|---|---|
| `/root/lunwen` | 原配对分支参考代码 |
| `/root/lunwen-querycomp` | 局部查询竞争实现，与本地实现提交一致 |
| `/hy-tmp/lunwen-assets` | 82,783张COCO图像、RefCOCO族缓存、BEiT3初始化及20%训练子集 |
| `/root/lunwen-data` | 指向数据实际目录的软链接 |
| `/hy-tmp/lunwen-runs/querycomp_matched_20261119` | 本次probe、smoke、共享warmup、control、treatment及summary |
| `/root/lunwen-experiment-records/querycomp_matched_20261119` | 独立小体积记录归档与校验汇总 |
| `/root/lunwen-diagnostics/pair-audit` | 配对审计与50对预览；不是经过验证的同类别负样本标签 |
| `/root/lunwen-setup-logs` | 环境和检查日志；不将整个root目录打包 |

目录保持原位，避免破坏复现路径。probe/smoke仅用于检查，不作为论文精度结果。临时盘在迁移或停止一定时间后可能清空；独立备份在OSS。

## OSS保留内容

| OSS路径 | 内容 |
|---|---|
| `oss://datasets/beit3.zip` | BEiT3初始化权重与SentencePiece |
| `oss://datasets/mscoco2014.zip` | COCO图像原包 |
| `oss://datasets/ref_data_shuffled.zip` | 原始标注缓存 |
| `oss://datasets/experiments/fused_train_20261118_1458.tar` | 旧SADB+TCGS融合实验备份 |
| `oss://datasets/experiments/querycomp_matched_20261119/warmup/checkpoints.tar` | 共享冻结阶段最佳/最新权重及记录 |
| `oss://datasets/experiments/querycomp_matched_20261119/control/finetune/checkpoints.tar` | 配对对照组最佳/最新权重及记录 |
| `oss://datasets/experiments/querycomp_matched_20261119/treatment/finetune/checkpoints.tar` | 局部竞争组最佳/最新权重及记录 |
| `oss://datasets/experiments/querycomp_matched_20261119/experiment_records.zip` | 单独日志、配置、采样清单、预测、源码快照与结果校验；不包含权重或数据集 |

权重归档含checkpoint.pth、存在时的best_checkpoint.pth及backup_manifest.json。分别用于续训和评估。归档按阶段覆盖更新，不代表保留每个中间epoch。恢复时先核对manifest SHA256，再解压到仓库外；原始图片zip不需要重复上传。

## 本次证据与限制

两组完成8轮，最佳均为第8轮。已保留原始逐轮日志、参数、初始化校验值和OSS归档清单。本地取回的是日志与元信息，并非已下载完整权重。2026-11-22已将服务器两组最佳/最新权重共4个文件做流式SHA256核对，全部与OSS归档清单一致，结果保存在独立records包。

旧普通基线与SADB单独实验的权重不在当前已确认的OSS清单里；不要把指标记录误称为已恢复模型。未删除或移动任何正式实验。
