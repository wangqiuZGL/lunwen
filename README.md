# OneRef 视觉指代定位实验记录

基于 [OneRef（NeurIPS 2024）](https://github.com/linhuixiao/OneRef) 的研究仓库，记录句法引导、中心网格辅助监督及同图实例辨别的实现和实验。**代码可运行不代表创新点有效；探索结果与正式消融分开记录。**

记录日期：2026-11-18；原始日志快照截至北京时间22:58。百分数为百分比，Δ为百分点；没有数据的位置保留“未核实/未评估”。

## 最新整理：2026-11-22

局部查询竞争实验已完成，两组都复用同一冻结阶段权重，seed42、20% RefCOCO+训练图片、完整val、微调8轮、batch16×累积2。验证最佳结果：

| 组别 | 竞争损失权重 | val Acc@0.5 | 同轮平均IoU |
|---|---:|---:|---:|
| 相同配对采样对照 | 0 | 73.3408% | 62.2440% |
| 局部查询竞争 | 0.1 | 73.8892% | 62.7179% |

相对配对对照提升0.5484和0.4738个百分点。单seed探索结果，尚无重复种子、testA/testB或与普通区域排序损失的完整对照，不声称稳定提升或已确立创新性。旧普通基线73.3965%仅作额外参考。

数据来源：从OSS权重归档中读取日志并与归档清单SHA256核对，两组归档均含第8轮断点；初始化SHA256相同。数值及原始日志见[本次记录](docs/experiments/querycomp-20261119/verified_comparison.json)和[CSV](docs/experiments/querycomp-20261119/results.csv)。代码实现提交为`f301e0a`。目录与备份见[数据索引](docs/experiments/DATA_INDEX.md)。

下文2026-11-18表格是当时的历史快照，其中“SADB运行中”是旧状态；其后本聊天现场核实最终Acc为73.4337%，mIoU为62.1113%，但该最终两轮日志未包含在旧快照中，不补写进原始日志。

## 目录与当前代码

| 目录 | 用途 |
|---|---|
| `OneRef-master/` | 上游代码参考副本 |
| `oneref/` | 早期句法实验代码 |
| `oneref-ptp/` | 当前单卡实验、SADB/TCGS与可选同图配对监督入口 |
| `docs/experiments/` | 历史记录、来源说明、后续实验表模板 |

上游安装和模型说明保留在 [OneRef README](oneref-ptp/README.md)。单卡运行说明见 [SERVER_EXPERIMENT.md](oneref-ptp/SERVER_EXPERIMENT.md)，新方法细节和命令见 [PAIR_EXPERIMENT.md](oneref-ptp/PAIR_EXPERIMENT.md)。本地当前主要开发目录为 `oneref-ptp/`。

## 创新方向及验证状态

| 方向 | 实际实现/目的 | 已有证据 | 当前判断 |
|---|---|---|---|
| SADB：句法引导注意力偏置 | 用句法结构调整文本区域注意力；开关 `enable_syntax_bias`，强度 `syntax_bias_lambda` | 融合实验已完成；单独消融已取回前6轮微调日志，最终值尚未完成 | 独立收益待验证，不能用代码检查替代效果证据 |
| TCGS：文本条件中心网格监督 | 文本条件的联合特征预测目标中心所属网格；旧代码名为 `ptp_*` | 16×16分类，辅助权重0.1；融合实验未超过基线；此前记录分类训练准确率约1.6% | 暂停扩大实验，需审查标签/特征与监督价值；不是完整PTP预训练 |
| 同图实例成对辨别（本次新增） | 同图不同框/描述候选配对，直接比较目标区域和特定干扰区域的匹配分数 | CPU逻辑/梯度/数据增强测试；真实训练尚未执行 | 待验证研究假设，不能声称涨点或已确立新颖性 |

新方法与原监督的区别：OneRef本来就通过focal＋Dice监督相似度图。本方案额外引入**两个具体实例间的相对排序**，必须通过采样对照证明它在原监督之外有贡献。缓存中的几何配对不是经过验证的同类别困难负样本。

## 2026-11-18：20% RefCOCO+ 探索实验

已复用此前提供的SSH连接信息，从服务器取回并核对原始逐轮日志。六份阶段日志见 [training-logs](docs/experiments/training-logs)，精确指标见 [CSV快照](docs/experiments/server-results-snapshot.csv) 和 [历史记录](docs/experiments/history.json)，来源、配置变化及SHA256见 [同步清单](docs/experiments/server-sync-manifest.json)。本次未下载模型权重。

| 实验 | SADB | TCGS | val Acc@0.5 ↑ | 同次报告的平均IoU ↑ | 相对基线 | 状态 |
|---|---|---|---:|---:|---:|---|
| OneRef 基线 | 关闭 | 关闭 | **73.40%** | **62.23%** | — | 日志核实：微调8/8轮完成，最佳epoch7 |
| OneRef＋SADB＋TCGS | λ=0.1 | 16×16，权重0.1 | 73.21% | 62.08% | Acc −0.19；mIoU −0.15 | 日志核实：微调8/8轮完成，最佳epoch7 |
| OneRef＋SADB | λ=0.1 | 关闭 | 71.32%（阶段最佳） | 59.97%（同轮） | 暂不作最终比较 | 快照仅微调6/8轮，最佳epoch5；不是最终成绩 |

### 训练配置与比较限制

| 项目 | 记录 |
|---|---|
| 数据 | 按训练图片ID抽取20%，保留所选图片的全部描述；完整官方验证集 |
| 数据子集种子 / 训练种子 | 42 / 42 |
| 模型 | `beit3_base_patch16_384`，REC框回归 |
| 初始化 | `beit3_base_indomain_patch16_224.pth` |
| 图像 / 文本 | 384 / 最长64 tokens |
| 冻结阶段 | 2轮，batch8，梯度累积4，lr=2.5e-4 |
| 微调阶段 | 8轮，lr=3e-5，cosine；启用原有框掩码约束 |
| 基线与SADB微调 | batch16，梯度累积2，有效batch32 |
| 融合微调 | 前2轮batch8×累积4，后改为batch16×累积2；过程中断点恢复 |
| 其他 | AMP；验证集挑选最佳；未取得testA/testB结果 |

**限制：**单种子、缩减训练集、融合组中途调整过单步batch。有效batch相同不能消除全部过程差异。已核实Acc与平均IoU来自各自最佳准确率的同一轮，表中epoch从0开始。这组已完成对照支持“本次融合未观察到提升”，不支持“两个模块均已被证明无效”。SADB阶段值不能与8轮完成结果作最终结论。

### 运行与备份记录

| 实验 | 此前记录的服务器路径/会话 | 备注 |
|---|---|---|
| 基线 | `/hy-tmp/lunwen-runs/baseline_train_20261118`；`baseline-rec` | 最佳权重此前记录在 `finetune/best_checkpoint.pth` |
| SADB单独消融 | `/hy-tmp/lunwen-runs/sadb_only_train_20261118`；`sadb-only` | 本次22:58日志快照：冻结2轮完成、微调6轮已记录；不是最终成绩 |
| 融合备份 | `oss://datasets/experiments/fused_train_20261118_1458.tar` | 此前报告已上传完整备份；本次未重新校验远端文件 |
| 融合记录备份 | `oss://datasets/experiments/fused_train_20261118_1458_records.zip` | 此前报告包含日志、配置、源码快照、环境、子集清单和校验记录 |

服务器临时盘、OSS路径和会话名称只用于找回记录，不表示本次仍在线或文件仍存在。仓库不存放权重和完整训练数据。

### 本次服务器代码同步

已将 `/root/lunwen` 的源码快照下载到本机 `C:/Users/LH/Desktop/lunwen/server-sync/20261118/source`，292个文件逐个通过SHA256校验；控制台日志、环境记录、训练源码ZIP等保留在同级 `training-records` 文件夹。未下载数据集或权重，也未修改服务器正在使用的代码。

服务器代码基于 `40a95a4`，未提交的单卡训练修改已包含在本机 `063dd52` 中。比较忽略CRLF/LF后，8个内容差异均为本机新增配对功能、README/说明、版本号修复或脚本引号差异；服务器的两个启动脚本已按原文同步。本机保留配对扩展，服务器原版独立保存。唯一额外的Notebook README检查点只留在快照，不混入训练源码。

GitHub推送认证单独核查：服务器与本机当前均无可用HTTPS写入凭据，复制源码不会自动获得仓库写权限；发布状态以实际push结果为准。

## 早期实验笔记（训练条件未完整核对）

来源为工作区原有 `实验效果.md`，已保留 [笔记快照](docs/experiments/legacy-notes.md)。以下不与上面的20%训练子集结果混用。“作者指标”沿用笔记，未在本次逐项复核论文。

| 数据集 / 方法 | val Acc@0.5 | testA / test | testB | 来源与解释 |
|---|---:|---:|---:|---|
| RefCOCO / 作者OneRef | 88.75 | 90.95 | 85.34 | 笔记所列作者值 |
| RefCOCO / OneRef复现 | 90.08 | 91.64 | 86.36 | 笔记复现汇总；缺完整配置与种子 |
| RefCOCO / 加diff模块 | 89.20 | 90.94 | 85.84 | 笔记汇总；另附epoch19日志val=89.0622%，不可视为同一个数值 |
| RefCOCO+ / 作者OneRef | 80.43 | 86.46 | 74.26 | 笔记所列作者值 |
| RefCOCO+ / 句法模块段落 | 82.40 | 86.93 | 75.99 | 原文写“加入语法模块：复现效果”；没有同配置未加模块对照，不能计算独立贡献 |
| RefCOCOg / 作者OneRef | 83.68 | 83.52 | — | 笔记未标清具体split协议 |
| RefCOCOg / 句法模块后续10轮 | 82.924 | 84.16 | — | 原文首组val不完整；仅列清楚的后续值 |
| ReferIt / OneRef复现 | — | 78.60 | — | 笔记未给val |
| ReferIt / 句法模块20轮 | 80.229 | 78.377 | — | 笔记记录 |
| ReferIt / 句法模块23轮 | 80.24 | 78.43 | — | 笔记记录 |
| ReferIt / 句法模块30轮 | 80.69 | 78.51 | — | val增长未对应更高的笔记基线test；不能用test反向选轮次 |
| Flickr30k / 句法模块 | 82.028 | 82.7510 | — | 笔记作者值83.61；没有同配置复现基线 |

笔记另含HiVG复现实验，原文保留于快照。表格只整理清楚可辨的OneRef数字，不把缺失或含糊记录补成结果。

## 新方向：A/B/C 如何运行与记录

先看 [完整实验说明](oneref-ptp/PAIR_EXPERIMENT.md)，审查配对覆盖率和预览，再决定训练。

| 组 | 冻结2轮 | 微调8轮 | 要判断的贡献 |
|---|---|---|---|
| A | 原始流程 | 原始流程 | 同配置基线 |
| B | 原始流程 | 成对采样/增强，排序权重0 | 数据组织的效果 |
| C | 原始流程 | 与B相同，排序权重0.1 | C−B为新损失的增量；C−A为整体变化 |

```bash
cd oneref-ptp
# 真实训练命令，串行执行；本次没有运行。
bash scripts/run_pair_ablation_single_gpu.sh A
bash scripts/run_pair_ablation_single_gpu.sh B
bash scripts/run_pair_ablation_single_gpu.sh C
```

新功能默认关闭；旧训练命令继续使用原有采样、增强和损失。`--enable_pair_sampling --pair_loss_weight 0`是B组，不等于关闭配对功能。模型文件和参数结构未修改，完整旧权重加载仍需实际资产验证。

每次运行保留代码SHA/diff/快照、环境、配置、子集与配对清单、逐轮指标、逐样本验证预测和最后/最佳权重；`scripts/collect_results.py`输出CSV/JSON，填入 [结果模板](docs/experiments/pair-results-template.csv)。不能只挑最佳seed；后续报告3个seed的均值/标准差及C−B、C−A，并标注训练中断或配置变化。没有收益的运行也保留。

## 开发验证

```bash
cd oneref-ptp
python -m unittest discover -s tests -v
```

测试覆盖配对筛选、固定预算无放回采样、双目标增强、排序梯度、零权重、空/重叠/padding区域、训练循环、预测导出和结果汇总。此次开发还归档此前未提交的AMP、梯度累积、断点scaler和验证日志修复，并修正torchvision版本号解析。**CPU测试通过不能替代真实数据CUDA smoke和精度实验。**
