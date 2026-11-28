# OneRef REC 持续搜索：独立结果—主张核查

日期：2026-11-27。审阅者：fresh native Codex reviewer，gpt-6-astra / ultra；路由见同目录 `reviewer_route.json`。仅阅读源代码、结果、协议与 manifest，做 CPU 计数和集合检查；未运行 GPU、修改实验代码或接触凭据。

## 结构化裁决

```yaml
intended_claim: 已找到超出探索集噪声、可复现的 OneRef REC 改善
primary_metric: Acc@IoU>=0.5
claim_supported: no
confidence: high
review_independence: same-family
acceptance_status: provisional
integrity_status: unavailable
route: pivot_to_bounded_evidence_gathering
paper_innovation_claim: not_authorized
```

**结论：当前审阅到的完成实验没有证成上述主张。** 这是一项对“已经找到可靠改善”的否定裁决，不是证明所有尝试的总体真实效应必然为零。未发现应把现有结果直接判为造假的证据；也没有完整独立重算或正式完整性审计，不能给 integrity pass。

- **what_results_support:** 在指定检查点、RefCOCO+、固定步数和已测划分上，分层混合、低学习率/冻结头门控、局部精修及短程继续训练均没有得到可确认的主指标改善；精修在 256 条上的 +1/+2 样本没有在扩大评估中形成稳健证据。数据与原生训练监督的分离在所读实现和 manifest 中基本清楚。
- **what_results_dont_support:** 稳定 Acc50 增益、跨数据集泛化、独立测试成功、架构贡献、新颖性，以及“层间信息/ROI/门控/额外训练数据均无效”等一般性结论。
- **missing_evidence:** 执行时源码/权重/数据哈希闭环；训练实际消费样本记录；原始标注上游来源的哈希认证；冻结规则后一次未被选择过程使用的确认；正增益若来自训练还需重复种子。六组已有预测的官方指标逐条重算及主要候选的图像聚类配对区间已在本审阅中补齐。
- **suggested_claim_revision:** “完成了一组受限、可追踪的探索与扩大验证；未确认主指标改善，继续以固定候选的错误互补性和训练集内选择规则做低成本证据收集。”
- **next_experiments_needed:** 先用已有原/翻转预测做 CPU 错误互补性与选择上限诊断；存在足够可纠正案例才运行一个训练集内确定、模型输出驱动的固定两候选选择规则，然后完整 val 配对分析，最后冻结后一次测试确认。详见末节。

## 数值证据核查

下列百分数来自直接读取 JSON；净变化同时由 corrected−regressed 重算。两个训练种子使用同一验证样本，不能把它们当成两份独立验证样本相加。

| 实验/候选 | 样本数 | 原始 Acc50 | 候选 Acc50（42 / 43） | 纠正/破坏（42；43） | 解释 |
|---|---:|---:|---:|---|---|
| screen01 最终层普通头 | 256 | 75.78125 | 75.00000 / 75.00000 | 见 diagnostic02 | 头继续拟合也未改善 |
| screen01 separate | 256 | 75.78125 | 75.390625 / 74.609375 | 0/1；0/3 | 无正增益 |
| refinement04 ROI | 256 | 75.78125 | 76.56250 / 76.171875 | 3/1；2/1 | 仅净 +2/+1 个样本，探索性 |
| confirm05 ROI | 2048 | 77.587890625 | 77.587890625 / 77.685546875 | 8/8；9/7 | 0 / +0.09765625 pp，未确认 |
| confirm05 global | 2048 | 77.587890625 | 77.734375 / 77.685546875 | 4/1；4/2 | +0.146484375 / +0.09765625 pp，且两种子 mIoU 都下降 |
| confirm05 native | 2048 | 77.587890625 | 77.34375 / 77.5390625 | 8/13；8/9 | 同数据原生头控制均不升 |
| confirm05 native_calibration | 2048 | 77.587890625 | 77.392578125 / 77.34375 | 8/12；11/16 | 额外监督控制均不升 |
| lastblocks06_fp32 native | 256 | 75.78125 | 75.390625 / 73.828125 | 2/3；1/6 | 本配置未改善 |
| lastblocks06_fp32 box_only | 256 | 75.78125 | 75.390625 / 74.609375 | 0/1；1/4 | 未证实移除辅助目标有效 |
| full_replay09 native | 2048 | 77.587890625 | 77.001953125 / 76.953125 | 53/65；44/57 | −0.5859375 / −0.634765625 pp |

ROI42 在扩大评估的 mIoU 为 66.72423，低于原始 66.75336；ROI43 为 66.86781，但 Acc75 59.57031 仍低于原始 59.86328。full_replay 两种子 mIoU 均约降 1.22 pp，Acc75 均约降 3.2 pp。没有多个指标一致支持的稳健正结论。

作为尺度感校验，对 confirm05 的 corrected/regressed 数量做忽略图像相关性的精确双侧二项配对检验，global42、global43、ROI42、ROI43 分别为 p=0.375、0.6875、1、0.803619。这些不是正式的图像聚类推断，也未处理候选搜索多重性；仅说明这里连乐观的描述级简单计算都没有强证据。正式结论必须使用图像聚类配对分析。

valid_projection08 在 256 条上净 0，在 2048 条上仅纠正 1 条（+0.048828125 pp）；mIoU 确有数值增加，但不能把这个单样本 Acc50 变化宣布为目标达成。full_replay 的事后预测插值诊断没有正 Acc50 结果，也不能视为预先冻结的独立确认。

### 独立原始预测复算

协调者随后补齐原始预测与 `official_repo_val.pth`，审阅者在本地CPU独立核算：直接从仓库源文件AST提取并执行原始 `bbox_iou`、`xywh2xyxy`、`trans_vg_eval_test_iou` 函数，与自定义实现、保存IoU及JSON逐条比较。以下六组所有候选的最大IoU差、Acc50/Acc75/mIoU报告差均为 **0**，阈值hit变化为 **0**：scale03、confirm05、full_replay09、lastblocks06_fp32、flip10、flip_confirm11。各组GT由官方仓库val的10,758行按manifest索引、image/phrase一致性断言和原生整数化/resize/pad重建，最大坐标差也全为 **0**。这是对当前仓库标注与运行产物的一致性验证，不是对上游原标注的认证。

独立使用 seed20261127、5,000次图像整簇有放回抽样，每次以采得图像中所有描述数作分母，得到下列配对95%百分位区间（pp）：

| 候选 | ΔAcc50 [95% CI] | ΔmIoU [95% CI] |
|---|---|---|
| confirm05 global42 | +0.1465 [−0.0491, +0.3893] | −0.0432 [−0.1121, +0.0249] |
| confirm05 global43 | +0.0977 [−0.1445, +0.3443] | −0.0530 [−0.1454, +0.0382] |
| confirm05 ROI42 | 0 [−0.3984, +0.3918] | −0.0291 [−0.1813, +0.1164] |
| confirm05 ROI43 | +0.0977 [−0.3055, +0.4912] | +0.1144 [−0.0198, +0.2417] |
| full_replay42 | −0.5859 [−1.5965, +0.4393] | −1.2351 [−1.8197, −0.6361] |
| full_replay43 | −0.6348 [−1.5455, +0.2957] | −1.2205 [−1.7830, −0.6434] |
| flip_confirm11 consensus_mean | −0.0488 [−0.4505, +0.3526] | +0.3266 [+0.1668, +0.4804] |
| flip_confirm11 consensus_projection | −0.0488 [−0.4505, +0.3526] | +0.4340 [+0.2722, +0.5868] |

后两个规则纠正9、破坏10，确有正的mIoU次指标信号，但没有Acc50提升。上述区间只条件于固定候选，未校正自适应搜索或训练种子不确定性。与协调者不同bootstrap种子的轻微区间差异符合重采样差异；点估计一致。

## GT 来源与指标忠实度

1. `prepare_layerwise_server.py` 从 `splits.zip` 的 `ref_data_shuffled/single_dataset/unc+/unc+_{train,val}.pth` 建立数据；旧 20% train 由排序图像列表、seed42、向上取整采样重建，断言描述数 24125。`prepare_unseen_train_cache.py` 从同一完整 train 取额外数据，明确排除旧训练图像。已独立读取补齐的仓库官方val并重建各预测GT，**未独立认证原始 archive、COCO 原标注或旧训练作业样本哈希**。
2. `datasets/data_loader.py` 从原行读取 bbox/phrase；框先 `np.array(..., dtype=int)`，再 xywh→xyxy，随后原生 resize/pad 转归一化 cxcywh。`enable_seg_mask=false` 时辅助 object mask 来自原 bbox 矩形，属于既有框监督，不能称额外人工分割标签。整数化属于现有仓库处理路径，不能把结果描述成已核对原始 COCO 浮点标注。
3. ROI 坐标来自冻结预测框；质量标签只在官方训练集 calibration 内根据候选/原始 IoU 差生成。val 的 oracle_correctable/AUC 是事后诊断，所读代码没有将其用于逐样本推理。failure_audit 的人工视觉分类仅是助手判断，不是金标准，也没有发现它进入所读训练脚本。
4. 自定义 `iou_giou(..., evaluation=True)` 与 `eval_utils.py` 都把预测转换角点并裁到 [0,1]，GT 不裁剪，阈值使用 `>=0.5`，因此主要语义一致。实现仍有一般性数值差异：自定义分母 `union.clamp(min=1e-8)`，官方 `union+1e-16`；但对已补齐六组实际预测逐条重算，IoU最大差和阈值变化均为0。不要将这个实测一致性延伸成任意退化输入下必然等价。
5. scale03 把不同缩放预测和 GT 映回共同 384 坐标，并断言 GT round-trip 误差 <2e-6；原尺度回放误差约 6e-8。它检验的是 384 画布中缩小图像内容的特定视图，不能泛称检验了所有多尺度推理。
6. image-content projection 的边界由图片尺寸、原生舍入与 padding 规则产生，推理不读 GT。只有 GT 完全位于有效图像矩形时才可直接获得裁剪不降低 IoU 的几何保证；现有报告有 3 个 GT 超出该矩形，不能宣称对所有样本无条件保证。样本中的 `iou_decreased=0` 是经验检查。

## 图像划分隔离与额外数据曝光

审阅者直接解析 manifest 并按 image 集合求交：

| 集合 | 描述数 | 唯一图像数 |
|---|---:|---:|
| 旧 screen train | 512 | 471 |
| 探索 val | 256 | 236 |
| unseen04 train | 1536 | 1413 |
| unseen04 calibration | 512 | 470 |
| confirm05 val | 2048 | 1008 |
| full_replay09 train | 8192 | 5917 |

旧 screen train、新 train、新 calibration 与探索/确认 val 未发现图像交集；新 train 与 calibration 互斥；探索与确认 val 互斥。confirm05 与 full_replay09 使用同一 2048 条验证 manifest。各集合 source_index 无内部重复。source_index 必须结合源 split 解释，train 与 val 相同数字并不表示同一记录。

**图像隔离不等于未被选择使用的测试集。** 原检查点已经使用官方 val 选择；256 条经历大量自适应搜索；2048 条现又用于后续路线决策。确认样本可检验当前 256 条上的偶然性，却不能恢复成独立 test。未来 bootstrap 也不能消除这种选择偏差。

refinement04 各主候选都看 1536 条新官方 train；quality 模型额外看 512 条和 128 步，native_calibration 提供额外标签曝光控制，但算力/容量/训练目标不同，不能称完全匹配。93,364 参数 ROI 与 95,284 参数 global 是近似容量控制，而 native 有 2,954,500 参数。任何未来增益都须先区分额外数据、优化方式与结构效应。

full_replay manifest 8192 条由旧图描述 4096 + 新图描述 4096 组成。每种子只训练 256×16=4096 次样本呈现，即 DataLoader 第一次打乱后的半个所选池，**没有完成该 8192 池的一轮**；实际消费的旧/新比例没有单独记录，不能宣称各消费4096。它与旧 screen train 有337图、新 train有389图、旧 calibration有124图交集，均属于训练数据再使用，不是 val 泄漏；但后续不可再把被重用的 calibration 图像称为从未训练见过。`full_replay09/protocol.json` 的 scope 仍写“New official-training images; reused exploratory256 val”，与实际混合8192/确认2048不符，应修正描述并保存原件。

## 冻结、优化器、恢复与精度

- screen01 的骨干 eval/frozen，head 每臂从原始模块 deepcopy，AdamW 独立新建；diagnostic02 冻结头臂仅训练 gate/norm/strength，并有冻结权重逐项相等断言。初始 residual strength=0 会使第一步 gate 梯度为0；后续记录非零梯度，所以“门控从未学到”不成立。
- refinement04 使用冻结缓存和原始预测；新 Correction 末层零初始化，框修正限幅于原框 logit 坐标的 `0.5*tanh`；主候选重新初始化，采用统一固定样本顺序、最终256步。ROI/global 的 L1+GIoU 与当前原生 box 两项的系数一致；被冻结的 mask 部分对 correction 是常数。质量模型在候选固定之后训练，阈值固定0.5。
- `train_lastblocks.py` 每 seed/arm 都 strict 恢复原始 state、新建 AdamW。last2 只解冻最后两块、最终norm和原头，并保持 model.eval；full_replay 解冻 beit3+原头且 model.train。后者既改变可训练范围也改变 dropout/drop-path，且训练池、评估集和精度均变，**不可将两个实验差异单因归给解冻范围或精度**。
- last2 使用原生随机训练图像增强，frozen cache 使用确定性 val 视图；均为合法但不同的训练条件。没有恢复原检查点 optimizer/scheduler，属于新开优化器的短程适配，不是精确续训原训练轨迹。
- screen01 实际缓存为 FP16 特征、FP32 head；diagnostic02 报告 train/val 缓存导致 hit 变化均0，最大预测误差约4.6e-6/4.7e-6；unseen04 最大回放误差6.79e-6。所读初始回放断言支持缓存误差很小，但最终候选阈值敏感样本仍应正式重算。
- lastblocks06_fp32 协议明确此前 FP16 溢出未完成首步，排除其结果合理。FP32完成结果可支持“此配置不升”，不能支持“混合精度方法无效”。full_replay09 训练 forward 为 BF16 autocast，几何损失显式转 FP32，GradScaler关闭；`evaluate()` 无 autocast，最终与 baseline 都经 FP32路径评估。故这不是 BF16 eval 与 FP32 eval 直接比较，也没有证明 BF16 是退步原因。
- 固定最终步避免单臂挑最好 val checkpoint，但没有消除跨轮架构/超参数选择。保存 state/pred/Iou 的代码与 baseline replay 断言是积极证据；后续补齐 `.pt` 后已独立重算预测级指标。full_replay/last2预测由协调者从较大的checkpoint提取，本审阅者没有直接加载完整模型做forward，也未建立训练时封存源码与执行文件的完整哈希链；不能把预测级复算冒充模型级重跑。

## 负结果的有效边界

screen01/diagnostic02 只能否定所测冻结骨干、小旧训练子集、短程优化和当前残差混合配置；旧 train 达98.24%而 val约75.78%，说明继续拟合旧样本的诊断价值有限。层6/9用最终层头直接读取并短训的失败也可能涉及表示/头不匹配。

ROI 的局部1.25倍、3×3采样及限幅修正主要服务局部框精修，不能据此排除解决远距离实例混淆的空间方法。质量门控只测到当前特征、标签“ΔIoU>0.01”、固定阈值0.5；ROI quality仅接受0/1样本，支持“该规则近乎不采取动作”，不支持“质量估计普遍无用”。

scale03 的尺度/均值规则均未提高，但不同尺度分别纠正7/9例，也破坏13/10例，说明存在错误互补性，尚未证明可由模型输出可靠选择。last2 第一批 box/aux 梯度cosine约−0.002/0.050，一批近正交并不证明全训练无冲突；box_only负结果也仅限该短程设置。

`run_spatial_readout_probe.py` 的旧 spatial probe 使用5×L1+2×GIoU且仅训练readout；不能将其负结果和原生1×L1+1×GIoU+mask约束结果混成同一严格消融。

## 明确的低算力下一方向与停止规则

**停止继续调同一个256条上的门控/ROI学习率。下一轮只检验“原/翻转候选有可识别的错误互补性”这个具体问题。**

1. 此项CPU筛查已独立完成：flip_confirm11中原对/翻对1486、原对/翻错103、原错/翻对78、两者均错381；理想GT选择上限81.396484375%，比原始高3.80859375pp。它表明互补性足以值得一次有界选择规则试验，但没有证明模型分数能够识别这78例。继续按图像和框一致性分层；GT oracle绝不保存为可用预测。
2. 如有足够潜力，只用未参与评估的官方train/calibration确定一个固定选择规则：将flip框和相似度图映回原坐标，在同一张固定图上比较两候选，或预先规定两视图对称打分；使用明确的面积归一化和有效图像遮罩，避免跨视图未校准cosine或大框偏好充当目标身份置信度。没有胜出证据时回退原框。先核对它确实预测“选谁可改善Acc50”，不要仅优化辅助分数或平均IoU。
3. 保留 original、固定mean/consensus、固定selector三个对照，明确额外一次forward的成本。规则、阈值与主指标冻结后完整val统一评估；该val仍作为开发证据，给图像聚类配对95%区间、corrected/regressed和Acc75/mIoU。若Acc50仍不升，不把次指标或一个样本改写成已满足目标。
4. 只有主指标出现实质、可复现正信号后，才用事先保留、未因当前结果反复挑选的测试集一次确认。若测试失败，不能在该测试上继续选规则再声称独立确认。纯确定性推理不需要伪造“随机种子复现”，而需要保存预测及再运行一致性；有训练参数则重复固定种子。

本报告已核实补齐后的flip_confirm11，其Acc50负、mIoU正，可以形成“该固定规则在此开发划分上改善定位质量次指标”的窄结论，不能升级为当前主指标目标达成。另读取verify_calibration12结果：官方train内512条，原始75.58594%，verified_consensus_0.02为76.36719%（5纠正/1破坏）。这只是训练校准结果；三个阈值的组合结果相同，不能当成三次独立重复。完整val的verify_full13尚未纳入本裁决，收到完成产物后应单独复核冻结规则与主指标。

## 证据范围与快照

已直接读取任务指定的七个主要实现、六组核心结果、full_replay协议/manifest、evidence_precheck、持续记录；补读缓存准备、layerwise实现、诊断、原生dataset/transforms/loss/box实现和scale/last2/projection结果。后续还读取六组原始预测、对应manifest、官方仓库val，以及flip确认/verify训练校准结果；执行上述独立GT重建、官方函数复算与图像bootstrap。

`evidence_precheck.json` 仅确认3个数值文本存在，不验证GT、实验运行或主张。未找到此范围内的 `EXPERIMENT_AUDIT.json`；因此该语义审查为 **same-family / provisional — no integrity audit available**。执行器负责后续 findings、pipeline与trace归档；本审阅者仅写本报告。

关键文件SHA256：

```text
screen01/results.json 147eb5b300fc111e541f9a679638b0f8c6eb8859189c5b1353ed13d244d71fc5
diagnostic02/results.json 2fbf1cdbe11acd4db4418a8a33a92968dc4311b79aaaf49c8e9a1ff34a2b29d3
refinement04/results.json 4590bcca869edc0b937c2cd79bb28ec3be1376fb3f51a546d67168026eeb1e02
confirm05/results.json 93598cd1a9c4766d25e48d394cf8d4b722562f79e10465fe2d70bdc637897dd3
full_replay09/results.json 8ac0866123ef5c6ffb6a0a2eb84aacf6437fce3fc385a4452eccd76f7de1758e
full_replay09/manifest.json eaad6762d1785ca40c415c72414292f2f3ae68fabdbb72f55de1431085f0c4f4
```

---

## 后续裁决一：完整 val 的冻结推理规则（2026-11-28）

**本节追加新证据，不覆盖上文截至旧候选/flip_confirm11的 `claim_supported: no` 历史。** 新候选为 `verified_consensus_0.02`：原/翻转候选一致时取均值；不一致时，只有两张已对齐的原生图文匹配图各自都以框内有效patch平均cosine优势>0.02支持翻转候选才换框；最后按输入有效图像区域裁剪。方位表达跳过翻转/融合，仍执行边界裁剪。参数固定、无新训练参数、推理不读GT，代价为两次原模型forward。

冻结依据包括此前已读的官方train-calibration512结果、协调者在完整val结果产生前的冻结通知，以及 `verify_full13/protocol.json` 的 `frozen_primary=verified_consensus_0.02`。其他margin只作为诊断；完整val的0.0虽分数更高，本裁决没有改选它。

```yaml
phase: full_validation
claim_supported: partial
supported_scope: 当前原检查点在匹配FP32完整val上的确定性推理改善
remaining_at_this_phase: 冻结后测试集确认
confidence: high
review_independence: same-family
acceptance_status: provisional
```

审阅者重新读取 `verify_full13/{results.json,protocol.json,manifest.json,predictions.pt,paired_statistics.json}`；独立从仓库官方val重建GT，直接执行官方源函数重算全部候选，并以保存的 `score_advantage`、`agreement`、`skip`、原/翻转框及图像边界重建主规则。GT误差、官方vs保存IoU、JSON指标误差、主规则框重建误差全部为 **0**。完整manifest按source_index完整覆盖10,758条，无抽掉错误样本。

| 项目 | 独立复算结果 |
|---|---|
| 样本/图像 | 10,758描述 / 1,500图像 |
| 原始正确数 → 主规则正确数 | 8,098 → 8,157 |
| Acc50 | 75.2742145380% → 75.8226436141%（精确计数比值） |
| 纠正 / 破坏 | 123 / 64，净 +59 |
| ΔAcc50 | +0.5484290760 pp，图像bootstrap95% [+0.278030,+0.815075] |
| ΔmIoU | +0.6048593619 pp，95% [+0.496697,+0.713663] |
| ΔAcc75 | +1.1061535601 pp，95% [+0.658670,+1.559618] |

上述bootstrap由审阅者独立使用seed20261127、5,000次图像整簇重采样，保留描述加权分母；与执行器区间的轻微差别来自抽样种子。保存JSON通过FP32均值输出的75.822647%与精确整数比例末位不同，只是浮点归约差异，不改变正确数或结论。

主规则对固定 `consensus_projection` 仍净增30例：43纠正/13破坏，+0.278862 pp，95% [+0.128937,+0.427315]；对单独 `valid_projection` 增+0.464770 pp，95% [+0.197314,+0.726649]。所以此次正信号不只是之前单样本的有效图像裁剪变化；但这仍是组合方法的经验效果，不能据此宣布语义身份“已被证明验证正确”。

按历史探索涉及的图像作事后分层：236探索图对应完整val的1,983描述，净+13；1,008确认图对应7,350描述，净+33；其余256图对应1,425描述，净+13。这说明增益并非只来自最初256描述。但该分层是事后诊断，且完整val历史上用于检查点选择，不能把最后这256图重新命名为独立test。

### 与历史75.2649%的差异

历史75.2649191299%=8,097/10,758；本次匹配FP32原模型75.2742145380%=8,098/10,758，恰差1例。**只使用本次同精度、同批评估的配对baseline计算方法增益：+59例，而非+60例。** 未直接取得并定位历史逐样本预测差异，不能断言这一例一定由AMP或其他特定因素导致；报告可说明来源尚未确定。

原protocol的方位描述“all methods fall back to original”不完全准确：跳过翻转后仍有有效区域裁剪，212个方向表达中62个框因此发生变化。保持冻结算法不动，以文字澄清“跳过翻转选择/融合，保留边界裁剪”；最终报告已使用这一准确描述。

完整val证据快照：

```text
verify_full13/results.json 32fb60566bf5997a0c5929ab7e59127f0141e3583765e5ca8a0281a66b2305ce
verify_full13/predictions.pt ab694c185a58f9a03af371168e6efd8df84d47d2c0645b3018ac9e7e9b30cf27
verify_full13/protocol.json e42b8251d87d4518725330a8203902ce95df3377c04fc331c9b538a3a022bd62
verify_full13/manifest.json 04cfcf1a76ad0636add55b442b04c60e7a7238789517daa47772dbeff47913a8
```

## 后续裁决二：最终 testA/testB 确认（2026-11-28）

**最终窄范围 `claim_supported: yes`。** 冻结于完整val之前、未按test结果改选的 `verified_consensus_0.02`，在同一原检查点、FP32、当前仓库RefCOCO+完整testA/testB上，相对同步评估的原推理得到可复算的Acc50改善；两测试集的图像聚类95%区间均高于0。此结论替代“当前仍未找到改善”作为最新项目状态，但不撤销前述旧候选的失败结论。

```yaml
phase: frozen_final_test_confirmation
claim_supported: yes
claim_scope: 同一现有OneRef检查点在当前RefCOCO+仓库完整val/testA/testB上的固定确定性推理改善
what_results_support: 冻结verified_consensus_0.02相对匹配FP32原推理的Acc50改善，testA/testB分别独立为正；两者mIoU也提高
what_results_dont_support: 训练算法收益、新学习模块收益、论文新颖性、SOTA、跨模型或跨数据集普遍改善
missing_evidence_for_this_narrow_claim: 无须追加性能搜索；模型级复现依赖已归档原检查点、源码和运行环境
next_experiments_needed_for_this_narrow_claim: none
confidence: high
review_independence: same-family
acceptance_status: provisional
integrity_status: unavailable
integrity_note: 单独experiment-audit审阅者未启动成功，协调者取消并记录REVIEW_UNAVAILABLE；不宣称完整audit通过
deterministic_prediction_and_metric_verification: passed
```

### 独立测试复算

直接读取 `final_testA14` 和 `final_testB14` 的原始预测、manifest、协议、结果及统计，读取对应 `official_repo_testA.pth` / `official_repo_testB.pth`，并补读实际集成模块 `utils/rec_refinement.py`。没有执行训练、GPU推理或新增候选。

- 两组manifest按source_index覆盖相应官方仓库划分全部行；image和phrase逐条匹配。用原生整数化框、resize、pad重建GT，最大差均0。
- 所有已保存候选的官方IoU、custom IoU、保存IoU相等；JSON Acc50/Acc75/mIoU误差均0。
- 从原/翻转预测、缓存分数优势、方向跳过和一致性信息重建主规则，testA/testB最大预测差均0，接受数分别949/891，与运行记录一致。
- testA和testB各750图像，彼此无图像交集；与完整val、已读旧screen train、新train/calibration、full_replay train均无图像交集。
- 实际集成模块仅使用图像、padding mask、文本和原模型输出，未接收GT。测试首批真实样本的集成smoke记录误差分别5.96e-8和0；这是16条首批集成一致性检查，不把它夸大为审阅者重跑完整神经网络。

| 指标 | testA | testB |
|---|---:|---:|
| 描述 / 图像 | 5,726 / 750 | 4,889 / 750 |
| 原始正确 → 改进正确 | 4,556 → 4,598 | 3,297 → 3,324 |
| 原始Acc50 | 79.56688788% | 67.43710370% |
| 主规则Acc50 | 80.30038421% | 67.98936388% |
| 纠正 / 破坏 | 72 / 30 | 65 / 38 |
| ΔAcc50 | **+0.733496 pp** | **+0.552260 pp** |
| 独立图像bootstrap95% | **[+0.345110,+1.135961]** | **[+0.104188,+1.013933]** |
| ΔmIoU | +0.843197 pp | +0.532958 pp |
| mIoU 95%区间 | [+0.696169,+0.998373] | [+0.362399,+0.693924] |
| ΔAcc75 | +1.449528 pp | +0.634076 pp |
| Acc75 95%区间 | [+0.813155,+2.081156] | [−0.020892,+1.309960] |

审阅者bootstrap同样为seed20261127、5,000次图像簇抽样。**testB的Acc75区间包含0，不应说所有次指标均达到统计显著。** 主指标Acc50和mIoU的两组正向结果支持本次狭义收益结论；这些区间不代表跨训练种子或跨模型的不确定性。

相对于固定consensus_projection，验证选择步骤在testA额外净增21例（24纠正/3破坏，+0.366748 pp，95% [+0.191091,+0.556912]），testB额外净增17例（18纠正/1破坏，+0.347719 pp，95% [+0.181887,+0.540684]）。这提供比“原图+翻转平均并裁剪”更直接的对照证据，仍不等于学术新颖性判定。

最终test文件也记录其他margin，但主规则始终为0.02；不能把0.0的更高测试数值替换为最终主结果。配对统计文件中沿用的“Val estimates remain exploratory”字符串是模板文字，实际本节评估对象以split、完整manifest和冻结流程为准；val开发性质与本轮冻结测试确认应分别描述。

### 最终允许与不允许的表述

允许：“在现有OneRef检查点上，固定的双视图验证/一致性融合与有效图像裁剪，在匹配FP32评估下将RefCOCO+ testA/testB Acc50分别提高0.7335/0.5523个百分点，图像聚类配对区间均为正；无需新增训练参数，增加一次骨干前向。”

不允许：“已证明新训练算法有效”“发现足够论文创新”“所有OneRef检查点/REC数据集都会提高”“每个样本均不退化”。此前失败的两训练种子也不能作为本确定性推理方案的两次训练复现。

对用户要求的“持续直到找到经验证的提升”，**当前狭义效果目标已经满足，可以停止新增搜索并完成代码、预测、协议和原检查点归档。** 若后续目标变成论文贡献或跨模型泛化，应另立新的研究问题，不能借本次效果确认越过该证据边界。

最终测试预测SHA256：

```text
final_testA14/predictions.pt 867e7dc39533f1793aa7bfbf74beb9d4310008fcdddf8b0e32ec590383926621
final_testB14/predictions.pt b4926b4b9f6f020afea444df05abff002e618b3a0abf6fc97798926f4064cda9
```
