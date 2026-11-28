# 相对改善质量筛选的双视图蒸馏：候选与首轮实验

日期2026-11-28。目标：将已验证的双前向教师优势转移给单原图前向学生，减少推理成本。创新候选是依赖训练标注的相对改善质量筛选蒸馏，不是首次使用一致性、增强或知识蒸馏；尚未通过完整查新。

相关近邻：SelfEQ(CVPR2024) https://openaccess.thecvf.com/content/CVPR2024/papers/He_Improved_Visual_Grounding_through_Self-Consistent_Explanations_CVPR_2024_paper.pdf 研究文本改写与解释图一致性；Language–Image Consistency Augmentation and Distillation Network for visual grounding https://www.sciencedirect.com/science/article/pii/S0031320325003231 有Consistency-Aware Self-Distillation，需全文对比后才能主张本候选的新颖性。

固定教师：原检查点+已冻结verified_consensus_0.02。训练官方train内512条，原unseen04 calibration划分改作本轮学生训练；与探索val256图像互斥。教师输出已生成，复用缓存避免重复骨干计算。对照原生继续训练、全部教师蒸馏、质量筛选蒸馏；相同原模型初始化、同批次序列，种子42/43，每组256步、batch32、头lr1e-5。骨干冻结，本轮是机制初筛，不是完整端到端训练。

候选loss：原one_ref_loss + 2 * batch_mean(gate * [L1_sum(student,teacher)+1-GIoU(student,teacher)])。gate=训练teacher真实IoU>=0.5且比原预测真实IoU提高>0.01。gate与教师均固定，无学生反馈自选标签；门控只用于训练，学生推理不读取GT、不执行翻转、无额外参数。all_teacher对照令gate=1。gate平均按全batch计算，明确保留覆盖率带来的有效loss权重差异，后续若有信号需补匹配有效权重对照。

验收：固定最终步，所有结果报告，不在testA/B调参。首轮只看候选是否超过原模型及两个对照，不以单个样本判成功。正向则扩大开发验证并开展原图单次前向的端到端公平对照；负向只否定本冻结尾部短训配置。每轮限时600秒，复用已验证3090环境。


首轮distill15已完成24.25秒，146/512接受教师监督，native两种子74.6094，all_teacher均75.3906，quality_gated均74.6094，原模型75.78125；该冻结头短训未提升。已备份oss://datasets/experiments/layerwise_20261127/distill15_records.tar.gz。

第二轮distill_anchor16仅补跑anchored_quality两种子：训练目标reference=gate*verified_teacher+(1-gate)*original_prediction，所有样本KD权重一致，控制原生训练导致的旧预测漂移。其余数据、初始化、批次、lr和256步全部同distill15，因此复用原native/all_teacher对照。仍为512/256缓存机制初筛，不能当作端到端或论文主结果。
