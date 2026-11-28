# 2026-11-29持续改进

用户授权：继续直到明显提升，可开始10轮冻结+20轮微调的基线/候选公平比较。已有goal active，不要因当前小收益就标记完成。

原GPU在线且空闲后已启动完整val评价：/hy-tmp/layerwise-20261127/confirm_pair_full.py --full，输出pair_full20/与pair_full20.log。评价原检查点及paired_distill19的native/anchored_quality各42/43，共5次10758样本。此进程需先完成，不同时抢GPU。脚本已按1024step权重与正确arms改写，保存完整raw预测。

旧启动exec本地session11871因SSH背景stdout仍等待，但远程日志确实前进，不能因此重复提交。先检查真实python进程、日志与results.json。ZIP初始化元数据辅助命令的shell引号有误（NameError beit3），不影响模型训练/完整val启动；以后用上传小Python文件读取，勿反复shell转义。

10+20长训尚未启动，必须如实说明。计划见LONG_TRAINING_PLAN.json。准备新长期入口：先10epoch冻结骨干，再20epoch全编码器微调，两组同初始任务检查点、同数据/顺序/优化预算；不从已适配学生开始。初次seed42两组，若确实正向再seed43。按4096×30轮/16，每组7680updates，不能称作前一1024step短训。冻结模型已任务训练，故明确属于续训不是从预训练重现论文。

缓存教师由确定性384原图生成。若长训引入增强，不能拿原教师框直接配随机裁剪/平移图。安全初版用确定性同坐标；更完整增强可只用无裁剪无翻转的scale/pad，并根据已知几何参数同步映射教师/原锚框，逐批断言。真实GT只在训练用于质量/几何，不在推理选框。

恢复点需保存模型/optimizer/stage/epoch/RNG；磁盘余量应先检查，长训两组checkpoint较大。可保留best/last并按已确认OSS成功管理本程序产生的过期恢复文件，不删除用户旧数据。旧run与源码快照不要覆盖。

Github分支research/consensus-distillation-20261129，最新已推送533941f；已有报告明确小样本区间跨0，不能夸大。新完整val与长训配置尚待追加GitHub。当前本地根目录SESSION_HANDOFF.md也需追加新goal状态，避免沿用GPU空闲旧结论。


最新：tools/train_long_consensus.py已上传且py_compile通过。run_long21.sh已启动，等待真实full-val进程退出且results.status=completed，再执行long_smoke21（冻结/全解冻各2步），通过后依次long_native21_seed42、long_anchored_quality21_seed42。每组10+20真实30epochs，各7680updates；每轮2048开发评价；最终及best分别保存；测试不参与。

当前smoke及长训尚未开始，必须以long21.status和实际进程确认。使用确定性与缓存教师同坐标变换，BF16train/FP32loss_eval；冻结头lr3e-5，微调encoder3e-6/head1e-5，阶段cosine至0.1lr。每轮原子保存模型、optimizer、RNG、stage和epoch，支持--resume。成功OSS上传后才清理本程序的已完成last_training恢复点及临时tar，best/final留在本机；已归档恢复点可从OSS恢复。

恢复只在整轮边界，完成run重复--resume不会重跑。代码需要继续检查smoke与存储状态，goal尚未达到。
