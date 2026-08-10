#!/bin/bash

# ========== 环境配置 ==========
# MRefM 预训练脚本（结合句法引导注意力）
# 使用 pretrain_oneref_with_mrefm.py 进行 MRefM 预训练
export CUDA_VISIBLE_DEVICES=0,1
echo "train_rec_mrefm_pretraining_base.sh (MRefM + Syntax-Guided Attention)"

# ========== 路径配置 ==========
DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output2B
# ==============================

# ========== 句法引导注意力参数 ==========
ENABLE_SYNTAX_BIAS=true  # 是否启用句法引导注意力
SYNTAX_BIAS_LAMBDA=0.1   # 句法偏置强度（建议值：0.05, 0.1, 0.2, 0.5）
# =====================================

# 创建输出目录
mkdir -p ${OUTPUT_DIR}/mrefm_pretrain

# 根据 ENABLE_SYNTAX_BIAS 设置参数
SYNTAX_ARGS=""
if [ "$ENABLE_SYNTAX_BIAS" = true ]; then
    SYNTAX_ARGS="--enable_syntax_bias --syntax_bias_lambda ${SYNTAX_BIAS_LAMBDA}"
    echo "启用句法引导注意力，lambda=${SYNTAX_BIAS_LAMBDA}"
else
    echo "不启用句法引导注意力"
fi

# MRefM 参数
MREFM_ARGS="--enable_mrefm --enable_dynamic_mim --dynamic_mask_ratio 0.75"

######### 1. Warmup 预训练（RefC Mixup + ReferIt，MRefM + 句法）—— 冻结 backbone
#echo "=== 阶段1: Warmup 预训练（冻结 backbone，10 epochs，MRefM + 句法引导注意力）==="
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#  --nproc_per_node=2 --master_port 28887 --use_env pretrain_oneref_with_mrefm.py \
#  --num_workers 8 --epochs 10 --batch_size 64 --lr 0.00025 --lr_scheduler cosine \
#  --aug_crop --aug_scale --aug_translate \
#  --imsize 384 --max_query_len 64 \
#  --model beit3_base_patch16_384 --task grounding \
#  --dataset mixup --frozen_backbone \
#  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
#  --tokenizer_weight ${BEIT3_CHECKPOINT}/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth \
#  --finetune ${BEIT3_CHECKPOINT}/beit3_base_indomain_patch16_224.pth \
#  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/mixup_with_refc_referit \
#  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_warmup \
#  ${MREFM_ARGS} ${SYNTAX_ARGS}

######## 2. Finetune 预训练（RefC Mixup + ReferIt，MRefM + 句法）—— 解冻所有层
echo "=== 阶段2: Finetune 预训练（解冻所有层，100 epochs，MRefM + 句法引导注意力）==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28888 --use_env pretrain_oneref_with_mrefm.py \
  --num_workers 8 --epochs 100 --batch_size 16 --lr 0.00005 --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset mixup \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --tokenizer_weight ${BEIT3_CHECKPOINT}/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth \
  --finetune ${OUTPUT_DIR}/mrefm_pretrain/mrefm_warmup/checkpoint.pth \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/mixup_with_refc_referit \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune \
  ${MREFM_ARGS} ${SYNTAX_ARGS}

######## 3. 评估（RefCOCO / unc / val）
echo "=== 阶段3: 模型评估 (RefCOCO val) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28889 --use_env eval.py \
  --num_workers 4 --batch_size 64 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc --eval_set val \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

######## 4. 评估（RefCOCO / unc / testA）
echo "=== 阶段4: 模型评估 (RefCOCO testA) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28890 --use_env eval.py \
  --num_workers 4 --batch_size 32 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc --eval_set testA \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

######## 5. 评估（RefCOCO / unc / testB）
echo "=== 阶段5: 模型评估 (RefCOCO testB) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28891 --use_env eval.py \
  --num_workers 4 --batch_size 32 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc --eval_set testB \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

######## 6. 评估（RefCOCO+ / unc+ / val）
echo "=== 阶段6: 模型评估 (RefCOCO+ val) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28892 --use_env eval.py \
  --num_workers 4 --batch_size 64 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc+ --eval_set val \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

######## 7. 评估（RefCOCO+ / unc+ / testA）
echo "=== 阶段7: 模型评估 (RefCOCO+ testA) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28893 --use_env eval.py \
  --num_workers 4 --batch_size 32 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc+ --eval_set testA \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

######## 8. 评估（RefCOCO+ / unc+ / testB）
echo "=== 阶段8: 模型评估 (RefCOCO+ testB) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28894 --use_env eval.py \
  --num_workers 4 --batch_size 32 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc+ --eval_set testB \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

######## 9. 评估（RefCOCOg / gref_umd / val）
echo "=== 阶段9: 模型评估 (RefCOCOg val) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28895 --use_env eval.py \
  --num_workers 4 --batch_size 64 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset gref_umd --eval_set val \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

######## 10. 评估（RefCOCOg / gref_umd / test）
echo "=== 阶段10: 模型评估 (RefCOCOg test) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28896 --use_env eval.py \
  --num_workers 4 --batch_size 32 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset gref_umd --eval_set test \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval \
  ${SYNTAX_ARGS}

echo "=========================================="
echo "RefC Mixup MRefM 预训练和评估完成！"
echo "最终模型保存在: ${OUTPUT_DIR}/mrefm_pretrain/mrefm_finetune/best_checkpoint.pth"
echo "评估结果保存在: ${OUTPUT_DIR}/mrefm_pretrain/mrefm_eval/"
echo "=========================================="

