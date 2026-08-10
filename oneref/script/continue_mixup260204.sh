#!/bin/bash

# ========== 环境配置 ==========
export CUDA_VISIBLE_DEVICES=0,1
echo "train_rec_mixup_grounding_pretraining_base.sh (with syntax-guided attention)"

# ========== 路径配置 ==========
DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output
# ==============================

# ========== 句法引导注意力参数 ==========
ENABLE_SYNTAX_BIAS=true  # 是否启用句法引导注意力
SYNTAX_BIAS_LAMBDA=0.1   # 句法偏置强度（建议值：0.05, 0.1, 0.2, 0.5）
# =====================================

# 创建输出目录（用于保存继续训练的结果）
CONTINUE_OUTPUT_DIR=${OUTPUT_DIR}/mixup_pretrain/mixup_finetune_continue
mkdir -p ${CONTINUE_OUTPUT_DIR}

# 根据 ENABLE_SYNTAX_BIAS 设置参数
SYNTAX_ARGS=""
if [ "$ENABLE_SYNTAX_BIAS" = true ]; then
    SYNTAX_ARGS="--enable_syntax_bias --syntax_bias_lambda ${SYNTAX_BIAS_LAMBDA}"
    echo "启用句法引导注意力，lambda=${SYNTAX_BIAS_LAMBDA}"
else
    echo "不启用句法引导注意力"
fi

######## 继续训练（RefC Mixup）—— 基于已训练的模型继续训练20轮
#echo "=== 继续训练：基于已训练的模型继续训练20个epoch ==="
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#  --nproc_per_node=2 --master_port 28897 --use_env train_oneref.py \
#  --num_workers 8 --epochs 40 --batch_size 22 --lr 0.00003 --lr_scheduler cosine \
#  --aug_crop --aug_scale --aug_translate \
#  --imsize 384 --max_query_len 64 \
#  --model beit3_base_patch16_384 --task grounding \
#  --dataset mixup --use_regress_box --use_box_mask_constraints \
#  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
#  --resume ${OUTPUT_DIR}/mixup_pretrain/mixup_finetune/best_checkpoint.pth \
#  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/mixup_with_refc \
#  --output_dir ${CONTINUE_OUTPUT_DIR} \
#
#  ${SYNTAX_ARGS}

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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
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
  --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth  \
  --output_dir ${CONTINUE_OUTPUT_DIR}/eval1  \
  ${SYNTAX_ARGS}

echo "RefC Mixup 预训练和评估完成！"
echo "最终模型保存在: ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth "
echo "评估结果保存在: ${CONTINUE_OUTPUT_DIR}/eval /"

