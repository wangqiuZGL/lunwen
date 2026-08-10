##!/bin/bash
#
## ========== 环境配置 ==========
#export CUDA_VISIBLE_DEVICES=0,1
#echo "恢复训练脚本 - 直接从微调阶段继续"
#
## ========== 路径配置 ==========
#DATA_ROOT=/root/shared-nvme/oneref/data/image_data
#SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
#BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
#OUTPUT_DIR=/root/shared-nvme/oneref/output
## ==============================
#
## ========== 句法引导注意力参数 ==========
#ENABLE_SYNTAX_BIAS=true  # 是否启用句法引导注意力
#SYNTAX_BIAS_LAMBDA=0.1   # 句法偏置强度
## =====================================
#
## 根据 ENABLE_SYNTAX_BIAS 设置参数
#SYNTAX_ARGS=""
#if [ "$ENABLE_SYNTAX_BIAS" = true ]; then
#    SYNTAX_ARGS="--enable_syntax_bias --syntax_bias_lambda ${SYNTAX_BIAS_LAMBDA}"
#    echo "启用句法引导注意力，lambda=${SYNTAX_BIAS_LAMBDA}"
#else
#    echo "不启用句法引导注意力"
#fi
#
######### 1. 直接从检查点继续微调训练
#echo "=== 阶段1: 从检查点继续微调训练 ==="
#echo "使用检查点: ${OUTPUT_DIR}/syntax_guided/gref_finetune/checkpoint.pth"
#
## 首先确认检查点是否存在
#if [ ! -f "${OUTPUT_DIR}/syntax_guided/gref_finetune/checkpoint.pth" ]; then
#    echo "错误: checkpoint.pth 不存在！"
#    echo "尝试使用 best_checkpoint.pth..."
#    if [ ! -f "${OUTPUT_DIR}/syntax_guided/gref_finetune/best_checkpoint.pth" ]; then
#        echo "错误: best_checkpoint.pth 也不存在！"
#        exit 1
#    fi
#    # 如果没有checkpoint.pth，只能从best_checkpoint.pth开始，但不能继续训练状态
#    RESUME_ARG="--finetune ${OUTPUT_DIR}/syntax_guided/gref_finetune/best_checkpoint.pth"
#else
#    RESUME_ARG="--resume ${OUTPUT_DIR}/syntax_guided/gref_finetune/checkpoint.pth"
#fi
#
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#  --nproc_per_node=2 --master_port 28888 --use_env train_oneref.py \
#  --num_workers 8 --epochs 30 --batch_size 22 --lr 0.00003 --lr_scheduler cosine \
#  --aug_crop --aug_scale --aug_translate \
#  --imsize 384 --max_query_len 64 \
#  --model beit3_base_patch16_384 --task grounding \
#  --dataset gref_umd --use_regress_box --use_box_mask_constraints \
#  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
#  ${RESUME_ARG} \
#  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
#  --output_dir ${OUTPUT_DIR}/syntax_guided/gref_finetune_continued \
#  ${SYNTAX_ARGS}
#
######### 2. 评估（RefCOCOg / gref_umd / val）
#echo "=== 阶段2: 模型评估 (val) ==="
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#  --nproc_per_node=2 --master_port 28889 --use_env eval.py \
#  --num_workers 4 --batch_size 64 \
#  --imsize 384 --max_query_len 64 \
#  --model beit3_base_patch16_384 --task grounding \
#  --dataset gref_umd --eval_set val \
#  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
#  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
#  --eval_model ${OUTPUT_DIR}/syntax_guided/gref_finetune_continued/best_checkpoint.pth \
#  --output_dir ${OUTPUT_DIR}/syntax_guided/gref_eval_continued  \
#  ${SYNTAX_ARGS}
#
#echo "恢复训练完成！结果保存在 ${OUTPUT_DIR}/syntax_guided/gref_finetune_continued"


#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output

# 继续训练10轮
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28897 --use_env train_oneref.py \
  --num_workers 8 --epochs 40 --batch_size 22 --lr 0.00003 --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset gref_umd --use_regress_box --use_box_mask_constraints \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --resume ${OUTPUT_DIR}/syntax_guided/gref_finetune_continued/checkpoint.pth \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --output_dir ${OUTPUT_DIR}/syntax_guided/gref_finetune_continued2 \
  --enable_syntax_bias --syntax_bias_lambda 0.2

echo "训练完成！"