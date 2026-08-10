#!/bin/bash

# ========== 环境配置 ==========
export CUDA_VISIBLE_DEVICES=0,1
echo "train_recoco+_with_ptp_only.sh (仅PTP辅助监督，不使用句法模块)"

# ========== 路径配置 ==========
DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output_test
# ==============================

# ========== PTP辅助监督参数 ==========
ENABLE_PTP_AUX=true      # 是否启用PTP辅助监督
PTP_GRID_SIZE=16         # Grid大小（16表示16x16=256个网格）
PTP_LOSS_WEIGHT=0.1      # PTP Loss权重（建议值：0.05, 0.1, 0.2）
# =====================================

# 创建输出目录
mkdir -p ${OUTPUT_DIR}/ptp_only

# 根据 ENABLE_PTP_AUX 设置参数
PTP_ARGS=""
if [ "$ENABLE_PTP_AUX" = true ]; then
    PTP_ARGS="--enable_ptp_aux --ptp_grid_size ${PTP_GRID_SIZE} --ptp_loss_weight ${PTP_LOSS_WEIGHT}"
    echo "启用PTP辅助监督，grid_size=${PTP_GRID_SIZE}，loss_weight=${PTP_LOSS_WEIGHT}"
else
    echo "不启用PTP辅助监督"
fi

######## 1. Warmup 训练（RefCOCO+ / unc+）—— 提高 batch_size 提高显存利用
echo "=== 阶段1: Warmup 训练（带PTP辅助监督）==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28887 --use_env train_oneref.py \
  --num_workers 8 --epochs 1 --batch_size 64 --lr 0.00025 --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc+ --use_regress_box --frozen_backbone \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --finetune ${BEIT3_CHECKPOINT}/beit3_base_indomain_patch16_224.pth \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --output_dir ${OUTPUT_DIR}/ptp_only/recoco+_warmup  \
  ${PTP_ARGS}

######## 2. Finetune 训练（RefCOCO+ / unc+）—— 使用PTP辅助监督
echo "=== 阶段2: Finetune 训练（带PTP辅助监督）==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28888 --use_env train_oneref.py \
  --num_workers 8 --epochs 1 --batch_size 22 --lr 0.00003 --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc+ --use_regress_box --use_box_mask_constraints \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --finetune ${OUTPUT_DIR}/ptp_only/recoco+_warmup/best_checkpoint.pth \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --output_dir ${OUTPUT_DIR}/ptp_only/recoco+_finetune \
  ${PTP_ARGS}

######## 3. 评估（RefCOCO+ / unc+ / val）
echo "=== 阶段3: 模型评估 (val) ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28889 --use_env eval.py \
  --num_workers 4 --batch_size 64 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc+ --eval_set val \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/ptp_only/recoco+_finetune/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/ptp_only/recoco+_eval

######### 4. 评估（RefCOCO+ / unc+ / testA）
#echo "=== 阶段4: 模型评估 (testA) ==="
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#  --nproc_per_node=2 --master_port 28890 --use_env eval.py \
#  --num_workers 4 --batch_size 32 \
#  --imsize 384 --max_query_len 64 \
#  --model beit3_base_patch16_384 --task grounding \
#  --dataset unc+ --eval_set testA \
#  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
#  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
#  --eval_model ${OUTPUT_DIR}/ptp_only/recoco+_finetune/best_checkpoint.pth \
#  --output_dir ${OUTPUT_DIR}/ptp_only/recoco+_eval
#
######### 5. 评估（RefCOCO+ / unc+ / testB）
#echo "=== 阶段5: 模型评估 (testB) ==="
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#  --nproc_per_node=2 --master_port 28891 --use_env eval.py \
#  --num_workers 4 --batch_size 32 \
#  --imsize 384 --max_query_len 64 \
#  --model beit3_base_patch16_384 --task grounding \
#  --dataset unc+ --eval_set testB \
#  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
#  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
#  --eval_model ${OUTPUT_DIR}/ptp_only/recoco+_finetune/best_checkpoint.pth \
#  --output_dir ${OUTPUT_DIR}/ptp_only/recoco+_eval
#
#echo "RefCOCO+数据集（仅PTP辅助监督）训练和评估完成！结果保存在 ${OUTPUT_DIR}/ptp_only"

