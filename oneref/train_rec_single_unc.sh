#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
echo "train_rec_single_dataset_finetuning_base_unc.sh"

# ========== 路径配置 ==========
DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output
# ==============================

######## 1. Warmup 训练（RefCOCO / unc）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28887 --use_env train_oneref.py \
  --num_workers 4 --epochs 10 --batch_size 64 --lr 0.00025 --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc --use_regress_box --frozen_backbone \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --finetune ${BEIT3_CHECKPOINT}/beit3_base_indomain_patch16_224.pth \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --output_dir ${OUTPUT_DIR}/v001/unc

######## 2. Finetune 训练（RefCOCO / unc）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28888 --use_env train_oneref.py \
  --num_workers 4 --epochs 20 --batch_size 22 --lr 0.00003 --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc --use_regress_box --use_box_mask_constraints \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --finetune ${OUTPUT_DIR}/v001/unc/best_checkpoint.pth \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --output_dir ${OUTPUT_DIR}/v002/unc

######## 3. 简单评估（RefCOCO / unc / val）
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28889 --use_env eval.py \
  --num_workers 4 --batch_size 60 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset unc --eval_set val \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/v002/unc/best_checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/v002/unc


