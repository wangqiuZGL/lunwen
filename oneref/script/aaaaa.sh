#!/bin/bash
set -e

# ========== 环境配置 ==========
export CUDA_VISIBLE_DEVICES=0,1
echo "=== Mixup Grounding Fine-tuning (Low LR, Stable) ==="

# ========== 路径配置 ==========
DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output

BASE_CKPT=${OUTPUT_DIR}/mixup_pretrain/mixup_finetune/best_checkpoint.pth
CONTINUE_OUTPUT_DIR=${OUTPUT_DIR}/mixup_pretrain/mixup_finetune_lowlr
mkdir -p ${CONTINUE_OUTPUT_DIR}

# ========== 句法引导注意力 ==========
ENABLE_SYNTAX_BIAS=true
SYNTAX_BIAS_LAMBDA=0.1

SYNTAX_ARGS=""
if [ "$ENABLE_SYNTAX_BIAS" = true ]; then
    SYNTAX_ARGS="--enable_syntax_bias --syntax_bias_lambda ${SYNTAX_BIAS_LAMBDA}"
    echo "✓ Syntax bias enabled (lambda=${SYNTAX_BIAS_LAMBDA})"
else
    echo "✗ Syntax bias disabled"
fi

# =====================================================
# 1️⃣ 低学习率继续训练（20 epoch，稳定精修）
# =====================================================
echo "=== Stage 1: Low-LR Fine-tuning (20 epochs) ==="

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28901 --use_env train_oneref.py \
  --num_workers 8 \
  --epochs 20 \
  --batch_size 22 \
  --lr 1e-6 \
  --lr_scheduler cosine \
  --weight_decay 1e-4 \
  --clip_max_norm 0.0 \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 \
  --max_query_len 64 \
  --model beit3_base_patch16_384 \
  --task grounding \
  --dataset mixup \
  --use_regress_box \
  --use_box_mask_constraints \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --resume ${BASE_CKPT} \
  --data_root ${DATA_ROOT} \
  --split_root ${SPLIT_ROOT}/mixup_with_refc \
  --output_dir ${CONTINUE_OUTPUT_DIR} \
  ${SYNTAX_ARGS}

# =====================================================
# 2️⃣ 评估：RefCOCO / RefCOCO+ / RefCOCOg
# =====================================================
echo "=== Stage 2: Evaluation ==="

run_eval () {
  DATASET=$1
  SPLIT=$2
  BS=$3
  PORT=$4

  echo "--- Eval ${DATASET} | ${SPLIT} ---"
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port ${PORT} --use_env eval.py \
    --num_workers 4 \
    --batch_size ${BS} \
    --imsize 384 \
    --max_query_len 64 \
    --model beit3_base_patch16_384 \
    --task grounding \
    --dataset ${DATASET} \
    --eval_set ${SPLIT} \
    --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
    --data_root ${DATA_ROOT} \
    --split_root ${SPLIT_ROOT}/single_dataset \
    --eval_model ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth \
    --output_dir ${CONTINUE_OUTPUT_DIR}/eval \
    ${SYNTAX_ARGS}
}

# RefCOCO
run_eval unc val   64 28910
run_eval unc testA 32 28911
run_eval unc testB 32 28912

# RefCOCO+
run_eval unc+ val   64 28913
run_eval unc+ testA 32 28914
run_eval unc+ testB 32 28915

# RefCOCOg
run_eval gref_umd val  64 28916
run_eval gref_umd test 32 28917

echo "=== Done ==="
echo "✓ Final checkpoint: ${CONTINUE_OUTPUT_DIR}/best_checkpoint.pth"
echo "✓ Eval results:     ${CONTINUE_OUTPUT_DIR}/eval/"