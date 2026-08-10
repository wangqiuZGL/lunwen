#!/bin/bash

# ========== 环境配置 ==========
CUDA_VISIBLE_DEVICES=0
echo "evaluate_gref_splits.sh"

# ========== 路径配置 ==========
DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output
# ==============================

# ========== 句法引导注意力参数 ==========
ENABLE_SYNTAX_BIAS=true  # 是否启用句法引导注意力
SYNTAX_BIAS_LAMBDA=0.1   # 句法偏置强度（与训练时保持一致）
# =====================================

# 根据 ENABLE_SYNTAX_BIAS 设置参数
SYNTAX_ARGS=""
if [ "$ENABLE_SYNTAX_BIAS" = true ]; then
    SYNTAX_ARGS="--enable_syntax_bias --syntax_bias_lambda ${SYNTAX_BIAS_LAMBDA}"
    echo "启用句法引导注意力，lambda=${SYNTAX_BIAS_LAMBDA}"
else
    echo "不启用句法引导注意力"
fi

# 评估模型路径
eval_model="${OUTPUT_DIR}/syntax_guided/gref_finetune/best_checkpoint.pth"

# 1. 评估 val 集
echo "=== 开始评估 val 集 ==="
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 28889 --use_env eval.py \
  --num_workers 4 --batch_size 64 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset gref_umd --eval_set val \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${eval_model} \
  --output_dir ${OUTPUT_DIR}/syntax_guided/gref_eval \
  ${SYNTAX_ARGS}

# 2. 评估 test 集（RefCOCOg 通常只有一个 test 集）(降低batch_size和num_workers以减少内存占用)
echo "=== 开始评估 test 集 ==="
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 28890 --use_env eval.py \
  --num_workers 2 --batch_size 32 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset gref_umd --eval_set test \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${eval_model} \
  --output_dir ${OUTPUT_DIR}/syntax_guided/gref_eval \
  ${SYNTAX_ARGS}

echo "RefCOCOg数据集所有评估完成！结果保存在 ${OUTPUT_DIR}/syntax_guided/gref_eval"
