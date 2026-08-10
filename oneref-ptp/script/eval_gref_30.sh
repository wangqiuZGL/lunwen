#!/bin/bash

# ========== 环境配置 ==========
export CUDA_VISIBLE_DEVICES=0,1
echo "评估test集合"

# ========== 路径配置 ==========
DATA_ROOT=/root/shared-nvme/oneref/data/image_data
SPLIT_ROOT=/root/shared-nvme/oneref/data/ref_data_shuffled
BEIT3_CHECKPOINT=/root/shared-nvme/oneref/data/beit3_checkpoint
OUTPUT_DIR=/root/shared-nvme/oneref/output
# ==============================

# ========== 句法引导注意力参数 ==========
ENABLE_SYNTAX_BIAS=true  # 是否启用句法引导注意力
SYNTAX_BIAS_LAMBDA=0.1   # 句法偏置强度
# =====================================

# 根据 ENABLE_SYNTAX_BIAS 设置参数
SYNTAX_ARGS=""
if [ "$ENABLE_SYNTAX_BIAS" = true ]; then
    SYNTAX_ARGS="--enable_syntax_bias --syntax_bias_lambda ${SYNTAX_BIAS_LAMBDA}"
    echo "启用句法引导注意力，lambda=${SYNTAX_BIAS_LAMBDA}"
else
    echo "不启用句法引导注意力"
fi


######## 2. 评估继续训练后模型的test集合
echo "=== 评估继续训练后模型的test集合 ==="
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port 28892 --use_env eval.py \
  --num_workers 4 --batch_size 64 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding \
  --dataset gref_umd --eval_set val \
  --sentencepiece_model ${BEIT3_CHECKPOINT}/beit3.spm \
  --data_root ${DATA_ROOT} --split_root ${SPLIT_ROOT}/single_dataset \
  --eval_model ${OUTPUT_DIR}/syntax_guided/gref_finetune_continued2/checkpoint.pth \
  --output_dir ${OUTPUT_DIR}/syntax_guided/gref_finetune_continued2 \
  ${SYNTAX_ARGS}

echo "test集合评估完成！"