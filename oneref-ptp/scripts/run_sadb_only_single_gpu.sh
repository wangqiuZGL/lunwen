#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate oneref
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ASSETS="${ASSETS:-/root/lunwen-data}"
RUN_ROOT="${RUN_ROOT:-/hy-tmp/lunwen-runs}"
MODE="${1:-train}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/sadb_only_${MODE}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
echo "RUN_DIR=$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git diff > "$RUN_DIR/code_changes.patch"
python scripts/snapshot_source.py "$RUN_DIR/source_snapshot.zip"
python -m pip freeze > "$RUN_DIR/environment.txt"
SPLITS="$ASSETS/subsets/refcocoplus20_seed42/single_dataset"
python scripts/prepare_rec_subset.py --source "$ASSETS/ref_data_shuffled/single_dataset" \
  --output "$SPLITS" --fraction 0.2 --seed 42
cp "$SPLITS/manifest.json" "$RUN_DIR/subset_manifest.json"
COMMON=(--num_workers "${NUM_WORKERS:-4}" --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding --dataset unc+ --use_regress_box \
  --sentencepiece_model "$ASSETS/beit3_checkpoints/beit3.spm" \
  --data_root "$ASSETS/image_data" --split_root "$SPLITS" \
  --enable_syntax_bias --syntax_bias_lambda 0.1 --seed 42 --amp)
INITIAL="$ASSETS/beit3_checkpoints/beit3_base_indomain_patch16_224.pth"
if [[ "$MODE" == smoke ]]; then
  echo "Use train mode for SADB-only ablation; fused-only smoke assertions are inapplicable." >&2; exit 2
  python scripts/smoke_fused.py "${COMMON[@]}" --epochs 1 --batch_size "${BATCH_SIZE:-2}" \
    --update_freq "${UPDATE_FREQ:-2}" --lr 0.00003 --use_box_mask_constraints \
    --limit_train_samples "${SMOKE_SAMPLES:-8}" --limit_val_samples 16 \
    --finetune "$INITIAL" --output_dir "$RUN_DIR/smoke"
elif [[ "$MODE" == train ]]; then
  # Exploratory schedule; use identical settings for later ablations.
  python train_oneref.py "${COMMON[@]}" --epochs "${WARMUP_EPOCHS:-2}" \
    --batch_size "${WARMUP_BATCH:-8}" --update_freq 4 --lr 0.00025 \
    --frozen_backbone --finetune "$INITIAL" --output_dir "$RUN_DIR/warmup"
  WARMUP="$RUN_DIR/warmup/best_checkpoint.pth"
  [[ -f "$WARMUP" ]] || WARMUP="$RUN_DIR/warmup/checkpoint.pth"
  python train_oneref.py "${COMMON[@]}" --epochs "${FINETUNE_EPOCHS:-8}" \
    --batch_size "${BATCH_SIZE:-16}" --update_freq "${UPDATE_FREQ:-2}" --lr 0.00003 \
    --use_box_mask_constraints --finetune "$WARMUP" --output_dir "$RUN_DIR/finetune"
else
  echo 'Usage: bash scripts/run_sadb_only_single_gpu.sh [smoke|train]' >&2
  exit 2
fi
echo "COMPLETED: $RUN_DIR"
