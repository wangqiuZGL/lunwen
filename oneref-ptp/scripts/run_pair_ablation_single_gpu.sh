#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate oneref
VARIANT="${1:-}"
case "$VARIANT" in
  A) PAIR_ARGS=() ;;
  B) PAIR_ARGS=(--enable_pair_sampling --pair_loss_weight 0) ;;
  C) PAIR_ARGS=(--enable_pair_sampling --pair_loss_weight "${PAIR_WEIGHT:-0.1}") ;;
  *) echo 'Usage: bash scripts/run_pair_ablation_single_gpu.sh A|B|C' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
ASSETS="${ASSETS:-/root/lunwen-data}"
RUN_ROOT="${RUN_ROOT:-/hy-tmp/lunwen-runs}"
SEED="${SEED:-42}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/pair_${VARIANT}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
# Refuse accidental overwrite/resume. Use train_oneref.py --resume explicitly.
if [[ -d "$RUN_DIR" ]] && [[ -n "$(ls -A "$RUN_DIR")" ]]; then
  echo "RUN_DIR must be empty: $RUN_DIR" >&2; exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
echo "RUN_DIR=$RUN_DIR VARIANT=$VARIANT SEED=$SEED"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git diff > "$RUN_DIR/code_changes.patch"
python scripts/snapshot_source.py "$RUN_DIR/source_snapshot.zip"
python -m pip freeze > "$RUN_DIR/environment.txt"
cp "$0" "$RUN_DIR/launcher.sh"
SPLITS="$ASSETS/subsets/refcocoplus20_seed42/single_dataset"
# Subset seed stays 42 even when training seed changes.
python scripts/prepare_rec_subset.py --source "$ASSETS/ref_data_shuffled/single_dataset" \
  --output "$SPLITS" --fraction 0.2 --seed 42
cp "$SPLITS/manifest.json" "$RUN_DIR/subset_manifest.json"
BATCH_SIZE="${BATCH_SIZE:-16}"
PAIR_FRACTION="${PAIR_FRACTION:-0.5}"
PAIR_MARGIN="${PAIR_MARGIN:-0.1}"
PAIR_MAX_IOU="${PAIR_MAX_IOU:-0.5}"
python scripts/audit_pairs.py --train-cache "$SPLITS/unc+/unc+_train.pth" \
  --output "$RUN_DIR/pair_audit" --batch-size "$BATCH_SIZE" --seed "$SEED" \
  --pair-fraction "$PAIR_FRACTION" --max-iou "$PAIR_MAX_IOU" \
  --image-root "$ASSETS/image_data/other/images/mscoco/images/train2014"
COMMON=(--num_workers "${NUM_WORKERS:-4}" --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 --task grounding --dataset unc+ --use_regress_box \
  --sentencepiece_model "$ASSETS/beit3_checkpoints/beit3.spm" \
  --data_root "$ASSETS/image_data" --split_root "$SPLITS" --seed "$SEED" --amp \
  --export_val_predictions)
INITIAL="$ASSETS/beit3_checkpoints/beit3_base_indomain_patch16_224.pth"
# Identical original warmup in all groups; pairing applies only to finetuning.
python train_oneref.py "${COMMON[@]}" --epochs "${WARMUP_EPOCHS:-2}" \
  --batch_size "${WARMUP_BATCH:-8}" --update_freq 4 --lr 0.00025 \
  --frozen_backbone --finetune "$INITIAL" --output_dir "$RUN_DIR/warmup"
WARMUP="$RUN_DIR/warmup/best_checkpoint.pth"
[[ -f "$WARMUP" ]] || WARMUP="$RUN_DIR/warmup/checkpoint.pth"
python train_oneref.py "${COMMON[@]}" "${PAIR_ARGS[@]}" \
  --pair_fraction "$PAIR_FRACTION" --pair_margin "$PAIR_MARGIN" --pair_max_iou "$PAIR_MAX_IOU" \
  --pair_min_patches "${PAIR_MIN_PATCHES:-1}" --epochs "${FINETUNE_EPOCHS:-8}" \
  --batch_size "$BATCH_SIZE" --update_freq "${UPDATE_FREQ:-2}" --lr 0.00003 \
  --use_box_mask_constraints --finetune "$WARMUP" --output_dir "$RUN_DIR/finetune"
python scripts/collect_results.py "$RUN_DIR" --output "$RUN_DIR/summary"
echo 'COMPLETED: verify checkpoint selection, predictions and diagnostics before claiming improvement.'
