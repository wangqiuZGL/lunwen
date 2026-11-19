#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate oneref
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ASSETS="${ASSETS:-/root/lunwen-data}"
RUN_DIR="${RUN_DIR:-/hy-tmp/lunwen-runs/querycomp_$(date +%Y%m%d_%H%M%S)}"
OSS_PREFIX="${OSS_PREFIX:-oss://datasets/experiments/$(basename "$RUN_DIR")}"
if [[ -d "$RUN_DIR" ]] && [[ -n "$(ls -A "$RUN_DIR")" ]]; then
  echo "Refusing to overwrite $RUN_DIR" >&2; exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/console.log") 2>&1
echo "RUN_DIR=$RUN_DIR OSS_PREFIX=$OSS_PREFIX"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git diff > "$RUN_DIR/code_changes.patch"
python scripts/snapshot_source.py "$RUN_DIR/source_snapshot.zip"
python -m pip freeze > "$RUN_DIR/environment.txt"
cp "$0" "$RUN_DIR/launcher.sh"
SPLITS="$ASSETS/subsets/refcocoplus20_seed42/single_dataset"
python scripts/prepare_rec_subset.py --source "$ASSETS/ref_data_shuffled/single_dataset" \
  --output "$SPLITS" --fraction .2 --seed 42
cp "$SPLITS/manifest.json" "$RUN_DIR/subset_manifest.json"
COMMON=(--num_workers 4 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 --model beit3_base_patch16_384 --task grounding \
  --dataset unc+ --use_regress_box --sentencepiece_model "$ASSETS/beit3_checkpoints/beit3.spm" \
  --data_root "$ASSETS/image_data" --split_root "$SPLITS" --seed 42 --amp)
PAIR=(--enable_pair_sampling --pair_loss_type query_competition --pair_temperature .1 \
  --pair_fraction .5 --pair_max_iou .5 --pair_min_patches 1)
INITIAL="$ASSETS/beit3_checkpoints/beit3_base_indomain_patch16_224.pth"
python scripts/probe_query_competition.py "${COMMON[@]}" "${PAIR[@]}" --pair_loss_weight .1 \
  --lr .00003 --use_box_mask_constraints --finetune "$INITIAL" --output_dir "$RUN_DIR/probe" --probe_steps 150
python scripts/smoke_query_competition.py "${COMMON[@]}" "${PAIR[@]}" --pair_loss_weight .1 \
  --epochs 1 --batch_size 16 --update_freq 2 --lr .00003 --use_box_mask_constraints \
  --finetune "$INITIAL" --output_dir "$RUN_DIR/smoke" --oss_backup_prefix "$OSS_PREFIX/smoke"
echo 'ALL_PREFLIGHT_GATES_PASSED; STARTING_FORMAL_TRAINING'
python train_oneref.py "${COMMON[@]}" --epochs 2 --batch_size 8 --update_freq 4 \
  --lr .00025 --frozen_backbone --finetune "$INITIAL" --export_val_predictions \
  --output_dir "$RUN_DIR/warmup" --oss_backup_prefix "$OSS_PREFIX/warmup"
WARMUP="$RUN_DIR/warmup/best_checkpoint.pth"
[[ -f "$WARMUP" ]] || WARMUP="$RUN_DIR/warmup/checkpoint.pth"
sha256sum "$WARMUP" > "$RUN_DIR/shared_initialization.sha256"
for STAGE in control treatment; do
  WEIGHT=0
  [[ "$STAGE" == treatment ]] && WEIGHT=.1
  mkdir -p "$RUN_DIR/$STAGE"
  cp "$RUN_DIR/shared_initialization.sha256" "$RUN_DIR/$STAGE/initialization.sha256"
  python train_oneref.py "${COMMON[@]}" "${PAIR[@]}" --pair_loss_weight "$WEIGHT" \
    --epochs 8 --batch_size 16 --update_freq 2 --lr .00003 --use_box_mask_constraints \
    --finetune "$WARMUP" --export_val_predictions --output_dir "$RUN_DIR/$STAGE/finetune" \
    --oss_backup_prefix "$OSS_PREFIX/$STAGE/finetune"
done
python scripts/collect_results.py "$RUN_DIR/control" "$RUN_DIR/treatment" --output "$RUN_DIR/summary"
echo "COMPLETED: $RUN_DIR"
