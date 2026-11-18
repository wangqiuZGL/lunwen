"""Collect logged validation results; never infer missing metrics or test results."""
import argparse
import csv
import json
from pathlib import Path


def summarize(run):
    run = Path(run)
    stage = run / 'finetune' if (run / 'finetune').is_dir() else run
    config_path = stage / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    rows = []
    path = stage / 'log.txt'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and 'epoch' in row and 'validation_accu' in row:
                rows.append(row)
    best = max(rows, key=lambda r: r['validation_accu']) if rows else {}
    epoch = best.get('epoch')
    prediction = stage / ('predictions_val_epoch_%03d.jsonl' % epoch) if epoch is not None else None
    return dict(run_id=run.name, stage=str(stage), seed=config.get('seed'),
                pair_sampling=config.get('enable_pair_sampling'), pair_weight=config.get('pair_loss_weight'),
                logged_epochs=len(rows), best_logged_epoch=epoch,
                best_logged_val_acc=best.get('validation_accu'),
                miou_at_best_logged_acc=best.get('validation_box_miou'),
                best_checkpoint_exists=(stage / 'best_checkpoint.pth').is_file(),
                prediction_file=str(prediction) if prediction and prediction.is_file() else None,
                expected_epochs=config.get('epochs'),
                log_complete=(bool(rows) and config.get('epochs') is not None
                              and max(r['epoch'] for r in rows) + 1 >= config['epochs']),
                selection_note='best logged epoch; checkpoint may instead be initialization or a prior resumed best')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    results = [summarize(run) for run in args.runs]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (args.output / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
