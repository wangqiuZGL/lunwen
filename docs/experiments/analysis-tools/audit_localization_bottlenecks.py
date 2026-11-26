"""Post-hoc REC bottleneck diagnostics from archived predictions; no training.

Size is normalized PADDED input area, not original-image area or COCO size.
Center-inside is a geometric proxy, never proof that the right object was chosen.
Uses recorded evaluator IoUs after checking their reconstruction.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_aligned_predictions import read, validate_recorded_ious


def summarize(records):
    pred = np.asarray([r['pred_box_cxcywh'] for r in records])
    gt = np.asarray([r['target_box_cxcywh'] for r in records])
    overlap = np.asarray([r['iou'] for r in records])
    area = gt[:, 2] * gt[:, 3]
    center_inside = (np.abs(pred[:, :2] - gt[:, :2]) <= gt[:, 2:] / 2).all(1)
    normalized_center_error = np.linalg.norm((pred[:, :2] - gt[:, :2]) / gt[:, 2:], axis=1)
    groups = {
        'all': np.ones(len(records), dtype=bool),
        'padded_area_lt_1pct': area < .01,
        'padded_area_1to10pct': (area >= .01) & (area < .1),
        'padded_area_ge_10pct': area >= .1,
    }
    result = {}
    for name, selected in groups.items():
        ids = np.flatnonzero(selected)
        failed = selected & (overlap < .5)
        result[name] = dict(
            n=len(ids),
            images=len({records[k]['image_id'] for k in ids}),
            accuracy_pct={str(t): float((overlap[selected] >= t).mean() * 100)
                          for t in [.3, .5, .6, .75, .9]} if len(ids) else {},
            mean_iou_pct=float(overlap[selected].mean() * 100) if len(ids) else None,
            failures_at_0_5=int(failed.sum()),
            near_threshold_failures_0_3_to_0_5=int((selected & (overlap >= .3) & (overlap < .5)).sum()),
            failed_prediction_center_inside_gt=int((failed & center_inside).sum()),
            median_center_error_in_gt_width_height=float(np.median(normalized_center_error[selected])) if len(ids) else None,
            median_predicted_to_gt_width_ratio=float(np.median(pred[selected, 2] / gt[selected, 2])) if len(ids) else None,
            median_predicted_to_gt_height_ratio=float(np.median(pred[selected, 3] / gt[selected, 3])) if len(ids) else None,
        )
    return dict(evaluator_reconstruction=validate_recorded_ious(records), groups=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plain', required=True)
    parser.add_argument('--fusion', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plain, fusion = read(args.plain), read(args.fusion)
    assert len(plain) == len(fusion) > 0
    for a, b in zip(plain, fusion):
        for key in ['sample_index', 'image_id', 'phrase', 'target_box_cxcywh', 'coordinate_frame']:
            assert a[key] == b[key], key
    n = 24 * 24
    result = dict(
        scope='Exploratory single-seed, previously inspected validation data; descriptive, not causal.',
        limitations=[
            'Padded area is confounded by image aspect ratio and differs from COCO small/medium/large.',
            'Center-inside and IoU intervals do not identify semantic error causes.',
            'Multiple expressions of one image or instance are correlated.',
            'Weight bounds describe the scalar pooling operator, not the information content of its values.',
        ],
        plain=summarize(plain), fusion=summarize(fusion),
        cosine_pooling_bounds=dict(
            assumption='Exact cosine logits in [-1,1], temperature=1, N=576; no token masking.',
            maximum_weight_ratio=math.exp(2),
            maximum_single_patch_mass=math.exp(2) / (n - 1 + math.exp(2)),
            maximum_mass_for_k_patches={str(k): k * math.exp(2) / (n-k+k*math.exp(2)) for k in [1, 6, 29, 58]},
            warning='Analytical upper bounds, not observed attention weights or accuracy predictions.',
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
