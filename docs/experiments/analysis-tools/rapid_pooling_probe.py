"""Training-free readout diagnostics. No GT enters feature construction.

This is a falsifiable probe, not a validated new model or novelty claim.
Inputs are query-conditioned projected patch features and cosine scores.
It cannot cache away backbone changes such as SADB/LoRA.
"""
import math
import torch


VARIANTS = ('baseline', 'valid_only', 'valid_tau05', 'valid_tau025')


def pool_candidates(features, scores, valid):
    if features.ndim != 3 or scores.shape != features.shape[:2] or valid.shape != scores.shape:
        raise ValueError('Expected B,N,C features and aligned B,N scores/validity')
    valid = valid.bool()
    if not valid.any(1).all():
        raise ValueError('An image has no valid patches')
    if not torch.isfinite(features).all() or not torch.isfinite(scores).all():
        raise ValueError('Nonfinite cache inputs')
    values = features.float()
    logits = scores.float()
    result = {}
    for name, temperature, mask in [('baseline', 1., False), ('valid_only', 1., True),
                                     ('valid_tau05', .5, True), ('valid_tau025', .25, True)]:
        candidate = logits / temperature
        if mask:
            candidate = candidate.masked_fill(~valid, -torch.inf)
        weights = candidate.softmax(1)
        result[name] = (weights.unsqueeze(-1)*values).sum(1)
    weights = logits.softmax(1)
    diagnostics = dict(
        padding_weight_mass=(weights*(~valid)).sum(1),
        normalized_entropy=-(weights*weights.clamp_min(1e-30).log()).sum(1)/math.log(max(2, scores.shape[1])))
    return result, diagnostics


def choose_head_split(image_ids):
    """Image-disjoint internal screening split; never an official test split.

    A backbone previously fitted on these images has still seen them. This
    split only diagnoses readout fitting; it is not a generalization claim.
    """
    import hashlib
    return torch.tensor([int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16) % 5 == 0
                         for x in image_ids], dtype=torch.bool)
