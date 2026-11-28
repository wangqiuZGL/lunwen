"""Optional REC inference refinements. No labels or training state are consumed."""
import math
import re
import torch
from torch.nn import functional as F


def corners(box):
    return torch.cat((box[:, :2]-box[:, 2:]/2, box[:, :2]+box[:, 2:]/2), -1)


def clip_boxes(box, bounds):
    c = corners(box)
    c = torch.maximum(torch.minimum(c, bounds[:, [2, 3, 2, 3]]), bounds[:, [0, 1, 0, 1]])
    return torch.cat(((c[:, :2]+c[:, 2:])/2, c[:, 2:]-c[:, :2]), -1)


def image_bounds(padding_mask):
    valid = ~padding_mask.bool()
    if not valid.flatten(1).any(-1).all():
        raise ValueError('An input image has no valid pixels')
    h, w = valid.shape[-2:]
    columns = valid.any(1); rows = valid.any(2)
    left = columns.to(torch.int64).argmax(1)
    right = w-columns.flip(1).to(torch.int64).argmax(1)
    top = rows.to(torch.int64).argmax(1)
    bottom = h-rows.flip(1).to(torch.int64).argmax(1)
    return torch.stack((left/w, top/h, right/w, bottom/h), -1)


def aligned_iou(first, second):
    p, q = corners(first), corners(second)
    intersection = (torch.minimum(p[:, 2:], q[:, 2:])-torch.maximum(p[:, :2], q[:, :2])).clamp_min(0).prod(-1)
    union = (p[:, 2:]-p[:, :2]).prod(-1)+(q[:, 2:]-q[:, :2]).prod(-1)-intersection
    return intersection/union.clamp_min(1e-8)


def choose_verified_box(base, reflected, score_maps, padding_mask, texts):
    """Fixed rule from train calibration; maps are both in the original frame."""
    b, _, n = score_maps.shape
    side = math.isqrt(n)
    if side*side != n:
        raise ValueError('Expected a square visual patch grid')
    frame = base.new_tensor([0, 0, 1, 1]).expand(b, -1)
    valid = F.adaptive_avg_pool2d((~padding_mask.bool()).float()[:, None], (side, side)).flatten(1) > .5
    if not valid.any(-1).all():
        raise ValueError('An image has no valid patch centers')
    axis = (torch.arange(side, device=base.device)+.5)/side
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    xy = torch.stack((xx, yy), -1).reshape(1, n, 2)
    skip = torch.tensor([bool(re.search(r'left|right|north|south|east|west|clock', text.lower())) for text in texts], device=base.device)
    reflected = torch.where(skip[:, None], base, reflected)
    support = []
    for proposal in (base, reflected):
        c = corners(clip_boxes(proposal, frame))
        inside = ((xy >= c[:, None, :2]) & (xy <= c[:, None, 2:])).all(-1) & valid
        empty = ~inside.any(-1)
        if empty.any():
            distance = (xy-proposal[:, None, :2]).square().sum(-1).masked_fill(~valid, torch.inf)
            inside[empty] = False
            inside[empty, distance[empty].argmin(-1)] = True
        support.append((score_maps*inside[:, None]).sum(-1)/inside.sum(-1)[:, None].clamp_min(1))
    accept = ((support[1]-support[0]) > .02).all(-1) & ~skip
    chosen = torch.where(accept[:, None], reflected, base)
    agreement = aligned_iou(clip_boxes(base, frame), clip_boxes(reflected, frame)) >= .5
    merged = torch.where(agreement[:, None], (base+reflected)/2, chosen)
    return clip_boxes(merged, image_bounds(padding_mask))


@torch.no_grad()
def predict_refined(model, images, padding_mask, texts, mode='none'):
    """Return the native OneRef tuple with only its REC box optionally refined.

    valid_projection needs one original forward; verified_flip needs two.
    REC box changes do not modify an optional segmentation output.
    """
    if mode not in ('none', 'valid_projection', 'verified_flip'):
        raise ValueError('Unknown REC refinement mode: '+mode)
    if model.training:
        raise ValueError('REC inference refinement requires model.eval()')
    if mode != 'verified_flip':
        output = model(images, padding_mask, texts)
        if mode == 'none':
            return output
        return (clip_boxes(output[0], image_bounds(padding_mask)), *output[1:])
    if not all(isinstance(text, str) for text in texts):
        raise ValueError('Verified flip supports plain-text referring expressions')
    module = getattr(model, 'module', model); captures = {}; handles = []
    for name in ('vision_head', 'language_head'):
        def hook(m, inputs, result, key=name):
            captures[key] = result.detach()
        handles.append(getattr(module, name).register_forward_hook(hook))
    def scores():
        return (F.normalize(captures['vision_head'], dim=-1)[:, 1:]*F.normalize(captures['language_head'], dim=-1)[:, 0, None]).sum(-1)
    try:
        output = model(images, padding_mask, texts); s0 = scores()
        reflected = model(images.flip(-1), padding_mask.flip(-1), texts)[0].clone()
        reflected[:, 0] = 1-reflected[:, 0]
        s1 = scores(); side = math.isqrt(s1.shape[-1])
        s1 = s1.reshape(-1, side, side).flip(-1).flatten(1)
        box = choose_verified_box(output[0], reflected, torch.stack((s0, s1), 1), padding_mask, texts)
        return (box, *output[1:])
    finally:
        for handle in handles:
            handle.remove()
