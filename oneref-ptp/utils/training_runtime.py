"""Optimizer updates shared by the single-GPU downstream training path."""
import torch


def backward_and_step(loss, model, optimizer, scaler, index, total_batches,
                      update_freq, max_norm=0):
    if update_freq < 1:
        raise ValueError('update_freq must be positive')
    window_start = (index // update_freq) * update_freq
    window_size = min(update_freq, total_batches - window_start)
    scaler.scale(loss / window_size).backward()
    update = (index + 1) % update_freq == 0 or index + 1 == total_batches
    if update:
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return update
