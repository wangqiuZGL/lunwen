import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pair_supervision import PairIndex, PairBatchSampler, pair_ranking_loss, validate_pair_config


def row(image, box, phrase):
    return (image, {'height': 8, 'width': 8}, box, phrase, None)


class PairIndexTest(unittest.TestCase):
    def setUp(self):
        self.rows = [row('a', [0, 0, 3, 8], 'red man'),
                     row('a', [0, 0, 3, 8], 'the man'),
                     row('a', [5, 0, 3, 8], 'blue man'),
                     row('b', [0, 0, 3, 8], 'red man')]

    def test_same_target_and_other_image_are_not_negatives(self):
        index = PairIndex(self.rows)
        self.assertEqual(index.partners[0], [2])
        self.assertEqual(index.partners[3], [])
        self.assertEqual(index.audit()['pairable_expressions'], 3)

    def test_sampler_has_fixed_budget_no_replacement_and_reproducible_epochs(self):
        index = PairIndex(self.rows)
        sampler = PairBatchSampler(index, 2, seed=42, pair_fraction=1.)
        batches = list(sampler)
        flatten = [i for batch in batches for key in batch
                   for i in (key if isinstance(key, tuple) else (key,))]
        self.assertEqual(sorted(flatten), list(range(4)))
        self.assertEqual(batches, list(sampler))
        self.assertEqual(len(batches), len(sampler))

    def test_overlap_and_identical_description_are_filtered(self):
        rows = [row('a', [0, 0, 4, 8], 'man'),
                row('a', [0, 0, 4.1, 8], 'other'),
                row('a', [5, 0, 3, 8], 'man')]
        self.assertEqual(PairIndex(rows).partners[0], [])

    def test_sampler_drops_only_tail_and_never_duplicates_samples_across_many_epochs(self):
        rows = [row(str(i // 6), [(i % 3)*5, 0, 3, 8], 'target %d' % i) for i in range(101)]
        sampler = PairBatchSampler(PairIndex(rows), 16, seed=42)
        for epoch in range(10):
            sampler.set_epoch(epoch)
            manifest = sampler.manifest()
            self.assertEqual(manifest['samples'], 96)
            self.assertEqual(manifest['unique_samples'], 96)
            self.assertEqual(len(manifest['dropped_indices']), 5)
            self.assertLessEqual(manifest['pairs'], 6 * 4)

    def test_config_rejects_unsupported_combinations_and_silent_unused_weight(self):
        with self.assertRaises(ValueError):
            validate_pair_config(SimpleNamespace(pair_loss_weight=.1))
        with self.assertRaises(ValueError):
            validate_pair_config(SimpleNamespace(enable_pair_sampling=True, pair_loss_weight=.1, distributed=True))
        with self.assertRaises(ValueError):
            validate_pair_config(SimpleNamespace(enable_pair_sampling=True, pair_loss_weight=.1,
                                                dataset='unc+', enable_ptp_aux=True))
        validate_pair_config(SimpleNamespace())


class PairLossTest(unittest.TestCase):
    def test_wrong_region_costs_more_and_gradients_correct_it(self):
        masks = torch.tensor([[[[1, 0], [1, 0]]], [[[0, 1], [0, 1]]]])
        scores = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]], requires_grad=True)
        loss, stats = pair_ranking_loss(scores, masks, torch.zeros(2, 2, 2), [(0, 1)], margin=.1)
        self.assertEqual(stats['pair_valid_comparisons'], 2)
        self.assertGreater(loss.item(), 1.)
        loss.backward()
        self.assertLess(scores.grad[0, 0].item(), 0.)
        good, _ = pair_ranking_loss(1 - scores.detach(), masks, torch.zeros(2, 2, 2), [(0, 1)])
        self.assertEqual(good.item(), 0.)

    def test_empty_or_padded_regions_skip_with_differentiable_zero(self):
        scores = torch.randn(2, 4, requires_grad=True)
        loss, stats = pair_ranking_loss(scores, torch.zeros(2, 1, 2, 2),
                                        torch.ones(2, 2, 2), [(0, 1)])
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(stats['pair_valid_comparisons'], 0)
        loss.backward()
        self.assertEqual(scores.grad.abs().sum().item(), 0.)

    def test_overlapping_regions_are_excluded(self):
        scores = torch.randn(2, 4, requires_grad=True)
        masks = torch.ones(2, 1, 2, 2)
        loss, stats = pair_ranking_loss(scores, masks, torch.zeros(2, 2, 2), [(0, 1)])
        self.assertEqual(stats['pair_valid_comparisons'], 0)


if __name__ == '__main__':
    unittest.main()
