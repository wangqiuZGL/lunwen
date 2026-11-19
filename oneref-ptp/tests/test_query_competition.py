import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import pair_supervision as ps


class QueryCompetitionTest(unittest.TestCase):
    def setUp(self):
        self.masks = torch.tensor([[[[1, 1], [0, 0]]], [[[0, 0], [1, 1]]]])
        self.padding = torch.zeros(2, 2, 2)

    def loss(self, scores, masks=None, padding=None, pairs=None, **kw):
        self.assertTrue(hasattr(ps, 'query_competition_loss'), 'Missing local query competition')
        return ps.query_competition_loss(
            scores, self.masks if masks is None else masks,
            self.padding if padding is None else padding,
            [(0, 1)] if pairs is None else pairs, **kw)

    def test_local_mistakes_hidden_by_region_average_are_penalized(self):
        uneven = torch.tensor([[.9, -.3, 0., 0.], [0., 0., .9, -.3]])
        even = torch.tensor([[.3, .3, 0., 0.], [0., 0., .3, .3]])
        self.assertEqual(float(ps.pair_ranking_loss(uneven, self.masks, self.padding, [(0, 1)])[0]), 0.)
        self.assertEqual(float(ps.pair_ranking_loss(even, self.masks, self.padding, [(0, 1)])[0]), 0.)
        self.assertGreater(float(self.loss(uneven)[0]), float(self.loss(even)[0]) + 1.)

    def test_gradient_prefers_correct_query_on_both_regions(self):
        scores = torch.zeros(2, 4, requires_grad=True)
        loss, stats = self.loss(scores)
        loss.backward()
        self.assertTrue((scores.grad[0, :2] < 0).all())
        self.assertTrue((scores.grad[1, :2] > 0).all())
        self.assertTrue((scores.grad[0, 2:] > 0).all())
        self.assertTrue((scores.grad[1, 2:] < 0).all())
        self.assertEqual(stats['pair_valid_comparisons'], 2)

    def test_overlap_padding_and_no_pairs_return_differentiable_zero(self):
        for masks, padding, pairs in [
                (torch.ones_like(self.masks), self.padding, [(0, 1)]),
                (self.masks, torch.ones_like(self.padding), [(0, 1)]),
                (self.masks, self.padding, [])]:
            scores = torch.zeros(2, 4, requires_grad=True)
            loss, stats = self.loss(scores, masks, padding, pairs)
            self.assertEqual(float(loss), 0.)
            self.assertEqual(stats['pair_valid_comparisons'], 0)
            loss.backward()
            self.assertEqual(float(scores.grad.abs().sum()), 0.)

    def test_two_targets_are_balanced_despite_different_sizes(self):
        masks = torch.tensor([[[[1, 1], [1, 0]]], [[[0, 0], [0, 1]]]])
        scores = torch.zeros(2, 4, requires_grad=True)
        self.loss(scores, masks=masks)[0].backward()
        self.assertAlmostEqual(float(scores.grad[0, :3].abs().sum()), float(scores.grad[0, 3].abs()))

    def test_ambiguous_coarse_patch_is_ignored(self):
        masks = torch.zeros(2, 1, 4, 4)
        masks[0, 0, 0, 0] = 1
        masks[1, 0, 1, 1] = 1
        scores = torch.zeros(2, 4, requires_grad=True)
        self.assertEqual(self.loss(scores, masks, torch.zeros(2, 4, 4))[1]['pair_valid_comparisons'], 0)

    def test_temperature_and_config_validation(self):
        for temperature in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.loss(torch.zeros(2, 4), temperature=temperature)
            with self.assertRaises(ValueError):
                ps.validate_pair_config(SimpleNamespace(pair_temperature=temperature))


if __name__ == '__main__':
    unittest.main()
