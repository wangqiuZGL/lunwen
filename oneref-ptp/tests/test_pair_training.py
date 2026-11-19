import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import train_one_epoch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]]))

    def forward(self, *args, **kwargs):
        boxes = torch.tensor([[.25, .5, .5, 1.], [.75, .5, .5, 1.]]) + self.scores.sum() * 0
        return boxes, None, self.scores, None, None, None, None, None, None, None


class PairTrainingTest(unittest.TestCase):
    def run_step(self, weight, loss_type='ranking'):
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        image = SimpleNamespace(tensors=torch.zeros(2, 3, 2, 2), mask=torch.zeros(2, 2, 2))
        image.to = lambda device: image
        masks = torch.tensor([[[[1, 0], [1, 0]]], [[[0, 1], [0, 1]]]])
        batch = (image, ('red', 'blue'), torch.tensor([[.25, .5, .5, 1.], [.75, .5, .5, 1.]]),
                 masks, {'pairs': [(0, 1)]})
        args = SimpleNamespace(amp=False, update_freq=1, enable_pair_sampling=True,
                               pair_loss_weight=weight, pair_margin=.1, pair_min_patches=1,
                               pair_loss_type=loss_type, pair_temperature=.1,
                               use_regress_box=True, use_contrastive_loss=False,
                               use_box_mask_constraints=False, enable_dynamic_mim=False,
                               use_mask_loss=False, enable_ref_mim=False, enable_ref_mlm=False,
                               enable_mim_vts=False, enable_mlm_sts=False)
        before = model.scores.detach().clone()
        stats = train_one_epoch(args, model, [batch], optimizer, torch.device('cpu'), 0, 0)
        return before, model.scores.detach(), stats

    def test_positive_weight_changes_scores_and_logs_valid_comparisons(self):
        before, after, stats = self.run_step(.1)
        self.assertGreater(after[0, 0], before[0, 0])
        self.assertAlmostEqual(stats['pair_valid_comparisons'], 2.)
        self.assertEqual(stats['pair_valid_comparisons_total'], 2)
        self.assertEqual(stats['pair_candidate_pairs_total'], 1)
        self.assertAlmostEqual(stats['pair_valid_fraction'], 1.)

    def test_sampling_only_zero_weight_has_no_ranking_gradient(self):
        before, after, stats = self.run_step(0.)
        torch.testing.assert_close(before, after)
        self.assertEqual(stats['loss_pair'], 0.)

    def test_query_competition_dispatch_and_zero_weight(self):
        before, after, stats = self.run_step(.1, 'query_competition')
        self.assertGreater(after[0, 0], before[0, 0])
        self.assertIn('pair_query_accuracy_valid_mean', stats)
        before, after, stats = self.run_step(0., 'query_competition')
        torch.testing.assert_close(before, after)
        self.assertEqual(stats['loss_pair'], 0.)


if __name__ == '__main__':
    unittest.main()
