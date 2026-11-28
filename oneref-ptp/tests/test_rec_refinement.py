import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('refinement', Path(__file__).resolve().parents[1] / 'utils/rec_refinement.py')
refinement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement)


class RefinementTests(unittest.TestCase):
    def test_content_bounds_and_projection(self):
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        mask[:, 2:, 1:7] = False
        bounds = refinement.image_bounds(mask)
        expected = torch.tensor([[.125, .25, .875, 1.]] * 2)
        self.assertTrue(torch.equal(bounds, expected))
        box = torch.tensor([[.5, .5, 1., 1.]] * 2)
        self.assertTrue(torch.allclose(refinement.corners(refinement.clip_boxes(box, bounds)), expected))

    def test_verification_and_direction_fallback(self):
        base = torch.tensor([[.125, .125, .25, .25]] * 2)
        reflected = torch.tensor([[.875, .875, .25, .25]] * 2)
        scores = torch.zeros(2, 2, 16)
        scores[:, :, 15] = 1
        actual = refinement.choose_verified_box(base, reflected, scores, torch.zeros(2, 8, 8, dtype=torch.bool), ['person', 'person on left'])
        self.assertTrue(torch.allclose(actual[0], reflected[0]))
        self.assertTrue(torch.allclose(actual[1], base[1]))

    def test_none_preserves_native_output(self):
        class Native(torch.nn.Module):
            def forward(self, *args):
                return (torch.tensor([[.5, .5, .4, .4]]), None, 'image', 'text', None)
        model = Native().eval()
        actual = refinement.predict_refined(model, None, None, ['object'], 'none')
        self.assertEqual(actual[1:], (None, 'image', 'text', None))
        self.assertTrue(torch.equal(actual[0], torch.tensor([[.5, .5, .4, .4]])))


if __name__ == '__main__':
    unittest.main()
