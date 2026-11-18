import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Load local transforms without importing the heavyweight datasets package.
spec = importlib.util.spec_from_file_location('local_transforms', ROOT / 'datasets/transforms.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
from utils.pair_data import PairedDataset, paired_collate


class FakeDataset:
    def __init__(self):
        self.transform = T.Compose([T.RandomResize([8]), T.ToTensor(), T.NormalizeAndPad(size=8)])

    def pull_item(self, idx):
        box = torch.tensor([0., 0., 3., 8.] if idx == 0 else [5., 0., 8., 8.])
        mask = torch.zeros(1, 8, 8)
        mask[:, :, :3] = idx == 0
        mask[:, :, 5:] = idx == 1
        return 'a', Image.new('RGB', (8, 8), 'white'), ['red', 'blue'][idx], box, box.numpy(), mask

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return ('original', idx)


class PairedDataTest(unittest.TestCase):
    def test_shared_geometry_and_collation(self):
        data = PairedDataset(FakeDataset())
        samples = data[(0, 1)]
        batch = paired_collate([samples])
        self.assertEqual(batch[4]['pairs'], [(0, 1)])
        self.assertEqual(batch[1], ('red', 'blue'))
        torch.testing.assert_close(batch[0].tensors[0], batch[0].tensors[1])
        torch.testing.assert_close(batch[2], torch.tensor([[.1875, .5, .375, 1.], [.8125, .5, .375, 1.]]))
        self.assertEqual(batch[3].shape, (2, 1, 8, 8))

    def test_extra_boxes_resize_pad_preserve_coordinates(self):
        boxes = torch.tensor([[0., 0., 2., 4.], [2., 0., 4., 4.]])
        result = T.Compose([T.RandomResize([8]), T.ToTensor(), T.NormalizeAndPad(size=12)])(
            dict(img=Image.new('RGB', (4, 4)), box=torch.tensor([0., 0., 4., 4.]),
                 text='a', obj_mask=torch.ones(2, 4, 4), pair_boxes=boxes))
        torch.testing.assert_close(result['pair_boxes'], torch.tensor([
            [1/3, .5, 1/3, 2/3], [2/3, .5, 1/3, 2/3]]))
        self.assertEqual(result['obj_mask'].shape, (2, 12, 12))

    def test_pair_augmentation_is_reproducible_and_preserves_targets(self):
        base = FakeDataset()
        base.transform = T.Compose([T.RandomResize([16]), T.RandomSizeCrop(8, 12),
                                    T.RandomResize([8]), T.ToTensor(), T.NormalizeAndPad(size=8)])
        data = PairedDataset(base)
        for seed in range(20):
            random.seed(seed)
            samples = data[(0, 1)]
            self.assertTrue(all(sample[8].sum() > 0 for sample in samples))
            self.assertTrue(all((sample[4][2:] > 0).all() for sample in samples))


if __name__ == '__main__':
    unittest.main()
