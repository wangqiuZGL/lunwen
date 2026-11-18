import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import validate


class PredictionExportTest(unittest.TestCase):
    def test_export_retains_ids_and_matches_validation_iou(self):
        target = torch.tensor([[.5, .5, .2, .2]])
        image = SimpleNamespace(tensors=torch.zeros(1, 3, 4, 4), mask=None)
        image.to = lambda device: image
        class Model(torch.nn.Module):
            def forward(self, *args):
                return target, None, None, None, None
        metadata = {'samples': [{'sample_index': 7, 'image_id': 'a.jpg', 'phrase': 'cup'}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'predictions.jsonl')
            stats = validate(SimpleNamespace(use_mask_loss=False, amp=False), Model(),
                             [(image, ['cup'], target, torch.zeros(1, 1, 4, 4), metadata)],
                             torch.device('cpu'), prediction_path=str(path))
            record = json.loads(path.read_text())
        self.assertEqual(record['sample_index'], 7)
        self.assertEqual(record['image_id'], 'a.jpg')
        self.assertEqual(record['iou'], stats['box_miou'])
        self.assertTrue(record['hit_at_0_5'])


if __name__ == '__main__':
    unittest.main()
