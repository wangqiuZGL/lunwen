import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import validate


class ValidationLoggingTest(unittest.TestCase):
    def test_validation_can_be_serialized_to_training_log(self):
        target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        image = SimpleNamespace(tensors=torch.zeros(1, 3, 4, 4), mask=None)
        image.to = lambda device: image
        class Model(torch.nn.Module):
            def forward(self, *args):
                return target, None, None, None, None
        stats = validate(SimpleNamespace(use_mask_loss=False, amp=False), Model(),
                         [(image, ['cup'], target, torch.zeros(1, 1, 4, 4))],
                         torch.device('cpu'))
        self.assertEqual(json.loads(json.dumps(stats))['accu'], 1.0)


if __name__ == '__main__':
    unittest.main()
