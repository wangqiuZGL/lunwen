import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.training_runtime import backward_and_step


class AccumulationTest(unittest.TestCase):
    def test_matches_large_batches_and_final_partial_window(self):
        x = torch.arange(1., 11.).reshape(5, 2)
        y = x * 0.3
        full = torch.nn.Linear(2, 2)
        micro = torch.nn.Linear(2, 2)
        micro.load_state_dict(full.state_dict())
        opt_full = torch.optim.SGD(full.parameters(), lr=0.001)
        opt_micro = torch.optim.SGD(micro.parameters(), lr=0.001)
        for start in (0, 3):
            opt_full.zero_grad()
            torch.nn.functional.mse_loss(full(x[start:start+3]), y[start:start+3]).backward()
            opt_full.step()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        opt_micro.zero_grad()
        updates = []
        for i in range(5):
            loss = torch.nn.functional.mse_loss(micro(x[i:i+1]), y[i:i+1])
            updates.append(backward_and_step(loss, micro, opt_micro, scaler, i, 5, 3))
        self.assertEqual(updates, [False, False, True, False, True])
        for a, b in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
