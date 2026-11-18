import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('collect_results', Path(__file__).resolve().parents[1] / 'scripts/collect_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CollectResultsTest(unittest.TestCase):
    def test_ignores_namespace_and_uses_accuracy_selected_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory, 'finetune')
            stage.mkdir()
            (stage / 'log.txt').write_text('Namespace(foo=1)\n' + '\n'.join(json.dumps(r) for r in [
                {'epoch': 0, 'validation_accu': .7, 'validation_box_miou': .8},
                {'epoch': 1, 'validation_accu': .75, 'validation_box_miou': .6}]))
            result = module.summarize(Path(directory))
            self.assertEqual(result['best_logged_epoch'], 1)
            self.assertEqual(result['best_logged_val_acc'], .75)
            self.assertEqual(result['miou_at_best_logged_acc'], .6)
            self.assertFalse(result['best_checkpoint_exists'])


if __name__ == '__main__':
    unittest.main()
