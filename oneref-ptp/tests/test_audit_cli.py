import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch


class AuditCliTest(unittest.TestCase):
    def test_trusted_cache_to_manifest_and_preview(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            records = []
            for image_index in range(2):
                image = 'image%d.jpg' % image_index
                Image.new('RGB', (32, 16), 'white').save(directory / image)
                for target in range(2):
                    for phrase in range(4):
                        records.append((image, {'width': 32, 'height': 16},
                                        [target*20, 0, 8, 16],
                                        'target %d description %d' % (target, phrase), None))
            cache = directory / 'train.pth'
            torch.save(records, cache)
            output = directory / 'audit'
            subprocess.run([sys.executable, str(root / 'scripts/audit_pairs.py'),
                            '--train-cache', str(cache), '--output', str(output),
                            '--image-root', str(directory), '--preview-count', '2'],
                           check=True, capture_output=True, text=True)
            audit = json.loads((output / 'data_audit.json').read_text())
            manifest = json.loads((output / 'pair_schedule_epoch_000.json').read_text())
            self.assertEqual(audit['pairable_expressions'], 16)
            self.assertEqual(manifest['unique_samples'], 16)
            self.assertEqual(len(list((output / 'previews').glob('*.jpg'))), 2)
            self.assertEqual(len((output / 'pair_index.jsonl').read_text().splitlines()), 32)


if __name__ == '__main__':
    unittest.main()
