import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BackupArchiveTest(unittest.TestCase):
    def test_worker_retries_noninteractive_overwrite_and_cleans_only_staging(self):
        from utils.oss_backup import _upload_worker
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = root / '.oss-backup' / 'epoch_000_test'
            pending.mkdir(parents=True)
            (pending / 'checkpoint.pth').write_bytes(b'checkpoint')
            archive, status = root / 'checkpoints.tar', root / 'status.json'
            with patch('utils.oss_backup.subprocess.run', side_effect=[
                    SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as run, \
                    patch('utils.oss_backup.time.sleep'):
                code = _upload_worker(pending, archive, 'oss://bucket/stage/checkpoints.tar', status)
            self.assertEqual(code, 0)
            self.assertEqual(run.call_count, 2)
            self.assertIn('-f', run.call_args.args[0])
            self.assertEqual(run.call_args.kwargs['timeout'], 900)
            self.assertTrue(json.loads(status.read_text())['success'])
            self.assertFalse(archive.exists())
            self.assertFalse(pending.exists())

    def test_failed_upload_retains_recoverable_archive_and_failure_status(self):
        from utils.oss_backup import _upload_worker
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending = root / '.oss-backup' / 'epoch_000_test'
            pending.mkdir(parents=True)
            (pending / 'checkpoint.pth').write_bytes(b'checkpoint')
            archive, status = root / 'checkpoints.tar', root / 'status.json'
            with patch('utils.oss_backup.subprocess.run', return_value=SimpleNamespace(returncode=1)), \
                    patch('utils.oss_backup.time.sleep'):
                code = _upload_worker(pending, archive, 'oss://bucket/stage/checkpoints.tar', status)
            self.assertEqual(code, 1)
            self.assertTrue(archive.exists())
            self.assertFalse(json.loads(status.read_text())['success'])

    def test_archive_contains_both_checkpoints_and_hashes_but_no_dataset(self):
        self.assertIsNotNone(importlib.util.find_spec('utils.oss_backup'), 'Missing checkpoint backup')
        from utils.oss_backup import build_archive
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'checkpoint.pth').write_bytes(b'latest checkpoint')
            (root / 'best_checkpoint.pth').write_bytes(b'best checkpoint')
            (root / 'config.json').write_text('{}')
            # Only explicitly snapshotted files are packaged by the worker.
            archive = root / 'archive.tar'
            build_archive(root, archive)
            with tarfile.open(archive) as tar:
                self.assertIn('checkpoint.pth', tar.getnames())
                self.assertIn('best_checkpoint.pth', tar.getnames())
                manifest = json.load(tar.extractfile('backup_manifest.json'))
                self.assertEqual(manifest['files']['checkpoint.pth']['bytes'], 17)
                self.assertEqual(tar.extractfile('checkpoint.pth').read(), b'latest checkpoint')

    def test_atomic_save_leaves_hardlinked_snapshot_unchanged(self):
        self.assertIsNotNone(importlib.util.find_spec('utils.oss_backup'), 'Missing checkpoint backup')
        import os
        import torch
        from utils.oss_backup import atomic_save
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, snapshot = root / 'checkpoint.pth', root / 'snapshot.pth'
            atomic_save({'epoch': 1}, target)
            os.link(target, snapshot)
            atomic_save({'epoch': 2}, target)
            self.assertEqual(torch.load(snapshot, weights_only=False)['epoch'], 1)
            self.assertEqual(torch.load(target, weights_only=False)['epoch'], 2)


if __name__ == '__main__':
    unittest.main()
