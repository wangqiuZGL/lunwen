"""Bounded asynchronous OSS checkpoint archives, without credentials in code."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time


def atomic_save(checkpoint, path):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def build_archive(snapshot, archive):
    snapshot, archive = Path(snapshot), Path(archive)
    allowed = {'checkpoint.pth', 'best_checkpoint.pth', 'config.json', 'snapshot.json'}
    files = [p for p in snapshot.rglob('*') if p.is_file() and
             (p.name in allowed or 'metadata' in p.relative_to(snapshot).parts)]
    if not any(p.name == 'checkpoint.pth' for p in files):
        raise ValueError('Snapshot is missing its latest checkpoint')
    manifest = dict(files={str(p.relative_to(snapshot)).replace('\\', '/'):
                          dict(bytes=p.stat().st_size, sha256=_digest(p)) for p in files})
    manifest_path = snapshot / 'backup_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w') as tar:
        for path in files + [manifest_path]:
            # Store hardlinked snapshots as independent regular tar members.
            tar.add(path, arcname=str(path.relative_to(snapshot)), recursive=False)
    return manifest


def _upload_worker(snapshot, archive, destination, status_path):
    snapshot, archive, status_path = Path(snapshot), Path(archive), Path(status_path)
    status = dict(success=False, destination=destination)
    try:
        build_archive(snapshot, archive)
        status.update(bytes=archive.stat().st_size, sha256=_digest(archive))
        for attempt in range(3):
            result = subprocess.run(['oss', 'cp', str(archive), destination, '-f'],
                                    stdin=subprocess.DEVNULL, timeout=900)
            if result.returncode == 0:
                status['success'] = True
                break
            time.sleep(5 * (attempt + 1))
        if status['success']:
            archive.unlink()
    except Exception as exc:
        status['error'] = repr(exc)
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        # Only remove this module's generated staging directory, never the run.
        if snapshot.name.startswith('epoch_') and snapshot.parent.name == '.oss-backup':
            shutil.rmtree(snapshot)
    return 0 if status['success'] else 1


class AsyncCheckpointBackup:
    def __init__(self, stage, prefix):
        if not prefix.startswith('oss://') or len(prefix.split('/')) < 4:
            raise ValueError('Expected an oss://bucket/prefix backup destination')
        self.stage = Path(stage).resolve()
        self.prefix = prefix.rstrip('/')
        self.job = None
        self.log = None
        self.status_path = self.stage / 'oss_backup_status.json'

    def _wait(self, required=False):
        if self.job is None:
            return
        code = self.job.wait()
        self.log.close()
        self.job = None
        if code:
            message = 'OSS_BACKUP_FAILED: inspect ' + str(self.stage / 'oss_backup.log')
            print(message, flush=True)
            if required:
                raise RuntimeError(message)

    def submit(self, epoch):
        # One worker per stage bounds disk usage and preserves upload order.
        self._wait()
        staging = self.stage / '.oss-backup'
        staging.mkdir(exist_ok=True)
        snapshot = Path(tempfile.mkdtemp(prefix='epoch_%03d_' % epoch, dir=staging))
        for name in ('checkpoint.pth', 'best_checkpoint.pth'):
            source = self.stage / name
            if source.is_file():
                os.link(source, snapshot / name)
        (snapshot / 'snapshot.json').write_text(json.dumps(dict(epoch=epoch,
            stage=str(self.stage), destination=self.prefix)), encoding='utf-8')
        names = ['log.txt', 'config.json', 'source_snapshot.zip', 'environment.txt',
                 'subset_manifest.json', 'git_commit.txt', 'code_changes.patch',
                 'launcher.sh', 'experiment.json', 'pair_audit.json',
                 'shared_initialization.sha256', 'initialization.sha256']
        for level, folder in enumerate([self.stage, self.stage.parent, self.stage.parent.parent]):
            targets = [folder / name for name in names]
            if level == 0:
                targets += list(folder.glob('predictions_val_epoch_*.jsonl'))
                targets += list(folder.glob('pair_schedule_epoch_*.json'))
            for source in targets:
                if source.is_file():
                    dest = snapshot / 'metadata' / str(level) / source.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, dest)
        self.log = (self.stage / 'oss_backup.log').open('a')
        destination = self.prefix + '/checkpoints.tar'
        self.job = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            str(snapshot), str(staging / 'checkpoints.tar'), destination, str(self.status_path)],
            stdout=self.log, stderr=subprocess.STDOUT)
        print('OSS_BACKUP_QUEUED', epoch, destination, flush=True)

    def close(self):
        self._wait(required=True)


if __name__ == '__main__':
    sys.exit(_upload_worker(*sys.argv[1:]))
