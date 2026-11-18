"""Archive reviewed source paths only, never datasets or model weights."""
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
names = subprocess.check_output(
    ['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z']
).decode().split('\0')
with ZipFile(sys.argv[1], 'w', ZIP_DEFLATED) as archive:
    for name in sorted(set(filter(None, names))):
        path = root / name
        source = name.startswith('oneref-ptp/') and path.suffix in ('.py', '.sh', '.md')
        config = name in ('.gitignore', 'oneref-ptp/requirements.txt', 'oneref-ptp/requirements-server.txt')
        if path.is_file() and (source or config):
            archive.write(path, name)
