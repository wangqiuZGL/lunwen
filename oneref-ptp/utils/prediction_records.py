"""Optional single-process prediction records, indexed in annotation order."""
import json
from pathlib import Path

from torch.utils.data import Dataset

from utils.misc import collate_fn


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        return sample, dict(sample_index=index, image_id=str(sample[5]), phrase=sample[6])


def indexed_collate(raw):
    samples, metadata = zip(*raw)
    return (*collate_fn(samples), {'samples': list(metadata)})


def write_predictions(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)
