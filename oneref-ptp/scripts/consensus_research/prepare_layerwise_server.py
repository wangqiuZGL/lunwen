"""Restore existing assets and only extract prespecified experiment images."""
import io,json,math,random,tarfile,zipfile,hashlib,difflib,sys
from pathlib import Path
import torch
root=Path('/hy-tmp/layerwise-20261127');assets=root/'assets';split=assets/'splits/unc+'
def extract_images(manifest):
 needed={r['image'] for group in manifest.values() for r in group}
 image_dir=assets/'image_data/other/images/mscoco/images/train2014';image_dir.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(root/'images.zip') as z:
  members={Path(n).name:n for n in z.namelist() if n.endswith('.jpg')}
  for name in sorted(needed):(image_dir/name).write_bytes(z.read(members[name]))
 print('EXTRACTED_IMAGES',len(needed),flush=True)
if '--images-only' in sys.argv:
 extract_images(json.loads((root/'manifest.json').read_text()));raise SystemExit(0)
split.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(root/'code.zip') as z:z.extractall(root/'code')
# Lazy NLP load is outside ordinary plain-text REC; no fallback parser is used.
p=root/'code/models/OneRef_model.py';original=p.read_text()
start=original.index('try:\n    nlp = spacy.load(');end=original.index('\n\ndef get_subject_phrase',start)
lazy='''class _LazyNLP:
    def __call__(self, text):
        import spacy
        global nlp
        nlp = spacy.load("en_core_web_md")
        return nlp(text)
nlp = _LazyNLP()
'''
updated=(original[:start]+lazy+original[end:]).replace('import spacy\n','',1).replace('from spacy.tokens.token import Token\n','')
p.write_text(updated)
(root/'plain_rec_lazy_nlp.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True))))
syntax=root/'code/models/syntax_guided_attention.py';source=syntax.read_text()
syntax.write_text(source.replace('import spacy\n','').replace('def _load_spacy_model():','def _load_spacy_model():\n        import spacy'))
with tarfile.open(root/'plain.tar') as z:
 for name in ('best_checkpoint.pth','metadata/0/config.json'):
  target=root/Path(name).name
  with z.extractfile(name) as source,target.open('wb') as destination:
   import shutil;shutil.copyfileobj(source,destination)
with zipfile.ZipFile(root/'beit3.zip') as z:(assets/'beit3.spm').write_bytes(z.read('beit3.spm'))
rows={}
with zipfile.ZipFile(root/'splits.zip') as z:
 for k in ('train','val'):rows[k]=torch.load(io.BytesIO(z.read('ref_data_shuffled/single_dataset/unc+/unc+_'+k+'.pth')))
images=sorted({r[0] for r in rows['train']});selected=set(random.Random(42).sample(images,math.ceil(len(images)*.2)))
rows['train']=[r for r in rows['train'] if r[0] in selected];assert len(rows['train'])==24125
assert not {r[0] for r in rows['train']} & {r[0] for r in rows['val']}
for k in rows:torch.save(rows[k],split/('unc+_'+k+'.pth'))
rng=random.Random(20261127);manifest={}
for k,n in [('train',512),('val',256)]:
 ids=rng.sample(range(len(rows[k])),n)
 manifest[k]=[dict(source_index=i,image=str(rows[k][i][0]),phrase=rows[k][i][3]) for i in ids]
(root/'manifest.json').write_text(json.dumps(manifest))
config=json.loads((root/'config.json').read_text());config.update(data_root=str(assets/'image_data'),split_root=str(assets/'splits'),sentencepiece_model=str(assets/'beit3.spm'))
(root/'probe_config.json').write_text(json.dumps(config,indent=2))
if '--metadata-only' not in sys.argv:extract_images(manifest)
print(json.dumps(dict(train=512,val=256,code_sha256=hashlib.sha256((root/'code.zip').read_bytes()).hexdigest())),flush=True)
