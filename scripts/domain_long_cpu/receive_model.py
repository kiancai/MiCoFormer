"""Validate staged immutable embeddings before publishing a model to CPU fits."""
import argparse
import shutil
from shared import *

parser = argparse.ArgumentParser(); parser.add_argument('--arm', required=True); args = parser.parse_args()
arm = args.arm; m = load_manifest(); assert arm in m['arms']
assert json.loads((EXP/'audit/static_acceptance.json').read_text())['status'] == 'PASS'
stage = EXP.parents[1]/'staging'/m['request_id']/'live'
src = stage/'tasks/resp/cache'/f'{arm}.npy'; receipt = stage/'tasks/resp/audit/embeddings'/f'{arm}.json'
r = json.loads(receipt.read_text()); extracted = json.loads((stage/'audit/extracted'/f'{arm}.json').read_text())
recipes = json.loads((EXP/'audit/recipes.json').read_text())
assert r['status'] == extracted['status'] == 'PASS' and extracted['arm'] == arm
assert r['manifest_sha256'] == sha(EXP/'tasks/resp/audit/manifest.json')
assert r['checkpoint_sha256'] == extracted['checkpoint_sha256']
assert r['training_manifest_sha256'] == recipes['models'][arm]['training_manifest_sha256']
assert r['rows_sha256'] == recipes['rows_sha256'] and r['rows'] == 90764
assert sha(src) == r['sha256']
a = np.load(src, mmap_mode='r')
assert a.shape == (90764,512) and a.dtype == np.float32 and np.isfinite(a).all() and (np.linalg.norm(a,axis=1)>0).all()
dest = EXP/'tasks/resp/cache'/src.name
if dest.exists(): assert sha(dest) == r['sha256']
else: os.link(src, dest)
shutil.copy2(receipt, EXP/'tasks/resp/audit/embeddings'/receipt.name)
shutil.copy2(stage/'audit/extracted'/f'{arm}.json', EXP/'audit/extracted'/f'{arm}.json')
dump(EXP/'audit/received'/f'{arm}.json', dict(status='PASS', time=now(), arm=arm, embedding_sha256=r['sha256'], manifest_sha256=sha(MANIFEST)))
