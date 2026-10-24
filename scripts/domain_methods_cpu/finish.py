"""Reuse accepted geometry, then package fully replayed per-model CPU results."""
import argparse
import tarfile
from geometry import compare
from shared import *

parser=argparse.ArgumentParser(); parser.add_argument('--arm',required=True); args=parser.parse_args()
arm=args.arm
receipt=json.loads((EXP/'audit/verified'/f'{arm}.json').read_text())
assert receipt['status']=='PASS' and receipt['fits']==104
cache=EXP/'tasks/resp/cache'
geometry=compare(np.load(cache/'weak_unifrac_e50.npy',mmap_mode='r'),np.load(cache/f'{arm}.npy',mmap_mode='r'))
dump(EXP/'results'/arm/'geometry.json',dict(model=arm,reference='weak_unifrac_e50',precision='BF16 production; FP32 storage; FP64 statistics',**geometry))
files=[p for directory in (EXP/'tasks/resp/results/fits').glob(arm+'_*') for p in directory.rglob('*') if p.is_file()]
files.extend((EXP/'results'/arm).rglob('*'))
files += [EXP/'audit/verified'/f'{arm}.json', EXP/'audit/cpu_complete'/f'{arm}.json', EXP/'audit/received'/f'{arm}.json']
files=sorted({p for p in files if p.is_file()})
delivery=EXP/'delivery'/f'{arm}.json'
dump(delivery,dict(status='PASS',time=now(),arm=arm,fits=104,manifest_sha256=sha(MANIFEST),
    files=[dict(file=str(p.relative_to(EXP)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
archive=EXP/'delivery'/f'{arm}.tar'; assert not archive.exists()
temporary=archive.with_suffix('.partial.tar')
with tarfile.open(temporary,'w') as output:
    for p in files+[delivery]: output.add(p,arcname=str(p.relative_to(EXP)),recursive=False)
os.replace(temporary,archive)
dump(EXP/'delivery'/f'{arm}.ready.json',dict(status='READY',time=now(),arm=arm,archive=record(archive),manifest_sha256=sha(MANIFEST)))
