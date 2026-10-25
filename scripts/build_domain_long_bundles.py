from pathlib import Path
import json,tarfile
from micoformer.domain_long_adaptation.prepare import RUN,EVAL,REPO
from micoformer.mpc_unifrac.workflow import sha256_file

def bundle(target,items,reused=None):
 if target.exists():raise FileExistsError(target)
 listing=[]
 with tarfile.open(target,'w',dereference=True) as out:
  for src,name in items:
   src=Path(src)
   if not src.is_file():raise RuntimeError(str(src))
   listing.append(dict(file=name,sha256=sha256_file(src),bytes=src.stat().st_size))
   out.add(src,arcname=name,recursive=False)
 record=dict(status='SEALED',path=str(target),sha256=sha256_file(target),bytes=target.stat().st_size,files=listing,reused_inputs=reused or [])
 target.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 return record

m=json.loads((RUN/'audit/manifest.json').read_text())
h100=[(REPO/n,'code/'+n) for n in m['source']['files']]
for n in ['settings.json','PLAN.md','AGENTS.md','audit/manifest.json','audit/manifest.sha256','audit/SOURCE.json']:
 h100.append((RUN/n,n))
for k,r in m['assets'].items():
 if k not in ['corpus','origin','prior']:h100.append((RUN/r['path'],r['path']))
for n in ['audit/execution_manifest.json','audit/recipes.json','tasks/resp/audit/manifest.json']:
 h100.append((EVAL/n,'evaluation/'+n))
reused=[]
for n in ['genus_ids.npy','rclr.npy','padding.npy','corpus_rows.npy']:
 p=EVAL/'tasks/resp/cache'/n
 reused.append(dict(file='evaluation/tasks/resp/cache/'+n,source='/disk1/shengan_group/wyn/caiqy/tmp/20261025_respiratory_peft_next50/evaluation/tasks/resp/cache/'+n,sha256=sha256_file(p),bytes=p.stat().st_size))
for key in ['corpus','origin','prior']:
 r=m['assets'][key]
 reused.append(dict(file=r['path'],source='/disk1/shengan_group/wyn/caiqy/tmp/20261025_respiratory_peft_next50/'+r['path'],sha256=r['sha256'],bytes=r['bytes']))
a=bundle(RUN/'transfer/h100_bundle.tar',h100,reused)
big=[]
for folder in ['scripts','audit','tasks/resp/scripts','tasks/resp/audit','tasks/resp/cache/folds']:
 for p in sorted((EVAL/folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:big.append((p,str(p.relative_to(EVAL))))
for n in ['PLAN.md','README.md','EXECUTION_LOG.md','tasks/resp/cache/corpus_rows.npy','tasks/resp/cache/y.npy','tasks/resp/cache/weak_unifrac_e50.npy']:
 big.append((EVAL/n,n))
names=[n for _,n in big]
if len(names)!=len(set(names)):raise RuntimeError('duplicate members')
b=bundle(EVAL/'static_bundle.tar',big)
print(json.dumps(dict(h100_bytes=a['bytes'],h100_sha256=a['sha256'],big_bytes=b['bytes'],big_sha256=b['sha256'])))
