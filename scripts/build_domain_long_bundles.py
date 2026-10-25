"""Create explicit, hash-listed H100 and BIG bundles from sealed recipes."""
from pathlib import Path
import json
import tarfile
from micoformer.domain_long_adaptation.prepare import RUN,EVAL,REPO
from micoformer.mpc_unifrac.workflow import sha256_file


def bundle(target,items):
    if target.exists():raise FileExistsError(target)
    listing=[]
    with tarfile.open(target,'w',dereference=True) as out:
        for src,name in items:
            src=Path(src)
            if not src.is_file():raise RuntimeError('bundle input absent: '+str(src))
            listing.append(dict(file=name,sha256=sha256_file(src),bytes=src.stat().st_size))
            out.add(src,arcname=name,recursive=False)
    record=dict(status='SEALED',path=str(target),sha256=sha256_file(target),bytes=target.stat().st_size,files=listing)
    target.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


def main():
    m=json.loads((RUN/'audit/manifest.json').read_text())
    h100=[(REPO/n,'code/'+n) for n in m['source']['files']]
    for n in ['settings.json','PLAN.md','audit/manifest.json','audit/manifest.sha256','audit/SOURCE.json']:
        h100.append((RUN/n,n))
    for k,r in m['assets'].items():
        if k not in ['corpus','origin','prior']:h100.append((RUN/r['path'],r['path']))
    for n in ['audit/execution_manifest.json','audit/recipes.json','tasks/resp/audit/manifest.json']:
        h100.append((EVAL/n,'evaluation/'+n))
    for n in ['genus_ids.npy','rclr.npy','padding.npy','corpus_rows.npy']:
        h100.append((EVAL/'tasks/resp/cache'/n,'evaluation/tasks/resp/cache/'+n))
    a=bundle(RUN/'transfer/h100_bundle.tar',h100)
    big=[]
    for folder in ['scripts','audit','tasks/resp/scripts','tasks/resp/audit','tasks/resp/cache/folds']:
        for p in sorted((EVAL/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:big.append((p,str(p.relative_to(EVAL))))
    for n in ['PLAN.md','README.md','EXECUTION_LOG.md','tasks/resp/cache/corpus_rows.npy','tasks/resp/cache/y.npy','tasks/resp/cache/weak_unifrac_e50.npy']:
        big.append((EVAL/n,n))
    # A file may be included via two tree roots; forbid duplicate archive members.
    names=[n for _,n in big]
    if len(names)!=len(set(names)):raise RuntimeError('duplicate static archive member')
    b=bundle(EVAL/'static_bundle.tar',big)
    print(json.dumps(dict(h100_bytes=a['bytes'],h100_sha256=a['sha256'],big_bytes=b['bytes'],big_sha256=b['sha256'])),flush=True)


if __name__=='__main__':main()
