#!/usr/bin/env python3
"""Create portable, fixed respiratory readouts using accepted unchanged head code."""
from pathlib import Path
import copy
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT/'tmp/20261024_respiratory_peft_mechanism_pilot'
E = EXP/'h100_long/evaluation'
OLD = ROOT/'tmp/20261022_domain_adaptive_pretraining/strengthened_lora/evaluation'
RID = '20261025_005_h100_respiratory_long'
REMOTE = '/hpcdisk1/limk_group/caiqy/project/260905_MCF/tasks/'+RID


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def remote(path): return REMOTE+'/'+str(path.relative_to(E))
def record(path): return dict(path=remote(path), sha256=sha(path), bytes=path.stat().st_size)


def main():
    if (E/'audit/execution_manifest.json').exists(): raise RuntimeError('evaluation already sealed')
    for directory in ['scripts','audit/extracted','audit/verified','audit/received','audit/cpu_progress','audit/cpu_complete','audit/head_smoke','audit/cpu_failure','audit/local_accepted','audit/local_failure','audit/extract_progress','results','logs','delivery','incoming','big_return']:
        (E/directory).mkdir(parents=True, exist_ok=True)
    training=EXP/'h100_long/audit/manifest.json'
    raw=json.loads(training.read_text());recipes={}
    for name in raw['arms']:
        for epoch in [10,20,30,50]:
            recipes[name+f'_e{epoch:02d}']=dict(kind='method',training_manifest='audit/manifest.json',
                training_manifest_sha256=sha(training),training_arm=name,epoch=epoch,source=raw['source']['worktree'])
    src = OLD/'tasks/resp'; dst = E/'tasks/resp'
    for directory in ['scripts','cache','audit/embeddings','audit/head_smoke','logs','results/fits']:
        (dst/directory).mkdir(parents=True, exist_ok=True)
    shutil.copytree(src/'audit/source', dst/'audit/source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(src/'cache/folds', dst/'cache/folds')
    for name in ['corpus_rows.npy', 'y.npy']:
        shutil.copy2(src/'cache'/name, dst/'cache'/name)
    for name in ['genus_ids.npy', 'rclr.npy', 'padding.npy']:
        (dst/'cache'/name).symlink_to((src/'cache'/name).resolve())
    for p in (src/'scripts').glob('*.py'): shutil.copy2(p, dst/'scripts'/p.name)
    # Only path resolution changes; fit.py and original head functions stay byte-identical.
    p = dst/'scripts/common.py'; s = p.read_text()
    s = s.replace('def sha(path):\n',
        'def sha(path):\n path=Path(path)\n if str(path).startswith('+repr(REMOTE)+"+'/'):path=EXP.parents[1]/path.relative_to("+repr(REMOTE)+')\n')
    p.write_text(s)
    task_manifest = json.loads((src/'audit/manifest.json').read_text())
    for name in ['genus_ids.npy', 'rclr.npy', 'padding.npy', 'corpus_rows.npy']:
        candidates=[digest for path,digest in task_manifest['input_hashes'].items() if Path(path).name==name]
        assert len(candidates)==1 and sha(src/'cache'/name)==candidates[0], name
    task_manifest['source_hashes'] = {remote(p): sha(p) for folder in ['scripts', 'audit/source'] for p in (dst/folder).rglob('*.py')}
    task_manifest['input_hashes'] = {remote(p): sha(p) for p in (dst/'cache').rglob('*') if p.is_file() and not p.is_symlink()}
    task_manifest['origin_manifest'] = dict(path=str(src/'audit/manifest.json'), sha256=sha(src/'audit/manifest.json'))
    dump(dst/'audit/manifest.json', task_manifest)
    s = (OLD/'scripts/shared.py').read_text(); s = s[:s.index('def load_model(')]
    s = s.replace("ARMS = ['respiratory_lr3e4', 'respiratory_rank64']", 'ARMS = '+repr(list(recipes)))
    s = s.replace('def sha(p):\n',
        'def sha(p):\n    p=Path(p)\n    if str(p).startswith('+repr(REMOTE)+"+'/'):p=EXP/p.relative_to("+repr(REMOTE)+')\n')
    start = s.index('def load_arrays(domain):'); end = s.index('\ndef forward(', start)
    s = s[:start]+'''def load_arrays(domain):
    c = EXP / 'tasks' / domain / 'cache'
    return {'rows': np.load(c/'corpus_rows.npy', mmap_mode='r')}
\n'''+s[end:]
    (E/'scripts/shared.py').write_text(s)
    s = (OLD/'scripts/verify.py').read_text().replace('len(records)==188', 'len(records)==104').replace('fits=188', 'fits=104')
    (E/'scripts/verify.py').write_text(s)
    s = (OLD/'scripts/run_cpu.py').read_text().replace("PYTHON='/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python'", 'import sys\nPYTHON=sys.executable')
    s = s.replace(";assert f'/job_{job}/' in Path('/proc/self/cgroup').read_text()", ";assert os.environ.get('SLURM_JOB_NODELIST')")
    s = s.replace('total=188', 'total=104').replace("    subprocess.run([PYTHON,str(EXP/'scripts/report.py')],env=env,check=True)\n", '')
    (E/'scripts/run_cpu.py').write_text(s)
    shutil.copy2(OLD/'audit/verified/weak_unifrac_e50.json', E/'audit/verified/weak_unifrac_e50.json')
    for name in ['preflight.py', 'finish.py', 'receive_model.py', 'geometry.py', 'cpu_model.sbatch', 'cpu_preflight.sbatch', 'transport.py']:
        shutil.copy2(Path(__file__).parent/'domain_long_cpu'/name, E/'scripts'/name)
    shutil.copy2(EXP/'h100_long/PLAN.md', E/'PLAN.md')
    (E/'README.md').write_text('# 呼吸道方法筛选固定读出\n\n16个终点，各52fold×2head=104拟合。合同见PLAN，输入与源码见audit。\n')
    (E/'EXECUTION_LOG.md').write_text('# CPU执行日志\n\n输入在BIG staging接收，校验后消费。\n')
    dump(E/'audit/expected_environment.json', dict(torch='2.6.0+cpu', numpy='1.26.4', sklearn='1.8.0', joblib='1.5.3',
        provenance='existing BIG rm_site_cpu_v1 metadata verified 2026-10-25 via approved SSH; no installation'))
    (dst/'cache/weak_unifrac_e50.npy').symlink_to((src/'cache/weak_unifrac_e50.npy').resolve())
    dump(E/'audit/baseline_embedding.json', dict(path=remote(dst/'cache/weak_unifrac_e50.npy'),
        sha256=sha(src/'cache/weak_unifrac_e50.npy'), rows=90764))
    task = copy.deepcopy(json.loads((OLD/'audit/execution_manifest.json').read_text())['tasks']['resp'])
    task['manifest'] = record(dst/'audit/manifest.json')
    manifest = dict(status='SEALED', arms=list(recipes), tasks=dict(resp=task), head_seed=42,
        expected_new_fits=1664, fits_per_model=104, source_hashes={}, plan=record(E/'PLAN.md'),
        environment=record(E/'audit/expected_environment.json'), baseline_record=record(E/'audit/baseline_embedding.json'),
        request_id=RID, remote_root=REMOTE, comparison_protocol='unchanged respiratory folds and Linear/MLP; exploratory')
    manifest['source_hashes'] = {remote(p): sha(p) for p in (E/'scripts').glob('*.py')}
    dump(E/'audit/execution_manifest.json', manifest)
    dump(E/'audit/recipes.json', dict(models=recipes, rows_sha256=sha(dst/'cache/corpus_rows.npy'),
        inference_inputs={name:sha(dst/'cache'/name) for name in ['genus_ids.npy','rclr.npy','padding.npy','corpus_rows.npy']},
        evaluation_manifest_sha256=sha(E/'audit/execution_manifest.json')))
    print(json.dumps(dict(status='SEALED', models=len(recipes), fits=1664, request_id=RID,
                         manifest_sha256=sha(E/'audit/execution_manifest.json'))), flush=True)


if __name__ == '__main__': main()
