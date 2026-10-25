"""Local delivery hashes/test-metric replay and progressive exploratory report."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
from .prepare import EVAL as E, RUN, now


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)


def accept(key):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score
    archive=E/'big_return'/f'{key}.tar'; ready=json.loads((E/'big_return'/f'{key}.ready.json').read_text())
    if ready['status']!='READY' or ready['arm']!=key or sha(archive)!=ready['archive']['sha256']:
        raise RuntimeError('CPU return archive mismatch')
    destination=E/'big_return/unpacked'/key
    if destination.exists():raise RuntimeError('partial unpack requires inspection')
    destination.mkdir(parents=True)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            name=Path(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():raise RuntimeError('unsafe result member')
        source.extractall(destination,filter='data')
    delivery=json.loads((destination/'delivery'/f'{key}.json').read_text())
    if (delivery['status']!='PASS' or delivery['fits']!=104
            or delivery['manifest_sha256']!=sha(E/'audit/execution_manifest.json')):
        raise RuntimeError('CPU delivery identity mismatch')
    for item in delivery['files']:
        src=destination/item['file']; target=E/item['file']
        if sha(src)!=item['sha256'] or src.stat().st_size!=item['bytes']:raise RuntimeError('CPU result file mismatch')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=item['sha256']:raise RuntimeError('conflicting existing CPU result')
        else:shutil.copy2(src,target)
    count=0
    for folder in sorted((E/'tasks/resp/results/fits').glob(key+'_*')):
        result=json.loads((folder/'result.json').read_text())
        if result['status']!='COMPLETE':raise RuntimeError('incomplete returned fit')
        with np.load(folder/'predictions.npz') as output:
            y=output['y_true'];pred=output['prediction'];scores=output['y_score']
        site=result['family']=='site'
        measured=dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=np.arange(10) if site else [0,1],average='macro',zero_division=0)))
        if site:
            auc=[float(roc_auc_score(y==i,scores[:,i])) for i in range(10)]
            measured.update(macro_ovr_auroc=float(np.mean(auc)),class_ovr_auroc=auc)
        else:
            recall=recall_score(y,pred,labels=[0,1],average=None,zero_division=0)
            measured.update(auroc=float(roc_auc_score(y,scores)),auprc=float(average_precision_score(y,scores)),
                prevalence=float(y.mean()),recall0=float(recall[0]),recall1=float(recall[1]))
        for name,value in result['metrics'].items():np.testing.assert_allclose(value,measured[name],atol=1e-10,rtol=1e-10)
        count+=1
    if count!=104:raise RuntimeError('returned fit coverage incomplete')
    dump(E/'audit/local_accepted'/f'{key}.json',dict(status='PASS',arm=key,fits=count,
        archive_sha256=ready['archive']['sha256'],all_return_file_hashes_verified=True,all_test_metrics_recomputed=True))


def report():
    import pandas as pd
    models=json.loads((E/'audit/recipes.json').read_text())['models'];frames=[];completed=[];geometry=[]
    for key in models:
        if not (E/'audit/local_accepted'/f'{key}.json').exists():continue
        frames.append(pd.read_csv(E/'results'/key/'per_fold.tsv',sep='\t'));completed.append(key)
        geometry.append(json.loads((E/'results'/key/'geometry.json').read_text()))
    old=ROOT/'tmp/20261022_domain_adaptive_pretraining/strengthened_lora/evaluation'
    for key in ['weak_unifrac_e50','respiratory_12l','respiratory_6l']:
        receipt=json.loads((old/'audit/references'/f'{key}.json').read_text());record=receipt['per_fold'];p=Path(record['path'])
        if sha(p)!=record['sha256']:raise RuntimeError('accepted reference changed')
        data=pd.read_csv(p,sep='\t');frames.append(data[data.domain=='resp'])
    previous=ROOT/'tmp/20261024_respiratory_peft_mechanism_pilot/h100_long/evaluation'
    for key in ['loraplus_r64_e50','lora_r128_e50','adapter_b256_e50','dora_r64_e50']:
        if (previous/'audit/local_accepted'/f'{key}.json').exists():frames.append(pd.read_csv(previous/'results'/key/'per_fold.tsv',sep='\t'))
    text=['# 呼吸道50轮适配：固定阶段的完整读出','',f'已完整验收{len(completed)}/6终点，每终点104拟合。',
        '训练从共同原始模型重开50轮计划；只有固定第50轮送评。单seed、既有开发面板；不按test选checkpoint。',
        '先对每种疾病的Study折等权平均，再对七种疾病等权；site另列。两头平均仅描述性汇总，不是集成分类器。','']
    if frames:
        rows=pd.concat(frames,ignore_index=True);rows.to_csv(E/'results/all_verified_folds.tsv',sep='\t',index=False)
        values=[]
        for (model,head,condition),g in rows.groupby(['model','head','condition'],dropna=False):
            metric='macro_f1' if (g.family=='site').all() else 'auroc'
            values.append(dict(model=model,head=head,condition=condition,disease=g.disease.iloc[0],metric=metric,value=float(g[metric].mean()),folds=len(g)))
        summary=pd.DataFrame(values);summary.to_csv(E/'results/task_summary.tsv',sep='\t',index=False)
        binary=summary[summary.metric=='auroc'];means=binary.groupby(['model','head']).agg(seven_disease_mean=('value','mean'),diseases=('disease','nunique')).reset_index()
        if not (means.diseases==7).all():raise RuntimeError('incomplete seven-disease average')
        means.to_csv(E/'results/mean_auroc.tsv',sep='\t',index=False)
        table=means.pivot(index='model',columns='head',values='seven_disease_mean');table['two_head_descriptive_mean']=table.mean(axis=1)
        text+=['七病等权AUROC（匹配RF seed42=0.614894；原三seedRF均值0.616148）：','',table.to_csv(sep='\t',float_format='%.4f'),'']
        for head in ['linear','mlp']:
            text+=[head+'各病/部位：','',summary[summary['head']==head].pivot(index='disease',columns='model',values='value').to_csv(sep='\t',float_format='%.4f'),'']
    if geometry:
        pd.DataFrame([dict(model=g['model'],centered_displacement=g['centered_relative_rms_displacement'],linear_cka=g['linear_cka'],centered_top20_retention=g['neighbors']['centered']['topk_retention_mean']) for g in geometry]).to_csv(E/'results/embedding_change.tsv',sep='\t',index=False)
        text+=['表示位移/CKA/邻域保持见embedding_change.tsv；变化幅度不等同质量。','']
    text+=['待完整验收：'+', '.join(k for k in models if k not in completed)]
    p=E/'results/REPORT.partial.md';p.write_text('\n'.join(text)+'\n');os.replace(p,E/'results/REPORT.md')


def collect():
    import fcntl
    with (E/'audit/collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic();models=json.loads((E/'audit/recipes.json').read_text())['models'];failed={}
        report()
        while time.monotonic()-started<12*3600:
            for key,recipe in models.items():
                if (E/'audit/local_accepted'/f'{key}.json').exists():continue
                if (E/'big_return'/f'{key}.ready.json').exists() and (E/'big_return'/f'{key}.tar').exists():
                    try:accept(key);failed.pop(key,None);report()
                    except Exception as error:failed[key]=repr(error);dump(E/'audit/local_failure'/f'{key}.json',dict(error=repr(error)))
            remote=E/'audit/big_transport_status.json'
            if remote.exists():
                state=json.loads(remote.read_text())
                for key,error in state.get('failures',{}).items():
                    if not (E/'audit/local_accepted'/f'{key}.json').exists():failed[key]=str(error)
                if state.get('status')=='FAILED':
                    for key in models:
                        if not (E/'audit/local_accepted'/f'{key}.json').exists():failed[key]='CPU transport failed: '+state.get('error','')
            for slot in [0,1]:
                p=RUN/f'audit/queue{slot}_status.json'
                if not p.exists():continue
                state=json.loads(p.read_text())
                if state['status']=='FAILED':
                    from .runner import QUEUES
                    methods=QUEUES[slot]
                    for key,recipe in models.items():
                        if recipe['training_arm'] in methods and not (E/'audit/extracted'/f'{key}.json').exists():failed[key]='GPU queue failed: '+state.get('error','')
            accepted=[k for k in models if (E/'audit/local_accepted'/f'{k}.json').exists()]
            status='COMPLETED' if len(accepted)==6 else 'PARTIAL_FAILURE' if len(accepted)+len(failed)==6 else 'RUNNING'
            dump(E/'audit/collector_status.json',dict(status=status,time=now(),pid=os.getpid(),accepted=accepted,failures=failed,total=6))
            if status!='RUNNING':
                if status=='COMPLETED':
                    # The final CPU endpoint can return before the last GPU status snapshot.
                    for _ in range(12):
                        if all((RUN/f'audit/queue{s}_status.json').exists() and json.loads((RUN/f'audit/queue{s}_status.json').read_text())['status']=='COMPLETED' for s in [0,1]):break
                        time.sleep(10)
                queues={str(s):json.loads((RUN/f'audit/queue{s}_status.json').read_text()) if (RUN/f'audit/queue{s}_status.json').exists() else {} for s in [0,1]}
                complete=status=='COMPLETED' and all(q.get('status')=='COMPLETED' for q in queues.values())
                dump(RUN/'audit/final_acceptance.json',dict(status='PASS' if complete else 'PARTIAL_FAILURE',time=now(),cpu_endpoints=len(accepted),fits=len(accepted)*104,queues=queues,failures=failed,placeholder_673251_untouched=True))
                report();return
            time.sleep(45)
        dump(E/'audit/collector_status.json',dict(status='WAIT_LIMIT_REACHED',time=now(),failures=failed))


if __name__=='__main__':collect()
