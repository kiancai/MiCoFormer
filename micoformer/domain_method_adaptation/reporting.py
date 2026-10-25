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
E=ROOT/'tmp/20261024_respiratory_peft_mechanism_pilot/method_screen/evaluation'


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
    models=json.loads((E/'audit/recipes.json').read_text())['models']
    frames=[];completed=[];geometry=[]
    for key in models:
        if not (E/'audit/local_accepted'/f'{key}.json').exists():continue
        frames.append(pd.read_csv(E/'results'/key/'per_fold.tsv',sep='\t'));completed.append(key)
        geometry.append(json.loads((E/'results'/key/'geometry.json').read_text()))
    old=ROOT/'tmp/20261022_domain_adaptive_pretraining/strengthened_lora/evaluation'
    for key in ['weak_unifrac_e50','respiratory_12l','respiratory_6l']:
        receipt=old/'audit/references'/f'{key}.json'
        if not receipt.exists():continue
        record=json.loads(receipt.read_text())['per_fold'];path=Path(record['path'])
        if sha(path)!=record['sha256']:raise RuntimeError('accepted comparison changed')
        data=pd.read_csv(path,sep='\t');frames.append(data[data.domain=='resp'])
    text=['# 呼吸道适配方法筛选：逐步验收结果','',f'新终点已验收 {len(completed)}/12。',
          '单seed、既有开发面板和预训练暴露；逐任务报告，未合成跨疾病总分。',
          '训练固定epoch10；读出只用val选head epoch。原始预测、完整重放和本地test指标复算均保留。','']
    if frames:
        rows=pd.concat(frames,ignore_index=True)
        rows.to_csv(E/'results/all_verified_folds.tsv',sep='\t',index=False)
        values=[]
        for (model,head,condition),group in rows.groupby(['model','head','condition'],dropna=False):
            site=bool((group.family=='site').all());metric='macro_f1' if site else 'auroc'
            values.append(dict(model=model,head=head,condition=condition,disease=group.disease.iloc[0],
                metric=metric,value=float(group[metric].mean()),folds=len(group)))
        summary=pd.DataFrame(values);summary.to_csv(E/'results/task_summary.tsv',sep='\t',index=False)
        contrasts=[]
        for reference in ['weak_unifrac_e50','r64_s42_window4096','r128_s42_window4096']:
            base=summary[summary.model==reference]
            for _,row in summary[summary.model.isin(completed)].iterrows():
                match=base[(base['head']==row['head'])&(base['condition']==row['condition'])]
                if len(match)==1:contrasts.append(dict(**row.to_dict(),reference=reference,delta=row['value']-match.iloc[0]['value']))
        pd.DataFrame(contrasts).to_csv(E/'results/contrasts.tsv',sep='\t',index=False)
        text+=['逐任务主表：task_summary.tsv；固定参照差异：contrasts.tsv；逐fold：all_verified_folds.tsv。','']
        for head in ['linear','mlp']:
            sub=summary[summary['head']==head]
            table=sub.pivot(index='disease',columns='model',values='value')
            text += [f'## {head}','',table.to_csv(sep='\t',float_format='%.4f'),'']
    if geometry:
        pd.DataFrame([dict(model=g['model'],centered_displacement=g['centered_relative_rms_displacement'],
            linear_cka=g['linear_cka'],centered_top20_retention=g['neighbors']['centered']['topk_retention_mean']) for g in geometry]).to_csv(E/'results/embedding_change.tsv',sep='\t',index=False)
        text+=['表示诊断见embedding_change.tsv；几何变化幅度不等同下游质量。','']
    text+=['待验收模型：'+', '.join(key for key in models if key not in completed)]
    temporary=E/'results/REPORT.partial.md';temporary.write_text('\n'.join(text)+'\n');os.replace(temporary,E/'results/REPORT.md')


def collect():
    started=time.monotonic();models=json.loads((E/'audit/recipes.json').read_text())['models'];failed={}
    report()
    while time.monotonic()-started<12*3600:
        remote_status=E/'audit/big_transport_status.json'
        if remote_status.exists():
            state=json.loads(remote_status.read_text())
            for key,value in state.get('failures',{}).items():
                if key in models and not (E/'audit/local_accepted'/f'{key}.json').exists():failed[key]=str(value)
            if state.get('status')=='FAILED':
                for key in models:
                    if not (E/'audit/local_accepted'/f'{key}.json').exists():failed[key]='BIG transport/preflight failed: '+state.get('error','')
        for key,recipe in models.items():
            if (E/'audit/extracted'/f'{key}.json').exists():continue
            family=(Path(recipe['training_manifest']).parents[1].name if recipe['kind']=='method'
                    else 'adapter' if key.startswith('r128_s42_') else 'loraplus' if key.startswith('r128_s43_') else 'dora')
            status=E.parent/'families'/family/'audit/controller_status.json'
            if status.exists() and json.loads(status.read_text()).get('status')=='FAILED':
                failed[key]='GPU branch failed before extraction; see '+str(status)
        for key in models:
            if key in failed or (E/'audit/local_accepted'/f'{key}.json').exists():continue
            if not (E/'big_return'/f'{key}.ready.json').exists() or not (E/'big_return'/f'{key}.tar').exists():continue
            try:accept(key);report()
            except Exception as error:
                failed[key]=repr(error);dump(E/'audit/local_failure'/f'{key}.json',dict(error=repr(error)))
        accepted=[key for key in models if (E/'audit/local_accepted'/f'{key}.json').exists()]
        dump(E/'audit/collector_status.json',dict(status='RUNNING',accepted=accepted,failures=failed,total=12))
        if len(accepted)+len(failed)==len(models):
            dump(E/'audit/collector_status.json',dict(status='COMPLETED' if not failed else 'PARTIAL_FAILURE',accepted=accepted,failures=failed,total=12))
            report();return not failed
        time.sleep(45)
    dump(E/'audit/collector_status.json',dict(status='WAIT_LIMIT_REACHED',failures=failed))
    return False
