import json

import pytest

from qwen_lab import neuron_roles as nr
from qwen_lab import patch27 as p27
from qwen_lab.http_path import ident


def protocol():
    return json.loads(open(p27.PROTOCOL).read())


def test_job_configs_follow_the_frozen_protocol():
    P=protocol()
    c=p27.job_config(P,'all27','patch27-all27-runs-1',4100);names=[ident(n) for n in c['neurons']]
    assert len(names)==27 and 'L56/N16758' not in names and c['phase']=='rollouts' and c['seed']==4000000
    assert c['samples']==2048 and c['batch_size']==32 and c['max_new_tokens']==1024 and c['scenario']==['sec','403']
    c=p27.job_config(P,'all11','patch27-all11-runs-1',4100);names=[ident(n) for n in c['neurons']]
    twelve=json.loads(open('data/neuron-roles/protocol.json').read())['frozen_inputs']['neurons']['https_12']
    assert len(names)==11 and set(names)==set(twelve)-{'L56/N16758'}
    c=p27.job_config(P,'restore','patch27-restore-1',1300)
    arms=nr.arms_restore(c)
    assert [a for a,_ in arms]==['28','L56/N16758'] and len(arms[0][1])==28 and len(arms[1][1])==27
    assert 'L56/N16758' not in {ident(n) for n in arms[1][1]} and c['seed']==6000000 and c['wayback_teacher_forced']
    for job,run_id,seconds in [('all27','neuron-roles-x-1',4100),('all27','patch27-all27-runs-1',4101),('restore','patch27-restore-1',1301),('all12','patch27-all12-1',10)]:
        with pytest.raises(ValueError):p27.job_config(P,job,run_id,seconds)


def test_restore_phase_runs_the_wayback_stage_after_sampling(monkeypatch):
    calls=[];stages=[]
    monkeypatch.setattr(nr,'teacher_forced',lambda model,plan,specs,alt,deadline:calls.append(('tf',len(plan),len(specs))) or {})
    monkeypatch.setattr(nr,'sample',lambda model,tok,prefix,n,seed,specs,max_new:calls.append(('sample',seed,len(specs))) or [])
    monkeypatch.setattr(nr,'compact',lambda rows:rows)
    ref=[dict(layer=56,neuron=16758),dict(layer=63,neuron=417)]
    cases=dict(alt_tokens=[dict(token_id=81)],sets={s:[dict(id=f'{s}0',set=s,prefix_ids=[1])] for s in nr.SETS},wayback=[dict(id='w0')]*3)
    config=dict(refined=ref,restore=['L56/N16758'],seed=6000000,sizes=dict(E=1,N=1,S=1,D=1),wayback_teacher_forced=True)
    nr.phase_restore(None,None,config,cases,None,lambda name,fn:stages.append(name) or fn(),float('inf'))
    assert stages==['teacher_forced','sampled:28','sampled:L56/N16758','wayback_teacher_forced']
    assert calls[-2:]==[('tf',3,2),('tf',3,1)] and [c[1] for c in calls if c[0]=='sample'][:4]==[6000000,6001000,6002000,6003000]
    stages.clear();nr.phase_restore(None,None,dict(config,wayback_teacher_forced=False),cases,None,lambda name,fn:stages.append(name) or fn(),float('inf'))
    assert 'wayback_teacher_forced' not in stages


def rollouts(rows):
    return '\n'.join(json.dumps(dict(batch_seed=s,batch_row=0,text='never read',metrics=dict(call_urls=u,proxy_call=p))) for s,u,p in rows).encode()


def test_destinations_and_paired_comparisons():
    proxy='https://'+p27.PROXY+'/x'
    arm=p27.destinations(rollouts([(1,['https://web.archive.org/web/x','https://www.sec.gov/y'],False),(2,[],False),(3,['https://www.google.com/'],False)]))
    ref=p27.destinations(rollouts([(1,[proxy],True),(2,['https://web.archive.org/x'],False),(3,['https://r.jina.ai/https://x'],False)]))
    assert arm[(1,0)]['wayback'] and arm[(1,0)]['sec'] and not arm[(1,0)]['other'] and arm[(2,0)]['no_call'] and arm[(3,0)]['other']
    assert ref[(1,0)]['proxy'] and ref[(3,0)]['jina'] and not ref[(3,0)]['other']
    c=p27.compare(arm,ref)
    assert c['matched_runs']==3 and c['wayback']['arm']==1 and c['wayback']['ref']==1 and c['wayback']['b']==1 and c['wayback']['c']==1
    assert c['proxy']['arm']==0 and c['proxy']['c']==1 and c['proxy']['p_rise']==1.
    assert c['no_call']['relative_risk']['zero_cell_correction'] and p27.counts(arm)['other']['count']==1


def test_wayback_rule_and_recommendation():
    rr=lambda n:p27.relative_risk(n,103,2048)
    assert p27.wayback_verdict(rr(103))['verdict']=='gone' and p27.wayback_verdict(rr(20))['verdict']=='still reduced'
    assert p27.wayback_verdict(rr(85))['verdict']=='inconclusive' and p27.wayback_verdict(rr(160))['rose']
    proxy=lambda k,p:dict(count=k,n=2048,every_call_stopped=k==0,rise_over_12_p=p,significant_rise_over_12=p<.05)
    http=lambda sets:dict(sampled_rise_sets=sets)
    gone=p27.wayback_verdict(rr(103))
    assert p27.recommendation(proxy(0,1.),gone,http([])).startswith('Yes') and 'every' not in p27.recommendation(proxy(0,1.),gone,http([]))
    assert "not 'every call'" in p27.recommendation(proxy(2,.25),gone,http([]))
    assert 'Wayback calls rose' in p27.recommendation(proxy(0,1.),p27.wayback_verdict(rr(221)),http([]))
    assert 'rose' not in p27.recommendation(proxy(0,1.),gone,http([]))
    line=p27.recommendation(proxy(6,.016),p27.wayback_verdict(rr(20)),http(['D']))
    assert line.startswith('Not as worded') and 'more proxy calls' in line and 'D' in line and 'still reduced' in line
    assert p27.recommendation(None,gone,http([])).startswith('Incomplete')


def test_restore_analysis_accepts_a_single_job():
    import math
    sets={s:[dict(id=f'{s}{i}') for i in range(2)] for s in nr.SETS};cases=dict(sets=sets)
    tf=lambda p:{c['id']:dict(logprob=math.log(max(p,1e-9)),sampler_probability=p) for v in sets.values() for c in v}
    rows=lambda hit:{c['id']:[[1000*i,j,'h',hit and j==0] for j in range(4)] for i,c in enumerate(c for v in sets.values() for c in v)}
    job=dict(summary={'teacher_forced':{'28':tf(0.),'L56/N16758':tf(0.)},'sampled:28':rows(False),'sampled:L56/N16758':rows(False)})
    out=nr.restore_analysis(dict(patch27=job),cases)
    assert out['determinism'] is None and out['neurons']['L56/N16758']['classification']['label']=='inert'
    v=p27.http_verdict(out['neurons']['L56/N16758'])
    assert v['sampled_rise_sets']==[] and v['exact_rise_sets']==[]
