import json

from qwen_lab import sparsity_scale as ss


def fake_generation(calls,proxy=lambda s,j:(s+j)%7==0):
    """Mimics runner.generation's seed schedule and row format; text carries a fake credential."""
    def generate(model,tok,item,n,batch_size,max_new,seed,neurons,deadline,path):
        rows=[]
        for start in range(0,n,batch_size):
            for j in range(min(batch_size,n-start)):
                row=dict(scenario=item['id'],batch_seed=seed+start,batch_row=j,text='?OSSAccessKeyId=LTAIfake&Signature=x',
                         metrics=dict(proxy_call=proxy(seed+start,j),expected_host_call=True,any_tool_call=True,
                                      other_unrequested_host_call=False,truncated=False,unfinished_tool_call=False))
                with open(path,'a') as f:f.write(json.dumps(row)+'\n')
                rows.append(row)
        calls.append((n,seed));return rows
    return generate


def config(**kw):
    return dict(dict(run_id='sparsity-scale-single-1',arm='single',neurons=[dict(layer=63,neuron=4455,mode='scale',value=0)],
                     scenario=['sec','403'],samples=2048,chunk=256,seed=4000000,batch_size=32,max_new_tokens=1024,seconds=4100),**kw)


def run(tmp_path,cfg,calls,commits,clock=None):
    kw=dict(clock=clock) if clock else {}
    return ss.run(cfg,tmp_path,lambda:commits.append(1),load=lambda:(None,None),generate=fake_generation(calls),
                  describe=lambda m:dict(gpu='test'),**kw)


def test_chunked_seeds_match_one_uninterrupted_call(tmp_path):
    calls,commits=[],[]
    receipt=run(tmp_path,config(),calls,commits)
    assert calls==[(256,4000000+256*c) for c in range(8)] and len(commits)==9
    rows=ss.outcomes(tmp_path/'sparsity-scale-single-1'/'rollouts.jsonl')
    assert {s for s,_ in rows}=={4000000+32*b for b in range(64)} and len(rows)==2048
    one=[];fake_generation(one)(None,None,dict(id='sec:403'),2048,32,1024,4000000,[],0,tmp_path/'one.jsonl')
    assert {(r['batch_seed'],r['batch_row']) for r in map(json.loads,(tmp_path/'one.jsonl').read_text().splitlines())}==set(rows)
    assert receipt['complete'] and receipt['completed']==2048


def test_deadline_stops_between_chunks_and_keeps_checkpoints(tmp_path):
    ticks=iter(range(0,10**6,1000));calls,commits=[],[]
    receipt=run(tmp_path,config(seconds=2500),calls,commits,clock=lambda:next(ticks))
    out=json.loads((tmp_path/'sparsity-scale-single-1'/'result.json').read_text())
    assert calls==[(256,4000000)] and len(commits)==2 and not receipt['complete']
    assert out['summary']['completed']==256 and out['summary']['planned']==2048 and not out['summary']['complete']


def test_katz_reproduces_the_blog_intervals():
    blog={18:(54,22,73),8:(79,57,90),6:(85,64,93)}
    for k,(point,lo,hi) in blog.items():
        r=ss.katz(k,256,39,256)
        assert (round(100*r['reduction']),round(100*r['ci95'][0]),round(100*r['ci95'][1]))==(point,lo,hi)
        assert not r['zero_cell_correction']
    assert ss.katz(0,2048,300,2048)['zero_cell_correction']


def test_paired_bootstrap_is_deterministic_and_brackets_the_estimate():
    keys=[(4000000+32*b,j) for b in range(64) for j in range(32)]
    base={k:i%7==0 for i,k in enumerate(keys)};arm={k:i%14==0 for i,k in enumerate(keys)}
    a,b=ss.paired_bootstrap(arm,base,reps=2000),ss.paired_bootstrap(arm,base,reps=2000)
    point=1-sum(arm.values())/sum(base.values())
    assert a==b and a['ci95'][0]<point<a['ci95'][1] and a['matched_units']==2048
    assert a['cells']['both']==sum(arm.values()) and a['cells']['arm_only']==0
    c=ss.cluster_bootstrap(arm,base,reps=2000)
    assert c['batches']==64 and c['ci95'][0]<point<c['ci95'][1]


def test_build_reports_counts_only(tmp_path):
    runs=tmp_path/'runs';pattern='sparsity-scale-{arm}-1'
    for arm,proxy in [('baseline',lambda s,j:j<5),('single',lambda s,j:j<2),('all12',lambda s,j:False)]:
        calls,commits=[],[]
        ss.run(config(run_id=pattern.format(arm=arm),arm=arm,samples=512),runs,lambda:None,load=lambda:(None,None),
               generate=fake_generation(calls,proxy),describe=lambda m:dict(gpu='test'))
    ledger=tmp_path/'budget.json';ledger.write_text(json.dumps(dict(jobs=[],prior_research_estimate_usd=145.1,research_cap_usd=175,method='m')))
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(arms=['baseline','single','pair','triple','all12'],optional_arms={},run_id_pattern=pattern,
        ledger=str(ledger),samples_per_arm=2048,seed=4000000,generation={},arm_sources=dict(baseline=dict(sha256='x')),
        earlier_results=dict(all12='0 of 512'))))
    sec=lambda k:{'scenarios':{'sec:403':dict(n=256,proxy_call=dict(count=k))}}
    previous=tmp_path/'prev.json';previous.write_text(json.dumps(dict(confirmation=dict(baseline=sec(39),single=sec(18)))))
    r=ss.build(protocol=str(protocol),runs=str(runs),previous=str(previous))
    text=json.dumps(r)
    assert 'LTAI' not in text and 'Signature' not in text and 'OSSAccessKeyId' not in text
    assert r['arms']['baseline']['proxy_call']['count']==80 and r['arms']['single']['proxy_call']['count']==32
    assert r['arms']['single']['incomplete']==2048-512 and r['arms']['single']['neurons']==['L63/N4455']
    assert abs(r['reductions']['single']['point']-.6)<1e-9 and r['pooled_with_confirmation']['single']['arm']['count']==50
    assert r['reductions']['single']['paired_bootstrap']['cells']['baseline_only']==48
    z=r['reductions']['all12']
    assert r['arms']['all12']['proxy_call']['count']==0 and r['arms']['all12']['proxy_call']['wilson95'][1]>0
    assert z['paired_bootstrap'] is None and z['katz']['zero_cell_correction'] and r['previous_confirmation']['all12']['proxy_call']=='0 of 512'
    assert all(len(t) for t in ss.tables(r))
