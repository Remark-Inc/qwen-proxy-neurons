"""Larger SEC 403 rerun of the frozen sparsity arms, for the blog review's confidence-interval question.

Generation reuses runner.generation unchanged. Chunks only add checkpoints: chunk c calls
generation(seed=S+c*chunk), so batch b still runs under torch.manual_seed(S+32*b), as one
uninterrupted call would. Tool calls are recorded as text and never executed. The analysis
reads outcome flags only, so no generated text (or memorized credential) reaches a report.
"""
import json,math,random,re,sys,time
from pathlib import Path

from qwen_lab.fixtures import scenario
from qwen_lab.metrics import paired_proxy_comparison,summarize,wilson

RATE=.001261+4*.0000131+96*.00000222
Z=1.959963984540054
KEYS=['proxy_call','expected_host_call','any_tool_call','other_unrequested_host_call','truncated','unfinished_tool_call']
SECRET=re.compile(r'OSSAccessKeyId|Signature=|LTAI')


def metadata(model):
    import torch,transformers
    from qwen_lab import MODEL,REVISION
    return dict(model=MODEL,revision=REVISION,torch=torch.__version__,transformers=transformers.__version__,
                gpu=torch.cuda.get_device_name(),reasoning_effort='medium',sampling=dict(temperature=1.,top_p=.95,top_k=20))


def run(config,root,commit,load=None,generate=None,describe=metadata,clock=time.monotonic):
    """Container side: one arm in checkpointed chunks. Rollouts stay on the volume."""
    from qwen_lab import runner
    load=load or runner.load_model;generate=generate or runner.generation
    started=clock();deadline=started+config['seconds']
    folder=Path(root)/config['run_id'];folder.mkdir(parents=True,exist_ok=False)
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    model,tok=load();meta=describe(model)
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    item=scenario(*config['scenario']);n,chunk=config['samples'],config['chunk'];rows=[]
    for start in range(0,n,chunk):
        if clock()>=deadline:break
        rows+=generate(model,tok,item,min(chunk,n-start),config['batch_size'],config['max_new_tokens'],
                       config['seed']+start,config['neurons'],deadline,folder/'rollouts.jsonl')
        progress=dict(completed=len(rows),planned=n,proxy_calls=sum(r['metrics']['proxy_call'] for r in rows),
                      elapsed_seconds=round(clock()-started,1))
        (folder/'progress.json').write_text(json.dumps(progress));commit()
        print(json.dumps(progress),flush=True)
    elapsed=clock()-started
    out=dict(summary=dict(scenario=item['id'],counts=summarize(rows),completed=len(rows),planned=n,complete=len(rows)==n),
             metadata=meta,config=config,elapsed_seconds=elapsed,estimated_compute_usd=elapsed*RATE,
             cost_note='Resource-rate estimate, not an invoice; excludes startup/build/storage.')
    (folder/'result.json').write_text(json.dumps(out,indent=2));commit()
    return dict(run_id=config['run_id'],complete=out['summary']['complete'],completed=len(rows),
                elapsed=elapsed,estimated_compute_usd=out['estimated_compute_usd'])


def outcomes(path):
    """(batch_seed, batch_row) -> outcome flags. Text is dropped here."""
    out={}
    for line in Path(path).read_text().splitlines():
        r=json.loads(line);out[(r['batch_seed'],r['batch_row'])]={k:bool(r['metrics'].get(k)) for k in KEYS}
    return out


def katz(a,n1,c,n2):
    """1 - RR for a/n1 against c/n2 with the Katz log-RR 95% interval. Adds 0.5 to every
    cell (Haldane-Anscombe) only when a count is zero, and says so."""
    corrected=not a or not c
    if corrected:a,n1,c,n2=a+.5,n1+1,c+.5,n2+1
    rr=(a/n1)/(c/n2);se=math.sqrt(1/a-1/n1+1/c-1/n2)
    return dict(reduction=1-rr,ci95=[1-rr*math.exp(Z*se),1-rr*math.exp(-Z*se)],zero_cell_correction=corrected)


def multinomial(rng,n,probs):
    out=[];left=n;mass=1.
    for p in probs[:-1]:
        k=rng.binomialvariate(left,min(1.,max(0.,p/mass))) if left and mass>0 else 0
        out.append(k);left-=k;mass-=p
    return out+[left]


def percentile_ci(vals):
    vals=sorted(vals);return [vals[int(.025*len(vals))],vals[int(.975*len(vals))-1]]


def paired_bootstrap(arm,base,reps=10000,seed=20261030):
    """Percentile 95% CI for 1 - RR, resampling matched (seed, row) units with replacement.
    A resample of n paired binary units is a multinomial draw over the four cell types."""
    keys=sorted(set(arm)&set(base));n=len(keys);cells=[0,0,0,0]  # both, arm only, baseline only, neither
    for k in keys:cells[2*(not arm[k])+(not base[k])]+=1
    rng=random.Random(seed);vals=[]
    for _ in range(reps):
        both,arm_only,base_only,_=multinomial(rng,n,[c/n for c in cells])
        if both+base_only:vals.append(1-(both+arm_only)/(both+base_only))
    return dict(ci95=percentile_ci(vals),matched_units=n,resamples=reps,rng_seed=seed,
                cells=dict(both=cells[0],arm_only=cells[1],baseline_only=cells[2],neither=cells[3]))


def cluster_bootstrap(arm,base,reps=10000,seed=20261031):
    """Sensitivity check: resample whole 32-row batches, which share one RNG seed."""
    batches={}
    for k in set(arm)&set(base):
        a,c=batches.get(k[0],(0,0));batches[k[0]]=(a+arm[k],c+base[k])
    units=list(batches.values());rng=random.Random(seed);vals=[]
    for _ in range(reps):
        draw=rng.choices(units,k=len(units));c=sum(u[1] for u in draw)
        if c:vals.append(1-sum(u[0] for u in draw)/c)
    return dict(ci95=percentile_ci(vals),batches=len(units),resamples=reps,rng_seed=seed)


def count(k,n):return dict(count=k,n=n,rate=k/n if n else None,wilson95=wilson(k,n))


def adjacent(x,y):
    keys=set(x)&set(y)
    rows=lambda d:[dict(scenario='sec:403',batch_seed=k[0],batch_row=k[1],metrics=dict(proxy_call=d[k])) for k in sorted(keys)]
    return paired_proxy_comparison(rows(x),rows(y))


def build(protocol='data/sparsity-scale/protocol.json',runs='runs',previous='sparsity-results.json',run_ids=None):
    P=json.loads(Path(protocol).read_text());ledger=json.loads(Path(P['ledger']).read_text())
    prev=json.loads(Path(previous).read_text())['confirmation'];jobs={j['run_id']:j for j in ledger['jobs']}
    run_ids=run_ids or {}
    arms,flags={},{}
    for arm in P['arms']+list(P['optional_arms']):
        run_id=run_ids.get(arm,P['run_id_pattern'].format(arm=arm));folder=Path(runs)/run_id
        if not (folder/'rollouts.jsonl').exists():continue
        rows=outcomes(folder/'rollouts.jsonl');flags[arm]={k:v['proxy_call'] for k,v in rows.items()}
        res=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
        job=jobs.get(run_id,{});n=len(rows)
        arms[arm]=dict(run_id=run_id,neurons=[f"L{s['layer']}/N{s['neuron']}" for s in res.get('config',{}).get('neurons',[])],
                       completed=n,planned=P['samples_per_arm'],incomplete=P['samples_per_arm']-n,
                       **{k:count(sum(v[k] for v in rows.values()),n) for k in KEYS},
                       gpu=res.get('metadata',{}).get('gpu'),elapsed_seconds=res.get('elapsed_seconds'),
                       app_wallclock_seconds=job.get('app_wallclock_seconds'),app_wallclock_estimate_usd=job.get('app_wallclock_estimate_usd'),
                       outcome_bits=''.join('1' if rows[k]['proxy_call'] else '0' for k in sorted(rows)))
    b=arms['baseline']['proxy_call'];pb=prev['baseline']['scenarios']['sec:403']
    reductions,pooled={},{}
    for arm,v in arms.items():
        if arm=='baseline':continue
        a=v['proxy_call'];zero=not a['count']
        reductions[arm]=dict(point=1-(a['count']/a['n'])/(b['count']/b['n']),katz=katz(a['count'],a['n'],b['count'],b['n']),
                             paired_bootstrap=None if zero else paired_bootstrap(flags[arm],flags['baseline']),
                             cluster_bootstrap=None if zero else cluster_bootstrap(flags[arm],flags['baseline']))
        if zero:reductions[arm]['note']='No proxy calls: bootstraps are degenerate; use the Wilson upper bound and the zero-cell-corrected Katz bound.'
        if arm in prev:
            pa=prev[arm]['scenarios']['sec:403'];k1,n1=a['count']+pa['proxy_call']['count'],a['n']+pa['n']
            k2,n2=b['count']+pb['proxy_call']['count'],b['n']+pb['n']
            pooled[arm]=dict(arm=count(k1,n1),baseline=count(k2,n2),point=1-(k1/n1)/(k2/n2),katz=katz(k1,n1,k2,n2))
    pairs=[('single','pair'),('pair','triple'),('triple','all12')]
    cost=sum(j.get('app_wallclock_estimate_usd',j['reservation_usd']) for j in ledger['jobs'] if j['status']!='not_started')
    out=dict(protocol=dict(file=protocol,seed=P['seed'],samples_per_arm=P['samples_per_arm'],scenario='sec:403',
                           generation=P['generation'],conditions_sha256=P['arm_sources']['baseline']['sha256']),
             arms=arms,reductions=reductions,pooled_with_confirmation=pooled,
             adjacent={f'{x}_vs_{y}':adjacent(flags[x],flags[y]) for x,y in pairs if x in flags and y in flags},
             previous_confirmation={**{k:dict(proxy_call=f"{v['scenarios']['sec:403']['proxy_call']['count']} of {v['scenarios']['sec:403']['n']}")
                                       for k,v in prev.items() if k in P['arms']+list(P['optional_arms'])},
                                    **{k:dict(proxy_call=v) for k,v in P.get('earlier_results',{}).items()}},
             cost=dict(phase_estimate_usd=cost,research_estimate_usd=ledger['prior_research_estimate_usd']+cost,
                       research_cap_usd=ledger['research_cap_usd'],method=ledger['method']),
             outcome_bits_order='Matched units sorted by (batch_seed, batch_row); 1 = proxy call')
    text=json.dumps(out,indent=2)
    if SECRET.search(text):raise ValueError('Credential pattern in results')
    return out


def pct(x):return f'{100*x:.0f}%'


def ci(c):return f'{100*c[0]:.0f}–{100*c[1]:.0f}%'


def tables(r):
    names=dict(baseline='0 (unmodified)',single='1 (L63/N4455)',pair='2 (+ L63/N417)',triple='3 (+ L63/N14734)',all12='12 (https:// patch)')
    head='| Neurons ablated | Proxy calls | Wilson 95% | Reduction vs matched unmodified (Katz 95%) | Paired bootstrap 95% | Earlier 256-run result |'
    rows=[head,'|---|---|---|---|---|---|']
    for arm,v in r['arms'].items():
        p=v['proxy_call'];red=r['reductions'].get(arm)
        rows.append(f"| {names[arm]} | {p['count']} of {p['n']} ({100*p['rate']:.1f}%) | {100*p['wilson95'][0]:.1f}–{100*p['wilson95'][1]:.1f}% | "
                    +(f"{pct(red['point'])} ({ci(red['katz']['ci95'])})" if red else '—')+' | '
                    +(ci(red['paired_bootstrap']['ci95']) if red and red['paired_bootstrap'] else '—')
                    +f" | {r['previous_confirmation'].get(arm,{}).get('proxy_call','—')} |")
    util=['| Arm | sec.gov calls | Any tool call | Other unrequested host | Truncated (1,024 tokens) | Unfinished tool call | Not run |','|---|---|---|---|---|---|---|']
    for arm,v in r['arms'].items():
        util.append(f"| {names[arm]} | "+' | '.join(f"{v[k]['count']} of {v[k]['n']}" for k in KEYS[1:])+f" | {v['incomplete']} |")
    pool=['| Neurons ablated | Pooled proxy calls | Pooled unmodified | Reduction (Katz 95%) |','|---|---|---|---|']
    for arm,v in r['pooled_with_confirmation'].items():
        pool.append(f"| {names[arm]} | {v['arm']['count']} of {v['arm']['n']} | {v['baseline']['count']} of {v['baseline']['n']} | {pct(v['point'])} ({ci(v['katz']['ci95'])}) |")
    return '\n'.join(rows),'\n'.join(util),'\n'.join(pool)


if __name__=='__main__':
    if sys.argv[1]=='build':
        r=build();Path('sparsity-scale-results.json').write_text(json.dumps(r,indent=2)+'\n')
        print('\n\n'.join(tables(r)))
    elif sys.argv[1]=='tables':print('\n\n'.join(tables(json.loads(Path('sparsity-scale-results.json').read_text()))))
