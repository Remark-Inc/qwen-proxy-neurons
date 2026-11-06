"""Coverage sweep: build coverage-sweep-results.json from the item-1, item-3 and item-7 runs.

Reads outcome flags, probabilities and hostnames only, never generated text.
"""
import collections,hashlib,json,math,sys
from pathlib import Path
from qwen_lab.coverage_forms import bash_analysis,bash_flags,forms_analysis
from qwen_lab.http_path import SECRET
from qwen_lab.metrics import PROXY,wilson
from qwen_lab.neuron_roles import mcnemar_two_sided
from qwen_lab.sparsity_scale import katz

PROTOCOL='data/coverage-sweep/protocol.json'
HF_UNMODIFIED=dict(run='sparsity-scale-baseline-1',proxy=287,n=2048)


def rr(x,n1,y,n2):
    """x/n1 against y/n2: relative risk with the Katz 95% interval (zero-cell corrected)."""
    if not x and not y:return None
    k=katz(x,n1,y,n2);return dict(rr=1-k['reduction'],ci95=[1-k['ci95'][1],1-k['ci95'][0]],zero_cell_correction=k['zero_cell_correction'])


def consistent(r):return None if r is None else ('consistent' if r['ci95'][0]<=1<=r['ci95'][1] else 'differs')


def vllm_rows(path):
    out=collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r=json.loads(line);m=r['metrics']
        out[(r.get('engine') or r['precision'],r['arm'])].append(dict(seed=r['batch_seed'],proxy=bool(m['proxy_call']),expected=bool(m['expected_host_call']),
                                                   any_call=bool(m['any_tool_call']),truncated=bool(m['truncated']),
                                                   hosts=sorted({u.split('://',1)[-1].split('/',1)[0].lower() for u in m['call_urls']})))
    return out


def vllm_analysis(runs,forms_tf):
    """Per (precision, arm): proxy calls with Wilson, engine-match checks, probes against the Transformers teacher forcing."""
    out=dict(arms={},engines={},matches={})
    for run in runs:
        R=json.loads((run/'result.json').read_text());rows=vllm_rows(run/'rollouts.jsonl') if (run/'rollouts.jsonl').exists() else {}
        for prec,e in R['summary'].get('engines',{}).items():
            out['engines'][f'{run.name}:{prec}']=dict(prepare=e.get('prepare'),timing=e.get('timing'),vllm=e.get('vllm'),gpu=e.get('gpu'),complete=e.get('complete'))
            for label,a in e.get('arms',{}).items():
                rs=rows.get((prec,label),[]);k=sum(r['proxy'] for r in rs);n=len(rs)
                probes={pid:dict(vllm=s,transformers=forms_tf.get(a['neurons'],{}).get(pid)) for pid,s in a['probe'].items()}
                for v in probes.values():
                    t=v['transformers'];v['difference_nats']=None if not t else v['vllm']['logprob']-t['logprob']
                out['arms'][f'{prec}:{label}']=dict(run=run.name,precision=prec,arm=label,neurons=a['neurons'],mode=a['mode'],n=n,proxy=k,wilson95=wilson(k,n),
                    expected_host=sum(r['expected'] for r in rs),truncated=sum(r['truncated'] for r in rs),receipts=a['receipts'],probes=probes,
                    generation_seconds=a['generation_seconds'],output_tokens=a['output_tokens'],
                    hosts=collections.Counter(h for r in rs for h in r['hosts']).most_common(8),
                    verdict=None if a['neurons']=='unmodified' else ('holds' if k==0 else 'leaks'))
            if R['summary'].get('errors'):out['engines'][f'{run.name}:errors']=R['summary']['errors']
    A=out['arms']
    if 'nvfp4-unedited:unmodified' in A and 'bf16:unmodified' in A:
        a,b=A['nvfp4-unedited:unmodified'],A['bf16:unmodified'];r=rr(a['proxy'],a['n'],b['proxy'],b['n'])
        out['matches']['nvfp4_b200_vs_vllm_bf16']=dict(arm=f"{a['proxy']} of {a['n']}",bf16=f"{b['proxy']} of {b['n']}",rr=r,verdict=consistent(r))
    if 'bf16:unmodified' in A:
        a=A['bf16:unmodified'];r=rr(a['proxy'],a['n'],HF_UNMODIFIED['proxy'],HF_UNMODIFIED['n'])
        out['matches']['vllm_bf16_vs_transformers']=dict(vllm=f"{a['proxy']} of {a['n']}",transformers=f"{HF_UNMODIFIED['proxy']} of {HF_UNMODIFIED['n']}",rr=r,verdict=consistent(r))
    for q in ('fp8','nvfp4'):
        if f'{q}:unmodified' in A and 'bf16:unmodified' in A:
            a,b=A[f'{q}:unmodified'],A['bf16:unmodified'];r=rr(a['proxy'],a['n'],b['proxy'],b['n'])
            out['matches'][f'{q}_vs_vllm_bf16']=dict(arm=f"{a['proxy']} of {a['n']}",bf16=f"{b['proxy']} of {b['n']}",rr=r,verdict=consistent(r))
    return out


def search_analysis(result,forms_result,cases):
    """1c: what the search added, overlap with the 28, and the held-out check against the 27 and unmodified."""
    from qwen_lab.coverage_forms import form_verdict
    from qwen_lab.neuron_roles import mean
    S=result['summary'];r=S.get('search_round') or {};P=json.loads(Path(PROTOCOL).read_text())
    order=json.loads(Path('data/neuron-roles/protocol.json').read_text())['frozen_inputs']['neurons']['order']
    out=dict(errors=S.get('errors'),timing=S.get('timing'),added=r.get('added'),neurons=r.get('neurons'),
             overlap_with_28=sorted(set(r.get('added') or [])&set(order)),l56_n16758_readded='L56/N16758' in (r.get('added') or []),
             start_objective=(r.get('start_objective') or {}).get('value'),final_objective=(r.get('objective') or {}).get('value'),
             final_gate={k:v for k,v in (r.get('final_gate') or {}).items() if k!='drops'},additions=r.get('additions'),pruned=r.get('pruned'),
             screened=(r.get('screened') or [])[:12],heldout={})
    TF=S.get('heldout_teacher_forced',{});SM=S.get('heldout_sampled',{});F=forms_result['summary']
    for name in cases['forms']:
        ids=[f"{name}:{t['id']}" for t in cases['transcripts']];tf=[TF[i] for i in ids if i in TF]
        rows=[x for i in ids for x in SM.get(i,[])];k=sum(x[3] for x in rows)
        u=[F['teacher_forced']['unmodified'][i] for i in ids];u_rows=[x for i in ids for x in F['sampled:unmodified'].get(i,[])]
        b=[F['teacher_forced']['27'][i] for i in ids];b_rows=[x for i in ids for x in F['sampled:27'].get(i,[])]
        exact=mean(x['sampler_probability'] for x in tf) if tf else None
        out['heldout'][name]=dict(search_set_exact=exact,search_set_k=k,search_set_n=len(rows),start_27_exact=mean(x['sampler_probability'] for x in b),
            start_27_k=sum(x[3] for x in b_rows),start_27_n=len(b_rows),unmodified_exact=mean(x['sampler_probability'] for x in u),unmodified_k=sum(x[3] for x in u_rows),
            verdict=None if exact is None else form_verdict(mean(x['sampler_probability'] for x in u),sum(x[3] for x in u_rows),exact,k if rows else 0),
            sampled=bool(rows))
    return out


def forms_teacher_forced(result):
    """arm label -> case id -> Transformers teacher-forced proxy-host log-probability."""
    return {a:{k:dict(logprob=v['logprob'],sampler_probability=v['sampler_probability']) for k,v in tf.items()} for a,tf in result['summary']['teacher_forced'].items()}


def build(protocol=PROTOCOL,runs='runs'):
    P=json.loads(Path(protocol).read_text());F=P['frozen_inputs'];root=Path(runs);out=dict(protocol=protocol)
    cases=json.loads((Path('data')/F['forms_cases']['path']).read_text())
    if hashlib.sha256((Path('data')/F['forms_cases']['path']).read_bytes()).hexdigest()!=F['forms_cases']['sha256']:raise ValueError('Case file changed')
    scan=json.loads(Path(F['host_scan']['path']).read_text())
    out['item3']=dict(other_hosts=scan['other_hosts'],by_host=scan['by_host'],records=sum(r['records'] for r in scan['scanned']),
                      runs=len({r['run'] for r in scan['scanned']}))
    forms=root/'coverage-forms-1'/'result.json';forms_tf={}
    if forms.exists():
        R=json.loads(forms.read_text());meta=dict(errors=R['summary'].get('errors'),timing=R['summary'].get('timing'),complete=R['summary'].get('complete'))
        top=root/'coverage-forms-27-rest-1'/'result.json'
        if top.exists():
            T=json.loads(top.read_text());extra=T['summary'].get('sampled:27',{})
            # The top-up samples only prefixes the first job did not reach; the 27 arm's counts combine both.
            if set(extra)&set(R['summary']['sampled:27']):raise ValueError('Top-up overlaps the first job')
            R['summary']['sampled:27']={**R['summary']['sampled:27'],**extra}
            a,b=R['summary']['teacher_forced']['27'],T['summary']['teacher_forced']['27']
            meta['top_up']=dict(run='coverage-forms-27-rest-1',sampled_prefixes=len(extra),errors=T['summary'].get('errors'),
                                teacher_forced_max_abs_logprob_diff_27=max(abs(a[k]['logprob']-b[k]['logprob']) for k in a.keys()&b.keys()))
        out['item1_forms']=forms_analysis(R,cases);forms_tf=forms_teacher_forced(R);out['item1_forms_meta']=meta
    search=root/'coverage-search-1'/'result.json'
    if search.exists() and forms.exists():out['item1_search']=search_analysis(json.loads(search.read_text()),R,cases)
    flags={a:bash_flags(root/f'coverage-bash-{a}-1'/'rollouts.jsonl') for a in ('unmodified','28','27') if (root/f'coverage-bash-{a}-1'/'rollouts.jsonl').exists()}
    if flags:out['item1_bash']=bash_analysis(flags)
    vruns=[root/f'coverage-vllm-{j}-1' for j in ('bf16','fp8','nvfp4','b200-nvfp4') if (root/f'coverage-vllm-{j}-1'/'result.json').exists()]
    if vruns:out['item7']=vllm_analysis(vruns,forms_tf)
    quant=root/'coverage-nvfp4-quant-1'/'result.json'
    if quant.exists():
        Q=json.loads(quant.read_text())['summary']
        out['item7_nvfp4_checkpoints']=dict(errors=Q.get('errors'),timing=Q.get('timing'),scales=Q.get('scales'),
            quantization_config=(Q.get('unedited') or {}).get('quantization_config'),bytes={k:(Q.get(k) or {}).get('bytes') for k in ('unedited','28_before')},
            after_receipts={k:(Q.get(k) or {}).get('receipts') for k in ('28_after','27_after')})
        after=next((root/f'coverage-nvfp4-after-{i}'/'result.json' for i in (2,1) if (root/f'coverage-nvfp4-after-{i}'/'result.json').exists()),root/'none')
        if after.exists():
            A=json.loads(after.read_text())['summary'];out['item7_nvfp4_checkpoints']['after_receipts']={k:A.get(k) for k in ('28_after','27_after')}
            out['item7_nvfp4_checkpoints']['after_errors']=A.get('errors')
    smoke=root/'coverage-vllm-smoke-1'/'result.json'
    if smoke.exists():out['item7_smoke']=vllm_analysis([smoke.parent],forms_tf)
    L=json.loads(Path(P['budget']['ledger']).read_text())
    out['cost']={k:L.get(k) for k in ('phase_cap_usd','research_cap_usd','phase_estimate_usd','research_estimate_usd','prior_research_estimate_usd')}|dict(
        jobs=[{k:j.get(k) for k in ('run_id','status','app_wallclock_seconds','app_wallclock_estimate_usd','measured_compute_usd')} for j in L['jobs']])
    text=json.dumps(out,indent=1)
    if SECRET.search(text):raise ValueError('Credential pattern in results')
    Path('coverage-sweep-results.json').write_text(text+'\n');return out


def pct(x,d=2):return '—' if x is None else f'{100*x:.{d}f}%'


def context_table(r):
    """The per-context table: context, unmodified, patched (28 / 27), verdict."""
    rows=['| Context | Unmodified | 28 off | 27 off | Verdict (28 / 27) |','|---|---|---|---|---|']
    for name,v in r.get('item1_forms',{}).items():
        cell=lambda a:f"{pct(v[a]['exact_rate'])} exact; {v[a]['k']}/{v[a]['n']} sampled"
        rows.append(f"| Decision point: {name} | {cell('unmodified')} | {cell('28')} | {cell('27')} | {v['28']['verdict']} / {v['27']['verdict']} |")
    b=r.get('item1_bash')
    if b:
        cell=lambda a:f"{b[a]['proxy']['count']} of {b[a]['n']} runs" if a in b else '—'
        rows.append(f"| SEC 403 runs, bash tool only | {cell('unmodified')} | {cell('28')} | {cell('27')} | {b.get('28',{}).get('verdict','—')} / {b.get('27',{}).get('verdict','—')} |")
    A=r.get('item7',{}).get('arms',{})
    for prec in ('bf16','fp8','nvfp4'):
        if f'{prec}:unmodified' not in A:continue
        u=A[f'{prec}:unmodified'];p28=A.get(f'{prec}:28') or A.get(f'{prec}:28 after');p27=A.get(f'{prec}:27') or A.get(f'{prec}:27 after')
        cell=lambda a:'—' if not a else f"{a['proxy']} of {a['n']} runs"
        word='rounding' if prec=='nvfp4' else 'quantization'
        extra=f"; edited before {word}: {A[f'{prec}:28 before']['proxy']} of {A[f'{prec}:28 before']['n']}" if f'{prec}:28 before' in A else ''
        name={'bf16':'vLLM bf16, batched','fp8':'vLLM FP8, batched','nvfp4':'vLLM batched, NVFP4-rounded weights (W4A16 emulation)'}[prec]
        rows.append(f"| SEC 403 runs, {name} | {cell(u)} | {cell(p28)}{extra} | {cell(p27)} | {(p28 or {}).get('verdict','—')} / {(p27 or {}).get('verdict','—')} |")
    if 'nvfp4-unedited:unmodified' in A:
        g=lambda k:A.get(k);cell=lambda a:'—' if not a else f"{a['proxy']} of {a['n']} runs"
        p28,p27,pb=g('nvfp4-28-after:28 after'),g('nvfp4-27-after:27 after'),g('nvfp4-28-before:28 before')
        rows.append(f"| SEC 403 runs, real NVFP4 checkpoints on a B200 (native FP4) | {cell(g('nvfp4-unedited:unmodified'))} | {cell(p28)}; edited before quantization: {cell(pb)} | {cell(p27)} | {(p28 or {}).get('verdict','—')} / {(p27 or {}).get('verdict','—')} |")
    return '\n'.join(rows)


if __name__=='__main__' and sys.argv[1:2]==['build']:
    r=build();print(context_table(r))
elif __name__=='__main__' and sys.argv[1:2]==['tables']:
    print(context_table(json.loads(Path('coverage-sweep-results.json').read_text())))
