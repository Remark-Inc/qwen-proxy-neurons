"""Patch-27 validation: the 28-neuron patch without L56/N16758 on the sparsity rerun's 2,048 SEC
403 runs (and the 12 without it), plus L56/N16758 restored on the held-out http:// sets.

Rollout arms reuse sparsity_scale.run and the restore job reuses neuron_roles.run (item 1's
sets and seeds, plus an optional Wayback teacher-forced stage). Reports read outcome flags,
probabilities and hostnames only, never generated text.
"""
import hashlib,json,math,sys
from pathlib import Path
from qwen_lab.http_path import SECRET,ident
from qwen_lab.metrics import PROXY,host,wilson
from qwen_lab.neuron_roles import SETS,WAYBACK,baseline_agreement,load_cases,mcnemar_one_sided,mcnemar_two_sided,restore_analysis,wayback_analysis
from qwen_lab.sparsity_scale import katz

PROTOCOL='data/patch-27/protocol.json'
NAMED=dict(proxy=PROXY,wayback=WAYBACK,sec='www.sec.gov',efts='efts.sec.gov',jina='r.jina.ai')
OUTCOMES=list(NAMED)+['other','no_call','truncated','unfinished']
NAMES=dict(unmodified='Unmodified',all12='12 off (https:// patch)',n16758_off='L56/N16758 off alone',
           all27='27 off (28 without L56/N16758)',all11='11 off (12 without L56/N16758)')


def frozen(path,sha,parse=True):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError(f'{path} differs from the frozen protocol')
    return json.loads(raw) if parse else raw


def job_config(P,job,run_id,seconds):
    """Container config for one frozen job; refuses anything outside the protocol's bounds."""
    if job not in P['jobs']:raise ValueError(job)
    J=P['jobs'][job];F=P['frozen_inputs']
    if not run_id.startswith(J['run_id_prefix']) or not 1<=seconds<=J['soft_seconds']:raise ValueError('Bounds exceeded')
    ref=frozen(F['neurons']['file'],F['neurons']['sha256'])
    if J['phase']=='rollouts':
        S=frozen(F['settings']['file'],F['settings']['sha256']);G=S['generation']
        off=[n for n in ref if ident(n) in J['neurons_off']]
        if len(off)!=len(J['neurons_off']):raise ValueError('Neuron missing')
        return dict(phase='rollouts',run_id=run_id,arm=job,neurons=off,scenario=S['scenario'],samples=S['samples_per_arm'],chunk=S['chunk'],
                    seed=S['seed'],batch_size=G['batch_size'],max_new_tokens=G['max_new_tokens'],seconds=seconds)
    R=frozen(F['roles_protocol']['file'],F['roles_protocol']['sha256']);RF=R['frozen_inputs'];I=R['item1']
    for e in RF['http_case_files']+[RF['roles_case_file']]:frozen(Path('data')/e['path'],e['sha256'])
    if not set(J['restore'])<={ident(n) for n in ref}:raise ValueError('Neuron missing')
    return dict(phase='restore',run_id=run_id,seconds=seconds,refined=ref,http_case_files=RF['http_case_files'],
                roles_case_file={k:RF['roles_case_file'][k] for k in ('path','sha256')},escape_id=RF['escape_id'],
                traces16=dict(run=RF['traces16']['run']),seed=I['sampling']['seed'],sizes=I['sampling']['sizes'],
                restore=J['restore'],wayback_teacher_forced=J['wayback_teacher_forced'])


# ---------------------------------------------------------------- local analysis

def destinations(raw):
    """(batch_seed, batch_row) -> outcome flags from rollouts.jsonl bytes. A run can call several hosts."""
    out={}
    for line in raw.decode().splitlines():
        r=json.loads(line);m=r['metrics'];hs={host(u) for u in m['call_urls']};f={k:h in hs for k,h in NAMED.items()}
        f.update(proxy=bool(m['proxy_call']),other=bool(hs-set(NAMED.values())),no_call=not hs,
                 truncated=bool(m.get('truncated')),unfinished=bool(m.get('unfinished_tool_call')))
        out[(r['batch_seed'],r['batch_row'])]=f
    return out


def counts(flags):
    n=len(flags);out=dict(n=n)
    for k in OUTCOMES:x=sum(f[k] for f in flags.values());out[k]=dict(count=x,wilson95=wilson(x,n))
    return out


def relative_risk(x,y,n):
    """x/y over n matched runs per arm with the Katz 95% interval (sparsity_scale.katz, zero-cell corrected)."""
    if not x and not y:return None
    k=katz(x,n,y,n);return dict(rr=1-k['reduction'],ci95=[1-k['ci95'][1],1-k['ci95'][0]],zero_cell_correction=k['zero_cell_correction'])


def compare(arm,ref):
    """Matched (batch_seed, batch_row) runs: counts, discordant pairs, exact McNemar and Katz relative risk."""
    keys=sorted(arm.keys()&ref.keys());n=len(keys);out=dict(matched_runs=n)
    for k in OUTCOMES:
        x=sum(arm[u][k] for u in keys);y=sum(ref[u][k] for u in keys)
        b=sum(arm[u][k] and not ref[u][k] for u in keys);c=sum(ref[u][k] and not arm[u][k] for u in keys)
        out[k]=dict(arm=x,ref=y,b=b,c=c,p_two_sided=mcnemar_two_sided(b,c),p_rise=mcnemar_one_sided(b,c),relative_risk=relative_risk(x,y,n))
    return out


def transitions(arm,ref,sources=('proxy','wayback','efts','jina')):
    """Exploratory: for matched runs where the reference called a host, how many runs of the arm called each host.
    Hosts are not exclusive, and paired runs diverge after the first differing token."""
    keys=arm.keys()&ref.keys();out={}
    for s in sources:
        units=[k for k in keys if ref[k][s]]
        out[s]=dict(runs=len(units),**{d:sum(arm[k][d] for k in units) for d in OUTCOMES[:7]})
    w=[k for k in arm if arm[k]['wayback']]
    out['wayback_runs_also_calling']={d:sum(arm[k][d] for k in w) for d in ('sec','efts','jina','other')}|dict(runs=len(w))
    return out


def wayback_verdict(rr,margin=2/3):
    """Frozen rule: 'gone' if the Katz lower bound against unmodified is >= 2/3, 'still reduced' if the
    upper bound is < 1, otherwise 'inconclusive'."""
    lo,hi=rr['ci95']
    return dict(verdict='gone' if lo>=margin else 'still reduced' if hi<1 else 'inconclusive',rose=lo>1,rr=rr['rr'],ci95=rr['ci95'])


def proxy_verdict(count,vs12):
    p=vs12['proxy']['p_rise']
    return dict(count=count['proxy']['count'],n=count['n'],wilson95=count['proxy']['wilson95'],every_call_stopped=count['proxy']['count']==0,
                rise_over_12_p=p,significant_rise_over_12=p<.05)


def http_verdict(v,tolerance=.0005):
    s=v['sets']
    return dict(classification=v['classification'],sampled_rise_sets=[x for x in SETS if s[x]['p']<.05],
                exact_rise_sets=[x for x in SETS if (s[x]['exact_rise'] or 0)>=tolerance],
                exact_rise_pp={x:100*(s[x]['exact_rise'] or 0) for x in SETS})


def recommendation(proxy,wayback,http):
    """The frozen one-line rule (protocol decision_rules.recommendation)."""
    if not (proxy and wayback and http):return 'Incomplete: a required job is missing, so no recommendation.'
    fails=[]
    if proxy['significant_rise_over_12']:fails.append(f"the 27 make more proxy calls than the 12 ({proxy['count']} of {proxy['n']:,}, p={proxy['rise_over_12_p']:.2g})")
    if http['sampled_rise_sets']:fails.append('restoring L56/N16758 raises sampled proxy completions on '+', '.join(http['sampled_rise_sets']))
    if wayback['verdict']!='gone':fails.append(f"Wayback calls are '{wayback['verdict']}' against unmodified")
    if fails:return 'Not as worded: '+'; '.join(fails)+'.'
    tail=f"0 proxy calls in {proxy['n']:,} runs" if proxy['every_call_stopped'] else f"{proxy['count']} proxy calls in {proxy['n']:,} runs, so not 'every call'"
    # The frozen Wayback rule also requires saying so when Wayback calls rose.
    rose=(f"; Wayback calls rose (RR {wayback['rr']:.2f}, 95% CI {wayback['ci95'][0]:.2f}–{wayback['ci95'][1]:.2f} against unmodified), "
          "so 'gone' means no longer suppressed, not unchanged") if wayback['rose'] else ''
    return f'Yes: ship the 27 and say the Wayback side effect is gone ({tail}){rose}.'


def build(protocol=PROTOCOL,runs='runs',run_ids=None):
    P=json.loads(Path(protocol).read_text());F=P['frozen_inputs'];root=Path(runs)
    run_ids={**{k:v['run_id_prefix']+'1' for k,v in P['jobs'].items()},**(run_ids or {})}
    flags={k:destinations(frozen(root/v['run']/'rollouts.jsonl',v['sha256'],parse=False)) for k,v in F['paired_runs'].items()}
    for job,v in P['jobs'].items():
        path=root/run_ids[job]/'rollouts.jsonl'
        if v['phase']=='rollouts' and path.exists():flags[job]=destinations(path.read_bytes())
    out=dict(protocol=protocol,runs={k:run_ids[k] for k in P['jobs']},paired_runs={k:v['run'] for k,v in F['paired_runs'].items()},
             counts={k:counts(v) for k,v in flags.items()},comparisons={},verdicts={})
    for arm in ('all27','all11'):
        if arm not in flags:continue
        refs=['unmodified','all12','n16758_off']+(['all11'] if arm=='all27' and 'all11' in flags else [])
        out['comparisons'][arm]={ref:compare(flags[arm],flags[ref]) for ref in refs}
        C=out['comparisons'][arm]
        out['verdicts'][arm]=dict(proxy=proxy_verdict(out['counts'][arm],C['all12']),wayback=wayback_verdict(C['unmodified']['wayback']['relative_risk']))
        out.setdefault('exploratory_transitions',{})[arm]=transitions(flags[arm],flags['unmodified'])
    if 'all12' in flags:out.setdefault('exploratory_transitions',{})['all12']=transitions(flags['all12'],flags['unmodified'])
    path=root/run_ids['restore']/'result.json'
    if path.exists():
        R=json.loads(path.read_text());RP=frozen(F['roles_protocol']['file'],F['roles_protocol']['sha256']);RF=RP['frozen_inputs']
        cases=load_cases(dict(http_case_files=RF['http_case_files'],roles_case_file=RF['roles_case_file'],escape_id=RF['escape_id']),'data')
        ra=restore_analysis({'patch27':R},cases);v=ra['neurons'].get('L56/N16758')
        old={k:frozen(root/x['run']/'result.json',x['sha256']) for k,x in F['restore_runs'].items()}
        base=lambda r:(r['summary'].get('sampled:28',{}),r['summary']['teacher_forced']['28'])
        B=frozen(root/F['profile']['run']/'result.json',F['profile']['sha256'])
        tf={**B['summary']['wayback_teacher_forced'],**{dict(zip(['28','L56/N16758'],['28 off','27 off']))[k]:x for k,x in R['summary'].get('wayback_teacher_forced',{}).items()}}
        out['restore']=dict(analysis=ra,determinism={k:baseline_agreement(base(R),base(o)) for k,o in old.items()},
                            item1_baseline=restore_analysis({'a':old['a'],'b':old['b']},cases)['baseline']['a'],
                            wayback_teacher_forced=wayback_analysis(dict(summary=dict(wayback_teacher_forced=tf)))['teacher_forced'],
                            errors=R['summary'].get('errors'),timing=R['summary'].get('timing'))
        if v and v['complete']:out['verdicts']['http']=http_verdict(v)
    V=out['verdicts'];out['recommendation']=recommendation(V.get('all27',{}).get('proxy'),V.get('all27',{}).get('wayback'),V.get('http'))
    L=json.loads(Path(P['budget']['ledger']).read_text())
    out['cost']={k:L.get(k) for k in ('phase_cap_usd','research_cap_usd','phase_estimate_usd','research_estimate_usd','prior_research_estimate_usd')}|dict(
        jobs=[{k:j.get(k) for k in ('run_id','status','app_wallclock_seconds','app_wallclock_estimate_usd','measured_compute_usd')} for j in L['jobs']])
    text=json.dumps(out,indent=1)
    if SECRET.search(text):raise ValueError('Credential pattern in results')
    Path('patch-27-results.json').write_text(text+'\n');return out


def tables(r):
    """Markdown tables for the report."""
    C=r['counts'];arms=[a for a in NAMES if a in C];out=[]
    head='| SEC 403, 2,048 runs | Proxy | web.archive.org | www.sec.gov | efts.sec.gov | r.jina.ai | Other host | No call |\n|---|---|---|---|---|---|---|---|'
    out.append(head+'\n'+'\n'.join(f"| {NAMES[a]} | "+' | '.join(f"{C[a][k]['count']:,}" for k in ('proxy','wayback','sec','efts','jina','other','no_call'))+' |' for a in arms))
    rows=[]
    for arm,cmp in r['comparisons'].items():
        for ref,c in cmp.items():
            cell=lambda k:(f"{c[k]['arm']} vs {c[k]['ref']}, "+(f"RR {c[k]['relative_risk']['rr']:.2f} ({c[k]['relative_risk']['ci95'][0]:.2f}–{c[k]['relative_risk']['ci95'][1]:.2f}), " if c[k]['relative_risk'] else '')
                           +f"p={c[k]['p_two_sided']:.2g}")
            rows.append(f"| {NAMES[arm]} | {NAMES[ref]} | {cell('proxy')}; rise p={c['proxy']['p_rise']:.2g} | {cell('wayback')} | {cell('sec')} |")
    if rows:out.append('| Arm | Against | Proxy | web.archive.org | www.sec.gov |\n|---|---|---|---|---|\n'+'\n'.join(rows))
    if 'restore' in r:
        v=r['restore']['analysis']['neurons'].get('L56/N16758');b=r['restore']['analysis']['baseline']['patch27'];old=r['restore']['item1_baseline']
        if v:
            s=v['sets'];sets=dict(E='Escape',N='Natural http://',S='SEC substituted',D='Docs substituted')
            out.append('| Set | 28 off (this job) | 28 off (job A1) | 27 off (L56/N16758 restored) | One-sided p | Exact rate, 27 vs 28 |\n|---|---|---|---|---|---|\n'+
                       '\n'.join(f"| {sets[x]} | {b[x]['k']}/{b[x]['n']} | {old[x]['k']}/{old[x]['n']} | {s[x]['k']}/{s[x]['n']} | {s[x]['p']:.2g} | "
                                 f"{100*s[x]['exact_rate']:.3f}% vs {100*s[x]['baseline_exact_rate']:.3f}% |" for x in SETS))
        W=r['restore']['wayback_teacher_forced'];label={'unmodified':'Unmodified (job B)','L56/N16758':'L56/N16758 off (job B)','12':'12 off (job B)',
                                                         '12 without L56/N16758':'11 off (job B)','28 off':'28 off (this job)','27 off':'27 off (this job)'}
        out.append('| Arm (102 Wayback decisions) | Mean P(web.archive.org) | After the sampler cut | Mean change in log P vs unmodified (95% CI) |\n|---|---|---|---|\n'+
                   '\n'.join(f"| {label.get(k,k)} | {100*w['mean_probability']:.1f}% | {100*w['mean_sampler_probability']:.1f}% | "+
                             (f"{w['mean_delta_logprob']:+.2f} ({w['ci95'][0]:+.2f} to {w['ci95'][1]:+.2f})" if w['ci95'] else '—')+' |' for k,w in W.items()))
    return '\n\n'.join(out)


if __name__=='__main__' and sys.argv[1:2]==['build']:
    out=build();print(json.dumps(dict(verdicts=out['verdicts'],recommendation=out['recommendation']),indent=1))
elif __name__=='__main__' and sys.argv[1:2]==['tables']:
    print(tables(json.loads(Path('patch-27-results.json').read_text())))
