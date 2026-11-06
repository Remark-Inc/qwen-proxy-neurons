"""Summarize the http:// path runs into http-path-results.json. Local, CPU only.

Every number in the results note comes from this file's output. Counts are
"X of N"; Wilson intervals are descriptive, not corrections for selection.
"""
import json,math,statistics,sys
from pathlib import Path
from qwen_lab.metrics import wilson
from qwen_lab.http_path import scheme_breakdown,redact

ARMS=['unpatched','all12','combined']


def load(path):return json.loads(Path(path).read_text())


def xn(k,n):return dict(k=k,n=n,rate=k/n if n else None,wilson95=wilson(k,n))


def case_groups(cases,escape_id):
    """Named, non-overlapping reporting groups of decision points."""
    g={}
    def add(name,pred):g[name]=[c for c in cases if pred(c)]
    pilot=lambda c:c['source']['run'].startswith('pilot:')
    add('escape',lambda c:c['id']==escape_id)
    add('natural_http_failures_discovery',lambda c:c['kind']=='natural_http_proxy' and c['split']=='discovery' and not pilot(c))
    add('natural_http_failures_heldout',lambda c:c['kind']=='natural_http_proxy' and c['split']=='heldout')
    add('pilot_http_failures_discovery',lambda c:c['kind']=='natural_http_proxy' and pilot(c))
    add('pilot_https_failures_discovery',lambda c:c['kind']=='natural_https_proxy' and pilot(c))
    add('forced_http_discovery',lambda c:c['kind']=='forced_http' and c['split']=='discovery')
    add('forced_http_heldout',lambda c:c['kind']=='forced_http' and c['split']=='heldout')
    add('natural_http_legit',lambda c:c['kind']=='natural_http_legit' and not pilot(c))
    add('natural_https_failures_twelve_discovery',lambda c:c['kind']=='natural_https_proxy' and c['split']=='monitor_discovery' and not pilot(c))
    add('natural_https_failures_heldout',lambda c:c['kind']=='natural_https_proxy' and c['split']=='heldout')
    return g


def conditional(validate,cases,escape_id):
    s=validate['summary'];out={}
    for name,group in case_groups(cases,escape_id).items():
        if not group:continue
        row=dict(cases=len(group))
        for arm in ARMS:
            tf=[math.exp(s['teacher_forced'][arm][c['id']]['logprob']) for c in group]
            cut=[s['teacher_forced'][arm][c['id']]['sampler_probability'] for c in group]
            sampled=[r for c in group for r in s['sampled'][arm].get(c['id'],[])]
            row[arm]=dict(sampled=xn(sum(r['proxy'] for r in sampled),len(sampled)),
                          sampled_aliyun=sum(r['aliyun'] for r in sampled),sampled_routify=sum(r['routify'] for r in sampled),
                          teacher_forced_mean=statistics.mean(tf),teacher_forced_max=max(tf),sampler_truncated_mean=statistics.mean(cut),
                          cases_with_any_proxy_sample=sum(any(r['proxy'] for r in s['sampled'][arm].get(c['id'],[])) for c in group))
        out[name]=row
    return out


def scheme_choice(validate):
    s=validate['summary'];out={}
    for arm in ARMS:
        rows=[r for v in s['scheme_sampled'][arm].values() for r in v]
        out[arm]=dict(n=len(rows),http=sum(r['scheme']=='http' for r in rows),https=sum(r['scheme']=='https' for r in rows),
                      proxy=sum(r['proxy'] for r in rows),proxy_after_http=sum(r['proxy'] and r['scheme']=='http' for r in rows),
                      proxy_after_https=sum(r['proxy'] and r['scheme']=='https' for r in rows))
    return out


def free(result):
    out={}
    for k,g in result['summary']['groups'].items():
        out[k]=dict(n=g['n'],proxy_calls=g['proxy_call']['count'],proxy_by_scheme=g['proxy_by_scheme'],
                    aliyun_calls=g['aliyun_call']['count'],requested_host_calls=g['expected_host_call']['count'],
                    any_tool_call=g['any_tool_call']['count'],other_unrequested_host_calls=g['other_unrequested_host_call']['count'],
                    truncated=g['truncated']['count'],unfinished_tool_call=g['unfinished_tool_call']['count'],
                    proxy_text_mentions=g['proxy_text_mention']['count'])
    rows=result['summary'].get('rollouts',[])
    for k in out:
        sub=[r for r in rows if r['scenario']==k]
        out[k]['expected_url_in_final']=sum(r['metrics']['expected_url_in_final'] for r in sub)
    return out


def legacy_free(result):
    """Earlier frozen-suite arms (evaluate stage) in the same format, with scheme breakdown."""
    out={}
    for k,g in result['summary'].items():
        sub=[r for r in result['rollouts'] if r['scenario']==k]
        out[k]=dict(n=g['n'],proxy_calls=g['proxy_call']['count'],proxy_by_scheme=scheme_breakdown(sub),aliyun_calls=g['aliyun_call']['count'],
                    requested_host_calls=g['expected_host_call']['count'],any_tool_call=g['any_tool_call']['count'],
                    other_unrequested_host_calls=g['other_unrequested_host_call']['count'],truncated=g['truncated']['count'],
                    unfinished_tool_call=g['unfinished_tool_call']['count'],proxy_text_mentions=g['proxy_text_mention']['count'],
                    expected_url_in_final=sum(r['metrics']['expected_url_in_final'] for r in sub))
    return out


def replay(result):
    """Fresh decision points harvested from finished suite runs, replayed under each arm."""
    out={};s=result['summary'];kinds={d['id']:d['kind'] for d in s['fresh_decisions']}
    for arm,rows in s['replay'].items():
        for cid,v in rows.items():
            key=(kinds[cid],arm);o=out.setdefault(key,dict(decisions=0,sampled_k=0,sampled_n=0,tf=[]))
            o['decisions']+=1;o['sampled_k']+=v['sampled']['k'];o['sampled_n']+=v['sampled']['n'];o['tf'].append(math.exp(v['teacher_forced']['logprob']))
    return {f'{k}|{a}':dict(decisions=o['decisions'],sampled=xn(o['sampled_k'],o['sampled_n']),teacher_forced_mean=statistics.mean(o['tf']),
                            teacher_forced_max=max(o['tf'])) for (k,a),o in out.items()}


def timing(validate,cases,escape_id):
    s=validate['summary'];groups=case_groups(cases,escape_id);out={}
    for variant,rows in s['timing'].items():
        for name in ['natural_http_failures_discovery','natural_http_failures_heldout','pilot_http_failures_discovery','natural_https_failures_heldout']:
            ids={c['id'] for c in groups.get(name,[])}
            by={}
            for r in rows:
                if r['id'] in ids:by.setdefault(r['window'],[]).append(math.exp(r['logprob']))
            if by:out[f'{variant}|{name}']={w:dict(mean=statistics.mean(v),n=len(v)) for w,v in by.items()}
    return out


def neurons(validate,cases):
    s=validate['summary'];http=[c for c in cases if c['kind']=='natural_http_proxy'];https=[c for c in cases if c['kind']=='natural_https_proxy' and c['split']=='monitor_discovery']
    ctl=[c for c in cases if c['role']=='control'];notes=[]
    def mp(scores,group):return statistics.mean(math.exp(scores[c['id']]['logprob']) for c in group)
    for name in s['new']:
        single=s['single'][name];restore=s['restore'][name];tf=s['teacher_forced']
        lens=next(l for l in s['lens'] if l['neuron']==name)
        acts=[s['decision_activations'][c['id']][name] for c in http];acts_s=[s['decision_activations'][c['id']][name] for c in https]
        notes.append(dict(neuron=name,
            single_http_mean_proxy_probability=dict(unpatched=mp(tf['unpatched'],http),single=mp(single,http)),
            restore_http_mean_proxy_probability=dict(combined=mp(tf['combined'],http),restored=mp(restore,http)),
            single_https_mean_proxy_probability=dict(unpatched=mp(tf['unpatched'],https),single=mp(single,https)),
            restore_https_mean_proxy_probability=dict(combined=mp(tf['combined'],https),restored=mp(restore,https)),
            control_worst_logprob_change_when_restored=min(restore[c['id']]['logprob']-tf['combined'][c['id']]['logprob'] for c in ctl),
            unpatched_decision_activation_http=dict(mean=statistics.mean(acts),min=min(acts),max=max(acts)),
            unpatched_decision_activation_https=dict(mean=statistics.mean(acts_s),min=min(acts_s),max=max(acts_s)),
            lens={k:[t['token'] for t in v[:10]] for k,v in lens['poles'].items()}))
    return notes


def build(runs='runs',legacy='./runs',out='http-path-results.json'):
    runs=Path(runs);protocol=load('data/http/protocol.json')
    cases=load('data/http/cases.json')+(load('data/http/discovery-harvest.json') if Path('data/http/discovery-harvest.json').exists() else [])
    d=load(runs/'http-discover-1/result.json');v=load(runs/'http-validate-1/result.json')
    suites={arm:load(runs/f'http-suite-http-{arm}-1/result.json') for arm in ARMS}
    https_combined=load(runs/'http-suite-https-combined-1/result.json')
    report=dict(protocol=dict(case_sha256=protocol['case_sha256'],combined_sha256=protocol.get('combined_sha256')),
        combined=protocol['combined_neurons'],added=d['summary']['added'],
        discovery=dict(objective=d['summary']['objective'],pilots=[{k:p[k] for k in ['label','seed','n','proxy_by_scheme','harvested_proxy_decisions']}|dict(proxy_calls=p['summary']['proxy_call']['count'],truncated=p['summary']['truncated']['count']) for p in d['summary']['pilots']],
            rounds=[dict(round=r['round'],neurons=len(r['neurons']),objective=r['objective'],stop=r.get('stop'),added=[a['added'] for a in r.get('additions',[]) if 'added' in a],
                         rejected=[a['rejected'] for a in r.get('additions',[]) if 'rejected' in a]) for r in d['summary']['rounds']],
            pruned=d['summary']['pruned'],final_control_gate={k:x for k,x in d['summary']['final_control_gate'].items() if k!='drops'}),
        conditional=conditional(v,cases,protocol['escape_case_id']),scheme_choice=scheme_choice(v),
        free_http={arm:free(r) for arm,r in suites.items()},
        free_https=dict(unpatched=legacy_free(load(Path(legacy)/'zero-validate-baseline-1/result.json')),
                        all12=legacy_free(load(Path(legacy)/'zero-validate-expanded-1/result.json')),combined=free(https_combined)),
        fresh_replay=replay(load(runs/'http-replay-2/result.json')),
        specificity=[{k:r[k] for k in ['arm','task','n','passed','requested_host_calls','proxy_calls','truncated']} for r in load(runs/'http-specificity-1/result.json')['summary']['specificity']],
        specificity_in_validation_run=[{k:r[k] for k in ['arm','task','n','passed','requested_host_calls','proxy_calls','truncated']} for r in v['summary']['specificity']],
        timing=timing(v,cases,protocol['escape_case_id']),weight_equivalence=v['summary']['weight_equivalence'],
        neuron_notes=neurons(v,cases),
        ledger={k:v for k,v in load('data/http/budget.json').items() if k in ('phase_cap_usd','phase_estimate_usd','research_estimate_usd','prior_research_estimate_usd')})
    report['free_http_timing']='Not run: its pre-declared condition (at least 5 proxy calls by the twelve on the http:// suite) was not met (0 of 672).'
    Path(out).write_text(json.dumps(json.loads(redact(json.dumps(report))),indent=1)+'\n')
    return report


ARMS4=['unpatched','all12','combined','refined']


def round3(runs='runs',legacy='./runs',out='http-path-results.json'):
    """All four arms side by side. Earlier arms come from http-validate-1 and http-replay-2;
    the refined arm and the new held-out docs pages come from http-validate-refined-1."""
    runs=Path(runs);protocol=load('data/http/protocol.json')
    cases=load('data/http/cases.json')+load('data/http/discovery-harvest.json')+load('data/http/refine-cases.json')
    byid={c['id']:c for c in cases}
    v2=load(runs/'http-validate-1/result.json')['summary'];v3=load(runs/'http-validate-refined-1/result.json')['summary']
    notes_run=runs/'http-validate-refined-notes-1/result.json'
    if notes_run.exists():
        # Stages the main run could not reach before its deadline, rerun alone on the same frozen sets.
        extra=load(notes_run)['summary']
        for k in ['single','restore','lens','decision_activations','profiles','escape_paths','weight_determinism']:
            if k in extra:v3[k]=extra[k];v3['errors'].pop(k,None)
        v3['errors'].update({f'notes:{k}':e for k,e in extra['errors'].items()})
    rp2=load(runs/'http-replay-2/result.json')['summary'];d2=load(runs/'http-discover-1/result.json')['summary'];d3=load(runs/'http-refine-1/result.json')['summary']
    tf3=v3['teacher_forced']  # all four arms on every case
    def samples(arm,cid):
        if arm=='refined':return v3['sampled_refined'].get(cid,[])
        if cid in v2['sampled'].get(arm,{}):return v2['sampled'][arm][cid]
        return v3.get('sampled_heldout_r3',{}).get(arm,{}).get(cid,[])
    escape=protocol['escape_case_id'];pilot=lambda c:c['source']['run'].startswith('pilot:')
    groups=dict(
        escape=[byid[escape]],
        natural_http_failures_discovery_other=[c for c in cases if c['kind']=='natural_http_proxy' and c['split']=='discovery' and not pilot(c) and c['id']!=escape],
        natural_http_failures_heldout=[c for c in cases if c['kind']=='natural_http_proxy' and c['split']=='heldout'],
        natural_http_legit=[c for c in cases if c['kind']=='natural_http_legit' and not pilot(c)],
        forced_http_sec_discovery=[c for c in cases if c['kind']=='forced_http' and c['split']=='discovery' and c.get('objective_group','forced')=='forced'],
        forced_http_sec_heldout=[c for c in cases if c['kind']=='forced_http' and c['split']=='heldout' and c['source']['scenario']=='sec:403'],
        docs_round2_heldout_now_discovery=[c for c in cases if c['id'].startswith('r3:') and c['split']=='discovery'],
        docs_heldout_round3=[c for c in cases if c['split']=='heldout_r3'],
        natural_https_failures_twelve_discovery=[c for c in cases if c['kind']=='natural_https_proxy' and c['split']=='monitor_discovery' and not pilot(c)],
        natural_https_failures_heldout=[c for c in cases if c['kind']=='natural_https_proxy' and c['split']=='heldout'])
    conditional={}
    for name,group in groups.items():
        row=dict(cases=len(group))
        for arm in ARMS4:
            # Retired docs prefixes were sampled for earlier arms under their original held-out IDs.
            rows=[r for c in group for r in samples(arm,c.get('retired_from',c['id']) if arm!='refined' else c['id'])]
            tf=[math.exp(tf3[arm][c['id']]['logprob']) for c in group]
            row[arm]=dict(sampled=xn(sum(r['proxy'] for r in rows),len(rows)),teacher_forced_mean=statistics.mean(tf),teacher_forced_max=max(tf),
                          prefixes_with_any_proxy_sample=sum(any(r['proxy'] for r in samples(arm,c.get('retired_from',c['id']) if arm!='refined' else c['id'])) for c in group))
        conditional[name]=row
    by_page={}
    for c in groups['docs_heldout_round3']:by_page.setdefault(c['source']['scenario'].split(':')[0],[]).append(c)
    docs_pages={page:{arm:xn(sum(r['proxy'] for c in cs for r in samples(arm,c['id'])),sum(len(samples(arm,c['id'])) for c in cs)) for arm in ARMS4} for page,cs in by_page.items()}
    # Fresh natural decisions: round-2 suites (earlier arms from http-replay-2, refined from round 3) and refined suites (all arms).
    kinds={d['id']:d['kind'] for d in rp2['fresh_decisions']}
    fresh_old={}
    for arm in ['unpatched','all12','combined']:
        for cid,val in rp2['replay'][arm].items():
            o=fresh_old.setdefault((kinds[cid],arm),[0,0]);o[0]+=val['sampled']['k'];o[1]+=val['sampled']['n']
    k3={d['id']:d['kind'] for d in v3['replay']['decisions_old']}
    for cid,val in v3['replay']['old'].items():
        o=fresh_old.setdefault((k3[cid],'refined'),[0,0]);o[0]+=val['sampled']['k'];o[1]+=val['sampled']['n']
    kn={d['id']:d['kind'] for d in v3['replay']['decisions_new']}
    fresh_new={}
    for arm,rows in v3['replay']['new'].items():
        for cid,val in rows.items():
            o=fresh_new.setdefault((kn[cid],arm),[0,0]);o[0]+=val['sampled']['k'];o[1]+=val['sampled']['n']
    sc={}
    for arm in ['unpatched','all12','combined']:
        rows=[r for x in v2['scheme_sampled'][arm].values() for r in x];sc[arm]=rows
    sc['refined']=[r for x in v3['scheme_sampled'].values() for r in x]
    scheme={arm:dict(n=len(rows),http=sum(r['scheme']=='http' for r in rows),proxy_after_http=sum(r['proxy'] and r['scheme']=='http' for r in rows),
                     proxy_after_https=sum(r['proxy'] and r['scheme']=='https' for r in rows)) for arm,rows in sc.items()}
    suites=dict(http={a:free(load(runs/f'http-suite-http-{a}-1/result.json')) for a in ARMS4},
                https=dict(unpatched=legacy_free(load(Path(legacy)/'zero-validate-baseline-1/result.json')),all12=legacy_free(load(Path(legacy)/'zero-validate-expanded-1/result.json')),
                           combined=free(load(runs/'http-suite-https-combined-1/result.json')),refined=free(load(runs/'http-suite-https-refined-1/result.json'))))
    spec=[{k:r[k] for k in ['arm','task','n','passed','requested_host_calls','proxy_calls','truncated']} for r in load(runs/'http-specificity-1/result.json')['summary']['specificity']]
    spec+=[{k:r[k] for k in ['arm','task','n','passed','requested_host_calls','proxy_calls','truncated']} for r in v3.get('specificity',[])]
    natural_ids={c['id'] for c in cases if c['kind']=='natural_http_proxy' and c['split']=='heldout'};https_ids={c['id'] for c in groups['natural_https_failures_heldout']}
    timing={}
    for variant,rows in v3.get('timing',{}).items():
        for label,ids in [('natural_http_heldout',natural_ids),('natural_https_heldout',https_ids)]:
            by={}
            for r in rows:
                if r['id'] in ids:by.setdefault(r['window'],[]).append(math.exp(r['logprob']))
            if by:timing[f'{variant}|{label}']={w:statistics.mean(x) for w,x in by.items()}
    def mp(scores,group):return statistics.mean(math.exp(scores[c['id']]['logprob']) for c in group)
    http=[c for c in cases if c['kind']=='natural_http_proxy'];docs=groups['docs_round2_heldout_now_discovery']+groups['docs_heldout_round3']
    notes=[]
    for name in v3.get('new',[]):
        lens=next(l for l in v3['lens'] if l['neuron']==name)
        acts=[v3['decision_activations'][c['id']][name] for c in http];acts_docs=[v3['decision_activations'][c['id']][name] for c in docs]
        acts_s=[v3['decision_activations'][c['id']][name] for c in groups['natural_https_failures_twelve_discovery']]
        notes.append(dict(neuron=name,single_http=dict(unpatched=mp(tf3['unpatched'],[c for c in http if c['id'] in v3['single'][name]]),single=mp(v3['single'][name],[c for c in http if c['id'] in v3['single'][name]])),
            restore_http=dict(refined=mp(tf3['refined'],http),restored=mp(v3['restore'][name],http)),
            activation=dict(http=statistics.mean(acts),docs_http=statistics.mean(acts_docs),https=statistics.mean(acts_s)),
            lens={k:[t['token'] for t in x[:10]] for k,x in lens['poles'].items()}))
    report=dict(arms=dict(unpatched=[],all12=v3['arms']['all12'],combined=v3['arms']['combined'],refined=v3['arms']['refined']),
        round2_added=d2['added'],round3_added=d3['added_this_run'],
        discovery=dict(round2=dict(objective=d2['objective'],pruned=[p['removed'] for p in d2['pruned']]),
                       round3=dict(objective=d3['objective'],stop=[r.get('stop') for r in d3['rounds']],escape_sampled=[r.get('escape_sampled') for r in d3['rounds']],
                                   escape_diagnostics=d3['escape_diagnostics'],gate={k:x for k,x in d3['final_control_gate'].items() if k!='drops'})),
        conditional=conditional,docs_heldout_by_page=docs_pages,
        fresh_round2_suites={f'{k}|{a}':xn(*v) for (k,a),v in fresh_old.items()},fresh_refined_suites={f'{k}|{a}':xn(*v) for (k,a),v in fresh_new.items()},
        scheme_choice=scheme,free=suites,specificity=spec,timing=timing,weight_equivalence=v3.get('weight_equivalence'),
        weight_determinism=v3.get('weight_determinism'),
        escape_paths=v3.get('escape_paths'),neuron_notes_round3=notes,errors=v3.get('errors'),
        ledger={k:x for k,x in load('data/http/budget.json').items() if k in ('phase_cap_usd','phase_estimate_usd','research_estimate_usd','prior_research_estimate_usd','research_cap_usd')})
    Path(out).write_text(json.dumps(json.loads(redact(json.dumps(report))),indent=1)+'\n')
    return report


if __name__=='__main__':round3(*sys.argv[1:])  # round 3 built the published http-path-results.json; build() is round 2
