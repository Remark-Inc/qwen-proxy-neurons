"""Conditional residual-failure search with the existing intervention active."""
import json,re,time
from qwen_lab.interventions import intervene
from qwen_lab.prefixes import observed_cases,first_distinguishing,synthetic_cases
from qwen_lab.scoring import score_cases,aggregate


def residual_search(model,tok,config,root,folder,deadline):
    from qwen_lab.runner import attribute,emit
    source_id=config['source_run']
    if not re.fullmatch(r'[A-Za-z0-9_-]+',source_id):
        raise ValueError('residual requires a source_run')
    source=json.loads((root/source_id/'result.json').read_text())
    base=source['config']['neurons']
    if config.get('neurons') and config['neurons']!=base:
        raise ValueError('Replay must retain the source intervention')
    cases=observed_cases(tok,source['rollouts'],limit=16)
    if not cases:raise ValueError('No exact-token residual proxy prefixes')
    examples=[first_distinguishing(c) for c in cases]
    with intervene(model,base):
        candidates=attribute(model,examples,deadline,telemetry=folder/'activation_telemetry.jsonl')
    existing={(n['layer'],n['neuron']) for n in base}
    candidates=[c for c in candidates if (c['layer'],c['neuron']) not in existing]
    before=score_cases(model,cases,base);baseline=aggregate(before)
    tests=[]
    for c in candidates:
        if time.monotonic()>=deadline:break
        spec=dict(layer=c['layer'],neuron=c['neuron'],mode='scale',value=0)
        scores=score_cases(model,cases,base+[spec]);value=aggregate(scores)
        test={**c,'scores':scores,'log_odds':value,'log_odds_reduction':baseline-value}
        tests.append(test);emit(folder/'single_additions.jsonl',test)
        print(json.dumps({'residual_single':[c['layer'],c['neuron']],'gain':baseline-value}),flush=True)
    ranked=sorted(tests,key=lambda c:c['log_odds'])
    selected=list(base);sets=[]
    # Greedy additions from the best 16 directly screened neurons. This remains
    # an exploratory search; independent generations, not these prefixes, judge success.
    for _ in range(min(config.get('additions',8),32-len(base))):
        choices=[]
        ids={(s['layer'],s['neuron']) for s in selected}
        for c in ranked[:16]:
            if (c['layer'],c['neuron']) in ids or time.monotonic()>=deadline:continue
            spec=dict(layer=c['layer'],neuron=c['neuron'],mode='scale',value=0)
            scores=score_cases(model,cases,selected+[spec])
            choices.append((aggregate(scores),spec,scores))
        if not choices:break
        value,spec,scores=min(choices,key=lambda x:x[0])
        selected=selected+[spec]
        row=dict(neurons=list(selected),log_odds=value,log_odds_reduction=baseline-value,scores=scores)
        sets.append(row);emit(folder/'greedy_sets.jsonl',row)
        print(json.dumps({'residual_size':len(selected),'gain':baseline-value,'added':spec}),flush=True)
    controls=synthetic_cases(tok,keys=('mckinsey','pathlib','requests','rust','jina','router'))
    generalization=[]
    if sets and time.monotonic()<deadline:
        generalization=[dict(label='base',neurons=base,scores=score_cases(model,controls,base)),
                        dict(label='expanded',neurons=selected,scores=score_cases(model,controls,selected))]
    telemetry=[json.loads(l) for l in (folder/'activation_telemetry.jsonl').read_text().splitlines()]
    for row in telemetry:
        row['bad_token_text']=tok.decode([row['bad_token']])
        row['good_token_text']=tok.decode([row['good_token']])
    return dict(status='residual_candidates_only',source_run=source_id,base_neurons=base,
        source_proxy_count=sum(r['metrics']['proxy_call'] for r in source['rollouts']),
        exact_replayed_count=len(cases),cases=cases,baseline_scores=before,
        baseline_log_odds=baseline,candidates=ranked,sets=sets,telemetry=telemetry,
        original_development_checks=generalization,
        caveat='These source failures are now discovery data. Activations and gradients are measured at the hostname decision under the source intervention. Direct ablation tests conditional influence, not necessity for spontaneous tool initiation. No free-generation success claim is made here.')
