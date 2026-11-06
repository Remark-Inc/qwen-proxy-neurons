import json
import random
import re
import time
from pathlib import Path

from qwen_lab import MODEL, REVISION
from qwen_lab.fixtures import scenario, suite
from qwen_lab.interventions import intervene, language_layers, protect_recurrent_cache_for_grad
from qwen_lab.metrics import PROXY, classify, summarize

def load_model():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    return model, tok

def render(tok, item):
    return tok.apply_chat_template(item["messages"], tools=item["tools"], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True,
                                   reasoning_effort="medium")

def emit(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")

def generation(model, tok, item, n, batch_size, max_new, seed, neurons, deadline, path):
    import torch
    prompt = item.get('prompt_override') or render(tok, item)
    response_prefill = prompt.rsplit('<|im_start|>assistant\n', 1)[-1]
    rows = []
    with intervene(model, neurons), torch.inference_mode():
        for start in range(0, n, batch_size):
            if time.monotonic() >= deadline:
                break
            count = min(batch_size, n-start)
            batch_seed = seed + start
            torch.manual_seed(batch_seed)
            inputs = tok([prompt]*count, return_tensors="pt", padding=True,
                         add_special_tokens=False).to("cuda")
            before = time.monotonic()
            generated = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id, use_cache=True)
            for j, sequence in enumerate(generated):
                ids = sequence[inputs.input_ids.shape[1]:].tolist()
                eos = model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                stop = next((i for i,t in enumerate(ids) if t in eos), None)
                truncated = stop is None and len(ids) == max_new
                if stop is not None:
                    ids = ids[:stop+1]
                text = tok.decode(ids, skip_special_tokens=False)
                row = dict(scenario=item["id"], fixture=item, prompt=prompt,
                    batch_seed=batch_seed, batch_row=j, batch_size=count,
                    token_ids=ids, text=text, response_prefill=response_prefill,
                    conditional_prefill=bool(item.get('prompt_override')), neurons=neurons,
                    metrics=classify(response_prefill+text, item["url"], truncated))
                emit(path, row)
                rows.append(row)
            print(json.dumps({"scenario": item["id"], "completed": len(rows),
                              "batch_seconds": round(time.monotonic()-before, 1),
                              "proxy_calls": sum(r["metrics"]["proxy_call"] for r in rows)}), flush=True)
    return rows

def attribute(model, examples, deadline, telemetry=None):
    import torch
    layers = language_layers(model)
    scores = torch.zeros(len(layers), layers[0].mlp.down_proj.in_features)
    completed = 0
    activity_records = []
    for ex in examples:
        if time.monotonic() >= deadline:
            break
        acts, handles = {}, []
        def hook_for(l):
            def hook(module, inputs):
                # Last-position additive perturbations yield independent scalar
                # derivatives for every neuron while preserving forward values.
                x = inputs[0]
                delta = torch.zeros_like(x[:, -1:, :], requires_grad=True)
                acts[l] = (x[:, -1, :].detach(), delta)
                return (torch.cat([x[:, :-1, :], x[:, -1:, :] + delta], dim=1), *inputs[1:])
            return hook
        try:
            # Decode one token with a detached prefix cache. Attribution is local
            # to this decision position, not all-token ablation attribution.
            ids = torch.tensor([ex["prefix_ids"]], device="cuda")
            with torch.no_grad():
                cache = model(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values
            protect_recurrent_cache_for_grad(cache)
            for l, layer in enumerate(layers):
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook_for(l)))
            out = model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits = out.logits[0,-1].float()
            objective = logits[ex["bad"]] - logits[ex["good"]]
            grads = torch.autograd.grad(objective, [acts[l][1] for l in range(len(layers))])
            if telemetry is not None:
                activation = torch.stack([acts[l][0][0].float().cpu() for l in range(len(layers))])
                gradient = torch.stack([g[0,0].float().detach().cpu() for g in grads])
                activity_records.append((ex['scenario'],activation,gradient))
                products = activation * gradient
                values, flat_indices = products.flatten().topk(48)
                width = products.shape[1]
                top = []
                for value, flat in zip(values.tolist(), flat_indices.tolist()):
                    l,n = divmod(flat,width)
                    top.append(dict(layer=l,neuron=n,activation=float(activation[l,n]),
                                    gradient=float(gradient[l,n]),attribution=value))
                probs=logits.softmax(-1)
                emit(telemetry,dict(scenario=ex['scenario'],bad_token=ex['bad'],good_token=ex['good'],
                     proxy_token_probability=float(probs[ex['bad']].detach()),
                     legitimate_token_probability=float(probs[ex['good']].detach()),top=top))
            for l, grad in enumerate(grads):
                scores[l] += (acts[l][0].float()*grad[:,0].float()).sum(0).detach().cpu()
            completed += 1
            print(f"Attributed decision {completed}/{len(examples)}", flush=True)
        finally:
            for h in handles:
                h.remove()
            acts.clear()
            if "out" in locals():
                del out, cache
            torch.cuda.empty_cache()
    if not completed:
        return []
    scores /= completed
    values, indices = scores.flatten().topk(32)
    width = scores.shape[1]
    candidates=[]
    for v,i in zip(values,indices):
        l,n=divmod(int(i),width)
        row=dict(layer=l,neuron=n,attribution=float(v),decisions=completed)
        if telemetry is not None:
            row['activity']=[dict(case=key,activation=float(a[l,n]),gradient=float(g[l,n]),
                                 attribution=float(a[l,n]*g[l,n])) for key,a,g in activity_records]
        candidates.append(row)
    return candidates

def discover(model, tok, cases, deadline, path):
    from qwen_lab.prefixes import first_distinguishing
    from qwen_lab.scoring import score_cases, aggregate
    examples = [first_distinguishing(c) for c in cases]
    candidates = attribute(model, examples, deadline)
    if not candidates:
        return {"status": "deadline_before_attribution", "candidates": []}
    baseline_scores = score_cases(model, cases)
    base = aggregate(baseline_scores)
    tests = []
    rng = random.Random(918)
    layers = language_layers(model)
    controls = [dict(layer=rng.randrange(len(layers)),
                     neuron=rng.randrange(layers[0].mlp.down_proj.in_features), random_control=True)
                for _ in range(8)]
    for c in candidates + controls:
        if time.monotonic() >= deadline:
            break
        intervention = {"layer": c["layer"], "neuron": c["neuron"], "mode": "scale", "value": 0}
        scored = score_cases(model, cases, [intervention])
        value = aggregate(scored)
        test = {**c, "log_odds": value, "log_odds_reduction": base-value, "scores": scored}
        tests.append(test)
        emit(path, test)
        print(json.dumps({k:v for k,v in test.items() if k!='scores'}), flush=True)
    ranked = sorted([t for t in tests if not t.get("random_control")], key=lambda t:t["log_odds"])
    selected, sets = [], []
    for size in range(1,4):
        choices = []
        for c in ranked[:12]:
            spec = {"layer":c["layer"], "neuron":c["neuron"], "mode":"scale", "value":0}
            if spec in selected or time.monotonic() >= deadline:
                continue
            scored = score_cases(model, cases, selected+[spec])
            choices.append((aggregate(scored),spec,scored))
        if not choices:
            break
        value, spec, scored = min(choices, key=lambda x:x[0])
        selected = selected+[spec]
        sets.append({"neurons":selected, "log_odds":value,
                     "log_odds_reduction":base-value, "scores":scored})
    return dict(status="conditional_candidates_only", baseline_log_odds=base,
                baseline_scores=baseline_scores, cases=cases,
                attribution_scope="last-position activation x gradient; first distinguishing token",
                causal_score="sum log P(full continuation | shared prefix), proxy minus legitimate",
                candidates=ranked, controls=[t for t in tests if t.get("random_control")],
                sets=sets, examples=examples,
                caveat="Conditional destination effects do not establish effects on tool initiation or free generation.")

def run(config, root):
    import torch
    started = time.monotonic()
    deadline = started + config["seconds"]
    run_id = config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain letters, numbers, underscore, or hyphen")
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder/"config.json").write_text(json.dumps(config, indent=2))
    model, tok = load_model()
    metadata = dict(model=MODEL, revision=REVISION, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(), reasoning_effort="medium",
                    sampling=dict(temperature=1.,top_p=.95,top_k=20),
                    neuron_count=sum(l.mlp.down_proj.in_features for l in language_layers(model)))
    (folder/"metadata.json").write_text(json.dumps(metadata, indent=2))
    stage = config["stage"]
    if stage == 'explain':
        from qwen_lab.explanation import explain
        result = dict(summary=explain(model,tok,config,root,folder,deadline),
                      metadata=metadata,config=config)
    elif stage == 'residual':
        from qwen_lab.residual import residual_search
        result = dict(summary=residual_search(model,tok,config,root,folder,deadline),
                      metadata=metadata,config=config)
    elif stage == 'extend':
        from qwen_lab.scoring import score_cases, aggregate
        source_id = config['source_run']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', source_id):
            raise ValueError('extend requires a discovery source_run')
        discovery = json.loads((root/source_id/'result.json').read_text())['summary']
        cases = discovery['cases']
        selected = config['neurons']
        if len(selected) != 3:
            raise ValueError('extend requires exactly three fixed neurons')
        baseline_scores = score_cases(model, cases)
        selected_scores = score_cases(model, cases, selected)
        selected_value = aggregate(selected_scores)
        tests = []
        selected_ids = {(n['layer'], n['neuron']) for n in selected}
        for c in discovery['candidates']:
            if (c['layer'], c['neuron']) in selected_ids:
                continue
            if time.monotonic() >= deadline:
                break
            added = dict(layer=c['layer'], neuron=c['neuron'], mode='scale', value=0)
            scored = score_cases(model, cases, selected+[added])
            value = aggregate(scored)
            test = dict(neurons=selected+[added], log_odds=value,
                        log_odds_reduction=aggregate(baseline_scores)-value,
                        incremental_reduction=selected_value-value, scores=scored)
            tests.append(test)
            emit(folder/'extension_screen.jsonl', test)
            print(json.dumps({'fourth': added, 'incremental_reduction': selected_value-value}), flush=True)
        ranked = sorted(tests, key=lambda x: x['log_odds'])
        result = dict(summary=dict(status='conditional_fourth_candidate_only', cases=cases,
            baseline_scores=baseline_scores, triple_scores=selected_scores,
            candidates=ranked, sets=ranked[:1],
            caveat='Fourth selected only on original discovery prefixes from the existing 32-candidate pool; not an exhaustive four-neuron search.'),
            metadata=metadata, config=config)
    elif stage == 'compare':
        from qwen_lab.prefixes import synthetic_cases
        from qwen_lab.scoring import score_cases
        source_id=config['source_run']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',source_id):
            raise ValueError('compare requires a discovery source_run')
        discovery=json.loads((root/source_id/'result.json').read_text())['summary']
        sets=discovery.get('sets',[])
        if not sets:
            raise ValueError('No candidate sets in discovery run')
        cases=[]
        compare_keys=config.get('case_keys') or ('asyncio','sec')
        for boundary in ('domain','before_tool','inside_domain'):
            cases += synthetic_cases(tok,keys=compare_keys,boundary=boundary)
        # Lexically matched requested hosts expose a generic 'r'/URL feature.
        if not config.get('case_keys'):
            cases += synthetic_cases(tok,keys=('requests','rust','jina'),boundary='domain')
        variants=[('baseline',[])]
        for s in sets:
            variants.append((f"ablate-{len(s['neurons'])}",s['neurons']))
        best=sets[-1]['neurons']
        for scale in (.5,1.,2.):
            variants.append((f'gain-{scale}',[{**n,'value':scale} for n in best]))
        controls=discovery.get('controls',[])[:len(best)]
        variants.append(('random-ablation',[dict(layer=c['layer'],neuron=c['neuron'],mode='scale',value=0) for c in controls]))
        comparisons=[]
        for label,neurons in variants:
            if time.monotonic()>=deadline:
                break
            row=dict(label=label,neurons=neurons,scores=score_cases(model,cases,neurons))
            comparisons.append(row);emit(folder/'comparisons.jsonl',row)
            print(f'Completed holdout prefix comparison: {label}',flush=True)
        lenses=[]
        with torch.inference_mode():
            for spec in best:
                column=language_layers(model)[spec['layer']].mlp.down_proj.weight[:,spec['neuron']]
                poles={}
                for sign in (1,-1):
                    logits=model.lm_head(model.model.language_model.norm((sign*column)[None,None]))[0,0]
                    top=logits.topk(20).indices.tolist()
                    poles[str(sign)]=[tok.decode([t]) for t in top]
                lenses.append(dict(layer=spec['layer'],neuron=spec['neuron'],poles=poles,
                    interpretation='Direct logit-lens heuristic; not a measured semantic label'))
        result=dict(summary=dict(status='heldout_conditional_comparison',cases=cases,comparisons=comparisons,lenses=lenses),
                    metadata=metadata,config=config)
    elif stage in {"discover", "prefix", "continuation"}:
        from qwen_lab.prefixes import synthetic_cases, observed_cases, custom_cases
        from qwen_lab.scoring import score_cases
        source_id = config["source_run"]
        if config.get('custom_cases'):
            cases = custom_cases(tok, config['custom_cases'])
        elif source_id:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
                raise ValueError("Invalid source run")
            source = [json.loads(l) for l in (root/source_id/"rollouts.jsonl").read_text().splitlines()]
            cases = observed_cases(tok, source)
            if not cases:
                raise ValueError("Source run has no usable positive trace; omit source_run for labeled synthetic discovery")
        else:
            keys = config.get('case_keys') or (('asyncio','sec') if config.get('heldout') else ('mckinsey','pathlib'))
            cases = synthetic_cases(tok, keys=keys, boundary=config.get('boundary','domain'))
        (folder/'prefix_cases.json').write_text(json.dumps(cases,indent=2))
        if stage == 'continuation':
            rows=[]
            for i,c in enumerate(cases):
                fixture = dict(c.get('fixture', {}))
                fixture.update(id=c['id'], url=fixture.get('url',''),
                               prompt_override=c.get('requested_prefix',c.get('prefix')),
                               synthetic=c['synthetic'], boundary=c['boundary'])
                rows += generation(model,tok,fixture,config['samples'],config['batch_size'],
                    config['max_new_tokens'],config['seed']+i*10000,config['neurons'],
                    deadline,folder/'rollouts.jsonl')
            result = dict(status='conditional_generations_only',cases=cases,
                          scores={c['id']:summarize([r for r in rows if r['scenario']==c['id']]) for c in cases})
        elif stage == 'prefix':
            scored=score_cases(model,cases,config['neurons'])
            result = dict(status='conditional_scores_only', cases=cases, scores=scored,
                          baseline_scores=score_cases(model,cases,[]) if config['neurons'] else scored)
        else:
            if config.get('boundary','domain') != 'domain':
                raise ValueError('Discover at the domain boundary; use prefix stage for other boundary diagnostics')
            result = discover(model, tok, cases, deadline, folder/"causal_screen.jsonl")
        result = {"summary":result, "metadata":metadata, "config":config}
        if stage == 'continuation':
            result['rollouts'] = rows
    else:
        rows = []
        if config.get('scenario_specs'):
            from qwen_lab.fixtures import validate_scenario_specs
            validate_scenario_specs(config['scenario_specs'])
            items = [{**scenario(s['key'],s['condition']), 'sample_count':s['samples']}
                     for s in config['scenario_specs']]
        else:
            items = ([scenario(k,'403') for k in config['case_keys']] if config.get('case_keys')
                     else suite("discovery" if stage == "baseline" else "heldout"))
        if stage == "evaluate" and not config.get('case_keys') and not config.get('scenario_specs'):
            items += [scenario("asyncio", "fresh"), scenario("pathlib", "recall")]
        for i, item in enumerate(items):
            rows += generation(model,tok,item,item.get('sample_count',config["samples"]),config["batch_size"],
                config["max_new_tokens"],config["seed"]+i*10000, config["neurons"],
                deadline,folder/"rollouts.jsonl")
        groups = {item["id"]:summarize([r for r in rows if r["scenario"]==item["id"]]) for item in items}
        result = {"summary":groups,"rollouts":rows,"metadata":metadata,"config":config}
    result["elapsed_seconds"] = time.monotonic()-started
    result["estimated_compute_usd"] = result["elapsed_seconds"]*(.001261 + 4*.0000131 + 96*.00000222)
    result["cost_note"] = "Resource-rate estimate, not an invoice; excludes startup/build/storage."
    (folder/"result.json").write_text(json.dumps(result,indent=2))
    return result
