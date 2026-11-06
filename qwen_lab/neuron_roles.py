"""Neuron roles for the 28-neuron patch: drop-one on the 16 http:// neurons, activation
profiles, effects on alternative first tokens, and a Wayback check for L56/N16758.

A decision position is the last token of ``scheme://`` before a hostname. Generated calls
stay text; nothing is fetched. Role statements are operational: measured effects and
activation patterns, not semantic labels.
"""
import collections,json,math,random,re,sys,time,zlib
from pathlib import Path
from qwen_lab.http_path import OPEN,SECRET,boundary,decision_points,ident,redact,sample,target_ids,truncate
from qwen_lab.metrics import PROXY,wilson

RATE=.001261+4*.0000131+96*.00000222
SETS=('E','N','S','D')  # escape, natural http:// failures, SEC substituted, docs substituted
ALT_HOSTS=[('r',PROXY),('www','www.sec.gov'),('efts','efts.sec.gov'),('web','web.archive.org'),
           ('developer','developer.mozilla.org'),('py','pytorch.org'),('en','en.wikipedia.org'),
           ('docs','docs.python.org'),('archive','archive.org'),('Way','Wayback.example'),('data','data.sec.gov')]
# Hosts whose first token is shared with a listed one; a first-token effect covers both.
SHARED={'r':['r.jina.ai'],'web':['webcache.googleusercontent.com']}
URLISH=re.compile(r'://|www\.|\b[a-z0-9-]+\.(?:com|org|gov|net|io|ai|ph|edu|cn)\b',re.I)
WAYBACK='web.archive.org'


# ---------------------------------------------------------------- CPU case building

def alt_tokens(tok):
    out=[]
    for label,host in ALT_HOSTS:
        ids={s:target_ids(tok,host+'/x',s)[0][0] for s in ('http','https')}
        if ids['http']!=ids['https']:raise ValueError(f'First token depends on scheme: {label}')
        out.append(dict(label=label,host=host,token_id=ids['https'],token=tok.decode([ids['https']]),shares_first_token_with=SHARED.get(label,[])))
    return out


def text_url_points(tok,row,source):
    """The first plain-text URL in a final answer without tool calls, cut after scheme://."""
    text=row['text'];ids=row['token_ids'];end=text.rfind('</think>')
    if end<0 or '<tool_call>' in text[end:]:return []
    final=end+len('</think>');m=re.search(r'(https?)://',text[final:])
    if not m:return []
    start=final+m.end();prior=text[:start]
    if PROXY in prior or SECRET.search(prior):return []
    k=boundary(tok,ids,text,start);think=boundary(tok,ids,text,final)
    if k is None:return []
    prompt_ids=tok.encode(row['prompt'],add_special_tokens=False)
    return [dict(scheme=m.group(1),prefix_ids=prompt_ids+ids[:k],prompt_length=len(prompt_ids),
                 think_end=None if think is None else len(prompt_ids)+think,
                 group=f"{row['scenario']}:{row.get('batch_seed')}:{row.get('batch_row')}",
                 source=dict(source,scenario=row['scenario'],batch_seed=row.get('batch_seed'),batch_row=row.get('batch_row')))]


def build(tok,validate_rows,rerun_rows):
    """Clean, plain-text and Wayback contexts from saved unpatched runs; never a credential."""
    out=dict(alt_tokens=alt_tokens(tok),clean_fresh=[],recall_text=[],wayback=[])
    keep=lambda c:{k:v for k,v in c.items() if k not in ('tail','scheme_prefix_ids')}
    for r in validate_rows:
        src=dict(run='zero-validate-baseline-1')
        if r['scenario']=='asyncio:fresh':
            for c in decision_points(tok,r,src)[:1]:out['clean_fresh'].append(dict(keep(c),id=f"c_fresh:{c['group']}"))
        if r['scenario']=='pathlib:recall':
            for c in text_url_points(tok,r,src):out['recall_text'].append(dict(c,id=f"c_recall_text:{c['group']}"))
    web=target_ids(tok,WAYBACK+'/x','https')[0]
    for r in rerun_rows:
        if WAYBACK not in r['text']:continue
        for c in decision_points(tok,r,dict(run='sparsity-scale-baseline-1')):
            if c['original_host']!=WAYBACK:continue
            out['wayback'].append(dict(keep(c),target_ids=web,id=f"wayback:{c['group']}:call{c['call_index']}"));break
    for c in out['clean_fresh']+out['recall_text']+out['wayback']:
        if SECRET.search(tok.decode(c['prefix_ids'],skip_special_tokens=False)) or PROXY in tok.decode(c['prefix_ids'],skip_special_tokens=False):
            raise ValueError(f"Prefix carries the proxy or a credential: {c['id']}")
    return out


def held_sets(http_cases,escape_id):
    """Item 1's held-out http:// sets, in a fixed order that sets the seeds."""
    by=lambda f:sorted((c for c in http_cases if f(c)),key=lambda c:c['id'])
    sets=dict(E=[c for c in http_cases if c['id']==escape_id],
              N=by(lambda c:c['kind']=='natural_http_proxy' and c['split']=='heldout'),
              S=by(lambda c:c['kind']=='forced_http' and c['split']=='heldout' and c['source']['scenario']=='sec:403'),
              D=by(lambda c:c['kind']=='forced_http' and c['split']=='heldout_r3'))
    return {k:[dict(c,set=k) for c in v] for k,v in sets.items()}


def families(http_cases):
    """Item 2's 403 families from the http:// case files (the 16 traces are built in the container)."""
    pick=lambda f:sorted((c for c in http_cases if f(c)),key=lambda c:c['id'])
    docs=lambda c:c['kind']=='forced_http' and c['source']['scenario']!='sec:403'
    return dict(a_https=pick(lambda c:c['kind']=='natural_https_proxy'),
                b_http_natural=pick(lambda c:c['kind']=='natural_http_proxy'),
                b_http_sec=pick(lambda c:c['kind']=='forced_http' and c['source']['scenario']=='sec:403'),
                b_http_docs=pick(docs))


# ---------------------------------------------------------------- positions (pure)

def first_call_position(tok,ids,prompt_length):
    """Last token of the first ``<parameter=url>\\nscheme://`` in the prompt: the synthetic first fetch."""
    text=tok.decode(ids[:prompt_length],skip_special_tokens=False)
    m=re.search(re.escape(OPEN)+r'(https?)://',text)
    if not m:return None
    k=boundary(tok,ids[:prompt_length],text,m.end())
    return None if k is None else k-1


def prose_positions(tok,ids,start,end,key,k=8,margin=32):
    """Up to k reasoning positions whose +-margin token window decodes without URL text."""
    if start is None or end is None:return []
    cand=list(range(start+margin,min(end,len(ids))-margin));random.Random(zlib.crc32(key.encode())).shuffle(cand);out=[]
    for p in cand:
        if not URLISH.search(tok.decode(ids[p-margin:p+margin+1],skip_special_tokens=False)):out.append(p)
        if len(out)==k:break
    return sorted(out)


def last_occurrence(ids,pattern):
    for j in range(len(ids)-len(pattern),-1,-1):
        if ids[j:j+len(pattern)]==pattern:return j
    return None


# ---------------------------------------------------------------- GPU measurements

def decision_scores(model,prefix_ids,target,alt_ids,top_k=20,top_p=.95):
    """host_logprob's proxy fields plus each alternative first token, raw and after the sampler's cut."""
    import torch
    dev=next(model.parameters()).device;t=torch.tensor(target,device=dev)
    ids=torch.tensor([prefix_ids+target[:-1]],device=dev)
    logits=model(input_ids=ids,use_cache=False,logits_to_keep=len(target)).logits[0].float()
    lp=logits.log_softmax(-1);raw=lp.gather(-1,t[:,None]).squeeze(-1)
    kept=truncate(logits,top_k,top_p).log_softmax(-1);tr=kept.gather(-1,t[:,None]).squeeze(-1)
    a=torch.tensor(alt_ids,device=dev)
    return dict(logprob=float(raw.sum()),first_logprob=float(raw[0]),sampler_probability=float(tr.sum().exp()),
                alt_logprob=[round(x,5) for x in lp[0,a].tolist()],alt_sampler_probability=kept[0,a].exp().tolist())


def teacher_forced(model,cases,specs,alt_ids,deadline):
    import torch
    from qwen_lab.interventions import intervene
    out={}
    with intervene(model,specs),torch.inference_mode():
        for c in cases:
            if time.monotonic()>=deadline:break
            out[c['id']]=decision_scores(model,c['prefix_ids'],c['target_ids'],alt_ids)
    return out


def compact(rows):
    """Sampled rows without continuation text: [batch seed, row, host, proxy]."""
    return [[r['seed'],r['row'],redact(r['host']),bool(r['proxy'])] for r in rows]


def unembedding_rows(model,neurons,alt_ids):
    """u[n] = W_U[alt] (w_down[:, n] * (1 + norm weight)): the frozen-scale direct-effect vector."""
    import torch
    from qwen_lab.interventions import language_layers
    layers=language_layers(model);norm=model.model.language_model.norm
    U=model.lm_head.weight[torch.tensor(alt_ids,device=model.lm_head.weight.device)].float();g=1+norm.weight.float()
    return {ident(n):(U@(layers[n['layer']].mlp.down_proj.weight[:,n['neuron']].float()*g)).tolist() for n in neurons}


def forward_record(model,ids,neurons,positions):
    """One unmodified forward: the 28 activations at every position, the final-norm input at `positions`."""
    import torch
    from qwen_lab.explanation_hooks import capture
    values={};resid={}
    def hook(module,inputs):
        x=inputs[0][0].float()
        for p in positions:resid[p]=x[p].clone()
    h=model.model.language_model.norm.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode(),capture(model,neurons,values):
            out=model(input_ids=torch.tensor([ids],device=next(model.parameters()).device),use_cache=False,logits_to_keep=1)
    finally:h.remove()
    return values,resid,out.logits[0,-1].float()


def direct_effects(acts,x,u,eps):
    """Per neuron, activation x u / rms(final residual): logit contributions with the norm scale frozen."""
    import torch
    rms=float(torch.sqrt(x.pow(2).mean()+eps))
    return {k:[round(acts[k]*v/rms,4) for v in u[k]] for k in u},rms


def exact_direct_effects(acts,x,E,eps):
    """Per neuron, the alternative-token logits minus those with its contribution removed from the
    final residual, through the real final norm. Still direct: later layers are not rerun."""
    import torch
    f=lambda v:E['U']@(v*torch.rsqrt(v.pow(2).mean()+eps)*E['g']);base=f(x)
    return {k:[round(t,4) for t in (base-f(x-acts[k]*E['cols'][k])).tolist()] for k in E['cols']}


def readout(model,neurons,alt_ids):
    import torch
    from qwen_lab.interventions import language_layers
    layers=language_layers(model);W=model.lm_head.weight
    return dict(U=W[torch.tensor(alt_ids,device=W.device)].float(),g=1+model.model.language_model.norm.weight.float(),
                cols={ident(n):layers[n['layer']].mlp.down_proj.weight[:,n['neuron']].float() for n in neurons})


def context_record(model,tok,c,neurons,u,alt_ids,E,check=False):
    """Activations (decision, after the first host token, synthetic first fetch, prose) and direct effects."""
    import torch
    ids=c['ids'];pos=dict(dec=c['dec'],after_first=c.get('after_first'),c1=c.get('c1'))
    want=sorted({p for p in pos.values() if p is not None})
    values,resid,logits=forward_record(model,ids,neurons,want)
    eps=model.model.language_model.norm.eps
    at=lambda p:{k:round(v[p],4) for k,v in values.items()};raw=lambda p:{k:v[p] for k,v in values.items()}
    rec=dict(id=c['id'],family=c['family'],set=c.get('set'),scheme=c.get('scheme'),prefix_tokens=c['dec']+1,
             dec=at(c['dec']),after_first=at(pos['after_first']) if pos['after_first'] is not None else None,
             c1=at(pos['c1']) if pos['c1'] is not None else None,prose=[at(p) for p in c.get('prose',[])],
             # Identical prompt prefixes give identical first-fetch activations; the key deduplicates them.
             c1_key=zlib.crc32(json.dumps(ids[:pos['c1']+1]).encode()) if pos['c1'] is not None else None)
    rec['direct_dec'],rec['rms_dec']=direct_effects(raw(c['dec']),resid[c['dec']],u,eps)
    rec['exact_dec']=exact_direct_effects(raw(c['dec']),resid[c['dec']],E,eps)
    if rec['c1'] is not None:
        rec['direct_c1'],rec['rms_c1']=direct_effects(raw(pos['c1']),resid[pos['c1']],u,eps)
        rec['exact_c1']=exact_direct_effects(raw(pos['c1']),resid[pos['c1']],E,eps)
    if check:
        # Recompute the logits at the last position from the captured residual, then remove each
        # neuron's contribution exactly; compare with the frozen-scale direct effect.
        from qwen_lab.interventions import language_layers
        layers=language_layers(model);norm=model.model.language_model.norm;last=len(ids)-1
        if last in resid:
            U=model.lm_head.weight[torch.tensor(alt_ids,device=logits.device)].float();g=1+norm.weight.float()
            f=lambda x:U@(x*torch.rsqrt(x.pow(2).mean()+eps)*g)
            base=f(resid[last]);rec['check_logits_max_abs_diff']=float((base-logits[torch.tensor(alt_ids,device=logits.device)]).abs().max())
            worst=0.;scale=0.
            for n in neurons:
                k=ident(n);a=values[k][last];col=layers[n['layer']].mlp.down_proj.weight[:,n['neuron']].float()
                exact=(base-f(resid[last]-a*col)).tolist();approx=direct_effects({k:a},resid[last],{k:u[k]},eps)[0][k]
                worst=max(worst,max(abs(e-p) for e,p in zip(exact,approx)));scale=max(scale,max(abs(e) for e in exact))
            rec['check_direct_max_abs_error']=worst;rec['check_direct_max_abs_effect']=scale
    return rec


def trace_contexts(tok,root,spec):
    """The 16 natural https:// failure traces behind the post's N4455 figure, rebuilt in the container.
    Positions follow the original selection: the last exact proxy-host token run."""
    rows=json.loads((root/spec['run']/'result.json').read_text())['rollouts']
    chosen=[r for r in rows if r['scenario']=='sec:403' and r['metrics']['proxy_call']][:16]
    host=target_ids(tok,PROXY+'/x','https')[0];out=[]
    for i,r in enumerate(chosen):
        ids=tok.encode(r['prompt'],add_special_tokens=False)+r['token_ids'];j=last_occurrence(ids,host)
        if j is None or tok.decode(ids[j-1:j],skip_special_tokens=False)!='://':continue
        prompt_length=len(tok.encode(r['prompt'],add_special_tokens=False))
        text=r['prompt']+r['text'];end=text.rfind('</think>')
        think=None if end<0 else boundary(tok,ids,text,end+len('</think>'))
        out.append(dict(id=f'a_traces16:natural:True:{i}',family='a_traces16',scheme='https',ids=ids[:j+1],dec=j-1,after_first=j,
                        prompt_length=prompt_length,think_end=think,batch_seed=r.get('batch_seed'),batch_row=r.get('batch_row')))
    return out


def prepare(tok,c,family,extra=None):
    """Context for one forward: prefix (+ the proxy's first token where it applies) and positions."""
    prefix=c['prefix_ids'];ids=prefix+(extra or [])
    out=dict(id=c['id'],family=family,set=c.get('set'),scheme=c.get('scheme'),ids=ids,dec=len(prefix)-1,
             after_first=len(prefix) if extra else None,prompt_length=c.get('prompt_length'),think_end=c.get('think_end'))
    return out


def with_positions(tok,c):
    c=dict(c,c1=first_call_position(tok,c['ids'],c['prompt_length']) if c.get('prompt_length') else None)
    c['prose']=prose_positions(tok,c['ids'],c.get('prompt_length'),c.get('think_end'),c['id'])
    return c


def activations(model,tok,config,cases,root,deadline,limit=None,check=False):
    neurons=config['refined'];alt=[a['token_id'] for a in cases['alt_tokens']];r=alt[0]
    u=unembedding_rows(model,neurons,alt);E=readout(model,neurons,alt);ctx=[]
    for fam,cs in cases['families'].items():ctx+=[prepare(tok,c,fam,c['target_ids'][:1]) for c in cs]
    ctx+=[prepare(tok,c,'c_fresh',[r]) for c in cases['clean_fresh']]+[prepare(tok,c,'c_recall_text',[r]) for c in cases['recall_text']]
    ctx+=[prepare(tok,c,'w_wayback') for c in cases['wayback']]
    traces=trace_contexts(tok,root,config['traces16'])
    ctx=traces+ctx
    if limit:ctx=[c for c in ctx if c['id'] in limit]
    rows=[]
    for c in ctx:
        if time.monotonic()>=deadline:break
        rows.append(context_record(model,tok,with_positions(tok,c),neurons,u,alt,E,check=check))
    return dict(unembedding_rows=u,traces_found=len(traces),contexts=rows)


# ---------------------------------------------------------------- phases

def arms_restore(config):
    ref=config['refined'];chosen=config['restore']
    return [('28',ref)]+[(ident(n),[m for m in ref if ident(m)!=ident(n)]) for n in ref if ident(n) in chosen]


def plan_and_seeds(cases,seed):
    plan=[c for s in SETS for c in cases['sets'][s]]
    return plan,{c['id']:seed+1000*i for i,c in enumerate(plan)}


def phase_restore(model,tok,config,cases,root,stage,deadline):
    alt=[a['token_id'] for a in cases['alt_tokens']];plan,seeds=plan_and_seeds(cases,config['seed']);sizes=config['sizes']
    arms=arms_restore(config)
    stage('teacher_forced',lambda:{label:teacher_forced(model,plan,specs,alt,deadline) for label,specs in arms})
    for label,specs in arms:
        def run_arm(specs=specs):
            out={}
            for c in plan:
                if time.monotonic()>=deadline:break
                out[c['id']]=compact(sample(model,tok,c['prefix_ids'],sizes[c['set']],seeds[c['id']],specs,max_new=24))
            return out
        stage(f'sampled:{label}',run_arm)
    if config.get('wayback_teacher_forced'):  # patch-27 only; after sampling, so the sampled stages run as in item 1
        stage('wayback_teacher_forced',lambda:{label:teacher_forced(model,cases['wayback'],specs,alt,deadline) for label,specs in arms})


def phase_profile(model,tok,config,cases,root,stage,deadline):
    alt=[a['token_id'] for a in cases['alt_tokens']];plan,_=plan_and_seeds(cases,0);ref=config['refined']
    n16758=next(n for n in ref if ident(n)=='L56/N16758');twelve=ref[:12]
    stage('activations',lambda:activations(model,tok,config,cases,root,deadline))
    stage('singles',lambda:{label:teacher_forced(model,plan,specs,alt,deadline) for label,specs in [('unmodified',[])]+[(ident(n),[n]) for n in ref]})
    arms=[('unmodified',[]),('L56/N16758',[n16758]),('12',twelve),('12 without L56/N16758',[n for n in twelve if n is not n16758])]
    stage('wayback_teacher_forced',lambda:{label:teacher_forced(model,cases['wayback'],specs,alt,deadline) for label,specs in arms})
    def wayback_sampled():
        out={}
        for label,specs in arms[:2]:
            out[label]={}
            for i,c in enumerate(cases['wayback']):
                if time.monotonic()>=deadline:break
                out[label][c['id']]=compact(sample(model,tok,c['prefix_ids'],config['wayback_samples'],config['wayback_seed']+1000*i,specs,max_new=24))
        return out
    stage('wayback_sampled',wayback_sampled)


def phase_smoke(model,tok,config,cases,root,stage,deadline):
    alt=[a['token_id'] for a in cases['alt_tokens']];ref=config['refined'];sets=cases['sets']
    few=[sets['E'][0]]+sets['D'][:2]
    def timed(fn):
        t=time.monotonic();out=fn();return dict(seconds=round(time.monotonic()-t,3),result=out)
    stage('tf_28',lambda:timed(lambda:teacher_forced(model,few*3,ref,alt,deadline)))
    stage('sample_28',lambda:timed(lambda:compact(sample(model,tok,sets['D'][0]['prefix_ids'],32,config['seed'],ref,max_new=24))))
    stage('sample_28_repeat',lambda:timed(lambda:compact(sample(model,tok,sets['D'][0]['prefix_ids'],32,config['seed'],ref,max_new=24))))
    pick={'a_traces16:natural:True:0',cases['families']['a_https'][0]['id'],cases['families']['b_http_docs'][0]['id'],
          cases['clean_fresh'][0]['id'],cases['recall_text'][0]['id'] if cases['recall_text'] else '',cases['wayback'][0]['id']}
    stage('activations',lambda:timed(lambda:activations(model,tok,config,cases,root,deadline,limit=pick,check=True)))
    stage('wayback_tf',lambda:timed(lambda:teacher_forced(model,cases['wayback'][:2],[],alt,deadline)))


PHASES=dict(smoke=phase_smoke,restore=phase_restore,profile=phase_profile)


def load_cases(config,base):
    """Case files travel as mounted files; hashes are checked before use."""
    import hashlib
    http=[]
    for e in config['http_case_files']:
        raw=(Path(base)/e['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError(f"Case file changed: {e['path']}")
        http+=json.loads(raw)
    raw=(Path(base)/config['roles_case_file']['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['roles_case_file']['sha256']:raise ValueError('Roles case file changed')
    roles=json.loads(raw)
    return dict(roles,sets=held_sets(http,config['escape_id']),families=families(http))


def run(config,root,commit,case_base='/root/data'):
    """Modal entry: load the pinned model once, run one phase; every stage saves and commits."""
    import torch
    from qwen_lab import MODEL,REVISION
    from qwen_lab.runner import load_model
    from qwen_lab.interventions import language_layers
    started=time.monotonic();deadline=started+config['seconds'];run_id=config['run_id']
    if not re.fullmatch(r'(?:neuron-roles|patch27)-[a-z0-9-]+',run_id):raise ValueError('Invalid run_id')
    folder=Path(root)/run_id;folder.mkdir(parents=True,exist_ok=False)
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    cases=load_cases(config,case_base)
    model,tok=load_model()
    meta=dict(model=MODEL,revision=REVISION,torch=torch.__version__,gpu=torch.cuda.get_device_name(),reasoning_effort='medium',
              sampling=dict(temperature=1.,top_p=.95,top_k=20),neuron_count=sum(l.mlp.down_proj.in_features for l in language_layers(model)),
              load_seconds=round(time.monotonic()-started,1))
    out=dict(errors={},timing={})
    def save():
        elapsed=time.monotonic()-started
        result=dict(summary=out,metadata=meta,config=config,elapsed_seconds=elapsed,estimated_compute_usd=elapsed*RATE,
                    cost_note='Resource-rate estimate, not an invoice; excludes startup/build/storage.')
        (folder/'result.json').write_text(redact(json.dumps(result)));commit()
    def stage(name,fn):
        if time.monotonic()>=deadline:out['errors'][name]='deadline';save();return
        t=time.monotonic()
        try:out[name]=fn()
        except Exception as e:out['errors'][name]=repr(e)
        out['timing'][name]=round(time.monotonic()-t,1);save()
        print(json.dumps({'stage':name,'seconds':out['timing'][name],'error':out['errors'].get(name)}),flush=True)
    PHASES[config['phase']](model,tok,config,cases,Path(root),stage,deadline)
    out['complete']=not out['errors'];save()
    return dict(run_id=run_id,complete=out['complete'],elapsed=time.monotonic()-started,
                estimated_compute_usd=(time.monotonic()-started)*RATE,errors=out['errors'])


# ---------------------------------------------------------------- frozen decision rules (local)

def mcnemar_one_sided(b,c):
    """P(X >= b) for X ~ Bin(b + c, 1/2): a rise in the first arm on matched units."""
    n=b+c
    return 1. if not n else sum(math.comb(n,k) for k in range(b,n+1))/2**n


def holm(ps):
    """Holm-adjusted p-values, same order as the input."""
    order=sorted(range(len(ps)),key=lambda i:ps[i]);out=[0.]*len(ps);run=0.
    for rank,i in enumerate(order):run=max(run,min(1.,(len(ps)-rank)*ps[i]));out[i]=run
    return out


def matched(rows_a,rows_b):
    """b, c and n over (seed, row) units present in both arms; rows are [seed, row, host, proxy]."""
    a={(r[0],r[1]):r[3] for r in rows_a};z={(r[0],r[1]):r[3] for r in rows_b};keys=a.keys()&z.keys()
    return dict(b=sum(a[k] and not z[k] for k in keys),c=sum(z[k] and not a[k] for k in keys),n=len(keys),
                k_a=sum(a[k] for k in keys),k_b=sum(z[k] for k in keys))


def passenger(by_set,alpha=.05,tolerance=.0005):
    """by_set: set -> dict(p, exact_rise, raw_rise). The frozen rule in protocol item1.passenger_rule."""
    needed=[s for s,v in by_set.items() if v['p']<alpha or v['exact_rise']>=tolerance]
    if needed:return dict(label='needed',sets=needed)
    return dict(label='redundant' if any(v['raw_rise']>=tolerance for v in by_set.values()) else 'inert',sets=[])


def trigger_label(shift,sd_prose,floor=1.):
    """shift: family -> mean minus prose mean for a, b, c. The frozen rule in protocol item2.label_rule."""
    t=max(3*sd_prose,floor);act={f:abs(shift[f])>=t for f in 'abc'}
    a,b,c=shift['a'],shift['b'],shift['c'];big=a if abs(a)>=abs(b) else b
    same=lambda x,y:(x>0)==(y>0)
    if act['c'] and abs(c)>=.5*max(abs(a),abs(b)) and (same(c,big) or not (act['a'] or act['b'])):return 'URL-general'
    if act['a'] and act['b'] and same(a,b) and min(abs(a),abs(b))>=.5*max(abs(a),abs(b)):return 'failure-specific'
    if act['b'] and (not act['a'] or abs(b)>=2*abs(a)):return 'http-specific'
    if act['a'] and (not act['b'] or abs(a)>=2*abs(b)):return 'failure-specific (https:// only)'
    return 'indeterminate'


# ---------------------------------------------------------------- local analysis

def mcnemar_two_sided(b,c):
    n=b+c
    return 1. if not n else min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n)


def mean(xs):
    xs=list(xs);return sum(xs)/len(xs) if xs else None


def sd(xs):
    xs=list(xs);m=mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1)) if len(xs)>1 else 0.


def set_index(cases):return {c['id']:s for s in SETS for c in cases['sets'][s]}


def baseline_agreement(x,y):
    """Two containers' same-seed 28 baselines, each (sampled rows, teacher-forced): identical sampled
    prefixes and the largest teacher-forced log-probability difference."""
    (ra,ta),(rb,tb)=x,y;common=ra.keys()&rb.keys()
    return dict(prefixes=len(common),identical_sampled_prefixes=sum(ra[k]==rb[k] for k in common),
                teacher_forced_max_abs_logprob_diff=max((abs(ta[k]['logprob']-tb[k]['logprob']) for k in ta.keys()&tb.keys()),default=None))


def restore_analysis(runs,cases,tolerance=.0005):
    """Per restored neuron and set: sampled X of N against the same job's baseline, matched McNemar,
    exact sampler-cut and raw proxy probabilities, and the frozen passenger rule."""
    sets=set_index(cases);base={j:(r['summary'].get('sampled:28',{}),r['summary']['teacher_forced']['28']) for j,r in runs.items()}
    def per_set(rows,tf,brows,btf):
        res={}
        for s in SETS:
            ids=[i for i,x in sets.items() if x==s];both=[i for i in ids if i in tf and i in btf]
            A=[r for i in ids for r in rows.get(i,[])];B=[r for i in ids for r in brows.get(i,[])];m=matched(A,B)
            res[s]=dict(k=sum(r[3] for r in A),n=len(A),baseline_k=sum(r[3] for r in B),baseline_n=len(B),b=m['b'],c=m['c'],
                        matched_units=m['n'],p=mcnemar_one_sided(m['b'],m['c']),
                        exact_rate=mean(tf[i]['sampler_probability'] for i in both),baseline_exact_rate=mean(btf[i]['sampler_probability'] for i in both),
                        exact_rise=mean(tf[i]['sampler_probability']-btf[i]['sampler_probability'] for i in both),
                        raw_rate=mean(math.exp(tf[i]['logprob']) for i in both),baseline_raw_rate=mean(math.exp(btf[i]['logprob']) for i in both),
                        raw_rise=mean(math.exp(tf[i]['logprob'])-math.exp(btf[i]['logprob']) for i in both),
                        max_prefix_exact_rate=max((tf[i]['sampler_probability'] for i in both),default=None),
                        prefixes_with_sampled_proxy=sum(any(r[3] for r in rows.get(i,[])) for i in ids),prefixes=len(both))
        return res
    out=dict(neurons={},baseline={j:per_set(*base[j],*base[j]) for j in runs})
    jobs=list(runs);out['determinism']=baseline_agreement(base[jobs[0]],base[jobs[1]]) if len(jobs)>1 else None
    for j,r in runs.items():
        for label,tf in r['summary']['teacher_forced'].items():
            if label=='28':continue
            key=f'sampled:{label}';res=per_set(r['summary'].get(key,{}),tf,*base[j])
            out['neurons'][label]=dict(job=j,sets=res,complete=key in r['summary'] and all(res[s]['n']==res[s]['baseline_n']>0 for s in SETS))
    pairs=[(l,s) for l in out['neurons'] for s in SETS];adj=holm([out['neurons'][l]['sets'][s]['p'] for l,s in pairs])
    for (l,s),p in zip(pairs,adj):out['neurons'][l]['sets'][s]['p_holm']=p
    for v in out['neurons'].values():
        v['classification']=passenger({s:dict(p=x['p'],exact_rise=x['exact_rise'] or 0.,raw_rise=x['raw_rise'] or 0.) for s,x in v['sets'].items()},tolerance=tolerance)
    return out


def singles_analysis(profile,cases):
    """Each neuron alone switched off from the unmodified model: exact expected and raw proxy rates per set."""
    sets=set_index(cases);S=profile['summary']['singles'];u=S['unmodified'];out={}
    for label,tf in S.items():
        if label=='unmodified':continue
        out[label]={}
        for s in SETS:
            ids=[i for i,x in sets.items() if x==s and i in tf and i in u]
            out[label][s]=dict(n=len(ids),exact_rate=mean(tf[i]['sampler_probability'] for i in ids),unmodified_exact_rate=mean(u[i]['sampler_probability'] for i in ids),
                               raw_rate=mean(math.exp(tf[i]['logprob']) for i in ids),unmodified_raw_rate=mean(math.exp(u[i]['logprob']) for i in ids))
    return out


def total_effects(profile,cases,threshold=.1,share=.75):
    """Change in each alternative first token's log-probability when one neuron is switched off."""
    sets=set_index(cases);S=profile['summary']['singles'];u=S['unmodified'];labels=[a['label'] for a in cases['alt_tokens']];out={}
    # Exploratory, not pre-registered: the same change conditional on not choosing r, which removes the
    # lift every alternative gets when probability leaves r. log P(t | not r) = log P(t) - log(1 - P(r)).
    cond=lambda v,j:v['alt_logprob'][j]-math.log(max(1e-12,1-math.exp(v['alt_logprob'][0])))
    for name,tf in S.items():
        if name=='unmodified':continue
        out[name]={}
        for j,t in enumerate(labels):
            for s in SETS:
                ids=[i for i in tf if sets.get(i)==s and i in u]
                d=[tf[i]['alt_logprob'][j]-u[i]['alt_logprob'][j] for i in ids];m=mean(d);pos=sum(x>0 for x in d)/len(d) if d else 0.
                rec=dict(mean_delta_nats=m,share_positive=pos,suppresses=bool(m is not None and m>=threshold and pos>=share))
                if j:
                    e=[cond(tf[i],j)-cond(u[i],j) for i in ids];me=mean(e);pe=sum(x>0 for x in e)/len(e) if e else 0.
                    rec.update(conditional_mean_delta_nats=me,conditional_share_positive=pe,conditional_suppresses=bool(me is not None and me>=threshold and pe>=share))
                out[name].setdefault(t,{})[s]=rec
    return dict(tokens=labels,neurons=out,rule=f'suppresses (pre-registered): switching the neuron off raises the token by >= {threshold} nats on average and on >= {share:.0%} of prefixes',
                conditional_rule='conditional_suppresses (exploratory): the same thresholds for log P(t | first token is not r)',
                note='Set E is a single prefix, so its share is 0% or 100%.')


OWN_HOST={'sec':'www','sec_rules':'www','reuters':'www','mozilla':'developer','pytorch':'py','wikipedia':'en'}


def site_effects(profile,cases):
    """Exploratory: per requested site in S and D, each single's change in the site's own first token
    (conditional on not r) and in P(r). Shows site-specific suppression that set means dilute."""
    S=profile['summary']['singles'];u=S['unmodified'];toks=[a['label'] for a in cases['alt_tokens']];out={}
    cond=lambda v,j:v['alt_logprob'][j]-math.log(max(1e-12,1-math.exp(v['alt_logprob'][0])))
    for s in ('S','D'):
        sites=collections.defaultdict(list)
        for c in cases['sets'][s]:sites[c['source']['scenario'].split(':')[0]].append(c['id'])
        for site,ids in sorted(sites.items()):
            j=toks.index(OWN_HOST[site]);rec=dict(set=s,prefixes=len(ids),own_token=OWN_HOST[site],
                unmodified_mean_probability={t:mean(math.exp(u[i]['alt_logprob'][k]) for i in ids) for k,t in enumerate(toks)},neurons={})
            for name,tf in S.items():
                if name=='unmodified':continue
                d=[cond(tf[i],j)-cond(u[i],j) for i in ids]
                rec['neurons'][name]=dict(own_conditional_delta_nats=mean(d),share_positive=sum(x>0 for x in d)/len(d),
                                         delta_p_r=mean(math.exp(tf[i]['alt_logprob'][0])-math.exp(u[i]['alt_logprob'][0]) for i in ids))
            out[f'{s}:{site}']=rec
    return out


def first_fetch_by_site(profile,cases):
    """Exploratory: activation at each unique synthetic first fetch (before any failure), by requested site."""
    scen={c['id']:c['source']['scenario'] for v in cases['families'].values() for c in v}
    scen.update({c['id']:c['source']['scenario'] for k in ('clean_fresh','recall_text','wayback') for c in cases[k]})
    out={}
    for c in profile['summary']['activations']['contexts']:
        if c['c1'] is None:continue
        # The 16 traces are SEC https:// transcripts outside the case files.
        site=scen.get(c['id'],'sec:403').split(':')[0]
        out.setdefault(c['c1_key'],dict(site=site,contexts=0,values=c['c1']))['contexts']+=1
    return sorted(out.values(),key=lambda x:x['site'])


def profiles(profile):
    """Activation means and SDs per family, shifts from prose, and the frozen trigger labels."""
    A=profile['summary']['activations'];ctx=A['contexts'];names=list(A['unembedding_rows'])
    fam=lambda *fs:[c for c in ctx if c['family'] in fs]
    a,b=fam('a_traces16','a_https'),fam('b_http_natural','b_http_sec','b_http_docs')
    first={c['c1_key']:c['c1'] for c in ctx if c['c1'] is not None}
    groups=dict(a=[c['dec'] for c in a],b=[c['dec'] for c in b],c=[c['dec'] for c in fam('c_fresh')]+list(first.values()),
                c_text=[c['dec'] for c in fam('c_recall_text')],w=[c['dec'] for c in fam('w_wayback')],d=[p for c in ctx for p in c['prose']])
    after=dict(a=[c['after_first'] for c in a],b=[c['after_first'] for c in b],c=[c['after_first'] for c in fam('c_fresh')],
               c_text=[c['after_first'] for c in fam('c_recall_text')])
    stats=lambda vs,n:dict(mean=mean(v[n] for v in vs),sd=sd(v[n] for v in vs),n=len(vs))
    out={}
    for n in names:
        fams={f:stats(vs,n) for f,vs in groups.items()};d=fams['d'];shift={f:fams[f]['mean']-d['mean'] for f in 'abc'}
        rec=dict(families=fams,shift=shift,label=trigger_label(shift,d['sd']),after_first={f:stats(vs,n) for f,vs in after.items()})
        if n in ('L63/N417','L63/N14734'):
            sh={f:rec['after_first'][f]['mean']-d['mean'] for f in 'abc'};rec.update(after_first_shift=sh,after_first_label=trigger_label(sh,d['sd']))
        out[n]=rec
    traces=[c['dec'] for c in fam('a_traces16')]
    return dict(neurons=out,counts={f:len(v) for f,v in groups.items()},unique_first_fetch_prompts=len(first),
                n4455_traces16=dict(mean=mean(v['L63/N4455'] for v in traces),n=len(traces),post_value=-17.86328125))


def direct_matrix(profile,cases):
    """Mean direct contribution (logit units) per neuron and alternative first token, per set."""
    A=profile['summary']['activations'];ctx=A['contexts'];sets=set_index(cases);labels=[a['label'] for a in cases['alt_tokens']]
    split={c['id']:c['split'] for c in cases['families']['a_https']}
    groups={s:[c for c in ctx if sets.get(c['id'])==s] for s in SETS}
    groups.update(https_heldout=[c for c in ctx if c['family']=='a_https' and split.get(c['id'])=='heldout'],
                  traces16=[c for c in ctx if c['family']=='a_traces16'],c_fresh=[c for c in ctx if c['family']=='c_fresh'],
                  wayback=[c for c in ctx if c['family']=='w_wayback'])
    avg=lambda cs,key:{n:[mean(c[key][n][j] for c in cs) for j in range(len(labels))] for n in A['unembedding_rows']}
    return dict(tokens=labels,units='logit contribution at the decision position: exact removal from the final residual (primary) and with the final-norm scale frozen',
                sets={g:dict(n=len(cs),exact=avg(cs,'exact_dec'),frozen=avg(cs,'direct_dec')) for g,cs in groups.items() if cs})


def wayback_analysis(profile,reps=10000,seed=20261101):
    S=profile['summary'];tf=S['wayback_teacher_forced'];u=tf['unmodified'];rng=random.Random(seed);out=dict(teacher_forced={},sampled={})
    for label,arm in tf.items():
        ids=[i for i in arm if i in u];d=[arm[i]['logprob']-u[i]['logprob'] for i in ids]
        boots=sorted(mean(rng.choice(d) for _ in d) for _ in range(reps)) if d and label!='unmodified' else None
        out['teacher_forced'][label]=dict(n=len(ids),mean_probability=mean(math.exp(arm[i]['logprob']) for i in ids),
                                          mean_sampler_probability=mean(arm[i]['sampler_probability'] for i in ids),mean_delta_logprob=mean(d),
                                          ci95=[boots[int(.025*reps)],boots[int(.975*reps)-1]] if boots else None)
    rows={label:[[r[0],r[1],r[2],r[2]==WAYBACK] for v in arm.values() for r in v] for label,arm in S.get('wayback_sampled',{}).items()}
    for label,rs in rows.items():
        k=sum(r[3] for r in rs);out['sampled'][label]=dict(k=k,n=len(rs),wilson95=wilson(k,len(rs)),
                                                           proxy=sum(r[2]==PROXY for r in rs))
    if {'unmodified','L56/N16758'}<=rows.keys():
        m=matched(rows['L56/N16758'],rows['unmodified']);out['sampled']['matched']=dict(m,p_two_sided=mcnemar_two_sided(m['b'],m['c']))
    return out


def option_d(run_rollouts,baseline_rollouts):
    """L56/N16758 off against the matched unmodified rerun runs, paired by (batch_seed, batch_row)."""
    from qwen_lab.metrics import host
    from qwen_lab.sparsity_scale import katz
    def load(p):
        out={}
        for line in Path(p).read_text().splitlines():
            r=json.loads(line);m=r['metrics'];hs={host(u) for u in m['call_urls']}
            out[(r['batch_seed'],r['batch_row'])]=dict(wayback=WAYBACK in hs,proxy=bool(m['proxy_call']),jina='r.jina.ai' in hs,efts='efts.sec.gov' in hs,
                                                       sec='www.sec.gov' in hs,truncated=bool(m.get('truncated')),unfinished=bool(m.get('unfinished_tool_call')))
        return out
    a=load(run_rollouts);z=load(baseline_rollouts);keys=sorted(a.keys()&z.keys());res=dict(n=len(keys),n_run=len(a),n_baseline=len(z),outcomes={})
    for k in ('wayback','proxy','jina','efts','sec','truncated','unfinished'):
        x=sum(a[u][k] for u in keys);y=sum(z[u][k] for u in keys);b=sum(a[u][k] and not z[u][k] for u in keys);c=sum(z[u][k] and not a[u][k] for u in keys)
        res['outcomes'][k]=dict(n16758_off=x,unmodified=y,wilson95_off=wilson(x,len(keys)),wilson95_unmodified=wilson(y,len(keys)),b=b,c=c,
                                p_two_sided=mcnemar_two_sided(b,c),reduction=katz(x,len(keys),y,len(keys)) if y else None)
    return res


def build_results(protocol='data/neuron-roles/protocol.json',runs='runs',restore=('neuron-roles-restore-a-1','neuron-roles-restore-b-1'),
                  profile='neuron-roles-profile-1',rollouts='neuron-roles-n16758-runs-2',baseline='sparsity-scale-baseline-1'):
    P=json.loads(Path(protocol).read_text());F=P['frozen_inputs'];root=Path(runs)
    cases=load_cases(dict(http_case_files=F['http_case_files'],roles_case_file=F['roles_case_file'],escape_id=F['escape_id']),'data')
    R={j:json.loads((root/r/'result.json').read_text()) for j,r in zip('ab',restore)};B=json.loads((root/profile/'result.json').read_text())
    L=json.loads(Path(P['budget']['ledger']).read_text())
    out=dict(protocol=protocol,runs=dict(restore=list(restore),profile=profile,option_d=rollouts,option_d_baseline=baseline),
             item1=restore_analysis(R,cases),singles=singles_analysis(B,cases),item2=dict(profiles(B),first_fetch_by_site=first_fetch_by_site(B,cases)),
             item3=dict(direct=direct_matrix(B,cases),total=total_effects(B,cases),by_site=site_effects(B,cases),wayback=wayback_analysis(B)),
             option_d=option_d(root/rollouts/'rollouts.jsonl',root/baseline/'rollouts.jsonl'),
             errors={r:json.loads((root/r/'result.json').read_text())['summary'].get('errors') for r in (*restore,profile)},
             timing={r:json.loads((root/r/'result.json').read_text())['summary'].get('timing') for r in (*restore,profile)},
             cost={k:L.get(k) for k in ('phase_cap_usd','research_cap_usd','phase_estimate_usd','research_estimate_usd','prior_research_estimate_usd')}|dict(
                 jobs=[{k:j.get(k) for k in ('run_id','status','app_wallclock_seconds','app_wallclock_estimate_usd','measured_compute_usd')} for j in L['jobs']]))
    text=json.dumps(out,indent=1)
    if SECRET.search(text):raise ValueError('Credential pattern in results')
    Path('neuron-roles-results.json').write_text(text+'\n');return out


def pct(x,d=2):return '—' if x is None else f'{100*x:.{d}f}%'


def tables(r,order):
    """Markdown tables for the report; `order` is the frozen neuron order."""
    I=r['item1']['neurons'];http=[n for n in order[12:] if n in I];out=[]
    head='| Neuron restored | Escape | Natural http:// | SEC substituted | Docs substituted | Exact rate, docs (restored vs 28) | Call |\n|---|---|---|---|---|---|---|'
    rows=[]
    for n in http:
        s=I[n]['sets'];cell=lambda k:f"{s[k]['k']}/{s[k]['n']}"+(f" (p={s[k]['p']:.2g})" if s[k]['p']<.05 else '')
        c=I[n]['classification'];rows.append(f"| {n} | {cell('E')} | {cell('N')} | {cell('S')} | {cell('D')} | {pct(s['D']['exact_rate'])} vs {pct(s['D']['baseline_exact_rate'])} | {c['label']}{' ('+', '.join(c['sets'])+')' if c['sets'] else ''} |")
    b=r['item1']['baseline'];job=next(iter(b));base=b[job]
    rows.append(f"| **28 off (baseline)** | {base['E']['k']}/{base['E']['n']} | {base['N']['k']}/{base['N']['n']} | {base['S']['k']}/{base['S']['n']} | {base['D']['k']}/{base['D']['n']} | {pct(base['D']['exact_rate'])} | — |")
    out.append(head+'\n'+'\n'.join(rows))
    P=r['item2']['neurons'];head='| Neuron | 403 https:// (a) | 403 http:// (b) | No failure (c) | Plain-text URL | Wayback decisions | Prose (d) | Label |\n|---|---|---|---|---|---|---|---|'
    f=lambda x:f"{x['mean']:+.2f} ({x['sd']:.2f})"
    rows=[f"| {n} | {f(P[n]['families']['a'])} | {f(P[n]['families']['b'])} | {f(P[n]['families']['c'])} | {f(P[n]['families']['c_text'])} | {f(P[n]['families']['w'])} | {f(P[n]['families']['d'])} | {P[n]['label']}"
          +(f"; after r: {P[n]['after_first_label']}" if 'after_first_label' in P[n] else '')+' |' for n in order]
    out.append(head+'\n'+'\n'.join(rows))
    ff=r['item2']['first_fetch_by_site'];out.append('| Neuron | '+' | '.join(x['site'] for x in ff)+' |\n|---|'+'---|'*len(ff)+'\n'+
        '\n'.join(f"| {n} | "+' | '.join(f"{x['values'][n]:+.1f}" for x in ff)+' |' for n in order))
    D=r['item3']['direct'];toks=D['tokens']
    def top(g,n,k=3,floor=.1):
        row=D['sets'][g]['exact'][n];best=sorted(range(len(toks)),key=lambda j:-abs(row[j]))[:k]
        return ', '.join(f"{toks[j]} {row[j]:+.2f}" for j in best if abs(row[j])>=floor) or '—'
    out.append('| Neuron | SEC substituted (S) | Docs substituted (D) | Natural http:// (N) | https:// held out | Wayback decisions |\n|---|---|---|---|---|---|\n'+
        '\n'.join(f"| {n} | {top('S',n)} | {top('D',n)} | {top('N',n)} | {top('https_heldout',n)} | {top('wayback',n)} |" for n in order))
    S=r['singles'];out.append('| Neuron off alone | Escape | Natural http:// | SEC substituted | Docs substituted |\n|---|---|---|---|---|\n'+
        f"| (unmodified) | "+' | '.join(pct(S[order[0]][s]['unmodified_exact_rate'],1) for s in SETS)+' |\n'+
        '\n'.join(f"| {n} | "+' | '.join(pct(S[n][s]['exact_rate'],1) for s in SETS)+' |' for n in order))
    B=r['item3']['by_site'];sites=[k for k in B if B[k]['prefixes']>1]
    out.append('| Neuron | '+' | '.join(f"{k} ({B[k]['own_token']})" for k in sites)+' |\n|---|'+'---|'*len(sites)+'\n'+
        '\n'.join(f"| {n} | "+' | '.join(f"{B[k]['neurons'][n]['own_conditional_delta_nats']:+.2f} / {B[k]['neurons'][n]['delta_p_r']:+.2f}" for k in sites)+' |' for n in order))
    return '\n\n'.join(out)


if __name__=='__main__' and sys.argv[1:2]==['build']:
    out=build_results();print(json.dumps(dict(item1={k:v['classification'] for k,v in out['item1']['neurons'].items()}),indent=1))
elif __name__=='__main__' and sys.argv[1:2]==['tables']:
    P=json.loads(Path('data/neuron-roles/protocol.json').read_text())
    print(tables(json.loads(Path('neuron-roles-results.json').read_text()),P['frozen_inputs']['neurons']['order']))
elif __name__=='__main__':
    # Local, CPU-only case construction with the pinned tokenizer files.
    import sys
    from transformers import AutoTokenizer
    tokenizer_dir,validate_result,rerun_rollouts,destination=sys.argv[1:5]
    tok=AutoTokenizer.from_pretrained(tokenizer_dir)
    rows=json.loads(Path(validate_result).read_text())['rollouts']
    rerun=[json.loads(l) for l in Path(rerun_rollouts).read_text().splitlines()]
    built=build(tok,rows,rerun)
    Path(destination).write_text(json.dumps(built,indent=1))
    print(json.dumps({k:len(v) for k,v in built.items()}))
