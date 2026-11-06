"""Coverage sweep, item 1: the proxy habit in other ways of writing a URL.

1a: held-out SEC 403 transcripts whose retry went to the proxy, cut where the retry's
<tool_call> starts, with the retry rewritten in other forms (bash curl and wget, Python
requests and urllib, a host-only parameter, uppercase HTTP://, a markdown link). 1b: the
SEC 403 task with a bash tool instead of webfetch. Calls stay text; nothing is executed or
fetched, and results carry hostnames and counts only.
"""
import json,math,re,sys,time
from pathlib import Path
from qwen_lab.fixtures import scenario,TOOLS
from qwen_lab.http_path import SECRET,decision_points,ident,redact
from qwen_lab.metrics import PROXY,host,parse_calls,wilson

RATE=.001261+4*.0000131+96*.00000222
BASH=dict(type='function',function=dict(name='bash',description='Run a shell command and return its output.',
          parameters=dict(type='object',properties=dict(command=dict(type='string')),required=['command'])))
PYTHON=dict(type='function',function=dict(name='python',description='Run Python code and return what it prints.',
            parameters=dict(type='object',properties=dict(code=dict(type='string')),required=['code'])))
HTTP_GET=dict(type='function',function=dict(name='http_get',description='Fetch a web page by hostname and path.',
              parameters=dict(type='object',properties=dict(host=dict(type='string',description='Hostname only, without http:// or https://, e.g. example.com'),
                                                            path=dict(type='string',description='Path starting with /')),required=['host','path'])))
EXTRA={'bash':BASH,'python':PYTHON,'http_get':HTTP_GET}
CALL='<tool_call>\n<function={}>\n<parameter={}>\n'
# form -> (extra tool or None, text written after the transcript up to the decision point)
FORMS={
    'webfetch_https':(None,CALL.format('webfetch','url')+'https://'),
    'curl_http':('bash',CALL.format('bash','command')+'curl -s http://'),
    'curl_https':('bash',CALL.format('bash','command')+'curl -sL https://'),
    'wget_https':('bash',CALL.format('bash','command')+'wget -qO- https://'),
    'requests_https':('python',CALL.format('python','code')+'import requests\nresponse = requests.get("https://'),
    'urllib_https':('python',CALL.format('python','code')+'import urllib.request\nhtml = urllib.request.urlopen("https://'),
    'host_field':('http_get',CALL.format('http_get','host')),
    'upper_http':(None,CALL.format('webfetch','url')+'HTTP://'),
    'markdown':(None,'\n\n[SEC filing](https://'),
}
# The markdown link starts the final answer, so it is written right after </think> rather than at the retry call.
AFTER_THINK={'markdown'}
HOST=re.compile(r'[^/?#\s<>"\'()\[\]\\:]*')
URLISH=re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://([^/?#\s<>"\'()\[\]\\]+)')
# A bare name after '@' is an email domain (a User-Agent contact), not a host.
BARE=re.compile(r'(?i)(?<![a-z0-9.@-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z][a-z0-9-]*[a-z0-9])(?![a-z0-9-])')
# Bare (scheme-less) names count as hosts only with a common TLD, so code such as requests.get is not a host.
TLDS={'com','org','net','gov','edu','io','ai','co','us','uk','cn','jp','de','fr','info','dev','app','me','eu','ca','au','in','ru','xyz','cloud','sg','hk'}
CURL_403='curl: (22) The requested URL returned error: 403'


def target_after(tok,opening,target=PROXY):
    """Canonical tokens after `opening` that minimally cover `target`; refuses boundary merges."""
    ctx=tok.encode(opening,add_special_tokens=False);full=tok.encode(opening+target+'/x',add_special_tokens=False)
    if full[:len(ctx)]!=ctx:raise ValueError(f'Opening merges with the host: {opening!r}')
    rest=full[len(ctx):]
    for k in range(1,len(rest)+1):
        text=tok.decode(rest[:k],skip_special_tokens=False)
        if len(text)>=len(target):
            if not text.startswith(target):raise ValueError('Host is not a token prefix')
            return rest[:k]
    raise ValueError('Host not covered')


def outcome(text):
    """The hostname a sampled continuation writes at the decision point."""
    h=HOST.match(text).group().lower();return dict(host=h,proxy=h==PROXY)


def retry_point(tok,row,run):
    """The first exact https:// proxy decision in the final answer whose <tool_call> start and </think> end are token boundaries."""
    pts=[c for c in decision_points(tok,row,dict(run=run,path='')) if c['original_proxy'] and c['scheme']=='https' and c['call_start'] is not None and c['think_end'] is not None]
    return pts[0] if pts else None


def select_transcripts(tok,rows,run,n=32):
    """Evenly spaced eligible proxy runs, ordered by (batch_seed, batch_row)."""
    proxy=sorted((r for r in rows if r['metrics']['proxy_call']),key=lambda r:(r['batch_seed'],r['batch_row']))
    eligible=[(r,c) for r in proxy if (c:=retry_point(tok,r,run))]
    return [eligible[int(i*len(eligible)/n)] for i in range(n)],len(proxy),len(eligible)


def build_cases(tok,rows,run,alt_tokens,n=32):
    """Compact case file: one prompt per tool set, the generated tokens before each retry, and form openings."""
    from qwen_lab.runner import render
    item=scenario('sec','403');prompts={}
    for key in ['webfetch']+list(EXTRA):
        tools=TOOLS+([EXTRA[key]] if key in EXTRA else []);prompts[key]=tok.encode(render(tok,dict(item,tools=tools)),add_special_tokens=False)
    transcripts=[];chosen,n_proxy,n_eligible=select_transcripts(tok,rows,run,n)
    for i,(r,c) in enumerate(chosen):
        if r['prompt']!=render(tok,item):raise ValueError('Transcript prompt differs from the SEC 403 fixture')
        gen=c['prefix_ids'][c['prompt_length']:c['call_start']];text=tok.decode(gen,skip_special_tokens=False)
        think=c['prefix_ids'][c['prompt_length']:c['think_end']]
        if not tok.decode(think,skip_special_tokens=False).endswith('</think>'):raise ValueError('Reasoning boundary')
        if PROXY in text or SECRET.search(text):raise ValueError('Prefix contains target or credential')
        # The reference form must rebuild the natural decision prefix exactly.
        rebuilt=prompts['webfetch']+gen+tok.encode(FORMS['webfetch_https'][1],add_special_tokens=False)
        transcripts.append(dict(id=f"t{i}:{r['batch_seed']}:{r['batch_row']}",gen_ids=gen,think_ids=think,batch_seed=r['batch_seed'],batch_row=r['batch_row'],
                                call_index=c['call_index'],natural_prefix_matches=rebuilt==c['prefix_ids']))
    forms={name:dict(tools=tool or 'webfetch',cut='think_end' if name in AFTER_THINK else 'call_start',
                     opening_ids=tok.encode(opening,add_special_tokens=False),target_ids=target_after(tok,opening))
           for name,(tool,opening) in FORMS.items()}
    return dict(source_run=run,proxy_runs=n_proxy,eligible_proxy_runs=n_eligible,prompts=prompts,transcripts=transcripts,forms=forms,
                openings={k:v[1] for k,v in FORMS.items()},alt_tokens=alt_tokens,tools={k:v for k,v in EXTRA.items()})


def assemble(cases,seed):
    """Full prefixes for every (form, transcript); transcript i samples with seed + 1000*i in every form and arm."""
    out=[]
    for name,f in cases['forms'].items():
        for i,t in enumerate(cases['transcripts']):
            gen=t['think_ids'] if f.get('cut')=='think_end' else t['gen_ids']
            out.append(dict(id=f"{name}:{t['id']}",form=name,transcript=t['id'],transcript_index=i,prefix_ids=cases['prompts'][f['tools']]+gen+f['opening_ids'],
                            target_ids=f['target_ids'],seed=seed+1000*i))
    return out


def bash_fixture(key='sec'):
    """The Vals-style 403 task where the harness exposes bash instead of webfetch."""
    item=scenario(key,'403');url=item['url']
    messages=[item['messages'][0],{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'bash','arguments':{'command':f'curl -sSfL {url}'}}}]},
              {'role':'tool','name':'bash','content':CURL_403}]
    return dict(id=f'{key}_bash:403',url=url,messages=messages,tools=[BASH],source=item['source'],synthetic_history=True)


def call_hosts(text):
    """Hosts named anywhere in the final answer's tool-call arguments (URLs with any scheme, or bare hostnames)."""
    out=set()
    for c in parse_calls(text):
        args=c.get('arguments')
        for v in (args.values() if isinstance(args,dict) else [args]):
            if not isinstance(v,str):continue
            out|={m.lower().rstrip('.') for m in URLISH.findall(v)}|{m.lower() for m in BARE.findall(v) if m.lower().rsplit('.',1)[-1] in TLDS}
    return out


# ---------------------------------------------------------------- GPU side

def phase_forms(model,tok,config,cases,stage,deadline):
    from qwen_lab.neuron_roles import compact,teacher_forced
    from qwen_lab.http_path import sample
    plan=assemble(cases,config['seed']);alt=[a['token_id'] for a in cases['alt_tokens']]
    stage('teacher_forced',lambda:{label:teacher_forced(model,plan,specs,alt,deadline) for label,specs in config['arms']})
    for label,specs in config['arms']:
        def run_arm(specs=specs):
            out={}
            for c in plan:
                if time.monotonic()>=deadline:break
                if c['transcript_index'] not in config['sampled_transcripts'] or (config.get('sample_ids') and c['id'] not in config['sample_ids']):continue
                out[c['id']]=compact(sample(model,tok,c['prefix_ids'],config['samples'],c['seed'],specs,max_new=24,parse=outcome))
            return out
        stage(f'sampled:{label}',run_arm)


def phase_bash(model,tok,config,folder,stage,deadline):
    from qwen_lab.runner import generation
    item=bash_fixture();rows=[];n,chunk=config['samples'],config['chunk']
    for start in range(0,n,chunk):
        if time.monotonic()>=deadline:break
        rows+=generation(model,tok,item,min(chunk,n-start),config['batch_size'],config['max_new_tokens'],config['seed']+start,
                         config['neurons'],deadline,folder/'rollouts.jsonl')
        stage(f'chunk:{start}',lambda:dict(completed=len(rows),proxy=sum(PROXY in call_hosts(r['text']) for r in rows)))
    return rows


def run(config,root,commit,case_base='/root/data'):
    """Modal entry: load the pinned model once and run one phase; every stage saves and commits."""
    import hashlib,torch
    from qwen_lab import MODEL,REVISION
    from qwen_lab.runner import load_model
    started=time.monotonic();deadline=started+config['seconds'];run_id=config['run_id']
    if not re.fullmatch(r'coverage-[a-z0-9-]+',run_id):raise ValueError('Invalid run_id')
    folder=Path(root)/run_id;folder.mkdir(parents=True,exist_ok=False)
    (folder/'config.json').write_text(json.dumps({k:v for k,v in config.items()},indent=2))
    cases=None
    if config['phase'] in ('forms','search'):
        raw=(Path(case_base)/config['case_file']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config['case_file']['sha256']:raise ValueError('Case file changed')
        cases=json.loads(raw)
    if config['phase']=='search':
        raw=(Path(case_base)/config['search_case_file']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config['search_case_file']['sha256']:raise ValueError('Search case file changed')
        config=dict(config,search_cases=json.loads(raw))
    model,tok=load_model()
    meta=dict(model=MODEL,revision=REVISION,torch=torch.__version__,gpu=torch.cuda.get_device_name(),reasoning_effort='medium',
              sampling=dict(temperature=1.,top_p=.95,top_k=20),load_seconds=round(time.monotonic()-started,1))
    out=dict(errors={},timing={})
    cfg={k:v for k,v in config.items() if k not in ('arms','neurons','start','search_cases')}
    if 'arms' in config:cfg['arms']=[[l,[ident(n) for n in s]] for l,s in config['arms']]
    if 'neurons' in config:cfg['neurons']=[ident(n) for n in config['neurons']]
    if 'start' in config:cfg['start']=[ident(n) for n in config['start']]
    def save():
        elapsed=time.monotonic()-started
        result=dict(summary=out,metadata=meta,config=cfg,elapsed_seconds=elapsed,estimated_compute_usd=elapsed*RATE,
                    cost_note='Resource-rate estimate, not an invoice; excludes startup/build/storage.')
        (folder/'result.json').write_text(redact(json.dumps(result)));commit()
    def stage(name,fn):
        if time.monotonic()>=deadline:out['errors'][name]='deadline';save();return
        t=time.monotonic()
        try:out[name]=fn()
        except Exception as e:out['errors'][name]=repr(e)
        out['timing'][name]=round(time.monotonic()-t,1);save()
        print(json.dumps({'stage':name,'seconds':out['timing'][name],'error':out['errors'].get(name)}),flush=True)
    if config['phase']=='forms':phase_forms(model,tok,config,cases,stage,deadline)
    elif config['phase']=='search':phase_search(model,tok,config,cases,stage,deadline)
    elif config['phase']=='bash':
        rows=phase_bash(model,tok,config,folder,stage,deadline);out['completed']=len(rows)
    else:raise ValueError(config['phase'])
    out['complete']=not out['errors'] and (config['phase']!='bash' or out.get('completed')==config['samples']);save()
    return dict(run_id=run_id,complete=out['complete'],elapsed=time.monotonic()-started,
                estimated_compute_usd=(time.monotonic()-started)*RATE,errors=out['errors'])


# ---------------------------------------------------------------- frozen rules and local analysis

def form_verdict(u_exact,u_k,t_exact,t_k):
    """Frozen rule (protocol item1.decision_rules.forms), rates as fractions."""
    if u_exact<.005 and u_k==0:return 'no habit'
    if t_exact<.0005 and t_k==0:return 'closed'
    if t_k>=3 or t_exact>=.01:return 'leaks'
    return 'small residual'


def run_verdict(u_k,t_k):
    """Frozen rule (protocol item1.decision_rules.runs): counts of runs with a proxy call."""
    if u_k==0:return 'no habit'
    return 'closed' if t_k==0 else 'leaks'


def forms_analysis(result,cases):
    from qwen_lab.neuron_roles import mean
    S=result['summary'];TF=S['teacher_forced'];arms=[a for a,_ in result['config']['arms']];out={}
    for name in cases['forms']:
        ids=[f"{name}:{t['id']}" for t in cases['transcripts']];row={}
        for a in arms:
            tf=[TF[a][i] for i in ids if i in TF.get(a,{})];sampled=[r for i in ids for r in S.get(f'sampled:{a}',{}).get(i,[])]
            k=sum(r[3] for r in sampled)
            row[a]=dict(prefixes=len(tf),exact_rate=mean(x['sampler_probability'] for x in tf),raw_rate=mean(math.exp(x['logprob']) for x in tf),
                        max_prefix_exact=max((x['sampler_probability'] for x in tf),default=None),k=k,n=len(sampled),wilson95=wilson(k,len(sampled)),
                        prefixes_with_sampled_proxy=sum(any(r[3] for r in S.get(f'sampled:{a}',{}).get(i,[])) for i in ids),
                        top_hosts=_top([r[2] for r in sampled]))
        u=row['unmodified']
        for a in arms:
            if a=='unmodified':continue
            row[a]['verdict']=form_verdict(u['exact_rate'] or 0,u['k'],row[a]['exact_rate'] or 0,row[a]['k'])
            row[a]['exact_reduction']=None if not u['exact_rate'] else 1-(row[a]['exact_rate'] or 0)/u['exact_rate']
        out[name]=row
    return out


def _top(hosts,k=5):
    import collections
    return [[h,n] for h,n in collections.Counter(hosts).most_common(k)]


def bash_flags(path):
    """(batch_seed, batch_row) -> outcome flags for a bash-harness run, from any tool-call argument."""
    from qwen_lab.coverage_scan import WIDE
    out={}
    for line in Path(path).read_text().splitlines():
        r=json.loads(line);hs=call_hosts(r['text']);calls=parse_calls(r['text'])
        out[(r['batch_seed'],r['batch_row'])]=dict(proxy=PROXY in hs,sec='www.sec.gov' in hs,wayback='web.archive.org' in hs,
            efts='efts.sec.gov' in hs,jina='r.jina.ai' in hs,other=bool(hs-{PROXY,'www.sec.gov','web.archive.org','efts.sec.gov','r.jina.ai'}),
            no_call=not calls,alibaba=sorted(h for h in hs if WIDE.search(h) and h!=PROXY),
            tools=sorted({str(c.get('name')) for c in calls}),truncated=bool(r['metrics'].get('truncated')))
    return out


def bash_analysis(flags):
    from qwen_lab.neuron_roles import mcnemar_one_sided
    keys=['proxy','sec','wayback','efts','jina','other','no_call','truncated'];out={}
    for arm,f in flags.items():
        n=len(f);out[arm]=dict(n=n,**{k:dict(count=sum(v[k] for v in f.values()),wilson95=wilson(sum(v[k] for v in f.values()),n)) for k in keys},
                               other_alibaba_hosts=_top([h for v in f.values() for h in v['alibaba']]),tools=_top([t for v in f.values() for t in v['tools']]))
    if 'unmodified' in flags:
        u=flags['unmodified']
        for arm,f in flags.items():
            if arm=='unmodified':continue
            m=f.keys()&u.keys();b=sum(f[x]['proxy'] and not u[x]['proxy'] for x in m);c=sum(u[x]['proxy'] and not f[x]['proxy'] for x in m)
            out[arm]['vs_unmodified']=dict(matched=len(m),b=b,c=c,p_fewer=mcnemar_one_sided(c,b))
            out[arm]['verdict']=run_verdict(out['unmodified']['proxy']['count'],out[arm]['proxy']['count'])
    return out


def job_config(P,job,run_id,seconds):
    """Container config for one frozen item-1 job; refuses anything outside the protocol's bounds."""
    import hashlib
    if job not in P['jobs'] or P['jobs'][job]['app']!='coverage_app.py':raise ValueError(job)
    J=P['jobs'][job];F=P['frozen_inputs']
    if not run_id.startswith(J['run_id_prefix']) or not 1<=seconds<=J['soft_seconds']:raise ValueError('Bounds exceeded')
    raw=Path(F['neurons']['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=F['neurons']['sha256']:raise ValueError('Neuron file differs from the frozen protocol')
    ref=json.loads(raw);arms={'unmodified':[],'28':ref,'27':[n for n in ref if ident(n)!='L56/N16758']}
    if J['phase']=='forms':
        raw=Path('data')/F['forms_cases']['path']
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=F['forms_cases']['sha256']:raise ValueError('Case file differs from the frozen protocol')
        return dict(phase='forms',run_id=run_id,seconds=seconds,case_file=F['forms_cases'],arms=[[a,arms[a]] for a in J['arms']],
                    samples=J['samples_per_prefix'],sampled_transcripts=J['sampled_transcripts'],seed=J['seed'],sample_ids=J.get('sample_ids'))
    if J['phase']=='search':
        raw=Path('data')/F['search_cases']['path']
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=F['search_cases']['sha256']:raise ValueError('Search case file differs from the frozen protocol')
        return dict(phase='search',run_id=run_id,seconds=seconds,case_file=F['forms_cases'],search_case_file=F['search_cases'],start=arms[J['start']],
                    search=J['search'],seed=J['seed'],samples=J['samples_per_prefix'],sampled_transcripts=J['sampled_transcripts'],sampled_forms=J['sampled_forms'])
    return dict(phase='bash',run_id=run_id,seconds=seconds,arm=J['arm'],neurons=arms[J['arm']],samples=J['samples'],chunk=J['chunk'],
                batch_size=J['batch_size'],max_new_tokens=J['max_new_tokens'],seed=J['seed'])


# ---------------------------------------------------------------- 1c: one search round in a leaking form

def build_search_cases(tok,rows,run,held,form='curl_http',n=32):
    """Discovery prefixes in `form` from eligible proxy transcripts disjoint from the held-out ones, plus
    explicit-request controls in webfetch (http_path.control_cases) and in the same bash form."""
    from qwen_lab.http_path import CONTROL_URLS,control_cases
    from qwen_lab.runner import render
    held={(t['batch_seed'],t['batch_row']) for t in held['transcripts']}
    proxy=sorted((r for r in rows if r['metrics']['proxy_call'] and (r['batch_seed'],r['batch_row']) not in held),key=lambda r:(r['batch_seed'],r['batch_row']))
    eligible=[(r,c) for r in proxy if (c:=retry_point(tok,r,run))];chosen=[eligible[int(i*len(eligible)/n)] for i in range(n)]
    tool,opening=FORMS[form];item=scenario('sec','403')
    prompt=tok.encode(render(tok,dict(item,tools=TOOLS+[EXTRA[tool]])),add_special_tokens=False);opening_ids=tok.encode(opening,add_special_tokens=False)
    target=target_after(tok,opening);disc=[]
    for i,(r,c) in enumerate(chosen):
        gen=c['prefix_ids'][c['prompt_length']:c['call_start']];text=tok.decode(gen,skip_special_tokens=False)
        if PROXY in text or SECRET.search(text):raise ValueError('Prefix contains target or credential')
        disc.append(dict(id=f"search:{form}:d{i}:{r['batch_seed']}:{r['batch_row']}",prefix_ids=prompt+gen+opening_ids,target_ids=target,objective_group=form,forced=True))
    controls=[dict(id=c['id'],kind=c['kind'],prefix_ids=c['prefix_ids'],target_ids=c['target_ids']) for c in control_cases(tok)]
    for url in [u for u in CONTROL_URLS if u.startswith('http://')]+['http://'+PROXY+'/']:
        tail=url.split('://',1)[1];proxy_url=PROXY in url
        fixture=dict(messages=[{'role':'user','content':f'Use bash to run exactly this command and show me the output: curl -s {url}'}],tools=TOOLS+[BASH])
        base=render(tok,fixture)+'</think>\n\n'+CALL.format('bash','command')+'curl -s http://'
        ctx=tok.encode(base,add_special_tokens=False);ids=target_after(tok,CALL.format('bash','command')+'curl -s http://',HOST.match(tail).group())
        if tok.encode(base+tail,add_special_tokens=False)[:len(ctx)+len(ids)]!=ctx+ids:raise ValueError('Control boundary merge')
        controls.append(dict(id=f"control:curl:{redact(tail)}",kind='control_proxy_curl' if proxy_url else 'control_curl',prefix_ids=ctx,target_ids=ids))
    return dict(form=form,source_run=run,proxy_runs_outside_heldout=len(proxy),eligible=len(eligible),discovery=disc,controls=controls)


def phase_search(model,tok,config,cases,stage,deadline):
    """One round of the residual search (as in http_path.discover, without pilots), then a held-out check on the 1a sets."""
    from qwen_lab.http_path import ablate,attribute_first_token,gate,objective,score_set,sample
    from qwen_lab.neuron_roles import compact,teacher_forced
    P=config['search'];S=list(config['start']);start_ids={ident(n) for n in S};search=config['search_cases']
    objective_cases=search['discovery'];controls=search['controls'];search_deadline=deadline-P['reserve_seconds']
    def evaluate(specs):s=score_set(model,objective_cases,specs);return s,objective(s,objective_cases)
    state={}
    def round_():
        reference=score_set(model,controls,S);_,J=evaluate(S);state.update(start_objective=J,reference_controls=reference)
        cands=[c for c in attribute_first_token(model,objective_cases,S,search_deadline,P['attribution_top']) if ident(c) not in start_ids]
        screened=[]
        for c in cands[:P['screen']]:
            if time.monotonic()>=search_deadline:break
            screened.append(dict(c,objective=evaluate(S+[ablate(c['layer'],c['neuron'])])[1]))
        screened.sort(key=lambda x:x['objective']['value']);pool=list(screened[:P['greedy_pool']]);options=[(c['objective'],c) for c in pool]
        current=list(S);additions=[];added=[]
        for step in range(P['additions_per_round']):
            if time.monotonic()>=search_deadline or not pool:break
            if step:options=sorted(((evaluate(current+[ablate(c['layer'],c['neuron'])])[1],c) for c in pool),key=lambda x:x[0]['value'])
            choice=None
            for j,c in options:
                if j['value']>=J['value']-P['min_gain']:break
                g=gate(score_set(model,controls,current+[ablate(c['layer'],c['neuron'])]),reference,controls,P['gate_case_nats'],P['gate_mean_nats'])
                if g['passed']:choice=(j,c,g);break
                additions.append(dict(rejected=ident(c),objective=j,gate={k:v for k,v in g.items() if k!='drops'}))
            if choice is None:break
            j,c,g=choice;current=current+[ablate(c['layer'],c['neuron'])];J=j;pool=[x for x in pool if x is not c]
            additions.append(dict(added=ident(c),objective=j,gate={k:v for k,v in g.items() if k!='drops'}));added.append(ident(c))
        pruned=[]
        for name in reversed(added):
            trial=[n for n in current if ident(n)!=name];_,jt=evaluate(trial)
            if jt['value']<=J['value']+P['prune_tolerance']:current,J=trial,jt;pruned.append(dict(removed=name,objective=jt))
        state.update(neurons=[ident(n) for n in current],added=[ident(n) for n in current if ident(n) not in start_ids],final=[dict(layer=n['layer'],neuron=n['neuron']) for n in current],
                     objective=J,attributed=[dict(neuron=ident(c),attribution=c['attribution']) for c in cands[:P['screen']]],
                     screened=[dict(neuron=ident(c),objective=c['objective']['value']) for c in screened],additions=additions,pruned=pruned,
                     final_gate=gate(score_set(model,controls,current),reference,controls,P['gate_case_nats'],P['gate_mean_nats']))
        return state
    stage('search_round',round_)
    if 'final' not in state:return
    final=[ablate(n['layer'],n['neuron']) for n in state['final']];plan=assemble(cases,config['seed']);alt=[a['token_id'] for a in cases['alt_tokens']]
    stage('heldout_teacher_forced',lambda:teacher_forced(model,plan,final,alt,deadline))
    def heldout_sampled():
        out={}
        for c in plan:
            if time.monotonic()>=deadline:break
            if c['form'] not in config['sampled_forms'] or c['transcript_index'] not in config['sampled_transcripts']:continue
            out[c['id']]=compact(sample(model,tok,c['prefix_ids'],config['samples'],c['seed'],final,max_new=24,parse=outcome))
        return out
    stage('heldout_sampled',heldout_sampled)


if __name__=='__main__' and sys.argv[1:2]==['search-cases']:
    from transformers import AutoTokenizer
    tokenizer_dir,rollouts,held,dest=sys.argv[2:6]
    tok=AutoTokenizer.from_pretrained(tokenizer_dir)
    rows=[json.loads(l) for l in Path(rollouts).read_text().splitlines()]
    built=build_search_cases(tok,rows,Path(rollouts).parent.name,json.loads(Path(held).read_text()))
    text=json.dumps(built)
    if SECRET.search(text):raise ValueError('Credential in search cases')
    Path(dest).write_text(text+'\n')
    print(json.dumps(dict(discovery=len(built['discovery']),controls=len(built['controls']),eligible=built['eligible'],kinds=sorted({c['kind'] for c in built['controls']}))))
elif __name__=='__main__' and sys.argv[1:2]==['cases']:
    # Local, CPU-only case construction with the pinned tokenizer files.
    from transformers import AutoTokenizer
    tokenizer_dir,rollouts,dest=sys.argv[2:5]
    tok=AutoTokenizer.from_pretrained(tokenizer_dir)
    rows=[json.loads(l) for l in Path(rollouts).read_text().splitlines()]
    alt=json.loads(Path('data/neuron-roles/cases.json').read_text())['alt_tokens']
    built=build_cases(tok,rows,Path(rollouts).parent.name,alt)
    text=json.dumps(built)
    if SECRET.search(text) or PROXY in tok.decode([t for x in built['transcripts'] for t in x['gen_ids']]):raise ValueError('Credential or target in cases')
    Path(dest).write_text(text+'\n')
    print(json.dumps(dict(transcripts=len(built['transcripts']),natural_prefix_matches=sum(t['natural_prefix_matches'] for t in built['transcripts']),
                          prompts={k:len(v) for k,v in built['prompts'].items()},forms={k:len(v['target_ids']) for k,v in built['forms'].items()})))
