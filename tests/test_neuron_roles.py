from qwen_lab import neuron_roles as nr
from qwen_lab.http_path import OPEN


class CharTok:
    """One token per character, so boundaries and decodes are exact."""
    def encode(self,text,add_special_tokens=False):return [ord(ch) for ch in text]
    def decode(self,ids,skip_special_tokens=False):return ''.join(map(chr,ids))


def test_first_fetch_position_is_the_last_token_of_the_prompt_scheme():
    tok=CharTok();prompt='user asks\n'+OPEN+'https://www.sec.gov/x\n403 Forbidden\n';ids=tok.encode(prompt+'thinking')
    p=nr.first_call_position(tok,ids,len(prompt))
    assert tok.decode(ids[:p+1]).endswith(OPEN+'https://') and tok.decode(ids[p+1:p+4])=='www'
    assert nr.first_call_position(tok,tok.encode('no call here'),12) is None


def test_prose_positions_avoid_url_windows_and_are_reproducible():
    tok=CharTok();text='a'*100+' see www.sec.gov now '+'b'*100;ids=tok.encode(text)
    ps=nr.prose_positions(tok,ids,0,len(ids),'ctx',k=8,margin=10)
    assert len(ps)==8 and ps==nr.prose_positions(tok,ids,0,len(ids),'ctx',k=8,margin=10)
    url=range(text.index('www')-10,text.index('gov')+13)
    assert not any(p in url for p in ps)
    assert nr.prose_positions(tok,ids,None,len(ids),'ctx')==[]


def test_last_occurrence_finds_the_final_run():
    assert nr.last_occurrence([1,2,3,9,1,2,3,4],[1,2,3])==4 and nr.last_occurrence([1,2],[3])is None


def test_held_sets_follow_the_frozen_split_and_order():
    def case(i,kind,split,scen='sec:403'):return dict(id=f'{kind}:{i}',kind=kind,split=split,source=dict(scenario=scen))
    cases=[case(2,'natural_http_proxy','heldout'),case(1,'natural_http_proxy','heldout'),case(0,'natural_http_proxy','discovery'),
           case(3,'forced_http','heldout'),case(4,'forced_http','heldout','mozilla:403'),case(5,'forced_http','heldout_r3','pytorch:403')]
    sets=nr.held_sets(cases,'natural_http_proxy:0')
    assert [c['id'] for c in sets['E']]==['natural_http_proxy:0'] and [c['id'] for c in sets['N']]==['natural_http_proxy:1','natural_http_proxy:2']
    assert [c['id'] for c in sets['S']]==['forced_http:3'] and [c['id'] for c in sets['D']]==['forced_http:5']
    plan,seeds=nr.plan_and_seeds(dict(sets=sets),6000000)
    assert [seeds[c['id']] for c in plan]==[6000000,6001000,6002000,6003000,6004000]
    fam=nr.families(cases)
    assert len(fam['b_http_docs'])==2 and len(fam['b_http_sec'])==1 and len(fam['b_http_natural'])==3


def test_mcnemar_and_holm():
    assert nr.mcnemar_one_sided(0,0)==1. and nr.mcnemar_one_sided(5,0)==1/32
    assert abs(nr.mcnemar_one_sided(3,1)-5/16)<1e-12
    assert nr.holm([.01,.04,.03])==[.03,.06,.06]


def test_matched_units_pair_by_seed_and_row():
    a=[[1,0,'r',True],[1,1,'w',False],[2,0,'r',True]];z=[[1,0,'r',False],[1,1,'r',True],[3,0,'r',True]]
    assert nr.matched(a,z)==dict(b=1,c=1,n=2,k_a=1,k_b=1)


def test_passenger_rule():
    quiet=dict(p=.4,exact_rise=0.,raw_rise=.0001)
    assert nr.passenger(dict(E=quiet,N=quiet))==dict(label='inert',sets=[])
    assert nr.passenger(dict(E=quiet,N=dict(quiet,raw_rise=.002)))['label']=='redundant'
    assert nr.passenger(dict(E=quiet,D=dict(quiet,exact_rise=.0005)))==dict(label='needed',sets=['D'])
    assert nr.passenger(dict(S=dict(quiet,p=.01)))==dict(label='needed',sets=['S'])


def test_trigger_labels():
    assert nr.trigger_label(dict(a=-18.,b=-15.,c=-1.),sd_prose=.5)=='failure-specific'
    assert nr.trigger_label(dict(a=-.2,b=6.,c=.3),sd_prose=.5)=='http-specific'
    assert nr.trigger_label(dict(a=5.,b=4.,c=4.),sd_prose=.5)=='URL-general'
    assert nr.trigger_label(dict(a=-9.,b=-1.,c=0.),sd_prose=.5)=='failure-specific (https:// only)'
    assert nr.trigger_label(dict(a=.5,b=-.4,c=.1),sd_prose=.5)=='indeterminate'
    assert nr.trigger_label(dict(a=4.,b=-4.,c=0.),sd_prose=.5)=='indeterminate'


def test_direct_effects_agree_for_small_contributions_and_match_exact_removal():
    import torch
    torch.manual_seed(0);h=64;x=torch.randn(h)*5;g=1+torch.randn(h)*.1;U=torch.randn(3,h);col=torch.randn(h)*.05;eps=1e-6
    E=dict(U=U,g=g,cols={'L1/N0':col});u={'L1/N0':(U@(col*g)).tolist()}
    frozen,rms=nr.direct_effects({'L1/N0':.5},x,u,eps);exact=nr.exact_direct_effects({'L1/N0':.5},x,E,eps)
    assert abs(rms-float(torch.sqrt(x.pow(2).mean()+eps)))<1e-6
    assert max(abs(a-b) for a,b in zip(frozen['L1/N0'],exact['L1/N0']))<.02
    f=lambda v:U@(v*torch.rsqrt(v.pow(2).mean()+eps)*g)
    assert torch.allclose(torch.tensor(exact['L1/N0']),f(x)-f(x-.5*col),atol=1e-3)


def test_restore_analysis_pairs_each_arm_with_its_own_jobs_baseline():
    import math
    sets={s:[dict(id=f'{s}{i}') for i in range(2)] for s in nr.SETS};cases=dict(sets=sets)
    tf=lambda p:{c['id']:dict(logprob=math.log(max(p,1e-9)),sampler_probability=p) for v in sets.values() for c in v}
    rows=lambda hit:{c['id']:[[1000*i,j,'h',hit and j==0] for j in range(4)] for i,c in enumerate(c for v in sets.values() for c in v)}
    job=lambda label,p,hit:dict(summary={'teacher_forced':{'28':tf(0.),label:tf(p)},'sampled:28':rows(False),f'sampled:{label}':rows(hit)})
    out=nr.restore_analysis(dict(a=job('L1/N1',0.,False),b=job('L2/N2',.01,True)),cases)
    assert out['neurons']['L1/N1']['classification']==dict(label='inert',sets=[])
    hot=out['neurons']['L2/N2']
    assert hot['classification']['label']=='needed' and hot['sets']['D']['k']==2 and hot['sets']['D']['b']==2 and hot['sets']['D']['n']==8
    assert out['determinism']['identical_sampled_prefixes']==8 and out['determinism']['teacher_forced_max_abs_logprob_diff']==0


def test_profile_analyses_run_on_the_container_record_shapes(tmp_path):
    import json,math
    names=['L63/N417','L63/N14734','L63/N4455'];alts=[dict(label=l) for l in ('r','www')]
    sets={s:[dict(id=f'{s}0')] for s in nr.SETS};cases=dict(sets=sets,alt_tokens=alts,families=dict(a_https=[dict(id='h0',split='heldout')]))
    act=lambda v:{n:v for n in names}
    def ctx(i,fam,v,c1=None,key=None):
        return dict(id=i,family=fam,dec=act(v),after_first=act(v/2),c1=act(c1) if c1 is not None else None,c1_key=key,prose=[act(0.),act(.1)],
                    direct_dec={n:[v,-v] for n in names},exact_dec={n:[v,-v] for n in names})
    contexts=[ctx('t0','a_traces16',-18.),ctx('h0','a_https',-16.,1.,7),ctx('E0','b_http_natural',-15.,1.,7),ctx('S0','b_http_sec',-14.,1.,8),
              ctx('D0','b_http_docs',-13.),ctx('N0','b_http_natural',-12.),ctx('f0','c_fresh',-.5),ctx('r0','c_recall_text',-.2),ctx('w0','w_wayback',.3)]
    tf=lambda p:{c:dict(logprob=math.log(p),sampler_probability=p,alt_logprob=[math.log(p),-1.]) for c in ['E0','N0','S0','D0']}
    way=lambda p:{f'w{i}':dict(logprob=math.log(p),sampler_probability=p) for i in range(4)}
    B=dict(summary=dict(activations=dict(unembedding_rows={n:[0.,1.] for n in names},contexts=contexts),
                        singles={'unmodified':tf(.5),'L63/N4455':tf(.25)},
                        wayback_teacher_forced={'unmodified':way(.9),'L56/N16758':way(.45)},
                        wayback_sampled={'unmodified':{'w0':[[1,0,'web.archive.org',False],[1,1,'www.sec.gov',False]]},
                                         'L56/N16758':{'w0':[[1,0,'www.sec.gov',False],[1,1,'www.sec.gov',False]]}}))
    p=nr.profiles(B)
    assert p['counts']['c']==3 and p['unique_first_fetch_prompts']==2 and p['n4455_traces16']['mean']==-18.
    assert p['neurons']['L63/N4455']['label']=='failure-specific' and 'after_first_label' in p['neurons']['L63/N417']
    t=nr.total_effects(B,cases)['neurons']['L63/N4455']
    assert t['www']['D']['mean_delta_nats']==0. and abs(t['r']['D']['mean_delta_nats']-math.log(.5))<1e-12
    w=nr.wayback_analysis(B,reps=200)
    assert abs(w['teacher_forced']['L56/N16758']['mean_delta_logprob']-math.log(.5))<1e-12 and w['sampled']['unmodified']['k']==1 and w['sampled']['matched']['c']==1
    d=nr.direct_matrix(B,cases)
    assert d['sets']['https_heldout']['n']==1 and d['sets']['E']['exact']['L63/N4455']==[-15.,15.]
    s=nr.singles_analysis(B,cases)['L63/N4455']['D']
    assert s['exact_rate']==.25 and s['unmodified_exact_rate']==.5
    row=lambda seed,urls,proxy:json.dumps(dict(batch_seed=seed,batch_row=0,metrics=dict(call_urls=urls,proxy_call=proxy)))
    (tmp_path/'off.jsonl').write_text('\n'.join([row(1,['https://www.sec.gov/x'],False),row(2,['https://web.archive.org/web/x'],False)]))
    (tmp_path/'base.jsonl').write_text('\n'.join([row(1,['https://web.archive.org/web/x'],False),row(2,['https://web.archive.org/x'],True)]))
    o=nr.option_d(tmp_path/'off.jsonl',tmp_path/'base.jsonl')['outcomes']
    assert o['wayback']['n16758_off']==1 and o['wayback']['unmodified']==2 and o['wayback']['c']==1 and o['proxy']['unmodified']==1
