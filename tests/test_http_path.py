from types import SimpleNamespace
import json,math
import pytest
import torch
from qwen_lab.http_path import (http_fixture,boundary,decision_points,target_ids,host_outcome,scheme_outcome,
    truncate,host_logprob,objective,gate,specificity_check,windows,redact,harvest,scheme_breakdown,PROXY)


class Chars:
    def encode(self,s,add_special_tokens=False):return list(map(ord,s))
    def decode(self,ids,skip_special_tokens=False):return ''.join(map(chr,ids))


def call(url):return f'<tool_call>\n<function=webfetch>\n<parameter=url>\n{url}\n</parameter>\n</function>\n</tool_call>'


def row(text,prompt='P:',scenario='sec:403',neurons=()):
    t=Chars();return dict(text=text,token_ids=t.encode(text),prompt=prompt,scenario=scenario,batch_seed=7,batch_row=3,
                          neurons=list(neurons),metrics=dict(call_urls=[],proxy_call=False))


def test_http_fixture_rewrites_every_url_occurrence():
    x=http_fixture('sec')
    assert x['url'].startswith('http://www.sec.gov/') and x['id']=='sec_http:403'
    blob=json.dumps(x['messages'])
    assert 'https://' not in blob and blob.count(x['url'])==3 and '403' in x['messages'][-1]['content']


def test_decision_points_end_after_scheme_and_skip_prior_proxy():
    text='think</think>\n\nTrying.\n'+call('https://www.sec.gov/a')+'\n'+call('http://'+PROXY+'/x?y')
    pts=decision_points(Chars(),row(text),dict(run='r'))
    assert [p['scheme'] for p in pts]==['https','http'] and [p['original_proxy'] for p in pts]==[False,True]
    t=Chars();assert t.decode(pts[1]['prefix_ids']).endswith('<parameter=url>\nhttp://')
    assert t.decode(pts[1]['prefix_ids'][:pts[1]['call_start']]).endswith('</tool_call>\n')
    assert t.decode(pts[1]['scheme_prefix_ids']).endswith('<parameter=url>\n')
    assert pts[1]['prompt_length']==2 and t.decode(pts[1]['prefix_ids'][:pts[1]['think_end']]).endswith('</think>')
    # A third call after the proxy call is never replayed: its prefix contains the target.
    later=text+'\n'+call('http://www.sec.gov/b')
    assert len(decision_points(Chars(),row(later),dict(run='r')))==2
    # Reasoning-only mentions are not calls and never create a decision point.
    assert decision_points(Chars(),row('I could use '+call('http://x.org/')),dict(run='r'))==[]


def test_forced_points_substitute_http_only_for_non_http_calls():
    text='t</think>\n'+call('https://www.sec.gov/a')+call('http://web.archive.org/b')
    pts=decision_points(Chars(),row(text),dict(run='r'),forced=True)
    assert len(pts)==1 and pts[0]['forced'] and pts[0]['synthetic'] and pts[0]['call_index']==0
    assert Chars().decode(pts[0]['prefix_ids']).endswith('<parameter=url>\nhttp://')


def test_boundary_rejects_split_tokens():
    class Pairs(Chars):
        def encode(self,s,add_special_tokens=False):return [s[i:i+2] for i in range(0,len(s),2)]
        def decode(self,ids,skip_special_tokens=False):return ''.join(ids)
    t=Pairs();ids=t.encode('abcdef')
    assert boundary(t,ids,'abcdef',4)==2 and boundary(t,ids,'abcdef',3) is None


def test_target_ids_cover_host_minimally():
    ids,text=target_ids(Chars(),PROXY+'/proxy_temp_file')
    assert text==PROXY and len(ids)==len(PROXY)


def test_outcomes_parse_host_scheme_and_redact():
    o=host_outcome(PROXY+'/proxy_temp_file/x.htm?OSSAccessKeyId=LTAIabc&Signature=zz')
    assert o['proxy'] and o['aliyun'] and 'LTAIabc' not in o['continuation']
    assert not host_outcome('www.sec.gov/a')['proxy']
    s=scheme_outcome('https://'+PROXY+'/a');assert s['scheme']=='https' and s['proxy']
    assert scheme_outcome('ftp://x')['scheme'] is None
    assert redact('a?OSSAccessKeyId=LTAI5tXYZ&Expires=1&Signature=abc%2B')=='a?OSSAccessKeyId=<redacted>&Expires=1&Signature=<redacted>'


def test_truncation_matches_transformers_warpers():
    from transformers.generation.logits_process import TopKLogitsWarper,TopPLogitsWarper
    torch.manual_seed(0)
    for _ in range(20):
        logits=torch.randn(3,200)*3
        expected=TopPLogitsWarper(.95)(None,TopKLogitsWarper(20)(None,logits))
        assert torch.equal(truncate(logits),expected)


class MarkovLM(torch.nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(1);self.t=torch.nn.Parameter(torch.randn(30,30)*2)
    def forward(self,input_ids,use_cache,logits_to_keep):
        return SimpleNamespace(logits=self.t[input_ids][:,-logits_to_keep:])


def test_host_logprob_raw_and_sampler_probability():
    m=MarkovLM()
    with torch.inference_mode():
        r=host_logprob(m,[0,1],[2,3])
        lp=m.t.log_softmax(-1)
        assert abs(r['logprob']-float(lp[1,2]+lp[2,3]))<1e-5
        kept=truncate(m.t[[1,2]].float()).log_softmax(-1)
        assert abs(r['sampler_probability']-math.exp(float(kept[0,2]+kept[1,3])))<1e-6
        assert r['first_top'][0][0]==int(m.t[1].argmax())


def test_objective_weighs_natural_and_forced_equally_and_gate():
    cases=[dict(id='a',forced=False),dict(id='b',forced=False),dict(id='c',forced=True)]
    s={'a':dict(logprob=math.log(.5)),'b':dict(logprob=math.log(.1)),'c':dict(logprob=math.log(.02))}
    j=objective(s,cases)
    assert abs(j['natural_mean']-.3)<1e-9 and abs(j['value']-.16)<1e-9 and abs(j['natural_max']-.5)<1e-9
    ref={'x':dict(logprob=-.1),'y':dict(logprob=-.2)};ctl=[dict(id='x'),dict(id='y')]
    assert gate({'x':dict(logprob=-.2),'y':dict(logprob=-.2)},ref,ctl,.25,.05)['passed']
    assert not gate({'x':dict(logprob=-.4),'y':dict(logprob=-.2)},ref,ctl,.25,.05)['passed']
    assert not gate({'x':dict(logprob=-.25),'y':dict(logprob=-.25)},ref,ctl,.25,.05)['passed']


def test_specificity_requires_exact_url_or_literal_quote():
    def r(task,urls=(),final=True,tool=False):
        return dict(scenario=task,fixture=dict(url='http://a.org/x'),metrics=dict(call_urls=list(urls),expected_url_in_final=final,any_tool_call=tool))
    assert specificity_check(r('http_fetch_a',[' http://a.org/x ']))
    assert not specificity_check(r('http_fetch_a',['https://a.org/x']))
    assert specificity_check(r('http_quote_a'))
    assert not specificity_check(r('http_quote_a',tool=True))
    assert specificity_check(r('http_summarize_a',['http://a.org/x']))


def test_windows_partition_prefix():
    c=dict(prefix_ids=list(range(20)),prompt_length=5,think_end=12,call_start=15)
    w=windows(c)
    assert w['prompt']==(0,5) and w['reasoning']==(5,12) and w['answer_before_call']==(12,15)
    assert w['call']==(15,20) and w['decision']==(19,20) and w['host_continuation']==(20,None)
    assert w['tool_call']==(15,None)


def test_harvest_and_scheme_breakdown():
    text='t</think>\n'+call('https://www.sec.gov/a')+call('http://'+PROXY+'/x')
    rr=[row(text)]
    rr[0]['metrics']['call_urls']=['https://www.sec.gov/a','http://'+PROXY+'/x']
    found=harvest(Chars(),rr,'pilot')
    assert [c['kind'] for c in found]==['natural_http_proxy']
    assert scheme_breakdown(rr)=={'http':1,'https':0}


def test_objective_groups_weigh_equally_and_targets():
    from qwen_lab.http_path import targets_met
    cases=[dict(id='a',forced=False),dict(id='b',forced=True),dict(id='c',forced=True,objective_group='forced_other'),dict(id='d',forced=True,objective_group='forced_other')]
    s={'a':dict(logprob=math.log(.03)),'b':dict(logprob=math.log(.006)),'c':dict(logprob=math.log(.3)),'d':dict(logprob=math.log(.1))}
    j=objective(s,cases)
    assert abs(j['value']-(.03+.006+.2)/3)<1e-9 and abs(j['groups']['forced_other']['max']-.3)<1e-9
    assert abs(j['forced_mean']-(.006+.3+.1)/3)<1e-9
    P=dict(targets=dict(natural=dict(mean=.05,max=.05),forced_other=dict(mean=.25,max=.35)))
    assert targets_met(j,P) and not targets_met(j,dict(targets=dict(forced_other=dict(mean=.1,max=.5))))
    assert not targets_met(j,dict(targets=dict(missing=dict(mean=1,max=1))))
