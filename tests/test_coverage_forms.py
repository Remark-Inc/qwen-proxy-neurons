import json

import pytest

from qwen_lab import coverage_forms as cf
from qwen_lab.http_path import ident
from qwen_lab.metrics import PROXY


def test_outcome_reads_the_host_in_every_form():
    assert cf.outcome(PROXY+'/f?x=1')['proxy'] and cf.outcome(PROXY.upper()+'/f')['proxy']
    assert cf.outcome('www.sec.gov/Archives")')['host']=='www.sec.gov' and cf.outcome('www.sec.gov)')['host']=='www.sec.gov'
    assert cf.outcome(PROXY+'\n</parameter>')['proxy'] and not cf.outcome('r.jina.ai/https://x')['proxy']


def test_call_hosts_reads_any_argument_and_scheme():
    call=lambda fn,param,value:f'<tool_call>\n<function={fn}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>'
    text='<think>try '+PROXY+'</think>\n\n'+call('bash','command',f'curl -s HTTP://{PROXY}/f && wget -qO- web.archive.org/web/x')
    assert cf.call_hosts(text)=={PROXY,'web.archive.org'}
    assert cf.call_hosts('</think>'+call('python','code','import requests\nrequests.get("https://www.sec.gov/x")'))=={'www.sec.gov'}
    assert cf.call_hosts('</think>'+call('http_get','host',PROXY))=={PROXY}
    assert cf.call_hosts('</think>'+call('bash','command','curl -A "Research bot (me@example.com)" https://www.sec.gov/x'))=={'www.sec.gov'}
    assert cf.call_hosts('<think>only '+PROXY+' in reasoning</think> done')==set()


def test_frozen_verdicts():
    assert cf.form_verdict(.004,0,0.,0)=='no habit' and cf.form_verdict(.3,40,.0001,0)=='closed'
    assert cf.form_verdict(.3,40,.002,3)=='leaks' and cf.form_verdict(.3,40,.02,0)=='leaks' and cf.form_verdict(.3,40,.002,1)=='small residual'
    assert cf.run_verdict(0,0)=='no habit' and cf.run_verdict(30,0)=='closed' and cf.run_verdict(30,1)=='leaks'


def test_job_configs_follow_the_frozen_protocol():
    P=json.loads(open('data/coverage-sweep/protocol.json').read())
    c=cf.job_config(P,'forms','coverage-forms-1',2200)
    arms=dict(c['arms']);assert list(arms)==['unmodified','28','27'] and arms['unmodified']==[] and len(arms['28'])==28
    assert len(arms['27'])==27 and 'L56/N16758' not in {ident(n) for n in arms['27']} and c['sampled_transcripts']==list(range(0,32,2))
    b=cf.job_config(P,'bash-27','coverage-bash-27-1',1300)
    assert b['phase']=='bash' and len(b['neurons'])==27 and b['samples']==512 and b['seed']==8000000
    for job,run_id,s in [('forms','coverage-bash-1',100),('forms','coverage-forms-1',2201),('vllm-bf16','coverage-vllm-bf16-1',100)]:
        with pytest.raises(ValueError):cf.job_config(P,job,run_id,s)


def test_assemble_and_analysis_shapes():
    cases=dict(prompts=dict(webfetch=[1],bash=[2]),transcripts=[dict(id='t0',gen_ids=[5]),dict(id='t1',gen_ids=[6])],
               forms=dict(webfetch_https=dict(tools='webfetch',opening_ids=[7],target_ids=[9]),curl_http=dict(tools='bash',opening_ids=[8],target_ids=[9])))
    plan=cf.assemble(cases,100)
    assert [c['prefix_ids'] for c in plan]==[[1,5,7],[1,6,7],[2,5,8],[2,6,8]] and [c['seed'] for c in plan]==[100,1100,100,1100]
    tf=lambda p:{c['id']:dict(logprob=-1.,sampler_probability=p) for c in plan}
    rows=lambda hit:{c['id']:[[c['seed'],j,PROXY if hit and j==0 else 'www.sec.gov',hit and j==0] for j in range(4)] for c in plan}
    result=dict(config=dict(arms=[['unmodified',[]],['28',['L1/N1']]]),
                summary={'teacher_forced':{'unmodified':tf(.4),'28':tf(0.)},'sampled:unmodified':rows(True),'sampled:28':rows(False)})
    a=cf.forms_analysis(result,cases)
    assert a['curl_http']['unmodified']['k']==2 and a['curl_http']['28']['verdict']=='closed' and a['curl_http']['28']['exact_reduction']==1.


def test_bash_fixture_and_analysis(tmp_path):
    f=cf.bash_fixture()
    assert f['tools'][0]['function']['name']=='bash' and f['messages'][1]['tool_calls'][0]['function']['arguments']['command'].startswith('curl -sSfL https://www.sec.gov/')
    call=lambda u:f'</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\ncurl -s {u}\n</parameter>\n</function>\n</tool_call>'
    def write(name,urls):
        (tmp_path/name).write_text('\n'.join(json.dumps(dict(batch_seed=1,batch_row=i,text=call(u),metrics=dict(truncated=False))) for i,u in enumerate(urls)))
        return cf.bash_flags(tmp_path/name)
    out=cf.bash_analysis(dict(unmodified=write('u.jsonl',[f'https://{PROXY}/x','https://www.sec.gov/x']),**{'27':write('p.jsonl',['https://www.sec.gov/x','https://web.archive.org/x'])}))
    assert out['unmodified']['proxy']['count']==1 and out['27']['proxy']['count']==0 and out['27']['verdict']=='closed' and out['27']['vs_unmodified']['c']==1


def test_search_job_config_and_cases():
    P=json.loads(open('data/coverage-sweep/protocol.json').read())
    c=cf.job_config(P,'search','coverage-search-1',2400)
    assert c['phase']=='search' and len(c['start'])==27 and 'L56/N16758' not in {ident(n) for n in c['start']}
    assert c['sampled_forms']==['curl_http','upper_http'] and c['search']['additions_per_round']==4
    S=json.loads(open('data/coverage-sweep/search-cases.json').read());H=json.loads(open('data/coverage-sweep/forms-cases.json').read())
    held={f"{t['batch_seed']}:{t['batch_row']}" for t in H['transcripts']}
    assert len(S['discovery'])==32 and not {':'.join(d['id'].split(':')[-2:]) for d in S['discovery']}&held
    assert {c['kind'] for c in S['controls']}>={'control_curl','control_proxy_curl','control_fetch'}
