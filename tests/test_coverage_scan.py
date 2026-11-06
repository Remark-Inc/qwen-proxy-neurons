import json

from qwen_lab import coverage_scan as cs
from qwen_lab.metrics import PROXY


def test_wide_hosts_and_places():
    text=('<think>maybe sanmaohe.oss-cn-beijing.aliyuncs.com or help.aliyun.com</think>\n'
          f'<tool_call>\n<function=bash>\n<parameter=command>\ncurl -s https://{PROXY}/f?OSSAccessKeyId=x\n</parameter>\n</function>\n</tool_call>\n'
          'See dashscope.aliyuncs.com and www.sec.gov.')
    p=cs.places(text)
    assert p[PROXY]=={'call'} and p['sanmaohe.oss-cn-beijing.aliyuncs.com']=={'reasoning'} and p['help.aliyun.com']=={'reasoning'}
    assert p['dashscope.aliyuncs.com']=={'final_text'} and 'www.sec.gov' not in p
    assert cs.hosts('routify-file-proxy-sg.example.com/x')=={'routify-file-proxy-sg.example.com'} and not cs.hosts('aliyuncs')


def test_records_find_rollouts_and_samples_but_skip_inputs():
    named={frozenset(['L1/N1']):'one'}
    doc=dict(config=dict(text='x',metrics={}),summary={'sampled:28':{'case:a:b:c:1':[[1,0,PROXY,True]]},
             'rows':[dict(text='a',metrics={},neurons=[dict(layer=1,neuron=1)],intervention_scope='prompt')],
             'generations':[dict(arm='AC',rows=[dict(text='b',metrics={})])],'outcome':dict(host='web.archive.org',continuation='...')})
    got=[(k,a if k=='sample' else cs.arm_label(r,a,named)) for k,a,r in cs.records(doc)]
    assert ('sample','summary/sampled:28') in got and ('rollout','one (prompt positions only)') in got and ('rollout','AC') in got
    assert ('sample','summary/outcome') in got and len(got)==4


def test_scan_counts_runs_per_host(tmp_path):
    run=tmp_path/'r1';run.mkdir()
    rows=[dict(text=f'</think><tool_call>{PROXY}</tool_call>',metrics={},neurons=[]),dict(text='</think>nothing',metrics={},neurons=[])]
    (run/'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    out=cs.scan([tmp_path],{})
    r=out[('r1','rollout','unmodified')]
    assert r['records']==2 and r['hosts'][f'{PROXY}|call']==1
    assert cs.summarize(out)['other_hosts']==[]
