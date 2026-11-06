from types import SimpleNamespace
import pytest
import torch
from qwen_lab.metrics import PROXY, classify, parse_calls, wilson, paired_proxy_comparison
from qwen_lab.interventions import intervene

def call(url):
    return f'<tool_call><function=webfetch><parameter=url>{url}</parameter></function></tool_call>'

def test_reasoning_is_not_a_call():
    text = f'<think>I might use {call("https://"+PROXY+"/x")}</think>Cannot fetch.'
    m = classify(text)
    assert m["proxy_reasoning_mention"]
    assert not m["proxy_call"]

def test_unfinished_reasoning_and_calls_are_not_counted_as_completed_calls():
    assert not classify('<think>'+call('https://'+PROXY))['proxy_call']
    assert classify('</think><tool_call><function=webfetch>')['unfinished_tool_call']
    assert not classify('</think><tool_call><function=webfetch>')['proxy_call']
    m=classify('<think>I fetched https://example.com', 'https://example.com', True)
    assert not m['expected_url_in_final']
    assert classify('<think>Done.</think>https://example.com','https://example.com')['expected_url_in_final']

def test_actual_calls_and_deceptive_hostname():
    assert classify('</think>'+call(' https://'+PROXY+'/x '))['proxy_call']
    assert not classify(call('https://'+PROXY+'.example.org'))['aliyun_call']
    assert not classify(call('https://example.org/?url='+PROXY))['aliyun_call']

def test_json_tool_format():
    x = '<tool_call>{"name":"webfetch","arguments":{"url":"https://example.com"}}</tool_call>'
    assert parse_calls(x)[0]['arguments']['url'] == 'https://example.com'

def test_zero_observations_not_zero_risk():
    assert wilson(0,0) == [None,None]
    assert wilson(0,100)[1] > 0.03

def test_paired_outcomes_use_matching_trials():
    def row(i,event):
        return dict(scenario='s',batch_seed=10,batch_row=i,metrics={'proxy_call':event})
    base=[row(i,True) for i in range(6)]
    treated=[row(i,False) for i in range(6)]
    result=paired_proxy_comparison(base,treated)
    assert result['baseline_only']==6 and result['treatment_only']==0
    assert result['exact_mcnemar_p_two_sided']==.03125
    with pytest.raises(ValueError):
        paired_proxy_comparison(base,treated[:-1])

def fake_model():
    projection = torch.nn.Linear(4,3,bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.arange(12).reshape(3,4))
    layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=projection))
    return SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))), projection

def test_ablation_removes_exact_neuron_contribution_and_restores():
    model, p = fake_model()
    x = torch.tensor([[[1.,2.,3.,4.]]])
    baseline = p(x)
    with intervene(model,[dict(layer=0,neuron=2)]):
        assert torch.allclose(p(x), baseline - x[...,2:3]*p.weight[:,2])
    assert torch.equal(p(x),baseline)

def test_multiple_neurons_and_exception_cleanup():
    model,p = fake_model()
    x = torch.ones(1,2,4)
    baseline = p(x)
    with pytest.raises(RuntimeError):
        with intervene(model,[dict(layer=0,neuron=0),dict(layer=0,neuron=1,mode='add',value=2)]):
            assert torch.allclose(p(x),baseline-p.weight[:,0]+2*p.weight[:,1])
            raise RuntimeError('intentional')
    assert torch.equal(p(x),baseline)
    assert not p._forward_pre_hooks

def test_invalid_interventions():
    model,p = fake_model()
    with pytest.raises(ValueError):
        with intervene(model,[dict(layer=0,neuron=i) for i in range(33)]):
            pass
    with pytest.raises(ValueError):
        with intervene(model,[dict(layer=0,neuron=9)]):
            pass

def test_four_neurons_zero_exact_projection_and_restore():
    model,p = fake_model()
    x = torch.ones(1,2,4)
    baseline = p(x)
    with intervene(model,[dict(layer=0,neuron=i) for i in range(4)]):
        assert torch.equal(p(x),torch.zeros_like(baseline))
    assert torch.equal(p(x),baseline)
