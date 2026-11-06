from types import SimpleNamespace
import torch
from qwen_lab.interventions import intervene
from qwen_lab.prefixes import observed_cases
from qwen_lab.metrics import PROXY

class CharTokenizer:
    def encode(self,s,add_special_tokens=False):return list(map(ord,s))
    def decode(self,ids,skip_special_tokens=False):return ''.join(map(chr,ids))

def test_replay_locates_real_call_not_prior_proxy_mention_and_preserves_treatment():
    t=CharTokenizer()
    text='</think> The host '+PROXY+' is mentioned here. <tool_call><function=webfetch><parameter=url>https://'+PROXY+'/x</parameter></function></tool_call>'
    neurons=[dict(layer=63,neuron=417,value=0)]
    row=dict(text=text,prompt='PROMPT',token_ids=t.encode(text),neurons=neurons,
             fixture={'url':'https://www.sec.gov/a'},metrics={'proxy_call':True},
             scenario='sec:403',batch_seed=100,batch_row=2)
    c=observed_cases(t,[row])[0]
    assert c['requested_prefix'].endswith('<parameter=url>https://')
    assert c['source_neurons']==neurons
    assert c['source_batch_seed']==100 and c['source_batch_row']==2
    assert c['exact_prefix_token_replay']
    assert c['proxy_already_in_prior_text']
    assert c['comparison_is_synthetic']
    bad={**row,'token_ids':[0]+row['token_ids'][1:]}
    assert not observed_cases(t,[bad])

def test_expanded_intervention_removes_only_the_selected_coordinates():
    p=torch.nn.Linear(16,3,bias=False)
    layer=SimpleNamespace(mlp=SimpleNamespace(down_proj=p))
    model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer])))
    x=torch.arange(16,dtype=torch.float32).reshape(1,1,16)
    original=p(x)
    specs=[dict(layer=0,neuron=i,value=0) for i in range(8)]
    expected=x.clone();expected[...,:8]=0
    with intervene(model,specs):assert torch.allclose(p(x),p.weight@expected[0,0])
    assert torch.equal(p(x),original)
    assert not p._forward_pre_hooks
