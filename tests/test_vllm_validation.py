from types import SimpleNamespace as NS
import math
import pytest
import torch
from qwen_lab.vllm_intervention import column_match,set_arm,digest,verify_owners
from qwen_lab.vllm_validation import token_logprobs,prompt_score


def test_original_column_found_in_one_padded_shard_and_reversibly_removed():
    torch.manual_seed(239);full=torch.randn(3,16,20,dtype=torch.bfloat16)
    expected=full[1,:,13].clone();matches=[]
    for rank in range(4):
        shard=torch.zeros(3,16,8,dtype=torch.bfloat16);shard[:,:,:5]=full[:,:,rank*5:(rank+1)*5]
        col=column_match(shard,1,expected);matches.append(col)
        if col is not None:
            spec={'layer':37,'expert':1,'neuron':13};target=shard[1,:,col]
            model=NS(_neuron_lab_columns=[dict(spec=spec,target=target,original=target.clone())])
            before=shard.clone();x=torch.randn(8)
            set_arm(model,[spec])
            delta=before[1].float()@x-shard[1].float()@x
            assert torch.allclose(delta,expected.float()*x[col],atol=1e-5)
            set_arm(model,[]);assert torch.equal(shard,before)
    assert matches==[None,None,3,None]


def test_ambiguous_or_quantized_layout_is_rejected():
    with pytest.raises(ValueError):column_match(torch.ones(2,4,3,dtype=torch.bfloat16),0,torch.ones(4))
    with pytest.raises(ValueError):column_match(torch.ones(2,4,3,dtype=torch.int8),0,torch.ones(4))


def test_replay_probability_is_original_not_renormalized_forced_probability():
    out=NS(outputs=[NS(token_ids=[4,7],finish_reason='length',logprobs=[{4:NS(logprob=math.log(.2))},{7:NS(logprob=math.log(.3))}])])
    result=token_logprobs(out,[4,7]);assert math.isclose(result['probability'],.06)
    with pytest.raises(ValueError):token_logprobs(out,[4,8])
    p=NS(prompt_logprobs=[None,None,{4:NS(logprob=math.log(.2))},{7:NS(logprob=math.log(.3))}])
    assert prompt_score(p,[1,2],[4,7])==result


def test_all_worker_owners_must_be_present_and_unique():
    spec={'layer':37,'expert':36,'neuron':149}
    rows=[dict(rank=r,world_size=4,columns=[dict(spec=spec,local_column=149 if r==0 else None,checkpoint_column_sha256='same')]) for r in range(4)]
    verify_owners(rows,[spec])
    with pytest.raises(ValueError):verify_owners(rows[:-1],[spec])
    rows[1]['columns'][0]['local_column']=1
    with pytest.raises(ValueError):verify_owners(rows,[spec])
