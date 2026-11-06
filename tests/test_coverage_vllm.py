import json

import pytest
import torch

from qwen_lab import coverage_vllm as cv


def test_find_down_and_column_handle_both_layouts():
    names=['model.language_model.layers.5.mlp.down_proj.weight','model.language_model.layers.50.mlp.down_proj.weight',
           'model.visual.blocks.50.mlp.down_proj.weight','model.language_model.layers.50.mlp.gate_up_proj.weight']
    assert cv.find_down(50,names)=='model.language_model.layers.50.mlp.down_proj.weight'
    with pytest.raises(ValueError):cv.find_down(7,names)
    w=torch.arange(cv.HIDDEN*cv.INTER,dtype=torch.float32).reshape(cv.HIDDEN,cv.INTER)
    assert torch.equal(cv.column(w,3),w[:,3]) and torch.equal(cv.column(w.t(),3),w[:,3])
    with pytest.raises(ValueError):cv.column(torch.zeros(3,4),0)
    q=torch.zeros(cv.HIDDEN,cv.INTER)
    assert cv.same_layout(w,q) is q and cv.same_layout(w.t(),q).shape==(cv.INTER,cv.HIDDEN)


def test_as_bytes_zeroes_fp8_columns_in_place():
    w=torch.ones(cv.INTER,cv.HIDDEN).to(torch.float8_e4m3fn)
    cv.as_bytes(cv.column(w,7)).zero_()
    assert w[7].float().abs().sum()==0 and w[6].float().sum()==cv.HIDDEN


def test_prompt_score_reads_target_positions():
    class L:
        def __init__(self,v):self.logprob=v
    class O:
        prompt_logprobs=[None,{5:L(-.1)},{6:L(-.2)},{7:L(-.5)}]
    s=cv.prompt_score(O,2,[6,7])
    assert abs(s['logprob']+.7)<1e-12 and s['first']==-.2


def test_vllm_jobs_are_frozen_with_engine_specs():
    P=json.loads(open('data/coverage-sweep/protocol.json').read());J=P['jobs']
    assert [e['precision'] for e in J['vllm-smoke']['engines']]==['bf16','fp8'] and J['vllm-fp8']['engines'][0]['samples']==512
    modes={a['label']:a['mode'] for a in J['vllm-fp8']['engines'][0]['arms']}
    assert modes=={'unmodified':'after','28 after':'after','28 before':'before','27 after':'after'}
    cases=json.loads(open('data/'+P['frozen_inputs']['vllm_cases']['path']).read())
    assert len(cases['probes'])==6 and len(cases['prompt_ids'])==482


def test_clean_quant_config_drops_unknown_fields(tmp_path,monkeypatch):
    import sys,types
    # Stand-ins with pydantic's model_fields mapping (pydantic is not installed locally).
    class QuantizationArgs:model_fields={'num_bits':None,'group_size':None}
    class QuantizationScheme:model_fields={'targets':None,'weights':None,'input_activations':None}
    mod=types.ModuleType('compressed_tensors.quantization');mod.QuantizationArgs=QuantizationArgs;mod.QuantizationScheme=QuantizationScheme
    monkeypatch.setitem(sys.modules,'compressed_tensors',types.ModuleType('compressed_tensors'));monkeypatch.setitem(sys.modules,'compressed_tensors.quantization',mod)
    cfg=dict(quantization_config=dict(config_groups=dict(group_0=dict(targets=['Linear'],format='nvfp4-pack-quantized',
             weights=dict(num_bits=4,group_size=16,scale_dtype='torch.float8_e4m3fn'),input_activations=dict(num_bits=4)))))
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    removed=cv.clean_quant_config(tmp_path)
    assert removed==['config_groups.group_0.format','config_groups.group_0.weights.scale_dtype']
    new=json.loads((tmp_path/'config.json').read_text())['quantization_config']['config_groups']['group_0']
    assert 'format' not in new and new['weights']=={'num_bits':4,'group_size':16} and (tmp_path/'config.json.orig').exists()
    assert cv.clean_quant_config(tmp_path)==[]
