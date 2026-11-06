"""The patch script edits the right weights, and the repository carries no access-key value."""
import importlib.util,json,re,sys
from pathlib import Path
import torch
from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration,Qwen3_5ForCausalLM

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('apply_patch',ROOT/'patch'/'apply_patch.py')
apply_patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(apply_patch)

TEXT=dict(vocab_size=64,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=2,
          num_key_value_heads=1,head_dim=8,layer_types=['linear_attention','full_attention'],
          linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,linear_conv_kernel_dim=4)
VISION=dict(depth=1,hidden_size=16,intermediate_size=24,num_heads=2,out_hidden_size=16,num_position_embeddings=16)


def test_apply_zeroes_only_the_listed_columns_on_both_model_classes():
    cfg=Qwen3_5Config(text_config=TEXT,vision_config=VISION)
    for cls in (Qwen3_5ForConditionalGeneration,Qwen3_5ForCausalLM):
        torch.manual_seed(0);m=cls(cfg) if cls is Qwen3_5ForConditionalGeneration else cls(cfg.text_config)
        layers=apply_patch.language_layers(m);before=[l.mlp.down_proj.weight.clone() for l in layers]
        n=apply_patch.apply(m,[dict(layer=0,neuron=3),dict(layer=1,neuron=20)])
        assert n==2
        assert torch.all(layers[0].mlp.down_proj.weight[:,3]==0) and torch.all(layers[1].mlp.down_proj.weight[:,20]==0)
        keep=torch.ones(24,dtype=torch.bool);keep[3]=False
        assert torch.equal(layers[0].mlp.down_proj.weight[:,keep],before[0][:,keep])
        keep=torch.ones(24,dtype=torch.bool);keep[20]=False
        assert torch.equal(layers[1].mlp.down_proj.weight[:,keep],before[1][:,keep])


def test_neuron_list_matches_the_frozen_28_minus_the_wayback_neuron():
    ref=json.loads((ROOT/'data'/'http'/'candidate-refined.json').read_text())
    want=[(n['layer'],n['neuron']) for n in ref if (n['layer'],n['neuron'])!=(56,16758)]
    assert [(n['layer'],n['neuron']) for n in apply_patch.NEURONS['neurons']]==want and len(want)==27
    assert all(0<=l<64 and 0<=n<17408 for l,n in want)


def test_no_access_key_value_in_text():
    """Generated URLs carry OSSAccessKeyId=<redacted>; no key ID or signature value survives."""
    key=re.compile(r'LTAI[A-Za-z0-9]{8,}');sig=re.compile(r'Signature=(?!<redacted>)[A-Za-z0-9%+/=]{8,}')
    bad=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or p.suffix in ('.png','.lock') or p==Path(__file__):continue
        try:s=p.read_text()
        except UnicodeDecodeError:continue
        if key.search(s) or sig.search(s):bad.append(str(p.relative_to(ROOT)))
    assert not bad,bad


def test_no_access_key_value_in_token_ids():
    """Decoding every token-ID array must not give a key or signature value either."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer
        path=try_to_load_from_cache(apply_patch.NEURONS['model'],'tokenizer.json',revision=apply_patch.NEURONS['revision'])
        if not isinstance(path,str):return  # tokenizer not cached locally; the text test above still runs
        tok=Tokenizer.from_file(path)
    except ImportError:return
    key=re.compile(r'LTAI[A-Za-z0-9]{8,}');sig=re.compile(r'Signature=(?!<redacted>)[A-Za-z0-9%+/=]{8,}');bad=[]
    def walk(o,f):
        if isinstance(o,dict):
            for v in o.values():walk(v,f)
        elif isinstance(o,list):
            if len(o)>8 and all(isinstance(x,int) and not isinstance(x,bool) for x in o):
                t=tok.decode(o,skip_special_tokens=False)
                if key.search(t) or sig.search(t):bad.append(f)
            else:
                for v in o:walk(v,f)
    for p in list((ROOT/'runs').rglob('*.json*'))+list((ROOT/'data').rglob('*.json*')):
        rel=str(p.relative_to(ROOT))
        if p.suffix=='.jsonl':
            for line in p.read_text().splitlines():
                if line.strip():walk(json.loads(line),rel)
        else:walk(json.loads(p.read_text()),rel)
    assert not bad,sorted(set(bad))
