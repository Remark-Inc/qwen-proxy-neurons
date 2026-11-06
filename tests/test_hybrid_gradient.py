"""Regression: a downstream recurrent layer must not overwrite saved gradients."""
import torch
from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
from qwen_lab.interventions import protect_recurrent_cache_for_grad

def test_cached_recurrent_gradient_matches_finite_difference():
    torch.manual_seed(2)
    cfg=Qwen3_5Config(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        layer_types=['linear_attention','linear_attention','full_attention'],
        linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=16,
        linear_value_head_dim=16,linear_conv_kernel_dim=4),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
                           out_hidden_size=32,num_position_embeddings=16))
    m=Qwen3_5ForConditionalGeneration(cfg).eval().requires_grad_(False)
    ids=torch.tensor([[1,2,3,4]])
    def forward(delta, protect=True):
        with torch.no_grad():
            cache=m(input_ids=ids[:,:-1],use_cache=True,logits_to_keep=1).past_key_values
        if protect:
            protect_recurrent_cache_for_grad(cache)
        def hook(mod,inp):
            mask=torch.zeros_like(inp[0]);mask[...,0]=1
            return (inp[0]+delta*mask,)
        handle=m.model.language_model.layers[0].mlp.down_proj.register_forward_pre_hook(hook)
        try:
            out=m(input_ids=ids[:,-1:],past_key_values=cache,use_cache=True,logits_to_keep=1)
            return out.logits[0,-1,7]-out.logits[0,-1,8]
        finally:
            handle.remove()
    d=torch.tensor(0.,requires_grad=True)
    grad=torch.autograd.grad(forward(d),d)[0]
    with torch.no_grad():
        assert torch.allclose(forward(torch.tensor(0.)),forward(torch.tensor(0.),False),atol=1e-6)
        fd=(forward(torch.tensor(.001))-forward(torch.tensor(-.001)))/.002
    assert torch.isfinite(grad)
    assert torch.allclose(grad,fd,atol=.001,rtol=.02)
