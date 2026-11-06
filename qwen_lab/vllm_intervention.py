"""Verified, reversible BF16 expert-column edits inside vLLM workers.

Locate the original checkpoint column by exact value, rather than assuming a
packed tensor's shard offset. The coordinator requires one owner per column.
"""
import hashlib
import re
from pathlib import Path
import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def column_match(weight, expert, expected):
    if weight.dtype!=torch.bfloat16 or weight.ndim!=3:
        raise ValueError('Only unquantized [expert, hidden, intermediate] BF16 weights are supported')
    if weight.shape[1]!=expected.numel() or not 0<=expert<weight.shape[0]:
        raise ValueError('Unexpected expert/hidden layout')
    matches=(weight[expert].detach().cpu()==expected.cpu().reshape(-1,1)).all(dim=0).nonzero().flatten().tolist()
    if len(matches)>1:raise ValueError('Checkpoint column is not uniquely identifiable')
    return matches[0] if matches else None


def prepare(model,specs,model_path):
    import json
    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
    if hasattr(model,'_neuron_lab_columns'):raise ValueError('Already prepared')
    parameters=dict(model.named_parameters());index=json.loads((Path(model_path)/'model.safetensors.index.json').read_text())['weight_map']
    saved=[];receipt=[]
    for spec in specs:
        layer,expert,neuron=(spec[k] for k in ('layer','expert','neuron'))
        pattern=re.compile(rf'(?:^|\.)layers\.{layer}\.mlp\..*w2_weight$')
        found=[(name,p) for name,p in parameters.items() if pattern.search(name)]
        if len(found)!=1:raise ValueError(f'Expected one down-projection parameter for layer {layer}, found {[n for n,p in found]}')
        name,weight=found[0]
        key=f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        with safe_open(str(Path(model_path)/index[key]),framework='pt',device='cpu') as f:
            expected=f.get_slice(key)[expert:expert+1,:,neuron:neuron+1].squeeze(0).squeeze(-1)
        col=column_match(weight,expert,expected)
        if col is not None:
            target=weight[expert,:,col]
            saved.append(dict(spec=spec,target=target,original=target.detach().clone()))
        receipt.append(dict(spec=spec,parameter=name,shape=list(weight.shape),stride=list(weight.stride()),
                            local_column=col,checkpoint_column_sha256=digest(expected)))
    model._neuron_lab_columns=saved
    return dict(rank=get_tensor_model_parallel_rank(),world_size=get_tensor_model_parallel_world_size(),columns=receipt)


def set_arm(model,active):
    chosen={(n['layer'],n['expert'],n['neuron']) for n in active};receipt=[]
    with torch.no_grad():
        for item in model._neuron_lab_columns:
            spec=item['spec'];target=item['target'];original=item['original']
            enabled=(spec['layer'],spec['expert'],spec['neuron']) in chosen
            target.copy_(original)
            if enabled:target.zero_()
            if enabled and torch.count_nonzero(target).item()!=0:raise ValueError('Ablation failed')
            if not enabled and not torch.equal(target,original):raise ValueError('Restoration failed')
            receipt.append(dict(spec=spec,ablated=enabled,original_sha256=digest(original),current_sha256=digest(target)))
    return receipt


def verify_owners(receipts,specs):
    if len(receipts)!=4 or {r['rank'] for r in receipts}!={0,1,2,3} or {r['world_size'] for r in receipts}!={4}:
        raise ValueError('Expected all four tensor-parallel workers')
    for spec in specs:
        rows=[c for r in receipts for c in r['columns'] if c['spec']==spec]
        if len({c['checkpoint_column_sha256'] for c in rows})!=1:raise ValueError('Workers disagree on checkpoint column')
        if sum(c['local_column'] is not None for c in rows)!=1:raise ValueError(f'Expected exactly one owner for {spec}')


class NeuronLabWorkerExtension:
    """Named worker RPCs with plain data; no pickle-based function transport."""
    def neuron_lab_prepare(self,specs,model_path):
        return prepare(self.get_model(),specs,model_path)

    def neuron_lab_set_arm(self,active):
        return set_arm(self.get_model(),active)
