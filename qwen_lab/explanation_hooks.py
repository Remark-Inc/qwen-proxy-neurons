"""Position-specific interventions; separate from the frozen validation hook."""
from contextlib import contextmanager
from collections import defaultdict
import torch
from qwen_lab.interventions import language_layers


def phase_mask(ids, prompt_length, markers, scope):
    """Classify positions by content already consumed, not a future boundary.

    The position consuming </think> predicts the next (final-answer) token.
    The position consuming <tool_call> predicts the call contents. Thus these
    are predictor-position windows, not labels on output tokens themselves.
    """
    phase='reasoning';out=[];history=[]
    for i,t in enumerate(ids):
        history.append(t)
        if i>=prompt_length:
            if history[-len(markers['end_think']):]==markers['end_think']:phase='final'
            if history[-len(markers['open_call']):]==markers['open_call']:phase='call'
            if history[-len(markers['close_call']):]==markers['close_call']:phase='final'
        p='prompt' if i<prompt_length else phase
        out.append(scope=='all' or p==scope or (scope=='generated' and p!='prompt'))
    return out


@contextmanager
def scoped_intervention(model, specs, start=0, end=None, dynamic=None):
    """Ablate/set selected coordinates over an exact interval or live phase.

    Dynamic generation uses the actual tokens separately for every batch row.
    No generated tool is executed. Static teacher forcing never uses KV caches.
    """
    layers=language_layers(model);grouped=defaultdict(list);handles=[];state={}
    for s in specs:
        if s.get('mode','scale') not in {'scale','set'}:raise ValueError('Unknown mode')
        grouped[s['layer']].append(s)
    def before(module,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is None:raise ValueError('Token IDs required for scoped hooks')
        if dynamic:
            rows=ids.detach().cpu().tolist()
            if ids.shape[1]>1 or 'tails' not in state:
                state.update(tails=[[] for _ in rows],phases=['reasoning']*len(rows),offset=0)
            if len(rows)!=len(state['tails']):raise ValueError('Batch changed')
            masks=[];markers=dynamic['markers'];scope=dynamic['scope']
            keep=max(map(len,markers.values()))
            for j,row in enumerate(rows):
                mask=[]
                for k,t in enumerate(row):
                    tail=(state['tails'][j]+[t])[-keep:];state['tails'][j]=tail
                    if state['offset']+k>=dynamic['prompt_length']:
                        if tail[-len(markers['end_think']):]==markers['end_think']:state['phases'][j]='final'
                        if tail[-len(markers['open_call']):]==markers['open_call']:state['phases'][j]='call'
                        if tail[-len(markers['close_call']):]==markers['close_call']:state['phases'][j]='final'
                    p='prompt' if state['offset']+k<dynamic['prompt_length'] else state['phases'][j]
                    mask.append(scope=='all' or p==scope or (scope=='generated' and p!='prompt'))
                masks.append(mask)
            state['offset']+=ids.shape[1]
            state['forward_calls']=state.get('forward_calls',0)+1
            state['selected_current']=sum(sum(mask) for mask in masks)
            state['selected_predictor_positions']=state.get('selected_predictor_positions',0)+state['selected_current']
            state['mask']=torch.tensor(masks,device=ids.device,dtype=torch.bool)
        else:
            pos=torch.arange(ids.shape[1],device=ids.device)
            state['mask']=((pos>=start)&(pos<(end if end is not None else ids.shape[1])))[None,:]
    def hook_for(items):
        def hook(module,inputs):
            x=inputs[0].clone();mask=state['mask']
            if dynamic:state['neuron_coordinate_applications']=state.get('neuron_coordinate_applications',0)+state['selected_current']*len(items)
            if x.shape[:2]!=mask.shape and mask.shape[0]!=1:raise ValueError('Mask shape mismatch')
            for s in items:
                n=s['neuron'];old=x[...,n];value=s.get('value',0)
                replacement=old*value if s.get('mode','scale')=='scale' else torch.full_like(old,value)
                x[...,n]=torch.where(mask,replacement,old)
            return (x,*inputs[1:])
        return hook
    try:
        handles.append(model.register_forward_pre_hook(before,with_kwargs=True))
        for l,items in grouped.items():handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(hook_for(items)))
        yield state
    finally:
        for h in handles:h.remove()


@contextmanager
def capture(model, specs, destination):
    """Capture only the chosen coordinates, at every supplied token position."""
    layers=language_layers(model);grouped=defaultdict(list);handles=[]
    for s in specs:grouped[s['layer']].append(s['neuron'])
    def hook_for(l,ns):
        def hook(module,inputs):
            a=inputs[0][0,:,ns].detach().float().cpu()
            for j,n in enumerate(ns):destination[f'L{l}/N{n}']=a[:,j].tolist()
        return hook
    try:
        for l,ns in grouped.items():handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(hook_for(l,ns)))
        yield
    finally:
        for h in handles:h.remove()
