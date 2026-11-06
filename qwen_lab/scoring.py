"""Teacher-forced sequence likelihoods. Never generate or execute tools here."""
import torch
from qwen_lab.interventions import intervene

def continuation_logprob(model, prefix_ids, suffix_ids):
    if not prefix_ids or not suffix_ids:
        raise ValueError('Prefix and continuation must both be nonempty')
    device=next(model.parameters()).device
    combined=prefix_ids+suffix_ids
    # Last K logits of (prefix + continuation[:-1]) predict all K suffix tokens.
    ids=torch.tensor([combined[:-1]],device=device)
    out=model(input_ids=ids,use_cache=False,logits_to_keep=len(suffix_ids))
    logits=out.logits[0,-len(suffix_ids):].float()
    targets=torch.tensor(suffix_ids,device=device)
    per_token=logits.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)
    return dict(logprob=float(per_token.sum()), tokens=len(suffix_ids),
                mean_logprob=float(per_token.mean()),per_token=per_token.cpu().tolist())

def score_cases(model,cases,neurons=()):
    rows=[]
    with intervene(model,neurons),torch.inference_mode():
        for c in cases:
            bad=continuation_logprob(model,c['prefix_ids'],c['bad_ids'])
            good=continuation_logprob(model,c['prefix_ids'],c['good_ids'])
            rows.append(dict(id=c['id'],boundary=c['boundary'],synthetic=c['synthetic'],
                scope=c['scope'],bad=bad,good=good,log_odds=bad['logprob']-good['logprob']))
    return rows

def aggregate(rows):
    return sum(r['log_odds'] for r in rows)/len(rows)
