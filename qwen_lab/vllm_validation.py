"""Scoring helpers for vLLM replay outputs (used by the coverage sweep's batched vLLM job)."""
import math


def token_logprobs(output,expected):
    completion=output.outputs[0]
    if list(completion.token_ids)!=list(expected):
        raise ValueError(f'Trace replay did not emit the requested tokens: expected={list(expected)}, actual={list(completion.token_ids)}, finish_reason={completion.finish_reason}')
    if len(completion.logprobs)!=len(expected):raise ValueError('Missing replay probabilities')
    values=[float(row[token].logprob) for token,row in zip(expected,completion.logprobs)]
    if any(not math.isfinite(x) or x>1e-5 for x in values):raise ValueError('Invalid log probabilities')
    total=sum(values)
    return dict(logprob=total,probability=math.exp(total),per_token=values)


def prompt_score(output,prefix,suffix):
    rows=output.prompt_logprobs[len(prefix):len(prefix)+len(suffix)]
    if len(rows)!=len(suffix):raise ValueError('Missing prompt scores')
    values=[float(row[token].logprob) for token,row in zip(suffix,rows)]
    return dict(logprob=sum(values),probability=math.exp(sum(values)),per_token=values)


def compare_scores(a,b):
    return dict(absolute_logprob_difference=abs(a['logprob']-b['logprob']),
                absolute_probability_difference=abs(a['probability']-b['probability']),
                first_token_probability_difference=abs(math.exp(a['per_token'][0])-math.exp(b['per_token'][0])))
