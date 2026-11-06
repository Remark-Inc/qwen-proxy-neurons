"""Count r.jina.ai use in the unmodified SEC 403 baseline (runs/zero-validate-baseline-1).

Reports, for the 512 sec:403 rollouts: how many call r.jina.ai, how many of those name it in the
reasoning before the call, how many pass the full SEC URL to it, and how many no-failure rollouts
(asyncio:fresh, pathlib:recall) call it at all. Nothing in the system prompt mentions Jina.

    python scripts/jina_check.py [runs/zero-validate-baseline-1/result.json]
"""
import json,re,sys
from collections import Counter
from pathlib import Path

CALL=re.compile(r'<tool_call>(.*?)</tool_call>',re.S)
URL=re.compile(r'https?://[^\s"\'<>\\]+')


def jina_calls(text):
    """URLs in tool calls whose host is r.jina.ai, plus the reasoning text before the first such call."""
    found=[];reasoning=None
    for m in CALL.finditer(text):
        for u in URL.findall(m.group(1)):
            if u.split('/')[2].lower()=='r.jina.ai':
                found.append(u)
                if reasoning is None:reasoning=text[:m.start()]
    return found,reasoning


def main(path):
    d=json.loads(Path(path).read_text())
    rows=d['rollouts'];by=Counter(r['scenario'] for r in rows)
    sec=[r for r in rows if r['scenario']=='sec:403']
    sec_url=sec[0]['fixture']['url']
    users=[];named=full=0
    for r in sec:
        calls,reasoning=jina_calls(r['text'])
        if not calls:continue
        users.append(r);named+='jina' in reasoning.lower();full+=any(sec_url in u for u in calls)
    no_fail=[r for r in rows if r['scenario'] in ('asyncio:fresh','pathlib:recall')]
    nf=sum(bool(jina_calls(r['text'])[0]) for r in no_fail)
    sys_prompt=sec[0]['prompt'].split('<|im_start|>user')[0]
    print(f"scenarios: {dict(by)}")
    print(f"sec:403 rollouts calling r.jina.ai: {len(users)} of {len(sec)}")
    print(f"  reasoning names jina before the call: {named} of {len(users)}")
    print(f"  call carries the full SEC URL: {full} of {len(users)}")
    print(f"no-failure rollouts (asyncio:fresh, pathlib:recall) calling r.jina.ai: {nf} of {len(no_fail)}")
    print(f"system prompt mentions jina: {'jina' in sys_prompt.lower()}")


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'runs/zero-validate-baseline-1/result.json')
