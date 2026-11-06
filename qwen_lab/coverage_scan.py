"""Coverage sweep, item 3: a wider Alibaba-host detector over every saved run, no GPU.

Counts hostnames in generated text only: full rollouts (any record with generated
'text' and 'metrics', including vLLM rows) and sampled decision-point
outcomes. Prompts, fixtures, case prefixes, profiles and configs are skipped.
Reports carry hostnames and counts, never text, URL paths, or the memorized access key.
"""
import collections,json,re,sys,zipfile
from pathlib import Path
from qwen_lab.http_path import SECRET,ident
from qwen_lab.metrics import PROXY

WIDE=re.compile(r'(?:^|\.)(?:aliyuncs\.com|alibaba-inc\.com|aliyun\.com)$|oss-cn-|dashscope|routify|sanmaohe')
HOSTNAME=re.compile(r'(?i)(?<![a-z0-9.-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z][a-z0-9-]*[a-z0-9])(?![a-z0-9-])')
CALL=re.compile(r'<tool_call>.*?(?:</tool_call>|$)',re.S)


def hosts(text):
    """Matching hostnames in one string, lowercased."""
    return {h.lower().rstrip('.') for h in HOSTNAME.findall(text) if WIDE.search(h.lower())}


def places(text):
    """host -> places ('call', 'final_text', 'reasoning') in one generated text."""
    end=text.rfind('</think>');reasoning,final=(text[:end],text[end:]) if end>=0 else ('',text)
    calls=''.join(CALL.findall(final));plain=CALL.sub('',final);out=collections.defaultdict(set)
    for place,part in (('call',calls),('final_text',plain),('reasoning',reasoning)):
        for h in hosts(part):out[h].add(place)
    return out


def arm_label(row,inherited,named):
    if row.get('effective_neurons') is not None or row.get('neurons') is not None:
        specs=row.get('effective_neurons') if row.get('effective_neurons') is not None else row['neurons']
        names=frozenset(ident(n) for n in specs if 'expert' not in n)|frozenset(f"L{n['layer']}/E{n['expert']}/N{n['neuron']}" for n in specs if 'expert' in n)
        scope=row.get('intervention_scope');scope='' if scope in (None,'all') or not names else f' ({scope} positions only)'
        return named.get(names,'unmodified' if not names else f'{len(names)} neurons (other set)')+scope
    own=row.get('arm') if isinstance(row.get('arm'),str) else row.get('explanation_variant')
    return own or inherited or 'unlabelled'


def sample_label(arm,path):
    """Arm for decision-point samples: an enclosing arm field, else the key path without case IDs or indices."""
    keys=[k for k in path if isinstance(k,str) and k.count(':')<3]
    return (f'{arm} @ ' if arm else '')+'/'.join(keys)


def documents(path):
    if path.suffix=='.json':yield path.name,json.loads(path.read_text())
    elif path.suffix=='.jsonl':
        for line in path.read_text().splitlines():
            if line.strip():yield path.name,json.loads(line)
    elif path.suffix=='.zip':
        z=zipfile.ZipFile(path)
        for name in z.namelist():
            if name.endswith('.json'):yield name,json.loads(z.read(name))
            elif name.endswith('.jsonl'):
                for line in z.read(name).decode().splitlines():
                    if line.strip():yield name,json.loads(line)


SKIP=('config','fixture','cases','profiles','prompt','messages','harvested_cases','recall_fixtures','scores','distributions')


def records(obj,arm=None,path=()):
    """Yield ('rollout', arm, row) for generated rows and ('sample', label, host) for decision-point samples."""
    if isinstance(obj,dict):
        if isinstance(obj.get('text'),str) and isinstance(obj.get('metrics'),dict):
            yield 'rollout',arm,obj;return
        if isinstance(obj.get('host'),str) and 'continuation' in obj:
            yield 'sample',sample_label(arm,path),obj['host'];return
        arm=obj['arm'] if isinstance(obj.get('arm'),str) else obj.get('explanation_variant',arm)
        for k,v in obj.items():
            if k not in SKIP:yield from records(v,arm,path+(k,))
    elif isinstance(obj,list):
        if len(obj)==4 and isinstance(obj[2],str) and isinstance(obj[3],bool) and all(isinstance(x,int) for x in obj[:2]):
            yield 'sample',sample_label(arm,path),obj[2];return
        for v in obj:yield from records(v,arm,path)


def scan(roots,named):
    """Per (run, arm): generated records and records per matching host and place."""
    out={}
    for root in roots:
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            for path in sorted(folder.rglob('*')):
                if path.suffix not in ('.json','.jsonl','.zip') or path.name in ('config.json','metadata.json','progress.json'):continue
                for name,doc in documents(path):
                    model=(doc.get('metadata') or {}).get('model') or doc.get('model') if isinstance(doc,dict) else None
                    for kind,arm,item in records(doc):
                        if kind=='rollout':
                            label=arm_label(item,arm,named);found=places(item['text'])
                        else:
                            label=arm;found={h:{'decision_sample'} for h in hosts(item)}
                        key=(folder.name,kind,label);r=out.setdefault(key,dict(run=folder.name,root=str(root),kind=kind,arm=label,
                                                                                 model=model,records=0,hosts=collections.Counter()))
                        r['records']+=1;r['model']=r['model'] or model
                        for h,ps in found.items():
                            for p in ps:r['hosts'][f'{h}|{p}']+=1
    return out


def summarize(scanned):
    """Every matching host other than the routify proxy, with counts per arm, plus the proxy for reference."""
    rows=[]
    for r in scanned.values():
        for k,n in r['hosts'].items():
            h,p=k.split('|');rows.append(dict(run=r['run'],kind=r['kind'],arm=r['arm'],model=r['model'],host=h,place=p,records_with_host=n,records=r['records']))
    other=sorted({x['host'] for x in rows if x['host']!=PROXY})
    by_host={h:sorted([x for x in rows if x['host']==h],key=lambda x:(x['run'],x['arm'],x['place'])) for h in other}
    totals=collections.defaultdict(lambda:collections.Counter())
    for r in scanned.values():totals[(r['kind'],r['arm'])]['records']+=r['records']
    for x in rows:totals[(x['kind'],x['arm'])][x['host']+'|'+x['place']]+=x['records_with_host']
    return dict(other_hosts=other,by_host=by_host,
                totals=[dict(kind=k,arm=a,**c) for (k,a),c in sorted(totals.items())],
                proxy=[x for x in rows if x['host']==PROXY])


if __name__=='__main__':
    roots=sys.argv[1:-1];dest=sys.argv[-1]
    P=json.loads(Path('data/neuron-roles/protocol.json').read_text())['frozen_inputs']['neurons']
    order=P['order'];https12=P['https_12'];drop='L56/N16758'
    named={frozenset(order):'28',frozenset(https12):'12',frozenset(n for n in order if n!=drop):'27',frozenset(n for n in https12 if n!=drop):'11',
           frozenset([drop]):'L56/N16758 alone',frozenset(['L63/N4455']):'1 (L63/N4455)',frozenset(['L63/N417','L63/N4455']):'2',
           frozenset(['L63/N417','L63/N14734','L63/N4455']):'3'}
    scanned=scan(roots,named);out=dict(roots=roots,scanned=[dict(v,hosts=dict(v['hosts'])) for v in scanned.values()],**summarize(scanned))
    text=json.dumps(out,indent=1)
    if SECRET.search(text):raise ValueError('Credential pattern in scan output')
    Path(dest).write_text(text+'\n');print(json.dumps(dict(other_hosts=out['other_hosts'],runs=len({v['run'] for v in scanned.values()})),indent=1))
