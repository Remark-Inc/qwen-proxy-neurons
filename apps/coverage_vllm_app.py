"""Ledger-budgeted coverage-sweep jobs on the pinned vLLM image: SEC 403 under batched vLLM in
bf16 and FP8, unmodified and with the patch's columns zeroed. No public endpoint; calls stay text."""
import hashlib,json
from pathlib import Path
import modal

IMAGE='vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34'
cache=modal.Volume.from_name('qwen-neuron-lab-models')
results=modal.Volume.from_name('qwen-neuron-lab-results')
# vLLM image and environment for the batched serving-engine checks.
image=(modal.Image.from_registry(IMAGE).entrypoint([])
    .run_commands('ln -sf /usr/bin/python3 /usr/bin/python')
    .env({'HF_HOME':'/models/hf','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
          'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_USE_V2_MODEL_RUNNER':'1'})
    .add_local_python_source('qwen_lab'))
app=modal.App('qwen-coverage-vllm')

@app.function(image=image,gpu='H200',cpu=4,memory=98304,timeout=4800,
              volumes={'/models':cache,'/results':results},max_containers=1,retries=0,
              scaledown_window=2,block_network=True)
def experiment(config:dict):
    from qwen_lab.coverage_vllm import run
    return run(config,Path('/results'),results.commit)


def frozen(path,sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError(f'{path} differs from the frozen protocol')
    return json.loads(raw)


@app.local_entrypoint()
def main(job:str='',run_id:str='',seconds:int=600):
    from qwen_lab.http_path import ident
    P=json.loads(Path('data/coverage-sweep/protocol.json').read_text());J=P['jobs'][job];F=P['frozen_inputs']
    if J['app']!='coverage_vllm_app.py' or not run_id.startswith(J['run_id_prefix']) or not 1<=seconds<=J['soft_seconds']:raise ValueError('Bounds exceeded')
    ref=frozen(F['neurons']['file'],F['neurons']['sha256']);cases=frozen(Path('data')/F['vllm_cases']['path'],F['vllm_cases']['sha256'])
    neurons={'unmodified':[],'28':ref,'27':[n for n in ref if ident(n)!='L56/N16758']}
    config=dict(run_id=run_id,seconds=seconds,engines=J['engines'],cases=cases,neurons=neurons,seed=P['item7']['seed'],max_new_tokens=P['item7']['max_new_tokens'])
    receipt=experiment.remote(config)
    folder=Path('runs')/run_id;folder.mkdir(parents=True,exist_ok=False)
    for name in ('result.json','rollouts.jsonl'):
        try:
            with (folder/name).open('wb') as f:
                for chunk in results.read_file(f'{run_id}/{name}'):f.write(chunk)
        except Exception as e:print(json.dumps({'missing':name,'error':repr(e)}))
    print(json.dumps({'run':run_id,**{k:v for k,v in receipt.items() if k!='errors'},'errors':receipt.get('errors')}))
