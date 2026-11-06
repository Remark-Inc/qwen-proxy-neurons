"""Ledger-budgeted item-7c jobs: build real NVFP4 checkpoints with llm-compressor on an H200, then
serve them natively (FP4 weights and activations) with the pinned vLLM image on a B200. Calls stay text."""
import hashlib,json
from pathlib import Path
import modal

VLLM_IMAGE='vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34'
cache=modal.Volume.from_name('qwen-neuron-lab-models')
results=modal.Volume.from_name('qwen-neuron-lab-results')
# The Transformers pins of every other job, plus llm-compressor 0.15.0 (it requires torch <= 2.14.1 and
# transformers <= 5.21.0, and brings compressed-tensors 0.20.1). No network at run time.
quant_image=(modal.Image.debian_slim(python_version='3.12')
             .pip_install('torch==2.14.1','transformers==5.21.0','accelerate==1.15.0','llmcompressor==0.15.0')
             .env({'HF_HOME':'/models/hf','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
             .add_local_python_source('qwen_lab')
             .add_local_dir('data/coverage-sweep','/root/data/coverage-sweep',ignore=['budget.json']))
# Same image and environment as the H200 vLLM jobs.
vllm_image=(modal.Image.from_registry(VLLM_IMAGE).entrypoint([])
            .run_commands('ln -sf /usr/bin/python3 /usr/bin/python')
            .env({'HF_HOME':'/models/hf','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                  'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_USE_V2_MODEL_RUNNER':'1'})
            .add_local_python_source('qwen_lab'))
app=modal.App('qwen-coverage-nvfp4')

@app.function(image=quant_image,gpu='H200',cpu=4,memory=196608,timeout=4800,
              volumes={'/models':cache,'/results':results},max_containers=1,retries=0,
              scaledown_window=2,block_network=True)
def quantize(config:dict):
    from qwen_lab.coverage_nvfp4 import quantize as run
    out=run(config,Path('/results'),results.commit);cache.commit();return out

@app.function(image=quant_image,cpu=4,memory=65536,timeout=3600,
              volumes={'/models':cache,'/results':results},max_containers=1,retries=0,
              scaledown_window=2,block_network=True)
def edit_after(config:dict):
    from qwen_lab.coverage_nvfp4 import edit_after as run
    out=run(config,Path('/results'),results.commit);cache.commit();return out

@app.function(image=vllm_image,gpu='B200',cpu=4,memory=98304,timeout=4800,
              volumes={'/models':cache,'/results':results},max_containers=1,retries=0,
              scaledown_window=2,block_network=True)
def serve(config:dict):
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
    if J['app']!='coverage_nvfp4_app.py' or not run_id.startswith(J['run_id_prefix']) or not 1<=seconds<=J['soft_seconds']:raise ValueError('Bounds exceeded')
    ref=frozen(F['neurons']['file'],F['neurons']['sha256'])
    neurons={'unmodified':[],'28':ref,'27':[n for n in ref if ident(n)!='L56/N16758']}
    if J['phase']=='nvfp4_after':
        config=dict(run_id=run_id,seconds=seconds,quantize_run=J['quantize_run'],labels=P['jobs']['nvfp4-quantize']['labels'],neurons=neurons)
        receipt=edit_after.remote(config);names=('result.json',)
    elif J['phase']=='nvfp4_quantize':
        frozen(Path('data')/F['nvfp4_calibration']['path'],F['nvfp4_calibration']['sha256'])
        config=dict(run_id=run_id,seconds=seconds,calibration=F['nvfp4_calibration'],labels=J['labels'],neurons=neurons)
        receipt=quantize.remote(config);names=('result.json',)
    else:
        cases=frozen(Path('data')/F['vllm_cases']['path'],F['vllm_cases']['sha256'])
        config=dict(run_id=run_id,seconds=seconds,engines=J['engines'],cases=cases,neurons=neurons,seed=P['item7']['seed'],max_new_tokens=P['item7']['max_new_tokens'])
        receipt=serve.remote(config);names=('result.json','rollouts.jsonl')
    folder=Path('runs')/run_id;folder.mkdir(parents=True,exist_ok=False)
    for name in names:
        try:
            with (folder/name).open('wb') as f:
                for chunk in results.read_file(f'{run_id}/{name}'):f.write(chunk)
        except Exception as e:print(json.dumps({'missing':name,'error':repr(e)}))
    print(json.dumps({'run':run_id,**{k:v for k,v in receipt.items() if k!='errors'},'errors':receipt.get('errors')}))
