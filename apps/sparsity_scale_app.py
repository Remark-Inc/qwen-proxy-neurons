"""Ledger-budgeted SEC 403 rerun of the frozen sparsity arms; tool calls are recorded, never executed."""
import hashlib,json
from pathlib import Path
import modal
cache=modal.Volume.from_name('qwen-neuron-lab-models')
results=modal.Volume.from_name('qwen-neuron-lab-results')
# Same pins as modal_app.py, which ran the seed-50000 confirmation. Only code is uploaded:
# the neuron list travels in the config after a local hash check.
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','transformers==5.21.0','accelerate==1.15.0',
                    'huggingface-hub==2.3.0','numpy==2.5.4')
       .env({'HF_HOME':'/models/hf','TOKENIZERS_PARALLELISM':'false'})
       .add_local_python_source('qwen_lab'))

app=modal.App('qwen-neuron-sparsity-scale')
@app.function(image=image,gpu='H200',volumes={'/models':cache,'/results':results},
              cpu=4,memory=98304,timeout=4800,max_containers=1,
              scaledown_window=2,retries=0,block_network=True)
def experiment(config:dict):
    from qwen_lab.sparsity_scale import run
    # Rollouts stay on the volume (block_network prevents blob returns); every chunk commits.
    return run(config,Path('/results'),results.commit)

@app.local_entrypoint()
def main(arm:str='baseline',run_id:str='',seconds:int=4100):
    P=json.loads(Path('data/sparsity-scale/protocol.json').read_text())
    if arm not in P['arms']+list(P['optional_arms']):raise ValueError(arm)
    if not run_id.startswith(f'sparsity-scale-{arm}-') or not 1<=seconds<=P['max_soft_deadline_seconds']:raise ValueError('Bounds exceeded')
    src=P['arm_sources'][arm];raw=Path(src['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=src['sha256']:raise ValueError('Neuron file differs from frozen hash')
    neurons=json.loads(raw)
    if 'condition' in src:neurons=neurons['conditions'][src['condition']]
    if 'same_as' in src:
        s=src['same_as'];other=Path(s['file']).read_bytes();key=lambda ns:[(n['layer'],n['neuron'],n['mode'],n['value']) for n in ns]
        if hashlib.sha256(other).hexdigest()!=s['sha256'] or key(json.loads(other)[:s['first']])!=key(neurons):raise ValueError('Neuron set differs from its reference')
    G=P['generation']
    config=dict(run_id=run_id,arm=arm,neurons=neurons,scenario=P['scenario'],samples=P['samples_per_arm'],chunk=P['chunk'],
                seed=P['seed'],batch_size=G['batch_size'],max_new_tokens=G['max_new_tokens'],seconds=seconds)
    receipt=experiment.remote(config)
    folder=Path('runs')/run_id;folder.mkdir(parents=True,exist_ok=False)
    for name in ('result.json','rollouts.jsonl'):
        with (folder/name).open('wb') as f:
            for chunk in results.read_file(f'{run_id}/{name}'):f.write(chunk)
    out=json.loads((folder/'result.json').read_text())
    assert out['config']['run_id']==run_id and out['summary']['completed']==receipt['completed']
    print(json.dumps({'run':run_id,**receipt}))
