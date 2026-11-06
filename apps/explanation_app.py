"""Ledger-budgeted follow-up; each invocation hard-capped at 90 minutes."""
import json
from pathlib import Path
import modal
cache=modal.Volume.from_name('qwen-neuron-lab-models')
results=modal.Volume.from_name('qwen-neuron-lab-results')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','transformers==5.21.0','accelerate==1.15.0',
                    'huggingface-hub==2.3.0','numpy==2.5.4')
       .env({'HF_HOME':'/models/hf','TOKENIZERS_PARALLELISM':'false'})
       .add_local_python_source('qwen_lab'))

app=modal.App('qwen-neuron-explanation')
@app.function(image=image,gpu='H200',volumes={'/models':cache,'/results':results},
              cpu=4,memory=98304,timeout=5400,max_containers=1,
              scaledown_window=2,retries=0,block_network=True)
def experiment(config:dict):
    from qwen_lab.runner import run
    out=run(config,Path('/results'));results.commit()
    # Large return values use an external blob upload, which block_network
    # correctly prevents. Return a compact receipt and read the durable file
    # through the local client's volume API instead.
    return dict(run_id=config['run_id'],complete=out['summary'].get('complete'),
                elapsed=out['elapsed_seconds'],estimated_compute_usd=out['estimated_compute_usd'])

@app.local_entrypoint()
def main(phase:str='smoke',run_id:str='explain-smoke-1',seconds:int=4800,samples:int=64,seed:int=370000):
    if phase not in {'smoke','conditional','contexts','timing','patches','profile','generate_restore','generate_single','generate_timing','failure_probe','blog_followup'}:raise ValueError(phase)
    if not 1<=seconds<=4800 or not 1<=samples<=128:raise ValueError('Bounds exceeded')
    ns=json.loads(Path('data/candidate-zero-proxy.json').read_text())
    config=dict(stage='explain',phase=phase,run_id=run_id,seconds=seconds,samples=samples,seed=seed,neurons=ns)
    if phase in {'failure_probe','blog_followup'}:config['failure_rows']=json.loads(Path('data/explanation/new-all12-failures.json').read_text())
    receipt=experiment.remote(config)
    folder=Path('runs')/run_id;folder.mkdir(parents=True,exist_ok=False)
    with (folder/'result.json').open('wb') as f:
        for chunk in results.read_file(run_id+'/result.json'):f.write(chunk)
    out=json.loads((folder/'result.json').read_text())
    assert out['config']['run_id']==run_id and out['summary'].get('complete')==receipt['complete']
    print(json.dumps({'run':run_id,**receipt}))
