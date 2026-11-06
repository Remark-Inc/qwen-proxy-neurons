"""Ledger-budgeted coverage-sweep jobs on the pinned Transformers stack: the proxy habit in other
URL forms (decision points) and the SEC 403 task with a bash tool. Calls stay text."""
import json
from pathlib import Path
import modal
cache=modal.Volume.from_name('qwen-neuron-lab-models')
results=modal.Volume.from_name('qwen-neuron-lab-results')
# Same pins as modal_app.py and the neuron-roles jobs. Ledgers change while launchers run;
# containers only read case files.
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','transformers==5.21.0','accelerate==1.15.0',
                    'huggingface-hub==2.3.0','numpy==2.5.4')
       .env({'HF_HOME':'/models/hf','TOKENIZERS_PARALLELISM':'false'})
       .add_local_python_source('qwen_lab')
       .add_local_dir('data/coverage-sweep','/root/data/coverage-sweep',ignore=['budget.json']))

app=modal.App('qwen-coverage-sweep')
@app.function(image=image,gpu='H200',volumes={'/models':cache,'/results':results},
              cpu=4,memory=98304,timeout=4800,max_containers=1,
              scaledown_window=2,retries=0,block_network=True)
def experiment(config:dict):
    # Results stay on the volume (block_network prevents blob returns); every stage commits.
    from qwen_lab.coverage_forms import run
    return run(config,Path('/results'),results.commit)


@app.local_entrypoint()
def main(job:str='',run_id:str='',seconds:int=600):
    from qwen_lab.coverage_forms import job_config
    config=job_config(json.loads(Path('data/coverage-sweep/protocol.json').read_text()),job,run_id,seconds)
    receipt=experiment.remote(config)
    folder=Path('runs')/run_id;folder.mkdir(parents=True,exist_ok=False)
    for name in ('result.json',)+(('rollouts.jsonl',) if config['phase']=='bash' else ()):
        with (folder/name).open('wb') as f:
            for chunk in results.read_file(f'{run_id}/{name}'):f.write(chunk)
    out=json.loads((folder/'result.json').read_text())
    assert out['config']['run_id']==run_id and out['summary'].get('complete')==receipt['complete']
    print(json.dumps({'run':run_id,**{k:v for k,v in receipt.items() if k!='errors'},'errors':receipt.get('errors')}))
