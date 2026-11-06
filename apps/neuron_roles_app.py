"""Ledger-budgeted neuron-roles jobs: drop-one on the 16 http:// neurons, activation profiles,
alternative-token effects, the Wayback check and option D's SEC runs. Calls stay text."""
import hashlib,json
from pathlib import Path
import modal
cache=modal.Volume.from_name('qwen-neuron-lab-models')
results=modal.Volume.from_name('qwen-neuron-lab-results')
# Same pins as modal_app.py and the sparsity rerun. Ledgers change while launchers run;
# containers only read case files.
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','transformers==5.21.0','accelerate==1.15.0',
                    'huggingface-hub==2.3.0','numpy==2.5.4')
       .env({'HF_HOME':'/models/hf','TOKENIZERS_PARALLELISM':'false'})
       .add_local_python_source('qwen_lab')
       .add_local_dir('data/http','/root/data/http',ignore=['budget.json'])
       .add_local_dir('data/neuron-roles','/root/data/neuron-roles',ignore=['budget.json']))

app=modal.App('qwen-neuron-roles')
@app.function(image=image,gpu='H200',volumes={'/models':cache,'/results':results},
              cpu=4,memory=98304,timeout=4800,max_containers=1,
              scaledown_window=2,retries=0,block_network=True)
def experiment(config:dict):
    # Results stay on the volume (block_network prevents blob returns); every stage commits.
    if config['phase']=='rollouts':
        from qwen_lab.sparsity_scale import run
        return run(config,Path('/results'),results.commit)
    from qwen_lab.neuron_roles import run
    return run(config,Path('/results'),results.commit)


def frozen(path,sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError(f'{path} differs from the frozen protocol')
    return json.loads(raw)


@app.local_entrypoint()
def main(phase:str='smoke',run_id:str='',seconds:int=600,job:str=''):
    P=json.loads(Path('data/neuron-roles/protocol.json').read_text());J=P['jobs']
    if phase not in J:raise ValueError(phase)
    if not run_id.startswith(J[phase]['run_id_prefix']) or not 1<=seconds<=J[phase]['soft_seconds']:raise ValueError('Bounds exceeded')
    F=P['frozen_inputs'];ref=frozen(F['neurons']['file'],F['neurons']['sha256'])
    if phase=='rollouts':
        D=P['option_d'];S=json.loads(Path(D['settings_from']).read_text());G=S['generation']
        neurons=[n for n in ref if f"L{n['layer']}/N{n['neuron']}" in D['neurons_off']]
        if len(neurons)!=len(D['neurons_off']):raise ValueError('Option D neuron missing')
        config=dict(phase=phase,run_id=run_id,arm=D['arm'],neurons=neurons,scenario=S['scenario'],samples=S['samples_per_arm'],chunk=S['chunk'],
                    seed=S['seed'],batch_size=G['batch_size'],max_new_tokens=G['max_new_tokens'],seconds=seconds)
    else:
        for e in F['http_case_files']+[F['roles_case_file']]:frozen(Path('data')/e['path'],e['sha256'])
        I=P['item1'];config=dict(phase=phase,run_id=run_id,seconds=seconds,refined=ref,http_case_files=F['http_case_files'],
                                 roles_case_file={k:F['roles_case_file'][k] for k in ('path','sha256')},escape_id=F['escape_id'],
                                 traces16=dict(run=F['traces16']['run']),seed=I['sampling']['seed'],sizes=I['sampling']['sizes'],
                                 wayback_seed=P['item3']['wayback']['seed'],wayback_samples=P['item3']['wayback']['samples_per_prefix'])
        if phase=='restore':
            if job not in I['jobs']:raise ValueError(job)
            config['restore']=I['jobs'][job]
    receipt=experiment.remote(config)
    folder=Path('runs')/run_id;folder.mkdir(parents=True,exist_ok=False)
    for name in ('result.json',)+(('rollouts.jsonl',) if phase=='rollouts' else ()):
        with (folder/name).open('wb') as f:
            for chunk in results.read_file(f'{run_id}/{name}'):f.write(chunk)
    out=json.loads((folder/'result.json').read_text())
    assert out['config']['run_id']==run_id and out['summary'].get('complete')==receipt['complete']
    print(json.dumps({'run':run_id,**{k:v for k,v in receipt.items() if k!='errors'},'errors':receipt.get('errors')}))
