"""Ledger-budgeted http:// path phases; each invocation hard-capped at 90 minutes."""
import hashlib,json
from pathlib import Path
import modal
cache=modal.Volume.from_name('qwen-neuron-lab-models')
results=modal.Volume.from_name('qwen-neuron-lab-results')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.14.1','transformers==5.21.0','accelerate==1.15.0',
                    'huggingface-hub==2.3.0','numpy==2.5.4')
       .env({'HF_HOME':'/models/hf','TOKENIZERS_PARALLELISM':'false'})
       .add_local_python_source('qwen_lab')
       # The ledger changes while other launchers run; containers only read case files.
       .add_local_dir('data/http','/root/data/http',ignore=['budget.json']))

app=modal.App('qwen-neuron-http-path')
@app.function(image=image,gpu='H200',volumes={'/models':cache,'/results':results},
              cpu=4,memory=98304,timeout=5400,max_containers=1,
              scaledown_window=2,retries=0,block_network=True)
def experiment(config:dict):
    from qwen_lab.http_path import run
    out=run(config,Path('/results'));results.commit()
    # Large results stay on the volume (block_network prevents blob returns).
    return dict(run_id=config['run_id'],complete=out['summary'].get('complete'),
                elapsed=out['elapsed_seconds'],estimated_compute_usd=out['estimated_compute_usd'])

@app.local_entrypoint()
def main(phase:str='smoke',run_id:str='http-smoke-1',seconds:int=900,arm:str='',suite:str='',scope:str='',samples:int=0,suite_runs:str='',suite_runs_old:str='',stages:str='',final:bool=False):
    if phase not in {'smoke','discover','validate','suite','replay','specificity','refine','validate_refined','jina'}:raise ValueError(phase)
    if not 1<=seconds<=4800:raise ValueError('Bounds exceeded')
    protocol=json.loads(Path('data/http/protocol.json').read_text())
    files=[dict(path='cases.json',sha256=protocol['case_sha256'])]
    if phase in {'validate','refine','validate_refined'}:
        files.append(dict(path='discovery-harvest.json',sha256=protocol['discovery_harvest_sha256']))
    if phase in {'refine','validate_refined'}:
        files.append(dict(path='refine-cases.json',sha256=protocol['round3']['refine_case_sha256']))
    for f in files:
        if hashlib.sha256(Path('data/http',f['path']).read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Case file differs from frozen protocol')
    ns=json.loads(Path('data/candidate-zero-proxy.json').read_text())
    config=dict(phase=phase,run_id=run_id,seconds=seconds,neurons=ns,case_files=files)
    if phase=='smoke':config['protocol']=protocol
    if phase=='discover':config['protocol']=protocol['discovery']
    if phase in {'validate','replay','specificity','refine','validate_refined'} or (phase=='suite' and arm in {'combined','refined','final27'}):
        combined=json.loads(Path(protocol['combined_file']).read_text())
        if hashlib.sha256(Path(protocol['combined_file']).read_bytes()).hexdigest()!=protocol['combined_sha256']:raise ValueError('Combined set differs from frozen hash')
        config.update(combined=combined)
    if phase in {'validate','suite','replay','specificity','validate_refined'}:
        config.update(protocol=dict(protocol['validation'],suite_max_new=protocol['suites']['max_new_tokens']))
    if phase=='refine':config.update(protocol=protocol['round3']['discovery'],start_neurons=config['combined'])
    if phase in {'validate_refined'} or (phase=='suite' and arm in {'refined','final27'}):
        r3=protocol['round3'];refined=json.loads(Path(r3['refined_file']).read_text())
        if hashlib.sha256(Path(r3['refined_file']).read_bytes()).hexdigest()!=r3['refined_sha256']:raise ValueError('Refined set differs from frozen hash')
        config.update(refined=refined)
    if phase=='validate_refined':
        sizes=protocol['round3']['validation_sizes'];config['protocol']['refined_sizes']={k:v for k,v in sizes.items() if k!='note'}
        if final:
            drop=lambda ns_:[n for n in ns_ if (n['layer'],n['neuron'])!=(56,16758)]
            config.update(final=drop(config['refined']),final11=drop(ns))
        config.update(escape_id=protocol['escape_case_id'],suite_runs_new=[x for x in suite_runs.split(',') if x],
                      suite_runs_old=[x for x in suite_runs_old.split(',') if x],stages=[x for x in stages.split(',') if x])
    if phase=='validate':config.update(escape_id=protocol['escape_case_id'],reference_sets={
        'all12_plus_first_followup_pair':ns+[dict(layer=51,neuron=8299,mode='scale',value=0),dict(layer=61,neuron=5269,mode='scale',value=0)]})
    if phase=='replay':config['suite_runs']=[x for x in suite_runs.split(',') if x]
    if phase=='jina':
        r3=protocol['round3'];refined=json.loads(Path(r3['refined_file']).read_text())
        if hashlib.sha256(Path(r3['refined_file']).read_bytes()).hexdigest()!=r3['refined_sha256']:raise ValueError('Refined set differs from frozen hash')
        drop=lambda ns_:[n for n in ns_ if (n['layer'],n['neuron'])!=(56,16758)]
        config.update(refined=refined,all27=drop(refined),all11=drop(ns),protocol=dict(protocol['validation']))
    if phase=='suite':
        drop=lambda ns_:[n for n in ns_ if (n['layer'],n['neuron'])!=(56,16758)]
        arms=dict(unpatched=[],all12=ns,combined=config.get('combined'),refined=config.get('refined'),final27=drop(config.get('refined') or []))
        spec=protocol['suites'][suite]
        if scope and scope not in {'call','reasoning'}:raise ValueError(scope)
        specs=spec['scenarios'] if not samples else [dict(s,samples=samples) for s in spec['scenarios'][:1]]
        config.update(arm=arm,arm_neurons=arms[arm],suite=specs,seed=spec['seed'],scope=scope or None)
    receipt=experiment.remote(config)
    folder=Path('runs')/run_id;folder.mkdir(parents=True,exist_ok=False)
    with (folder/'result.json').open('wb') as f:
        for chunk in results.read_file(run_id+'/result.json'):f.write(chunk)
    out=json.loads((folder/'result.json').read_text())
    assert out['config']['run_id']==run_id and out['summary'].get('complete')==receipt['complete']
    print(json.dumps({'run':run_id,**receipt}))
