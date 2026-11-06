"""Coverage sweep, item 7: the patch under batched vLLM, in bf16 and FP8.

Edits zero each ablated neuron's down_proj input column inside the vLLM worker. For FP8,
"after" zeroes the online-quantized FP8 column; "before" re-quantizes the edited bf16
checkpoint weight with the op vLLM uses at load and writes weight and scale, which is only
used if re-quantizing the unedited weight reproduces the loaded tensors bit for bit. Each
arm runs a teacher-forced probe so an edit that does not reach the forward pass is caught.
Calls stay text; results carry counts, probabilities and hostnames.
"""
import json,math,re,subprocess,sys,time
from pathlib import Path

MODEL='Qwen/Qwen3.8-27B';REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
HIDDEN,INTER=5120,17408
RATE=.001261+4*.0000131+96*.00000222


def checkpoint_path():
    return Path('/models/hf/hub')/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION


def find_down(layer,names):
    pattern=re.compile(rf'(?:^|\.)layers\.{layer}\.mlp\.down_proj\.weight$')
    found=[n for n in names if pattern.search(n) and 'visual' not in n]
    if len(found)!=1:raise ValueError(f'Expected one down_proj weight for layer {layer}, found {found}')
    return found[0]


def column(w,n):
    """The input column n of a down_proj weight stored as [hidden, intermediate] or its transpose."""
    if tuple(w.shape)==(HIDDEN,INTER):return w[:,n]
    if tuple(w.shape)==(INTER,HIDDEN):return w[n,:]
    raise ValueError(f'Unexpected down_proj shape {tuple(w.shape)}')


def as_bytes(t):
    import torch
    return t.view(torch.uint8) if t.element_size()==1 else t.view(torch.int16) if t.element_size()==2 else t.view(torch.int32)


def checkpoint_down(layer):
    from safetensors import safe_open
    root=checkpoint_path();index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    keys=[k for k in index if re.search(rf'(?:^|\.)layers\.{layer}\.mlp\.down_proj\.weight$',k) and 'visual' not in k]
    if len(keys)!=1:raise ValueError(f'Checkpoint key for layer {layer}: {keys}')
    with safe_open(str(root/index[keys[0]]),framework='pt',device='cpu') as f:return keys[0],f.get_tensor(keys[0])


def fp8_quantize(bf):
    """The op vLLM's online FP8 path applies to a bf16 weight at load (dynamic per-tensor scale)."""
    from vllm import _custom_ops as ops
    return ops.scaled_fp8_quant(bf.cuda(),scale=None)


E2M1=(0.,.5,1.,1.5,2.,3.,4.,6.)


def nvfp4_roundtrip(w,rows=1024):
    """NVFP4 weight rounding, dequantized back to w's dtype: E2M1 values in groups of 16 along the input
    dimension, E4M3 group scales and one FP32 global scale per tensor (the ModelOpt / compressed-tensors
    recipe). Exact ties (measure zero for real weights) round to the lower magnitude."""
    import torch
    grid=torch.tensor(E2M1,device=w.device,dtype=torch.float32);mids=(grid[1:]+grid[:-1])/2
    out_dim,in_dim=w.shape
    if in_dim%16:raise ValueError('Input dimension is not a multiple of 16')
    gscale=(448.*6.)/w.detach().abs().max().float().clamp(min=1e-12);out=torch.empty_like(w)
    for r in range(0,out_dim,rows):
        x=w[r:r+rows].float().reshape(-1,in_dim//16,16);amax=x.abs().amax(-1,keepdim=True)
        s=((amax/6.)*gscale).clamp(max=448.).to(torch.float8_e4m3fn).float()/gscale
        q=grid[torch.bucketize((x.abs()/s.clamp(min=1e-30)).clamp(max=6.),mids)]
        out[r:r+rows]=(q*x.sign()*s).reshape(-1,in_dim).to(w.dtype)
    return out


def same_layout(stored,q):
    """q (as quantized, [hidden, intermediate]) in the stored parameter's orientation."""
    return q if tuple(stored.shape)==tuple(q.shape) else q.t()


class Extension:
    """Named worker RPCs with plain data."""
    def lab_nvfp4(self):
        """Round every language-model linear weight to NVFP4 in place (weight-only emulation in the bf16 engine)."""
        import torch
        from vllm.model_executor.layers.linear import LinearBase
        done=[];skipped=[];count=0
        with torch.no_grad():
            for name,m in self.get_model().named_modules():
                w=getattr(m,'weight',None)
                if not isinstance(m,LinearBase) or w is None or 'visual' in name:continue
                # GEMM weights only: NVFP4 builds leave convolution kernels and odd shapes in 16-bit.
                if w.dtype!=torch.bfloat16 or w.ndim!=2 or w.shape[1]%16 or 'conv' in name:skipped.append([name,list(w.shape),str(w.dtype)]);continue
                w.copy_(nvfp4_roundtrip(w));done.append(name);count+=w.numel()
        return dict(modules=len(done),parameters=count,examples=done[:8],skipped=skipped[:40],skipped_count=len(skipped))

    def lab_linear_methods(self):
        out={}
        for name,m in self.get_model().named_modules():
            q=getattr(m,'quant_method',None)
            if q is not None and re.search(r'layers\.(0|63)\.(mlp\.down_proj|self_attn\.o_proj|linear_attn\.out_proj)$',name):
                scheme=getattr(m,'scheme',None)
                out[name]=dict(method=type(q).__name__,scheme=type(scheme).__name__ if scheme is not None else None,
                               params={k:[list(v.shape),str(v.dtype)] for k,v in m.named_parameters(recurse=False)})
        return out

    def lab_prepare(self,specs,precision):
        import torch
        model=self.get_model();params=dict(model.named_parameters());names=list(params);state={};receipts=[]
        for layer in sorted({s['layer'] for s in specs}):
            name=find_down(layer,names);w=params[name];scale=params.get(name[:-len('weight')]+'weight_scale')
            key,bf=checkpoint_down(layer)
            rec=dict(layer=layer,parameter=name,shape=list(w.shape),stride=list(w.stride()),dtype=str(w.dtype),checkpoint_key=key,
                     scale=None if scale is None else dict(shape=list(scale.shape),dtype=str(scale.dtype),value=scale.detach().float().flatten()[:4].tolist()))
            if precision=='bf16':
                rec['matches_checkpoint']=bool(torch.equal(same_layout(w,bf.to(w.device)).contiguous(),w.contiguous()))
            elif precision=='nvfp4':
                rec['matches_nvfp4_rounding']=bool(torch.equal(same_layout(w,nvfp4_roundtrip(bf.to(w.device))).contiguous(),w.contiguous()))
            else:
                q,s=fp8_quantize(bf);q=same_layout(w,q)
                rec['requantize_reproduces_weight']=bool(torch.equal(as_bytes(q.contiguous()),as_bytes(w.detach().contiguous())))
                rec['requantize_reproduces_scale']=scale is not None and bool(torch.equal(s.float().flatten(),scale.detach().float().flatten()))
            state[layer]=dict(weight=w,scale=scale,original=w.detach().clone(),original_scale=None if scale is None else scale.detach().clone())
            receipts.append(rec)
        self._lab=state;self._lab_precision=precision
        return receipts

    def lab_set_arm(self,off,mode):
        import torch
        S=self._lab;receipts=[];by_layer={}
        for s in off:by_layer.setdefault(s['layer'],[]).append(s['neuron'])
        with torch.no_grad():
            for e in S.values():
                e['weight'].copy_(e['original'])
                if e['scale'] is not None:e['scale'].copy_(e['original_scale'])
            for layer,ns in sorted(by_layer.items()):
                e=S[layer];w=e['weight']
                if mode=='before' and self._lab_precision=='nvfp4':
                    _,bf=checkpoint_down(layer);bf=bf.clone();bf[:,ns]=0;w.copy_(same_layout(w,nvfp4_roundtrip(bf.to(w.device))))
                elif mode=='before':
                    _,bf=checkpoint_down(layer);bf=bf.clone();bf[:,ns]=0;q,s=fp8_quantize(bf)
                    w.copy_(same_layout(w,q));e['scale'].copy_(s.reshape(e['scale'].shape))
                else:
                    for n in ns:as_bytes(column(w,n)).zero_()
                if any(torch.count_nonzero(as_bytes(column(w,n))).item() for n in ns):raise ValueError('Ablation failed')
                keep=torch.ones(INTER,dtype=torch.bool,device=w.device);keep[ns]=False
                diff=(as_bytes(w)!=as_bytes(e['original']));diff=diff[:,keep] if tuple(w.shape)==(HIDDEN,INTER) else diff[keep,:]
                receipts.append(dict(layer=layer,neurons=ns,mode=mode,changed_elements_outside_columns=int(diff.sum().item()),
                                     scale_changed=None if e['scale'] is None else not torch.equal(e['scale'],e['original_scale'])))
        return receipts


def clean_quant_config(path):
    """Drop quantization-config fields the installed compressed-tensors does not define (checkpoints from a
    newer release); the original config is kept as config.json.orig. Returns the removed paths."""
    from compressed_tensors.quantization import QuantizationArgs,QuantizationScheme
    p=Path(path)/'config.json';cfg=json.loads(p.read_text());q=cfg.get('quantization_config');removed=[]
    if not q:return removed
    args,scheme=set(QuantizationArgs.model_fields),set(QuantizationScheme.model_fields)
    for g,group in (q.get('config_groups') or {}).items():
        for k in [k for k in group if k not in scheme]:group.pop(k);removed.append(f'config_groups.{g}.{k}')
        for part in ('weights','input_activations','output_activations'):
            a=group.get(part)
            if isinstance(a,dict):
                for k in [k for k in a if k not in args]:a.pop(k);removed.append(f'config_groups.{g}.{part}.{k}')
    if removed:
        if not (Path(path)/'config.json.orig').exists():(Path(path)/'config.json.orig').write_text(p.read_text())
        p.write_text(json.dumps(cfg,indent=2))
    return removed


def prompt_score(output,prefix_len,target):
    rows=output.prompt_logprobs[prefix_len:prefix_len+len(target)]
    values=[float(row[t].logprob) for t,row in zip(target,rows)]
    return dict(logprob=sum(values),probability=math.exp(sum(values)),first=values[0])


def engine(config_path):
    """One engine in its own process: load, verify edits, then per arm a probe and the batched runs."""
    import torch,vllm
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    from qwen_lab.metrics import classify
    cfg=json.loads(Path(config_path).read_text());folder=Path(cfg['folder']);t0=time.monotonic()
    name=cfg.get('name',cfg['precision']);edits=cfg.get('edits',True)
    out=dict(precision=cfg['precision'],name=name,arms={},timing={},vllm=vllm.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    def save():(folder/f"engine-{name}.json").write_text(json.dumps(out))
    path=str(cfg.get('model_path') or checkpoint_path());tok=AutoTokenizer.from_pretrained(str(checkpoint_path()),local_files_only=True)
    if cfg.get('model_path'):
        removed=clean_quant_config(path)
        out['checkpoint']=dict(path=path,removed_config_fields=removed,quantization_config=json.loads((Path(path)/'config.json').read_text()).get('quantization_config'))
    kwargs=dict(model=path,tokenizer=path,dtype='bfloat16',tensor_parallel_size=1,max_model_len=4096,max_num_seqs=256,gpu_memory_utilization=.9,
                seed=0,language_model_only=True,worker_extension_cls='qwen_lab.coverage_vllm.Extension',disable_log_stats=True)
    if cfg['precision']=='fp8':kwargs['quantization']='fp8'
    out['engine_arguments']=kwargs;save()
    llm=LLM(**kwargs);out['timing']['engine_load']=round(time.monotonic()-t0,1);out['resolved_engine_config']=str(llm.llm_engine.vllm_config)[:4000]
    out['linear_methods']=llm.collective_rpc('lab_linear_methods')
    if cfg['precision']=='nvfp4':
        t=time.monotonic();out['nvfp4']=llm.collective_rpc('lab_nvfp4');out['timing']['nvfp4_rounding']=round(time.monotonic()-t,1);save()
    out['prepare']=llm.collective_rpc('lab_prepare',args=(cfg['neurons']['28'],cfg['precision'])) if edits else None;save()
    cases=cfg['cases'];url=cases['url'];n=cfg['samples']
    for arm in cfg['arms']:
        label,off,mode=arm['label'],cfg['neurons'][arm['neurons']],arm['mode'];t=time.monotonic()
        rec=llm.collective_rpc('lab_set_arm',args=(off,mode)) if edits else 'edited in the checkpoint'
        probe=llm.generate([{'prompt_token_ids':p['prefix_ids']+p['target_ids']} for p in cases['probes']],
                           SamplingParams(max_tokens=1,prompt_logprobs=1,temperature=1.,detokenize=False),use_tqdm=False)
        scores={p['id']:prompt_score(o,len(p['prefix_ids']),p['target_ids']) for p,o in zip(cases['probes'],probe)}
        params=[SamplingParams(n=1,temperature=1.,top_p=.95,top_k=20,min_p=0.,repetition_penalty=1.,max_tokens=cfg['max_new_tokens'],
                               seed=cfg['seed']+i,skip_special_tokens=False,spaces_between_special_tokens=False) for i in range(n)]
        g=time.monotonic();outs=llm.generate([{'prompt_token_ids':cases['prompt_ids']}]*n,params,use_tqdm=False);gen_seconds=time.monotonic()-g
        rows=[]
        with (folder/'rollouts.jsonl').open('a') as f:
            for i,o in enumerate(outs):
                c=o.outputs[0];ids=list(c.token_ids);text=tok.decode(ids,skip_special_tokens=False)
                m=classify(text,url,truncated=c.finish_reason=='length')
                row=dict(arm=label,precision=cfg['precision'],engine=name,batch_seed=cfg['seed']+i,batch_row=0,token_ids=ids,text=text,finish_reason=c.finish_reason,metrics=m)
                f.write(json.dumps(row)+'\n');rows.append(m)
        out['arms'][label]=dict(neurons=arm['neurons'],mode=mode,receipts=rec,probe=scores,n=len(rows),proxy=sum(r['proxy_call'] for r in rows),
                                expected_host=sum(r['expected_host_call'] for r in rows),truncated=sum(r['truncated'] for r in rows),
                                generation_seconds=round(gen_seconds,1),output_tokens=sum(len(o.outputs[0].token_ids) for o in outs),seconds=round(time.monotonic()-t,1))
        save();print(json.dumps(dict(arm=label,proxy=out['arms'][label]['proxy'],n=n,seconds=out['arms'][label]['seconds'])),flush=True)
    out['complete']=True;save()


def run(config,root,commit):
    """Modal entry: each precision runs in its own engine subprocess; the parent always saves."""
    from qwen_lab.http_path import redact
    started=time.monotonic();run_id=config['run_id']
    if not re.fullmatch(r'coverage-vllm-[a-z0-9-]+',run_id):raise ValueError('Invalid run_id')
    folder=Path(root)/run_id;folder.mkdir(parents=True,exist_ok=False)
    out=dict(engines={},errors={})
    def save():
        elapsed=time.monotonic()-started
        (folder/'result.json').write_text(redact(json.dumps(dict(summary=out,config={k:v for k,v in config.items() if k not in ('cases','neurons')},
            elapsed_seconds=elapsed,estimated_compute_usd=elapsed*RATE))));commit()
    for spec in config['engines']:
        name=spec.get('name',spec['precision']);left=config['seconds']-(time.monotonic()-started)
        if left<120:out['errors'][name]='deadline';continue
        path=folder/f"engine-{name}-config.json"
        path.write_text(json.dumps(dict(spec,folder=str(folder),cases=config['cases'],neurons=config['neurons'],samples=spec['samples'],
                                        seed=config['seed'],max_new_tokens=config['max_new_tokens'])))
        try:
            p=subprocess.run([sys.executable,'-m','qwen_lab.coverage_vllm','engine',str(path)],timeout=left,capture_output=True,text=True)
            if p.returncode:out['errors'][name]=redact(p.stderr[-3000:])
        except subprocess.TimeoutExpired:out['errors'][name]='timeout'
        result=folder/f"engine-{name}.json"
        if result.exists():out['engines'][name]=json.loads(result.read_text())
        path.unlink();save()
        if name in out['errors'] and spec.get('stop_on_error'):break
    out['complete']=not out['errors'] and all(e.get('complete') for e in out['engines'].values());save()
    return dict(run_id=run_id,complete=out['complete'],elapsed=time.monotonic()-started,estimated_compute_usd=(time.monotonic()-started)*RATE,errors=out['errors'])


if __name__=='__main__' and sys.argv[1:2]==['engine']:engine(sys.argv[2])
