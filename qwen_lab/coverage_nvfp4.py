"""Coverage sweep, item 7c on Blackwell: real NVFP4 checkpoints served natively by vLLM on a B200.

Checkpoints are built here, not downloaded: llm-compressor's NVFP4 scheme (FP4 weights and
activations, groups of 16, E4M3 group scales, FP32 global scales) with the ignore list of
llm-compressor's Qwen3.5 recipe (lm_head, the vision tower, the Gated DeltaNet layers), calibrated
on our own unmodified transcripts. Variants: unedited; the 28 and the 27 zeroed after quantization
(FP4 codes of the neurons' down_proj columns set to 0 with compressed-tensors' own pack/unpack,
scales kept); the 28 zeroed before quantization (bf16 columns zeroed, then calibrated and quantized).
"""
import hashlib,json,re,shutil,sys,time
from pathlib import Path

from qwen_lab import MODEL,REVISION
IGNORE=['lm_head','re:.*visual.*','re:.*linear_attn.*']
QUANT_RATE=.001261+4*.0000131+192*.00000222
B200_RATE=.001736+4*.0000131+96*.00000222
CPU_RATE=4*.0000131+64*.00000222
ROOT=Path('/models/nvfp4')


def snapshot():
    return Path('/models/hf/hub')/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION


def calibration_rows(run_dirs,n=256,max_tokens=2048):
    """Unmodified rollouts (no neurons switched off), evenly spread over scenarios, with no proxy host or
    credential in the text: token ids of prompt plus generation, cut to max_tokens."""
    import collections
    from qwen_lab.http_path import SECRET,iter_rollouts
    from qwen_lab.metrics import PROXY
    by=collections.defaultdict(list)
    for d in run_dirs:
        d=Path(d)
        docs=[json.loads(l) for l in (d/'rollouts.jsonl').read_text().splitlines()] if (d/'rollouts.jsonl').exists() else \
             [r for _,r in iter_rollouts(json.loads((d/'result.json').read_text()))] if (d/'result.json').exists() else []
        for r in docs:
            if r.get('neurons') or r.get('conditional_prefill') or 'prompt' not in r:continue
            if PROXY in r['prompt']+r['text'] or SECRET.search(r['prompt']+r['text']):continue
            by[r['scenario']].append((d.name,r))
    keys=sorted(by);out=[];i=0
    while len(out)<n and any(by[k] for k in keys):
        k=keys[i%len(keys)];i+=1
        if by[k]:out.append(by[k].pop(0))
    return out


def build_calibration(tok,run_dirs,dest,n=256,max_tokens=2048):
    rows=calibration_rows(run_dirs,n,max_tokens)
    seqs=[(tok.encode(r['prompt'],add_special_tokens=False)+r['token_ids'])[:max_tokens] for _,r in rows]
    meta=dict(samples=len(seqs),max_tokens=max_tokens,tokens=sum(map(len,seqs)),
              scenarios=sorted({r['scenario'] for _,r in rows}),runs=sorted({n for n,_ in rows}))
    Path(dest).write_text(json.dumps(dict(meta=meta,input_ids=seqs))+'\n');return meta


# ---------------------------------------------------------------- quantization job (Transformers + llm-compressor)

def fp4_codec():
    """compressed-tensors' FP4 pack and unpack functions (the module moved between releases)."""
    import importlib,pkgutil
    known=('compressed_tensors.compressors.nvfp4.helpers','compressed_tensors.compressors.quantized_compressors.fp4_quantized',
           'compressed_tensors.compressors.quantized_compressors.nvfp4_quantized','compressed_tensors.compressors.quantized_compressors')
    import compressed_tensors
    walked=[m.name for m in pkgutil.walk_packages(compressed_tensors.__path__,'compressed_tensors.')]
    for mod in list(known)+walked:
        try:
            m=importlib.import_module(mod)
            if hasattr(m,'pack_fp4_to_uint8') and hasattr(m,'unpack_fp4_from_uint8'):return m.pack_fp4_to_uint8,m.unpack_fp4_from_uint8
        except Exception:continue
    raise ImportError('compressed-tensors FP4 pack/unpack not found')


def zero_fp4_columns(packed,columns):
    """Set the FP4 codes of input columns to 0 in a [out, in/2] packed tensor, using compressed-tensors' own
    unpack and pack (so the nibble order is the library's)."""
    import torch
    pack_fp4_to_uint8,unpack_fp4_from_uint8=fp4_codec()
    out_dim,half=packed.shape;vals=unpack_fp4_from_uint8(packed,out_dim,half*2,dtype=torch.float32)
    if not torch.equal(pack_fp4_to_uint8(vals),packed):raise ValueError('FP4 pack/unpack does not round-trip')
    vals[:,columns]=0;new=pack_fp4_to_uint8(vals)
    check=unpack_fp4_from_uint8(new,out_dim,half*2,dtype=torch.float32);keep=torch.ones(half*2,dtype=torch.bool);keep[columns]=False
    if check[:,columns].abs().sum()!=0 or not torch.equal(check[:,keep],unpack_fp4_from_uint8(packed,out_dim,half*2,dtype=torch.float32)[:,keep]):
        raise ValueError('Column edit changed other weights')
    return new


def copy_with_edits(src,dest,edits):
    """A full copy of a compressed checkpoint with the listed down_proj columns' FP4 codes zeroed.
    edits: layer -> columns. Returns receipts."""
    from safetensors.torch import load_file,save_file
    # The destination may hold a partial copy from an interrupted attempt: files already there at the
    # source's size are kept, everything else is (re)written.
    src,dest=Path(src),Path(dest);dest.mkdir(parents=True,exist_ok=True)
    index=json.loads((src/'model.safetensors.index.json').read_text())['weight_map'];receipts=[]
    keys={L:[k for k in index if re.search(rf'(?:^|\.)layers\.{L}\.mlp\.down_proj\.weight_packed$',k) and 'visual' not in k] for L in edits}
    for L,ks in keys.items():
        if len(ks)!=1:raise ValueError(f'Packed down_proj key for layer {L}: {ks}')
    touched={index[ks[0]] for ks in keys.values()};linked=[]
    kept=[]
    for f in src.iterdir():
        if f.name in touched:continue
        d=dest/f.name
        if d.exists() and d.stat().st_size==f.stat().st_size:kept.append(f.name);continue
        # Unchanged shards are symlinked where the volume allows it, otherwise copied.
        try:
            if f.suffix!='.safetensors':raise OSError
            d.symlink_to(f.resolve());linked.append(f.name)
        except OSError:shutil.copy2(f,d)
    for name in sorted(touched):
        tensors=load_file(str(src/name))
        for L,ks in keys.items():
            if index[ks[0]]!=name:continue
            before=tensors[ks[0]];tensors[ks[0]]=zero_fp4_columns(before,edits[L])
            receipts.append(dict(layer=L,key=ks[0],columns=edits[L],shape=list(before.shape),
                                 changed_bytes=int((tensors[ks[0]]!=before).sum()),sha256=hashlib.sha256(tensors[ks[0]].numpy().tobytes()).hexdigest()))
        save_file(tensors,str(dest/name),metadata={'format':'pt'})
    return dict(edits=receipts,rewritten_shards=sorted(touched),linked_shards=len(linked),kept_from_earlier_attempt=len(kept))


def quantize(config,root,commit):
    """H200 job: build the NVFP4 variants on the models volume, then record receipts."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    started=time.monotonic();run_id=config['run_id'];folder=Path(root)/run_id;folder.mkdir(parents=True,exist_ok=False)
    out=dict(variants={},timing={},errors={});cal=json.loads((Path('/root/data')/config['calibration']['path']).read_text())
    def save():
        (folder/'result.json').write_text(json.dumps(dict(summary=out,config={k:v for k,v in config.items() if k!='neurons'},
            elapsed_seconds=time.monotonic()-started,estimated_compute_usd=(time.monotonic()-started)*QUANT_RATE)));commit()
    def step(name,fn):
        t=time.monotonic()
        try:out[name]=fn()
        except Exception as e:out['errors'][name]=repr(e)[-3000:]
        out['timing'][name]=round(time.monotonic()-t,1);save();print(json.dumps({'step':name,'seconds':out['timing'][name],'error':out['errors'].get(name)}),flush=True)
        return name not in out['errors']
    src=snapshot();tok=AutoTokenizer.from_pretrained(str(src),local_files_only=True)
    ds=Dataset.from_dict(dict(input_ids=cal['input_ids'],attention_mask=[[1]*len(x) for x in cal['input_ids']]))
    def load(zero=None):
        from qwen_lab.interventions import language_layers
        m=Qwen3_5ForConditionalGeneration.from_pretrained(str(src),dtype=torch.bfloat16,local_files_only=True);layers=language_layers(m)
        if zero:
            with torch.no_grad():
                for n in zero:layers[n['layer']].mlp.down_proj.weight[:,n['neuron']]=0
        return m
    base=ROOT/run_id
    def build(label,zero=None):
        dest=base/label;m=load(zero)
        recipe=QuantizationModifier(targets='Linear',scheme='NVFP4',ignore=IGNORE)
        oneshot(model=m,processor=tok,dataset=ds,recipe=recipe,num_calibration_samples=len(cal['input_ids']),max_seq_length=cal['meta']['max_tokens'])
        m.save_pretrained(str(dest),save_compressed=True);tok.save_pretrained(str(dest))
        for f in src.iterdir():
            if f.suffix in ('.json','.jinja','.txt') and f.name not in ('config.json','model.safetensors.index.json') and not (dest/f.name).exists():shutil.copy2(f,dest/f.name)
        q=json.loads((dest/'config.json').read_text()).get('quantization_config');del m;torch.cuda.empty_cache()
        return dict(path=str(dest),quantization_config=q,files=sorted(p.name for p in dest.iterdir()),bytes=sum(p.stat().st_size for p in dest.iterdir()))
    edits=lambda specs:{L:[n['neuron'] for n in specs if n['layer']==L] for L in sorted({n['layer'] for n in specs})}
    if step('unedited',lambda:build(config['labels']['unedited'])):
        step('28_after',lambda:dict(path=str(base/config['labels']['28_after']),receipts=copy_with_edits(base/config['labels']['unedited'],base/config['labels']['28_after'],edits(config['neurons']['28']))))
        step('27_after',lambda:dict(path=str(base/config['labels']['27_after']),receipts=copy_with_edits(base/config['labels']['unedited'],base/config['labels']['27_after'],edits(config['neurons']['27']))))
    step('28_before',lambda:build(config['labels']['28_before'],zero=config['neurons']['28']))
    from qwen_lab.coverage_nvfp4 import scale_receipts
    step('scales',lambda:scale_receipts(base/config['labels']['unedited'],base/config['labels']['28_before'],sorted({n['layer'] for n in config['neurons']['28']})))
    out['complete']=not out['errors'];save()
    return dict(run_id=run_id,complete=out['complete'],elapsed=time.monotonic()-started,estimated_compute_usd=(time.monotonic()-started)*QUANT_RATE,errors=out['errors'])


def edit_after(config,root,commit):
    """CPU job: the after-quantization variants as copies of the unedited checkpoint with FP4 codes zeroed."""
    started=time.monotonic();run_id=config['run_id'];folder=Path(root)/run_id;folder.mkdir(parents=True,exist_ok=False);out=dict(errors={},timing={})
    base=ROOT/config['quantize_run'];edits=lambda specs:{L:[n['neuron'] for n in specs if n['layer']==L] for L in sorted({n['layer'] for n in specs})}
    for key,arm in (('28_after','28'),('27_after','27')):
        t=time.monotonic()
        try:out[key]=dict(path=str(base/config['labels'][key]),**copy_with_edits(base/config['labels']['unedited'],base/config['labels'][key],edits(config['neurons'][arm])))
        except Exception as e:out['errors'][key]=repr(e)[-3000:]
        out['timing'][key]=round(time.monotonic()-t,1)
        (folder/'result.json').write_text(json.dumps(dict(summary=out,elapsed_seconds=time.monotonic()-started)));commit()
        print(json.dumps({'step':key,'seconds':out['timing'][key],'error':out['errors'].get(key)}),flush=True)
    out['complete']=not out['errors'];(folder/'result.json').write_text(json.dumps(dict(summary=out,elapsed_seconds=time.monotonic()-started,
        estimated_compute_usd=(time.monotonic()-started)*CPU_RATE)));commit()
    return dict(run_id=run_id,complete=out['complete'],elapsed=time.monotonic()-started,estimated_compute_usd=(time.monotonic()-started)*CPU_RATE,errors=out['errors'])


def scale_receipts(a,b,layers):
    """Global and group scales of the edited layers' down_proj, unedited against quantized-after-editing."""
    import torch
    from safetensors import safe_open
    def tensors(d,L):
        index=json.loads((Path(d)/'model.safetensors.index.json').read_text())['weight_map'];out={}
        for suffix in ('weight_global_scale','weight_scale','input_global_scale','weight_packed'):
            k=[x for x in index if re.search(rf'(?:^|\.)layers\.{L}\.mlp\.down_proj\.{suffix}$',x) and 'visual' not in x]
            if k:
                with safe_open(str(Path(d)/index[k[0]]),framework='pt') as f:out[suffix]=f.get_tensor(k[0])
        return out
    rows=[]
    for L in layers:
        x,y=tensors(a,L),tensors(b,L)
        rows.append(dict(layer=L,weight_global_scale=[float(x['weight_global_scale'].float().flatten()[0]),float(y['weight_global_scale'].float().flatten()[0])],
                         input_global_scale=[float(x['input_global_scale'].float().flatten()[0]),float(y['input_global_scale'].float().flatten()[0])] if 'input_global_scale' in x else None,
                         group_scales_changed=int((x['weight_scale'].view(torch.uint8)!=y['weight_scale'].view(torch.uint8)).sum()),
                         packed_bytes_changed=int((x['weight_packed']!=y['weight_packed']).sum())))
    return rows


if __name__=='__main__' and sys.argv[1:2]==['calibration']:
    from transformers import AutoTokenizer
    tokenizer_dir,dest=sys.argv[2:4];runs=sys.argv[4:]
    print(json.dumps(build_calibration(AutoTokenizer.from_pretrained(tokenizer_dir),runs,dest)))
