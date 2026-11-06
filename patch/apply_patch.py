"""Zero the 27 patch neurons in Qwen3.8-27B and optionally save a drop-in patched checkpoint.

    python patch/apply_patch.py --save /path/to/patched-checkpoint

Each neuron (layer, n) is removed by zeroing column n of that layer's MLP down-projection, so the
neuron's activation can no longer reach the residual stream. `down_proj` has no bias, so the edit
is exact. This is the edit validated in reports/patch-27-results.md and
reports/coverage-sweep-results.md; it is not a general fix for every fetch fallback (see README).

Loading needs about 54 GB of memory in bf16 (`--device-map auto` spreads it over available GPUs
and CPU). The checkpoint is `Qwen3_5ForConditionalGeneration`; under `AutoModelForCausalLM` the
text-only class is returned and the layers sit at `model.model.layers` instead, which `apply`
also handles.
"""
import argparse,json,shutil
from pathlib import Path

NEURONS=json.loads((Path(__file__).with_name('neurons-27.json')).read_text())


def language_layers(model):
    inner=model.model
    return getattr(inner,'language_model',inner).layers


def apply(model,neurons=None):
    import torch
    layers=language_layers(model);neurons=neurons or NEURONS['neurons']
    with torch.no_grad():
        for n in neurons:
            layers[n['layer']].mlp.down_proj.weight[:,n['neuron']].zero_()
    return len(neurons)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model',default=NEURONS['model']);ap.add_argument('--revision',default=NEURONS['revision'])
    ap.add_argument('--device-map',default='auto');ap.add_argument('--save',help='directory to write the patched checkpoint to')
    a=ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import Qwen3_5ForConditionalGeneration
    src=Path(snapshot_download(a.model,revision=a.revision))
    model=Qwen3_5ForConditionalGeneration.from_pretrained(src,dtype=torch.bfloat16,device_map=a.device_map)
    print(f'zeroed {apply(model)} down_proj columns')
    if a.save:
        dest=Path(a.save);model.save_pretrained(dest)
        # Everything that is not a weight shard (tokenizer, chat template, generation config, ...)
        # is copied as is, so the result loads like the original.
        for f in src.iterdir():
            if f.is_file() and f.suffix!='.safetensors' and f.name!='model.safetensors.index.json' and not (dest/f.name).exists():
                shutil.copy2(f,dest/f.name)
        print(f'saved to {dest}')


if __name__=='__main__':main()
