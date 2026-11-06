import json

import torch

from qwen_lab import coverage_nvfp4 as cn
from qwen_lab.metrics import PROXY

GRID=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])


def codec():
    """A reference E2M1 codec with the even column in the low nibble."""
    def code(v):
        mag=(v.abs().unsqueeze(-1)==GRID).float().argmax(-1);return (mag|((v<0).long()<<3)).to(torch.uint8)
    def pack(x):c=code(x);return c[:,0::2]|(c[:,1::2]<<4)
    def unpack(a,m,n,dtype=torch.float32):
        lo,hi=a&0x0F,a>>4;c=torch.stack([lo,hi],-1).reshape(m,n).long()
        return (GRID[c&7]*torch.where(c>=8,-1.,1.)).to(dtype)
    return pack,unpack


def test_zero_fp4_columns_changes_only_those_columns(monkeypatch):
    monkeypatch.setattr(cn,'fp4_codec',codec)
    torch.manual_seed(0);x=GRID[torch.randint(1,8,(6,32))]*torch.where(torch.rand(6,32)<.5,-1.,1.)
    pack,unpack=codec();packed=pack(x);new=cn.zero_fp4_columns(packed,[3,10])
    y=unpack(new,6,32)
    assert y[:,[3,10]].abs().sum()==0 and torch.equal(y[:,[0,1,2,4,11,31]],x[:,[0,1,2,4,11,31]])


def test_b200_jobs_are_frozen():
    P=json.loads(open('data/coverage-sweep/protocol.json').read());J=P['jobs']
    q,b=J['nvfp4-quantize'],J['vllm-b200-nvfp4']
    assert set(q['labels'])=={'unedited','28_after','27_after','28_before'} and q['rate_per_second']>.0015
    assert [e['name'] for e in b['engines']]==['nvfp4-unedited','nvfp4-28-after','nvfp4-27-after','nvfp4-28-before']
    assert all(e['model_path'].startswith('/models/nvfp4/'+b['quantize_run']+'/') and not e['edits'] for e in b['engines'])
    assert b['engines'][0]['stop_on_error'] and abs(b['rate_per_second']-(.001736+4*.0000131+96*.00000222))<1e-12
    cal=json.loads(open('data/'+P['frozen_inputs']['nvfp4_calibration']['path']).read())
    assert cal['meta']['samples']==256 and max(map(len,cal['input_ids']))<=2048
