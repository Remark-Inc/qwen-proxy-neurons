from types import SimpleNamespace
import torch
from qwen_lab.scoring import continuation_logprob
from qwen_lab.prefixes import paired_tokens,first_distinguishing

class MarkovLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transitions=torch.nn.Parameter(torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.],
                                                         [0.,1.,2.,4.],[2.,3.,4.,1.]]))
    def forward(self,input_ids,use_cache,logits_to_keep):
        return SimpleNamespace(logits=self.transitions[input_ids][:,-logits_to_keep:])

def test_teacher_forcing_has_correct_shift_and_no_prefix_score():
    m=MarkovLM()
    with torch.inference_mode():
        got=continuation_logprob(m,[0,1],[2,3])
        expected=m.transitions.log_softmax(-1)
        assert abs(got['logprob']-float(expected[1,2]+expected[2,3]))<1e-6
        assert got['tokens']==2
        assert abs(got['per_token'][0]-float(expected[1,2]))<1e-6

class BoundaryTokenizer:
    def encode(self,s,add_special_tokens=False):
        ids=[]
        while s:
            if s.startswith('xa'):
                ids.append(1000);s=s[2:]
            else:
                ids.append(ord(s[0]));s=s[1:]
        return ids
    def decode(self,ids,skip_special_tokens=False):
        return ''.join('xa' if x==1000 else chr(x) for x in ids)

def test_shared_prefix_backs_up_across_bpe_boundary():
    t=BoundaryTokenizer()
    c=paired_tokens(t,'zx','abc','bcd')
    assert c['effective_prefix']=='z'
    assert c['prefix_boundary_backoff_chars']==1
    assert t.decode(c['prefix_ids']+c['bad_ids'])=='zxabc'
    assert t.decode(c['prefix_ids']+c['good_ids'])=='zxbcd'

def test_first_token_screen_does_not_reuse_common_tokens():
    c=dict(id='test',prefix_ids=[1,2],bad_ids=[3,4,5],good_ids=[3,9,8])
    ex=first_distinguishing(c)
    assert ex['prefix_ids']==[1,2,3]
    assert (ex['bad'],ex['good'])==(4,9)
