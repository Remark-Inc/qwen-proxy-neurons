from types import SimpleNamespace
import torch
from qwen_lab.explanation_hooks import phase_mask,scoped_intervention,capture

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.proj=torch.nn.Linear(3,3,bias=False)
        with torch.no_grad():self.proj.weight.copy_(torch.eye(3))
        self.model=SimpleNamespace(language_model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=self.proj))]))
    def forward(self,input_ids):return self.proj(input_ids.float()[...,None].expand(-1,-1,3))


def test_static_interval_and_nested_rescue():
    model=Toy();ids=torch.tensor([[1,2,3,4]])
    specs=[dict(layer=0,neuron=1,value=0)]
    with scoped_intervention(model,specs,start=1,end=3):
        out=model(input_ids=ids)
        assert out[0,:,1].tolist()==[1,0,0,4]
        assert out[0,:,0].tolist()==[1,2,3,4]
        with scoped_intervention(model,[dict(layer=0,neuron=1,mode='set',value=-7)],start=2,end=3):
            assert model(input_ids=ids)[0,:,1].tolist()==[1,0,-7,4]
    assert model(input_ids=ids)[0,:,1].tolist()==[1,2,3,4]
    assert not model._forward_pre_hooks and not model.proj._forward_pre_hooks


def test_phase_boundaries_do_not_look_ahead():
    markers=dict(end_think=[8,9],open_call=[10],close_call=[11])
    ids=[1,2,3,8,9,4,10,5,11,6]
    assert phase_mask(ids,2,markers,'reasoning')==[False,False,True,True,False,False,False,False,False,False]
    assert phase_mask(ids,2,markers,'call')==[False]*6+[True,True,False,False]


def test_dynamic_masks_independent_batch_and_reset():
    model=Toy();specs=[dict(layer=0,neuron=1,value=0)]
    dynamic=dict(prompt_length=2,markers=dict(end_think=[8],open_call=[9],close_call=[10]),scope='call')
    with scoped_intervention(model,specs,dynamic=dynamic) as state:
        assert model(input_ids=torch.tensor([[1,2],[1,2]]))[0,:,1].tolist()==[1,2]
        out=model(input_ids=torch.tensor([[9],[3]]))
        assert out[:,:,1].flatten().tolist()==[0,3]
        out=model(input_ids=torch.tensor([[4],[9]]))
        assert out[:,:,1].flatten().tolist()==[0,0]
        out=model(input_ids=torch.tensor([[10],[4]]))
        assert out[:,:,1].flatten().tolist()==[10,0]
        out=model(input_ids=torch.tensor([[1,2],[1,2]]))
        assert out[:,:,1].tolist()==[[1,2],[1,2]]
        assert state['forward_calls']==5
        assert state['selected_predictor_positions']==4
        assert state['neuron_coordinate_applications']==4


def test_capture_signed_values_and_hook_cleanup_on_exception():
    model=Toy();values={}
    try:
        with capture(model,[dict(layer=0,neuron=1)],values):
            model(input_ids=torch.tensor([[-2,3]]));raise RuntimeError('test')
    except RuntimeError:pass
    assert values=={'L0/N1':[-2,3]}
    assert not model.proj._forward_pre_hooks


def test_weight_surgery_column_zeroing_matches_hook():
    from qwen_lab.interventions import intervene
    model=Toy();ids=torch.tensor([[2,5,7]])
    specs=[dict(layer=0,neuron=1,mode='scale',value=0)]
    with intervene(model,specs):
        hook_out=model(input_ids=ids).clone()
    saved=model.proj.weight[:,1].clone()
    try:
        with torch.no_grad():
            model.proj.weight[:,1].zero_()
        weight_out=model(input_ids=ids).clone()
    finally:
        with torch.no_grad():
            model.proj.weight[:,1].copy_(saved)
    assert torch.equal(hook_out,weight_out)
