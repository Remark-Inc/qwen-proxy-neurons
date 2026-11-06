"""True neuron intervention at the input of down_proj, before neurons mix."""
from contextlib import contextmanager
from collections import defaultdict

def protect_recurrent_cache_for_grad(cache):
    """Isolate values saved for backward from Transformers' cache copy_ updates.

    The one-token forward and subsequent cache contents are unchanged. A clone
    prevents an update from overwriting the prior recurrent state autograd saved.
    """
    read = cache.get_recurrent_state
    def cloned_state(*args, **kwargs):
        state = read(*args, **kwargs)
        return None if state is None else state.clone()
    cache.get_recurrent_state = cloned_state
    return cache

def language_layers(model):
    layers = model.model.language_model.layers
    for layer in layers:
        if not hasattr(layer.mlp, "down_proj"):
            raise ValueError("Dense MLP required; MoE needs expert-specific hooks")
    return layers

@contextmanager
def intervene(model, neurons):
    if len(neurons) > 32:
        raise ValueError("Interventions are limited to 32 neurons")
    layers = language_layers(model)
    grouped = defaultdict(list)
    seen = set()
    for spec in neurons:
        l, n = int(spec["layer"]), int(spec["neuron"])
        if (l,n) in seen:
            raise ValueError("Duplicate neuron")
        seen.add((l,n))
        if not 0 <= l < len(layers) or not 0 <= n < layers[l].mlp.down_proj.in_features:
            raise ValueError("Neuron outside model dimensions")
        mode = spec.get("mode", "scale")
        if mode not in {"scale", "add"}:
            raise ValueError("mode must be scale or add")
        grouped[l].append((n, mode, float(spec.get("value", 0))))
    handles = []
    def make_hook(specs):
        def hook(module, inputs):
            x = inputs[0].clone()
            for n, mode, value in specs:
                if mode == "scale":
                    x[..., n] *= value
                else:
                    x[..., n] += value
            return (x, *inputs[1:])
        return hook
    try:
        for l, specs in grouped.items():
            handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(specs)))
        yield
    finally:
        for h in handles:
            h.remove()
