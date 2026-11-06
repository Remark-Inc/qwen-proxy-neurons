"""Run bounded research jobs; generated tool calls are recorded, never executed."""
import json
from pathlib import Path
import modal
from qwen_lab import MODEL, REVISION

app = modal.App("qwen-neuron-lab")
cache = modal.Volume.from_name("qwen-neuron-lab-models", create_if_missing=True)
results = modal.Volume.from_name("qwen-neuron-lab-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "transformers==5.21.0", "accelerate==1.15.0",
                      "huggingface-hub==2.3.0", "numpy==2.5.4")
         .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_python_source("qwen_lab"))

@app.function(image=image, volumes={"/models": cache}, timeout=3600, cpu=4,
              memory=16384, max_containers=1, retries=0)
def download():
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION,
                             allow_patterns=["*.json", "*.jinja", "*.safetensors"])
    cache.commit()
    return {"model": MODEL, "revision": REVISION, "path": path}

@app.function(image=image, gpu="H200", volumes={"/models": cache, "/results": results},
              cpu=4, memory=98304, timeout=7200, max_containers=1,
              scaledown_window=2, retries=0, block_network=True)
def experiment(config: dict):
    from qwen_lab.runner import run
    result = run(config, Path("/results"))
    results.commit()
    return result

@app.local_entrypoint()
def main(stage: str = "download", run_id: str = "pilot", samples: int = 16,
         max_new_tokens: int = 768, batch_size: int = 4, seconds: int = 1800,
         source_run: str = "", neurons: str = "", seed: int = 42,
         boundary: str = "domain", heldout: bool = False, cases_file: str = "",
         case_keys: str = "", suite_file: str = "", additions: int = 8):
    if stage == "download":
        print(json.dumps(download.remote(), indent=2))
        return
    if stage not in {"baseline", "discover", "evaluate", "prefix", "continuation", "compare", "extend", "residual"}:
        raise ValueError("stage must be download, baseline, discover, evaluate, prefix, continuation, or compare")
    if boundary not in {"domain", "before_tool", "inside_domain"}:
        raise ValueError("Unknown prefix boundary")
    if not 1 <= seconds <= 6600 or not 1 <= samples <= 1024:
        raise ValueError("Bound seconds to 1..6600 and samples to 1..1024")
    if not 1 <= batch_size <= 32 or not 1 <= max_new_tokens <= 4096:
        raise ValueError("Bound batch size to 1..32 and max_new_tokens to 1..4096")
    if not 1 <= additions <= 16:
        raise ValueError('Bound additions to 1..16')
    config = dict(stage=stage, run_id=run_id, samples=samples,
                  max_new_tokens=max_new_tokens, batch_size=batch_size,
                  seconds=seconds, source_run=source_run, seed=seed,
                  boundary=boundary, heldout=heldout,
                  case_keys=[x.strip() for x in case_keys.split(',') if x.strip()],
                  neurons=json.loads(neurons) if neurons else [], additions=additions)
    if cases_file:
        config["custom_cases"] = json.loads(Path(cases_file).read_text())
    if suite_file:
        config['scenario_specs'] = json.loads(Path(suite_file).read_text())
        from qwen_lab.fixtures import validate_scenario_specs
        validate_scenario_specs(config['scenario_specs'])
    output = experiment.remote(config)
    path = Path("runs") / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps(output, indent=2))
    for i, candidate in enumerate(output.get("summary", {}).get("sets", []), 1):
        (path / f"candidate-{i}.json").write_text(json.dumps(candidate["neurons"], indent=2))
    print(json.dumps(output.get("summary", output), indent=2))
