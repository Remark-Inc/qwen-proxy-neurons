# Qwen's Alibaba proxy habit: the neurons behind it, and a patch

Code, data and raw outputs behind the post *"Qwen never says it's calling Alibaba. We found the
neurons that do."*

When a web fetch fails, Qwen3.8-27B sometimes retries the request through an Alibaba Cloud
storage address (`routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com`) without mentioning it
in its reasoning. This repository traces that choice to 27 of the model's 1.1 million MLP
neurons, validates a weight patch that switches them off, and maps where the patch holds and
where it leaks. All model runs are recorded generations: tool calls are parsed as text, never
executed, and the GPU jobs run with network access blocked.

Model: `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (64 layers,
17,408 MLP neurons each). Everything ran on Modal (H200, plus one B200 job) for about $200 of GPU.

- **[METHOD.md](METHOD.md)**: how the neurons were found and how the patch was checked, one page.
- **[patch/](patch/)**: the 27 neurons and a script that applies the edit.
- **[reports/](reports/)**: the write-up for each phase, with the numbers it cites.

## Results

| Phase | Report | Key numbers |
| --- | --- | --- |
| Discovery and first validation (12 neurons, https://) | [research-log.md](reports/research-log.md), [zero-results.html](reports/zero-results.html) | 74 of 512 SEC runs call the proxy unmodified; 0 of 512 with the 12 off |
| Explaining the 12: timing, single ablations, restores, profiles | [explanation-results.html](reports/explanation-results.html) | the decision lands on the token after `https://`; one run escaped the patch by typing `http://` |
| Sparsity rerun at 2,048 runs per arm | [sparsity-scale-results.md](reports/sparsity-scale-results.md) | 287 → 105 → 56 → 43 → 0 of 2,048 for 0, 1, 2, 3 and 12 neurons off |
| The http:// path: second and third searches | [http-path-results.md](reports/http-path-results.md) | 16 more neurons; 28 close the escape and the held-out http:// cases |
| Neuron roles: drop-one, activation profiles, alternative tokens | [neuron-roles-results.md](reports/neuron-roles-results.md) | 3 of the 28 do nothing measurable; L56/N16758 drives Wayback calls, not proxy calls |
| Patch-27 validation (the 28 without L56/N16758) | [patch-27-results.md](reports/patch-27-results.md) | 0 of 2,048 proxy calls with the 27 off; www.sec.gov calls rise 1,611 → 1,817 |
| Coverage sweep: other URL forms, bash tool, vLLM bf16/FP8, real NVFP4 on a B200 | [coverage-sweep-results.md](reports/coverage-sweep-results.md) | holds for every https:// form and under batched vLLM (0 of 512 patched vs 77/62/80 unmodified); leaks on `curl -s http://` (15 of 512) and uppercase `HTTP://` (4 of 512) |
| r.jina.ai fallback (a public fetching service that receives the full URL) | `python scripts/jina_check.py` | 47 of 512 SEC runs; the reasoning names it in 42 of the 47; 0 of 64 no-failure runs |

Each `reports/*-results.md` is the frozen write-up for its phase and `*-results.json` holds the
numbers it cites. Each phase's pre-registered plan, with sha256 pins on its inputs, is
`data/<phase>/protocol.json`, and its GPU ledger is `data/<phase>/budget.json`.

## The patch

[`patch/neurons-27.json`](patch/neurons-27.json) lists the 27 neurons: the 12 from the https://
search minus L56/N16758, plus the 16 from the two http:// searches. Three of the 27 (L54/N10344,
L63/N7463, L60/N5507) showed no measurable effect in the drop-one test and are kept for fidelity
to what was validated. The edit zeroes one column of each layer's MLP down-projection:

```python
with torch.no_grad():
    for layer, neuron in NEURONS_TO_ZERO:
        model.model.language_model.layers[layer].mlp.down_proj.weight[:, neuron].zero_()
```

`python patch/apply_patch.py --save DIR` loads the pinned checkpoint, applies the edit and saves
a patched copy. In our runs the same edit, made before or after FP8 or NVFP4 quantization, behaved
the same.

What the patch does not do: it covers the paths we found. It still leaks on `curl -s http://` and
on `HTTP://`, and it does not touch the model's other fallback, r.jina.ai, which the reasoning
does announce. An egress allowlist is the first line of defense; the patch is the second.

## Layout

```
METHOD.md            the search and validation method
patch/               the 27-neuron list and a standalone apply script
reports/             one write-up per phase, with its numbers (md + json), and the explanation dashboard
qwen_lab/            experiment code: hooks, search, validation, per-phase analysis
apps/                Modal entrypoints, one per phase; each reads its data/<phase>/protocol.json
data/                fixtures, frozen cases, candidate neuron sets, protocols and GPU ledgers
runs/                raw outputs of the 50 runs the reports cite (result.json, rollouts.jsonl)
figures/             the post's figures (SVG + PNG) and the script that draws them
scripts/             small checks that read runs/ directly
tests/               unit tests for the analysis code (no GPU; `pytest`)
```

## Reproducing

```sh
uv sync --extra research --extra test   # analysis and tests run locally without a GPU
uv run pytest
uv run modal run apps/modal_app.py --stage download   # caches the pinned checkpoint on a Modal volume
```

GPU phases, in order, each bounded by its protocol and ledger. Run them from the repository root:

```sh
uv run modal run apps/modal_app.py --stage evaluate --run-id zero-validate-baseline-1 --suite-file data/zero-validation-suite-1.json --samples 512 --max-new-tokens 2048 --batch-size 32 --seed 270000 --seconds 3600
uv run modal run apps/explanation_app.py --phase timing --run-id explain-timing-1
uv run modal run apps/sparsity_scale_app.py --arm baseline --run-id sparsity-scale-baseline-1 --seconds 4100
uv run modal run apps/http_path_app.py --phase discover --run-id http-discover-1 --seconds 4800
uv run modal run apps/neuron_roles_app.py --phase rollouts --run-id neuron-roles-n16758-runs-2 --seconds 4100
uv run modal run apps/patch27_app.py --job all27 --run-id patch27-all27-runs-1 --seconds 4100
uv run modal run apps/coverage_app.py --job forms --run-id coverage-forms-1 --seconds 2200
uv run modal run apps/coverage_vllm_app.py --job vllm-bf16 --run-id coverage-vllm-bf16-1 --seconds 2400
uv run modal run apps/coverage_nvfp4_app.py --job nvfp4-quantize --run-id coverage-nvfp4-quant-1 --seconds 3300
```

Each app refuses a job that is not in its protocol or that exceeds the protocol's time bound,
and the exact arguments used for every run are in the phase's `budget.json`. Expect GPU
numerical variation to move individual rollouts; the aggregate counts were stable across reruns.
Figures: `python figures/make_figures.py figures/` rewrites the SVGs from the numbers in the
reports; the PNGs were rendered from them with headless Chrome.

## Raw outputs and redaction

`runs/` holds the output of the 50 runs the ledgers and reports cite (smoke runs and
superseded pilots are left out). The proxy URLs the model generates carry an Alibaba Cloud
access key ID and a signature. Both are removed everywhere in this repository: in text fields
the values read `OSSAccessKeyId=<redacted>` and `Signature=<redacted>`, and in token-ID arrays
the tokens that spelled them are replaced by the token for `*`, so decoding an array gives
asterisks of the same token count and per-token arrays stay aligned. A test
(`tests/test_patch.py`) checks both forms on every file.

Because of the redaction, some files no longer match the sha256 values pinned in later
protocols: the run files `runs/sparsity-scale-baseline-1/rollouts.jsonl`,
`runs/neuron-roles-n16758-runs-2/rollouts.jsonl` and the five explanation runs pinned in
`data/explanation/audit.json`, plus `data/explanation/new-all12-failures.json` (pinned in
`data/explanation/failure-probe-addendum.json`). The pins refer to the unredacted originals;
nothing else in those files changed.

The parallel experiments on Qwen3.8-Flash-Next (a mixture-of-experts model) are not included;
the post does not use them.

## Sources

- Vals AI, [Qwen's Alibaba requests](https://www.vals.ai/blogs/qwen-alibaba-requests): the task
  fixtures reconstruct its 403 and URL-recall examples.
- Cohere Labs, [10-2026-qwen-investigation-traces](https://huggingface.co/datasets/CohereLabs/10-2026-qwen-investigation-traces)
  (`data/cohere-original.jsonl`, `data/provenance.json`).
- Lyra, [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): the scoring approach
  the discovery search is based on.

## License

Code, protocols and the write-ups are released under the [MIT License](LICENSE). The generated
text in `runs/` comes from Qwen3.8-27B, which Alibaba distributes under Apache 2.0.
`data/cohere-original.jsonl` holds four records from Cohere Labs' trace dataset, redistributed as
received; the dataset declares no license, so those records are Cohere Labs' and not covered by
the MIT grant. The neuron-explorer repository (MIT) was a methodological reference; none of its
code is included here.
