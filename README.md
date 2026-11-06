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
17,408 MLP neurons each). Everything ran on Modal (H200, plus one B200 job) for about $105 of GPU (table below).

- **[METHOD.md](METHOD.md)**: how the neurons were found and how the patch was checked, one page.
- **[patch/](patch/)**: the 27 neurons and a script that applies the edit.
- **[reports/](reports/)**: the write-up for each phase, with the numbers it cites.

## Results

| Phase | Report | Key numbers |
| --- | --- | --- |
| Discovery and first validation (12 neurons, https://) | [research-log.md](reports/research-log.md), [zero-results.html](reports/zero-results.html); protocols `data/zero-target-protocol.json`, `data/sparsity-followup-protocol.json` | 74 of 512 SEC runs call the proxy unmodified; 0 of 512 with the 12 off |
| Explaining the 12: timing, single ablations, restores, profiles | [explanation-results.html](reports/explanation-results.html), [explanation-results.json](reports/explanation-results.json) | the decision lands on the token after `https://`; one run escaped the patch by typing `http://` |
| Sparsity rerun at 2,048 runs per arm | [sparsity-scale-results.md](reports/sparsity-scale-results.md) | 287 → 105 → 56 → 43 → 0 of 2,048 for 0, 1, 2, 3 and 12 neurons off (the 11-neuron row of the post's table is `counts.all11` in patch-27-results.json) |
| The http:// path: second and third searches | [http-path-results.md](reports/http-path-results.md) | 16 more neurons; the 28 close the escape and all but 3 of 392 held-out docs-page cases |
| Neuron roles: drop-one, activation profiles, alternative tokens | [neuron-roles-results.md](reports/neuron-roles-results.md) | 3 of the 28 do nothing measurable; L56/N16758 drives Wayback calls, not proxy calls |
| Patch-27 validation (the 28 without L56/N16758) | [patch-27-results.md](reports/patch-27-results.md) | 0 of 2,048 proxy calls with the 27 off; www.sec.gov calls rise 1,611 → 1,817 |
| Coverage sweep: other URL forms, bash tool, vLLM bf16/FP8, real NVFP4 on a B200 | [coverage-sweep-results.md](reports/coverage-sweep-results.md) | holds for every https:// form and under batched vLLM (0 of 512 patched vs 77/62/80 unmodified); leaks on `curl -s http://` (15 of 512) and uppercase `HTTP://` (4 of 512); a real NVFP4 checkpoint on a B200 goes 40 → 0 of 512 |
| r.jina.ai fallback (a public fetching service that receives the full URL) | `python scripts/jina_check.py`, [jina-explicit-check.md](reports/jina-explicit-check.md) | 47 of 512 SEC runs; the reasoning names it in 42 of the 47; 0 of 64 no-failure runs; asked by name, every arm fetches through it 16 of 16 |

Each `reports/*-results.md` is the frozen write-up for its phase and `*-results.json` holds the
numbers it cites. From the sparsity rerun on, each phase's pre-registered plan, with sha256 pins
on its inputs, is `data/<phase>/protocol.json`, and its GPU ledger is `data/<phase>/budget.json`.
`zero-results.json` and `explanation-results.json` are frozen outputs; their generators are not
included. Several GPU phases read earlier runs from the Modal results volume
(`qwen-neuron-lab-results`), so to rerun one, upload the runs it names first.

## Finding your way from the post

| Post section | Phase | Report | Data | Runs | App | Module |
| --- | --- | --- | --- | --- | --- | --- |
| How We Tested It; What We Found | first search and validation | `reports/research-log.md`, `reports/zero-results.*` | `data/zero-*.json`, `data/candidate-zero-proxy.json` | `zero-validate-*` (`treatment` = the 8-neuron set, `expanded` = the 12) | `apps/modal_app.py` | `runner`, `residual`, `prefixes`, `scoring` |
| When the Decision Happens (Figure 1) | explanation | `reports/explanation-results.*` | `data/explanation/` | `explain-*` | `apps/explanation_app.py` | `explanation`, `explanation_hooks` |
| How Many Neurons Does It Take? (Figure 2) | sparsity-scale | `reports/sparsity-scale-results.*` | `data/sparsity-scale/` | `sparsity-scale-*` | `apps/sparsity_scale_app.py` | `sparsity_scale` |
| What the Patch Missed; Running the Search Again (Table 3) | http-path | `reports/http-path-results.*` | `data/http/` (`combined` = the 19, `refined` = the 28) | `http-*` | `apps/http_path_app.py` | `http_path`, `http_report` |
| What the Neurons Do (Figure 3) | neuron-roles | `reports/neuron-roles-results.*` | `data/neuron-roles/` | `neuron-roles-*` | `apps/neuron_roles_app.py` | `neuron_roles` |
| Making the Fix Permanent (the 27 and the 11) | patch-27 | `reports/patch-27-results.*` | `data/patch-27/` | `patch27-*` | `apps/patch27_app.py` | `patch27` |
| Some gaps remain (other forms, vLLM, NVFP4) | coverage-sweep | `reports/coverage-sweep-results.*` | `data/coverage-sweep/` | `coverage-*` | `apps/coverage_app.py`, `apps/coverage_vllm_app.py`, `apps/coverage_nvfp4_app.py` | `coverage_forms`, `coverage_vllm`, `coverage_nvfp4`, `coverage_scan`, `coverage_report` |

The post's "first and second pass" for http:// are "search 1" and "search 2" in the http-path report.

GPU cost by phase, from each phase's ledger (whole-app estimates at Modal's published rates,
not invoices):

| Phase | Cost |
| --- | --- |
| First search and validation | $17.58 |
| Explanation (incl. a $1.68 follow-up run not included here) | $10.05 |
| http:// searches | $22.69 |
| 2,048-run rerun | $18.21 |
| Neuron roles | $16.24 |
| Patch-27 | $8.86 |
| Coverage sweep | $12.73 |
| Total | about $106 |

What each module in `qwen_lab/` is for:

- `__init__`: the pinned model and revision.
- `fixtures`: the 403 tasks and the clean-fetch and recall controls.
- `metrics`: parses tool calls out of generated text; a proxy call is one whose host is the bucket.
- `interventions`: the `down_proj` input hook that switches a neuron off.
- `explanation_hooks`: the same hook scoped by position, and activation capture.
- `scoring`: teacher-forced log-probabilities of a continuation.
- `prefixes`: replay prefixes that end at the hostname decision.
- `runner`: model loading, generation, the first search (`discover`, `extend`) and `evaluate`.
- `residual`: the residual round that took the set from 4 to 12.
- `explanation`: the experiments that explain the 12.
- `sparsity_scale`: the 2,048-run neuron-count arms and their analysis.
- `http_path`: the http:// searches, validation and suites.
- `http_report`: builds http-path-results.json (`round3`).
- `neuron_roles`: what each of the 28 does.
- `patch27`: the 27 and the 11.
- `coverage_forms`: other URL forms, the bash harness, the curl search round.
- `coverage_vllm`: vLLM bf16, FP8 and NVFP4-rounded arms.
- `coverage_nvfp4`: real NVFP4 checkpoints.
- `coverage_scan`: the scan for any other Alibaba host.
- `coverage_report`: builds coverage-sweep-results.json.

## The patch

[`patch/neurons-27.json`](patch/neurons-27.json) lists the 27 neurons: the 12 from the https://
search minus L56/N16758, plus the 16 from the two http:// searches. Three of the 27 showed no
measurable effect (L54/N10344 in single and restore ablations, L63/N7463 and L60/N5507 in the
drop-one test on the 16) and are kept for fidelity to what was validated. The edit zeroes one column of each layer's MLP down-projection:

```python
with torch.no_grad():
    for layer, neuron in NEURONS_TO_ZERO:
        model.model.language_model.layers[layer].mlp.down_proj.weight[:, neuron].zero_()
```

`python patch/apply_patch.py --save DIR` loads the pinned checkpoint, applies the edit and saves
a patched copy. In our runs the edit behaved the same whether made before or after FP8 or NVFP4
quantization (tested with the 28-neuron set; the 27 were tested after quantization).

What the patch does not do: it covers the paths we found. It still leaks on `curl -s http://` and
on `HTTP://`. It was not built for the model's other fallback, r.jina.ai, which the reasoning does
announce; those calls happen to fall too (157 → 13 of 2,048 with the 27), but the fallback itself
is untouched. An egress allowlist is the first line of defense; the patch is the second.

## Layout

```
METHOD.md            the search and validation method
patch/               the 27-neuron list and a standalone apply script
reports/             one write-up per phase, with its numbers (md + json), and the explanation dashboard
qwen_lab/            experiment code: hooks, search, validation, per-phase analysis
apps/                Modal entrypoints, one per phase; the later phases read data/<phase>/protocol.json
data/                frozen cases, candidate neuron sets, protocols and GPU ledgers
runs/                raw outputs of the 50 runs the reports cite (result.json, rollouts.jsonl)
figures/             the post's figures (SVG + PNG) and the script that draws them
scripts/             jina_check.py, which reads the baseline run directly
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
Figures: `python figures/make_figures.py figures/` rewrites the SVGs; the numbers are written
into the script from the reports. The PNGs were rendered from the SVGs with headless Chrome.

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
