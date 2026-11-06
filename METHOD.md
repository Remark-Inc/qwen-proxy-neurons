# How the neurons were found, and how the patch was checked

This is the method behind the post, in one page, with pointers into the code.

## 1. The task, and what counts as a proxy call

Every run is a recorded generation on `Qwen/Qwen3.8-27B` (pinned revision) with a `webfetch`
tool in the system prompt. The tool is never executed: the model's tool calls are parsed out of
the generated text (`qwen_lab/metrics.py`) and classified by host. A proxy call is any call whose
host is `routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com`.

The main fixture (`qwen_lab/fixtures.py`) reconstructs Vals AI's example: the user asks for a
summary of an SEC filing, the first `webfetch` has already returned a 403, and the model
continues from there. Other fixtures do the same with other 403 pages (SEC rules, MDN, PyTorch,
Reuters, Wikipedia), and two no-failure tasks (asyncio, pathlib) check that nothing fires
without a failure. Sampling is the model's default (temperature 1, top-p 0.95, top-k 20,
reasoning effort "medium"); the token budget was 2,048 for the first validation and the http://
suites and 1,024 for the 2,048-run arms and the coverage runs.

## 2. The decision point

In a failing transcript the choice is made at one position: the token right after `https://`
(or `http://`) inside the retry's URL. There the model either starts the proxy host (`r`) or a
legitimate one (`www`, `efts`, `developer`, ...). The post's "When the Decision Happens" section
rests on two phases of `qwen_lab/explanation.py`: `generate_timing` switches the 12 off during
the prompt only, the reasoning only, or the tool call only in full generations (14, 19, 17 and
0 of 128 proxy calls), and `timing` does the same at replayed decision points (the 74%, 2% and
1% in Figure 1, from 13 natural replays).

Every search below replays real or constructed failing transcripts up to that position and
measures the model's preference for the proxy host there. Those are conditional numbers. Only
fresh generations (section 5) decide whether a set works.

## 3. First search: the 12 neurons for `https://` (`qwen_lab/runner.py`, `residual.py`)

1. **Cases.** Six constructed 403 transcripts (`qwen_lab/prefixes.py`, `synthetic_cases`), for
   McKinsey, pathlib, requests, rust-lang, r.jina.ai and the Vue router docs, each replayed to the
   first token where the proxy continuation and the legitimate continuation diverge. Four of the
   legitimate hosts also start with `r`, so the contrast is not just the letter.
2. **Attribution.** At that position, activation × gradient of the logit difference (proxy minus
   legitimate first token) for every MLP neuron in the language model: 64 layers × 17,408
   neurons. This is a first-order ranking only.
3. **Screening by ablation.** Each top-ranked neuron is actually switched off (its `down_proj`
   column zeroed via a forward pre-hook, `qwen_lab/interventions.py`) and the log-odds of the full
   proxy continuation against the full legitimate continuation is re-measured
   (`qwen_lab/scoring.py`). Eight random neurons from random layers are scored the same way as
   controls.
4. **Greedy selection.** Neurons are added one at a time by the largest measured reduction:
   L63/N417, then L63/N14734, then L63/N4455. The post's 1- and 2-neuron rows were chosen
   afterwards from the subsets of that triple by fewest proxy calls in 128 SEC generations
   (L63/N4455 alone, then with L63/N417).
5. **A fourth neuron.** The `extend` stage of `runner.run` screened a 32-candidate pool on the same prefixes; the
   neuron with the best worst-case gain, L54/N10344, was added (the best mean gain, L63/N6560,
   was validated separately and dropped; `data/robust-fourth-amendment.json`).
6. **Residual round.** With those four switched off, the five SEC generations that still called
   the proxy became new cases, and `residual.py` repeated steps 2–4 on them with eight greedy
   additions. That is how the set went from 4 to 12. The 8- and 12-neuron candidates were
   piloted on 128 runs each and then validated on fresh seeds (section 5).

## 4. Second search: the 16 neurons for `http://` (`qwen_lab/http_path.py`)

One validation run got past the 12 by typing `http://`. The second search starts with the 12
switched off and uses the same loop (`attribute_first_token`, `score_set`, `discover`) with three
additions:

- **Objective.** Mean teacher-forced probability of the proxy host over a case set, with
  natural decision points (from real `http://` failures) and forced ones (the same transcripts
  with `https://` rewritten to `http://`) weighted equally (`objective`).
- **Gate.** A candidate is accepted only if the explicit-request controls lose at most 0.25 nats
  on any case and 0.05 nats on average relative to the 12-neuron patch (`gate`). The controls ask
  for the proxy URL by name, for legitimate `http://` URLs (CERN, rust-lang.org,
  raw.githubusercontent.com, neverssl.com, r.jina.ai, the SEC page), and include quote and
  scheme controls. That is what keeps the patch from removing the model's ability to type a host
  on request.
- **Pilots and pruning.** The search regenerates the SEC task, requested over `http://`, with the
  current set switched off: once at the start and again whenever the objective's targets are
  met. New failures are harvested as cases. At the end, neurons added in this search are pruned
  backwards if removing them does not hurt the objective.

A first pass on the SEC page found 7 neurons; a second pass on the docs pages that still leaked
found 9 more. The frozen result is `data/http/candidate-refined.json` (28 neurons: 12 + 16).

## 5. Validation (`sparsity_scale.py`, `patch27.py`)

Validation never reuses search cases. Each arm generates fresh rollouts from the SEC 403
fixture, 2,048 per arm on a shared seed block, and counts proxy calls. Wilson intervals on the
counts and a Katz interval on the ratio against the unmodified arm give the reductions in the
post's table. `patch27` repeats this for the final 27 (the 28 minus L56/N16758) and for the 11
(the 12 minus it), and also counts calls to every other host so side effects such as the
Wayback rise are visible. The explicit-request checks (16 of 16 on each task) were run on the
unmodified model, the 12, the 19 and the 28.

## 6. What each neuron does (`neuron_roles.py`)

Three measurements: the direct effect of each of the 28 on the alternative first tokens (`r`,
`www`, `efts`, `developer`, ...) at the decision point, its activation profile across positions,
and drop-one on the 16 http:// neurons with the other 27 off. These give the role descriptions in
the post and found two of the 16 with no measurable effect. L56/N16758's role came from a
separate 2,048-run arm with only it switched off: Wayback calls fell 103 → 18 while proxy calls
did not move.

## 7. Coverage (`coverage_forms.py`, `coverage_vllm.py`, `coverage_nvfp4.py`)

The patch is tested outside the search's conditions: other URL forms at the decision point
(curl, wget, urllib, bare hostname, markdown links, uppercase `HTTP://`), a bash tool instead of
`webfetch`, batched serving in vLLM in bf16 and FP8, and a real NVFP4 checkpoint served
natively on a B200. `coverage_scan.py` searches every saved run for any other memorized Alibaba
host. The results, including the two forms where the patch leaks, are in
`reports/coverage-sweep-results.md`.

## 8. The patch itself

Switching a neuron off is one weight edit: zero column *n* of layer *l*'s
`mlp.down_proj.weight`. `http_path.weight_equivalence` checked, for the 19 and the 28, that this
in-memory edit gives the same logits and sampled tokens as the hook used during search; the vLLM
and NVFP4 arms checked the edited checkpoints with teacher-forced probes instead, and editing
before or after quantization gave the same results. `patch/apply_patch.py` performs the edit;
`patch/neurons-27.json` lists the neurons.

## Reusing the method on another model

Start from `interventions.intervene` (the hook), `http_path.attribute_first_token` (the ranking),
`score_set` and `objective` (the measurement), `gate` (the safety constraint) and `discover`
(the loop). A case is a prefix of token IDs ending at the decision position plus the target
token IDs for the unwanted continuation (`data/http/cases.json` has the fields). The
model-specific parts are `interventions.language_layers` (where the MLP layers live), the
`language_model.norm` and `lm_head` paths used for the output-token lens, and
`protect_recurrent_cache_for_grad`, which this hybrid-attention model needs so that gradients
through a cached decode step are correct.
