# How the neurons were found, and how the patch was checked

This is the method behind the post, in one page. Section names match the code.

## 1. The task, and what counts as a proxy call

Every run is a recorded generation on `Qwen/Qwen3.8-27B` (pinned revision) with a `webfetch`
tool in the system prompt. The tool is never executed: the model's tool calls are parsed out of
the generated text (`qwen_lab/metrics.py`) and classified by host. A proxy call is any call whose
host is `routify-file-proxy-sg.oss-ap-southeast-1.aliyuncs.com`.

The main fixture (`qwen_lab/fixtures.py`) reconstructs Vals AI's example: the user asks for a
summary of an SEC filing, the first `webfetch` has already returned a 403, and the model
continues from there. Other fixtures do the same with other 403 pages (SEC rules, MDN, PyTorch,
Reuters, Wikipedia), and two no-failure tasks (asyncio, pathlib) check that nothing fires
without a failure. Sampling is the model's default (temperature 1, top-p 0.95, top-k 20).

## 2. The decision point

In a failing transcript the choice is made at one position: the token right after `https://`
(or `http://`) inside the retry's URL. There the model either starts the proxy host (`r`) or a
legitimate one (`www`, `efts`, `developer`, ...). The post's "What Qwen thinks vs. what it
types" section is this observation; `qwen_lab/explanation.py` (the `timing` phase) measures it
by switching the neurons off during the prompt only, the reasoning only, or the tool call only.

Every search below replays real failing transcripts up to that position, teacher-forces the
proxy host, and measures its log-probability. Those are conditional numbers. Only fresh
generations (section 5) decide whether a set works.

## 3. First search: the 12 neurons for `https://` (`qwen_lab/runner.py`, `residual.py`)

1. **Cases.** Failing SEC rollouts replayed to the first token where the proxy continuation and
   the legitimate continuation diverge (`qwen_lab/prefixes.py`).
2. **Attribution.** At that position, activation × gradient of the log-odds (proxy minus
   legitimate continuation) for every MLP neuron in the language model: 64 layers × 17,408
   neurons. This is a first-order ranking only.
3. **Screening by ablation.** Each top-ranked neuron is actually switched off (its `down_proj`
   column zeroed via a forward pre-hook, `qwen_lab/interventions.py`) and the log-odds
   re-measured. Eight random neurons from random layers are scored the same way as controls.
4. **Greedy selection.** Neurons are added one at a time by the largest measured reduction,
   giving the 1-, 2- and 3-neuron sets (L63/N4455, then L63/N417, then L63/N14734).
5. **Residual search.** With the current set switched off, the rollouts that still call the
   proxy become new cases, and steps 2–4 repeat on them, up to a cap. That is how the set grew
   from 3 to 12.

## 4. Second search: the 16 neurons for `http://` (`qwen_lab/http_path.py`)

One validation run got past the 12 by typing `http://`. The second search starts with the 12
switched off and uses the same loop with three additions:

- **Objective.** Mean teacher-forced probability of the proxy host over a case set, with
  natural decision points (from real `http://` failures) and forced ones (the same transcripts
  with `https://` rewritten to `http://`) weighted equally.
- **Gate.** A candidate is accepted only if explicit-request controls, where the user asks for
  the proxy URL by name, lose at most a fixed number of nats relative to the 12-neuron patch.
  That is what keeps the patch from removing the model's ability to type the host on request.
- **Pilots and pruning.** After each round the search regenerates the SEC task with the
  current set switched off; any new failures are harvested as cases. At the end, neurons added
  in this search are pruned backwards if removing them does not hurt the objective.

A first pass on the SEC page found 7 neurons; a second pass on the docs pages that still leaked
found 9 more. The frozen result is `data/http/candidate-refined.json` (28 neurons: 12 + 16).

## 5. Validation (`sparsity_scale.py`, `patch27.py`)

Validation never reuses search cases. Each arm generates fresh rollouts from the SEC 403
fixture, 2,048 per arm on a shared seed block, and counts proxy calls. Wilson intervals on the counts and a
Katz interval on the ratio against the unmodified arm give the reductions in the post's table.
`patch-27` repeats this for the final 27 (the 28 minus L56/N16758) and for the 11 (the 12 minus
it), and also counts calls to every other host so side effects such as the Wayback rise are
visible.

## 6. What each neuron does (`neuron_roles.py`)

Three measurements, all at the decision point: the direct effect of each neuron on the
alternative first tokens (`r`, `www`, `efts`, `developer`, ...), its activation profile across
positions, and drop-one from the full set. These give the role descriptions in the post and
identified L56/N16758 as a Wayback-fallback neuron rather than a proxy neuron, and three
neurons with no measurable effect.

## 7. Coverage (`coverage_forms.py`, `coverage_vllm.py`, `coverage_nvfp4.py`)

The patch is tested outside the search's conditions: other URL forms at the decision point
(curl, wget, urllib, bare hostname, markdown links, uppercase `HTTP://`), a bash tool instead of
`webfetch`, batched serving in vLLM in bf16 and FP8, and a real NVFP4 checkpoint served
natively on a B200. `coverage_scan.py` searches every saved run for any other memorized Alibaba
host. The results, including the two forms where the patch leaks, are in
`reports/coverage-sweep-results.md`.

## 8. The patch itself

Switching a neuron off is one weight edit: zero column *n* of layer *l*'s
`mlp.down_proj.weight`. The hook used during search and the weight edit used for the patched
checkpoints were checked to be equivalent (`http_path.weight_equivalence`), and editing before
or after FP8/NVFP4 quantization gave the same results. `patch/apply_patch.py` performs the
edit; `patch/neurons-27.json` lists the neurons.
