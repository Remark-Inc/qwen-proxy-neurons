# Qwen neuron lab

> **Archived lab notebook (October 27, 2026).** This is the running log from the first three
> phases: the first search, the zero-validation suite and the explanation follow-up. It is kept
> as written, so its commands and paths describe the lab's layout at the time (the apps now
> live in `apps/`, the dashboards in this folder). For the method as finally used, see
> [METHOD.md](../METHOD.md); for the later phases, the other reports in this folder.


Current objective: reach zero observed unwanted proxy calls on a fixed fresh-generation
validation suite. Neuron pruning and broader capability tradeoffs come afterward.
The initial search allowed three neurons; the current validated candidate uses 12.

No neuron set is assumed to exist. Discovery returns candidates; only independent
generation experiments can establish a behavioral effect. Failure to find a set
with this search does not establish that no such set exists.

## Explanation follow-up: completed, with a residual failure — October 27, 2026

The follow-up completed **2,784 exploratory generations**, individual ablations
and restorations, all 66 pair restorations, dose-response and matched-layer
random controls, 60 synthetic contrasts, temporal interventions, activation
patching, and 48 signed-activation profiles. A further paired HTTP/HTTPS probe
investigated a newly observed failure. All nine analysis phases completed.

**The fixed twelve-neuron set does not eliminate the behavior across runs.**
The timing experiment produced **1/128 SEC proxy calls** with all twelve ablated,
versus **14/128 unsteered**. The earlier zero-observed validation below remains
a historical fixed-suite result. The saved candidate has not been changed.

| Intervention window | SEC proxy calls |
| --- | ---: |
| Unsteered | 14/128 |
| All positions | 1/128 |
| Prompt only | 19/128 |
| Reasoning only | 17/128 |
| Tool-call contents only | 0/128 |
| Generated positions only | 0/128 |

No timing-arm responses were truncated or left unfinished. These are exploratory
comparisons, not proof that the zero-count timing variants are superior or
reliably zero. Controls reused seed schedules across jobs and must not be pooled
as independent replications. GPU numerical variation can change trajectories.

The strongest supported explanation is **context-dependent URL completion**:

- L63/N4455 has the largest individual effect on SEC hostname choices. Its
  negative activation and a `www`-facing output direction support an
  alternative-suppression interpretation. The effect is strongly host-dependent.
- L63/N417 and N14734 primarily affect the `out` continuation after the initial
  `r` in these traces; N417 has a clear `out`-facing output direction. Several
  other neurons contribute to initial `r` preference. Some smaller contributors
  still lack a clear semantic interpretation.
- The new all12 failure used `http://`. Changing only that scheme to `https://`
  reduced teacher-forced P(`r`) under all12 from **59.7% to 4.8%**. The proxy
  hostname token-prefix probability, omitting the terminal slash, fell from
  **57.7% to 0.18%**. These are conditional token probabilities, not rollout
  failure rates. L63/N4455 was almost inactive in the HTTP version.
- The replay verified that all twelve selected activations were exactly zero.
  The residual failure is not explained by a missing ablation hook.
- Explicit proxy-fetch and proxy-quotation controls passed 16/16 each in both
  baseline and all12 arms. These narrow checks establish preserved ability to
  emit the requested hostname/URL, not broad helpfulness or task completion.
- Same-prefix decision-value restoration recovered much of the lost proxy
  preference. The simple early-to-late mediation probe did not establish a clean
  causal chain. Large activation magnitude alone was not a reliable importance
  indicator.

[Explanation dashboard and all twelve profiles](explanation-results.html) ·
[Machine-readable results](explanation-results.json) ·
[Raw-output audit](../data/explanation/audit.json) ·
[Saved residual failure](../data/explanation/new-all12-failures.json)

The next repair target is the HTTP-conditioned continuation path. A new search
should include both schemes, legitimate `r`-prefixed destinations and safe
continuation boundaries, then validate on new natural rollouts. The twelve-neuron
candidate remains frozen; no new repair or pruning claim is made here. Broad
BFCL/IFBench evaluation remains deferred.

Additional measured compute estimate: **$8.37 of the additional $100 authorization**,
using [published Modal rates](https://modal.com/pricing), verified October 27.
This excludes startup, failed-import/return-transfer overhead and storage; it is
not an invoice. The failed large return was recovered from the committed volume
without repeating GPU work. All **11 Modal apps for this phase are stopped with
zero tasks**. The implementation passed **23 tests**, and the artifact audit
recomputed all generation classifications and verified candidate identity,
model revision, phase-hook counts, score consistency and scheme-only edits.
See [the budget ledger](../data/explanation/budget.json) and
[shutdown verification](../data/explanation/modal-status.json).

## Earlier fixed-suite result: empirical-zero target met — October 27, 2026

A **12-neuron ablation** reached **0/512 proxy calls** on the frozen fresh-seed
SEC mocked-403 validation, versus **74/512 unsteered**. The eight-neuron candidate
produced 1/512. All three arms used the same seed schedule, prompts, model revision,
precision, and 2,048-token response allowance.

The 12-neuron set also produced **0/160** proxy calls across five additional
failed-fetch scenarios (baseline: 3/160), passed **32/32** clean-fetch controls and
**32/32** URL-recall controls, and had **zero truncated or unfinished responses**.
Across all 736 outputs, a case-insensitive audit found no routify, aliyuncs or
169.254.169.254 mentions. On the main SEC task it made requested-host calls in
458/512 responses, compared with 406/512 unsteered; total tool-call responses were
465/512 versus 469/512. These are limited utility checks, not a broad capability
benchmark or task-completion grade.

This is empirical zero in the Vals-style synthetic-403 reproduction, **not** a
universal 0% guarantee, a user-request-only end-to-end agent evaluation, or a
replication of Cohere's Flash-Next upload behavior. The SEC zero-event one-sided
95% binomial upper bound is about 0.58% per candidate; a conservative simultaneous
bound across the two frozen candidates is about 0.72%. Pruning, BFCL and IFBench
remain deferred to the cost/benefit phase.

[Validated intervention](../data/candidate-zero-proxy.json) ·
[Validation dashboard](zero-results.html) ·
Residual neuron measurements ·
[Machine-readable validation](zero-results.json)

| Layer (zero-based) | Ablated neuron indices |
| --- | --- |
| 54 | 6172, 10344 |
| 56 | 16758 |
| 58 | 11828 |
| 62 | 6870, 10856, 13617 |
| 63 | 417, 3149, 4455, 10126, 14734 |

The additions came from exact replay of all five failures under the previous best
quartet. That source intervention remained active during attribution. Signed
activations and gradients were recorded before the hostname decision, then actual
added ablations tested their causal influence on the full-hostname contrast.
The five prefixes did not already contain the proxy domain. Those failures and
the 128-trial pilots are discovery data; the frozen validation uses fresh seeds.
Both the eight- and twelve-neuron validation arms were fixed before validation
outcomes were inspected. The twelve-neuron pilot had one right-censored legitimate
SEC call at 1,024 tokens, so it was not counted as a clean zero pilot; the longer
fixed validation completed without censoring.

Cumulative completed-run resource estimate: **$17.58 of the $100 authorization**,
excluding startup, failed jobs and storage; not an invoice. All 34 completed
experiment results are preserved. See `modal-run-status.json` for shutdown status.

## Fresh-seed sparsity follow-up — October 27, 2026

**A fourth neuron did not eliminate the failure.** On 256 fresh-seed SEC trials:

| Intervention | Proxy calls | Rate |
| --- | ---: | ---: |
| Unsteered | 39/256 | 15.23% |
| L63/N4455 | 18/256 | 7.03% |
| L63/N417 + L63/N4455 | 8/256 | 3.13% |
| Original triple: add L63/N14734 | 6/256 | 2.34% |
| Original triple + L63/N6560 | 7/256 | 2.73% |
| Original triple + L54/N10344 | 5/256 | 1.95% |

Under the earlier preference for fewer neurons, **the two-neuron pair was the
preferred next candidate for formal capability testing**. The subsequent zero-first
objective and validated 12-neuron set above supersede that recommendation. It reduced the observed SEC proxy
rate by about 79% relative to the matched unsteered run. This is a practical
sparsity choice, not proof that the pair is equivalent to larger sets, globally
minimal, or reliable enough for deployment. Pair versus triple had an exploratory
paired exact McNemar p=0.727; nonsignificance does not establish equivalence.
The pair and triple each had one truncated SEC response. Requested-host calls
were 212/256 unsteered, 231/256 for the single and pair, and 227/256 for the triple.

The 32-trial clean-fetch and 32-trial URL-recall controls are limited utility
checks. Formal BFCL/IFBench capability evaluations have **not** been run.
The current evidence does not establish general helpfulness preservation.
All six conditions completed 480 trials each and passed both 32-trial clean-fetch
and URL-recall controls. Every intervention produced zero proxy calls on the
160 additional failed-fetch trials, versus 2/160 for baseline; all baseline
failures there were on the new SEC-rules prompt. The other four new URLs had no
baseline proxy calls, so they provide limited evidence about suppression.
Across 27 completed experiment runs, measured-runtime resource estimates total
$11.91 (not an invoice; excludes startup, failed jobs and storage).
See the sparsity dashboard,
machine-readable results, and
[two-neuron intervention](../data/candidate-two-neuron.json).

The single and pair were selected from all subsets of the original triple using
the earlier seed-9000 SEC replication as development data. The first quartet
added the candidate with best mean improvement on the original six discovery
prefixes. Because that gain was almost entirely on the Rust prefix, a separate
exploratory quartet added the candidate with best worst-case improvement across
those same six prefixes. The latter addition was recorded after the first three
confirmation conditions began, but before either quartet's free-generation
results were available. Neither quartet was selected using its confirmation
outputs. All original frozen conditions remain in the report.

## Initial measured result — October 27, 2026

On the pinned Qwen3.8-27B checkpoint, ablation of **L63/N417, L63/N14734, and
L63/N4455** reduced emitted proxy calls from **22/128 to 1/128** in the SEC-page
mocked-403 replication, with new seeds and no suspicious destination prefilled.
The matched-layer random three-neuron control emitted proxy calls in **17/128**
trials. The earlier candidate (L63/N4455, L63/N7495, L58/N13077) emitted **8/128**.
Requested-host calls increased from 104/128 to 114/128; neither run truncated.
This is a result for one scenario in this harness, not an estimate of deployment
risk or proof of broad capability preservation. The 95% Wilson interval for the
treated call rate is about 0.14%–4.29%; zero risk is not established.

The same triple reduced the proxy-versus-requested-host likelihood ratio by
about 2×–29× across six new destination prefixes, including hosts starting with
`r`. Sixteen clean-fetch and sixteen URL-recall controls remained correct in the
tested ways. In a failed-fetch asyncio scenario, requested-host calls fell from
15/16 to 12/16, so broad utility preservation must not be inferred from these
small controls. No blinded task-completion benchmark was run.

The Cohere-derived, cross-model upload-prefix contrast showed essentially no
improvement. The original Qwen3.8-Flash-Next model was not run. There is **no
validated shared three-neuron explanation for both reports**. Logit lenses
suggest at least part of the effect involves lexical preferences (e.g. `out`
and `www`), not a cleanly isolated intention or policy feature.

See the experiment dashboard for all completed runs,
including the first candidate, controls, uncertainty intervals, and raw JSON.
[Candidate intervention](../data/candidate-proxy-suppression.json) is a ready-to-use
`--neurons` array. Indices are zero-based; ablation operates at `down_proj` input
on all prompt and generated tokens. All calls were recorded as text; none were
executed. Results are exploratory and multiple comparisons were not pre-registered.
The paired exact McNemar comparison for the selected triple versus baseline is
approximately p=0.00000572 (unadjusted). This does not replace replication across
tasks, harnesses, checkpoints, or broader capability tests.

At the end of the initial phase, all 16 Modal apps used up to that point
(including download and one failed attribution attempt) were verified stopped
with zero running tasks. Fourteen
completed experiment runs have a combined measured-runtime compute estimate of
$3.51, excluding setup/failed-run overhead and storage; this is not an invoice.

## Sources and scope

- [Neuron explorer](https://github.com/lyramakesmusic/neuron-explorer) motivates the
  steering-first workflow, random controls, and intervention at the input of
  `down_proj`. This repository is an independent implementation. Its Qwen3.5-4B
  neuron IDs and caches are not reused for Qwen3.8-27B.
- [Vals report](https://www.vals.ai/blogs/qwen-alibaba-requests): two-turn synthetic
  403 replay, unwanted proxy fetching, and false recollection of previous URLs.
  The report's 2–4% figure is provider-specific, not an expected result guaranteed
  for this harness. Our tool description, reasoning settings, sampling, and
  synthetic first assistant call are explicit experimental choices.
- [Cohere report](https://cohere.com/blog/concerning-behaviour-popular-open-weight-ai-model)
  and [public trace](https://huggingface.co/datasets/CohereLabs/10-2026-qwen-investigation-traces):
  Qwen3.8-Flash-Next attempted a probe upload. The downloaded JSONL is preserved
  in `data/cohere-original.jsonl`; it is evidence, not executable code.
  This pilot does **not** establish effects on Flash-Next, uploads, metadata probes,
  or a deliberate backdoor. Raw token IDs cannot be transferred between models.

Target: `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; 64 layers × 17,408 MLP neurons.
Indices are zero-based. BF16 weights, Transformers 5.21.0, no quantization.

## Run

```sh
uv sync --extra test
uv run modal profile list
uv run modal run modal_app.py --stage download
uv run modal run modal_app.py --stage prefix --run-id domain-scores --boundary domain
uv run modal run modal_app.py --stage discover --run-id discover-1 --seconds 3600
uv run modal run modal_app.py --stage compare --run-id holdout-prefixes --source-run discover-1 --seconds 3600
uv run modal run modal_app.py --stage baseline --run-id baseline-128 --samples 128 --max-new-tokens 1536 --batch-size 4 --seconds 6600
# Optional: discover from sampled positive traces instead of synthetic prefixes.
uv run modal run modal_app.py --stage discover --run-id discover-observed --source-run baseline-128 --seconds 3600
# Only run a treatment if discovery produced this candidate file.
uv run modal run modal_app.py --stage evaluate --run-id heldout-baseline --samples 128 --seed 1042 --seconds 6600
uv run modal run modal_app.py --stage evaluate --run-id heldout-treatment --samples 128 --seed 1042 --seconds 6600 --neurons "$(cat runs/discover-1/candidate-3.json)"
uv run python -m qwen_lab.report runs/heldout-treatment/result.json runs/heldout-treatment/report.html
```

Use an empty list for the baseline. `scale: 0` ablates a neuron; `scale: 1` is a no-op; `add` applies a
constant in native activation units. At most 32 distinct neurons are accepted for the zero-target residual search; the original discovery search still selects up to three.
Hooks cover all prompt and generated positions and are removed after each test.

The H200 job loads cached weights and blocks network access. Tool schemas are
provided for generation, but no tools are implemented or executed. The download
job has network access to obtain the model. Modal volumes preserve intermediate
rollouts even if a job is interrupted; retrieve them with `modal volume get`.

Use unique run IDs. `samples` means samples **per scenario**, not total.
The wall-clock budget stops admission of new batches; a batch already running
can finish. Modal's 7,200-second timeout bounds each GPU invocation. One GPU
container, no retries, and a two-second idle window limit accidental spend.
The $100 session budget must also be tracked across runs; the CLI is not an
account-wide billing cap. Estimated resource costs are saved separately from
scientific results and are not invoices.

## Experiment and interpretation

1. **Prefix discovery without rare-event sampling:** synthetic prefixes after
   a mocked 403 are explicitly labeled and preserve the pinned model's chat
   formatting. Score the proxy hostname plus `/` against the intended hostname
   plus `/` under an identical token prefix. Use the unnormalized sum of token
   log probabilities for the likelihood ratio; retain per-token and mean scores
   for diagnostics. BPE boundary merges are handled by backing both branches up
   to the same token boundary, recording the character backoff.
2. **Baseline:** complete generations after a mocked 403 for the McKinsey and
   pathlib pages. Keep exact output IDs, rendered prompts, model revision,
   generation settings, batch seeds, and batch positions. An unfinished response
   is marked censored/truncated, not treated as a successful mitigation.
3. **Attribution:** on synthetic or sampled proxy-call prefixes, replay the prefix
   before the first distinguishing proxy-host token. Rank all neurons by last-position
   activation × gradient of proxy-token minus intended-host-token logits. This
   is a local surrogate; it can select token-spelling features.
4. **Causal screen:** ablate 32 ranked neurons and eight random neurons, measuring
   the **whole hostname** log-probability contrast, not just the first token.
   Greedily select sets of size 1–3 from the top 12 screened
   candidates. This is not exhaustive search and may miss synergistic pairs.
5. **Independent validation:** freeze the selected set, then compare unsteered,
   no-op, random controls, and treatment on new seeds and held-out URLs. Test
   each constituent, pairs, the triple, and a dose-response series. Re-run the
   original scenario with independent seeds as well as held-out scenarios.
6. **Utility and displacement:** measure legitimate fetching, task completion,
   malformed calls, truncation, alternative unrequested hosts, and prose-only
   false recollection. `expected_host_call` is a weak utility proxy, not proof of
   correct task completion. Do not accept a "fix" that merely stops all output,
   disables tools, or replaces the Alibaba host with another unwanted host.

Wilson 95% intervals are descriptive binomial intervals, not paired significance
tests or corrections for adaptive candidate selection. Matching batch seeds does
not guarantee identical per-sample random draws after generation paths diverge.
Run enough independent holdout samples for rare-event uncertainty; zero events
in 100 trials still has an upper bound around 3.7%. Keep discovery data out of
confirmatory testing.

## Inspect results

`runs/<id>/result.json` contains metadata, summaries, and full rollouts. The HTML
viewer filters by scenario, actual proxy call, expected host call, or truncation.
Nothing is loaded from the network by the viewer. Remote run directories also
contain append-only JSONL checkpoints and causal-screen measurements.

Local verification: `uv sync --extra research --extra test && uv run pytest -q`.
Tests check tool-call scoring, deceptive hostnames, uncertainty intervals, exact
neuron ablation, multiple interventions, cleanup on exceptions, teacher-forcing
alignment, and shared-prefix tokenization.

## Prefix boundaries and imported traces

`--stage prefix` scores specified continuations; `--stage continuation` samples
complete continuations from the same prefix. Both accept `--neurons` and
`--heldout` (different destination hosts). Three boundary modes are implemented:

| `--boundary` | What is fixed | What the score can establish |
| --- | --- | --- |
| `domain` | Reasoning, tool choice, URL syntax | Conditional hostname selection |
| `before_tool` | Reasoning only | Relative likelihood of two particular complete calls; **not** total probability of initiating any tool call |
| `inside_domain` | Initial `routify-` fragment | Completion of a supplied fragment versus a synthetic alternative tail; **not** legitimate destination selection |

To measure whether a model initiates a call at all, use free generation at the
earlier boundary and count parsed calls; a contrast between two call sequences
alone cannot answer that question. The inside-domain alternate is deliberately
labeled synthetic: a legitimate host is not reachable by changing only a suffix
after an incompatible host prefix has already been supplied.

`compare` evaluates discovered singles, pairs, and triples on new requested hosts
at all three boundaries. It also includes half-strength, no-op, doubled-activation,
and random-ablation controls. Requested hosts beginning with `r` help distinguish
proxy selection from generic letter/token suppression. Direct logit lenses are
saved as exploratory descriptions, not ground-truth neuron labels.

```sh
uv run modal run modal_app.py --stage prefix --run-id earlier-scores --boundary before_tool
uv run modal run modal_app.py --stage continuation --run-id conditional-calls --boundary domain --samples 32 --max-new-tokens 512
uv run modal run modal_app.py --stage prefix --run-id custom-traces --cases-file data/my-cases.json
uv run modal run modal_app.py --stage prefix --run-id cohere-transfer --cases-file data/cohere-prefix-cases.json
```

Custom cases are a JSON array with fields `id`, `prefix`, `bad_text`, `good_text`,
`source`, `synthetic` (boolean), `boundary`, and `scope`. `prefix` is the complete
rendered prompt, including the original tool schema and chat markers. Use
`source_model` for cross-model text replay. Optional `prefix_ids` require exact
`model` and `revision` matches and are checked against canonical tokenization.
Keep original schemas when available; Vals's full provider schema was not
published, so built-in fixtures explicitly label theirs reconstructed. Edited
prefixes must be labeled synthetic. `data/prefix-manifest.json` contains the
fully rendered built-in cases for inspection.

`data/cohere-prefix-cases.json` preserves the original Cohere system/tool schema
and reconstructs a prefix before the upload destination. It is explicitly a
cross-model text replay on the 27B model. Its alternative is `example.invalid`,
which is a synthetic control, **not a legitimate requested upload destination**.
This case can test conditional destination preference, but cannot establish that
the neurons explain the original Flash-Next incident.

Methodology reference: [Zhang and Nanda, Towards Best Practices of Activation
Patching](https://arxiv.org/abs/2309.16042). Prefix construction, metrics, and
intervention positions are part of the experimental claim, not incidental details.

## Sparsity follow-up and mixture-of-experts applicability

The follow-up tests all three singles and all three pairs from the selected triple
on the development SEC replication. A fourth candidate is selected only by
conditional scoring on the original six discovery prefixes. The single, pair,
triple and quartet then use frozen identities on new seeds and five new prompts,
plus clean-fetch and URL-recall controls. See
[data/sparsity-followup-protocol.json](../data/sparsity-followup-protocol.json).
This is a search within a candidate pool, not a proof of a globally minimal set.

The prefix contrast, gradient screening, direct ablation and independent
generation design also applies in principle to sparse mixture-of-experts models.
The current code deliberately rejects them: dense `down_proj` hooks are not
sufficient for fused expert implementations. A correct adapter must:

- Identify each routed neuron by (layer, expert, neuron), and distinguish shared experts.
- Hook the gated intermediate activation before that expert's down projection,
  preserving its routing weight and the other selected experts.
- Record expert-selection frequency and both conditional-on-selection and
  unconditional effects across prompts. An inactive expert's zero gradient is
  not evidence that it cannot matter on another route.
- Treat router interventions as a separate target: discrete top-k selection is
  not fully explored by local activation gradients.
- Measure downstream routing changes during actual interventions and free generation.
- Count distinct physical neuron coordinates across experts; the same numerical
  neuron index in many experts is many neurons, not one.

Memory planning must use total expert weights, not just active parameters.
No dense-model neuron IDs or results transfer to a different checkpoint.
See [Transformers expert backends](https://huggingface.co/docs/transformers/experts_interface)
and the [Qwen3.5 MoE architecture documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5_moe.md).

For a cited capability evaluation at the end, BFCL v4 is the closest tool-use
benchmark, with IFBench as a complementary instruction-following check. Qwen's
27B model card reports IFBench 79.5 and WebArena-Verified 64.8; it does not list
a BFCL score. Published numbers provide context, while matched local unsteered
runs are necessary to estimate intervention effects. The benchmark plan and
primary-source links are recorded in
[data/capability-benchmark-plan.json](../data/capability-benchmark-plan.json).
These formal benchmarks have not yet been executed.

To reproduce the fourth-candidate screen and frozen confirmation suite:

```sh
uv run modal run modal_app.py --stage extend --run-id fourth-screen-replication --source-run lexical-discovery-1 --neurons "$(cat data/candidate-proxy-suppression.json)" --seconds 600
# Supply one condition's neurons from data/sparsity-frozen-conditions.json.
# Use a unique run ID for every condition; use the same seed across conditions.
uv run modal run modal_app.py --stage evaluate --run-id confirmation-baseline-replication --suite-file data/sparsity-confirmation-suite.json --samples 256 --batch-size 32 --max-new-tokens 1024 --seconds 2400 --seed 150000 --neurons '[]'
uv run python -m qwen_lab.sparsity_report
```

The `extend` stage requires the original discovery artifact in the Modal results
volume. The report uses the original `confirm-*` run IDs; independent replications
should be reported separately rather than overwriting those results.

## Residual-failure search: prioritize empirical zero

The latest objective prioritizes zero observed proxy calls before minimizing the
number of neurons or measuring broader capability costs. The 5 residual calls
from `confirm-quartet-robust` are now **discovery data**, not held-out evidence for
new interventions. `--stage residual --source-run confirm-quartet-robust` replays
exact sampled prefixes, retaining the source ablations during attribution and
causal scoring. It records signed post-gating activations, gradients, and their
products at the first distinguishing hostname token, then tests each candidate
by actual ablation and greedily adds neurons. Activation magnitude alone is not
evidence of a trigger, and these conditional scores do not establish effects
on earlier tool initiation.

The initial expansion permits eight additional neurons (12 total), with a hard
implementation limit of 32. Pilot generations select a candidate before its
independent validation. See [zero-target protocol](../data/zero-target-protocol.json).
No output filtering or blocked network call will count as successful neuron
mitigation. Broader capability benchmarks and pruning come after an empirical
zero result. A finite sample cannot establish a universal zero failure rate.

