# Sparsity rerun: the 1-, 2-, 3- and 12-neuron arms at 2,048 SEC runs each

Oct 30, 2026. This rerun narrows the confidence intervals for the small neuron sets. The seed-50000 confirmation ran only 256 SEC runs per arm. This rerun uses 2,048 per arm, on a fresh seed block shared by every arm.

- **Numbers:** every number below comes from `sparsity-scale-results.json`, built with `python -m qwen_lab.sparsity_scale build`.
- **Protocol:** `data/sparsity-scale/protocol.json` was frozen before any generation. One amendment, also recorded before any generation, added the 12-neuron arm.

## Headline

| Neurons switched off | Proxy-call runs | Wilson 95% | Reduction vs matched unmodified (Katz 95%) | Paired bootstrap 95% | Earlier result |
|---|---|---|---|---|---|
| 0 (unmodified) | 287 of 2,048 (14.0%) | 12.6–15.6% | — | — | 39 of 256 |
| 1 (L63/N4455) | 105 of 2,048 (5.1%) | 4.3–6.2% | 63% (55–70%) | 56–70% | 18 of 256 |
| 2 (+ L63/N417) | 56 of 2,048 (2.7%) | 2.1–3.5% | 80% (74–85%) | 75–85% | 8 of 256 |
| 3 (+ L63/N14734) | 43 of 2,048 (2.1%) | 1.6–2.8% | 85% (79–89%) | 80–89% | 6 of 256 |
| 12 (https:// patch) | 0 of 2,048 (0.0%) | 0.0–0.19% | at least 97%¹ | —¹ | 0 of 512² |

¹ With no events, the bootstrap is degenerate. The bound is Katz with a 0.5 correction on every cell (97.2%).
² From zero-validate-expanded-1: seed 270000, 2,048-token budget, 74 of 512 unmodified.

**Old and new reductions** (95% CI):

| Neurons | Earlier (256 per arm) | Now (2,048 per arm) |
|---|---|---|
| 1 | 54% (22–73%) | **63% (55–70%)** |
| 2 | 79% (57–90%) | **80% (74–85%)** |
| 3 | 85% (64–93%) | **85% (79–89%)** |
| 12 | 0 of 512 | **0 of 2,048**; true rate below 0.19% at 95% |

- **Precision:** the intervals narrowed about as projected.
- **The single neuron's estimate rose from 54% to 63%.** The 256-run arm landed low by chance. Its interval (22–73%) contains the new estimate, and 54% now sits just below the new interval.

## What ran
- **Arms:**
  - unmodified;
  - L63/N4455;
  - + L63/N417;
  - + L63/N14734;
  - the 12-neuron https:// set: L54/N6172, L54/N10344, L56/N16758, L58/N11828, L62/N6870, L62/N10856, L62/N13617, L63/N417, L63/N3149, L63/N4455, L63/N10126 and L63/N14734.
- **Neuron sources:** the 1-, 2- and 3-neuron sets come exactly as frozen in `data/sparsity-frozen-conditions.json`, with no re-selection. The 12 come from `data/candidate-zero-proxy.json`, identical to the first 12 entries of `data/http/candidate-refined.json`. Hashes were checked at launch.
- **Prompt:** the SEC 403 prompt, 2,048 runs per arm.
- **Seeds:** batch b used seed 4,000,000 + 32·b in every arm. All five arms covered exactly the same 2,048 seed/row units. The block was never used before.
- **Settings:** the same prompt, sampler (temperature 1, top-k 20, top-p 0.95), batch size 32, 1,024-token budget, engine and image pins as the seed-50000 confirmation, with unchanged generation code. Tool calls were recorded as text and never executed.
- **Jobs:** five parallel H200 jobs ran from 16:30 to 17:14 EDT. Every job finished all 2,048 runs, and none reached its time limit.

## Requested host, truncation and other calls

| Arm | sec.gov calls | Any tool call | Other unrequested host | Truncated (1,024 tokens) | Unfinished tool call | Not run |
|---|---|---|---|---|---|---|
| 0 (unmodified) | 1,611 | 1,853 | 1,345 | 1 | 0 | 0 |
| 1 (L63/N4455) | 1,797 | 1,867 | 719 | 0 | 0 | 0 |
| 2 (+ L63/N417) | 1,784 | 1,851 | 744 | 0 | 0 | 0 |
| 3 (+ L63/N14734) | 1,782 | 1,849 | 717 | 1 | 1 | 0 |
| 12 (https:// patch) | 1,832 | 1,854 | 271 | 1 | 0 | 0 |

All counts are out of 2,048. "Other unrequested host" includes the proxy and any host other than `www.sec.gov`.

- **sec.gov calls:** every patched arm calls sec.gov more often than the unmodified model (87–89% of runs against 79%) and calls other hosts less often.
- **Tool use:** overall it stays at 90–91% in every arm.
- **Truncation:** at most one run per arm hit the 1,024-token budget, and at most one ended with an unfinished tool call.

## Pooled with the earlier confirmation (secondary)

| Neurons | Pooled proxy-call runs | Pooled unmodified | Reduction (Katz 95%) |
|---|---|---|---|
| 1 | 123 of 2,304 | 326 of 2,304 | 62% (54–69%) |
| 2 | 64 of 2,304 | 326 of 2,304 | 80% (74–85%) |
| 3 | 49 of 2,304 | 326 of 2,304 | 85% (80–89%) |

## Adjacent comparisons on matched seeds (exploratory, unadjusted)
- **1 → 2 neurons:** proxy-call runs drop from 105 to 56. On matched seeds, 78 runs call the proxy with one neuron off but not with two, and 29 do the reverse. Exact McNemar p = 2×10⁻⁶.
- **2 → 3 neurons:** 56 to 43, with 36 against 23 discordant runs and p = 0.12. At this size, the third neuron's added effect can't be separated from noise.
- **3 → 12 neurons:** 43 to 0, p = 2×10⁻¹³.

## Interval methods
- **Katz log relative risk** is the blog's method. From the earlier counts (18 of 256 against 39 of 256) it reproduces 54% (22–73%) exactly.
- **The paired bootstrap** resamples matched seed/row units 10,000 times.
  - Shared seeds correlate outcomes across arms. For example, 64 runs call the proxy both unmodified and with one neuron off, where about 15 would be expected if the arms were independent.
  - So the paired intervals come out slightly narrower.
  - Resampling whole 32-run batches instead gives the same intervals to within a point.

## Caveats
- **One prompt:** SEC with a synthetic 403 history. The other pages and the clean-fetch and recall controls were out of scope for this rerun.
- **One setup:** one checkpoint and one harness (BF16 Transformers with SDPA attention, on H200).
- **Zero counts** bound the 12-neuron rate. They don't prove it's zero.
- **Selection was on earlier data:** the small sets were chosen on development seeds and frozen on Oct 27. The 12 were validated before on seed 270000. These arms test all of them on fresh seeds.

## Cost
- **Total:** five jobs, $18.21 by whole-app lifetime ($18.04 in-container compute). Each arm took 37–44 min and cost $3.42–$4.00.
- **Research total:** $163.31.

## Files
- **`sparsity-scale-results.json`:** counts, intervals, adjacent tests and per-run outcome bits, with no generated text.
- **Protocol and ledger:** `data/sparsity-scale/protocol.json`, `data/sparsity-scale/budget.json`.
- **Code and tests:** `apps/sparsity_scale_app.py`, `qwen_lab/sparsity_scale.py`, `tests/test_sparsity_scale.py`.
- **Raw rollouts:** `../runs/sparsity-scale-*-1/` (the model's text, with the proxy URLs' key and signature redacted).
