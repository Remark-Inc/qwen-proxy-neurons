# Patch-27 validation: the 28-neuron patch without L56/N16758

Nov 2, 2026. The neuron-roles follow-up found that L56/N16758 drives Wayback calls and does nothing measurable to proxy calls. This run asks whether the post can ship the other 27 neurons and say the Wayback side effect is gone.

**How the run was set up:**
- **Protocol:** `data/patch-27/protocol.json` holds the protocol and its decision rules. Both were frozen before any generation.
- **Numbers:** every number comes from `patch-27-results.json`, built with `python -m qwen_lab.patch27 build`. The tables come from `python -m qwen_lab.patch27 tables`.

## Headline
- **Proxy calls:** the 27 make 0 in 2,048 SEC 403 runs (Wilson 95% upper bound 0.19%). The 12 also make 0, and the unmodified model makes 287.
- **Wayback calls rise rather than return.**
  - **Counts:** 221 of 2,048 with the 27, against 103 unmodified, 20 with the 12 and 18 with L56/N16758 off alone.
  - **Against unmodified:** RR 2.15 (95% CI 1.71–2.69).
  - **Frozen verdict:** "gone", because the lower bound clears 2/3. The rule also requires saying that Wayback calls rose.
- **www.sec.gov calls:** 1,817 with the 27, against 1,611 unmodified and 1,832 with the 12.
- **Other routes shrink:**

  | Destination | 27 off | Unmodified | 12 off |
  |---|---|---|---|
  | efts.sec.gov | 68 | 798 | 151 |
  | r.jina.ai | 13 | 157 | 48 |
  | Other hosts | 52 | 112 | 52 |

- **Held-out http:// sets:** switching L56/N16758 back on, with the other 27 off, adds no sampled proxy completions on any set.

  | Set | 28 off | 27 off |
  |---|---|---|
  | Escape | 0/128 | 0/128 |
  | Natural | 0/192 | 0/192 |
  | SEC substituted | 0/480 | 0/480 |
  | Docs pages | 12/1,568 | 9/1,568 |

- **11 off:** 0 of 2,048 proxy calls, so "12 stopped every call" also holds for 11. Wayback calls are 100 against 103 unmodified (RR 0.97, 95% CI 0.74–1.27), so the Wayback rise under the 27 comes from switching off the 16 http:// neurons.

## SEC 403 destinations
2,048 runs per arm on the sparsity rerun's seeds (batch b under 4,000,000 + 32·b), so runs are paired across arms.
- **Settings:** the same sampler and the same 1,024-token limit in every arm.
- **Counting:** a run counts under every host it calls. Hosts come from the parsed final-answer tool calls, read the same way as option D.

| SEC 403, 2,048 runs | Proxy | web.archive.org | www.sec.gov | efts.sec.gov | r.jina.ai | Other host | No call |
|---|---|---|---|---|---|---|---|
| Unmodified | 287 | 103 | 1,611 | 798 | 157 | 112 | 195 |
| 12 off (https:// patch) | 0 | 20 | 1,832 | 151 | 48 | 52 | 194 |
| L56/N16758 off alone | 312 | 18 | 1,650 | 846 | 153 | 107 | 187 |
| **27 off (28 without L56/N16758)** | **0** | **221** | 1,817 | 68 | 13 | 52 | 182 |
| 11 off (12 without L56/N16758) | 0 | 100 | 1,817 | 141 | 44 | 54 | 198 |

| Comparison (matched runs) | Proxy | web.archive.org | www.sec.gov |
|---|---|---|---|
| 27 off vs unmodified | 0 vs 287 (p = 8×10⁻⁸⁷) | 221 vs 103, RR 2.15 (1.71–2.69), p = 3×10⁻¹³ | 1,817 vs 1,611, RR 1.13 (1.10–1.16) |
| 27 off vs 12 off | 0 vs 0; one-sided rise p = 1 | 221 vs 20, RR 11.1 (7.0–17.4) | 1,817 vs 1,832, p = 0.41 |
| 27 off vs L56/N16758 off alone | 0 vs 312 | 221 vs 18, RR 12.3 (7.6–19.8) | 1,817 vs 1,650 |
| 11 off vs unmodified | 0 vs 287 (p = 8×10⁻⁸⁷) | 100 vs 103, RR 0.97 (0.74–1.27), p = 0.85 | 1,817 vs 1,611, RR 1.13 (1.10–1.16) |
| 11 off vs 12 off | 0 vs 0; one-sided rise p = 1 | 100 vs 20, RR 5.0 (3.1–8.1) | 1,817 vs 1,832, p = 0.25 |
| 27 off vs 11 off | 0 vs 0 | 221 vs 100, p = 2×10⁻¹⁴ | 1,817 vs 1,817 |

The p-values are exact two-sided McNemar tests on matched runs. The RRs use Katz 95% intervals, which ignore the pairing, so they are conservative.

**Where the calls went (exploratory).** Paired runs diverge after the first differing token, so these are descriptive only.
- **The 287 runs where the unmodified model called the proxy:** the matched 27-off runs called www.sec.gov in 260, Wayback in 34, efts.sec.gov in 8, another host in 14, and nothing in 22. With the 12 off, the same runs called www.sec.gov in 276 and Wayback in 3.
- **Most of the new Wayback runs:** they come from runs that called efts.sec.gov unmodified, 100 of 798.
- **Wayback as a second call:** under the 27, Wayback mostly comes alongside www.sec.gov, in 180 of 221 Wayback runs. Unmodified, that was 61 of 103.

## L56/N16758 restored on the held-out http:// sets
This check uses the drop-one design from the neuron-roles follow-up (item 1), with the same sets, seeds, sampler and sample sizes. It runs two arms: all 28 off, and L56/N16758 switched back on (27 off).

| Set | 28 off (this job) | 28 off (job A1) | 27 off (L56/N16758 restored) | One-sided p | Exact rate, 27 vs 28 |
|---|---|---|---|---|---|
| Escape | 0/128 | 0/128 | 0/128 | 1 | 0.000% vs 0.000% |
| Natural http:// | 0/192 | 0/192 | 0/192 | 1 | 0.000% vs 0.000% |
| SEC substituted | 0/480 | 0/480 | 0/480 | 1 | 0.050% vs 0.000% |
| Docs substituted | 12/1,568 | 12/1,568 | 9/1,568 | 1 | 0.285% vs 0.434% |

- **No sampled rise on any set.** On docs pages, restoring it removes 3 of the baseline's 12 proxy completions and adds none.
- **One borderline exact-rate rise.** Under the frozen item-1 rule, L56/N16758 counts as "needed" on SEC substituted. That rests only on a 0.0505-point rise in the exact expected rate, just over the 0.05-point tolerance.
  - **Source:** two of the 30 prefixes, where the proxy's first token moves just inside the sampler's kept set, to 1.05% and 0.46%.
  - **Docs pages:** it lowers the exact rate by 0.15 points.
  - **Raw probability:** it falls on the escape and the natural set.
  - **Under this phase's rule:** only a sampled rise counts against shipping, so this is reported as a cost of at most about 0.05 points on one set.
- **Reproducibility:** this job's 28 baseline reproduced item 1's counts exactly.
  - **Sampled outputs:** 85 of 86 prefixes match jobs A1 and A2. The one docs prefix that differs is the same one on which A1 and A2 disagreed.
  - **Teacher-forced values:** they match exactly.

**Wayback at the 102 decision points (exploratory).** These are teacher-forced P(web.archive.org) at the first Wayback decision of each unmodified rerun run.
- **Containers:** job B's arms are from the neuron-roles profile job, and the last two rows are from this job.
- **Reading the last column:** it is the mean change in log P against unmodified. The 27 row's −0.31 means some prefixes drop even though the mean probability is unchanged.

| Arm | Mean P(web.archive.org) | After the sampler cut | Mean change in log P vs unmodified (95% CI) |
|---|---|---|---|
| Unmodified (job B) | 62.8% | 65.1% | — |
| L56/N16758 off (job B) | 34.0% | 34.9% | −0.91 (−1.01 to −0.80) |
| 12 off (job B) | 28.7% | 29.2% | −1.68 (−1.95 to −1.42) |
| 11 off (job B) | 55.5% | 57.3% | −0.38 (−0.52 to −0.25) |
| 28 off (this job) | 32.9% | 33.1% | −1.91 (−2.29 to −1.55) |
| 27 off (this job) | 64.5% | 65.8% | −0.31 (−0.50 to −0.14) |

At the decision itself:
- **The 28** cut the Wayback probability about as much as the 12 do.
- **The 27** leave it where it was.

At the run level, more runs reach a Wayback decision under the 27, because the efts.sec.gov and r.jina.ai routes shrink.

## The 12 without L56/N16758 (11 off)
- **Proxy calls:** 0 of 2,048 (Wilson 95% upper bound 0.19%). The post's "12 stopped every call in 2,048 runs" also holds for 11.
- **Wayback calls:** 100 of 2,048, against 103 unmodified (RR 0.97, 95% CI 0.74–1.27). The frozen verdict is "gone", with no rise. So L56/N16758 accounts for the 12's Wayback cut, from 103 to 20.
- **The rise under the 27 comes from the 16 http:// neurons.** The 27 and the 11 differ only by those 16.
  - **Wayback:** 221 against 100 (p = 2×10⁻¹⁴).
  - **efts.sec.gov and r.jina.ai:** these fall further, to 68 from 141 and to 13 from 44.
  - **Fit with earlier findings:** this matches item 3 of the neuron-roles follow-up, where L62/N8518, L60/N8520 and L63/N2250 push down `web`.

## What ran
| Job | What | App time | Estimate |
|---|---|---|---|
| `patch27-all27-runs-1` | 27 off, 2,048 SEC 403 runs | 2,321 s | $3.54 |
| `patch27-restore-1` | 28 off and L56/N16758 restored on the 86 held-out http:// prefixes, plus Wayback teacher forcing | 853 s | $1.30 |
| `patch27-all11-runs-1` | 11 off, 2,048 SEC 403 runs | 2,631 s | $4.02 |

- **Rollout arms:** `qwen_lab.sparsity_scale.run` and `runner.generation`, unchanged.
- **Restore job:** `qwen_lab.neuron_roles.phase_restore`. Two changes, both recorded before generation:
  - an optional Wayback teacher-forced stage that runs after sampling;
  - a `patch27-` run-ID prefix.
- **Neuron-roles results:** they rebuild byte for byte after the code change.

## Caveats
- **One task:** this is the SEC 403 task only, and the Wayback Machine is a legitimate fallback. The rise is a change in behavior, not a leak.
- **The 28 themselves:** they have not been run on these 2,048 seeds. The 12 and the 27 have.
- **Conservative CIs:** Katz intervals ignore the pairing. The paired McNemar tests are given next to them.
- **Exploratory sections:** the "where the calls went" and Wayback decision-point sections were not pre-registered.
- **The "needed (SEC substituted)" label** for L56/N16758 rests on 0.0005 of exact rate over two prefixes.

## Cost
- **This phase:** $8.86 of the $16 cap. That is the whole-app estimate; measured in-container compute was $8.80.
- **Research total:** $188.42 of the $280 cap.

**Verdict (from the frozen decision rules):** the 27 can be used without L56/N16758; its Wayback suppression is gone (0 proxy calls in 2,048 runs), and Wayback calls rose (RR 2.15, 95% CI 1.71–2.69 against unmodified), so 'gone' means no longer suppressed, not unchanged.
