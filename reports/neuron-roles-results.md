# Neuron roles in the 28-neuron patch: drop-one, activation profiles and alternative destinations

Oct 30–31, 2026. This follow-up tests the role claims made after the explanation phase. Those rested on the output-token lens plus single ablations; this rerun adds three measurements:

- **Drop-one:** restore each of the 16 http:// neurons on held-out http:// failures.
- **Activation profiles:** where each of the 28 neurons fires.
- **Effects on alternative destinations:** each neuron's effect on alternative first tokens, including a test of the Wayback guess for L56/N16758.

Role statements below are operational: measured effects and activation patterns, not semantic labels. The protocol, `data/neuron-roles/protocol.json`, was frozen before any generation; the amendments are listed under "What ran".

**Every number comes from `neuron-roles-results.json`.** It is built with `python -m qwen_lab.neuron_roles build`, and the tables are rendered with `python -m qwen_lab.neuron_roles tables`.

## Headline
- **Drop-one on the 16 http:// neurons:** two are passengers on the held-out http:// sets. Restoring L63/N7463 or L60/N5507 alone, with the other 27 still off, gives no significant rise in sampled proxy completions and no measurable rise in the expected rate.
  - **Clearly needed (four search-2 neurons):** L63/N6311, L62/N16573, L63/N6342 and L63/N1945. Their sampled rises on docs pages survive Holm correction across all 64 tests.
  - **Needed with significant unadjusted rises (three):** L51/N8299, L61/N5269 and L62/N8518.
  - **Needed only by small expected-rate rises (seven):** 0.05 to 0.25 percentage points, with no significant sampled rise.
- **Trigger types:**
  - **Fire after failures in both schemes:** 4 of the original 12.
  - **http:// only:** L51/N8299 (+34 after http://), L61/N5269 and three others.
  - **Fire at every URL decision, including clean fetches:** L62/N4422.
  - **Site-specific:** several suppressors track the requested site and fire at the first fetch, before any failure.
    - `www` sites: L63/N4455, L62/N10856, L63/N7463.
    - MDN: L63/N6311.
    - PyTorch: L63/N1945, L62/N16573, L63/N6342.
- **Alternative suppression, operational:**
  - **Requested site's first token:**
    - **`www`:** L63/N4455 (−2.9 logits on https:// failures), L62/N10856 and L63/N7463.
    - **MDN's `developer`:** L63/N6311.
    - **PyTorch's `py`:** L63/N1945, L62/N16573 and L63/N6342.
  - **Wayback's `web`:** L62/N8518, L60/N8520 and L63/N2250.
  - **Never suppressed:** efts.sec.gov, docs.python.org and en.wikipedia.org.
- **Wayback (L56/N16758):** the guess holds.
  - **Direct effect:** it raises `archive`, `Way` and `web`.
  - **Activation:** it is most active at Wayback-bound decisions (+27.9).
  - **At 102 Wayback-bound decision points:** switching it off cuts web.archive.org completions from 2,110 to 1,124 of 3,264.
  - **Across 2,048 full SEC runs:** Wayback calls fall from 103 to 18 (−82%, 95% CI 71–89%).
  - **Proxy calls don't fall:** 312 against 287 (p = 0.17).

## Item 1: drop-one on the 16 http:// neurons

**Design:**
- **Arms:** with all 28 off, one http:// neuron at a time was switched back on (27 off).
- **Measure:** proxy-host completion at the decision token, with the deployed sampler (24 new tokens).
- **Seeds:** the same seeds in every arm.
- **Sets:** the held-out round-3 sets.
- **Teacher forcing:** every prefix also got the exact expected sampled rate, the proxy-host probability after the sampler's top-k/top-p cut.

| Neuron restored | Escape | Natural http:// | SEC substituted | Docs substituted | Exact rate, docs (restored vs 28) | Call |
|---|---|---|---|---|---|---|
| L51/N8299 | 10/128 (p=0.00098) | 1/192 | 7/480 (p=0.0078) | 25/1568 (p=0.0022) | 1.07% vs 0.43% | needed (E, N, S, D) |
| L61/N5269 | 0/128 | 0/192 | 0/480 | 22/1568 (p=0.00098) | 1.00% vs 0.43% | needed (S, D) |
| L63/N9024 | 0/128 | 0/192 | 0/480 | 15/1568 | 0.65% vs 0.43% | needed (D) |
| L62/N8518 | 5/128 (p=0.031) | 2/192 | 0/480 | 16/1568 | 0.75% vs 0.43% | needed (E, N, D) |
| L50/N14454 | 0/128 | 0/192 | 0/480 | 11/1568 | 0.47% vs 0.43% | needed (S) |
| L60/N8520 | 0/128 | 0/192 | 0/480 | 14/1568 | 0.59% vs 0.43% | needed (S, D) |
| L63/N7463 | 0/128 | 0/192 | 0/480 | 12/1568 | 0.46% vs 0.43% | **passenger (redundant)** |
| L63/N6311 | 0/128 | 0/192 | 0/480 | 40/1568 (p=3.7e-09) | 2.39% vs 0.43% | needed (D) |
| L63/N2250 | 0/128 | 0/192 | 0/480 | 14/1568 | 0.68% vs 0.43% | needed (D) |
| L61/N9789 | 0/128 | 0/192 | 0/480 | 15/1568 | 0.58% vs 0.43% | needed (D) |
| L62/N4422 | 0/128 | 0/192 | 0/480 | 12/1568 | 0.49% vs 0.43% | needed (D) |
| L63/N1945 | 0/128 | 0/192 | 0/480 | 24/1568 (p=0.00024) | 1.12% vs 0.43% | needed (D) |
| L62/N16573 | 0/128 | 0/192 | 0/480 | 52/1568 (p=9.1e-13) | 2.67% vs 0.43% | needed (D) |
| L60/N5507 | 0/128 | 0/192 | 0/480 | 13/1568 | 0.46% vs 0.43% | **passenger (redundant)** |
| L61/N168 | 0/128 | 0/192 | 0/480 | 13/1568 | 0.45% vs 0.43% | needed (S) |
| L63/N6342 | 0/128 | 0/192 | 0/480 | 48/1568 (p=1.5e-11) | 2.54% vs 0.43% | needed (D) |
| **28 off (baseline)** | 0/128 | 0/192 | 0/480 | 12/1568 | 0.43% | — |

p-values are one-sided exact McNemar tests on matched seed/row units; only p < 0.05 is shown. A neuron counts as **needed** if, on some set, either:
- the sampled rise is significant, or
- the mean exact rate rises by at least 0.05 percentage points.

How the 14 needed neurons split:

- **Strong, sampled rise survives Holm across 64 tests:** L63/N6311 (Holm p = 2×10⁻⁷), L62/N16573 (6×10⁻¹¹), L63/N6342 (9×10⁻¹⁰) and L63/N1945 (0.015). All four rise on docs pages.
- **Significant unadjusted sampled rise:**
  - L51/N8299: escape 10/128, SEC 7/480, docs 25/1,568; Holm 0.06 at best.
  - L61/N5269: docs 22/1,568.
  - L62/N8518: escape 5/128.
- **Exact-rate rule only, no significant sampled rise:**

  | Neuron | Set | Exact-rate rise |
  |---|---|---|
  | L63/N2250 | docs | +0.25 pt |
  | L63/N9024 | docs | +0.22 pt |
  | L61/N9789 | docs | +0.15 pt |
  | L60/N8520 | docs | +0.15 pt |
  | L60/N8520 | SEC | +0.13 pt |
  | L61/N168 | SEC | +0.08 pt |
  | L50/N14454 | SEC | +0.07 pt |
  | L62/N4422 | docs | +0.06 pt |

- **Passengers:** L63/N7463 and L60/N5507. Both count as **redundant** rather than inert: restoring them raises the raw, uncut proxy probability by a little more than 0.05 points (SEC +0.12 and +0.06), but not past the sampler's cut while the other 27 are off.
- **Escape and natural failures:** apart from L51/N8299 and L62/N8518, no single restore lets the escape or the natural http:// failures through. Those sets have redundancy to spare.
- **Cross-container determinism:** each job ran the 28 baseline. The two baselines gave identical teacher-forced values and identical proxy counts. Sampled rows matched on 85 of 86 prefixes; on the 86th, 2 of 32 Wikipedia rows swapped en.wikipedia.org and en.m.wikipedia.org. Every comparison above uses the same job's baseline.

**Single ablations from the unmodified model** (job B, teacher-forced exact expected rates; covers the nine search-2 neurons and the rest of the 28):

| Neuron off alone | Escape | Natural http:// | SEC substituted | Docs substituted |
|---|---|---|---|---|
| (unmodified) | 100.0% | 79.9% | 89.9% | 78.9% |
| L63/N417 | 100.0% | 79.9% | 86.5% | 77.4% |
| L63/N14734 | 100.0% | 79.9% | 89.2% | 78.9% |
| L63/N4455 | 100.0% | 77.8% | 88.8% | 78.0% |
| L54/N10344 | 100.0% | 79.7% | 90.2% | 77.7% |
| L62/N6870 | 100.0% | 58.8% | 78.4% | 71.3% |
| L63/N3149 | 100.0% | 75.5% | 90.0% | 74.8% |
| L62/N13617 | 95.8% | 63.6% | 82.7% | 71.5% |
| L54/N6172 | 100.0% | 70.5% | 82.1% | 71.1% |
| L62/N10856 | 100.0% | 78.7% | 89.6% | 78.9% |
| L63/N10126 | 94.7% | 60.7% | 80.3% | 65.7% |
| L56/N16758 | 100.0% | 92.2% | 90.4% | 82.6% |
| L58/N11828 | 100.0% | 77.5% | 89.2% | 79.2% |
| L51/N8299 | 88.1% | 55.4% | 76.7% | 52.7% |
| L61/N5269 | 94.7% | 62.2% | 82.2% | 70.7% |
| L63/N9024 | 100.0% | 73.2% | 87.7% | 74.8% |
| L62/N8518 | 100.0% | 73.6% | 89.6% | 76.8% |
| L50/N14454 | 100.0% | 79.0% | 87.9% | 70.3% |
| L60/N8520 | 100.0% | 79.3% | 89.4% | 79.5% |
| L63/N7463 | 100.0% | 78.1% | 88.9% | 78.7% |
| L63/N6311 | 100.0% | 79.9% | 90.0% | 66.5% |
| L63/N2250 | 100.0% | 79.7% | 90.1% | 78.6% |
| L61/N9789 | 100.0% | 79.9% | 90.1% | 78.9% |
| L62/N4422 | 100.0% | 79.2% | 89.6% | 78.5% |
| L63/N1945 | 100.0% | 79.9% | 90.0% | 74.0% |
| L62/N16573 | 100.0% | 80.1% | 90.0% | 78.1% |
| L60/N5507 | 100.0% | 78.0% | 89.7% | 78.9% |
| L61/N168 | 100.0% | 79.2% | 89.8% | 79.0% |
| L63/N6342 | 100.0% | 80.1% | 90.0% | 76.5% |

- **Search-2 neurons on their own:** only L63/N6311 (docs −12.4 points) and L63/N1945 (docs −4.9) move a set by more than 2.5 points. None moves the escape or the SEC set.
- **L56/N16758 alone raises the proxy rate:** natural http:// failures go from 79.9% to 92.2%. See the Wayback section.
- **L54/N10344 does nothing here too,** matching what the post says about it.

## Item 2: where each neuron fires

**Measurement:** the down_proj input, the value the patch zeroes, at the decision position (`scheme://` before the host) in the unmodified model. Each cell is the mean, with the SD in parentheses.

**Families:**

| Family | Contexts |
|---|---|
| (a) 403 failures, https:// | the 16 traces behind the post's −17.9 figure plus 45 held-out and monitor prefixes |
| (b) 403 failures, http:// | 158: natural, SEC substituted and docs substituted |
| (c) no failure | 32 model-written asyncio:fresh first fetches plus the 7 distinct synthetic first fetches in the prompts |
| Plain-text URL | 32 pathlib:recall answers |
| Wayback | 102 Wayback-bound decisions |
| (d) prose | 1,295 reasoning positions at least 32 tokens from any URL |

| Neuron | 403 https:// (a) | 403 http:// (b) | No failure (c) | Plain-text URL | Wayback decisions | Prose (d) | Label |
|---|---|---|---|---|---|---|---|
| L63/N417 | -0.65 (0.14) | -1.22 (0.48) | -1.36 (0.30) | -1.65 (0.37) | -0.70 (0.21) | -0.95 (0.95) | indeterminate; after r: URL-general |
| L63/N14734 | +0.01 (0.00) | +0.00 (0.02) | -0.05 (0.09) | -0.00 (0.01) | +0.02 (0.01) | +0.01 (0.13) | indeterminate; after r: URL-general |
| L63/N4455 | -17.77 (8.19) | -0.32 (2.14) | -2.09 (5.81) | +0.22 (0.58) | -1.95 (2.73) | -0.31 (0.78) | failure-specific (https:// only) |
| L54/N10344 | -0.41 (0.26) | +3.06 (2.98) | +0.15 (1.40) | -0.08 (0.10) | +0.22 (0.41) | +0.12 (0.82) | http-specific |
| L62/N6870 | -12.99 (1.57) | -12.97 (6.34) | +1.04 (1.02) | -0.54 (1.36) | -9.53 (2.45) | -1.11 (3.00) | failure-specific |
| L63/N3149 | -5.54 (1.38) | -1.53 (2.07) | +0.65 (0.92) | -0.07 (0.49) | -6.28 (1.51) | -1.14 (1.30) | failure-specific (https:// only) |
| L62/N13617 | -11.16 (3.81) | -19.82 (8.00) | +0.47 (1.29) | +0.89 (0.59) | -8.44 (4.33) | -0.30 (2.35) | failure-specific |
| L54/N6172 | -19.23 (1.23) | -25.13 (5.14) | -0.10 (0.27) | +0.17 (0.06) | -19.75 (3.54) | -0.93 (1.93) | failure-specific |
| L62/N10856 | -10.32 (1.85) | -1.91 (1.71) | -0.68 (2.74) | -1.75 (0.64) | -7.38 (2.04) | -0.85 (1.89) | failure-specific (https:// only) |
| L63/N10126 | +10.08 (2.00) | +15.30 (3.45) | -0.29 (2.56) | -1.48 (1.11) | +7.43 (2.63) | -0.13 (0.74) | failure-specific |
| L56/N16758 | +8.02 (4.02) | +16.90 (8.69) | +0.21 (0.11) | -0.04 (0.19) | +27.93 (10.35) | +4.36 (8.64) | indeterminate |
| L58/N11828 | +7.46 (1.08) | +3.41 (3.54) | -0.00 (0.29) | +0.01 (0.03) | +3.92 (2.04) | +0.36 (0.75) | failure-specific (https:// only) |
| L51/N8299 | -0.59 (0.22) | +34.34 (4.61) | -0.07 (0.11) | +0.09 (0.12) | +2.76 (11.05) | -0.08 (0.53) | http-specific |
| L61/N5269 | -3.08 (2.59) | -12.90 (6.62) | +1.12 (0.40) | +1.06 (0.16) | -5.67 (3.83) | +0.56 (1.15) | http-specific |
| L63/N9024 | -1.60 (0.18) | -1.60 (0.28) | -0.97 (0.27) | -0.66 (0.17) | -1.75 (0.21) | -0.12 (0.55) | indeterminate |
| L62/N8518 | -0.20 (1.95) | -2.53 (3.68) | -0.00 (0.22) | -0.10 (0.39) | -7.33 (3.39) | -1.18 (4.41) | indeterminate |
| L50/N14454 | +4.99 (0.39) | -9.20 (1.95) | -0.03 (0.04) | +0.08 (0.21) | +3.51 (4.31) | -0.02 (0.21) | indeterminate |
| L60/N8520 | +1.90 (2.61) | +6.83 (4.49) | -1.06 (0.99) | +0.02 (0.17) | +8.32 (3.78) | +1.99 (4.26) | indeterminate |
| L63/N7463 | -4.94 (1.99) | -1.42 (2.11) | +0.02 (2.80) | +0.77 (0.22) | -0.99 (1.44) | -0.01 (0.98) | failure-specific (https:// only) |
| L63/N6311 | +0.11 (0.08) | -3.02 (6.59) | -0.87 (3.40) | +0.22 (0.07) | +0.17 (0.09) | +0.02 (0.38) | http-specific |
| L63/N2250 | +0.27 (1.82) | -0.89 (1.90) | -4.18 (3.24) | -1.03 (2.12) | +4.14 (3.28) | +0.33 (5.01) | indeterminate |
| L61/N9789 | +0.86 (1.18) | +3.66 (3.69) | -0.49 (0.43) | -0.08 (0.22) | -0.11 (1.22) | +0.01 (0.31) | http-specific |
| L62/N4422 | -13.57 (2.16) | -14.88 (5.19) | -12.12 (5.83) | -4.26 (3.74) | -11.11 (1.72) | -2.30 (3.25) | URL-general |
| L63/N1945 | -0.57 (0.08) | -1.46 (4.43) | -0.99 (1.51) | -0.56 (0.08) | -0.59 (0.39) | -0.50 (0.23) | indeterminate |
| L62/N16573 | +0.03 (0.04) | -2.78 (6.60) | -4.96 (5.23) | -0.86 (0.66) | -0.01 (0.39) | +0.08 (1.11) | URL-general |
| L60/N5507 | -7.97 (0.99) | -2.17 (1.85) | -0.04 (0.63) | -0.06 (0.06) | -4.65 (1.45) | -0.44 (1.18) | failure-specific (https:// only) |
| L61/N168 | +2.88 (1.04) | +0.70 (1.10) | +0.33 (0.98) | +0.31 (0.39) | +0.32 (0.82) | +0.14 (1.47) | indeterminate |
| L63/N6342 | -0.88 (0.34) | -1.58 (1.42) | +2.09 (1.72) | +0.58 (1.09) | -1.28 (0.23) | -1.05 (0.87) | URL-general |

- **N4455 lines up with the post.** On the same 16 traces, at the position that predicts `r`, its mean is **−17.89** (the post says −17.86). The traces are rebuilt to end at the proxy token, so bf16 numerics differ slightly.
- **The two finishers are quiet at the decision and fire on the next position, after `r`, in every context.**

  | Neuron | https:// failures | http:// failures | Clean fetches with `r` forced | Plain-text URLs |
  |---|---|---|---|---|
  | L63/N417 | +13.4 | +15.7 | +17.7 | +7.2 |
  | L63/N14734 | −83.2 | −68.0 | −70.6 | −75.5 |

  So they respond to `r` after `://` regardless of a failure: they finish the proxy's spelling rather than signal a failure.
- **How to read the labels.** They follow the frozen rule:
  - each family's shift from prose must be at least max(3 × prose SD, 1);
  - the label is chosen from the shifts in families a, b and c.

  The rule has two blind spots:
  - **High prose variance hides activity.** It makes four neurons "indeterminate" even where their activity is concentrated:
    - L56/N16758: prose SD 8.6, Wayback decisions +27.9;
    - L62/N8518: prose SD 4.4, Wayback −7.3;
    - L60/N8520;
    - L63/N2250.
  - **"No failure" mixes very different contexts.** It pools 32 docs.python.org first fetches with 7 synthetic first fetches. "Failure-specific (https:// only)" therefore means: among decisions after a failure, the neuron fires after https://. It does not mean the neuron is silent without a failure. The table below shows the difference.

**Activation at the synthetic first fetch, before any failure, by requested site** (exploratory; the number of contexts sharing each prompt is in the JSON):

| Neuron | mozilla | pathlib | pytorch | reuters | sec | sec_rules | wikipedia |
|---|---|---|---|---|---|---|---|
| L63/N417 | -1.6 | -1.7 | -2.1 | -1.4 | -1.2 | -0.8 | -1.4 |
| L63/N14734 | +0.0 | -0.1 | +0.1 | -0.0 | -0.0 | -0.0 | -0.0 |
| L63/N4455 | -1.2 | +0.0 | +0.1 | **-14.1** | **-11.4** | **-33.2** | -1.1 |
| L54/N10344 | **+8.7** | -0.1 | -0.2 | -0.1 | +0.1 | -0.0 | -0.2 |
| L62/N6870 | +0.7 | +0.7 | +1.2 | -3.4 | -2.3 | -0.7 | +0.9 |
| L63/N3149 | +0.5 | -3.9 | +1.0 | -0.5 | -1.4 | -1.1 | +1.2 |
| L62/N13617 | -6.5 | +0.9 | +0.7 | +1.4 | +0.1 | -2.5 | +1.6 |
| L54/N6172 | +0.1 | -0.2 | +0.0 | +0.0 | +0.5 | -1.5 | +0.3 |
| L62/N10856 | +0.3 | -1.2 | +0.4 | -4.8 | -5.7 | **-15.6** | +0.7 |
| L63/N10126 | -2.5 | -0.7 | -0.6 | +7.9 | +6.0 | +9.0 | -1.3 |
| L56/N16758 | -0.2 | +0.3 | +0.4 | +0.2 | -0.1 | +0.3 | +0.1 |
| L58/N11828 | -0.4 | +0.2 | +1.3 | -0.1 | +0.1 | +1.0 | -0.3 |
| L51/N8299 | +0.2 | -0.1 | +0.0 | -0.1 | +0.3 | +0.2 | -0.3 |
| L61/N5269 | +0.5 | +0.7 | +0.4 | +0.1 | +0.9 | +1.5 | +1.1 |
| L63/N9024 | -1.3 | -1.0 | -1.2 | -1.6 | -0.8 | -1.5 | -1.2 |
| L62/N8518 | +0.8 | +0.0 | -0.7 | -0.6 | -0.0 | -0.3 | -0.0 |
| L50/N14454 | -0.1 | -0.1 | -0.0 | -0.1 | -0.1 | +0.0 | -0.1 |
| L60/N8520 | -0.7 | -2.0 | -2.2 | +1.6 | +0.2 | -2.7 | +0.6 |
| L63/N7463 | +0.4 | +1.1 | +0.2 | -9.1 | -2.4 | **-13.7** | +0.3 |
| L63/N6311 | **-21.5** | -0.1 | +0.4 | -0.0 | +0.0 | +0.1 | -0.8 |
| L63/N2250 | -4.1 | -5.6 | +1.9 | +1.3 | +1.1 | +0.9 | -1.4 |
| L61/N9789 | +0.0 | -0.6 | +0.9 | -0.4 | -0.8 | -0.9 | +0.4 |
| L62/N4422 | -4.3 | -4.0 | **-19.9** | -5.5 | -4.9 | **-18.5** | **-20.0** |
| L63/N1945 | -0.5 | -0.8 | **-9.7** | -0.5 | -1.0 | -0.3 | -1.3 |
| L62/N16573 | +0.4 | +0.2 | **-9.8** | +0.3 | +0.1 | -0.1 | +0.4 |
| L60/N5507 | -0.2 | +0.4 | +0.0 | -0.9 | -0.1 | +0.5 | -3.7 |
| L61/N168 | +0.9 | -2.2 | +0.8 | -3.6 | -2.8 | -0.9 | +1.1 |
| L63/N6342 | -0.9 | +0.6 | **+5.7** | -1.2 | +1.0 | -1.7 | -1.5 |

The suppressors from item 3 already fire at the first fetch, when the requested site comes next. L63/N4455, L62/N10856 and L63/N7463 fire for `www` sites, L63/N6311 for MDN, and L63/N1945 and L62/N16573 for PyTorch.

**Their combined direct effect on the requested site's first token is negative at those first fetches:**

| Site | Token | Combined effect (logits) | Main contributors |
|---|---|---|---|
| SEC rules | `www` | −5.9 | N4455 −4.3 |
| MDN | `developer` | −2.7 | N6311 −2.6 |
| PyTorch | `py` | −1.0 | N1945 −1.0, N16573 −0.5, N6342 +0.8 |

- **Before a failure:** the model still copies the requested URL, so this suppression doesn't change the first fetch.
- **After a 403:** the same suppression is one way the proxy wins.
- **For the patch:** zeroing the 28 directly lifts the requested site at clean first fetches.

## Item 3: effects on alternative destinations

**Direct effect** is the exact change in each alternative first token's logit at the decision position when the neuron's contribution is removed from the final residual. It passes through the real final norm; later layers aren't rerun. The table lists up to three tokens with |effect| ≥ 0.1 logits. Negative means the neuron lowers that token.

| Neuron | SEC substituted (S) | Docs substituted (D) | Natural http:// (N) | https:// held out | Wayback decisions |
|---|---|---|---|---|---|
| L63/N417 | — | — | — | — | — |
| L63/N14734 | — | — | — | — | — |
| L63/N4455 | — | — | — | www -2.90, web -0.24, developer -0.11 | www -0.29 |
| L54/N10344 | — | — | — | — | — |
| L62/N6870 | r +2.85 | r +1.16 | r +2.34 | r +1.61 | r +1.24 |
| L63/N3149 | — | r +0.29 | r +0.21 | r +0.56 | r +0.66 |
| L62/N13617 | r +2.17, efts -0.16, www -0.10 | r +1.20 | r +1.85, efts -0.13 | r +0.84 | r +0.63 |
| L54/N6172 | en +0.40, docs +0.31, developer -0.28 | en +0.25, docs +0.21, developer -0.20 | en +0.39, docs +0.30, developer -0.29 | en +0.23, docs +0.18, developer -0.14 | en +0.23, docs +0.18, developer -0.17 |
| L62/N10856 | — | www -0.13 | www -0.23 | www -0.68, docs -0.12 | www -0.48 |
| L63/N10126 | r +1.20, data -0.19, developer -0.13 | r +0.92, data -0.15, developer -0.11 | r +1.03, data -0.16, developer -0.11 | r +0.63, data -0.10 | r +0.47 |
| L56/N16758 | Way +0.64, archive +0.54, web +0.21 | Way +0.79, archive +0.65, web +0.23 | Way +1.13, archive +0.90, web +0.32 | Way +0.32, archive +0.27, web +0.10 | Way +1.11, archive +0.90, web +0.26 |
| L58/N11828 | r +0.12 | — | — | r +0.11 | — |
| L51/N8299 | en +0.42, www +0.39, archive +0.31 | www +0.37, en +0.36, archive +0.29 | en +0.44, www +0.41, archive +0.30 | — | — |
| L61/N5269 | r +1.22, data -0.11 | r +0.53 | r +1.07 | r +0.19 | r +0.34 |
| L63/N9024 | r +0.36 | r +0.28 | r +0.39 | r +0.30 | r +0.34 |
| L62/N8518 | web -0.12 | web -0.61 | web -0.89 | — | web -1.14 |
| L50/N14454 | www -0.12 | www -0.14 | www -0.13 | — | — |
| L60/N8520 | www -0.38, web -0.37 | www -0.50, web -0.48, docs -0.10 | www -0.57, web -0.55, docs -0.12 | www -0.14, web -0.14 | www -0.55, web -0.53, docs -0.11 |
| L63/N7463 | www -0.21 | — | www -0.18 | www -0.28 | — |
| L63/N6311 | — | developer -0.92 | — | — | — |
| L63/N2250 | — | web +0.24 | — | — | web -0.56 |
| L61/N9789 | — | r +0.33 | — | — | — |
| L62/N4422 | developer -0.16, efts -0.11 | — | developer -0.14, efts -0.10 | developer -0.11 | — |
| L63/N1945 | — | py -0.25 | — | — | — |
| L62/N16573 | — | py -0.28 | — | — | — |
| L60/N5507 | — | — | — | — | — |
| L61/N168 | — | — | — | www -0.28 | — |
| L63/N6342 | py -0.11 | py -0.34 | py -0.15 | py -0.12 | py -0.19 |

**The four main patterns:**

- **Push `r` directly:**

  | Neuron | Effect on `r` (logits) |
  |---|---|
  | L62/N6870 | +1.2 to +2.9 |
  | L62/N13617 | +0.6 to +2.2 |
  | L63/N10126 | +0.5 to +1.2 |
  | L61/N5269 | +0.2 to +1.2 |
  | L63/N3149 | +0.2 to +0.7 |
  | L63/N9024 | +0.3 to +0.4 |
  | L61/N9789 | +0.3 (docs pages) |
  | L58/N11828 | +0.1 |

- **Lower the requested site's first token:**
  - `www`: L63/N4455 (−2.9, https:// only), L62/N10856, L63/N7463, L61/N168, L60/N8520 and L50/N14454;
  - `developer`: L63/N6311;
  - `py`: L63/N6342, L63/N1945 and L62/N16573.
- **Lower `web` (web.archive.org, and webcache, which shares the token):** L62/N8518, L60/N8520, and L63/N2250 at Wayback-bound decisions.
- **Raise alternatives directly:** L56/N16758 (`archive`, `Way`, `web`), L51/N8299 (`www`, `en`, `archive`) and L54/N6172 (`en`, `docs`).
  - These three reduce proxy calls in other ways. Their direct effects don't explain it.

**Total effects (singles).** Switching off a neuron that pushes `r` lifts every alternative, because probability leaves `r`. That trips the pre-registered "suppresses" flag (at least 0.1 nats on at least 75% of prefixes) for almost every token, so the flags are in the JSON but don't separate one alternative from another.

The table below is an exploratory alternative. It uses log P(token | not `r`) on each requested site's own first token, and gives the change in P(`r`) after the slash. Cells are mean change in nats when the neuron is off / change in P(`r`). Only neurons with a cell above 0.1 nats or 0.1 in P(`r`) are listed:

| Neuron | S:sec (www) | D:mozilla (developer) | D:pytorch (py) | D:sec_rules (www) | D:wikipedia (en) |
|---|---|---|---|---|---|
| L63/N6311 | -0.00 / +0.00 | **+0.33 / -0.36** | -0.00 / +0.00 | -0.00 / -0.00 | -0.00 / -0.00 |
| L63/N1945 | -0.01 / +0.00 | -0.00 / -0.00 | **+0.18 / -0.14** | +0.00 / +0.00 | -0.00 / -0.00 |
| L63/N6342 | -0.00 / +0.00 | -0.00 / -0.00 | **+0.11 / -0.07** | +0.01 / -0.00 | -0.00 / -0.00 |
| L62/N16573 | -0.00 / -0.00 | -0.00 / -0.00 | +0.10 / -0.02 | -0.00 / -0.00 | -0.00 / -0.00 |
| L54/N6172 | **+0.39** / -0.07 | +0.06 / -0.04 | +0.14 / -0.05 | +0.31 / -0.05 | -0.03 / -0.22 |
| L51/N8299 | +0.31 / -0.09 | -0.11 / -0.16 | -0.42 / -0.18 | -1.09 / -0.18 | +0.08 / **-0.61** |
| L50/N14454 | +0.07 / -0.01 | -0.16 / -0.02 | -0.20 / -0.03 | -0.26 / +0.02 | +0.07 / **-0.42** |
| L56/N16758 | +0.17 / +0.00 | +0.15 / -0.00 | +0.16 / +0.02 | **+0.86 / +0.17** | +0.00 / +0.01 |
| L60/N8520 | +0.07 / -0.01 | +0.01 / +0.00 | -0.00 / +0.00 | +0.49 / +0.01 | -0.01 / +0.01 |
| L62/N6870 | -0.05 / -0.12 | -0.01 / -0.03 | -0.01 / -0.03 | +0.00 / -0.08 | -0.00 / -0.25 |
| L63/N10126 | -0.00 / -0.10 | +0.02 / -0.10 | +0.00 / -0.14 | -0.02 / -0.17 | +0.00 / -0.15 |
| L62/N13617 | -0.04 / -0.07 | -0.05 / -0.05 | -0.05 / -0.06 | -0.03 / -0.12 | -0.01 / -0.10 |
| L61/N5269 | -0.05 / -0.08 | -0.05 / -0.06 | -0.05 / -0.12 | -0.04 / -0.10 | -0.00 / -0.05 |

Full per-site values for all 28 neurons are in the JSON (`item3.by_site`).

**What each site shows:**

- **MDN:** switching off L63/N6311 raises `developer` on all 16 prefixes and cuts P(proxy) by 0.36.
- **PyTorch:** L63/N1945 and L63/N6342 do the same for `py`.
- **Wikipedia:** no neuron lifts `en` by more than 0.08. L51/N8299 and L50/N14454 cut P(proxy) by 0.61 and 0.42 there by working through the proxy itself.

**The brief's three specific alternatives:**

- **efts.sec.gov:** no neuron suppresses it. Direct effects on its first token (`eft`) are −0.16 at most, and conditional total effects are +0.06 at most. L54/N6172 favors it.
- **docs.python.org:** no neuron suppresses `docs` by more than 0.15. L62/N4422 reaches −0.15, at clean docs.python.org fetches.
- **en.wikipedia.org:** no neuron suppresses it.
- **Shared first tokens:**
  - r.jina.ai shares `r` with the proxy, so every `r` effect covers both. In option D, r.jina.ai calls didn't change (157 → 153).
  - webcache.googleusercontent.com shares `web` with web.archive.org.

## The Wayback test (L56/N16758)

**Decision points headed to the Wayback Machine.** There are 102 of them, from unmodified rerun runs that called web.archive.org; the outcome counted is completing the web.archive.org host.

| | Unmodified | L56/N16758 off | The 12 off | The 12 off, L56/N16758 on |
|---|---|---|---|---|
| Teacher-forced P(web.archive.org), mean | 62.8% | 34.0% | 28.7% | 55.5% |
| Mean change in log P vs unmodified (95% bootstrap CI) | — | −0.91 (−1.01 to −0.80) | −1.68 (−1.95 to −1.42) | −0.38 (−0.52 to −0.25) |
| Sampled completions to web.archive.org (32 per prefix) | 2,110 / 3,264 (64.6%) | 1,124 / 3,264 (34.4%) | — | — |

- **Matched seeds:** 990 completions left web.archive.org with L56/N16758 off and 4 moved to it (two-sided exact McNemar, p ≈ 5×10⁻²⁸⁹).
- **Where the displaced completions went:**

  | Destination | Unmodified | L56/N16758 off |
  |---|---|---|
  | www.sec.gov | 629 | 1,318 |
  | The proxy | 262 | 438 |
  | efts.sec.gov | 93 | 142 |
  | webcache | 67 | 124 |

- **L56/N16758 accounts for most of the 12's effect here.**

**Whole runs (option D).** L56/N16758 off for 2,048 SEC 403 runs, on the sparsity rerun's seeds, paired with its unmodified arm.

| Outcome (runs with a final-answer call) | L56/N16758 off | Unmodified | Matched change | Two-sided p |
|---|---|---|---|---|
| web.archive.org | **18 (0.9%, 0.6–1.4%)** | 103 (5.0%, 4.2–6.1%) | 97 runs lost it, 12 gained it | 1×10⁻¹⁷ |
| Proxy | 312 (15.2%) | 287 (14.0%) | 140 lost, 165 gained | 0.17 |
| r.jina.ai | 153 | 157 | 91 lost, 87 gained | 0.82 |
| efts.sec.gov | 846 | 798 | 254 lost, 302 gained | 0.046 |
| www.sec.gov | 1,650 | 1,611 | 163 lost, 202 gained | 0.047 |

- **Wayback calls drop by 82%** (Katz 95% CI 71–89%).
- **Truncated runs:** 1 in each arm. No unfinished calls.
- **The guess holds in operational terms.** L56/N16758 directly raises `archive`, `Way` and `web`, and it is most active at Wayback-bound decisions. Removing it nearly eliminates Wayback calls.
- **The effect is larger over whole runs (−82%) than at the decision point (−46% of completions).** This fits with it also firing during reasoning (prose SD 8.6).
- **On its own it doesn't reduce proxy calls.** In whole runs the proxy count doesn't change, and at decision points proxy completions rise. Its value inside the 12 is cutting an alternative route, not the proxy.

## Roles of all 28

**Passengers among the 16.** By the frozen drop-one rule, L63/N7463 and L60/N5507 do nothing measurable on the held-out http:// sets once the other 27 are off. Both are "redundant": they move the raw proxy probability slightly, but not into the sampler's range.

L54/N10344 was already a passenger among the original 12 (single and restore ablations in the explanation phase), and its single ablation here confirms that on http:// prefixes. So the count is:

> 28 neurons, and 3 of them (L54/N10344, L63/N7463, L60/N5507) appear to do nothing; 25 do measurable work.

Two caveats belong next to that sentence:
- seven of those 25 http:// neurons matter only by small amounts: 0.05 to 0.25 points of expected proxy rate on docs or SEC pages, with no significant sampled rise;
- "nothing" is relative to these held-out sets and the deployed sampler.

**Trigger types for all 28** (operational; frozen label, plus where the neuron fires and its main direct effect):

| Neuron | Set | Frozen label | Where it fires (activation at the decision position unless noted) | Main direct effect at the decision (logits) | Drop-one (16 only) |
|---|---|---|---|---|---|
| L63/N417 | https:// 12 (finisher) | indeterminate; URL-general after `r` | Quiet at the decision (−0.7 to −1.4). After `r`: +13.4 at https:// failures, +15.7 at http:// failures, +17.7 at clean fetches | Acts one position later, on `out` | — |
| L63/N14734 | https:// 12 (finisher) | indeterminate; URL-general after `r` | Quiet at the decision. After `r`: −83 at https:// failures, −68 at http:// failures, −71 at clean fetches | Acts one position later | — |
| L63/N4455 | https:// 12 | failure-specific (https:// only) | −17.8 at https:// failures, −0.3 after http://. Also −11 to −33 at the first fetch of `www` sites, before any failure | www −2.9 (https:// held out) | — |
| L54/N10344 | https:// 12 | http-specific | +3.1 after http://; +8.7 at MDN's first fetch | none ≥ 0.1 | — |
| L62/N6870 | https:// 12 | failure-specific | −13.0 at failures in both schemes; quiet at clean fetches | r +1.2 to +2.9 | — |
| L63/N3149 | https:// 12 | failure-specific (https:// only) | −5.5 https://, −1.5 http:// | r +0.2 to +0.7 | — |
| L62/N13617 | https:// 12 | failure-specific | −11.2 https://, −19.8 http:// | r +0.6 to +2.2 | — |
| L54/N6172 | https:// 12 | failure-specific | −19.2 https://, −25.1 http:// | Small and mixed (r −0.1 to −0.25); its proxy effect isn't direct | — |
| L62/N10856 | https:// 12 | failure-specific (https:// only) | −10.3 https://, −1.9 http://; −5 to −16 at first fetches of `www` sites | www −0.7 (https://), −0.1 to −0.2 (http://) | — |
| L63/N10126 | https:// 12 | failure-specific | +10.1 https://, +15.3 http://; +6 to +9 at first fetches of `www` sites | r +0.5 to +1.2 | — |
| L56/N16758 | https:// 12 | indeterminate (fires in reasoning too) | +8.0 https://, +16.9 http://, **+27.9 at Wayback decisions** | archive +0.3 to +0.9, Way +0.3 to +1.1, web +0.1 to +0.3 | — |
| L58/N11828 | https:// 12 | failure-specific (https:// only) | +7.5 https://, +3.4 http:// | r +0.1 | — |
| L51/N8299 | http:// search 1 | http-specific | +34.3 after http://, −0.6 after https:// | www, en, archive +0.3 to +0.5; its proxy effect isn't direct | needed, all sets |
| L61/N5269 | http:// search 1 | http-specific | −12.9 http://, −3.1 https:// | r +0.2 to +1.2 | needed (docs 22 vs 12) |
| L63/N9024 | http:// search 1 | indeterminate | −1.6 at failures, −1.0 at clean fetches; just under the 1.64 threshold | r +0.3 to +0.4 | small (docs +0.22 pt) |
| L62/N8518 | http:// search 1 | indeterminate | −7.3 at Wayback decisions | web −0.6 to −1.1 (−0.1 on SEC) | needed (escape 5/128) |
| L50/N14454 | http:// search 1 | indeterminate (opposite signs) | +5.0 https://, −9.2 http:// | www −0.1 | small (SEC +0.07 pt) |
| L60/N8520 | http:// search 1 | indeterminate | +6.8 http://, +8.3 at Wayback decisions | www and web −0.4 to −0.6 | small (SEC +0.13, docs +0.15 pt) |
| L63/N7463 | http:// search 1 | failure-specific (https:// only) | −4.9 https://, −1.4 http://; −2 to −14 at first fetches of `www` sites | www −0.2 to −0.3 | **passenger** |
| L63/N6311 | http:// search 2 | http-specific | −3.0 mean after http:// (MDN prefixes); −21.5 at MDN's first fetch | developer −0.9 (docs) | needed (docs 40 vs 12) |
| L63/N2250 | http:// search 2 | indeterminate | −4.2 at clean fetches, +4.1 at Wayback decisions | web −0.6 at Wayback decisions | small (docs +0.25 pt) |
| L61/N9789 | http:// search 2 | http-specific | +3.7 http://, +0.9 https:// | r +0.3 (docs) | small (docs +0.15 pt) |
| L62/N4422 | http:// search 2 | **URL-general** | −13.6 https://, −14.9 http://, −12.1 at clean fetches; −4 to −20 at every first fetch | Small negatives on most alternatives (≤ 0.16) | small (docs +0.06 pt) |
| L63/N1945 | http:// search 2 | indeterminate | Quiet on average; −9.7 at PyTorch's first fetch | py −0.25 (docs) | needed (docs 24 vs 12) |
| L62/N16573 | http:// search 2 | URL-general | −5.0 at clean fetches; −9.8 at PyTorch's first fetch | py −0.3 | needed (docs 52 vs 12) |
| L60/N5507 | http:// search 2 | failure-specific (https:// only) | −8.0 https://, −2.2 http://; −3.7 at Wikipedia's first fetch | none ≥ 0.1 | **passenger** |
| L61/N168 | http:// search 2 | indeterminate | +2.9 https:// (threshold 4.4) | www −0.3 (https://) | small (SEC +0.08 pt) |
| L63/N6342 | http:// search 2 | URL-general (by rule) | +2.1 at clean fetches, +5.7 at PyTorch's first fetch; −0.9 and −1.6 at failures | py −0.1 to −0.3 at failures | needed (docs 48 vs 12) |

**Alternative suppression, per neuron.** Measured, most of the 16 have an operational role:

- **Push `r`:**
  - L61/N5269 and L63/N9024 at every decision;
  - L61/N9789 on docs pages.
- **Lower the requested site:**
  - `www`: L63/N7463, L61/N168, L50/N14454 and L60/N8520;
  - MDN's `developer`: L63/N6311;
  - PyTorch's `py`: L63/N1945, L62/N16573 and L63/N6342. Switching them off raises `py` and cuts P(proxy) on PyTorch pages.
- **Lower `web`:** L62/N8518 and L60/N8520 at failures; L63/N2250 at Wayback-bound decisions.
- **Act on the proxy without a direct push:** L51/N8299, the strongest http:// neuron.
- **No operational role found:** L60/N5507, one of the two passengers.
- **Never suppressed:** efts.sec.gov, docs.python.org and en.wikipedia.org.

**"Didn't get in the way."** Three neurons fire at clean URL decisions too: L62/N4422 everywhere, and L62/N16573 and L63/N6342 for PyTorch and Python docs.

- **Direct effects at clean fetches:** small for L62/N4422 (≤ 0.15 logits, mostly negative). At the synthetic first fetches for SEC rules, MDN and PyTorch, the 28 together *lower* the requested site's first token by 1 to 6 logits; on Wikipedia the combined effect is about 0.
- **So zeroing them helps the requested site.**
- **Restoring L62/N4422 alone** changes held-out proxy completions by only 0.06 points.
- **Untested here:** clean PyTorch fetches weren't in the earlier suite. Direct effects say the patch favors `py` there on net, but nothing was sampled.

**Wayback.** L56/N16758 is a Wayback-fallback neuron, not a proxy neuron:

> Switching off L56/N16758 alone cut Wayback calls from 103 to 18 in 2,048 SEC runs (−82%, 95% CI 71–89%) without reducing proxy calls (312 vs 287). At the first token of a Wayback-bound URL it roughly halves the chance of completing web.archive.org (65% → 34% of 3,264 sampled continuations). Its direct output raises `archive`, `Way` and `web`.

## What ran
- **Model and sampler:** Qwen3.8-27B (revision 1d4bf0f2), BF16, Transformers 5.21.0, one H200 per job, with the deployed sampler (temperature 1, top-k 20, top-p 0.95). Tool calls stay text and are never executed.
- **Protocol:** frozen before any generation.
- **Amendments**, after the GPU smoke test and before any results job:
  1. **Exact direct effect added and made primary.** The frozen-scale version differed from exact removal by up to 0.46 logits where activations are large; it's still recorded.
  2. **Wider job time limits.** Sampling took about 4 s per 32-row batch; the plan assumed 2.9 s. This happened inside the same $25 cap.
  3. **Note on the rebuilt traces.** They reproduce the original values within bf16 differences, for example trace 0's N4455: −15.25 against −15.06.
- **Exploratory analyses (not pre-registered), all in the JSON:**
  - the "conditional on not `r`" total effects;
  - the per-site breakdowns;
  - the first-fetch-by-site table.
- **Jobs:**

  | Job | Arms or contents | Time | Cost |
  |---|---|---|---|
  | A1 | 28 baseline + 8 restores | 54 min | $4.96 |
  | A2 | 28 baseline + 8 restores | 56 min | $5.11 |
  | B | activations, singles, Wayback test | 26 min | $2.40 |
  | D | 2,048 runs | 40 min | $3.64 |
  | Smoke test | | | $0.12 |
  | D, first launch | failed after 7 s; the app didn't pass the job type | | $0.01 |

- **Seeds:**
  - item 1: 6,000,000 + 1000·i;
  - Wayback prefixes: 7,000,000 + 1000·i;
  - option D: the sparsity rerun's 4,000,000 + 32·b.

  A scan of 303 data files and run configs found no earlier use of any seed between 5M and 9M.

## Caveats
- **Narrow inputs.** Item 1 used held-out SEC and docs-page 403 prefixes with http:// at the decision. Option D used the SEC prompt only.
  - Passenger calls are relative to these sets and the deployed sampler.
  - The escape and natural http:// sets have so much redundancy that most single restores can't show up there.
- **Small-effect calls are close to the 0.05-point line.** Restoring L62/N4422 adds 0.055 points and L50/N14454 adds 0.065. A different threshold would move them.
- **Labels follow the frozen rule.** The per-site tables show where the rule's pooled "no failure" family and high prose variance hide site-specific firing.
- **Direct effects are logit contributions through the final norm only.** Neurons in layers 50–62 also act through later layers, which the singles' total effects capture.
- **One checkpoint and one harness** (BF16 Transformers with SDPA attention, H200). Decision-point results are prefix-level, not whole runs, except option D.
- **Nothing here re-tests the original 12 by drop-one.** That data is in explanation-results.json.

## Cost
- **This phase:** $16.24 by whole-app lifetime ($16.12 measured compute), against a $25 phase cap.
- **Research total:** $179.56 of the $280 cap.

## Files
- **`neuron-roles-results.json`:** counts, rates, activations aggregated by family, direct and total effects, per-site breakdowns, the Wayback test, option D and costs. It holds no generated text.
- **Protocol, cases and ledger:** `data/neuron-roles/protocol.json`, `data/neuron-roles/cases.json`, `data/neuron-roles/budget.json`.
- **Code and tests:** `apps/neuron_roles_app.py`, `qwen_lab/neuron_roles.py`, `tests/test_neuron_roles.py` (11 tests).
- **Raw results:** `../runs/neuron-roles-*/` (option D's rollouts hold the model's text, with the proxy URLs' key and signature redacted).
