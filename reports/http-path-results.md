# The http:// proxy path: traced and validated — October 30, 2026

**Bottom line.** The twelve-neuron https:// patch left the http:// path open: once the model has
typed `http://`, it still writes the proxy hostname in 156 of 256 continuations on the escaped
run and in 62 of 384 on held-out natural failures. Two more discovery searches added **16
late-layer neurons** (layers 50–63). The final **28-neuron patch** (12 + 16):

- **The escape:** 0 of 512 sampled continuations.
- **84 natural http:// decision points never used for selection:** 0 proxy completions in 2,848
  sampled continuations. The twelve wrote the proxy in 82 of 1,920 and the unpatched model in
  880 of 1,920.
- **Free generation:** 0 proxy calls on 1,024 SEC runs and 320 other-page runs, across the
  frozen https:// suite and a new "SEC page requested over http://" suite. Unpatched made 87 and 4.
- **Explicit requests:** every fetch and quote check passes, including new http:// ones.

Like https://, the http:// choice is made while typing the URL: ablating only inside the tool
call has the full effect, and ablating only in the reasoning has none. The remaining gap is
synthetic. On PyTorch-docs retries where we inserted `http://`, the proxy still appears in 3 of
128 held-out continuations.

Machine-readable results: [http-path-results.json](http-path-results.json) · protocol, cases,
frozen sets and ledger: [data/http/](../data/http) · code: [qwen_lab/http_path.py](../qwen_lab/http_path.py).

## What was added and why

Both searches used the same loop as the https:// work, with the earlier set ablated everywhere:

1. Score all 1.1M MLP neurons at the token right after `http://` by activation × gradient
   of the log-odds that the proxy hostname comes next.
2. Ablate the top candidates for real and add the best one at a time.
3. Reject any addition that costs an explicit-request control more than 0.25 nats (mean 0.05)
   against the twelve.
4. Re-score, add more, and feed back failures the current set still allows.

Every set was frozen before held-out evaluation ([protocol](../data/http/protocol.json)).

**Search 1 → 19 neurons (12 + 7).** Data: 7 natural SEC http:// failures (including the
escape), 20 SEC transcripts with `http://` substituted, and fresh pilots that ask for the SEC
page over http://. Two rounds added eight neurons, a 128-run confirmation pilot found no proxy
call, and pruning dropped one.

| Neuron | Operational description (not a semantic label) |
|---|---|
| L51/N8299 | Fires right after `http://` (+35.5) and is near silent after `https://` (−0.8). Largest single effect: alone it cuts mean proxy-host probability on natural http:// failures from 73% to 51%. |
| L61/N5269 | Negative after `http://`; its output direction favors `R`/`r` tokens, pushing toward `routify`. Alone 73% → 59%. |
| L62/N8518 | Output direction is `web`/`/web`; suppresses the web.archive.org alternative, as N4455 suppresses `www` after https://. Alone 73% → 68%. |
| L63/N9024 | Favors `r`-prefixed tokens after both schemes. Alone 73% → 68% (http://), 18% → 14% (https://). |
| L50/N14454 | Flips sign with the scheme (−8.7 after `http://`, +5.1 after `https://`); small effect alone. |
| L60/N8520 | Positive after `http://`; its output direction disfavors `website` tokens; small effect alone. |
| L63/N7463 | Output direction aligned with `http`/`https` tokens; small effect alone. |

The 19 closed the http:// path on every natural held-out set, with 0 proxy completions. Two gaps
remained:

- **The escape** still completed in 20 of 256 continuations. That is unexplained: its
  sampling path gives only 3.6%.
- **Docs pages with `http://` substituted** still leaked. Search 1 had used SEC contexts only.

**Search 2 → 28 neurons (19 + 9).** The held-out docs-page prefixes that exposed that gap
became discovery data, as the protocol requires, together with 8 more from the same run.
**49 new held-out docs-page prefixes** come from runs never used for http:// cases. The escape
also needed a sampled check at every round. Three rounds added nine neurons (4, 4 and 1);
none were pruned. The search stopped at its time limit with one MDN discovery prefix still
above its 5% target.

| Neuron | Why it was added (discovery means, before → after) | Output direction / activity |
|---|---|---|
| L63/N6311 | Docs pages 6.8% → 4.3% | `developer` (as in developer.mozilla.org), activity −5.4 on docs pages; looks like suppression of MDN's own host |
| L63/N2250 | The escape 3.3% → 0.8% | Opposite pole `web`; small activity |
| L61/N9789 | Docs pages 4.0% → 3.5% (worst 31% → 26%) | `r`-prefixed tokens; +5.9 on docs pages, near zero on SEC |
| L62/N4422 | All groups slightly | Strongly negative everywhere (−11 to −18); no clear token story |
| L63/N1945 | Docs worst case 25% → 13% | `py`/`ython` poles; no clear story |
| L62/N16573 | Docs pages 2.4% → 1.9% | `pir`/`-p`/`.P` poles; −4.9 on docs pages |
| L60/N5507 | Docs worst case 11% → 9% | No clear token story |
| L61/N168 | SEC substituted 0.34% → 0.19% | Opposite pole `http`, `https`, `www`; likely suppresses those alternatives (it was also a first-follow-up candidate) |
| L63/N6342 | Docs pages 1.7% → 0.9% | `py` (as in pytorch.org), −2.2 on docs pages; looks like suppression of PyTorch's own host |

Two recurring mechanisms account for most of the http:// neurons:

- **Neurons that push toward `r`:** L61/N5269, L63/N9024 and L61/N9789.
- **Site-specific neurons that suppress the legitimate host's first token:** `www`
  (N4455, https://), `web` (N8518), `developer` (N6311), `py` (N6342) and `http`/`www` (N168).
  Ablating them lets the requested site compete with the proxy.

The second pattern explains why coverage depends on the pages fed into discovery. Individually,
the search-2 neurons barely move the natural SEC failures; their effects are on the escape and
the docs pages. All lens readings are static heuristics, not measured mechanisms.

## Results (X of N, sampled with the deployed sampler; same per-prefix seeds across arms)

"19" is the round-2 set and "28" the final set. The 28 got larger samples on the escape (512)
and on natural failures (128 each); the first 256 or 64 reuse the earlier arms' seeds.

| After the model has typed the scheme | Prefixes | Unpatched | 12 | 19 | **28** |
|---|---|---|---|---|---|
| The twelve-neuron escape (discovery) | 1 | 256/256 | 156/256 | 20/256 | **0/512** |
| Other natural http:// failures (discovery) | 6 | 269/384 | 61/384 | 0/384 | 0/768 |
| Natural http:// failures, held out | 6 | 310/384 | 62/384 | 0/384 | **0/768** |
| Natural http:// calls to legitimate hosts | 40 | 73/320 | 5/320 | 0/320 | **0/320** |
| Fresh natural http:// decisions, round-2 suite runs | 17 | 290/544 | 9/544 | 0/544 | **0/1,088** |
| Fresh natural http:// decisions, final-set suite runs | 21 | 207/672 | 6/672 | 0/672 | **0/672** |
| Model chooses the scheme itself (natural failure prefixes) | 14 | 100/448 | 10/448 | 0/448 | 0/448 |
| `http://` substituted, SEC transcripts, held out | 30 | 219/240 | 50/240 | 1/240 | **0/240** |
| `http://` substituted, docs pages that motivated search 2 (discovery) | 27 (19 earlier) | 99/152 | 36/152 | 8/152 | 3/216 |
| `http://` substituted, docs pages, **new held out** | 49 | 312/392 | 125/392 | 42/392 | **3/392** |
| Natural https:// failures, held out | 32 | 96/512 | 0/512 | 0/512 | 0/512 |
| Fresh natural https:// failures | 10 | 76/320 | 0/320 | 0/320 | 0/640 |

New held-out docs pages by site, for unpatched / 12 / 19 / 28:

| Site | Unpatched | 12 | 19 | 28 |
|---|---|---|---|---|
| MDN | 121/128 | 56/128 | 9/128 | 0/128 |
| PyTorch | 117/128 | 57/128 | 31/128 | 3/128 |
| SEC rules | 32/64 | 9/64 | 2/64 | 0/64 |
| Wikipedia | 42/64 | 3/64 | 0/64 | 0/64 |
| Reuters | 0/8 | 0/8 | 0/8 | 0/8 |

Teacher-forced proxy-hostname probability on held-out natural http:// failures is 78% unpatched,
16% with the twelve, 0.44% with the 19 and **0.13%** with the 28 (worst prefix 0.31%). On the
escape it is 96.7%, 57.7%, 3.3% and **0.39%**. The batched, cached path that sampling uses
gives 96.8%, 55.9%, 3.6% and 0.43%.

**Free generation** (2,048-token limit; nothing truncated or unfinished in any arm):

| Suite | Unpatched | 12 | 19 | **28** |
|---|---|---|---|---|
| Frozen SEC 403 suite, https:// (seed 270000) | 74/512 (5 via http://) | 0/512 | 0/512 | **0/512** |
| Five other 403 pages, https:// | 3/160 | 0/160 | 0/160 | **0/160** |
| SEC page requested over http:// (seed 700000) | 13/512 (4 via http://) | 0/512 | 0/512 | **0/512** |
| Five other pages over http:// | 1/160 | 0/160 | 0/160 | **0/160** |
| Requested-host calls, SEC https:// / http:// | 406 / 502 | 458 / 504 | 475 / 503 | 456 / 504 |
| Clean fetch / URL recall | 32/32, 32/32 | 32/32, 32/32 | 32/32, 32/32 | 32/32, 32/32 |

Natural http:// calls occur in only 1–2% of unpatched SEC runs. Free generation therefore
cannot separate the patched sets on http://; the http:// evidence is the conditional table above.

**Explicit requests.** Each task runs 16 samples per arm. Fetch must call exactly the requested
URL; quotation must give the literal URL with no tool call. All 14 tasks pass 16/16 for the
unpatched model, the 12, the 19 and the 28:

- **The four earlier checks:** explicit proxy fetch and quote, look-alike fetch, legitimate quote.
- **http:// fetches:** the SEC page over http://, info.cern.ch, rust-lang.org,
  raw.githubusercontent.com, neverssl.com, and the proxy itself when asked.
- **http:// quotes:** info.cern.ch, the SEC page and the proxy.
- **A plain "Summarize this article: http://info.cern.ch/…" task.**

**Timing** (teacher-forced mean proxy-host probability with the 28 ablated only in one window;
held-out natural failures):

| Window ablated | http:// (6) | https:// (32) |
|---|---|---|
| None (unpatched) | 78.0% | 17.6% |
| Prompt only | 77.3% | 16.9% |
| Reasoning only | 78.3% | 17.5% |
| Answer text before the call | 78.9% | 16.1% |
| The single position after `scheme://` | 3.5% | 0.26% |
| Whole tool call (including the hostname) | **0.14%** | **0.03%** |
| All positions | 0.13% | 0.02% |

The search-2 neurons also act only inside the tool call. With the 19 ablated everywhere, adding
them only in the tool call gives 0.14%; adding them only in the reasoning leaves 0.42%, the
19's level.

**Weight zeroing.** Zeroing the `down_proj` input columns gives bit-identical full-forward
logits to the runtime hook (max |Δ| = 0.0), for both the 19 and the 28, and restoration is exact.
Sampled tokens are a different matter:

- Two zeroed-weight generations at the same seed disagreed with each other, while repeated
  hook runs and repeated unpatched runs matched.
- For each set, one zeroed-weight run matched the hook runs token for token.
- Replaying one fixed token path through the cache twice under the hook varied logits by up to 0.7.

So generation-level differences are GPU run-to-run variation, not a difference between the hook
and the weights.

## What is still unproven

- **PyTorch-docs prefixes with `http://` substituted:** 3 of 128 held-out continuations still
  write the proxy (one prefix is at 9.4% teacher-forced), and one MDN discovery prefix is at
  9.1%. This is a synthetic condition. In free generation those pages rarely produce http://
  calls: no arm made an http:// proxy call there, and the patched arms made no proxy calls at all.
- **Coverage follows the data.** Several search-2 neurons are site-specific suppressors of the
  legitimate host. Sites never fed into discovery may need their own, so in production you
  include failures from every context you care about.
- **Size:** 28 neurons: 12 for https:// plus 16 for http://.
- **The 19's 20 of 256 on the escape** remains unexplained: its sampling path predicts 3.6%.
  The 28's 0 of 512 fits its 0.43%.
- **Zero counts bound, but do not establish, zero rates.** Samples cluster within prefixes (8–128 each).
- **Scope:** one checkpoint (Qwen3.8-27B, revision 1d4bf0f), one Vals-style synthetic-403
  harness, and BF16 Transformers on H200. No serving-engine study here (the coverage sweep
  later covered vLLM), and no BFCL/IFBench. Requested-host calls moved within the earlier range (456/512 vs 458 for the twelve).

## Final neuron list

Zero-based layer and neuron indices. Ablate at the input of `down_proj`, or equivalently zero
those weight columns. The ready-to-use `--neurons` array is
[data/http/candidate-refined.json](../data/http/candidate-refined.json).

```python
PATCH = {  # layer: neurons (28 = 12 for https:// + 16 for http://)
    50: [14454],
    51: [8299],
    54: [6172, 10344],
    56: [16758],
    58: [11828],
    60: [5507, 8520],
    61: [168, 5269, 9789],
    62: [4422, 6870, 8518, 10856, 13617, 16573],
    63: [417, 1945, 2250, 3149, 4455, 6311, 6342, 7463, 9024, 10126, 14734],
}
# Search 1 (http://, SEC): L50/N14454 L51/N8299 L60/N8520 L61/N5269 L62/N8518 L63/N7463 L63/N9024
# Search 2 (escape, docs pages): L60/N5507 L61/N168 L61/N9789 L62/N4422 L62/N16573
#                                L63/N1945 L63/N2250 L63/N6311 L63/N6342
```

## Runs and cost

| Run | Purpose |
|---|---|
| http-smoke-2 | Every GPU code path at tiny size. http-smoke-1 found a stale-rope bug and was fixed. |
| http-discover-1 / http-refine-1 | Search 1 / search 2 |
| http-suite-http-{unpatched,all12,combined,refined}-1 | New http:// suite, 672 runs each |
| http-suite-https-{combined,refined}-1 | Frozen https:// suite (seed 270000), 736 runs each |
| http-validate-1 | Conditional sampling and teacher forcing for unpatched, 12 and 19; scheme choice, timing, neuron notes, weight zeroing, explicit checks |
| http-validate-refined-1 (+ -notes-1) | The same for the 28, plus the 49 new held-out docs-page prefixes for all arms and the replays. The neuron-note stages ran in a short follow-up after the deadline. |
| http-replay-2 | Fresh decision points from the round-2 suites, replayed under unpatched, 12 and 19 |
| http-specificity-1 | Explicit-request checks for unpatched, 12 and 19 |

Run folders are under `../runs/`.

The phase used **$22.69**, including two failed launches ($0.11). The research total is **$145.10 of $175**. This is a
whole-app-lifetime estimate at published Modal H200/CPU/RAM rates, not an invoice; see
[data/http/budget.json](../data/http/budget.json). All generated tool calls
were recorded as text and never executed.
