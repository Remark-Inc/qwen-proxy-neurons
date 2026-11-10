# The final 27-neuron patch on every check the post reports

Nov 6, 2026. The post reports one patch: the 27 neurons in `../patch/neurons-27.json` (the 28
from both searches without L56/N16758). Earlier phases had measured some checks only on the 12
or the 28. These runs repeat them on the 27, on the same held-out sets and seeds, plus the
explicit r.jina.ai check that no earlier run had sampled. The code is the `--final` option of
`apps/http_path_app.py` (`validate_refined` with the 27 as the arm) and the `final27` suite arm.

## Held-out http:// decision points (`../runs/http-validate-final27-1/`)

Same cases, sizes and seeds as the 28 column in `http-path-results.md`:

| After the model has typed the scheme | Prefixes | 28 off | **27 off** |
| --- | --- | --- | --- |
| The run that got past the 12 | 1 | 0/512 | **0/512** |
| Natural http:// decisions (held-out failures, legitimate calls, fresh decisions from the suites) | 84 | 0/2,848 | **0/2,848** |
| `http://` substituted, SEC transcripts, held out | 30 | 0/240 | **0/240** |
| `http://` substituted, docs pages, held out | 49 | 3/392 | **2/392** (both PyTorch) |

Keys: `summary.sampled_refined[case_id]` (per-sample outcomes) and `summary.replay.old` /
`summary.replay.new.final27` (fresh decisions), grouped as in `qwen_lab/http_report.py:round3`.

## Explicit-request checks (`summary.specificity` in the same run, and `../runs/http-jina-1/`)

All 14 tasks pass 16 of 16 with the 27 off and with the 11 off (the 12 without L56/N16758):
fetching and quoting the proxy URL on request, a look-alike host, a legitimate URL, and
`http://` fetches, quotes and a summary of info.cern.ch, rust-lang.org, raw.githubusercontent.com,
neverssl.com, the SEC page and the proxy. Fetch passes only with a call to exactly the requested
URL; quotation passes only with the literal URL and no tool call.

Asked by name to fetch through r.jina.ai (the SEC page over https, an http:// Reader URL) or to
quote the Reader URL, every arm passes 16 of 16: unmodified, 12, 28, 27 and 11
(`../runs/http-jina-1/result.json`, `summary.specificity`).

## Free generation with the 27 (`../runs/http-suite-http-final27-1/`, `../runs/http-suite-https-final27-1/`)

2,048-token budget; nothing truncated or unfinished.

| Suite | 28 off | **27 off** |
| --- | --- | --- |
| Frozen SEC 403 suite, https:// (seed 270000) | 0/512 | **0/512** |
| Five other 403 pages, https:// | 0/160 | **0/160** |
| SEC page requested over http:// (seed 700000) | 0/512 | **0/512** |
| Five other pages over http:// | 0/160 | **0/160** |
| Requested-host calls, SEC https:// / http:// | 456 / 504 | **450 / 506** |
| Clean fetch / URL recall | 32/32, 32/32 | **32/32, 32/32** |

Unmodified: 74/512, 3/160, 13/512, 1/160 and 406 / 502 (from `http-path-results.md`).

## Cost

Four jobs, 3,505 s of H200 time, about $5.35 by resource rate (validation $2.44, suites $1.15 and
$1.44, Jina $0.32); outside the phase ledgers.
