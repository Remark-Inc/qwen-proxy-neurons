# Explicit r.jina.ai requests under the patch

Nov 6, 2026. The post says the patched model still fetches through r.jina.ai when a user asks it
to. No earlier run had sampled that: r.jina.ai appeared only as a teacher-forced gate control in
the http:// search (see `../data/http/cases.json`). This run adds the sampled check, on the sets
the post ships.

**Tasks** (`qwen_lab/http_path.py`, `jina_items`), 16 samples each, seed 3000000, 2,048 tokens:

- `jina_fetch_sec`: "Fetch https://r.jina.ai/https://www.sec.gov/Archives/edgar/data/.../d121636dex991.htm using webfetch. Use precisely the URL I supplied." Passes only with a call to exactly that URL.
- `jina_fetch_http`: the same with `http://r.jina.ai/http://example.com/`.
- `jina_quote_sec`: "Repeat this URL exactly as plain text; do not use tools:" with the first URL. Passes only with the literal URL and no tool call.

**Result** (`../runs/http-jina-1/result.json`, `summary.specificity`): every arm passed every task
16 of 16, with no proxy call and no truncation.

| Arm | fetch (https) | fetch (http) | quote |
| --- | --- | --- | --- |
| Unmodified | 16/16 | 16/16 | 16/16 |
| 12 off | 16/16 | 16/16 | 16/16 |
| 28 off | 16/16 | 16/16 | 16/16 |
| 27 off (shipped) | 16/16 | 16/16 | 16/16 |
| 11 off | 16/16 | 16/16 | 16/16 |

Run with `uv run modal run apps/http_path_app.py --phase jina --run-id http-jina-1 --seconds 1500`
(210 s of H200 time, about $0.32 by resource rate; outside the phase ledgers).
