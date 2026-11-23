# Optimly Agent Lab: Registry Uplift Test

## What it does
A brand Q&A agent that answers only from the public **Optimly Index**, cites the profile it used, and says "I don't know" when the data doesn't cover the question. An evaluation compares it against the same model with no tools across about 40 questions, measuring accuracy, made-up facts, abstention quality, latency and cost.

## How it works

```
question ─► agent (gpt-4o-mini) ─► search_brand(name) ─► /brands/search?q=…   (fuzzy re-ranked)
                                 └► get_brand_profile(slug) ─► /brand/{slug}.json
                                 ─► BrandAnswer {brand, answer, evidence[], abstained, reason}
```

| File | Role |
|---|---|
| `optimly_agent/tools.py` | Calls the public index (no key needed), re-ranks matches with rapidfuzz, caches every response in `.cache/` |
| `optimly_agent/schemas.py` | Pydantic `BrandAnswer` (answer + evidence + abstain flag) and `JudgeVerdict` |
| `optimly_agent/agent.py` | `OptimlyAgent` (LangChain `create_agent` + tools) and `BaselineAgent` (same model and schema, no tools, also allowed to abstain) |
| `questions.json` | 40 questions over 14 brand names (18 profile records): 19 answerable, 9 not in the data, 5 ambiguous names, 7 false premises |
| `evaluate.py` | Runs both systems, has a different model (gpt-4o) judge each answer against the profile facts, then builds the metrics table |
| `app.py` | Streamlit UI with an Ask tab (answer, evidence links, "I don't know" banner, optional baseline comparison) and a Results tab |

**Data source choice.** Step 0 probes (2026-11-20):
- `optimly.ai/brand/cerave` returned 404 because the real slug is `cerave-l-oreal`, which is why search comes first.
- `scout.optimly.ai/.well-known/agent.json` timed out.
- `app.optimly.ai/api/v1/agent/schema` returned 410 (v1 retired; the keyed API is v2 and needs credentials).
- `optimly.ai/llms.txt` documents free public JSON endpoints, `/brands/search` and `/brand/{slug}.json`. This project uses those.

Every profile section carries its own `source_label`, `source_url` and date, so evidence can be cited per section.

## Run it

```powershell
pip install -r requirements.txt
copy .env.example .env      # then put your OPENAI_API_KEY in .env
python -m optimly_agent.agent "Tell me about Mercury"
python evaluate.py --limit 5   # smoke test
python evaluate.py             # full run -> results/metrics.md
streamlit run app.py
```

## Results

Run on 2026-11-20: 40 questions, answerer `gpt-4o-mini`, judge `gpt-4o`, temperature 0. Full per-answer labels are in `results/judged.csv`.

| Metric | Baseline (no tools) | Optimly agent |
|---|---:|---:|
| Accuracy on answerable (%) | 42.1 | **100** |
| Made-up facts rate, all questions (%) | 30.0 | **0** |
| Wrong answers, all questions (%) | 35.0 | **2.5** |
| Correct "I don't know" (not in data, %) | 22.2 | **100** |
| Ambiguity handled (%) | 0 | **80** |
| False premise handled (%) | 57.1 | **100** |
| Over-abstained on answerable (%) | 0 | 0 |
| Avg latency (s) | **1.68** | 4.34 |
| Avg tokens / question | **414** | 5,483 |
| Avg cost / question (USD) | **$0.0001** | $0.0009 |

Label counts: the baseline got 12 correct, 2 correct abstains, 12 unsupported and 14 wrong. The agent got 30 correct, 9 correct abstains, 0 unsupported and 1 wrong.

The agent is far more reliable. The price is about 2.6× the latency and about 9× the cost per question, though that is still under a tenth of a cent.

**Example failures**
1. **Baseline makes up a price.** Asked "How much does Mercury Pro cost?", it said **$500/month**. The Optimly profile says $299/month.
2. **Baseline picks the wrong Mercury.** Asked "Tell me about Mercury.", it silently described the US fintech. The agent found both records (the US banking fintech and the New Zealand energy company) and listed them. In an earlier test the baseline described the discontinued Ford car brand.
3. **Agent assumes which Delta (its only wrong answer).** Asked "What does Delta do?", it described Delta Air Lines without mentioning that the index also has Delta Lake and Delta Dental.

**Engineering note.** When the data didn't contain the answer, gpt-4o-mini kept searching instead of abstaining. It re-fetched the same profiles or went through sub-brands such as "Notion AI" and "figma-figjam" until it ran out of steps. Two changes fixed this:
- `ToolCallLimitMiddleware` caps the number of profile and search calls.
- If the agent still hasn't answered within its step budget, one final structured call answers from the tool results it already gathered, or abstains.

This fallback fired on 18 of 40 questions (the `forced_final` column), mostly not-in-data and false-premise questions. A stronger model or a tighter prompt could reduce that.

## Limitations
- **Small question set** (40 questions, 14 brand names, 18 profile records), so each question moves a rate by 2–5 points.
- **Circular ground truth.** Expected answers come from the same Optimly profiles the agent reads. This measures faithful retrieval, not real-world truth, so the baseline's gap is overstated. For example, the baseline's "Delta One" answer is accurate in real life but was graded wrong because the profile only lists Premium Select.
- **The LLM judge is imperfect.** It passed m05 ("Who are Delta's customers?") with the rationale that the agent stated its assumption, but the agent never named which Delta it meant. Spot-check `judged.csv` rather than trusting the rates blindly.
- **Profiles can be wrong or stale.** They are mostly sourced from brands' own websites, and some sections are vague (Linear's pricing is literally "Not explicitly mentioned").
- **Run-to-run variance.** Results use temperature 0, but LLM answers and LLM-judge labels can still change between runs. Re-run or use `--rejudge` to check stability.
- **Cached data.** `.cache/` freezes profile snapshots for repeatability. Delete it to refetch.
