# Legal Citation Hallucination Benchmark

A benchmark for evaluating the tendency of LLMs to hallucinate legal citations when answering legal questions without tool access.

## What It Measures

Whether LLMs fabricate, misattribute, or correctly cite legal cases and statutes across UK and UAE jurisdictions.

## Rubric

| Category | Description | Severity |
|----------|-------------|----------|
| **Correct** | Real, relevant citation | - |
| **Correct Refusal** | Model acknowledges uncertainty | - |
| **Misattribution** | Real case, wrong context | Medium |
| **Fabrication** | Entirely invented case | High |

## Dimensions

- **Jurisdiction**: UK, UAE
- **Legal domain**: Contracts, criminal, property, corporate, employment, family
- **Case obscurity**: Landmark → well-known → jurisdiction-specific → DB-only
- **Prompt language**: English, Arabic (~20% of dataset)

## Pipeline

1. **Dataset** — provided as `data/input/benchmark.csv` (see [Dataset construction](#dataset-construction) below)
2. **LLM evaluation** — pass prompts to models (GPT, Gemini, Claude), collect responses
3. **Verification** — LLM-as-judge extracts citations and categorizes each response
4. **Analysis** — hallucination rates across all dimensions and model families

## Setup

```bash
uv sync
```

## Running the pipeline

**Step 2 — collect model responses:**

```bash
python -m evaluation.run --model gpt-5.1
```

Responses land in `data/output/{model_id}_responses.csv`. Re-running skips items already completed for that model.

**Step 3 — judge each response:**

```bash
python -m citation_extractor.run --model gpt-5.1 --judge-model gpt-5.5 --concurrency 5
```

Uses an LLM as judge to extract citations from each response and assign one of `correct` / `correct_refusal` / `misattribution` / `fabrication`. Resumable. Results land in `data/results/{model_id}_evaluated.csv`. Pass `--limit N` for a smoke test.

**Step 4 — generate the report:**

```bash
python -m analysis.report
```

Joins all `data/results/*_evaluated.csv` with the benchmark and writes `analysis/summary.md` with overall + per-dimension breakdowns, plus PNG charts in `analysis/plots/`.

### Note on judge bias

Verification is done by an LLM-as-judge, since open-source citation lookup tools don't cover UAE / DIFC / ADGM well. Using a model from the same family as one being judged (e.g. GPT-5.1 judging GPT-5.1) carries known self-preference bias — interpret the cross-model comparison with that caveat in mind. A future iteration could swap in Claude or Gemini as judge for cross-family verification.

## Dataset construction

The benchmark CSV was built once, off-repo, by sampling a private legal research corpus covering UK statutes and judgments, UAE federal laws, and DIFC/ADGM instruments. Each sampled record was:

- classified into a legal domain (contracts, criminal, property, corporate, employment, family) from title + content,
- bucketed by obscurity (landmark, well-known, jurisdiction-specific, DB-only) based on prominence signals,
- turned into a natural-language prompt that asks the model for a citation in that domain/jurisdiction.

A set of **refusal-test items** was added — prompts about non-existent statutes or fabricated case names — to measure whether models will invent citations under pressure or correctly decline.

Roughly **20% of UAE items** were translated to Arabic via GPT-4 to probe whether hallucination rates shift with prompt language.

The extraction and curation code is not included in this repo; the resulting CSV is the artifact downstream stages consume.

## Project Structure

```
data/input/         — benchmark.csv (frozen input dataset)
data/output/        — per-model raw response CSVs
data/results/       — per-model judged result CSVs (EvaluationResult schema)
evaluation/         — LLM evaluation pipeline (Phase 2)
citation_extractor/ — LLM-as-judge citation extraction + categorization (Phase 3)
analysis/           — aggregation, summary.md, and plots/ (Phase 4)
models.py           — data models (BenchmarkItem, LLMResponse, EvaluationResult)
```
