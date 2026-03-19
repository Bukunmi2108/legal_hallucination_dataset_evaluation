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

1. **Data curation** — source citations from legal research DB + public datasets + edge cases
2. **LLM evaluation** — pass prompts to models (GPT, Gemini, Claude), collect responses
3. **Verification** — extract citations → compare to ground truth → DB lookup → categorize
4. **Analysis** — hallucination rates across all dimensions and model families

## Setup

```bash
uv sync
```

## Project Structure

```
data/           — input dataset and output results
evaluation/     — LLM evaluation pipeline
analysis/       — results analysis and visualization
citation_extractor/ — citation extraction from LLM responses
models.py       — data models (BenchmarkItem, EvaluationResult)
```
