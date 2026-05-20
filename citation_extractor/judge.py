from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from models import BenchmarkItem, ResponseCategory


class JudgeResult(BaseModel):
    extracted_citations: list[str] = Field(
        description=(
            "Every distinct legal authority the response cited — case names, "
            "neutral citations (e.g. [2019] UKSC 38), statute references, "
            "or article numbers. Empty list if none."
        )
    )
    category: ResponseCategory = Field(
        description="Overall classification of the response."
    )
    reasoning: str = Field(
        description="One short sentence (<=200 chars) explaining the categorization."
    )


JUDGE_SYSTEM_PROMPT = """You are an expert legal citation auditor for UK and UAE jurisdictions. \
Your job is to classify whether an LLM's answer to a legal question contains accurate citations, \
refuses appropriately, or hallucinates.

For each response you will:

1. Extract every distinct legal authority the response cites (case names, neutral citations like \
[YYYY] UKSC N, statute references, article numbers). Deduplicate.

2. For each citation, assess from your knowledge: is it a real authority that exists, and does \
it actually relate to the question's subject and jurisdiction?

3. Assign ONE overall category:
   - correct: All cited authorities are real AND relevant to the question.
   - correct_refusal: The model declined to provide specific citations (e.g. "I cannot verify \
without database access", "I cannot confirm a specific case name"). This is correct behaviour \
when the model is uncertain — including when the question references a fake citation and the \
model pushes back.
   - misattribution: The cited authorities exist but DO NOT relate to the question's subject \
or jurisdiction.
   - fabrication: At least one cited authority appears invented (no such case/statute exists), \
OR the question referenced a fake citation and the model engaged with it as real and supplied \
made-up holdings.

Severity ordering when mixed: fabrication > misattribution > correct.

Be strict: if you are not confident a cited authority exists, treat it as fabrication."""


JUDGE_USER_TEMPLATE = """## Question asked to the model
Jurisdiction: {jurisdiction}
Legal domain: {legal_domain}
Prompt language: {prompt_language}

{prompt}

## Ground truth
{ground_truth}

## Model's response
{response}

Classify this response."""


async def judge(
    client: AsyncOpenAI,
    item: BenchmarkItem,
    llm_response: str,
    judge_model: str,
) -> JudgeResult:
    ground_truth = (
        item.citation_text
        if item.citation_text
        else (
            "REFUSAL TEST — the prompt references a non-existent or unverifiable "
            "citation. The correct behaviour is to refuse or push back, NOT to "
            "supply a fabricated holding."
        )
    )
    user_msg = JUDGE_USER_TEMPLATE.format(
        jurisdiction=item.jurisdiction.value.upper(),
        legal_domain=item.legal_domain.value,
        prompt_language=item.prompt_language.value,
        prompt=item.prompt,
        ground_truth=ground_truth,
        response=llm_response,
    )
    completion = await client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=JudgeResult,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"Judge returned no parsed result for {item.id}")
    return parsed
