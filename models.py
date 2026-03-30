from enum import Enum
from pydantic import BaseModel


class Jurisdiction(str, Enum):
    UK = "uk"
    UAE = "uae"


class LegalDomain(str, Enum):
    CONTRACTS = "contracts"
    CRIMINAL = "criminal"
    PROPERTY = "property"
    CORPORATE = "corporate"
    EMPLOYMENT = "employment"
    FAMILY = "family"


class CaseObscurity(str, Enum):
    LANDMARK = "landmark"
    WELL_KNOWN = "well_known"
    JURISDICTION_SPECIFIC = "jurisdiction_specific"
    DB_ONLY = "db_only"


class PromptLanguage(str, Enum):
    ENGLISH = "en"
    ARABIC = "ar"


class ResponseCategory(str, Enum):
    CORRECT = "correct"
    CORRECT_REFUSAL = "correct_refusal"
    MISATTRIBUTION = "misattribution"
    FABRICATION = "fabrication"


class BenchmarkItem(BaseModel):
    """Input dataset schema — prepared before running the LLM."""

    id: str
    prompt: str
    legal_domain: LegalDomain
    jurisdiction: Jurisdiction
    case_obscurity: CaseObscurity
    citation_text: str | None  # None for refusal-test prompts
    prompt_language: PromptLanguage


class LLMResponse(BaseModel):
    """Phase 3 output — raw LLM response before verification."""

    benchmark_item_id: str
    model_id: str
    llm_response: str
    output_language: str  # "en" or "ar"


class EvaluationResult(BaseModel):
    """Output dataset schema — populated after LLM evaluation."""

    benchmark_item_id: str
    model_id: str
    llm_response: str
    output_language: str
    extracted_citations: list[str]
    category: ResponseCategory
