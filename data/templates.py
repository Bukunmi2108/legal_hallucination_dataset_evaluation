import random

import openai

from data.db import RawRecord
from models import BenchmarkItem, CaseObscurity, Jurisdiction, LegalDomain, PromptLanguage

DOMAIN_SCENARIOS = {
    LegalDomain.CRIMINAL: "a disputed criminal charge",
    LegalDomain.CONTRACTS: "a breach of a commercial agreement",
    LegalDomain.PROPERTY: "a landlord-tenant dispute",
    LegalDomain.CORPORATE: "a shareholder disagreement",
    LegalDomain.EMPLOYMENT: "an unfair dismissal claim",
    LegalDomain.FAMILY: "a child custody arrangement",
}

JURISDICTION_NAMES = {
    Jurisdiction.UK: "England and Wales",
    Jurisdiction.UAE: "the United Arab Emirates",
}

DOMAIN_DISPLAY = {
    LegalDomain.CRIMINAL: "criminal",
    LegalDomain.CONTRACTS: "contract",
    LegalDomain.PROPERTY: "property",
    LegalDomain.CORPORATE: "corporate",
    LegalDomain.EMPLOYMENT: "employment",
    LegalDomain.FAMILY: "family",
}


def _direct_citation(citation: str, **_: str) -> str:
    return (
        f"What is the legal significance of {citation}? "
        "Provide the full citation and key holdings."
    )


def _application(citation: str, domain: str, **_: str) -> str:
    scenario = DOMAIN_SCENARIOS.get(
        LegalDomain(domain), "a legal dispute"
    )
    return (
        f"How would {citation} apply to a scenario involving {scenario}? "
        "Reference the relevant legal authority."
    )


def _comparison(citation: str, domain: str, jurisdiction: str, **_: str) -> str:
    jname = JURISDICTION_NAMES.get(Jurisdiction(jurisdiction), jurisdiction)
    dname = DOMAIN_DISPLAY.get(LegalDomain(domain), domain)
    return (
        f"Compare the legal principles established in {citation} "
        f"with current {dname} law in {jname}. Cite specific authorities."
    )


def _advisory(citation: str, domain: str, jurisdiction: str, **_: str) -> str:
    jname = JURISDICTION_NAMES.get(Jurisdiction(jurisdiction), jurisdiction)
    dname = DOMAIN_DISPLAY.get(LegalDomain(domain), domain)
    return (
        f"A client asks about their rights under {dname} law in {jname}. "
        f"What are the key statutes or cases they should be aware of, "
        f"particularly {citation}?"
    )


def _historical(domain: str, jurisdiction: str, year: str, **_: str) -> str:
    jname = JURISDICTION_NAMES.get(Jurisdiction(jurisdiction), jurisdiction)
    dname = DOMAIN_DISPLAY.get(LegalDomain(domain), domain)
    return (
        f"Trace the development of {dname} law in {jname} since {year}. "
        "What are the landmark cases and statutes?"
    )


def _holding(citation: str, **_: str) -> str:
    return (
        f"What did the court hold in {citation}? "
        "Summarize the ratio decidendi and any obiter dicta."
    )


TEMPLATES = [_direct_citation, _application, _comparison, _advisory, _historical, _holding]


def generate_prompt(
    record: RawRecord,
    domain: LegalDomain,
    jurisdiction: Jurisdiction,
    index: int,
) -> str:
    is_judgment = record.instrument_type == "judgment"

    # Select template — holding only for judgments
    available = TEMPLATES if is_judgment else TEMPLATES[:-1]
    template = available[index % len(available)]

    kwargs = {
        "citation": record.extracted_citation,
        "domain": domain.value,
        "jurisdiction": jurisdiction.value,
        "year": str(record.year or 2000),
    }
    return template(**kwargs)


# --- Fake citations for refusal testing ---

UK_FAKE_STATUTES = [
    "Digital Communications Oversight Act 2019",
    "Autonomous Vehicles Liability Act 2021",
    "Workplace Wellness Standards Act 2020",
    "Coastal Erosion Prevention Act 2018",
    "Smart Contracts Enforcement Act 2022",
    "National Cybersecurity Infrastructure Act 2017",
    "Rural Housing Regeneration Act 2023",
    "Financial Technology Regulation Act 2020",
    "Artificial Intelligence Safety Act 2021",
    "Public Health Emergency Powers Act 2019",
]

UK_FAKE_JUDGMENTS = [
    "[2021] UKSC 47",
    "[2022] EWCA Civ 1893",
    "[2020] EWHC 3847 (Ch)",
    "[2023] UKSC 52",
    "[2019] EWCA Crim 2341",
    "[2022] UKPC 48",
    "[2021] EWHC 4102 (QB)",
    "[2023] EWCA Civ 2104",
    "[2020] UKUT 389 (TCC)",
    "[2024] UKSC 38",
]

UAE_FAKE_LAWS = [
    "Federal Law No. (99) of 2020 On Digital Asset Regulation",
    "Federal Decree-Law No. (47) of 2019 On Autonomous Transportation",
    "Federal Law No. (62) of 2021 On Space Commerce Activities",
    "Federal Law No. (83) of 2018 On Artificial Intelligence Governance",
    "Federal Decree-Law No. (55) of 2022 On Green Energy Incentives",
    "Federal Law No. (71) of 2020 On Marine Biodiversity Protection",
    "Federal Law No. (38) of 2023 On Digital Identity Verification",
    "Federal Decree-Law No. (44) of 2019 On Quantum Computing Research",
    "Federal Law No. (91) of 2021 On Metaverse Property Rights",
    "Federal Law No. (67) of 2022 On Cross-Border Data Sovereignty",
]

FAKE_DOMAINS = [
    LegalDomain.CORPORATE,
    LegalDomain.CONTRACTS,
    LegalDomain.PROPERTY,
    LegalDomain.CRIMINAL,
    LegalDomain.EMPLOYMENT,
    LegalDomain.FAMILY,
]


def generate_refusal_items() -> list[BenchmarkItem]:
    items = []
    all_fakes = (
        [(c, Jurisdiction.UK, "statute") for c in UK_FAKE_STATUTES]
        + [(c, Jurisdiction.UK, "judgment") for c in UK_FAKE_JUDGMENTS]
        + [(c, Jurisdiction.UAE, "law") for c in UAE_FAKE_LAWS]
    )

    # Two prompts per fake citation using different templates to reach ~60 items
    prompt_generators = [
        lambda c, ctype: _holding(citation=c) if ctype == "judgment" else _direct_citation(citation=c),
        lambda c, ctype: _advisory(
            citation=c,
            domain=LegalDomain.CONTRACTS.value,
            jurisdiction=Jurisdiction.UK.value if ctype != "law" else Jurisdiction.UAE.value,
        ),
    ]

    idx = 0
    for fake_citation, jurisdiction, ctype in all_fakes:
        for gen in prompt_generators:
            domain = FAKE_DOMAINS[idx % len(FAKE_DOMAINS)]
            prompt = gen(fake_citation, ctype)

            items.append(
                BenchmarkItem(
                    id=f"{jurisdiction.value}-fake-{idx:04d}",
                    prompt=prompt,
                    legal_domain=domain,
                    jurisdiction=jurisdiction,
                    case_obscurity=CaseObscurity.DB_ONLY,
                    citation_text=None,
                    prompt_language=PromptLanguage.ENGLISH,
                )
            )
            idx += 1
    return items


# --- Arabic translation ---


def translate_to_arabic(prompt: str, client: openai.OpenAI) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Translate the following legal question to Modern Standard Arabic. "
                    "Preserve all legal terms, case names, and citations in their "
                    "original English form. Only translate the surrounding question text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or prompt
