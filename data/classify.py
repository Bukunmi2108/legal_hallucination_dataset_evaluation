from models import CaseObscurity, Jurisdiction, LegalDomain

from data.db import RawRecord

DOMAIN_KEYWORDS: dict[LegalDomain, list[str]] = {
    LegalDomain.CRIMINAL: [
        "criminal", "offence", "offense", "sentence", "murder", "theft", "fraud",
        "prosecution", "penal", "prison", "bail", "assault", "manslaughter",
        "robbery", "burglary", "drug", "trafficking",
    ],
    LegalDomain.CONTRACTS: [
        "contract", "agreement", "breach", "damages", "warranty", "consideration",
        "liability", "indemnity", "obligation", "sale of goods", "commercial",
        "trading", "supply", "petroleum", "telecommunications",
    ],
    LegalDomain.PROPERTY: [
        "property", "land", "tenant", "lease", "mortgage", "conveyance",
        "planning permission", "freehold", "leasehold", "real estate", "rent",
        "housing", "building",
    ],
    LegalDomain.CORPORATE: [
        "company", "companies", "director", "shareholder", "insolvency",
        "corporate", "securities", "merger", "acquisition", "winding up",
        "dividend", "partnership", "foundation", "financial",
    ],
    LegalDomain.EMPLOYMENT: [
        "employment", "worker", "dismissal", "redundancy", "wage", "wages",
        "discrimination", "tribunal", "unfair dismissal", "employer", "labour",
        "labor", "occupational", "health and safety",
    ],
    LegalDomain.FAMILY: [
        "family", "divorce", "custody", "child", "children", "marriage",
        "adoption", "matrimonial", "guardian", "maintenance", "alimony",
        "personal status",
    ],
}

LANDMARK_ACTS = {
    "human rights act 1998",
    "companies act 2006",
    "sale of goods act 1979",
    "consumer rights act 2015",
    "equality act 2010",
    "data protection act 2018",
    "employment rights act 1996",
    "landlord and tenant act 1954",
    "law of property act 1925",
    "theft act 1968",
    "fraud act 2006",
    "insolvency act 1986",
    "partnership act 1890",
    "arbitration act 1996",
    "limitation act 1980",
    "misrepresentation act 1967",
    "unfair contract terms act 1977",
    "contract (rights of third parties) act 1999",
    "children act 1989",
    "matrimonial causes act 1973",
}

LANDMARK_CASES = {
    "donoghue v stevenson",
    "carlill v carbolic smoke ball",
    "salomon v salomon",
    "caparo v dickman",
    "hadley v baxendale",
    "rylands v fletcher",
    "entick v carrington",
    "anns v merton",
    "pepper v hart",
    "balfour v balfour",
    "fisher v bell",
    "re sigsworth",
    "heydon's case",
    "r v r",
    "woolmington v dpp",
}


def classify_domain(title: str, content_snippet: str) -> LegalDomain:
    text = f"{title} {content_snippet}".lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return LegalDomain.CONTRACTS
    return best


def assign_obscurity(record: RawRecord) -> CaseObscurity:
    title_lower = record.title.lower()

    # DIFC/ADGM are always jurisdiction-specific
    if record.source_table in ("documents_difc", "documents_adgm"):
        return CaseObscurity.JURISDICTION_SPECIFIC

    # Check landmark acts
    for act in LANDMARK_ACTS:
        if act in title_lower:
            return CaseObscurity.LANDMARK

    # Check landmark cases
    for case in LANDMARK_CASES:
        if case in title_lower:
            return CaseObscurity.LANDMARK

    # Recent items are more likely well-known
    if record.year and record.year >= 2015:
        return CaseObscurity.WELL_KNOWN

    return CaseObscurity.DB_ONLY


def map_jurisdiction(source_table: str) -> Jurisdiction:
    if source_table == "documents_uk":
        return Jurisdiction.UK
    return Jurisdiction.UAE
