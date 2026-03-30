import os
import re
from dataclasses import dataclass, field

import psycopg
from dotenv import load_dotenv
from psycopg import sql

load_dotenv()

UK_NEUTRAL_CITATION_RE = re.compile(
    r"\[(\d{4})\]\s+"
    r"(UKSC|UKHL|EWCA\s+(?:Civ|Crim)|EWHC|UKPC|UKUT|UKFTT)"
    r"\s+(\d+)"
)


@dataclass
class RawRecord:
    db_id: str
    title: str
    content_snippet: str
    instrument_type: str
    year: int | None
    source_link: str | None
    source_table: str
    extracted_citation: str = field(default="")

    def __post_init__(self):
        if not self.extracted_citation:
            self.extracted_citation = self.title


def get_connection() -> psycopg.Connection:
    return psycopg.connect(os.environ["PG_CONN_STR"])


def discover_instrument_types(table: str) -> list[tuple[str, int]]:
    with get_connection() as conn:
        query = sql.SQL(
            "SELECT instrument_type, COUNT(*) as cnt FROM {} "
            "GROUP BY instrument_type ORDER BY cnt DESC"
        ).format(sql.Identifier(table))
        rows = conn.execute(query).fetchall()
    return [(r[0], r[1]) for r in rows]


def _fetch_records(query: str | sql.SQL, source_table: str) -> list[RawRecord]:
    records = []
    with get_connection() as conn:
        rows = conn.execute(sql.SQL(query) if isinstance(query, str) else query).fetchall()
    for row in rows:
        db_id, title, content_snippet, instrument_type, year, source_link = row
        record = RawRecord(
            db_id=db_id,
            title=title or "",
            content_snippet=content_snippet or "",
            instrument_type=instrument_type or "",
            year=year,
            source_link=source_link,
            source_table=source_table,
        )
        # Extract neutral citation for UK judgments
        if source_table == "documents_uk" and instrument_type == "judgment":
            match = UK_NEUTRAL_CITATION_RE.search(record.content_snippet)
            if match:
                record.extracted_citation = match.group(0)
        records.append(record)
    return records


def fetch_uk_statutes(limit: int = 60) -> list[RawRecord]:
    query = f"""
        SELECT id, title, LEFT(content, 2000), instrument_type,
               date_of_issue_year, source_link
        FROM documents_uk
        WHERE instrument_type = 'law'
          AND title IS NOT NULL
          AND date_of_issue_year IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
    """
    return _fetch_records(query, "documents_uk")


def fetch_uk_judgments(limit: int = 60) -> list[RawRecord]:
    query = f"""
        SELECT id, title, LEFT(content, 2000), instrument_type,
               date_of_issue_year, source_link
        FROM documents_uk
        WHERE instrument_type = 'judgment'
          AND title IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
    """
    return _fetch_records(query, "documents_uk")


def fetch_uae_laws(limit: int = 60) -> list[RawRecord]:
    query = f"""
        SELECT id, title, LEFT(content, 2000), instrument_type,
               date_of_issue_year, source_link
        FROM documents_uae
        WHERE instrument_type IN ('law', 'decree')
          AND language IN ('en', 'en-AE', 'en-GB')
          AND title IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
    """
    return _fetch_records(query, "documents_uae")


def fetch_difc(limit: int = 30) -> list[RawRecord]:
    query = f"""
        SELECT id, title, LEFT(content, 2000), instrument_type,
               date_of_issue_year, source_link
        FROM documents_difc
        WHERE instrument_type IN ('law', 'judgment')
          AND title IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
    """
    return _fetch_records(query, "documents_difc")


def fetch_adgm(limit: int = 30) -> list[RawRecord]:
    query = f"""
        SELECT id, title, LEFT(content, 2000), instrument_type,
               date_of_issue_year, source_link
        FROM documents_adgm
        WHERE title IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
    """
    return _fetch_records(query, "documents_adgm")


def fetch_all_records() -> list[RawRecord]:
    records = []
    records.extend(fetch_uk_statutes())
    records.extend(fetch_uk_judgments())
    records.extend(fetch_uae_laws())
    records.extend(fetch_difc())
    records.extend(fetch_adgm())
    return records
