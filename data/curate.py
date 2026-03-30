import random

import openai
import pandas as pd
from dotenv import load_dotenv

from data.classify import assign_obscurity, classify_domain, map_jurisdiction
from data.db import fetch_all_records
from data.templates import generate_prompt, generate_refusal_items, translate_to_arabic
from models import BenchmarkItem, Jurisdiction, PromptLanguage


def main() -> None:
    load_dotenv()

    # 1. Fetch records from DB
    print("Fetching records from legal research DB...")
    records = fetch_all_records()
    print(f"  Fetched {len(records)} records")

    # 2. Classify and generate benchmark items
    print("Generating benchmark items...")
    benchmark_items: list[BenchmarkItem] = []

    for i, record in enumerate(records):
        domain = classify_domain(record.title, record.content_snippet)
        obscurity = assign_obscurity(record)
        jurisdiction = map_jurisdiction(record.source_table)

        prompt = generate_prompt(record, domain, jurisdiction, i)

        item = BenchmarkItem(
            id=f"{jurisdiction.value}-{record.source_table.replace('documents_', '')}-{i:04d}",
            prompt=prompt,
            legal_domain=domain,
            jurisdiction=jurisdiction,
            case_obscurity=obscurity,
            citation_text=record.extracted_citation,
            prompt_language=PromptLanguage.ENGLISH,
        )
        benchmark_items.append(item)

    print(f"  Generated {len(benchmark_items)} real citation items")

    # 3. Add refusal items
    refusal_items = generate_refusal_items()
    benchmark_items.extend(refusal_items)
    print(f"  Added {len(refusal_items)} refusal-test items")

    # 4. Translate ~20% of UAE items to Arabic
    uae_items = [it for it in benchmark_items if it.jurisdiction == Jurisdiction.UAE]
    arabic_count = max(1, len(uae_items) // 5)
    arabic_candidates = random.sample(uae_items, min(arabic_count, len(uae_items)))

    print(f"  Translating {len(arabic_candidates)} UAE items to Arabic...")
    try:
        client = openai.OpenAI()
        arabic_items = []
        for item in arabic_candidates:
            arabic_prompt = translate_to_arabic(item.prompt, client)
            arabic_item = item.model_copy(
                update={
                    "id": item.id + "-ar",
                    "prompt": arabic_prompt,
                    "prompt_language": PromptLanguage.ARABIC,
                }
            )
            arabic_items.append(arabic_item)
        benchmark_items.extend(arabic_items)
        print(f"  Added {len(arabic_items)} Arabic items")
    except Exception as e:
        print(f"  Skipping Arabic translation (OPENAI_API_KEY missing or error): {e}")

    # 5. Shuffle and write CSV
    random.shuffle(benchmark_items)
    df = pd.DataFrame([item.model_dump(mode="json") for item in benchmark_items])
    output_path = "data/input/benchmark.csv"
    df.to_csv(output_path, index=False)
    print(f"\nWritten {len(benchmark_items)} items to {output_path}")

    # 6. Summary stats
    print("\n--- Distribution Summary ---")
    print(f"\nBy jurisdiction:\n{df['jurisdiction'].value_counts().to_string()}")
    print(f"\nBy legal domain:\n{df['legal_domain'].value_counts().to_string()}")
    print(f"\nBy case obscurity:\n{df['case_obscurity'].value_counts().to_string()}")
    print(f"\nBy prompt language:\n{df['prompt_language'].value_counts().to_string()}")
    print(f"\nRefusal items (null citation): {df['citation_text'].isna().sum()}")


if __name__ == "__main__":
    main()
