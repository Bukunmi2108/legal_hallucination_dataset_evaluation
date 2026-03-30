import argparse
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from evaluation.config import MODEL_REGISTRY, OUTPUT_DIR, SYSTEM_PROMPT
from evaluation.providers import create_provider, LLMProvider
from models import BenchmarkItem, LLMResponse


def load_benchmark() -> list[BenchmarkItem]:
    df = pd.read_csv("data/input/benchmark.csv")
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
    return [BenchmarkItem(**r) for r in records]


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path)
    return set(df["benchmark_item_id"])


def detect_language(text: str) -> str:
    arabic_count = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return "ar" if arabic_count / max(len(text), 1) > 0.3 else "en"


async def evaluate_item(
    item: BenchmarkItem,
    provider: LLMProvider,
    model_id: str,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_path: Path,
    progress: dict[str, int],
) -> None:
    async with semaphore:
        try:
            raw_response = await provider.generate(SYSTEM_PROMPT, item.prompt)
            response = LLMResponse(
                benchmark_item_id=item.id,
                model_id=model_id,
                llm_response=raw_response,
                output_language=detect_language(raw_response),
            )
            async with write_lock:
                df = pd.DataFrame([response.model_dump(mode="json")])
                header = not output_path.exists()
                df.to_csv(output_path, mode="a", header=header, index=False)
        except Exception as e:
            print(f"  ERROR {item.id}: {e}")
        finally:
            progress["done"] += 1
            done, total = progress["done"], progress["total"]
            if done % 10 == 0 or done == total:
                print(f"  Progress: {done}/{total}")


async def run(model_id: str) -> None:
    config = MODEL_REGISTRY[model_id]
    provider = create_provider(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{model_id}_responses.csv"

    items = load_benchmark()
    completed = load_completed_ids(output_path)
    remaining = [item for item in items if item.id not in completed]

    print(f"Model: {model_id}")
    print(f"Total: {len(items)}, Already done: {len(completed)}, Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    semaphore = asyncio.Semaphore(config.max_concurrency)
    write_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(remaining)}

    tasks = [
        evaluate_item(item, provider, model_id, semaphore, write_lock, output_path, progress)
        for item in remaining
    ]
    await asyncio.gather(*tasks)
    print(f"\nDone. Results in {output_path}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run LLM evaluation benchmark")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to evaluate",
    )
    args = parser.parse_args()
    asyncio.run(run(args.model))


if __name__ == "__main__":
    main()
