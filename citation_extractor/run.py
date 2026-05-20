import argparse
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from citation_extractor.judge import judge
from evaluation.config import MODEL_REGISTRY, OUTPUT_DIR
from models import BenchmarkItem, EvaluationResult

RESULTS_DIR = Path("data/results")
DEFAULT_JUDGE_MODEL = "gpt-5.5"


def load_benchmark_index() -> dict[str, BenchmarkItem]:
    df = pd.read_csv("data/input/benchmark.csv")
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
    return {r["id"]: BenchmarkItem(**r) for r in records}


def load_responses(model_id: str) -> list[dict]:
    df = pd.read_csv(OUTPUT_DIR / f"{model_id}_responses.csv")
    return df.to_dict(orient="records")


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path)
    return set(df["benchmark_item_id"])


async def evaluate_one(
    client: AsyncOpenAI,
    judge_model: str,
    item: BenchmarkItem,
    response_row: dict,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    output_path: Path,
    progress: dict[str, int],
) -> None:
    async with semaphore:
        try:
            result = await judge(client, item, response_row["llm_response"], judge_model)
            eval_result = EvaluationResult(
                benchmark_item_id=item.id,
                model_id=response_row["model_id"],
                llm_response=response_row["llm_response"],
                output_language=response_row["output_language"],
                extracted_citations=result.extracted_citations,
                category=result.category,
                judge_reasoning=result.reasoning,
            )
            async with write_lock:
                df = pd.DataFrame([eval_result.model_dump(mode="json")])
                header = not output_path.exists()
                df.to_csv(output_path, mode="a", header=header, index=False)
        except Exception as e:
            print(f"  ERROR {item.id}: {type(e).__name__}: {e}")
        finally:
            progress["done"] += 1
            done, total = progress["done"], progress["total"]
            if done % 10 == 0 or done == total:
                print(f"  Progress: {done}/{total}")


async def run(model_id: str, judge_model: str, concurrency: int, limit: int | None) -> None:
    items_index = load_benchmark_index()
    rows = load_responses(model_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{model_id}_evaluated.csv"
    completed = load_completed_ids(output_path)

    remaining = [
        r for r in rows
        if r["benchmark_item_id"] not in completed
        and r["benchmark_item_id"] in items_index
    ]
    if limit is not None:
        remaining = remaining[:limit]

    print(f"Target model: {model_id} | Judge model: {judge_model}")
    print(
        f"Responses: {len(rows)} | "
        f"Already judged: {len(completed)} | "
        f"Remaining: {len(remaining)}"
    )

    if not remaining:
        print("Nothing to do.")
        return

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(remaining)}

    tasks = [
        evaluate_one(
            client,
            judge_model,
            items_index[r["benchmark_item_id"]],
            r,
            semaphore,
            write_lock,
            output_path,
            progress,
        )
        for r in remaining
    ]
    await asyncio.gather(*tasks)
    print(f"\nDone. Results in {output_path}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Extract citations from LLM responses and judge them."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which evaluated model's responses to judge.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Model used as judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N items (useful for smoke testing).",
    )
    args = parser.parse_args()
    asyncio.run(run(args.model, args.judge_model, args.concurrency, args.limit))


if __name__ == "__main__":
    main()
