import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("data/results")
PLOTS_DIR = Path("analysis/plots")
SUMMARY_PATH = Path("analysis/summary.md")
BENCHMARK_PATH = Path("data/input/benchmark.csv")

HALLUCINATION_CATEGORIES = {"fabrication", "misattribution"}
DIMENSIONS = ["jurisdiction", "legal_domain", "case_obscurity", "prompt_language"]


def load_all_results() -> pd.DataFrame:
    benchmark = (
        pd.read_csv(BENCHMARK_PATH)
        .rename(columns={"id": "benchmark_item_id"})
        .drop(columns=["prompt"])
    )
    frames = []
    for path in sorted(RESULTS_DIR.glob("*_evaluated.csv")):
        df = pd.read_csv(path)
        merged = df.merge(benchmark, on="benchmark_item_id", how="left")
        frames.append(merged)
    if not frames:
        raise SystemExit(
            f"No evaluated CSVs found in {RESULTS_DIR}/. "
            "Run `python -m citation_extractor.run --model <id>` first."
        )
    return pd.concat(frames, ignore_index=True)


def _counts(group: pd.DataFrame) -> dict[str, int | float]:
    total = len(group)
    raw = group["category"].value_counts().to_dict()
    cat = {k: int(raw.get(k, 0) or 0) for k in (
        "correct", "correct_refusal", "misattribution", "fabrication"
    )}
    bad = cat["fabrication"] + cat["misattribution"]
    return {
        "n": total,
        **cat,
        "hallucination_rate": round(bad / total, 4) if total else 0.0,
    }


def overall_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"model_id": m, **_counts(g)} for m, g in df.groupby("model_id")]
    return pd.DataFrame(rows).sort_values("hallucination_rate")


def breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(["model_id", by]):
        m, v = key  # type: ignore[misc]
        rows.append({"model_id": m, by: v, **_counts(g)})
    return pd.DataFrame(rows).sort_values(["model_id", by])


def plot_breakdown(table: pd.DataFrame, by: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pivot = (
        table.pivot(index=by, columns="model_id", values="hallucination_rate")
        .fillna(0)
    )
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"Hallucination rate by {by}")
    ax.set_ylabel("Hallucination rate (fabrication + misattribution)")
    ax.set_xlabel(by)
    ax.set_ylim(0, max(pivot.values.max() * 1.15, 0.05))
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = PLOTS_DIR / f"{by}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return "\n".join([header, sep, body])


def write_summary(
    df: pd.DataFrame,
    overall: pd.DataFrame,
    breakdowns: dict[str, pd.DataFrame],
    plot_paths: dict[str, Path],
) -> None:
    lines: list[str] = ["# Hallucination Benchmark Results", ""]
    lines.append(
        f"_Across {df['model_id'].nunique()} models, "
        f"{len(df):,} judged responses, "
        f"{df['benchmark_item_id'].nunique()} benchmark items._"
    )
    lines.append("")
    lines.append("**Hallucination rate** = (fabrication + misattribution) / n.")
    lines.append("")
    lines.append("## Overall (by model)")
    lines.append("")
    lines.append(df_to_md(overall))
    lines.append("")
    for dim in DIMENSIONS:
        lines.append(f"## By {dim}")
        lines.append("")
        lines.append(df_to_md(breakdowns[dim]))
        lines.append("")
        rel = plot_paths[dim].relative_to(SUMMARY_PATH.parent).as_posix()
        lines.append(f"![{dim}]({rel})")
        lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hallucination analysis report")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating PNG charts"
    )
    args = parser.parse_args()

    df = load_all_results()
    overall = overall_by_model(df)
    breakdowns = {d: breakdown(df, d) for d in DIMENSIONS}
    if args.no_plots:
        plot_paths = {d: PLOTS_DIR / f"{d}.png" for d in DIMENSIONS}
    else:
        plot_paths = {d: plot_breakdown(breakdowns[d], d) for d in DIMENSIONS}
    write_summary(df, overall, breakdowns, plot_paths)
    print(f"Wrote {SUMMARY_PATH}")
    print(f"  {df['model_id'].nunique()} models · {len(df):,} judged responses")


if __name__ == "__main__":
    main()
