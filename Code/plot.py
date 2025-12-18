#!/usr/bin/env python3
"""
make_dcr_prompt_plots.py

Reads all *.jsonl metric files in the current folder, combines them into
a single pandas DataFrame, and produces comparison plots for:

- quality_score
- num_generation_attempts
- total_time_seconds

Grouped by:
- form (AB101 vs SP242)
- prompt_condition (raw vs structured)
- model_name (gpt-4.1, gpt-4.1-mini, gpt-5.1)

Usage:
    python make_dcr_prompt_plots.py

This will create PNG files in a local "figures" folder that you can
use directly in your LaTeX report with \\includegraphics.
"""

import json
import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------
# 1. Helper: infer labels
# ----------------------

def infer_form_from_filename(fname: str) -> str:
    """Infer which form (AB101 / SP242) from the filename."""
    lower = fname.lower()
    if "ab101" in lower:
        return "AB101"
    if "sp242" in lower or "s0242" in lower:  # handle your typo'd filename too
        return "SP242"
    return "UNKNOWN"


def infer_prompt_condition_from_filename(fname: str) -> str:
    """Infer prompt condition from filename: raw vs structured."""
    lower = fname.lower()
    # Heuristic: filenames with 'raw' are raw legal text; 'clean'/'cleaned' are structured SPEC
    if "raw" in lower:
        return "raw"
    if "clean" in lower:
        return "structured"
    return "unknown"


# ----------------------
# 2. Load all JSONL files
# ----------------------

def load_metrics_from_jsonl() -> pd.DataFrame:
    rows = []
    for path in glob.glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Skip broken lines
                    continue

                rec["source_file"] = os.path.basename(path)
                rec["form"] = infer_form_from_filename(rec["source_file"])
                rec["prompt_condition"] = infer_prompt_condition_from_filename(
                    rec["source_file"]
                )

                # Ensure model_name field exists (fallback if not)
                if "model_name" not in rec:
                    # You can tweak this if you encoded it in task_label instead
                    rec["model_name"] = rec.get("model", "UNKNOWN")

                rows.append(rec)

    if not rows:
        raise SystemExit("No .jsonl metric files found in the current directory.")

    df = pd.DataFrame(rows)
    return df


# ----------------------
# 3. Plotting helpers
# ----------------------

def ensure_figures_folder() -> Path:
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    return out_dir


def plot_metric_bar(df: pd.DataFrame,
                    metric: str,
                    out_dir: Path,
                    ylabel: str,
                    title_prefix: str):
    """
    For each form (AB101, SP242), create a bar plot:

        x-axis: model_name
        bars: prompt_condition (raw / structured)
        y: mean(metric)

    Saves one PNG per form.
    """
    # Only keep rows where we actually have this metric
    if metric not in df.columns:
        print(f"Warning: metric '{metric}' not found in DataFrame columns.")
        return

    forms = sorted(df["form"].dropna().unique())
    for form in forms:
        if form == "UNKNOWN":
            continue

        sub = df[df["form"] == form].copy()
        if sub.empty:
            continue

        # Group and pivot
        grouped = (
            sub.groupby(["model_name", "prompt_condition"])[metric]
            .mean()
            .reset_index()
        )

        if grouped.empty:
            continue

        pivot = grouped.pivot(
            index="model_name", columns="prompt_condition", values=metric
        )

        # Sort models roughly in a sensible order if present
        model_order = ["gpt-4.1", "gpt-4.1-mini", "gpt-5.1"]
        existing_models = [m for m in model_order if m in pivot.index]
        other_models = [m for m in pivot.index if m not in existing_models]
        pivot = pivot.loc[existing_models + other_models]

        # Plot
        plt.figure(figsize=(7, 4))
        pivot.plot(kind="bar")
        plt.xlabel("Model")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} – {form}")
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Save
        out_path = out_dir / f"{metric}_{form}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {metric} plot for {form} to {out_path}")


def main():
    df = load_metrics_from_jsonl()
    out_dir = ensure_figures_folder()

    # Optional: print a quick overview
    print("Loaded metrics:")
    print(df[["source_file", "form", "prompt_condition", "model_name",
              "quality_score", "difficulty_score",
              "num_generation_attempts", "total_time_seconds"]]
          .sort_values(["form", "model_name", "prompt_condition"]))

    # 1) Quality score plots
    plot_metric_bar(
        df,
        metric="quality_score",
        out_dir=out_dir,
        ylabel="Quality score (0–100)",
        title_prefix="Prompt quality by model and condition",
    )

    # 2) Attempts plots
    plot_metric_bar(
        df,
        metric="num_generation_attempts",
        out_dir=out_dir,
        ylabel="Number of generation attempts",
        title_prefix="Attempts by model and condition",
    )

    # 3) Time plots
    plot_metric_bar(
        df,
        metric="total_time_seconds",
        out_dir=out_dir,
        ylabel="Total time (seconds)",
        title_prefix="Run time by model and condition",
    )

    # 4) (Optional) Difficulty score plots – uncomment if you want them
    # plot_metric_bar(
    #     df,
    #     metric="difficulty_score",
    #     out_dir=out_dir,
    #     ylabel="Difficulty score (1–10)",
    #     title_prefix="Prompt difficulty by model and condition",
    # )


if __name__ == "__main__":
    main()
