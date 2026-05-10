"""End-to-end demo: load data, pick 3 representative samples, run the agent.

Run with:
    python demo.py

Steps:
  1. data_loader.main()            - download / synthesize / clean
  2. diversity_analysis.main()     - compute diversity, save plots
  3. Pick 3 sample IDs:
        - highest Shannon (high diversity / 'healthy')
        - lowest Shannon  (low diversity)
        - most extreme F/B ratio
  4. agent.run_agent() on each
  5. Print transcript + save report to outputs/reports/
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import data_loader
import diversity_analysis as da
import agent

HERE = Path(__file__).resolve().parent
PROCESSED_DIR = HERE / "data" / "processed"
REPORTS_DIR = HERE / "outputs" / "reports"


def pick_demo_samples(div_df: pd.DataFrame, fb: pd.Series) -> dict[str, str]:
    """Return {label: sample_id} for the three demo cases."""
    high_div = div_df["shannon"].idxmax()
    low_div = div_df["shannon"].idxmin()

    fb_clean = fb.dropna()
    if fb_clean.empty:
        extreme_fb = high_div
    else:
        # Pick whichever extreme is further from the median, but distinct
        med = fb_clean.median()
        deviations = (fb_clean - med).abs().sort_values(ascending=False)
        extreme_fb = next(
            (sid for sid in deviations.index if sid not in (high_div, low_div)),
            deviations.index[0],
        )

    return {
        "high_diversity":   high_div,
        "low_diversity":    low_div,
        "extreme_fb_ratio": extreme_fb,
    }


def write_report(label: str, sample_id: str, result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}_{sample_id}.txt"

    transcript_lines = []
    for entry in result["transcript"]:
        kind = entry["kind"]
        if kind == "thought":
            transcript_lines.append(f"--- THOUGHT (step {entry['step']}) ---\n{entry['text']}\n")
        elif kind == "tool_use":
            transcript_lines.append(
                f"--- TOOL_USE step={entry['step']} {entry['name']}({json.dumps(entry['input'])}) ---"
            )
        elif kind == "tool_result":
            preview = json.dumps(entry["output"], default=str, indent=2)
            if len(preview) > 1500:
                preview = preview[:1500] + "\n... [truncated]"
            transcript_lines.append(
                f"--- TOOL_RESULT step={entry['step']} {entry['name']} ---\n{preview}\n"
            )

    text = (
        f"# Sample: {sample_id} ({label})\n\n"
        "## Final report\n\n"
        f"{result['text']}\n\n"
        "## Agent transcript\n\n"
        + "\n".join(transcript_lines)
    )
    path.write_text(text)
    return path


def main() -> None:
    print("=" * 70)
    print("STEP 1: data loading / preprocessing")
    print("=" * 70)
    data_loader.main()

    print("\n" + "=" * 70)
    print("STEP 2: diversity analysis + plots")
    print("=" * 70)
    da.main()

    print("\n" + "=" * 70)
    print("STEP 3: pick demo samples")
    print("=" * 70)
    otu, meta, tax = da.load_processed()
    div = da.alpha_diversity_per_sample(otu)
    fb = da.firmicutes_bacteroidetes_ratio(otu, tax)

    picks = pick_demo_samples(div, fb)
    for label, sid in picks.items():
        print(f"  {label:18s} -> {sid}  shannon={div.loc[sid, 'shannon']:.2f}  "
              f"observed={div.loc[sid, 'observed_otus']:.0f}  "
              f"F/B={fb.get(sid, float('nan')):.2f}")

    print("\n" + "=" * 70)
    print("STEP 4: run agent on each sample")
    print("=" * 70)
    for label, sid in picks.items():
        print(f"\n>>> {label}: {sid}")
        try:
            result = agent.run_agent(sid)
        except Exception as e:
            print(f"[demo] agent failed for {sid}: {e!s}")
            continue
        out = write_report(label, sid, result, REPORTS_DIR)
        print(f"[demo] saved {out}")

    print("\n[demo] done.")


if __name__ == "__main__":
    main()
