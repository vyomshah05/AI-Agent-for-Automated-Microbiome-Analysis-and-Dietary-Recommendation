"""Presentation figures.

  Figure 1: Pipeline diagram (boxes + arrows, schematic)
  Figure 2: Diversity comparison of the 3 demo samples
  Figure 3: Taxonomic composition heatmap (samples x top-20 genera)
  Figure 4: Example agent reasoning chain as a styled flowchart

Run:
    python visualizations.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import diversity_analysis as da
from demo import pick_demo_samples

HERE = Path(__file__).resolve().parent
PLOTS_DIR = HERE / "outputs" / "plots"


# ============================================================
# Figure 1: pipeline diagram
# ============================================================

def fig1_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        ("Raw\nOTU table\n(BIOM)",       0.3, "#A8D8EA"),
        ("Preprocess\nfilter + relabund", 2.6, "#A8D8EA"),
        ("Diversity\n(Shannon, Chao1, F/B)", 5.0, "#AA96DA"),
        ("Claude Agent\n(tool-using)",      7.6, "#FCBAD3"),
        ("PubMed\nliterature",              7.6, "#FFFFD2"),
        ("Microbiome\nhealth report",      10.6, "#7FB069"),
    ]
    # First five sit on a single row; PubMed sits on a side-feeder above the agent.
    positions = [
        (0.3, 1.5), (2.6, 1.5), (5.0, 1.5), (7.6, 1.5), (7.6, 3.0), (10.6, 1.5),
    ]
    w, h = 2.0, 1.4
    for (label, _x, color), (x, y) in zip(boxes, positions):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                             linewidth=1.5, facecolor=color, edgecolor="#444"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, weight="bold")

    # Arrows: raw -> preprocess -> diversity -> agent -> report
    arrow_kw = dict(arrowstyle="->", color="#444", lw=2)
    for (xs, ys), (xe, ye) in [
        ((2.3, 2.2), (2.6, 2.2)),
        ((4.6, 2.2), (5.0, 2.2)),
        ((7.0, 2.2), (7.6, 2.2)),
        ((9.6, 2.2), (10.6, 2.2)),
    ]:
        ax.annotate("", xy=(xe, ye), xytext=(xs, ys), arrowprops=arrow_kw)

    # PubMed feeds the agent
    ax.annotate("", xy=(8.6, 2.95), xytext=(8.6, 2.95 + 0.0),
                arrowprops=arrow_kw)
    ax.annotate("", xy=(8.6, 2.95), xytext=(8.6, 3.0),
                arrowprops=arrow_kw)
    ax.annotate("", xy=(8.6, 2.95), xytext=(8.6, 3.0),
                arrowprops=arrow_kw)
    ax.annotate("", xy=(8.6, 2.95), xytext=(8.6, 3.0), arrowprops=arrow_kw)
    # Cleaner: a single arrow from PubMed box down to agent box
    ax.annotate("", xy=(8.6, 2.92), xytext=(8.6, 3.0), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.6, 2.92), xytext=(8.6, 3.0), arrowprops=arrow_kw)

    ax.set_title("American Gut Agentic AI Pipeline", fontsize=14, weight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig1] {out}")


# ============================================================
# Figure 2: diversity comparison of the 3 demo samples
# ============================================================

def fig2_demo_comparison(out: Path) -> None:
    otu, meta, tax = da.load_processed()
    div = da.alpha_diversity_per_sample(otu)
    fb = da.firmicutes_bacteroidetes_ratio(otu, tax)
    picks = pick_demo_samples(div, fb)

    metrics = ["shannon", "observed_otus", "chao1", "fb_ratio"]
    rows = []
    for label, sid in picks.items():
        row = {"label": f"{label}\n({sid})"}
        row["shannon"] = float(div.loc[sid, "shannon"])
        row["observed_otus"] = float(div.loc[sid, "observed_otus"])
        row["chao1"] = float(div.loc[sid, "chao1"])
        row["fb_ratio"] = float(fb.get(sid, np.nan))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("label")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#7FB069", "#D7263D", "#3E7CB1"]
    for ax, m in zip(axes, metrics):
        ax.bar(range(len(df)), df[m].values, color=colors)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, fontsize=8, rotation=20, ha="right")
        ax.set_title(m)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Diversity comparison: 3 demo samples", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig2] {out}")


# ============================================================
# Figure 3: heatmap of top-20 genera across all stool samples
# ============================================================

def fig3_heatmap(out: Path) -> None:
    otu, meta, tax = da.load_processed()
    # Aggregate to genus level
    def genus_of(t: str) -> str:
        if not isinstance(t, str):
            return "Unknown"
        for part in t.split(";"):
            p = part.strip()
            if p.startswith("g__"):
                return p[3:] or "Unknown"
        return "Unknown"

    g = tax.loc[otu.index, "taxonomy"].map(genus_of)
    by_genus = otu.groupby(g).sum()
    by_genus = by_genus.loc[by_genus.index != ""]
    top20 = by_genus.mean(axis=1).sort_values(ascending=False).head(20).index
    mat = by_genus.loc[top20]

    # log transform for visualization
    mat_log = np.log10(mat + 1e-6)

    fig, ax = plt.subplots(figsize=(min(14, 0.35 * mat.shape[1] + 4), 7))
    sns.heatmap(mat_log, ax=ax, cmap="viridis", cbar_kws={"label": "log10(relabund)"})
    ax.set_title("Top 20 genera across samples (log relative abundance)", fontsize=12, weight="bold")
    ax.set_xlabel("sample")
    ax.set_ylabel("genus")
    plt.xticks(rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig3] {out}")


# ============================================================
# Figure 4: agent reasoning chain
# ============================================================

def fig4_agent_reasoning(out: Path) -> None:
    """Stylized flowchart of a representative agent loop. Schematic, not
    pulled from a live transcript so it's deterministic for slides."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")

    steps = [
        ("User: analyze sample AG_S007",                "#FFFFD2", 8.0),
        ("Tool: calculate_diversity(AG_S007)",          "#A8D8EA", 7.0),
        ("Thought: Shannon low (1.8 vs cohort 3.4) - dysbiosis flag",  "#EFE2BA", 6.0),
        ("Tool: get_top_taxa(AG_S007, n=10)",          "#A8D8EA", 5.0),
        ("Thought: Bacteroides + Prevotella dominate - check literature", "#EFE2BA", 4.0),
        ("Tool: search_pubmed('Bacteroides')",          "#A8D8EA", 3.0),
        ("Tool: search_pubmed('Prevotella')",           "#A8D8EA", 2.0),
        ("Tool: generate_report(AG_S007)",              "#A8D8EA", 1.0),
        ("Final report: 5/10, recommend fiber + fermented foods", "#7FB069", 0.0),
    ]
    box_w, box_h = 7.0, 0.7
    for label, color, y in steps:
        x = (10 - box_w) / 2
        ax.add_patch(mpatches.FancyBboxPatch((x, y), box_w, box_h,
                                             boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor="#444", linewidth=1.2))
        ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center", fontsize=10)

    # Arrows between successive boxes
    arrow_kw = dict(arrowstyle="->", color="#444", lw=1.5)
    for i in range(len(steps) - 1):
        y_from = steps[i][2]
        y_to = steps[i + 1][2] + box_h
        ax.annotate("", xy=(5, y_to), xytext=(5, y_from), arrowprops=arrow_kw)

    ax.set_title("Example agent reasoning chain", fontsize=13, weight="bold", pad=10)
    legend_handles = [
        mpatches.Patch(color="#FFFFD2", label="user input"),
        mpatches.Patch(color="#A8D8EA", label="tool call"),
        mpatches.Patch(color="#EFE2BA", label="model thought"),
        mpatches.Patch(color="#7FB069", label="final answer"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig4] {out}")


# ============================================================
# Driver
# ============================================================

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig1_pipeline(PLOTS_DIR / "fig1_pipeline.png")
    fig2_demo_comparison(PLOTS_DIR / "fig2_demo_comparison.png")
    fig3_heatmap(PLOTS_DIR / "fig3_heatmap.png")
    fig4_agent_reasoning(PLOTS_DIR / "fig4_agent_reasoning.png")
    print("[viz] done.")


if __name__ == "__main__":
    main()
