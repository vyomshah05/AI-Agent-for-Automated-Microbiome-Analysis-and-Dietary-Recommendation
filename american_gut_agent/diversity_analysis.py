"""Alpha diversity, F/B ratio, dysbiosis flagging, and presentation plots.

Reads the processed CSVs produced by data_loader.py:
  data/processed/otu_clean.csv      relative-abundance OTU table (taxa x samples)
  data/processed/metadata_clean.csv sample metadata
  data/processed/taxonomy.csv       per-OTU taxonomy strings

Computes:
  Shannon, observed_otus, Chao1 per sample (scikit-bio)
  Firmicutes / Bacteroidetes ratio per sample
  Dysbiosis flag: any metric > 1.5 SD from cohort mean

Saves three plots under outputs/plots/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
PROCESSED_DIR = HERE / "data" / "processed"
PLOTS_DIR = HERE / "outputs" / "plots"

# Counts-derived diversity wants integer-ish counts. We approximate by
# scaling relative abundances to a fixed "depth" before computing Chao1
# and observed_otus. Shannon is scale-invariant.
SCALE_DEPTH = 50_000


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    otu = pd.read_csv(PROCESSED_DIR / "otu_clean.csv", index_col=0)
    meta = pd.read_csv(PROCESSED_DIR / "metadata_clean.csv", index_col=0)
    tax = pd.read_csv(PROCESSED_DIR / "taxonomy.csv", index_col=0)
    return otu, meta, tax


# ============================================================
# Diversity
# ============================================================

def alpha_diversity_per_sample(otu_relabund: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed by sample_id with columns
    shannon, observed_otus, chao1.
    """
    from skbio.diversity import alpha_diversity

    # Approximate counts from relative abundances for richness-based metrics.
    counts = (otu_relabund.T * SCALE_DEPTH).round().astype(int)
    sample_ids = list(counts.index)

    shannon = alpha_diversity("shannon", counts.values, ids=sample_ids)
    observed = alpha_diversity("observed_otus", counts.values, ids=sample_ids)
    chao1 = alpha_diversity("chao1", counts.values, ids=sample_ids)

    out = pd.DataFrame({
        "shannon": shannon,
        "observed_otus": observed,
        "chao1": chao1,
    })
    out.index.name = "sample_id"
    return out


def _phylum_of(tax: str) -> str:
    """Extract phylum from a semicolon-delimited QIIME-style taxonomy string."""
    if not isinstance(tax, str):
        return "Unknown"
    for part in tax.split(";"):
        p = part.strip()
        if p.startswith("p__"):
            return p[3:] or "Unknown"
    return "Unknown"


def phylum_table(otu_relabund: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the OTU table to the phylum level. Returns phyla x samples."""
    phyla = taxonomy.loc[otu_relabund.index, "taxonomy"].map(_phylum_of)
    grouped = otu_relabund.groupby(phyla).sum()
    grouped.index.name = "phylum"
    return grouped


def firmicutes_bacteroidetes_ratio(otu_relabund: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.Series:
    phy = phylum_table(otu_relabund, taxonomy)
    firm = phy.loc["Firmicutes"] if "Firmicutes" in phy.index else pd.Series(0.0, index=phy.columns)
    bact = phy.loc["Bacteroidetes"] if "Bacteroidetes" in phy.index else pd.Series(0.0, index=phy.columns)
    # Avoid div-by-zero; tiny epsilon mirrors how clinicians eyeball ratios.
    return (firm / bact.replace(0, np.nan)).rename("fb_ratio")


# ============================================================
# Reference ranges & dysbiosis
# ============================================================

def healthy_reference_ranges(div_df: pd.DataFrame, fb: pd.Series, k_sd: float = 1.5) -> dict:
    """Cohort mean ± k_sd*std for each metric. The 'healthy reference' for the
    demo is the cohort itself — i.e. dysbiosis is relative to the population."""
    ref = {}
    for col in div_df.columns:
        s = div_df[col].dropna()
        ref[col] = {"mean": float(s.mean()), "std": float(s.std()),
                    "low": float(s.mean() - k_sd * s.std()),
                    "high": float(s.mean() + k_sd * s.std())}
    fb_clean = fb.dropna()
    ref["fb_ratio"] = {
        "mean": float(fb_clean.mean()),
        "std": float(fb_clean.std()),
        "low": float(fb_clean.mean() - k_sd * fb_clean.std()),
        "high": float(fb_clean.mean() + k_sd * fb_clean.std()),
    }
    return ref


def flag_dysbiosis(div_df: pd.DataFrame, fb: pd.Series, k_sd: float = 1.5) -> tuple[pd.Series, dict]:
    """Mark a sample dysbiotic if any of its metrics is more than k_sd SD
    away from the cohort mean. Returns (bool series, per-sample reasons).
    """
    ref = healthy_reference_ranges(div_df, fb, k_sd=k_sd)
    flags = pd.Series(False, index=div_df.index)
    reasons: dict[str, list[str]] = {sid: [] for sid in div_df.index}

    for metric in ["shannon", "observed_otus", "chao1"]:
        r = ref[metric]
        for sid, v in div_df[metric].items():
            if pd.isna(v):
                continue
            if v < r["low"]:
                flags[sid] = True
                reasons[sid].append(f"{metric} low ({v:.2f} < {r['low']:.2f})")
            elif v > r["high"]:
                flags[sid] = True
                reasons[sid].append(f"{metric} high ({v:.2f} > {r['high']:.2f})")

    r = ref["fb_ratio"]
    for sid, v in fb.items():
        if pd.isna(v) or sid not in flags.index:
            continue
        if v < r["low"]:
            flags[sid] = True
            reasons[sid].append(f"F/B ratio low ({v:.2f} < {r['low']:.2f})")
        elif v > r["high"]:
            flags[sid] = True
            reasons[sid].append(f"F/B ratio high ({v:.2f} > {r['high']:.2f})")

    return flags, reasons


# ============================================================
# Plots
# ============================================================

def plot_top10_taxa(otu_relabund: pd.DataFrame, taxonomy: pd.DataFrame, out: Path) -> None:
    mean_ab = otu_relabund.mean(axis=1).sort_values(ascending=False).head(10)
    labels = []
    for oid in mean_ab.index:
        tax = taxonomy.loc[oid, "taxonomy"] if oid in taxonomy.index else oid
        # Last non-empty rank for readability
        parts = [p.strip() for p in str(tax).split(";") if p.strip().split("__")[-1]]
        labels.append(parts[-1] if parts else oid)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(mean_ab))[::-1], mean_ab.values, color="#3E7CB1")
    ax.set_yticks(range(len(mean_ab))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean relative abundance")
    ax.set_title("Top 10 most abundant taxa across cohort")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_shannon_box(div_df: pd.DataFrame, dysbiosis: pd.Series, out: Path,
                      meta: Optional[pd.DataFrame] = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    df = div_df.copy()
    df["dysbiosis"] = dysbiosis.map({True: "flagged", False: "normal"}).fillna("normal")
    sns.boxplot(data=df, x="dysbiosis", y="shannon", ax=ax,
                palette={"normal": "#7FB069", "flagged": "#D7263D"})
    sns.stripplot(data=df, x="dysbiosis", y="shannon", ax=ax, color="black", size=3, alpha=0.5)
    ax.set_title("Shannon diversity by dysbiosis flag")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_phylum_composition(phy: pd.DataFrame, out: Path, max_samples: int = 30) -> None:
    """Stacked bar across (up to max_samples) samples."""
    samples = phy.columns[:max_samples]
    sub = phy[samples].T  # samples x phyla
    sub = sub.div(sub.sum(axis=1), axis=0)
    # Order phyla by overall mean for nicer visuals
    sub = sub[sub.mean(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11, 5))
    sub.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.95)
    ax.set_ylabel("relative abundance")
    ax.set_xlabel("sample")
    ax.set_title("Phylum-level composition")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


# ============================================================
# Driver
# ============================================================

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[diversity] loading processed data")
    otu, meta, tax = load_processed()

    print("[diversity] computing alpha diversity")
    div = alpha_diversity_per_sample(otu)

    print("[diversity] computing F/B ratio")
    fb = firmicutes_bacteroidetes_ratio(otu, tax)

    print("[diversity] flagging dysbiosis")
    flags, reasons = flag_dysbiosis(div, fb)

    # Persist combined frame for downstream tools (agent + visualizations)
    combined = div.copy()
    combined["fb_ratio"] = fb
    combined["dysbiotic"] = flags
    combined["reasons"] = [" | ".join(reasons.get(sid, [])) for sid in combined.index]
    combined.to_csv(PROCESSED_DIR / "diversity.csv")
    print(f"[save] {PROCESSED_DIR / 'diversity.csv'}")

    print("[diversity] generating plots")
    plot_top10_taxa(otu, tax, PLOTS_DIR / "top10_taxa.png")
    plot_shannon_box(div, flags, PLOTS_DIR / "shannon_boxplot.png", meta=meta)
    phy = phylum_table(otu, tax)
    plot_phylum_composition(phy, PLOTS_DIR / "phylum_composition.png")

    n_flagged = int(flags.sum())
    print(f"[diversity] done. {n_flagged}/{len(flags)} samples flagged as dysbiotic.")


if __name__ == "__main__":
    main()
