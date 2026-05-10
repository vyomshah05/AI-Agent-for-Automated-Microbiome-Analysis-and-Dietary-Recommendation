"""Download, parse, filter, and persist American Gut Project OTU data.

Pipeline:
  download_american_gut() -> data/raw/
  load_biom() + load_mapping()
  filter_to_stool()
  filter_low_abundance()
  to_relative_abundance()
  save_processed() -> data/processed/{otu_clean.csv, metadata_clean.csv}

If the real American Gut FTP is unreachable, falls back to a small
synthetic-but-realistic dataset so demo.py always runs end-to-end.
"""
from __future__ import annotations

import ftplib
import io
import os
import socket
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ----- paths -----
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
PROCESSED_DIR = HERE / "data" / "processed"

OTU_OUT = PROCESSED_DIR / "otu_clean.csv"
META_OUT = PROCESSED_DIR / "metadata_clean.csv"
TAXONOMY_OUT = PROCESSED_DIR / "taxonomy.csv"

# Candidate filenames the American Gut FTP has used at one time or another.
FTP_HOST = "ftp.microbio.me"
FTP_DIR = "/AmericanGut/latest/"
BIOM_CANDIDATES = ["ag.biom", "ag-cleaned.biom", "ag_fecal.biom", "03-otus/ag.biom"]
MAP_CANDIDATES = ["ag_map_with_alpha.txt", "ag-cleaned.txt", "ag_map.txt"]


# ============================================================
# Download
# ============================================================

def _try_ftp_download(filename: str, dest: Path, timeout: int = 30) -> bool:
    """Try to download a single file from the American Gut FTP."""
    print(f"  [ftp] trying {filename} ...", flush=True)
    try:
        with ftplib.FTP(FTP_HOST, timeout=timeout) as ftp:
            ftp.login()
            ftp.cwd(FTP_DIR)
            buf = io.BytesIO()
            ftp.retrbinary(f"RETR {filename}", buf.write)
            dest.write_bytes(buf.getvalue())
        print(f"  [ftp] ok -> {dest.name} ({dest.stat().st_size} bytes)")
        return True
    except (ftplib.all_errors, socket.timeout, OSError) as e:
        print(f"  [ftp] failed ({e!s})")
        return False


def download_american_gut(dest: Path = RAW_DIR) -> tuple[Optional[Path], Optional[Path]]:
    """Try to fetch a BIOM table + mapping file. Returns (biom_path, map_path).

    Either may be None if the corresponding download failed. Caller decides
    whether to fall back to the synthetic dataset.
    """
    dest.mkdir(parents=True, exist_ok=True)
    print("[download] attempting American Gut FTP...")

    biom_path: Optional[Path] = None
    for name in BIOM_CANDIDATES:
        out = dest / Path(name).name
        if out.exists() and out.stat().st_size > 0:
            print(f"  [skip] {out.name} already present")
            biom_path = out
            break
        if _try_ftp_download(name, out):
            biom_path = out
            break

    map_path: Optional[Path] = None
    for name in MAP_CANDIDATES:
        out = dest / Path(name).name
        if out.exists() and out.stat().st_size > 0:
            print(f"  [skip] {out.name} already present")
            map_path = out
            break
        if _try_ftp_download(name, out):
            map_path = out
            break

    return biom_path, map_path


# ============================================================
# Parsing
# ============================================================

def load_biom(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a BIOM file. Returns (counts_df: taxa x samples, taxonomy_df).

    taxonomy_df is indexed by OTU id with one column 'taxonomy' (semicolon
    string). If the BIOM file has no taxonomy metadata the column is empty.
    """
    print(f"[biom] loading {path}")
    from biom import load_table

    table = load_table(str(path))
    counts = table.matrix_data.toarray()
    obs_ids = list(table.ids("observation"))
    sample_ids = list(table.ids("sample"))
    counts_df = pd.DataFrame(counts, index=obs_ids, columns=sample_ids)

    taxonomy = []
    for oid in obs_ids:
        meta = table.metadata(oid, axis="observation") or {}
        tax = meta.get("taxonomy") if isinstance(meta, dict) else None
        if isinstance(tax, list):
            tax = ";".join(tax)
        taxonomy.append(tax or "")
    tax_df = pd.DataFrame({"taxonomy": taxonomy}, index=obs_ids)

    print(f"  taxa={len(obs_ids)}  samples={len(sample_ids)}")
    return counts_df, tax_df


def load_mapping(path: Path) -> pd.DataFrame:
    """Load the QIIME-style tab-separated mapping/metadata file."""
    print(f"[meta] loading {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    # Normalize column names: lowercase, strip
    df.columns = [c.strip().lower().lstrip("#") for c in df.columns]
    if "sampleid" in df.columns:
        df = df.rename(columns={"sampleid": "sample_id"})
    if "sample_id" not in df.columns:
        # fallback to first column
        df = df.rename(columns={df.columns[0]: "sample_id"})
    df = df.set_index("sample_id")
    print(f"  {len(df)} samples, {len(df.columns)} metadata columns")
    return df


# ============================================================
# Filtering / normalization
# ============================================================

STOOL_VALUES = {"uberon:feces", "uberon:0001988", "feces", "stool"}


def filter_to_stool(counts: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only stool/fecal samples in both counts (taxa x samples) and mapping."""
    site_cols = [c for c in ("body_site", "body_habitat", "env_material") if c in mapping.columns]
    if not site_cols:
        print("[filter] no body_site column found — keeping all samples")
        common = counts.columns.intersection(mapping.index)
        return counts[common], mapping.loc[common]

    keep_mask = pd.Series(False, index=mapping.index)
    for c in site_cols:
        keep_mask = keep_mask | mapping[c].astype(str).str.lower().isin(STOOL_VALUES)
    keep_ids = mapping.index[keep_mask]
    common = counts.columns.intersection(keep_ids)
    print(f"[filter] stool samples: {len(common)} / {len(mapping)}")
    return counts[common], mapping.loc[common]


def filter_low_abundance(counts: pd.DataFrame, min_mean_relabund: float = 1e-4) -> pd.DataFrame:
    """Drop taxa whose mean relative abundance across samples is below threshold.

    Default threshold is 0.01% (1e-4). Operates on raw counts; computes
    relative abundance internally for the filter, then returns the filtered
    raw counts (so downstream code can choose to renormalize).
    """
    col_sums = counts.sum(axis=0).replace(0, np.nan)
    relabund = counts.divide(col_sums, axis=1).fillna(0)
    means = relabund.mean(axis=1)
    keep = means >= min_mean_relabund
    print(f"[filter] taxa kept: {int(keep.sum())} / {len(keep)} (mean relabund >= {min_mean_relabund})")
    return counts.loc[keep]


def to_relative_abundance(counts: pd.DataFrame) -> pd.DataFrame:
    """Column-normalize raw counts to relative abundance (sums to 1 per sample)."""
    col_sums = counts.sum(axis=0).replace(0, np.nan)
    return counts.divide(col_sums, axis=1).fillna(0)


# ============================================================
# Persistence
# ============================================================

def save_processed(otu_relabund: pd.DataFrame, mapping: pd.DataFrame, taxonomy: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    otu_relabund.to_csv(OTU_OUT)
    mapping.to_csv(META_OUT)
    taxonomy.to_csv(TAXONOMY_OUT)
    print(f"[save] {OTU_OUT}  ({otu_relabund.shape[0]} taxa x {otu_relabund.shape[1]} samples)")
    print(f"[save] {META_OUT}")
    print(f"[save] {TAXONOMY_OUT}")


# ============================================================
# Synthetic fallback
# ============================================================

# Realistic phylum-level priors for healthy western gut microbiomes.
SYNTHETIC_PHYLA = {
    "Firmicutes":       0.50,
    "Bacteroidetes":    0.35,
    "Actinobacteria":   0.06,
    "Proteobacteria":   0.05,
    "Verrucomicrobia":  0.03,
    "Fusobacteria":     0.01,
}

# A handful of representative genera per phylum (for taxonomy strings).
SYNTHETIC_GENERA = {
    "Firmicutes":       ["Faecalibacterium", "Roseburia", "Ruminococcus", "Lactobacillus", "Blautia", "Eubacterium", "Clostridium", "Streptococcus"],
    "Bacteroidetes":    ["Bacteroides", "Prevotella", "Parabacteroides", "Alistipes"],
    "Actinobacteria":   ["Bifidobacterium", "Collinsella"],
    "Proteobacteria":   ["Escherichia", "Klebsiella", "Sutterella"],
    "Verrucomicrobia":  ["Akkermansia"],
    "Fusobacteria":     ["Fusobacterium"],
}


def make_synthetic_fallback(n_samples: int = 30, n_taxa_per_genus: int = 6, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a small but realistic synthetic OTU table + metadata + taxonomy.

    Diversity varies sample-to-sample so the agent has interesting cases to
    flag. F/B ratio is also intentionally varied.
    """
    print("[synthetic] generating fallback dataset")
    rng = np.random.default_rng(seed)

    # Build taxonomy strings: kingdom; phylum; ...; genus; species_id
    taxa: list[str] = []
    tax_strings: list[str] = []
    phylum_per_taxon: list[str] = []
    for phylum, genera in SYNTHETIC_GENERA.items():
        for genus in genera:
            for i in range(n_taxa_per_genus):
                tid = f"{genus}_OTU{i}"
                taxa.append(tid)
                tax_strings.append(
                    f"k__Bacteria;p__{phylum};c__;o__;f__;g__{genus};s__{genus}_sp{i}"
                )
                phylum_per_taxon.append(phylum)

    # Per-sample composition: draw a phylum mixture, then draw genus shares
    # within phylum, then split among that genus's OTUs.
    sample_ids = [f"AG_S{i:03d}" for i in range(n_samples)]
    counts = np.zeros((len(taxa), n_samples), dtype=float)

    for s_idx in range(n_samples):
        # Vary phylum priors so some samples are dysbiotic.
        phylum_alpha = np.array([SYNTHETIC_PHYLA[p] for p in SYNTHETIC_PHYLA]) * 50
        if s_idx % 7 == 0:  # low-diversity / high-Firmicutes
            phylum_alpha[0] *= 4.0
            phylum_alpha[1] *= 0.2
        elif s_idx % 11 == 0:  # high-Bacteroidetes (low F/B)
            phylum_alpha[0] *= 0.2
            phylum_alpha[1] *= 4.0
        elif s_idx % 13 == 0:  # bloom in Proteobacteria (dysbiotic signature)
            phylum_alpha[3] *= 6.0

        phylum_mix = rng.dirichlet(phylum_alpha)
        depth = rng.integers(20_000, 60_000)

        for p_idx, phylum in enumerate(SYNTHETIC_PHYLA):
            phylum_taxa_idx = [i for i, ph in enumerate(phylum_per_taxon) if ph == phylum]
            if not phylum_taxa_idx:
                continue
            taxon_alpha = rng.uniform(0.2, 1.5, size=len(phylum_taxa_idx))
            within = rng.dirichlet(taxon_alpha)
            counts[phylum_taxa_idx, s_idx] = phylum_mix[p_idx] * depth * within

    counts = np.round(counts).astype(int)
    counts_df = pd.DataFrame(counts, index=taxa, columns=sample_ids)

    # Metadata
    meta = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    meta["body_site"] = "UBERON:feces"
    meta["age_years"] = rng.integers(20, 70, size=n_samples).astype(str)
    meta["sex"] = rng.choice(["male", "female"], size=n_samples)
    meta["diet_type"] = rng.choice(
        ["omnivore", "vegetarian", "vegan"], size=n_samples, p=[0.7, 0.2, 0.1]
    )
    meta["bmi_cat"] = rng.choice(
        ["normal", "overweight", "obese", "underweight"], size=n_samples, p=[0.5, 0.3, 0.15, 0.05]
    )

    tax_df = pd.DataFrame({"taxonomy": tax_strings}, index=taxa)
    return counts_df, meta, tax_df


# ============================================================
# Driver
# ============================================================

def main(force: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not force and OTU_OUT.exists() and META_OUT.exists():
        print(f"[main] processed files already present at {PROCESSED_DIR}, skipping. "
              "Pass force=True to regenerate.")
        return

    biom_path, map_path = download_american_gut(RAW_DIR)

    if biom_path and map_path:
        try:
            counts, taxonomy = load_biom(biom_path)
            mapping = load_mapping(map_path)
            counts, mapping = filter_to_stool(counts, mapping)
            if counts.shape[1] == 0:
                raise RuntimeError("no stool samples after filter — falling back")
            counts = filter_low_abundance(counts)
            taxonomy = taxonomy.loc[taxonomy.index.intersection(counts.index)]
        except Exception as e:
            print(f"[main] real-data path failed ({e!s}) — using synthetic fallback")
            counts, mapping, taxonomy = make_synthetic_fallback()
            counts = filter_low_abundance(counts)
            taxonomy = taxonomy.loc[counts.index]
    else:
        print("[main] download failed — using synthetic fallback")
        counts, mapping, taxonomy = make_synthetic_fallback()
        counts = filter_low_abundance(counts)
        taxonomy = taxonomy.loc[counts.index]

    relabund = to_relative_abundance(counts)
    save_processed(relabund, mapping, taxonomy)
    print("[main] done.")


if __name__ == "__main__":
    main(force=os.environ.get("FORCE_RELOAD") == "1")
