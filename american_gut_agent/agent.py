"""Gemini tool-using agent for microbiome analysis.

Loads the processed CSVs once, defines four tools, and runs a standard
Gemini function-calling loop until the agent emits a final report.

Tools:
  calculate_diversity(sample_id)
  get_top_taxa(sample_id, n=10)
  search_pubmed(genus_name)
  generate_report(sample_id)

Entry point:
  run_agent(sample_id) -> {"text": str, "transcript": list[dict]}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

import diversity_analysis as da
import pubmed_tool

load_dotenv()

HERE = Path(__file__).resolve().parent
PROCESSED_DIR = HERE / "data" / "processed"

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_TOOL_ITERATIONS = 12

# ============================================================
# Cached data (loaded once at import time)
# ============================================================

_OTU: pd.DataFrame | None = None
_META: pd.DataFrame | None = None
_TAX: pd.DataFrame | None = None
_DIV: pd.DataFrame | None = None
_FB: pd.Series | None = None
_FLAGS: pd.Series | None = None
_REASONS: dict[str, list[str]] | None = None
_REF: dict | None = None


def _ensure_loaded() -> None:
    global _OTU, _META, _TAX, _DIV, _FB, _FLAGS, _REASONS, _REF
    if _OTU is not None:
        return
    print("[agent] loading processed data + diversity")
    _OTU, _META, _TAX = da.load_processed()
    _DIV = da.alpha_diversity_per_sample(_OTU)
    _FB = da.firmicutes_bacteroidetes_ratio(_OTU, _TAX)
    _FLAGS, _REASONS = da.flag_dysbiosis(_DIV, _FB)
    _REF = da.healthy_reference_ranges(_DIV, _FB)
    print(f"[agent] ready. {len(_DIV)} samples, {_OTU.shape[0]} taxa")


def _genus_of(tax: str) -> str:
    if not isinstance(tax, str):
        return ""
    for part in tax.split(";"):
        p = part.strip()
        if p.startswith("g__"):
            return p[3:] or ""
    return ""


# ============================================================
# Tool implementations
# ============================================================

def tool_calculate_diversity(sample_id: str) -> dict[str, Any]:
    _ensure_loaded()
    if sample_id not in _DIV.index:
        return {"error": f"sample_id '{sample_id}' not found"}
    row = _DIV.loc[sample_id]
    fb = float(_FB.get(sample_id, float("nan")))
    flagged = bool(_FLAGS.get(sample_id, False))
    return {
        "sample_id": sample_id,
        "shannon": float(row["shannon"]),
        "observed_otus": float(row["observed_otus"]),
        "chao1": float(row["chao1"]),
        "firmicutes_bacteroidetes_ratio": fb,
        "dysbiotic": flagged,
        "reasons": _REASONS.get(sample_id, []),
        "cohort_reference_ranges": _REF,
    }


def tool_get_top_taxa(sample_id: str, n: int = 10) -> dict[str, Any]:
    _ensure_loaded()
    if sample_id not in _OTU.columns:
        return {"error": f"sample_id '{sample_id}' not found"}
    n = max(1, min(int(n), 50))
    col = _OTU[sample_id].sort_values(ascending=False).head(n)
    out = []
    for oid, ab in col.items():
        tax = _TAX.loc[oid, "taxonomy"] if oid in _TAX.index else ""
        out.append({
            "otu_id": oid,
            "taxonomy": tax,
            "genus": _genus_of(tax),
            "relative_abundance": float(ab),
        })
    return {"sample_id": sample_id, "top_taxa": out}


def tool_search_pubmed(genus_name: str) -> dict[str, Any]:
    hits = pubmed_tool.search_pubmed(genus_name)
    return {"genus": genus_name, "hits": [dict(h) for h in hits]}


def tool_generate_report(sample_id: str) -> dict[str, Any]:
    """Compile a structured payload the agent will rephrase as prose."""
    _ensure_loaded()
    div = tool_calculate_diversity(sample_id)
    if "error" in div:
        return div
    top = tool_get_top_taxa(sample_id, n=10)["top_taxa"]

    if sample_id in _OTU.columns:
        cohort_mean = _OTU.mean(axis=1)
        sample_col = _OTU[sample_id]
        dev = (sample_col - cohort_mean).sort_values(ascending=False).head(5)
        flagged_high = []
        for oid, d in dev.items():
            tax = _TAX.loc[oid, "taxonomy"] if oid in _TAX.index else ""
            flagged_high.append({
                "otu_id": oid,
                "genus": _genus_of(tax),
                "taxonomy": tax,
                "sample_relabund": float(sample_col[oid]),
                "cohort_mean_relabund": float(cohort_mean[oid]),
                "delta": float(d),
            })
    else:
        flagged_high = []

    metadata = {}
    if sample_id in _META.index:
        metadata = {k: (None if pd.isna(v) else v) for k, v in _META.loc[sample_id].to_dict().items()}

    return {
        "sample_id": sample_id,
        "diversity": div,
        "top_taxa": top,
        "flagged_high_taxa": flagged_high,
        "metadata": metadata,
    }


TOOL_REGISTRY = {
    "calculate_diversity": tool_calculate_diversity,
    "get_top_taxa": tool_get_top_taxa,
    "search_pubmed": tool_search_pubmed,
    "generate_report": tool_generate_report,
}


# ============================================================
# Tool schemas (google.genai types format)
# ============================================================

def _build_tools():
    from google.genai import types

    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="calculate_diversity",
            description=(
                "Compute alpha diversity (Shannon, observed OTUs, Chao1), the "
                "Firmicutes/Bacteroidetes ratio, and a dysbiosis flag for a single "
                "sample. Also returns cohort reference ranges for comparison."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"sample_id": types.Schema(type=types.Type.STRING, description="Sample ID to analyze.")},
                required=["sample_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_top_taxa",
            description="Return the top N most abundant taxa for a sample with relative abundance and taxonomy.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "sample_id": types.Schema(type=types.Type.STRING),
                    "n": types.Schema(type=types.Type.INTEGER, description="Number of top taxa (1-50, default 10)."),
                },
                required=["sample_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_pubmed",
            description="Search PubMed for up to 3 abstracts linking a bacterial genus to diet, nutrition, probiotics, or health outcomes.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"genus_name": types.Schema(type=types.Type.STRING, description="Bacterial genus, e.g. 'Bifidobacterium'.")},
                required=["genus_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="generate_report",
            description=(
                "Compile a structured evidence payload (diversity, top taxa, "
                "flagged-high taxa, metadata) for a sample. Call this when you "
                "are ready to write the final report."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"sample_id": types.Schema(type=types.Type.STRING)},
                required=["sample_id"],
            ),
        ),
    ])]


SYSTEM_PROMPT = """You are a microbiome analysis assistant for a bioinformatics class demo.

You are given a single sample ID from the American Gut Project (or a synthetic
fallback dataset with the same shape). Your job:

1. Call calculate_diversity to get alpha-diversity metrics + dysbiosis flag.
2. Call get_top_taxa to see what dominates the community.
3. Call generate_report once you understand the sample, to fetch flagged-high taxa.
4. For the 2-3 most interesting genera (those flagged as unusually high or
   conspicuous in the top taxa), call search_pubmed to ground recommendations
   in literature.
5. Synthesize a final plain-language MICROBIOME HEALTH REPORT with these
   sections, in order:

   - Sample summary (1-2 sentences, include any metadata like diet/age)
   - Community composition (top dominant taxa, what they typically do)
   - Diversity assessment vs cohort reference ranges (cite actual numbers)
   - Top 3 flagged taxa (high or low) with brief context
   - Dietary recommendations (3-5 bullets), each citing a PubMed PMID where possible
   - Probiotic suggestions (1-3 bullets) when supported by literature
   - Overall microbiome health score: X/10 with one-sentence justification
   - Educational disclaimer (this is NOT medical advice)

Write the final report in clear prose. Use markdown headings. When you cite
a paper, use the format: (PMID: 12345678, Year). Be specific with numbers.
Do not invent PMIDs — only cite IDs that came back from search_pubmed.

Reason briefly between tool calls about what you learned and what to do
next. Stop calling tools once you have enough to write the report."""


# ============================================================
# Agent loop
# ============================================================

def run_agent(sample_id: str, model: str = DEFAULT_MODEL, verbose: bool = True) -> dict[str, Any]:
    """Run the Gemini agent on a single sample. Returns final text + transcript."""
    _ensure_loaded()
    if sample_id not in _DIV.index:
        raise ValueError(f"sample_id '{sample_id}' not in dataset")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.template -> .env and add your key "
            "(get one free at https://aistudio.google.com/app/apikey)."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=_build_tools(),
    )

    user_msg = (
        f"Please analyze sample_id='{sample_id}' from the American Gut Project "
        f"dataset and produce the final microbiome health report."
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_msg)])
    ]

    transcript: list[dict[str, Any]] = []
    final_text: str | None = None

    if verbose:
        print(f"\n{'=' * 70}\n[agent] starting on {sample_id} (model={model})\n{'=' * 70}")

    for step in range(MAX_TOOL_ITERATIONS):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        fn_calls = []
        text_parts = []
        for part in candidate.content.parts:
            if part.function_call:
                fn_calls.append(part.function_call)
            elif part.text:
                text_parts.append(part.text)

        for t in text_parts:
            transcript.append({"step": step, "kind": "thought", "text": t})
            if verbose:
                print(f"\n[step {step}] thought:\n{t}\n")

        if not fn_calls:
            final_text = "\n".join(text_parts).strip()
            if verbose:
                print(f"\n[agent] finished after {step + 1} steps.")
            break

        # Execute each tool and build function-response parts.
        response_parts: list[types.Part] = []
        for fc in fn_calls:
            name = fc.name
            args = dict(fc.args)
            transcript.append({"step": step, "kind": "tool_use", "name": name, "input": args})
            if verbose:
                print(f"[step {step}] tool_use: {name}({json.dumps(args)})")

            fn = TOOL_REGISTRY.get(name)
            if fn is None:
                result: Any = {"error": f"unknown tool '{name}'"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}

            transcript.append({"step": step, "kind": "tool_result", "name": name, "output": result})
            if verbose:
                preview = json.dumps(result, default=str)[:300]
                print(f"[step {step}] tool_result[{name}]: {preview}...")

            response_parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )

        contents.append(types.Content(role="user", parts=response_parts))
    else:
        if verbose:
            print(f"[agent] stopped at iteration cap ({MAX_TOOL_ITERATIONS}).")
        final_text = next(
            (e["text"] for e in reversed(transcript) if e["kind"] == "thought"),
            "(no final report produced — iteration cap hit)",
        )

    return {"text": final_text or "", "transcript": transcript}


if __name__ == "__main__":
    import sys

    sid = sys.argv[1] if len(sys.argv) > 1 else None
    if sid is None:
        _ensure_loaded()
        sid = _DIV.index[0]
        print(f"[agent] no sample_id given, using first: {sid}")
    result = run_agent(sid)
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result["text"])
