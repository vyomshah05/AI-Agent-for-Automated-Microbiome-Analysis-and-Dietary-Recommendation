"""NCBI Entrez (PubMed) search wrapper for the agent.

Given a bacterial genus, returns the top N abstracts of papers linking it
to diet / health outcomes. Handles rate-limiting via exponential backoff
and caches results in-process.

Usage:
    from pubmed_tool import search_pubmed
    hits = search_pubmed("Bifidobacterium")
    for h in hits:
        print(h["pmid"], h["title"])
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv

load_dotenv()


# ============================================================

def _entrez():
    """Lazy import + email setup so importing this module doesn't trip on
    a missing biopython during tests."""
    from Bio import Entrez

    email = os.environ.get("ENTREZ_EMAIL", "anonymous@example.com")
    Entrez.email = email
    return Entrez


def _retry(fn, *, attempts: int = 4, base_delay: float = 1.0):
    """Run fn() with exponential backoff on Entrez failures."""
    delay = base_delay
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except (HTTPError, URLError, OSError, RuntimeError) as e:
            last_exc = e
            print(f"  [pubmed] attempt {i + 1}/{attempts} failed: {e!s}; sleeping {delay:.1f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"PubMed search failed after {attempts} attempts: {last_exc!s}")


# ============================================================

@lru_cache(maxsize=128)
def search_pubmed(genus: str, max_results: int = 3) -> tuple[dict[str, Any], ...]:
    """Search PubMed for papers linking ``genus`` to diet/health outcomes.

    Returns a tuple (so the result is hashable / cacheable) of dicts with
    keys: pmid, title, abstract, year, journal.

    Failures (network, rate limit) yield an empty tuple after retries.
    """
    print(f"[pubmed] search '{genus}' (top {max_results})")
    try:
        Entrez = _entrez()
    except ImportError:
        print("[pubmed] biopython not installed — returning empty result")
        return ()

    query = (
        f'"{genus}"[Title/Abstract] AND '
        '(diet OR dietary OR probiotic OR prebiotic OR fiber OR nutrition OR microbiome)'
    )

    def _esearch():
        with Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance") as h:
            return Entrez.read(h)

    try:
        es = _retry(_esearch)
    except Exception as e:
        print(f"[pubmed] esearch failed: {e!s}")
        return ()

    pmids = es.get("IdList", [])
    if not pmids:
        print(f"[pubmed] no hits for '{genus}'")
        return ()

    def _efetch():
        with Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml") as h:
            return Entrez.read(h)

    try:
        records = _efetch()
    except Exception as e:
        print(f"[pubmed] efetch failed: {e!s}")
        return ()

    out: list[dict[str, Any]] = []
    for art in records.get("PubmedArticle", []):
        try:
            citation = art["MedlineCitation"]
            article = citation["Article"]
            pmid = str(citation["PMID"])
            title = str(article.get("ArticleTitle", "")).strip()
            journal = str(article.get("Journal", {}).get("Title", "")).strip()

            year = ""
            try:
                year = str(article["Journal"]["JournalIssue"]["PubDate"].get("Year", ""))
            except Exception:
                pass

            abstract = ""
            abst = article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abst, list):
                abstract = " ".join(str(x) for x in abst)
            else:
                abstract = str(abst)
            abstract = abstract.strip()

            out.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract[:1500],  # cap to keep agent context small
                "year": year,
                "journal": journal,
            })
        except Exception as e:
            print(f"[pubmed] skipped a record (parse error: {e!s})")

    print(f"[pubmed] returned {len(out)} record(s)")
    return tuple(out)


if __name__ == "__main__":
    import json
    import sys

    genus = sys.argv[1] if len(sys.argv) > 1 else "Bifidobacterium"
    hits = search_pubmed(genus)
    print(json.dumps([dict(h) for h in hits], indent=2))
