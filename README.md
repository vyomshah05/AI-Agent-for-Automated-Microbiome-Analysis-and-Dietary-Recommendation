# Gut Agentic AI Pipeline

A Claude-powered agent that analyses gut microbiome OTU data from the
American Gut Project, detects dysbiosis, queries PubMed for diet evidence,
and produces a plain-language microbiome health report.

## Quick start

```bash
cd american_gut_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
# edit .env and fill in GEMINI_API_KEY and ENTREZ_EMAIL
python demo.py
```

`demo.py` runs the full pipeline end-to-end: downloads (or generates) data,
computes diversity, picks 3 sample IDs, and runs the agent on each.

## Project layout

```
american_gut_agent/
├── requirements.txt      pinned dependencies
├── .env.template         copy to .env and add your keys
├── data/
│   ├── raw/              downloaded BIOM + mapping (or synthetic fallback)
│   └── processed/        otu_clean.csv, metadata_clean.csv
├── outputs/
│   ├── plots/            PNG figures
│   └── reports/          per-sample agent reports
├── data_loader.py        download + BIOM parsing + filtering
├── diversity_analysis.py alpha diversity, F/B ratio, dysbiosis flagging
├── pubmed_tool.py        NCBI Entrez search wrapper
├── agent.py              Claude tool-using agent
├── demo.py               end-to-end runner
└── visualizations.py     presentation figures
```

## Data source

The American Gut Project's official FTP (`ftp.microbio.me`) has been
intermittent for years. `data_loader.py` will:

1. Try the FTP, then a few HTTPS mirror candidates.
2. If all downloads fail, generate a small synthetic-but-realistic dataset
   (~30 samples × ~150 taxa) so the demo always runs.

If you have BIOM/mapping files already, drop them in `data/raw/` and the
loader will skip the download step.

## Troubleshooting

- **`scikit-bio` install fails on macOS** — make sure Xcode CLT is installed
  (`xcode-select --install`) and you're on Python 3.11+.
- **PubMed throttles** — set `ENTREZ_EMAIL` so NCBI gives you the polite
  rate limit. The tool already retries with exponential backoff.
- **`GEMINI_API_KEY` missing** — `agent.py` will refuse to start. Get a
  free key at https://aistudio.google.com/app/apikey.

## Disclaimer

This pipeline is for educational and demonstrational use only. The reports
produced are not medical advice.
