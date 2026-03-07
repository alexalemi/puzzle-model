# Puzzle Model

Bayesian latent factor model for speed puzzling competition times. Uses NumPyro for inference.

## Project Layout

```
src/puzzle_model/     # Main package (src layout, hatchling build)
  data.py             # Loading, filtering, encoding, train/test split
  basis.py            # Piece-count scaling basis functions
  model.py            # NumPyro models 0-4 (IRT through robust Student-t)
  inference.py        # MCMC (NUTS) and SVI runners
  predict.py          # Predictions, rankings, cold-start
  evaluate.py         # RMSE, MAE, coverage, model comparison

scraper/              # Data collection
  speedpuzzling.py    # PDF scraper for speedpuzzling.com results
  usajigsaw.py        # Scraper for USA Jigsaw Puzzle Association results
  myspeedpuzzling.py  # Scraper for myspeedpuzzling.com (puzzles, times, players, images)
  combine.py          # Merges all sources into combined_results.csv

notebooks/            # Marimo notebooks
  01_eda.py           # Exploratory data analysis
  02_model_fit.py     # Model fitting and diagnostics

data/
  raw/pdfs/           # ~680 downloaded result PDFs from speedpuzzling.com
  processed/          # Generated CSVs (do not edit by hand)
```

## Data Sources

Three data sources, combined by `python -m scraper.combine`:

### 1. speedpuzzling.com (`source=speedpuzzling`)
- US-based virtual/in-person competitions with controlled conditions
- Same puzzle for all competitors, timed, first-attempt
- Scraped from PDF result sheets → `data/processed/speedpuzzling_results.csv`
- Competitor names: "Last, First" format (real names)
- ~15K rows, ~2K puzzlers, ~200 puzzles
- Columns: `puzzle_brand` (manufacturer), `puzzle_name` (actual title), `puzzle_pieces`
- Run with: `python -m scraper.speedpuzzling`

### 2. usajigsaw (`source=usajigsaw`)
- USA Jigsaw Puzzle Association championship results
- Small dataset (~1K rows) with nationals/regional competition data
- No `puzzle_name` or `puzzle_brand` columns
- `division` uses "individual" (not "solo") — normalized downstream
- Has `time_limit_seconds`

### 3. myspeedpuzzling.com (`source=myspeedpuzzling`)
- Global self-reported solving times from myspeedpuzzling.com
- Scraped by `scraper/myspeedpuzzling.py` → `data/raw/myspeedpuzzling/`
- Two files: `puzzles.csv` (catalog with names, brands, EANs) and `solving_times.csv`
- Competitor names: "Display Name (msp:uuid_prefix)" for uniqueness
- `first_attempt` field distinguishes unseen vs repeat solves
- ~37K rows, ~3.6K puzzlers, ~400 puzzles
- Heavily European population (DE, AU, CZ, SE, FI) vs SP's US focus

### Combined Schema (`data/processed/combined_results.csv`)

```
source, event_id, year, division, round, rank, competitor_name, origin,
time_seconds, completed, pieces_completed, puzzle_pieces, puzzle_brand,
puzzle_name, time_limit_seconds, first_attempt
```

## Key Data Pipeline

```python
from puzzle_model.data import load_solo_completed, create_puzzle_id, encode_indices

df = load_solo_completed()          # Filter to solo + completed + valid times
df = create_puzzle_id(df)           # puzzle_id = event_id + puzzle_name + puzzle_pieces
df, puzzler_map, puzzle_map = encode_indices(df)  # Integer indices for NumPyro
```

- `create_puzzle_id` builds IDs like `sp_139_Backyard Heroes_500` or `msp_018c0d75_Tuscany Hills_500`
- No shared players between SP and MSP (different name formats, populations)
- Minimal shared puzzles between sources

## Model Architecture

Models 0-4, incrementally complex:
- Model 0: Basic IRT (`mu + alpha_i + beta_j`)
- Model 1: + global piece-count effect
- Model 2: + K=1 latent interaction
- Model 3: + K=3 latent interactions
- Model 4: Student-t likelihood (robust to outliers)

Uses non-centered parameterization via `LocScaleReparam`. Factor dimensions use separate plates per k.

## DEVLOG.md

A lab-notebook-style development log. Newer entries at the top, organized by date. Records decisions, observations, hypotheses, and plans as they happen. When asked to add notes, record what was said faithfully (fixing grammar/spelling/readability) without making things up or editorializing. Include motivations and reasoning, not just bare facts.

## Tools & Commands

- **Python >=3.14**, managed with `uv`
- Run any module: `uv run python -m scraper.speedpuzzling`
- Combine data: `uv run python -m scraper.combine`
- Notebooks: `uv run marimo edit notebooks/01_eda.py`
