# Puzzle Model

Bayesian latent factor model for speed puzzling competition times. Uses NumPyro for inference.

## Project Layout

```
src/puzzle_model/     # Main package (src layout, hatchling build)
  data.py             # Loading, filtering, encoding, train/test split, repeat features
  basis.py            # DEPRECATED: old normalized basis functions (retained for compat)
  model.py            # NumPyro models 1t, 2c, 2r (fixed mu, physical basis, Student-t)
  inference.py        # MCMC (NUTS) and SVI runners
  predict.py          # Predictions, rankings, cold-start
  evaluate.py         # RMSE, MAE, coverage, WAIC

scraper/              # Data collection
  speedpuzzling.py    # PDF scraper for speedpuzzling.com results
  usajigsaw.py        # Scraper for USA Jigsaw Puzzle Association results
  wjpf.py             # Scraper for WJPC (World Jigsaw Puzzle Championship) results
  wjpf_common.py      # Shared Playwright/table-parsing utilities for worldjigsawpuzzle.org
  myspeedpuzzling.py  # Scraper for myspeedpuzzling.com (puzzles, times, players, images)
  combine.py          # Merges all 5 sources into combined_results.csv

scripts/              # Analysis & utility scripts
  refit_all.py        # Refit solo models (1t, 2c, 2r), regenerate explorer_data.json
  refit_team.py       # Refit joint solo+team model, regenerate explorer_team_data.json
  slim_data.py        # Compress explorer_data.json (round floats, subsample)
  suggest_player_links.py  # Find candidate MSP→SP player matches for review
  compute_waic.py     # WAIC computation for model comparison
  diagnostic_sources.py    # Check residuals by data source
  analyze_noise.py    # Residual analysis, outputs noise_data.json for noise.html

notebooks/            # Marimo notebooks
  01_eda.py           # Exploratory data analysis
  02_model_fit.py     # Model fitting and diagnostics

data/
  raw/pdfs/           # ~680 downloaded result PDFs from speedpuzzling.com
  raw/myspeedpuzzling/  # puzzles.csv, solving_times.csv, players.csv, images/
  raw/html/           # Downloaded HTML pages
  processed/          # Generated CSVs (do not edit by hand)
  mappings/           # player_links.csv (manual MSP→SP identity links)
  MalloryPuzzleData.csv  # Personal puzzle dataset (1 puzzler)

explorer.html         # D3.js interactive explorer (distill.pub style)
explorer_data.json    # Precomputed data for explorer (~7.7 MB)
explorer_puzzler_obs.json  # Per-observation posterior predictive data (~23 MB)
deploy.sh             # Deployment script
```

## Data Sources

Five data sources, combined by `python -m scraper.combine`:

### 1. speedpuzzling.com (`source=speedpuzzling`)
- US-based virtual/in-person competitions with controlled conditions
- Same puzzle for all competitors, timed, first-attempt
- Scraped from PDF result sheets → `data/processed/speedpuzzling_results.csv`
- Competitor names: "Last, First" format (real names)
- ~29K rows, ~2K puzzlers
- Columns: `puzzle_brand` (manufacturer), `puzzle_name` (actual title), `puzzle_pieces`
- Run with: `python -m scraper.speedpuzzling`

### 2. usajigsaw (`source=usajigsaw`)
- USA Jigsaw Puzzle Association championship results
- Small dataset (~1K rows) with nationals/regional competition data
- No `puzzle_name` or `puzzle_brand` columns
- `division` uses "individual" (not "solo") — normalized downstream
- Has `time_limit_seconds`

### 3. wjpf (`source=wjpf`)
- World Jigsaw Puzzle Championship (WJPC) results from worldjigsawpuzzle.org
- International competitions with controlled conditions, same puzzle per round
- Scraped by `scraper/wjpf.py` → `data/processed/wjpf_results.csv`
- Shared Playwright infrastructure with usajigsaw.py (same website, `scraper/wjpf_common.py`)
- WJPC years: 2019, 2022-2025; divisions: individual (500pc), pairs (1000pc), teams (2000pc)
- Multiple rounds per year: qualifying (A-F), semifinals (S1/S2), final
- No `puzzle_name` or `puzzle_brand` (secret competition puzzles); puzzle_name filled with `{event_id}_{round}` in combine.py
- Competitor names: "First Last" format → normalized to "Last, First" in combine.py
- Country extracted from flag `<img>` tags, appended to `origin`
- `time_limit_seconds` and `finished_date` extracted from page JavaScript
- ~4.5K rows, ~2K+ puzzlers (heavily European)
- Run with: `python -m scraper.wjpf`

### 4. myspeedpuzzling.com (`source=myspeedpuzzling`)
- Global self-reported solving times from myspeedpuzzling.com
- Scraped by `scraper/myspeedpuzzling.py` → `data/raw/myspeedpuzzling/`
- Files: `puzzles.csv`, `solving_times.csv`, `players.csv`, `competitions.csv`, `images/`
- Competitor names: "Display Name (msp:uuid_prefix)" for uniqueness
- `first_attempt` field distinguishes unseen vs repeat solves
- ~302K rows (~243K first-attempt + 59K repeats), ~5K puzzlers
- Heavily European population (DE, AU, CZ, SE, FI) vs SP's US focus

### 5. mallory (`source=mallory`)
- Personal puzzle dataset from `data/MalloryPuzzleData.csv`
- 167 rows, 1 puzzler; canonical name "Alemi, Mallory"
- Linked to MSP identity via `data/mappings/player_links.csv`

### Combined Dataset (~338K rows, ~10K puzzlers, ~20.5K puzzles)

Schema (`data/processed/combined_results.csv`):

```
source, event_id, year, division, round, rank, competitor_name, origin,
time_seconds, completed, pieces_completed, puzzle_pieces, puzzle_brand,
puzzle_name, finished_date, first_attempt, time_limit_seconds
```

### Player Disambiguation

- UJ individual names normalized to SP "Last, First" format → ~237 linked
- WJPF "First Last" names normalized to "Last, First": SP-matched first, then default "last word = surname"
- Manual corrections in `WJPF_NAME_CORRECTIONS` for multi-word surnames (Spanish, etc.)
- `data/mappings/player_links.csv`: manual MSP player_id → SP name links (applied in combine.py)
- `scripts/suggest_player_links.py`: finds candidate matches for human review
- Cross-source overlaps: SP∩UJ=237 players, SP∩MSP=1 (Mallory), SP∩WJPF≈50-100 (US competitors at Worlds)

## Key Data Pipeline

```python
from puzzle_model.data import load_solo_completed, create_puzzle_id, encode_indices
from puzzle_model.data import prepare_model_data, add_repeat_features

df = load_solo_completed()          # Filter to solo + completed + valid times
df = create_puzzle_id(df)           # puzzle_id = puzzle_name + "_" + puzzle_pieces
df, puzzler_map, puzzle_map = encode_indices(df)  # Integer indices for NumPyro
data_dict = prepare_model_data(df)  # Computes mu_fixed = mean(log_time)
df = add_repeat_features(df)        # solve_number, days_since_last, days_since_first
```

- `puzzle_id = puzzle_name + "_" + puzzle_pieces` — shared across events/sources
- Same puzzle at different events shares one beta parameter (~139 bridge puzzles)
- Response variable: milliBels (`log_time = 1000 * log10(time_seconds)`)

## Model Architecture

Three models, incrementally complex. All use fixed mu and Student-t likelihood:

- **Model 1t**: `mu_fixed + alpha_i + beta_j + c*log(N)` — basic IRT with piece-count effect
- **Model 2c**: + physical piece-count basis `[√N, N, N·log(N), N²]` + per-puzzler velocity
- **Model 2r**: extends 2c + repeat-solve practice effect `gamma*log(solve_number)`

Key design choices:
- **mu is fixed** to `mean(log_time)` from training data, not sampled (eliminates identifiability issue)
- Non-centered parameterization via `LocScaleReparam`
- Physical basis functions defined in `model.py` (`N_REF=500`, `PHYS_BASIS_NAMES`)
- Models 1t, 2c fitted on first-attempt data only; Model 2r on all data including repeats
- Fitted via SVI (AutoNormal guide, 5000 steps, lr=0.005)

## Explorer

- `explorer.html` renders from `explorer_data.json` and `explorer_puzzler_obs.json`
- Shows: data overview, model specs, basis functions, model comparison, deep dive (Model 2r)
- Puzzler/puzzle rankings with mB ratings and Wilson bounds
- Puzzler table projected to 2026 with velocity (ΔmB/yr)
- Regenerate with: `uv run python -m scripts.refit_all`
- Serve with: `python -m http.server 8010`

## DEVLOG.md

A lab-notebook-style development log. Newer entries at the top, organized by date. Records decisions, observations, hypotheses, and plans as they happen. When asked to add notes, record what was said faithfully (fixing grammar/spelling/readability) without making things up or editorializing. Include motivations and reasoning, not just bare facts.

## Tools & Commands

- **Python >=3.14**, managed with `uv`
- Run any module: `uv run python -m scraper.speedpuzzling`
- Combine data: `uv run python -m scraper.combine`
- Refit models: `uv run python scripts/refit_all.py`
- Notebooks: `uv run marimo edit notebooks/01_eda.py`
