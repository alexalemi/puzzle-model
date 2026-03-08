# Puzzle Model

A statistical model for speed puzzling — rating every puzzler and puzzle from competitive solving times.

**[Explore the results →](https://alexalemi.com/puzzle-model)**

## What is this?

Speed puzzling is competitive jigsaw puzzle solving. Competitors race to complete the same puzzle, and times vary enormously depending on the puzzler's skill, the puzzle's difficulty, the number of pieces, and whether you're solving solo or on a team.

This project fits a Bayesian model to roughly **400,000 solving times** across **8,600 puzzlers** and **22,000 puzzles**, producing a skill rating for every puzzler and a difficulty rating for every puzzle in the dataset. Think of it like an Elo system, but for jigsaw puzzles.

The [interactive explorer](https://alexalemi.com/puzzle-model) lets you browse puzzler and puzzle rankings, see how piece count affects solving time, and look up individual puzzlers' histories.

## How the ratings work

Every solving time gets decomposed into contributions from:

- **Puzzler skill** — how fast you are relative to average
- **Puzzle difficulty** — how hard this particular puzzle is relative to average
- **Piece count** — a learned curve capturing how time scales with puzzle size
- **Velocity** — whether a puzzler is improving (or declining) over time
- **Practice** — how much faster you get on repeat solves of the same puzzle

Ratings are measured in **milliBels (mB)**, a logarithmic unit where every 100 mB corresponds to about a 26% difference in solving time. A puzzler rated 200 mB better than another will, all else equal, finish about 60% faster.

### Piece-count scaling

A 1000-piece puzzle doesn't just take twice as long as a 500-piece puzzle — the relationship is more complex. The model breaks piece-count scaling into four physical processes that contribute to solving time:

| Process | Scales as | What it captures |
|---------|-----------|-----------------|
| Border assembly | √N | Edge pieces grow with the perimeter |
| Per-piece work | N | Each piece takes roughly constant time to place |
| Search | N log N | Finding where a piece goes requires scanning |
| Pairwise matching | N² | Worst-case trial-and-error fitting |

The model learns the relative contribution of each process from the data. At typical competition sizes (300–1000 pieces), per-piece work and search dominate. The quadratic term is heavily suppressed — competitive puzzlers rarely resort to brute-force matching.

### Team predictions

The model also handles team solving. The key insight is that team members work in parallel — their individual solving *rates* add up. But there's coordination overhead: some fraction of the work is serial (you can't both work on the border simultaneously). The model learns this serial fraction from the data and can predict how fast a team of any composition would solve a given puzzle, purely from the individuals' solo ratings.

## Data sources

The model combines data from three public sources:

| Source | Records | Puzzlers | Description |
|--------|---------|----------|-------------|
| [myspeedpuzzling.com](https://myspeedpuzzling.com) | ~372K | ~5K | Global self-reported times, includes repeat solves |
| [speedpuzzling.com](https://speedpuzzling.com) | ~29K | ~2K | US virtual and in-person competitions |
| [USA Jigsaw](https://usajigsaw.org) | ~1K | ~400 | National championship results |

These sources are linked together by shared puzzles — the same puzzle appearing in multiple sources anchors the rating scales together.

## Technical details

The model is a Bayesian [Item Response Theory](https://en.wikipedia.org/wiki/Item_response_theory) (IRT) factorization fitted with [NumPyro](https://num.pyro.ai/) (JAX) using stochastic variational inference. It uses non-centered parameterization, a Student-t likelihood for robustness to outliers, and physically-motivated basis functions for piece-count scaling. The [interactive explorer](https://alexalemi.com/puzzle-model) is built with D3.js.

For full model specifications, see [`src/puzzle_model/model.py`](src/puzzle_model/model.py).

## Project layout

```
src/puzzle_model/        # Core package
  model.py               # NumPyro model definitions
  data.py                # Loading, filtering, encoding
  inference.py           # SVI and MCMC runners
  predict.py             # Predictions, rankings
  evaluate.py            # RMSE, MAE, coverage, WAIC

scraper/                 # Data collection scripts
index.html               # Interactive explorer (joint solo+team model)
explorer_solo.html       # Solo-only model explorer
```
