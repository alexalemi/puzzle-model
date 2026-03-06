# Development Log

## 2026-03-06 (evening)

### Added all data sources — no source offset needed

Re-enabled all three data sources (speedpuzzling.com, myspeedpuzzling.com, mallory) after previously narrowing to MSP-only to fix the bimodal alpha issue. The question was whether competition times (SP) would show systematic bias vs self-reported times (MSP), requiring a source-specific gamma offset.

**Diagnostic results**: No offset needed. Mean residuals on the test set:
- speedpuzzling: -3.95 mB (N=1,783)
- myspeedpuzzling: -4.35 mB (N=24,561)
- mallory: -19.22 mB (N=14, just 1 puzzler)

The <1 mB difference between SP and MSP is negligible. The model naturally handles population differences through puzzler-level alphas. SP puzzlers are systematically slower on shared puzzles (e.g. "Be Mine_300": SP mean=3468 vs MSP=3296), but this reflects the different populations, not measurement bias — and the individual alphas absorb it.

SP data has notably lower noise (test RMSE 91 vs 121 mB, mean LPD -5.27 vs -5.60), consistent with controlled competition conditions vs self-reported timing.

**Combined dataset**: 113K train / 26K test, 6,157 puzzlers, 16,493 puzzles. 144 shared puzzles serve as bridge puzzles for cross-source calibration.

**Model performance** (all sources, model_2c): test mean LPD = -5.5905, essentially identical to MSP-only (-5.5906). Adding 11K SP observations didn't degrade fit at all.

Changed `load_solo_completed` default from `source="myspeedpuzzling"` to `source=None`.

## 2026-03-06 (afternoon)

### Fixed bimodal puzzler ratings by pinning mu

The bimodality turned out to be caused by an identifiability gap in `log_time = mu + alpha_i + beta_j + ...`. With mu sampled from a diffuse prior (Normal(3500, 900)), any constant could shift freely between mu, alpha, and beta. During SVI fitting, mu would land somewhere arbitrary, so alpha=0 didn't correspond to the true population center. The hierarchical prior shrinks n=1 puzzlers toward alpha=0, but that was the wrong target — hence two modes: well-observed puzzlers at the data-driven location, n=1 puzzlers shrunk to the wrong center.

**Fix**: Pin mu to the empirical mean of log_time from training data (~3587 mB). This completely removes the additive degree of freedom. Now alpha=0 genuinely means "average puzzler" and shrinkage works correctly. Implemented via `numpyro.deterministic("mu", mu_fixed)` so it still appears in the trace for downstream code.

Also tightened hyperpriors on sigma_alpha/sigma_beta from HalfNormal(500) to HalfNormal(300) and simplified the active model set to just three: model_1t, model_2, model_2c (dropped 0, 1, 3). Removed the post-hoc centering block in refit_all.py since it's no longer needed.

All three models improved in test mean LPD:
- model_2c: -5.62 (was -5.63)
- model_2: -5.66 (was -5.70)
- model_1t: -5.70 (was -5.74)

### Explorer page improvements

Iterated on the explorer and it's in a very good place now:
- Piece-count distribution chart uses log-scaled x-axis
- Puzzle difficulty distribution now shows mB (mu + beta) instead of raw beta
- "Predicted time curves by puzzler type" replaced with "Learned piece-count correction" showing just the Σ c_k φ_k(N) basis correction term
- Added a zoomed uncertainty panel below the correction curve showing posterior deciles as filled bands (the main curve's uncertainty is sub-pixel on a 700 mB y-range, so the zoom panel shows deviation from mean on a ±7 mB scale)

### Next steps

Model is well-behaved on MSP-only data. Next: add back the other data sources (speedpuzzling.com, usajigsaw) and see if the model still holds up. May need to reintroduce the gamma (source offset) parameter.

## 2026-03-06

### Simplifying to MSP-only data

Narrowed the dataset down to just the myspeedpuzzling.com (MSP) source to really nail down the model before adding complexity. With only one source, we removed the gamma (source offset) parameter entirely. The idea is to reduce moving parts so we can focus on getting the core model right.

### Bimodality in puzzler ratings persists

The puzzler rating distribution still shows clear bimodality: n=1 puzzlers cluster at much higher ratings than puzzlers with more observations. We'd like to fix this. One hypothesis is that the post-hoc recentering transformations we added earlier (converting raw alpha to mB ratings with offsets) may be contributing to or exaggerating this effect. Worth investigating whether the bimodality is intrinsic to the posterior or an artifact of how we display the results.

### Plan going forward

1. Really nail down the model on MSP data alone — get the parameterization right, work out the basis function terms, and resolve the bimodality issue.
2. Once we're very happy with the model and its parameterization, then mix back in the other data sources (speedpuzzling.com, usajigsaw).
