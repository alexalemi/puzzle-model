# Development Log

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
