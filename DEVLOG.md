# Development Log

## 2026-03-28

### Added worldjigsawpuzzle.org (WJPF) as 5th data source

Added a scraper for World Jigsaw Puzzle Championship (WJPC) results from worldjigsawpuzzle.org. The existing `usajigsaw.py` already scraped USAJPA nationals from this site; the new `wjpf.py` scraper covers the international WJPC championships (2019, 2022-2025).

**Data added**: 7,927 rows across 57 event pages (4,605 solo, 2,363 duo, 959 group), 4,299 unique competitors. All divisions (individual 500pc, pairs 1000pc, teams 2000pc) and all rounds (qualifying A-F, semifinals S1/S2, finals). Shared Playwright infrastructure via new `wjpf_common.py`.

**Cross-source links**: 166 competitors overlap between SP and WJPF (US competitors who attend Worlds: Peng, Roiter, Walter, Black, etc.). These bridge players connect the US-centric SP dataset with the European-heavy WJPC competitor pool.

**Name normalization**: WJPF uses "First Last" format. Two-pass normalization in combine.py: (1) try all word splits against SP canonical "Last, First" names, (2) default to "last word = surname" with manual corrections for multi-word surnames (Spanish double surnames like "Clemente León").

**Puzzle identity**: WJPC puzzles are secret (no puzzle_name/brand). Each round's puzzle gets a unique identity via `puzzle_name = "{event_id}_{round}"` in combine.py.

### Also scraped 2026 USAJPA Nationals prelims (in progress)

The 2026 Nationals is happening in Atlanta (March 27-29). Scraped available prelim results: individual rounds A-D (798 rows) and pairs rounds A-B (400 rows). Added 2026 URLs to usajigsaw.py. Finals not yet posted.

### Out-of-sample prediction accuracy: 2026 Nationals prelims

The model was refitted with WJPF data but *before* 2026 nationals data was added, making the 2026 prelims a true out-of-sample test. Matched 455 of 552 competitors (with completed times) to model ratings.

**Model accuracy** (455 matched competitors):

| Metric | Our Model (mB) | JPAR |
|--------|:---:|:---:|
| Spearman ρ | 0.882 | 0.898 |
| Kendall τ | 0.700 | 0.719 |
| Top 10 accuracy | 3/10 | 5/10 |
| Top 20 accuracy | 12/20 | 13/20 |
| Top 50 accuracy | 36/50 | 34/50 |
| Median rank error | 30 | 29 |

Comparison uses 356 competitors matched to both systems. JPAR has a slight edge (ρ = 0.898 vs 0.882), which makes sense: JPAR is computed exclusively from USAJPA-sanctioned events with the same 500pc competition format, while our model trains on a much broader dataset (MSP self-reported times, different piece counts, international data) and predicts using `alpha` alone without puzzle-specific difficulty.

Both systems strongly predict rankings (ρ > 0.88). The model's main disadvantage is a +16 min systematic bias in absolute times (it assumes average puzzle difficulty; competition puzzles are curated to be easier). After bias correction, RMSE drops from 23.6 to 17.0 min.

**Notable predictions**:
- Alice Rowe: Model #3, JPAR #1, Actual #1
- Mari Black: Model #1, JPAR #15, Actual #25 (both overrated her, model more so)
- Kelly Walter: Model #13, JPAR #4, Actual #3 (JPAR called this better)
- Iryna Shvydchenko: Model #2, JPAR #2, Actual #8 (both overrated, but she's still top tier)

The model's competitive performance despite training on heterogeneous data (300K+ MSP self-reported times, not just sanctioned competitions) is encouraging. The JPAR comparison uses December 2025 published ratings.

### Three-way comparison: Production vs Autoresearch vs JPAR

Ran the autoresearch/mar13 model (20 kept improvements over production, heldout LPD -5.356 vs production's -5.394) on the same data (with WJPF, without 2026 nationals) to see if the improved model closes the gap to JPAR. 352 competitors matched to all three systems.

| Metric | Production (main) | Autoresearch (mar13) | JPAR (Dec 2025) |
|--------|:---:|:---:|:---:|
| Spearman ρ | 0.8812 | 0.8817 | **0.8996** |
| Kendall τ | 0.7005 | 0.7051 | **0.7208** |
| Top 10 | 3/10 | 4/10 | **5/10** |
| Top 20 | 11/20 | **13/20** | **13/20** |
| Top 50 | 36/50 | **37/50** | 35/50 |
| Median rank error | 30 | **28** | **28** |

The autoresearch model shows consistent improvement over production across all metrics: better ρ, τ, top-N accuracy, and median rank error. It ties JPAR on top-20 (13/20) and beats it on top-50 (37 vs 35). But JPAR still wins overall rank correlation (ρ = 0.900 vs 0.882).

**Notable autoresearch improvements in top-20 predictions**:
- Kelly Walter: Production #13 → Autoresearch #5 → Actual #3 (much better)
- Hannah Scott: Production #54 → Autoresearch #40 → Actual #12 (better but still off)
- Amber Whitmill: Production #25 → Autoresearch #20 → Actual #5 (improved)

**Why JPAR still wins**: JPAR is computed exclusively from USAJPA-sanctioned events — the exact same format (500pc, timed, controlled) as the test set. Our model trains on 264K observations across 5 heterogeneous sources including self-reported MSP times, international WJPC data, and multiple piece counts, then predicts using `alpha` alone without puzzle-specific difficulty. The fact that we're within ρ = 0.02 of a purpose-built rating system using a general-purpose Bayesian model is encouraging. The autoresearch improvements (per-puzzler discrimination, noise scaling, practice rate) help but don't fully bridge the format-specificity gap.

### 2026 Nationals puzzle difficulty estimates

Estimated puzzle difficulty (beta) for each prelim round by using velocity-projected alpha values for all matched competitors and computing the residual. Positive beta = harder than average; negative = easier.

**Individual prelims** (500pc):

| Round | Beta (mB) | Avg puzzler time | Matched |
|-------|:---------:|:----------------:|:-------:|
| A | -45.0 | 54m | 118 |
| C | -46.3 | 54m | 112 |
| D | -18.8 | 57m | 118 |
| B | -0.1 | 60m | 107 |

Rounds A and C were the easiest puzzles; Round B was the hardest (nearly average difficulty). All are easier than the training-set average (negative beta), consistent with competition puzzles being curated for completability.

**Pairs prelims** (1000pc):

| Round | Beta (mB) | Avg pair time | Matched |
|-------|:---------:|:-------------:|:-------:|
| A | +218.8 | 99m | 111 |
| B | +352.2 | 135m | 105 |

Pairs Round B was substantially harder than Round A — about 36 minutes longer for an average pair. The positive betas reflect the 1000pc piece count (more pieces = harder, absorbed into beta since we're estimating it as a residual).

### Per-puzzler predictions vs actuals (Mallory's team)

Using velocity-projected alphas and the estimated puzzle betas above.

**Individual:**

| Name | α_proj | Round | Predicted | Actual | Diff |
|------|:------:|:-----:|:---------:|:------:|:----:|
| Mallory Alemi | +2.7 | A | 54.4m | 56.9m | +2.5m |
| Kathy Robbins | +98.7 | C | 67.7m | 64.3m | -3.4m |
| Amanda Messinger | -14.5 | A | 52.3m | 54.9m | +2.6m |
| Megan Bailey | -67.4 | D | 49.2m | 56.3m | +7.1m |

All within ~7 minutes. Mallory and Amanda are predicted almost exactly. Kathy slightly outperformed. Megan's high velocity (+116 mB/yr) may be overshooting.

**Pairs** (using logsumexp + Amdahl + eta_2 team model):

| Pair | Team α | Round | Predicted | Actual | Diff | Rank |
|------|:------:|:-----:|:---------:|:------:|:----:|:----:|
| Mallory & Kathy | -410.2 | B | 52.5m | 56.9m | +4.4m | 82/200 |
| Amanda & Megan | -501.1 | A | 31.3m | 35.1m | +3.8m | 42/200 |

Amanda & Megan are the stronger pair — Megan's projected α of -67 and Amanda's -15 give a team α of -501 vs Mallory & Kathy's -410. The logsumexp aggregation means the faster member dominates.

**Team alpha decomposition** (Amdahl serial fraction s = 0.278, duo correction η₂ = -12.1):
- Mallory (+2.7) & Kathy (+98.7): parallel = -643.6, + Amdahl +245.5, + η₂ -12.1 = **-410.2**
- Amanda (-14.5) & Megan (-67.4): parallel = -734.4, + Amdahl +245.5, + η₂ -12.1 = **-501.1**

The 28% serial fraction (s = 0.278) means about 28% of the work can't be parallelized (sorting, coordination), so two puzzlers are ~1.5x as fast as the faster member alone rather than 2x.

## 2026-03-06 (late night)

### Physical basis model promoted to production

Replaced the old 5-basis model (which added [log(N), √N, N, N·log(N), N²] in log-time space) with a physically-motivated model that sums processes additively in time space, then takes log₁₀. The old approach caused time to grow as 10^(c·N²) — catastrophically wrong for extrapolation.

**New model**: time(N) = Σ w_k g_k(N) where g = [√N, N, N·log(N), N²], weights positive via w = exp(log_w). Piece correction centered at N_REF=500. At large N, log(time) grows as at most 2·log(N) — physically sensible.

**Results**: Test mean LPD improved from -5.59 (old basis) to -5.49 (physical). Model 1t baseline: -5.66.

**Changes made**:
- `model.py`: Removed model_2, _compute_phi, basis imports. Added N_REF=500, PHYS_BASIS_NAMES, physical model_2c with log_w parameter.
- `refit_all.py`: Removed basis normalization. Stats now has `log_w`, `N_REF`, `phys_basis_names` instead of `c_basis`/`basis_mean`/`basis_std`. Added `basis_components` for per-component mB curves.
- `explorer.html`: Updated model table (2 models), basis chart (physical components), prediction calculator (physical formula), scalar params (log_w_0..3), correction curve title, model spec text.
- `basis.py`: Marked deprecated (retained for backward compat with old scripts).

Learned log_w values: [√N: 0.99, N: 4.04, N·log(N): 3.28, N²: -1.25]. The dominant process is N (per-piece work), with N·log(N) (search) secondary. N² is suppressed (log_w ≈ -1.25, so w ≈ 0.29).

## 2026-03-06 (night)

### Model 3 experiment: more latent factors hurt

Fitted model_3 (K=3 latent puzzler×puzzle interactions through basis functions) to see if richer interactions improve on model_2's K=1. Result: worse across the board.

- **Test mean LPD**: -5.7078 (model_2: -5.6599)
- **Test RMSE**: 146 mB (model_2: 134 mB)
- **Test MAE**: 72 mB (model_2: 68 mB)

The ~70K extra parameters (3× the latent loadings and factors) overwhelm the available signal. Puzzle identity via beta_j already captures most puzzle-specific variation, leaving little for additional latent dimensions to explain. Model 3 is removed from the active set.

### Survey of potential model improvements

Investigated six directions for improving on model_2c. Recording findings here for future reference — none implemented yet.

**1. Heteroscedastic σ(N) — most promising**
Noise clearly varies with puzzle size and source. SP test RMSE = 91 mB vs MSP = 121 mB; large puzzles are noisier than small ones. Parameterize as `log(σ) = σ_0 + σ_pieces * log(N)`, giving piece-count-dependent noise without per-puzzle σ parameters. Most likely to improve LPD since the current model uses a single global σ (plus Student-t df).

**2. Brand random effect**
Hierarchical structure: `beta_j ~ Normal(gamma_brand, sigma_within)`. 1,099 distinct brands in the data, but highly imbalanced — Ravensburger alone is 49% of observations (74K obs). Other top brands: Trefl (4.3%), Buffalo Games (3.5%). ICC = 0.055 (brand explains ~5.5% of puzzle difficulty variance). At 500 pieces, brand effects range from -30 to +75 mB (Educa hardest, Schmidt/Trefl easiest). Small overall effect but helps cold-start predictions for new puzzles. Prerequisite: normalize brand names ("Masterpieces" vs "MasterPieces" etc.).

**3. Source-specific σ**
Fit one sigma per source (SP, MSP, mallory). Simple to implement but overlaps with σ(N) since source and piece-count distribution are confounded. Could combine: `log(σ) = σ_source + σ_pieces * log(N)`.

**4. Discrimination parameter (2PL IRT)**
Replace `alpha_i + beta_j` with `a_j * alpha_i + beta_j` — some puzzles are more skill-revealing than others. Adds N_puzzle parameters and creates identifiability risk (a_j and alpha_i can trade off). Standard in psychometrics but harder to fit with SVI.

**5. Mixture IRT**
Detect latent subpopulations (e.g., casual vs competitive puzzlers) with different skill distributions. Complex inference — mixture models are hard with SVI's AutoNormal guide (mode-seeking, label switching). Would need custom guide or MCMC.

**6. Non-linear learning curves**
Power law or exponential decay instead of current linear velocity (delta_i). Theoretically more realistic, but current per-puzzler data is too sparse to reliably fit non-linear curves. Linear velocity is a reasonable approximation given the data.

**Decision**: None implemented for now. Model 2c remains the best model (test mean LPD = -5.5905). Heteroscedastic σ(N) is the most promising next step when we revisit model improvements.

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
