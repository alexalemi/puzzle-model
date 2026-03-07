# Development Log

## 2026-03-07

### Rethinking the physical model: PPM, multiplicative scaling, and teams

Brainstorming session about the fundamental model structure. Three interrelated questions: (1) why PPM is incomplete, (2) whether the piece-count effect should be multiplicative, and (3) how to model duos/teams.

#### What the current model actually says in time space

The model is additive in milliBels (log-time), which means it's **fully multiplicative in real-time space**:

```
time = K * speed_i * difficulty_j * T(N)/T(N_ref) * trend * practice
```

where `speed_i = 10^(alpha_i/1000)`, etc. The piece-count function T(N) has the **same shape** for every puzzler — faster puzzlers just compress the entire curve by a constant factor. If puzzler A is 2x faster than B at 300 pieces, they're 2x faster at 1000 pieces too. This is a strong assumption worth examining.

#### PPM: the zeroth-order model

Puzzle enthusiasts use PPM (pieces per minute), which assumes `time = N / rate`. In log space: `log(time) = log(N) - log(rate)`. This is essentially model_1t with a forced slope of 1.

The learned physical basis weights show why PPM is incomplete but not crazy:
- **Dominant term**: `w1*N` (linear per-piece work) — this IS the PPM model
- **Strong secondary**: `w2*N*log(N)` — search complexity; for each piece, you search through O(log N) candidate locations (heuristic narrowing by color/shape/region)
- **Weak**: `w0*sqrt(N)` (setup/edge overhead), `w3*N^2` (pairwise interactions, suppressed)

Over the typical competition range (300-1000 pieces), `log(N)` varies from 5.7 to 6.9 — only 20%. So PPM is a decent approximation within this narrow range, but extrapolation to 2000+ pieces would systematically underpredict times.

#### Should piece-count scaling be multiplicative?

The current multiplicative structure (`time = speed * difficulty * T(N)`) assumes every puzzler performs the same physical processes and their speed advantage is a uniform multiplier across all of them. This is correct if: a puzzler's hands move 2x faster, eyes scan 2x faster, and brain pattern-matches 2x faster — everything scales together.

It breaks down if:
- **Expert search strategies have different complexity**: A skilled puzzler might use better heuristics that reduce the effective search from N*log(N) toward N — their T(N) would have a different *shape*, not just a different *scale*
- **Strategy shifts with puzzle size**: Small puzzles might be "scan and place" (linear), while large puzzles require sorting, grouping, edge-first strategies that change the functional form
- **Difficulty interacts with piece count non-uniformly**: A "hard" puzzle (unusual cut, monochrome) might be hard specifically because of search — so its difficulty multiplier grows with N rather than being constant

A lightweight way to model this: give each puzzler a "search efficiency" parameter that modulates the N*log(N) term:

```
T_i(N) = w1*N + lambda_i * w2*N*log(N) + ...
```

where `lambda_i > 1` means "worse at search" and `lambda_i < 1` means "better at search." This adds only 1 parameter per puzzler and is related to the 2PL IRT discrimination idea. However, it makes the model non-separable (puzzler x puzzle interaction through piece count), complicating inference.

**Empirical test**: Check whether residuals show systematic skill x piece-count interaction. If fast puzzlers consistently have negative residuals on large puzzles (faster than the multiplicative model predicts), that's evidence the multiplicative assumption is too simple. This should be checked before adding model complexity.

#### Team/duo physics: rates add

The cleanest physical model for teams comes from **rate space**. If two puzzlers work independently on different pieces, their solving rates add:

```
r_duo = r_1 + r_2
time_duo = 1 / (r_1 + r_2) = (time_1 * time_2) / (time_1 + time_2)
```

This is the **harmonic mean** of solo times, not the arithmetic or geometric mean. In milliBel space, with `rate_i proportional to 10^(-alpha_i/1000)`:

```
alpha_team = -1000 * log10(sum_k 10^(-alpha_k/1000))
```

For k puzzlers of equal ability, this gives `delta = -1000*log10(k)`:
- Duo: -301 mB (2x speedup)
- Trio: -477 mB (3x speedup)
- Quad: -602 mB (4x speedup)

**Why perfect rate-addition is too optimistic** — real teams face overhead:
1. **Spatial contention**: Can't both work on the same area; edge pieces are limited; physically reaching around each other
2. **Amdahl's law**: Some work is inherently serial — initial sort, final assembly, edge frame, strategy coordination
3. **Communication**: "Has anyone seen a blue piece with two tabs?" — can be positive (avoiding duplicate search) or negative (interruption)
4. **Diminishing returns**: For a 500-piece puzzle with 4 people, each handles ~125 pieces, but spatial/coordination bottlenecks dominate

A practical model adds an overhead term:

```
log_time_team = mu + beta_j + piece_correction(N)
                - 1000*log10(sum_k 10^(-alpha_k/1000))   [rate addition]
                + eta(team_size)                           [coordination overhead]
```

For solo (team_size=1), the rate-addition term simplifies to `+alpha_i`, recovering the current model exactly.

#### Interesting prediction: fast puzzler dominates

The rate-addition model predicts that heterogeneous teams are carried by their best member. Example (all in mB):

| Team composition | Combined rate | Effective alpha |
|---|---|---|
| Two fast (-200 mB each) | 3.17 | -501 mB |
| One fast (-200) + one slow (+200) | 2.22 | -346 mB |
| Two slow (+200 mB each) | 1.26 | -100 mB |

The mixed team is much closer to the fast-fast team than to the slow-slow team. The fast puzzler contributes ~71% of the total rate even though they're only 50% of the headcount. This is testable if team member identities can be linked to solo ratings.

#### Connection to PPM and rates

PPM is literally the rate model: `PPM_i = N / time_i`. For teams, PPM should approximately add: `PPM_team ≈ PPM_1 + PPM_2` (minus overhead). This gives team modeling a natural user-facing interpretation — "our duo has a combined PPM of 12" is intuitive even if the underlying model is more nuanced.

The piece-count question and the team question connect here: if we think in rate space, the question is whether each puzzler has a rate that's constant (PPM model), depends on N (the current model's T(N)), or depends on N in a puzzler-specific way (the lambda_i extension). Teams then just sum whatever the individual rate functions are.

#### Open questions for future work

1. **Empirical check**: Do model 2r residuals show skill x piece-count interaction? (Fast puzzlers systematically beating predictions on large puzzles would motivate per-puzzler scaling.)
2. **Team data parsing**: ~400 pair and ~300 team records exist from USA Jigsaw. Team member names are concatenated and would need parsing + linking to solo identities to test the rate-addition model.
3. **Amdahl fraction**: What fraction of puzzle work is serial? Could be estimated from the team data if the rate-addition + overhead model is fit.
4. **PPM as user-facing metric**: Even if PPM isn't the right model, converting alpha to effective PPM at a reference puzzle size (e.g., 500 pieces) would be intuitive for the community.

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
