// Puzzle Model: Bayesian Latent Factor Model for Speed Puzzling
// Working paper / method outline

#set document(
  title: "A Bayesian Latent Factor Model for Speed Puzzling Times",
  author: "Alexander Alemi",
  date: datetime(year: 2026, month: 3, day: 7),
)

#set page(paper: "us-letter", margin: (x: 1.25in, y: 1.25in), numbering: "1")
#set text(size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 14pt, weight: "bold", it)
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold", it)
  v(0.3em)
}

// ── Title ──

#align(center)[
  #text(size: 20pt, weight: "bold")[
    A Bayesian Latent Factor Model \ for Speed Puzzling Times
  ]

  #v(0.8cm)

  #text(size: 12pt)[Alexander Alemi]

  #v(0.3cm)

  #text(size: 10pt, fill: luma(100))[Draft — March 2026]
]

#v(1cm)

// ── Abstract ──

#block(inset: (left: 2em, right: 2em))[
  #text(style: "italic")[
    We develop a Bayesian latent factor model for jigsaw puzzle solving times, drawing on item response theory (IRT) to jointly estimate puzzler skill and puzzle difficulty from heterogeneous competition and self-reported data. The model uses a logarithmic response scale (milliBels), physically motivated piece-count basis functions, per-puzzler velocity trends, a practice effect for repeat solves, and a team model that derives team performance from individual solo skills via logsumexp rate combination and Amdahl's law. Inference is performed via stochastic variational inference (SVI) with NumPyro. We combine data from four sources totaling ~333K solving times across ~8.6K puzzlers and ~20.5K puzzles.
  ]
]

#v(0.5cm)

= Introduction

Speed puzzling — competitive jigsaw puzzle solving — has grown rapidly as a recreational and competitive activity. Competitions such as those organized by speedpuzzling.com and the USA Jigsaw Puzzle Association draw hundreds of competitors, while platforms like myspeedpuzzling.com enable thousands to self-report solving times globally.

A natural question arises: can we build a principled rating system for puzzlers and puzzles simultaneously? The structure of the problem — many puzzlers solving overlapping subsets of puzzles — maps directly onto the framework of item response theory (IRT), widely used in educational testing and game rating systems.

This paper develops a hierarchy of Bayesian models for puzzle solving times, culminating in a unified model that handles both solo and team solving and includes:
- Latent puzzler skill and puzzle difficulty parameters,
- Physically motivated piece-count scaling,
- Per-puzzler temporal velocity (improvement rate),
- A practice effect for repeat solves,
- A team model deriving group performance from individual skills,
- A Student-$t$ likelihood for robustness to outliers.

We combine four heterogeneous data sources and use shared puzzle identities to anchor cross-source calibration, even when the puzzler populations barely overlap.


= Data <data>

== Sources

We combine four data sources into a unified dataset of ~333K solving times:

+ *speedpuzzling.com* (~29K observations, ~2K puzzlers). US-based virtual and in-person competitions under controlled conditions — same puzzle for all competitors, timed, first attempt. Results scraped from PDF result sheets.

+ *USA Jigsaw Puzzle Association* (~1K observations). National and regional championship results with structured divisions.

+ *myspeedpuzzling.com* (~302K observations, ~5K puzzlers). A global self-reporting platform. Includes both first-attempt and repeat solves, with timestamps. The population skews European relative to the US-centric competition data.

+ *Personal dataset* (167 observations, 1 puzzler). A single puzzler's comprehensive log, serving as a cross-source anchor.

== Preprocessing

We filter to solo, completed solves with valid times and piece counts. A unified puzzle identity is constructed as #raw("puzzle_name") + `"_"` + $N$ where $N$ is the piece count. This allows the same physical puzzle appearing across events or sources to share a single difficulty parameter, enabling cross-source calibration through ~139 bridge puzzles.

Player identities are disambiguated across sources through a combination of name normalization and manual linking via a curated mapping file.

== Response variable

We work on the *milliBel* (mB) scale:
$ y = 1000 dot log_10(t) $ <eq-mb>
where $t$ is the solving time in seconds. This logarithmic scale has several advantages: solving times are log-normally distributed to first approximation, the scale is additive (parameters combine linearly), and has interpretable units — 1000 mB corresponds to a factor of 10 in time, and 100 mB (1 dB) corresponds to approximately a 26% change.


= Model <model>

== General structure

The model takes the form of an additive latent factor model on the milliBel scale. For observation $k$ of puzzler $i$ solving puzzle $j$:
$ y_k = mu + alpha_i + beta_j + f(N_j) + v_(i k) + r_k + epsilon_k $ <eq-general>
where:
- $mu$ is the grand mean (fixed),
- $alpha_i$ is the puzzler effect (skill; lower = faster),
- $beta_j$ is the puzzle effect (difficulty; higher = harder),
- $f(N_j)$ is a piece-count correction,
- $v_(i k)$ is a velocity (temporal trend) term,
- $r_k$ is a repeat-solve practice effect,
- $epsilon_k$ is observation noise.

The grand mean $mu$ is *fixed* to the empirical mean of $y$ in the training data, rather than given a prior and sampled. This eliminates the additive identifiability between $mu$, $alpha$, and $beta$ that would otherwise make the model unidentifiable up to a global shift. With $mu$ fixed, $alpha_i = 0$ genuinely represents an average-skill puzzler, ensuring that hierarchical shrinkage pulls sparse observations toward the correct center.

== Priors

The puzzler and puzzle effects receive hierarchical normal priors:
$
  alpha_i | sigma_alpha &tilde cal(N)(0, sigma_alpha), quad& sigma_alpha &tilde "HalfNormal"(300) \
  beta_j | sigma_beta &tilde cal(N)(0, sigma_beta), quad& sigma_beta &tilde "HalfNormal"(300)
$

All latent effects use a *non-centered parameterization* via `LocScaleReparam` to improve sampling and variational inference geometry.

== Likelihood

All models use a Student-$t$ likelihood for robustness to outliers:
$ y_k | dots tilde "Student"-t(nu, hat(y)_k, sigma) $
with:
$
  nu &tilde "Gamma"(2, 0.1) \
  sigma &tilde "HalfNormal"(500)
$

The heavy-tailed likelihood naturally downweights anomalous solving times (interrupted attempts, transcription errors) without explicit outlier detection.


== Model 1t: Baseline with log piece-count

The simplest model includes a single scalar coefficient for the log of piece count:
$ hat(y)_k = mu + alpha_i + beta_j + c dot log N_j $ <eq-1t>
where $c tilde cal(N)(0, 500)$. This captures the broad trend that larger puzzles take longer, but cannot capture the physical nonlinearity of the relationship.

== Model 2c: Physical basis + velocity

=== Piece-count correction

Rather than a single log-linear term, we model the piece-count dependence using physically motivated basis functions that combine additively in *time* space:
$ T(N) = sum_(l=1)^4 w_l g_l (N) $ <eq-phys-basis>
where the basis functions $g_l$ correspond to distinct physical processes:
$
  g_1(N) &= sqrt(N) quad &&"(edge assembly)" \
  g_2(N) &= N quad &&"(linear scanning)" \
  g_3(N) &= N log N quad &&"(sorting / search)" \
  g_4(N) &= N^2 quad &&"(exhaustive matching)"
$

The weights are constrained to be positive via $w_l = exp(xi_l)$ with $xi_l tilde cal(N)(0, 5)$. The contribution in mB is then centered at a reference piece count $N_"ref" = 500$:
$ f(N) = 1000 / (ln 10) dot [ln T(N) - ln T(N_"ref")] $ <eq-piece-correction>

This formulation ensures that: (a) the correction is zero at the reference count, so $beta_j$ retains its interpretation; (b) extrapolation to large $N$ is physically bounded (log-time grows at most as $2 log N$); and (c) the model can learn which physical processes dominate.

=== Per-puzzler velocity

To capture improvement (or decline) over time, each puzzler gets a velocity parameter:
$ v_(i k) = (delta_0 + delta_i) dot (t_k - t_"ref") $ <eq-velocity>
where $t_k$ is the fractional year of observation, $t_"ref" = 2025$, $delta_0 tilde cal(N)(0, 100)$ is a population-level trend, and $delta_i tilde cal(N)(0, sigma_delta)$ with $sigma_delta tilde "HalfNormal"(100)$.

== Model 2r: Repeat-solve practice effect

Model 2r extends 2c with a practice effect for repeat solves of the same puzzle:
$ r_k = gamma dot ln(s_k) $ <eq-practice>
where $s_k$ is the solve number (1-indexed, so the effect vanishes for first attempts since $ln 1 = 0$), and $gamma tilde cal(N)(0, 200)$. The logarithmic form captures diminishing returns — each doubling of solve count yields the same marginal improvement.

For the myspeedpuzzling data, when the platform reports a repeat solve but no prior solve appears in our data, we conservatively set $s_k = 2$.

Models 1t and 2c are fitted on first-attempt data only; Model 2r is fitted on all data including repeats.

== Team model: Joint solo and team solving <team-model>

The team model extends Model 2r to jointly handle solo, duo, and group solving within a single unified framework. For solo observations ($K=1$), all team corrections vanish and the model reduces exactly to Model 2r. For teams ($K >= 2$), three additional mechanisms apply.

=== Parallel combination of skills

When $K$ puzzlers solve a puzzle together, their individual solving _rates_ (not times) combine. Each team member $m$ has a time-adjusted skill:
$ alpha_m (t) = alpha_m + (delta_0 + delta_m) dot (t - t_"ref") $

The pure-parallel team skill is obtained via the logsumexp of individual rates. Defining the rate of puzzler $m$ as $rho_m = exp(-alpha_m (t) "/" M)$ where $M = 1000 "/" ln 10$ is the mB scale constant, the combined team skill is:
$ alpha_"parallel" = -M dot ln sum_(m=1)^K rho_m $ <eq-logsumexp>

This is the exact formula for adding rates on the mB scale: if two equally-skilled puzzlers work in perfect parallel, their combined time is half the solo time, corresponding to a shift of $M dot ln 2 approx 301$ mB.

=== Amdahl's law correction

In practice, not all work can be parallelized. A serial fraction $s$ bounds the speedup achievable by a team. The Amdahl correction adds a positive (slower) term:
$ a_K = M dot ln(1 + s dot (K - 1)) $ <eq-amdahl>
where $s = sigma("logit"_s)$ and $"logit"_s tilde cal(N)(0, 2)$. For $K=1$ this vanishes ($ln 1 = 0$), and for large teams the slowdown approaches $M dot ln(s dot K)$.

=== Per-bucket residual corrections

To absorb any remaining systematic differences between team sizes, we include residual corrections $eta_b$ for three team-size buckets ($K=2$, $K=3$, $K >= 4$):
$ eta_b tilde cal(N)(0, 50), quad b in {1, 2, 3} $
These are zero for solo observations.

=== Combined team prediction

The effective puzzler contribution for a team observation is:
$ alpha_"eff" = alpha_"parallel" + a_K + eta_(b(K)) $ <eq-team-alpha>

The full model is then:
$ hat(y)_k = mu + alpha_"eff" + beta_j + f(N_j) + r_k $
with a separate observation scale $sigma_"team" tilde "HalfNormal"(500)$ for team observations (solo observations continue to use $sigma$).



= Inference <inference>

== Stochastic variational inference

We use *stochastic variational inference* (SVI) with an `AutoNormal` mean-field guide in NumPyro. The guide approximates the posterior with independent normal distributions for each (unconstrained) parameter. Optimization uses Adam with learning rate $0.005$ for $5000$ steps.

SVI is preferred over MCMC (NUTS) for this problem due to the large number of latent variables (~8.6K puzzler effects + ~20.5K puzzle effects), making full MCMC prohibitively expensive.

== Non-centered parameterization

All hierarchical effects use a non-centered parameterization:
$ alpha_i = sigma_alpha dot tilde(alpha)_i, quad tilde(alpha)_i tilde cal(N)(0,1) $
This reparameterization reduces the posterior geometry's funnel-like structure, dramatically improving both MCMC and SVI convergence.


= Evaluation <evaluation>

We evaluate models using:
- *Train/test split*: 80/20 random split stratified by puzzler.
- *Test mean log predictive density*: The primary comparison metric — higher values indicate better calibrated predictions. Model 2c achieves the best test LPD among first-attempt models.
- *RMSE and MAE*: Point prediction accuracy in mB.
- *Coverage*: Calibration of posterior predictive intervals.


= Results <results>

// TODO: Fill in with current results


= Discussion <discussion>

The model structure — additive latent factors on a log scale with physically motivated corrections — provides an interpretable and extensible framework for rating puzzlers and puzzles simultaneously. Key design choices include:

- *Fixed $mu$* eliminates the identifiability problem inherent in additive IRT models, ensuring shrinkage works correctly.
- *Physical basis functions* provide principled extrapolation across piece counts, unlike purely empirical polynomials.
- *Student-$t$ likelihood* provides automatic robustness without explicit outlier modeling.
- *Cross-source calibration* through shared puzzle identities anchors the different data populations to a common scale.
- *Unified team model* derives team performance from individual solo skills without separate team-level parameters, leveraging the much larger solo dataset to inform team predictions.

Future directions include incorporating puzzle-level covariates such as brand or image complexity, and extending the per-puzzler velocity model to capture non-linear improvement trajectories.


// ── Bibliography placeholder ──

// #bibliography("refs.bib")
