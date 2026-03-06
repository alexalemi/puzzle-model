import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# Speed Puzzling: Model Fitting & Comparison")
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import jax
    import numpyro

    numpyro.set_host_device_count(4)

    from puzzle_model.data import (
        load_solo_completed,
        create_puzzle_id,
        encode_indices,
        train_test_split,
        prepare_model_data,
    )
    from puzzle_model.model import MODELS
    from puzzle_model.inference import run_mcmc, run_svi, check_diagnostics, to_arviz, posterior_predictive
    from puzzle_model.evaluate import evaluate_predictions, naive_baselines
    from puzzle_model.predict import puzzler_rankings, puzzle_rankings

    return (
        MODELS,
        create_puzzle_id,
        encode_indices,
        evaluate_predictions,
        load_solo_completed,
        naive_baselines,
        np,
        posterior_predictive,
        prepare_model_data,
        puzzle_rankings,
        puzzler_rankings,
        run_mcmc,
        to_arviz,
        train_test_split,
    )


@app.cell
def _(
    create_puzzle_id,
    encode_indices,
    load_solo_completed,
    prepare_model_data,
    train_test_split,
):
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df, puzzler_lookup, puzzle_lookup = encode_indices(df)
    train_df, test_df = train_test_split(df)

    train_data = prepare_model_data(train_df)
    test_data = prepare_model_data(test_df)

    print(f"Train: {len(train_df):,} observations")
    print(f"Test:  {len(test_df):,} observations")
    print(f"Puzzlers: {train_data['n_puzzlers']:,}, Puzzles: {train_data['n_puzzles']:,}")
    return puzzle_lookup, puzzler_lookup, test_data, train_data


@app.cell
def _(mo):
    mo.md("""
    ## Model 0: Basic IRT (mu + alpha_i + beta_j)
    """)
    return


@app.cell
def _(MODELS, run_mcmc, train_data):
    mcmc_0 = run_mcmc(
        MODELS["model_0"],
        train_data,
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
    )
    mcmc_0.print_summary(exclude_deterministic=True)
    return (mcmc_0,)


@app.cell
def _(
    MODELS,
    evaluate_predictions,
    mcmc_0,
    mo,
    naive_baselines,
    np,
    posterior_predictive,
    test_data,
    train_data,
):
    # Evaluate Model 0
    pred_0 = posterior_predictive(mcmc_0, MODELS["model_0"], test_data)
    metrics_0 = evaluate_predictions(np.array(pred_0["log_time"]), np.array(test_data["log_time"]))
    baselines = naive_baselines(np.array(train_data["log_time"]), np.array(test_data["log_time"]))

    mo.md(f"""
    ### Model 0 Results
    | Metric | Model 0 | Global Mean Baseline |
    |--------|---------|---------------------|
    | RMSE (log) | {metrics_0['rmse_log']:.4f} | {baselines['global_mean_rmse_log']:.4f} |
    | RMSE (sec) | {metrics_0['rmse_seconds']:.1f} | {baselines['global_mean_rmse_seconds']:.1f} |
    | MAE (log)  | {metrics_0['mae_log']:.4f} | — |
    | 90% coverage | {metrics_0['coverage_90']:.1%} | — |
    | 50% coverage | {metrics_0['coverage_50']:.1%} | — |
    """)
    return baselines, metrics_0


@app.cell
def _(mo):
    mo.md("""
    ## Model 1: + Global log(N) Effect
    """)
    return


@app.cell
def _(MODELS, run_mcmc, train_data):
    mcmc_1 = run_mcmc(
        MODELS["model_1"],
        train_data,
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
    )
    mcmc_1.print_summary(exclude_deterministic=True)
    return (mcmc_1,)


@app.cell
def _(
    MODELS,
    baselines,
    evaluate_predictions,
    mcmc_1,
    metrics_0,
    mo,
    np,
    posterior_predictive,
    test_data,
):
    pred_1 = posterior_predictive(mcmc_1, MODELS["model_1"], test_data)
    metrics_1 = evaluate_predictions(np.array(pred_1["log_time"]), np.array(test_data["log_time"]))

    mo.md(f"""
    ### Model 1 Results
    | Metric | Model 0 | Model 1 | Baseline |
    |--------|---------|---------|----------|
    | RMSE (log) | {metrics_0['rmse_log']:.4f} | {metrics_1['rmse_log']:.4f} | {baselines['global_mean_rmse_log']:.4f} |
    | RMSE (sec) | {metrics_0['rmse_seconds']:.1f} | {metrics_1['rmse_seconds']:.1f} | {baselines['global_mean_rmse_seconds']:.1f} |
    | 90% coverage | {metrics_0['coverage_90']:.1%} | {metrics_1['coverage_90']:.1%} | — |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Puzzler & Puzzle Rankings (from Model 1)
    """)
    return


@app.cell
def _(
    mcmc_1,
    mo,
    puzzle_lookup,
    puzzle_rankings,
    puzzler_lookup,
    puzzler_rankings,
):
    top_puzzlers = puzzler_rankings(mcmc_1, puzzler_lookup)[:20]
    top_puzzles = puzzle_rankings(mcmc_1, puzzle_lookup)[:20]

    puzzler_table = "| Rank | Puzzler | Speed Effect (alpha) | Std |\n|------|---------|---------------------|-----|\n"
    for i, (name, mean, std) in enumerate(top_puzzlers):
        puzzler_table += f"| {i+1} | {name} | {mean:.3f} | {std:.3f} |\n"

    puzzle_table = "| Rank | Puzzle | Difficulty (beta) | Std |\n|------|--------|------------------|-----|\n"
    for i, (name, mean, std) in enumerate(top_puzzles):
        puzzle_table += f"| {i+1} | {name} | {mean:.3f} | {std:.3f} |\n"

    mo.md(f"""
    ### Top 20 Fastest Puzzlers
    {puzzler_table}

    ### Top 20 Hardest Puzzles
    {puzzle_table}
    """)
    return


@app.cell
def _(MODELS, mcmc_0, mcmc_1, mo, to_arviz, train_data):
    import arviz as az

    _idata_0 = to_arviz(mcmc_0, model=MODELS["model_0"], data=train_data)
    _idata_1 = to_arviz(mcmc_1, model=MODELS["model_1"], data=train_data)

    _comparison = az.compare({"model_0": _idata_0, "model_1": _idata_1})
    mo.md(f"""
    ### Model Comparison (WAIC/LOO)
    ```
    {_comparison.to_string()}
    ```
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
