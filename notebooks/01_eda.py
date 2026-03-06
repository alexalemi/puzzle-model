import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# Speed Puzzling: Exploratory Data Analysis")
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from puzzle_model.data import load_solo_completed, create_puzzle_id, encode_indices
    return create_puzzle_id, load_solo_completed, plt


@app.cell
def _(create_puzzle_id, load_solo_completed):
    df = load_solo_completed()
    df = create_puzzle_id(df)
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Dataset Summary

    - **Total solo completed records**: {len(df):,}
    - **Unique puzzlers**: {df['competitor_name'].nunique():,}
    - **Unique puzzles**: {df['puzzle_id'].nunique():,}
    - **Piece counts**: {sorted(df['puzzle_pieces'].unique())}
    - **Log-time mean**: {df['log_time'].mean():.3f}, std: {df['log_time'].std():.3f}
    """)
    return


@app.cell
def _(df, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    # Distribution of log times
    _axes[0].hist(df["log_time"], bins=80, edgecolor="white", alpha=0.8)
    _axes[0].set_xlabel("log(time_seconds)")
    _axes[0].set_ylabel("Count")
    _axes[0].set_title("Distribution of Log Times")

    # Time vs pieces (log-log)
    _axes[1].scatter(df["puzzle_pieces"], df["time_seconds"], alpha=0.05, s=3)
    _axes[1].set_xscale("log")
    _axes[1].set_yscale("log")
    _axes[1].set_xlabel("Pieces")
    _axes[1].set_ylabel("Time (seconds)")
    _axes[1].set_title("Time vs Pieces (log-log)")

    # Piece count distribution
    _piece_counts = df["puzzle_pieces"].value_counts().sort_index()
    _axes[2].bar(_piece_counts.index.astype(str), _piece_counts.values, edgecolor="white")
    _axes[2].set_xlabel("Piece Count")
    _axes[2].set_ylabel("Number of Records")
    _axes[2].set_title("Records by Piece Count")
    _axes[2].tick_params(axis="x", rotation=45)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(df, plt):
    # Puzzler frequency distribution
    _puzzler_counts = df.groupby("competitor_name").size()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].hist(_puzzler_counts, bins=50, edgecolor="white", alpha=0.8)
    _axes[0].set_xlabel("Number of Records")
    _axes[0].set_ylabel("Number of Puzzlers")
    _axes[0].set_title("Puzzler Observation Frequency")
    _axes[0].set_yscale("log")

    # Puzzle frequency distribution
    _puzzle_counts = df.groupby("puzzle_id").size()
    _axes[1].hist(_puzzle_counts, bins=50, edgecolor="white", alpha=0.8)
    _axes[1].set_xlabel("Number of Records")
    _axes[1].set_ylabel("Number of Puzzles")
    _axes[1].set_title("Puzzle Observation Frequency")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(df, plt):
    # Log time distribution by piece count (top categories)
    _top_pieces = df["puzzle_pieces"].value_counts().head(6).index.sort_values()
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 8))

    for _ax, _n_pieces in zip(_axes.flat, _top_pieces):
        _subset = df[df["puzzle_pieces"] == _n_pieces]
        _ax.hist(_subset["log_time"], bins=40, edgecolor="white", alpha=0.8)
        _ax.set_title(f"{_n_pieces} pieces (n={len(_subset):,})")
        _ax.set_xlabel("log(time)")
        _ax.set_ylabel("Count")

    _fig.suptitle("Log-Time Distributions by Piece Count", y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(df, mo):
    # Sparsity analysis
    _puzzler_counts = df.groupby("competitor_name").size()
    mo.md(f"""
    ## Sparsity Analysis

    - Puzzlers with 1 observation: {(_puzzler_counts == 1).sum()} ({(_puzzler_counts == 1).mean():.0%})
    - Puzzlers with 1-2 observations: {(_puzzler_counts <= 2).sum()} ({(_puzzler_counts <= 2).mean():.0%})
    - Puzzlers with 5+ observations: {(_puzzler_counts >= 5).sum()} ({(_puzzler_counts >= 5).mean():.0%})
    - Puzzlers with 10+ observations: {(_puzzler_counts >= 10).sum()} ({(_puzzler_counts >= 10).mean():.0%})
    - Max observations for a puzzler: {_puzzler_counts.max()}
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
