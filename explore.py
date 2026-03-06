import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import matplotlib
    return


@app.cell
def _():
    import pandas as pd

    df = pd.read_csv("data/processed/combined_results.csv")
    df
    return (df,)


@app.cell
def _(df):
    df[df.division=='solo'].plot.scatter(x='puzzle_pieces', y='time_seconds')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
