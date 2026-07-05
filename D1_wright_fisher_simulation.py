"""
Wright-Fisher genetic drift simulation
Converted from the original R script (WF.r) to executable Python.

Run:
    python wright_fisher_simulation.py

Outputs:
    - wright_fisher_results.csv
    - wright_fisher_simulation.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_wright_fisher(
    sizes: list[int] | tuple[int, ...] = (50, 100, 1000, 5000),
    starting_ps: list[float] | tuple[float, ...] = (0.01, 0.1, 0.5, 0.8),
    n_gen: int = 100,
    n_reps: int = 50,
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate neutral allele-frequency drift under a diploid Wright-Fisher model.

    Parameters
    ----------
    sizes:
        Diploid effective population sizes (N). The number of allele copies is 2N.
    starting_ps:
        Initial allele frequencies.
    n_gen:
        Number of generations to simulate.
    n_reps:
        Number of replicate populations for each parameter combination.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns: replicate, N, gen, p0, p.
    """
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []

    for N in sizes:
        for p0 in starting_ps:
            # One allele-frequency trajectory per replicate.
            p = np.full(n_reps, p0, dtype=float)

            # Record generation 0.
            rows.append(
                pd.DataFrame(
                    {
                        "replicate": np.arange(1, n_reps + 1),
                        "N": N,
                        "gen": 0,
                        "p0": p0,
                        "p": p,
                    }
                )
            )

            # Wright-Fisher update: X_{t+1} ~ Binomial(2N, p_t).
            for gen in range(1, n_gen + 1):
                allele_counts = rng.binomial(2 * N, p)
                p = allele_counts / (2 * N)
                rows.append(
                    pd.DataFrame(
                        {
                            "replicate": np.arange(1, n_reps + 1),
                            "N": N,
                            "gen": gen,
                            "p0": p0,
                            "p": p,
                        }
                    )
                )

    return pd.concat(rows, ignore_index=True)


def plot_wright_fisher(df: pd.DataFrame, output_png: str = "wright_fisher_simulation.png") -> None:
    """Create a faceted trajectory plot similar to the original ggplot output."""
    sizes = sorted(df["N"].unique())
    starting_ps = sorted(df["p0"].unique())

    fig, axes = plt.subplots(
        nrows=len(sizes),
        ncols=len(starting_ps),
        figsize=(14, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for i, N in enumerate(sizes):
        for j, p0 in enumerate(starting_ps):
            ax = axes[i, j]
            sub = df[(df["N"] == N) & (df["p0"] == p0)]
            for _, traj in sub.groupby("replicate"):
                ax.plot(traj["gen"], traj["p"], alpha=0.45, linewidth=0.8)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"N={N}, p0={p0}")
            if i == len(sizes) - 1:
                ax.set_xlabel("Generation")
            if j == 0:
                ax.set_ylabel("Allele frequency")

    fig.suptitle("Wright-Fisher genetic drift simulation", fontsize=16)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    df = simulate_wright_fisher()
    df.to_csv("wright_fisher_results.csv", index=False)
    plot_wright_fisher(df)
    print("Done. Wrote wright_fisher_results.csv and wright_fisher_simulation.png")


if __name__ == "__main__":
    main()
