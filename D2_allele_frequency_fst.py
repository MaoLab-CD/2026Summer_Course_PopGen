"""
Class exercise: Allele frequency and FST formula

This version follows the lecture-style definition:

    Var(p1 - p2) = 2 * FST * p * (1 - p)

For a single toy SNP, we use the classroom approximation:

    FST ≈ (p1 - p2)^2 / [2 * pbar * (1 - pbar)]

where:
    pbar = (p1 + p2) / 2

The toy genotype data are chosen so that the classroom approximation stays
within a reasonable 0-1 range for all examples.

Run:
    python allele_frequency_fst_price_formula.py

Outputs:
    toy_genotypes.csv
    allele_frequencies.csv
    simple_pairwise_fst_price_formula.csv
    mean_simple_fst_price_formula.csv
    mean_simple_fst_price_formula.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_toy_data() -> pd.DataFrame:
    """Create a small toy genotype table.

    Genotype coding:
        0 = zero copies of the counted allele
        1 = one copy
        2 = two copies

    Each row is one individual. Each SNP column is one variant.
    """
    return pd.DataFrame(
        {
            "individual": [
                "CEU_1", "CEU_2", "CEU_3", "CEU_4",
                "YRI_1", "YRI_2", "YRI_3", "YRI_4",
                "CHB_1", "CHB_2", "CHB_3", "CHB_4",
            ],
            "population": [
                "CEU", "CEU", "CEU", "CEU",
                "YRI", "YRI", "YRI", "YRI",
                "CHB", "CHB", "CHB", "CHB",
            ],
            # Allele frequencies by population are deliberately moderate
            # so the lecture-style toy FST approximation does not exceed 1.
            # CEU frequencies: 0.25, 0.375, 0.25, 0.75, 0.125, 0.625
            "SNP1": [0, 0, 1, 1,   2, 2, 1, 1,   1, 1, 0, 1],
            "SNP2": [1, 1, 0, 1,   1, 1, 1, 2,   0, 0, 1, 0],
            "SNP3": [0, 0, 1, 1,   0, 0, 1, 1,   2, 2, 1, 1],
            "SNP4": [2, 1, 2, 1,   0, 0, 1, 1,   1, 1, 1, 0],
            "SNP5": [0, 1, 0, 0,   0, 0, 0, 1,   0, 0, 0, 0],
            "SNP6": [1, 1, 1, 2,   1, 1, 0, 1,   2, 2, 1, 1],
        }
    )


def calculate_allele_frequencies(data: pd.DataFrame, snp_cols: list[str]) -> pd.DataFrame:
    """Calculate allele frequency by population.

    For diploid genotypes coded as 0/1/2:
        allele frequency = mean genotype / 2
    """
    return data.groupby("population")[snp_cols].mean() / 2


def fst_price_formula(p1: float, p2: float) -> float:
    """Classroom FST approximation following the formula.

    Lecture formula:
        Var(p1 - p2) = 2 * FST * p * (1 - p)

    Toy single-SNP approximation:
        FST ≈ (p1 - p2)^2 / [2 * pbar * (1 - pbar)]

    where pbar = (p1 + p2) / 2.

    For real data, use a formal estimator such as Weir-Cockerham FST.
    """
    pbar = (p1 + p2) / 2
    denominator = 2 * pbar * (1 - pbar)
    if denominator == 0:
        return np.nan
    return (p1 - p2) ** 2 / denominator


def calculate_pairwise_fst(allele_freq: pd.DataFrame, snp_cols: list[str]) -> pd.DataFrame:
    """Calculate pairwise toy FST for each SNP and each population pair."""
    population_pairs = [("CEU", "YRI"), ("CEU", "CHB"), ("YRI", "CHB")]
    rows = []

    for pop1, pop2 in population_pairs:
        for snp in snp_cols:
            p1 = float(allele_freq.loc[pop1, snp])
            p2 = float(allele_freq.loc[pop2, snp])
            rows.append(
                {
                    "population_pair": f"{pop1}-{pop2}",
                    "SNP": snp,
                    "p1": p1,
                    "p2": p2,
                    "simple_FST_price_formula": fst_price_formula(p1, p2),
                }
            )

    return pd.DataFrame(rows)


def plot_mean_fst(mean_fst: pd.DataFrame, output_png: str = "mean_simple_fst_price_formula.png") -> None:
    """Make a simple bar plot of average pairwise toy FST."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(mean_fst["population_pair"], mean_fst["mean_simple_FST_price_formula"])
    ax.set_xlabel("Population pair")
    ax.set_ylabel("Mean FST")
    ax.set_title("Pairwise genetic differentiation: lecture-style toy formula")
    ax.set_ylim(0, max(0.05, float(mean_fst["mean_simple_FST_price_formula"].max()) * 1.25))

    for i, value in enumerate(mean_fst["mean_simple_FST_price_formula"]):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    snp_cols = ["SNP1", "SNP2", "SNP3", "SNP4", "SNP5", "SNP6"]

    data = make_toy_data()
    data.to_csv("toy_genotypes.csv", index=False)

    allele_freq = calculate_allele_frequencies(data, snp_cols)
    allele_freq.to_csv("allele_frequencies.csv")

    fst_df = calculate_pairwise_fst(allele_freq, snp_cols)
    fst_df.to_csv("simple_pairwise_fst_price_formula.csv", index=False)

    mean_fst = (
        fst_df.groupby("population_pair", as_index=False)["simple_FST_price_formula"]
        .mean()
        .rename(columns={"simple_FST_price_formula": "mean_simple_FST_price_formula"})
    )
    mean_fst.to_csv("mean_simple_fst_price_formula.csv", index=False)

    plot_mean_fst(mean_fst)

    print("\nAllele frequencies by population:")
    print(allele_freq.round(3))

    print("\nPairwise simple FST by SNP, using lecture-style formula:")
    print(fst_df.round(3))

    print("\nMean simple FST by population pair:")
    print(mean_fst.round(3))

    print("\nDone. Files written:")
    print("  toy_genotypes.csv")
    print("  allele_frequencies.csv")
    print("  simple_pairwise_fst_price_formula.csv")
    print("  mean_simple_fst_price_formula.csv")
    print("  mean_simple_fst_price_formula.png")


if __name__ == "__main__":
    main()
