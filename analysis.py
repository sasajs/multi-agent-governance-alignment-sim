# analysis.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze simulation results from run_experiment.py outputs.")
    p.add_argument("--summary", type=str, default="outputs/simulation_summary.csv", help="Run-level summary CSV")
    p.add_argument("--outdir", type=str, default="outputs/figures", help="Where to save figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.summary)

    # ANOVA (you can swap for Kruskal–Wallis later, but ANOVA is a quick screening)
    anova_reward = smf.ols("Avg_Reward ~ C(Regime)", data=data).fit()
    anova_alignment = smf.ols("Avg_Alignment ~ C(Regime)", data=data).fit()

    print("\nANOVA on Reward by Regime")
    print(sm.stats.anova_lm(anova_reward, typ=2))

    print("\nANOVA on Alignment by Regime")
    print(sm.stats.anova_lm(anova_alignment, typ=2))

    print("\nCorrelation (Reward vs Alignment)")
    print(data[["Avg_Reward", "Avg_Alignment"]].corr())

    lm = smf.ols("Avg_Reward ~ Avg_Alignment", data=data).fit()
    print("\nLinear Regression: Reward ~ Alignment")
    print(lm.summary())

    # Plots
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="Avg_Alignment", y="Avg_Reward", hue="Regime")
    plt.title("Reward vs. Alignment by Regime")
    plt.tight_layout()
    plt.savefig(outdir / "scatter_reward_alignment_by_regime.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x="Regime", y="Avg_Reward")
    plt.xticks(rotation=45, ha="right")
    plt.title("Reward Distribution by Regime (Run-Level)")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_reward_by_regime.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x="Regime", y="Avg_Alignment")
    plt.xticks(rotation=45, ha="right")
    plt.title("Alignment Distribution by Regime (Run-Level)")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_alignment_by_regime.png", dpi=200)
    plt.close()

    print(f"\nSaved figures to: {outdir}")


if __name__ == "__main__":
    main()