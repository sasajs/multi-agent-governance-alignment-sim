# run_experiment.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from governance_simulation import run_simulation, summarize_run
from regimes import get_regimes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run governance simulation experiments across regimes.")
    p.add_argument("--regime", type=str, default="all", help="Regime name in regimes.py or 'all'")
    p.add_argument("--replications", type=int, default=10, help="Number of independent runs per regime")
    p.add_argument("--seed", type=int, default=42, help="Base random seed (each run uses seed+run_id)")
    p.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    regimes = get_regimes()

    if args.regime != "all":
        if args.regime not in regimes:
            raise ValueError(f"Unknown regime '{args.regime}'. Available: {list(regimes.keys())}")
        selected = {args.regime: regimes[args.regime]}
    else:
        selected = regimes

    all_raw: List[pd.DataFrame] = []
    all_summaries: List[Dict] = []

    run_counter = 0
    for name, cfg in selected.items():
        for r in range(args.replications):
            run_id = run_counter
            seed = args.seed + run_id
            df = run_simulation(cfg, run_id=run_id, seed=seed)
            all_raw.append(df)
            all_summaries.append(summarize_run(df))
            run_counter += 1

    raw_df = pd.concat(all_raw, ignore_index=True)
    sum_df = pd.DataFrame(all_summaries)

    raw_path = outdir / "simulation_raw.csv"
    sum_path = outdir / "simulation_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    sum_df.to_csv(sum_path, index=False)

    print(f"Saved raw agent-level data to: {raw_path}")
    print(f"Saved run-level summary to: {sum_path}")
    print(f"Rows (raw): {len(raw_df):,} | Runs (summary): {len(sum_df):,}")


if __name__ == "__main__":
    main()