/code
    governance_simulation.py      # Core simulation engine
    regimes.py                    # Governance vector definitions (V1–V16)
    run_experiment.py             # Executes baseline and robustness runs
    analysis.py                   # Output aggregation utilities
    stat_tests.py                 # Statistical tests (Kruskal–Wallis, Dunn, Spearman, Levene)

/outputs
    /baseline
        simulation_summary.csv    # 100 replication results
    /rep150
        simulation_summary.csv    # 150 replication robustness results
    /seed42
        simulation_summary.csv    # Fixed-seed validation (42)
    /seed999
        simulation_summary.csv    # Fixed-seed validation (999)

dunn_reward_holm.csv              # Holm-adjusted Dunn post-hoc matrix
requirements.txt                  # Python dependencies
README.md                         # Documentation
