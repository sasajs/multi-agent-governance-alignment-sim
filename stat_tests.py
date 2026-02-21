import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from scipy.stats import spearmanr, levene

# -------------------------
# 1. LOAD DATA
# -------------------------

base = pd.read_csv("outputs/simulation_summary.csv")
rep150 = pd.read_csv("outputs_rep150/simulation_summary.csv")
seed42 = pd.read_csv("outputs_seed42/simulation_summary.csv")
seed999 = pd.read_csv("outputs_seed999/simulation_summary.csv")

# -------------------------
# 2. IDENTIFY COLUMNS
# -------------------------

print("Columns:", base.columns)

VERSION_COL = "Label"
REWARD_COL  = "Avg_Reward"
ALIGN_COL   = "Avg_Alignment"

# -------------------------
# 3. KRUSKAL–WALLIS (REWARD)
# -------------------------

groups = [g[REWARD_COL].values for _, g in base.groupby(VERSION_COL)]
H, p = stats.kruskal(*groups)

k = base[VERSION_COL].nunique()
n = len(base)
epsilon_sq = (H - k + 1) / (n - k)

print("\nKruskal–Wallis (Reward)")
print("H =", H)
print("p =", p)
print("Effect size ε² =", epsilon_sq)

# -------------------------
# 4. DUNN POST-HOC
# -------------------------

dunn = sp.posthoc_dunn(base, val_col=REWARD_COL,
                       group_col=VERSION_COL,
                       p_adjust="holm")

dunn.to_csv("dunn_reward_holm.csv")
print("\nDunn post-hoc saved to dunn_reward_holm.csv")

# -------------------------
# 5. RANK STABILITY (100 vs 150)
# -------------------------

mean_base = base.groupby(VERSION_COL)[REWARD_COL].mean()
mean_150 = rep150.groupby(VERSION_COL)[REWARD_COL].mean()

rho, pval = spearmanr(mean_base, mean_150)

print("\nRank Stability (100 vs 150 reps)")
print("Spearman rho =", rho)
print("p =", pval)

# -------------------------
# 6. SEED STABILITY
# -------------------------

mean_42 = seed42.groupby(VERSION_COL)[REWARD_COL].mean()
mean_999 = seed999.groupby(VERSION_COL)[REWARD_COL].mean()

rho_seed, p_seed = spearmanr(mean_42, mean_999)

print("\nSeed Stability (42 vs 999)")
print("Spearman rho =", rho_seed)
print("p =", p_seed)

# -------------------------
# 7. VARIANCE TEST (OPTIONAL)
# -------------------------

reg_groups = [g[REWARD_COL].values for _, g in base.groupby(VERSION_COL)]
W, p_var = levene(*reg_groups, center="median")

print("\nBrown–Forsythe Variance Test")
print("W =", W)
print("p =", p_var)