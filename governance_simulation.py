# governance_simulation.py
from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from regimes import RegimeConfig, to_param_dict


class AIAgent:
    def __init__(self, agent_id: int, cfg: RegimeConfig):
        self.id = agent_id
        self.cfg = cfg

        # Alignment initialization
        if cfg.alignment_enforced:
            lo, hi = cfg.alignment_init_range_enforced
        else:
            lo, hi = cfg.alignment_init_range_unenforced
        self.alignment = random.uniform(lo, hi)

        # Belief initialization (optional)
        self.belief = None
        if cfg.belief_enabled:
            self.belief = self.alignment + random.uniform(-cfg.belief_init_noise, cfg.belief_init_noise)
            self.belief = max(0.0, min(1.0, self.belief))

        self.reward = 0
        self.memory: List[int] = []  # 1 if aligned action, 0 otherwise

    def update_belief(self):
        if not self.cfg.belief_enabled or self.belief is None:
            return
        delta = random.uniform(-self.cfg.belief_feedback_strength, self.cfg.belief_feedback_strength)
        self.belief = max(0.0, min(1.0, self.belief + delta))

    def decide_aligned_action(self) -> bool:
        """
        Core decision rule.
        - If belief enabled: act using belief probability
        - Else: act using alignment probability
        - If memory enabled: boost probability when recent coordination is high
        """
        p = self.belief if (self.cfg.belief_enabled and self.belief is not None) else self.alignment

        if self.cfg.memory_enabled and len(self.memory) >= self.cfg.memory_window:
            recent = self.memory[-self.cfg.memory_window :]
            if sum(recent) > self.cfg.memory_threshold:
                p = min(1.0, p + self.cfg.memory_alignment_bonus)

        # Optional "randomness/noise" that flips actions
        aligned = (random.random() < p)
        if self.cfg.randomness_enabled and self.cfg.noise_action_flip_prob > 0:
            if random.random() < self.cfg.noise_action_flip_prob:
                aligned = not aligned

        # Optional adversarial mechanism: sabotage reduces alignment probability
        if self.cfg.adversarial_enabled and self.cfg.adversarial_factor > 0:
            if random.random() < self.cfg.adversarial_factor:
                aligned = False

        return aligned

    def apply_social_influence(self, population_avg_alignment: float):
        """
        Simple social influence: nudge the agent's alignment slightly toward population mean.
        (This is a placeholder mechanism; you can refine it for your specific V* definitions.)
        """
        if not self.cfg.social_influence_enabled:
            return
        alpha = self.cfg.social_influence_factor
        self.alignment = (1 - alpha) * self.alignment + alpha * population_avg_alignment
        self.alignment = max(0.0, min(1.0, self.alignment))
        if self.cfg.belief_enabled and self.belief is not None:
            self.belief = max(0.0, min(1.0, self.belief))  # keep in bounds

    def receive_reward(self, aligned: bool):
        if aligned:
            lo, hi = self.cfg.reward_aligned_range
            r = random.randint(lo, hi)
            self.reward += r
            self.memory.append(1)
        else:
            lo, hi = self.cfg.reward_unaligned_range
            r = random.randint(lo, hi)
            self.reward += r
            self.memory.append(0)

    def audit(self):
        # Penalty if misaligned relative to probability proxy
        if random.random() > self.alignment:
            lo, hi = self.cfg.penalty_range
            penalty = random.randint(lo, hi)
            self.reward -= penalty


def run_simulation(cfg: RegimeConfig, run_id: int, seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)

    agents = [AIAgent(i, cfg) for i in range(cfg.n_agents)]

    for step in range(cfg.n_steps):
        # Population-level influence (optional)
        if cfg.social_influence_enabled:
            pop_avg = sum(a.alignment for a in agents) / len(agents)
            for a in agents:
                a.apply_social_influence(pop_avg)

        for agent in agents:
            agent.update_belief()
            aligned = agent.decide_aligned_action()
            agent.receive_reward(aligned)

            if step in cfg.audit_schedule:
                agent.audit()

    # Record per-agent end-state
    rows: List[Dict] = []
    for a in agents:
        rows.append(
            {
                "Regime": cfg.name,
                "Label": cfg.label,
                "Theme": cfg.theme,
                "run": run_id,
                "agent_id": a.id,
                "alignment": round(a.alignment, 5),
                "belief": None if a.belief is None else round(a.belief, 5),
                "reward": a.reward,
            }
        )

    df = pd.DataFrame(rows)

    # Add regime parameters for transparency (repeated per row for convenience; can drop later)
    params = to_param_dict(cfg)
    for k, v in params.items():
        if k in df.columns:
            continue
        df[k] = str(v) if isinstance(v, (tuple, list, dict)) else v

    return df


def summarize_run(df: pd.DataFrame) -> Dict:
    """Run-level summary used by analysis.py (Avg/Std across agents)."""
    return {
        "Regime": df["Regime"].iloc[0],
        "Label": df["Label"].iloc[0],
        "Theme": df["Theme"].iloc[0],
        "run": int(df["run"].iloc[0]),
        "Avg_Reward": float(df["reward"].mean()),
        "Std_Reward": float(df["reward"].std(ddof=1)),
        "Avg_Alignment": float(df["alignment"].mean()),
        "Std_Alignment": float(df["alignment"].std(ddof=1)),
        "N_agents": int(df.shape[0]),
    }