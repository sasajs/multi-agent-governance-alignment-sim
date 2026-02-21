# regimes.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RegimeConfig:
    # Identity / documentation
    name: str
    label: str
    description: str
    theme: str  # e.g., "Baseline", "Dynamic Beliefs", "Social Influence"

    # Core simulation settings
    n_agents: int = 10
    n_steps: int = 100

    # Initialization
    alignment_enforced: bool = True
    alignment_init_range_enforced: Tuple[float, float] = (0.4, 0.9)
    alignment_init_range_unenforced: Tuple[float, float] = (0.0, 0.3)

    # Rewards / penalties
    reward_aligned_range: Tuple[int, int] = (5, 15)
    reward_unaligned_range: Tuple[int, int] = (10, 20)
    penalty_range: Tuple[int, int] = (10, 30)

    # Auditing
    audit_schedule: Tuple[int, ...] = (25, 50, 75)

    # Optional mechanisms (toggle features rather than separate files)
    memory_enabled: bool = False
    memory_window: int = 5
    memory_threshold: int = 3  # "mostly aligned" threshold in last window
    memory_alignment_bonus: float = 0.1

    belief_enabled: bool = False
    belief_init_noise: float = 0.1
    belief_feedback_strength: float = 0.1

    social_influence_enabled: bool = False
    social_influence_factor: float = 0.0  # 0..1 influence magnitude

    adversarial_enabled: bool = False
    adversarial_factor: float = 0.0  # probability of sabotage / miscoordination

    randomness_enabled: bool = False
    noise_action_flip_prob: float = 0.0  # probability of flipping action


def to_param_dict(cfg: RegimeConfig) -> Dict:
    """Convenience for writing configs into output files."""
    return asdict(cfg)


def get_regimes() -> Dict[str, RegimeConfig]:
    """
    Starter set:
      - baseline_enforced   ~ your 'alignment incentives' baseline
      - baseline_control    ~ your no-incentives control
      - memory_coordination ~ resembles V14 (memory-based reinforcement)
      - dynamic_beliefs     ~ resembles V16 (belief + belief updates)
    Add the remaining versions by adding more RegimeConfig entries.
    """

    regimes: Dict[str, RegimeConfig] = {}

    regimes["baseline_enforced"] = RegimeConfig(
        name="baseline_enforced",
        label="V_Baseline_Enforced",
        theme="Baseline",
        description="Baseline with alignment incentives + scheduled audits.",
        alignment_enforced=True,
        memory_enabled=False,
        belief_enabled=False,
    )

    regimes["baseline_control"] = RegimeConfig(
        name="baseline_control",
        label="V_Baseline_Control",
        theme="Baseline",
        description="Control without alignment incentives and no audits.",
        alignment_enforced=False,
        audit_schedule=(),
        memory_enabled=False,
        belief_enabled=False,
    )

    regimes["memory_coordination"] = RegimeConfig(
        name="memory_coordination",
        label="V_Memory_Coordination",
        theme="Multi-Agent Cooperation",
        description="Memory-based reinforcement: recent aligned actions increase likelihood of aligned action.",
        alignment_enforced=True,
        memory_enabled=True,
        memory_window=5,
        memory_threshold=3,
        memory_alignment_bonus=0.1,
    )

    regimes["dynamic_beliefs"] = RegimeConfig(
        name="dynamic_beliefs",
        label="V_Dynamic_Beliefs",
        theme="Dynamic Beliefs",
        description="Agents act on a belief (perceived alignment) that evolves via stochastic feedback.",
        alignment_enforced=True,
        belief_enabled=True,
        belief_init_noise=0.1,
        belief_feedback_strength=0.1,
    )

    return regimes