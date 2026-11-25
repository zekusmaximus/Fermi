"""Local Solar System null constraint.

If a front reached the Solar System, existing searches would have found evidence
with probability ``p_ss``. Without anthropic conditioning the null likelihood is

    L_ss = 1 - P(Earth reached) * p_ss.

When the engine conditions on Earth being unreached, the Solar System is
unreached by construction, so this channel carries no further information and
contributes exactly 1.0 (it must never re-apply the Earth-unreached factor).
"""

from __future__ import annotations


def null_likelihood(
    p_earth_reached: float, p_artifact_detect: float, *, conditioned_on_earth_unreached: bool
) -> float:
    if not 0.0 <= p_earth_reached <= 1.0 or not 0.0 <= p_artifact_detect <= 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    if conditioned_on_earth_unreached:
        return 1.0
    return 1.0 - p_earth_reached * p_artifact_detect
