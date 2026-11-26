"""Mid-IR waste-heat candidate-count model and calibration-reversal sweep.

Each of ``n`` surveyed stars hosts a genuine waste-heat signature with
probability ``q = f_ir_host * f_ir_loud``. A calibration pair
``(completeness C, false-positive rate alpha)`` gives a per-star flag probability

    pi(q) = q C + (1 - q) alpha,

and the observed candidate count ``k`` is Binomial(n, pi). The likelihood ratio
against a silent sky (q = 0) is therefore exactly 1.0 whenever q = 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


@dataclass(frozen=True)
class IRCalibration:
    id: str
    completeness: float
    false_positive_rate: float

    def __post_init__(self) -> None:
        if not 0.0 < self.completeness <= 1.0 or not 0.0 <= self.false_positive_rate < 1.0:
            raise ValueError(f"Invalid calibration {self}")


def flag_probability(signature_fraction: np.ndarray, cal: IRCalibration) -> np.ndarray:
    q = np.asarray(signature_fraction, float)
    return q * cal.completeness + (1.0 - q) * cal.false_positive_rate


def log_likelihood(
    k: int, n: int, signature_fraction: np.ndarray, cal: IRCalibration
) -> np.ndarray:
    return binom.logpmf(k, n, flag_probability(signature_fraction, cal))


def log_marginal_likelihood(
    k: int, n: int, signature_grid: np.ndarray, cal: IRCalibration
) -> float:
    """Log of the prior-averaged likelihood over an equal-weight signature-fraction grid."""
    grid = np.atleast_1d(np.asarray(signature_grid, float))
    return float(logsumexp(log_likelihood(k, n, grid, cal)) - np.log(grid.size))


def log_lr_vs_silent(k: int, n: int, signature_grid: np.ndarray, cal: IRCalibration) -> float:
    """log [ P(k | hypothesis) / P(k | q = 0) ] for one calibration."""
    if n == 0:
        return 0.0
    ref = float(log_likelihood(k, n, np.array([0.0]), cal)[0])
    return log_marginal_likelihood(k, n, signature_grid, cal) - ref


def log_marginal_likelihood_calibration_mixture(
    k: int, n: int, signature_grid: np.ndarray, cals: Sequence[IRCalibration]
) -> float:
    """Raw log P(k | H) with calibrations marginalised as an equal-weight mixture."""
    if n == 0:
        return 0.0
    vals = [log_marginal_likelihood(k, n, signature_grid, c) for c in cals]
    return float(logsumexp(vals) - np.log(len(vals)))


def log_lr_vs_silent_calibration_mixture(
    k: int, n: int, signature_grid: np.ndarray, cals: Sequence[IRCalibration]
) -> float:
    """Likelihood ratio vs. silence with calibrations marginalised as an equal-weight mixture."""
    raw = log_marginal_likelihood_calibration_mixture(k, n, signature_grid, cals)
    null = log_marginal_likelihood_calibration_mixture(k, n, np.array([0.0]), cals)
    return raw - null


@dataclass(frozen=True)
class CalibrationSweep:
    per_calibration: list[dict]
    min_lr: float
    max_lr: float
    reverses: bool
    min_calibration: str
    max_calibration: str


def sweep_calibrations(
    k: int,
    n: int,
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    cals: Sequence[IRCalibration],
) -> CalibrationSweep:
    """LR(a : b) = P(k | a) / P(k | b) at every calibration; flags a preference reversal."""
    rows = []
    for c in cals:
        la = log_marginal_likelihood(k, n, grid_a, c)
        lb = log_marginal_likelihood(k, n, grid_b, c)
        rows.append(
            {
                "calibration_id": c.id,
                "completeness": c.completeness,
                "false_positive_rate": c.false_positive_rate,
                "expected_false_positives": c.false_positive_rate * n,
                "log10_lr": float((la - lb) / np.log(10.0)),
                "lr": float(np.exp(la - lb)),
            }
        )
    lrs = np.array([r["lr"] for r in rows])
    i_min, i_max = int(np.argmin(lrs)), int(np.argmax(lrs))
    return CalibrationSweep(
        per_calibration=rows,
        min_lr=float(lrs[i_min]),
        max_lr=float(lrs[i_max]),
        reverses=bool(lrs[i_min] < 1.0 < lrs[i_max]),
        min_calibration=rows[i_min]["calibration_id"],
        max_calibration=rows[i_max]["calibration_id"],
    )
