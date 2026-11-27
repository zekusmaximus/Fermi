"""Expected Information Gain (bits) and cost-adjusted utility for queue jobs.

EIG is the mutual information I(H; Y) between a stated discrete prior over
hypotheses (or calibrations) H and a job's discrete outcome Y:

    I(H; Y) = sum_h w_h sum_y p(y|h) [log2 p(y|h) - log2 p(y)],  p(y) = sum_h w_h p(y|h).

It is exact for finite outcome spaces. Count outcomes are evaluated exactly on a
common support truncated where every hypothesis's tail mass is < ``tail``; the
tail is lumped into a single final outcome, so each predictive distribution sums
to one and the result is the exact MI of that (slightly coarsened) variable.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom


def entropy_bits(prior: Mapping[str, float]) -> float:
    w = np.array([v for v in prior.values() if v > 0], float)
    return float(-np.sum(w * np.log2(w)))


def _validate(prior: Mapping[str, float], likelihoods: Mapping[str, np.ndarray]) -> None:
    if set(prior) != set(likelihoods):
        raise ValueError("prior and likelihood keys differ")
    if not np.isclose(sum(prior.values()), 1.0, atol=1e-12) or min(prior.values()) < 0:
        raise ValueError("prior must be a probability vector")
    sizes = {np.asarray(v).size for v in likelihoods.values()}
    if len(sizes) != 1:
        raise ValueError("all predictive distributions must share one outcome space")
    for h, v in likelihoods.items():
        v = np.asarray(v, float)
        if np.any(v < 0) or not np.isclose(v.sum(), 1.0, atol=1e-9):
            raise ValueError(f"predictive distribution for {h} must sum to 1")


def mutual_information_bits(
    prior: Mapping[str, float], likelihoods: Mapping[str, np.ndarray]
) -> float:
    _validate(prior, likelihoods)
    keys = sorted(prior)
    w = np.array([prior[k] for k in keys])
    py_h = np.vstack([np.asarray(likelihoods[k], float) for k in keys])
    p_y = w @ py_h
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(py_h > 0, py_h * (np.log2(py_h) - np.log2(p_y)[None, :]), 0.0)
    mi = float(np.sum(w[:, None] * terms))
    return max(mi, 0.0)


def binary_outcome(p_zero: float) -> np.ndarray:
    """Outcome space {no detection, >= 1 detection}."""
    return np.array([p_zero, 1.0 - p_zero])


def binomial_count_outcomes(
    n: int, rates: Mapping[str, float], tail: float = 1e-15
) -> dict[str, np.ndarray]:
    """Binomial(n, rate_h) pmfs on a common support 0..K-1 plus one lumped tail outcome."""
    k_max = max(int(binom.isf(tail, n, r)) + 1 for r in rates.values())
    k = np.arange(k_max)
    out = {}
    for h, r in rates.items():
        pmf = binom.pmf(k, n, r)
        out[h] = np.append(pmf, max(0.0, 1.0 - pmf.sum()))
    return out


def posterior(prior: Mapping[str, float], likelihood: Mapping[str, float]) -> dict[str, float]:
    """Posterior over hypotheses after one observed outcome with ``likelihood[h] = p(y | h)``."""
    if set(prior) != set(likelihood):
        raise ValueError("prior and likelihood keys differ")
    joint = {h: prior[h] * float(likelihood[h]) for h in prior}
    z = sum(joint.values())
    if not z > 0:
        raise ValueError("the observed outcome has zero probability under every hypothesis")
    return {h: v / z for h, v in joint.items()}


def realized_information_bits(prior: Mapping[str, float], likelihood: Mapping[str, float]) -> float:
    """Information actually gained from ONE observed outcome: KL(posterior || prior) in bits.

    The expected value of this quantity over the outcome distribution is the EIG
    (``mutual_information_bits``); a single realisation can be smaller or larger than it.
    """
    post = posterior(prior, likelihood)
    return float(sum(p * (math.log2(p) - math.log2(prior[h])) for h, p in post.items() if p > 0))


def binary_eig_range(
    prior: Mapping[str, float], p_zero_bounds: Mapping[str, tuple[float, float]]
) -> tuple[float, float]:
    """(min, max) of the binary-outcome EIG when each hypothesis's P(no detection) may lie
    anywhere in its own interval ``[lower, upper]``.

    Used with the Harris-FKG lower and Frechet upper bounds on P(no detection): the exact
    value under the model's positive spatial correlation lies between them, so the EIG lies
    in the returned range.

    The EIG is convex in the likelihoods and zero exactly where all hypotheses predict the
    outcome equally. Its maximum over the box of intervals is therefore attained at a
    corner (exact). Its minimum is 0 when the intervals share a common point; otherwise,
    for two hypotheses, it is attained at the corner where the two intervals are closest
    (exact). For more than two hypotheses the corner minimum is an upper bound on the true
    minimum.
    """
    keys = sorted(prior)
    if set(keys) != set(p_zero_bounds):
        raise ValueError("prior and bound keys differ")
    if len(keys) > 16:
        raise ValueError("too many hypotheses for a corner scan")
    bounds = []
    for k in keys:
        lo, hi = (float(x) for x in p_zero_bounds[k])
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"bounds for {k} must satisfy 0 <= lower <= upper <= 1")
        bounds.append((lo, hi))
    corners = [
        mutual_information_bits(
            prior, {k: binary_outcome(p) for k, p in zip(keys, combo, strict=True)}
        )
        for combo in itertools.product(*bounds)
    ]
    overlap = max(lo for lo, _ in bounds) <= min(hi for _, hi in bounds)
    return (0.0 if overlap else float(min(corners))), float(max(corners))


def utility_bits_per_hour(bits: float, cpu_seconds: float, reviewer_minutes: float) -> float:
    hours = cpu_seconds / 3600.0 + reviewer_minutes / 60.0
    return bits / hours if hours > 0 else 0.0


@dataclass(frozen=True)
class EIGResult:
    bits: float
    prior: dict
    basis: str
    details: dict
