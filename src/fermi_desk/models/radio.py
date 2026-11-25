"""Breakthrough Listen narrowband forward model (flat vs. distance-scaled EIRP).

Sensitivity follows Enriquez et al. 2017 (arXiv:1709.03491) Sec. 5.1. For a
transmitter narrower than one spectrometer channel, the minimum detectable
*flux* (W m^-2) is

    F_min = (S/N)_min * SEFD * sqrt(delta_nu / (n_pol * tau_obs)),

equivalently their S_min = 17 Jy quoted for a delta_nu_t = 1 Hz transmitter. The
minimum detectable EIRP at distance d is EIRP_min = 4 pi d^2 F_min.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import astropy.units as u
import numpy as np
from scipy.special import ndtr

from fermi_desk.errors import UnitValidationError

RadioMode = Literal["flat_probability", "distance_scaled_eirp"]
RADIO_MODES: tuple[str, ...] = ("flat_probability", "distance_scaled_eirp")


@dataclass(frozen=True)
class Radiometer:
    sefd: u.Quantity
    snr_threshold: float
    channel_bandwidth: u.Quantity
    integration_time: u.Quantity
    n_pol: int

    def __post_init__(self) -> None:
        for name, unit in (("sefd", u.Jy), ("channel_bandwidth", u.Hz), ("integration_time", u.s)):
            q = getattr(self, name)
            if not isinstance(q, u.Quantity) or not q.unit.is_equivalent(unit):
                raise UnitValidationError(f"Radiometer.{name} must be convertible to {unit}")


def min_detectable_flux(rad: Radiometer) -> u.Quantity:
    """Narrowband minimum detectable flux in W m^-2."""
    root = np.sqrt(rad.channel_bandwidth / (rad.n_pol * rad.integration_time))
    return (rad.snr_threshold * rad.sefd * root).to(u.W / u.m**2)


def min_detectable_eirp(distance: u.Quantity, f_min: u.Quantity) -> u.Quantity:
    """EIRP_min = 4 pi d^2 F_min, in watts."""
    if not isinstance(distance, u.Quantity) or not distance.unit.is_equivalent(u.pc):
        raise UnitValidationError("distance must be a length Quantity")
    return (4.0 * np.pi * distance**2 * f_min).to(u.W)


@dataclass(frozen=True)
class LogNormalEIRP:
    """Transmitter luminosity function: log10(EIRP / W) ~ Normal(log10(median), sigma_dex)."""

    median: u.Quantity
    sigma_dex: float

    def __post_init__(self) -> None:
        if not isinstance(self.median, u.Quantity) or not self.median.unit.is_equivalent(u.W):
            raise UnitValidationError("LogNormalEIRP.median must be a power Quantity")
        if self.sigma_dex <= 0:
            raise ValueError("sigma_dex must be positive")

    def survival(self, eirp_min: u.Quantity) -> np.ndarray:
        """P(EIRP >= eirp_min)."""
        x = np.log10(eirp_min.to_value(u.W)) - np.log10(self.median.to_value(u.W))
        return ndtr(-x / self.sigma_dex)


def detection_probabilities(
    mode: str,
    p_radio: float,
    distances: u.Quantity,
    radiometer: Radiometer | None = None,
    luminosity_function: LogNormalEIRP | None = None,
) -> np.ndarray:
    """Per-target probability that a *reached* system is detected.

    ``flat_probability``: p_i = p_radio.
    ``distance_scaled_eirp``: p_i = p_radio * P(EIRP >= 4 pi d_i^2 F_min).
    """
    if not 0.0 <= p_radio <= 1.0:
        raise ValueError("p_radio must be in [0, 1]")
    if not isinstance(distances, u.Quantity) or not distances.unit.is_equivalent(u.pc):
        raise UnitValidationError("distances must be a length Quantity")
    n = int(np.size(distances))
    if mode == "flat_probability":
        return np.full(n, float(p_radio))
    if mode == "distance_scaled_eirp":
        if radiometer is None or luminosity_function is None:
            raise ValueError("distance_scaled_eirp needs a radiometer and luminosity function")
        eirp = min_detectable_eirp(np.atleast_1d(distances), min_detectable_flux(radiometer))
        return float(p_radio) * luminosity_function.survival(eirp)
    raise ValueError(f"radio mode must be one of {RADIO_MODES}, got {mode!r}")


@dataclass(frozen=True)
class AnalyticBounds:
    """Bounds on P(no detections) from marginal reach probabilities q_i.

    * ``independent``: prod(1 - p_i q_i). Reach indicators of a Poisson-Boolean
      model are increasing in the point configuration, so by Harris-FKG this is a
      LOWER bound on the exact value under the model's positive correlation.
    * ``markov``: max(0, 1 - sum p_i q_i), the Frechet/Markov lower bound valid
      under ANY dependence.
    * ``frechet_upper``: min_i (1 - p_i q_i), valid under any dependence.
    """

    independent: float
    markov: float
    frechet_upper: float
    expected_detections: float


def zero_detection_bounds(p_detect: np.ndarray, q_reach: np.ndarray) -> AnalyticBounds:
    pq = np.asarray(p_detect, float) * np.asarray(q_reach, float)
    exp_det = float(pq.sum())
    return AnalyticBounds(
        independent=float(np.exp(np.sum(np.log1p(-pq)))) if pq.size else 1.0,
        markov=float(max(0.0, 1.0 - exp_det)),
        frechet_upper=float(np.min(1.0 - pq)) if pq.size else 1.0,
        expected_detections=exp_det,
    )
