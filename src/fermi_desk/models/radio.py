"""Breakthrough Listen narrowband forward model (flat vs. distance-scaled EIRP).

Sensitivity follows Enriquez et al. 2017 (arXiv:1709.03491) Sec. 5.1. For a
transmitter narrower than one spectrometer channel, the minimum detectable
*flux* (W m^-2) is

    F_min = (S/N)_min * SEFD * sqrt(delta_nu / (n_pol * tau_obs)),

equivalently their S_min = 17 Jy quoted for a delta_nu_t = 1 Hz transmitter. The
minimum detectable EIRP at distance d is EIRP_min = 4 pi d^2 F_min.

Off-axis stars (v0.2.1). A star at angular offset theta from the beam centre is seen
through the Gaussian primary beam B(theta) = exp(-4 ln2 (theta / FWHM)^2), so

    EIRP_min(d, theta) = 4 pi d^2 F_min / B(theta),

with B(0) = 1 (the on-axis formula, bit for bit) and B(FWHM / 2) = 1/2. Background-field
stars inside the primary beams are carried as compact shells (``BackgroundShells``):
per-pointing counts on a (distance, EIRP_min) grid. This module never touches expansion
physics: reach probabilities are passed in by ``inference/engine.py``.

Spatial-correlation assumption for merged primary + background evaluations
(``SHELL_CORRELATION_ASSUMPTION``): the primaries keep the exact correlated Monte Carlo;
the shells use the independent product of marginal conditional-reach probabilities.
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
# GBT L-band primary-beam FWHM (Wlodarczyk-Sroka, Garrett & Siemion 2020, Sec. 2).
GBT_LBAND_BEAM_FWHM = 8.4 * u.arcmin
SHELL_CORRELATION_ASSUMPTION = (
    "primary targets: exact correlated Monte Carlo over shared settlement fronts; "
    "background shells: independent product of marginal conditional-reach probabilities, "
    "prod_cells (1 - p_c q_c)^N_c, multiplied onto the primary term. Reach indicators of a "
    "Poisson-Boolean front model are positively associated (Harris-FKG), so this product is "
    "a LOWER bound on the true P(no detection | H): it can only overstate how strongly the "
    "null disfavours a hypothesis, never understate it. The Frechet upper bound "
    "min_c (1 - p_c q_c), valid under any dependence, is reported beside it; the exact value "
    "lies between the two. The sign of the effect on a Bayes factor between two hypotheses "
    "is not determined by the bound."
)


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


def _angle(q: object, name: str) -> u.Quantity:
    if not isinstance(q, u.Quantity) or not q.unit.is_equivalent(u.arcmin):
        raise UnitValidationError(f"{name} must be an angle Quantity (e.g. astropy.units.arcmin)")
    return q


def beam_response(theta: u.Quantity, fwhm: u.Quantity = GBT_LBAND_BEAM_FWHM) -> np.ndarray:
    """Gaussian primary-beam power response B(theta) = exp(-4 ln2 (theta / FWHM)^2).

    Evaluated as ``2 ** (-4 (theta / FWHM)^2)``, the same function, so that B(0) == 1.0
    and B(FWHM / 2) == 0.5 hold exactly in floating point. ``theta`` is the angular
    offset from the beam centre (>= 0); the default FWHM is the GBT L-band 8.4 arcmin.
    """
    theta, fwhm = _angle(theta, "theta"), _angle(fwhm, "fwhm")
    ratio = np.asarray((theta / fwhm).to_value(u.dimensionless_unscaled), float)
    if not np.all(np.isfinite(ratio)) or np.any(ratio < 0) or not fwhm.value > 0:
        raise ValueError("beam_response needs finite theta >= 0 and fwhm > 0")
    return np.power(2.0, -4.0 * ratio**2)


def min_detectable_eirp_off_axis(
    distance: u.Quantity,
    theta: u.Quantity,
    f_min: u.Quantity,
    fwhm: u.Quantity = GBT_LBAND_BEAM_FWHM,
) -> u.Quantity:
    """EIRP_min(d, theta) = 4 pi d^2 F_min / B(theta), in watts.

    ``theta = 0`` reproduces :func:`min_detectable_eirp` bit for bit (division by 1.0).
    """
    return (min_detectable_eirp(distance, f_min) / beam_response(theta, fwhm)).to(u.W)


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


# ------------------------------------------------ background-field shells (v0.2.1)
@dataclass(frozen=True)
class ShellBinning:
    """Global log grids of the background shells: distance [pc] and EIRP_min [W].

    Bin ``k`` covers ``[10^(min + k dex), 10^(min + (k + 1) dex))``; its representative is
    the geometric centre. Values outside the grid raise (never clipped silently).
    """

    distance_bin_dex: float = 0.05
    distance_log10_pc_min: float = 0.0
    n_distance_bins: int = 81  # 1 pc .. 10^4.05 pc (the 10 kpc cut sits inside the last bin)
    eirp_bin_dex: float = 0.05
    eirp_log10_w_min: float = 8.0
    n_eirp_bins: int = 240  # 1e8 W .. 1e20 W

    def __post_init__(self) -> None:
        if not (self.distance_bin_dex > 0 and self.eirp_bin_dex > 0):
            raise ValueError("shell bin widths must be positive")
        if self.n_distance_bins < 1 or self.n_eirp_bins < 1:
            raise ValueError("shell grids need at least one bin")

    @staticmethod
    def _index(values: np.ndarray, lo: float, dex: float, n: int, what: str) -> np.ndarray:
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f"{what} must be finite and positive")
        idx = np.floor((np.log10(values) - lo) / dex).astype(np.int64)
        if np.any((idx < 0) | (idx >= n)):
            raise ValueError(
                f"{what} outside the shell grid [1e{lo:g}, 1e{lo + n * dex:g}) "
                f"(got {values.min():.6g} .. {values.max():.6g})"
            )
        return idx

    def distance_bin(self, distance: u.Quantity) -> np.ndarray:
        if not isinstance(distance, u.Quantity) or not distance.unit.is_equivalent(u.pc):
            raise UnitValidationError("distance must be a length Quantity")
        return self._index(
            np.atleast_1d(distance.to_value(u.pc)),
            self.distance_log10_pc_min,
            self.distance_bin_dex,
            self.n_distance_bins,
            "distance [pc]",
        )

    def eirp_bin(self, eirp_min: u.Quantity) -> np.ndarray:
        if not isinstance(eirp_min, u.Quantity) or not eirp_min.unit.is_equivalent(u.W):
            raise UnitValidationError("eirp_min must be a power Quantity")
        return self._index(
            np.atleast_1d(eirp_min.to_value(u.W)),
            self.eirp_log10_w_min,
            self.eirp_bin_dex,
            self.n_eirp_bins,
            "EIRP_min [W]",
        )

    def distance_centre(self, index: np.ndarray) -> u.Quantity:
        k = np.asarray(index, float) + 0.5
        return 10.0 ** (self.distance_log10_pc_min + k * self.distance_bin_dex) * u.pc

    def eirp_centre(self, index: np.ndarray) -> u.Quantity:
        k = np.asarray(index, float) + 0.5
        return 10.0 ** (self.eirp_log10_w_min + k * self.eirp_bin_dex) * u.W

    def as_dict(self) -> dict:
        return {
            "distance_bin_dex": self.distance_bin_dex,
            "distance_log10_pc_min": self.distance_log10_pc_min,
            "n_distance_bins": self.n_distance_bins,
            "eirp_bin_dex": self.eirp_bin_dex,
            "eirp_log10_w_min": self.eirp_log10_w_min,
            "n_eirp_bins": self.n_eirp_bins,
            "representative": "geometric_bin_centre",
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShellBinning:
        if d.get("representative") != "geometric_bin_centre":
            raise ValueError("unsupported shell representative convention")
        return cls(**{k: v for k, v in d.items() if k != "representative"})


DEFAULT_SHELL_BINNING = ShellBinning()


@dataclass(frozen=True)
class BinnedStars:
    """Per-star bins and the sorted sparse cell table of a background star list."""

    cells: np.ndarray  # (n_cells, 4) int64: pointing_index, distance_bin, eirp_min_bin, count
    distance_bin: np.ndarray  # per star
    eirp_min_bin: np.ndarray  # per star
    eirp_min: u.Quantity  # per star, W (exact, unbinned)


def bin_background_stars(
    pointing_index: np.ndarray,
    distance: u.Quantity,
    theta: u.Quantity,
    f_min: u.Quantity,
    binning: ShellBinning = DEFAULT_SHELL_BINNING,
    fwhm: u.Quantity = GBT_LBAND_BEAM_FWHM,
) -> BinnedStars:
    """Bin de-duplicated background stars into per-pointing (distance, EIRP_min) cells.

    Pure and deterministic. Cells are sorted lexicographically by
    ``(pointing_index, distance_bin, eirp_min_bin)``; only non-empty cells are returned.
    """
    pidx = np.asarray(pointing_index, np.int64).reshape(-1)
    n = pidx.size
    if np.size(distance) != n or np.size(theta) != n:
        raise ValueError("pointing_index, distance and theta must have equal length")
    if n == 0:
        empty = np.zeros(0, np.int64)
        return BinnedStars(np.zeros((0, 4), np.int64), empty, empty, np.zeros(0) * u.W)
    if np.any(pidx < 0):
        raise ValueError("pointing_index must be non-negative")
    eirp = min_detectable_eirp_off_axis(np.atleast_1d(distance), np.atleast_1d(theta), f_min, fwhm)
    d_bin, e_bin = binning.distance_bin(np.atleast_1d(distance)), binning.eirp_bin(eirp)
    keys, counts = np.unique(np.column_stack([pidx, d_bin, e_bin]), axis=0, return_counts=True)
    return BinnedStars(np.column_stack([keys, counts]).astype(np.int64), d_bin, e_bin, eirp)


@dataclass(frozen=True)
class BackgroundShells:
    """Background-field stars of one pointing group as compact shells.

    ``pointing_unit_xyz`` holds the Galactic Cartesian unit vector of every beam centre
    (x toward the Galactic centre, z toward the NGP; the same frame as ``TargetTable``).
    Each cell's stars sit at ``unit[pointing] * distance_centre`` with the cell's
    ``eirp_centre``. The stars are already de-duplicated against the primary targets
    (``union_targets_by_pointing``); this container carries no star twice by construction.
    """

    pointing_unit_xyz: np.ndarray
    pointing_index: np.ndarray
    distance_bin: np.ndarray
    eirp_min_bin: np.ndarray
    count: np.ndarray
    binning: ShellBinning
    data_tier: str
    survey_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        unit = np.asarray(self.pointing_unit_xyz, float)
        if unit.ndim != 2 or unit.shape[1] != 3:
            raise ValueError("pointing_unit_xyz must be shaped (n_pointings, 3)")
        if unit.size and not np.allclose(np.linalg.norm(unit, axis=1), 1.0, rtol=0, atol=1e-9):
            raise ValueError("pointing_unit_xyz rows must be unit vectors")
        cols = [
            np.asarray(c, np.int64).reshape(-1)
            for c in (self.pointing_index, self.distance_bin, self.eirp_min_bin, self.count)
        ]
        n = cols[0].size
        if any(c.size != n for c in cols):
            raise ValueError("shell columns must have equal length")
        limits = (len(unit), self.binning.n_distance_bins, self.binning.n_eirp_bins)
        for c, lim in zip(cols[:3], limits, strict=True):
            if n and (c.min() < 0 or c.max() >= lim):
                raise ValueError("shell cell index outside its grid")
        if n and cols[3].min() < 1:
            raise ValueError("shell counts must be >= 1")
        for name, c in zip(
            ("pointing_index", "distance_bin", "eirp_min_bin", "count"), cols, strict=True
        ):
            object.__setattr__(self, name, c)
        object.__setattr__(self, "pointing_unit_xyz", unit)

    @property
    def n_cells(self) -> int:
        return int(self.count.size)

    @property
    def n_stars(self) -> int:
        return int(self.count.sum())

    def cell_distance(self) -> u.Quantity:
        return self.binning.distance_centre(self.distance_bin)

    def cell_eirp_min(self) -> u.Quantity:
        return self.binning.eirp_centre(self.eirp_min_bin)

    def reach_nodes(self) -> tuple[u.Quantity, np.ndarray]:
        """Distinct ``(pointing, distance_bin)`` positions and each cell's node index.

        Reach probabilities depend on position only, so the engine evaluates the expansion
        kernel once per node and shares it across that node's EIRP_min cells.
        """
        nd = self.binning.n_distance_bins
        nodes, inverse = np.unique(
            self.pointing_index * nd + self.distance_bin, return_inverse=True
        )
        p_idx, d_idx = np.divmod(nodes, nd)
        dist = self.binning.distance_centre(d_idx).to_value(u.pc)
        return self.pointing_unit_xyz[p_idx] * dist[:, None] * u.pc, inverse.reshape(-1)


def shell_detection_probabilities(
    mode: str,
    p_radio: float,
    shells: BackgroundShells,
    luminosity_function: LogNormalEIRP | None = None,
) -> np.ndarray:
    """Per-cell probability that a *reached* background star is detected.

    ``flat_probability``: p_radio for every cell (sensitivity-blind, as for the primaries).
    ``distance_scaled_eirp``: p_radio * P(EIRP >= EIRP_min(d, theta)) at the cell centre,
    where EIRP_min already contains the off-axis beam attenuation.
    """
    if not 0.0 <= p_radio <= 1.0:
        raise ValueError("p_radio must be in [0, 1]")
    if mode == "flat_probability":
        return np.full(shells.n_cells, float(p_radio))
    if mode == "distance_scaled_eirp":
        if luminosity_function is None:
            raise ValueError("distance_scaled_eirp needs a luminosity function")
        return float(p_radio) * luminosity_function.survival(shells.cell_eirp_min())
    raise ValueError(f"radio mode must be one of {RADIO_MODES}, got {mode!r}")


@dataclass(frozen=True)
class ShellZeroDetection:
    """P(no detection among the shell stars) under the independent-reach assumption.

    ``log_p_zero_independent = sum_c N_c log(1 - p_c q_c)`` is the Harris-FKG LOWER bound
    on the exact log-probability (see ``SHELL_CORRELATION_ASSUMPTION``); ``markov`` and
    ``frechet_upper`` are the dependence-free lower and upper bounds on P itself.
    """

    log_p_zero_independent: float
    markov: float
    frechet_upper: float
    expected_detections: float
    expected_reached: float


def shell_zero_detection(
    p_detect: np.ndarray, q_reach: np.ndarray, counts: np.ndarray
) -> ShellZeroDetection:
    p = np.asarray(p_detect, float).reshape(-1)
    q = np.asarray(q_reach, float).reshape(-1)
    n = np.asarray(counts, float).reshape(-1)
    if not (p.size == q.size == n.size):
        raise ValueError("p_detect, q_reach and counts must have equal length")
    if np.any((p < 0) | (p > 1)) or np.any((q < 0) | (q > 1)) or np.any(n < 0):
        raise ValueError("need probabilities in [0, 1] and non-negative counts")
    pq = p * q
    with np.errstate(divide="ignore"):
        log_p0 = float(np.sum(n * np.log1p(-pq))) if pq.size else 0.0
    exp_det = float(np.sum(n * pq))
    return ShellZeroDetection(
        log_p_zero_independent=log_p0,
        markov=float(max(0.0, 1.0 - exp_det)),
        frechet_upper=float(np.min(1.0 - pq)) if pq.size else 1.0,
        expected_detections=exp_det,
        expected_reached=float(np.sum(n * q)),
    )
