"""v0.2.2 (Sprint 3): frozen-method blend screens on unsmoothed unWISE W3 coadds.

Pure kernels (no I/O, no network). The opt-in live fetch lives in
``scripts/run_unwise_gate.py``; a promoted summary is re-verified offline by
``registry.load_unwise_gate``. ``models/hephaistos_fits.py`` (Gate #2, AllWISE Atlas) is
imported, never edited, so the locked Gate #2 artifact keeps recomputing byte-identically.

Two questions, both asked out of sample on a FRESH control sample that shares no patch and
no star with Gate #1 / Gate #2:

1. Does removing the Atlas matched-filter smoothing (13.69 arcsec empirical FWHM) bring the
   W3 PSF back towards 6.5 arcsec and improve image-level two-PSF blend recovery?
2. Does a correlated-noise (GLS) statistic or a distribution-free threshold restore null
   false-alarm control where the scalar-variance chi^2(3) cutoff failed (5/77)?

Geometry is declared in ARCSEC and is identical to Gate #2 (fit radius 16.5", sky beyond
22", primary search +-2.75", secondary search 11"); :class:`Geometry` converts it to pixels,
and at the Atlas scale (1.375"/px) it reproduces Gate #2's pixel constants exactly, so the
scalar statistic below IS the Gate #2 statistic (unit-tested).

Frozen on the calibration half only, before any withheld cutout exists:

* **Empirical PSF** (:func:`stack_empirical_psf`): sky-subtracted, windowed-centroid
  aligned stamps, pixel-wise median, renormalised to unit sum: the Gate #2 recipe with ONE
  deliberate change. Gate #2 divided every stamp by its own summed flux; on 2.75 arcsec
  pixels that sum runs over ~225 pixels of which ~10 hold the star, so it is noise-dominated
  at W3 SNR 8-20, and dropping the stamps whose sum is not positive leaves a positive
  pedestal in the stack (unit-tested). Here each stamp is divided by its least-squares
  amplitude against the nominal 6.5 arcsec Gaussian, whose noise is the star's own 1 / SNR.
* **Pixel-noise covariance** (:func:`collect_noise_samples`, :func:`regularise_lag_covariance`):
  pooled, sigma-clipped autocovariance of the sky annulus (r >= 22"; never the PSF-subtracted
  core, so it does not depend on the PSF model) in units of the per-cutout scalar variance,
  tapered, projected onto non-negative power on a torus, and shrunk towards white noise. Every
  per-star matrix is a principal submatrix of that circulant covariance: strictly positive
  definite with eigenvalues >= ``shrinkage * V(0)``, by construction and deterministically.
* **Screen A** ``image_scalar_chi2``: scalar-variance profile Delta chi^2, threshold
  ``s * chi2.isf(FA, 3)`` with the median-matched scale ``s`` (the Gate #2 rule verbatim; the
  direct control for question 1).
* **Screen B** ``image_gls_chi2``: the same profile Delta chi^2 after whitening data and
  templates with the Cholesky factor of ``Sigma = var_star * V`` (sky projected out in the
  whitened space), thresholded by the same median-matched rule (so A vs B isolates the
  noise model).
* **Screen C** ``image_scalar_conformal``: the scalar statistic with a distribution-free
  threshold, the ``(k+1)``-th largest LEAVE-ONE-OUT calibration null. A new exchangeable
  null exceeds it with probability exactly ``(k+1) / (n_cal + 1)`` (continuous scores; ``<=``
  with ties): no chi^2(3) tail assumption (:func:`conformal_threshold`).
* ``image_gls_conformal``: the remaining corner of the 2x2 (GLS statistic, conformal
  threshold), predeclared as a SECONDARY screen so it is never evaluated post hoc.
* Reference screens as in Gate #2: the centroid screen at the LOCKED 0.4356" floor and,
  side by side, at a floor re-measured on this calibration half; and the adversarial
  ``naive_flux_excess`` rule.

Leave-one-out calibration scores (PSF and covariance rebuilt without the star itself) feed
the conformal thresholds only; A and B keep Gate #2's in-sample scores so that they stay a
like-for-like control. Both sets are stored.

Zero-leakage state machine: every fitting step refuses non-calibration cutouts; a
:class:`FrozenCalibration` must be sealed with a UTC timestamp no earlier than its newest
calibration cutout; :func:`evaluate_withheld` refuses an unsealed state, any withheld
cutout whose fetch timestamp is missing or not strictly later than the seal, and any star
shared with the calibration set. All of these raise ``LeakageError``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from fermi_desk.errors import LeakageError, SchemaValidationError, UnitValidationError
from fermi_desk.models import hephaistos
from fermi_desk.models import hephaistos_fits as hf
from fermi_desk.models.hephaistos import (
    FWHM_PER_SIGMA,
    RAYLEIGH_3SIGMA_FALSE_ALARM,
    clopper_pearson,
)

RULE_VERSION = "unwise-gate-v1"
UNWISE_PIXEL_SCALE = 2.75 * u.arcsec  # Lang (2014) unWISE coadd pixel scale, all bands
NOMINAL_W3_FWHM_ARCSEC = hf.NOMINAL_W3_FWHM_ARCSEC  # 6.5 arcsec single-frame W3 FWHM
SEPARATIONS_ARCSEC = hf.SEPARATIONS_ARCSEC  # the Gate #2 Arm A axes, unchanged
FLUX_FRACTIONS = hf.FLUX_FRACTIONS
SURVIVAL_MAX_SEPARATION_ARCSEC = hf.SURVIVAL_MAX_SEPARATION_ARCSEC
CHI2_DOF = hf.CHI2_DOF
NAIVE_EXCESS_SIGMA = hf.NAIVE_EXCESS_SIGMA
SPLITS = hf.SPLITS

# Angular geometry: Gate #2's pixel constants times its 1.375 arcsec pixel.
FIT_RADIUS = 16.5 * u.arcsec
SKY_INNER = 22.0 * u.arcsec
STAMP_HALF = 20.625 * u.arcsec  # rounded DOWN to whole pixels (7 px = 19.25" on unWISE)
PRIMARY_RANGE = 2.75 * u.arcsec
SECONDARY_MAX = 11.0 * u.arcsec
FINE_STEP = 0.34375 * u.arcsec
COARSE_STEP = 0.6875 * u.arcsec

MIN_PSF_STAMPS = 5
HALFMAX_OVERSAMPLE = 8  # half-max area is measured on the spline, 8x finer than the pixels
# Noise-covariance estimator (all frozen constants of rule version unwise-gate-v1).
NOISE_CLIP_SIGMA = 4.0  # robust clip of sky-annulus sources before the autocovariance
NOISE_TAPER_FLAT = 0.5  # lags <= 0.5 * max_lag untouched, cosine roll-off to zero beyond
NOISE_POWER_FLOOR = 1.0e-3  # spectral floor, in units of the mean power V(0)
NOISE_SHRINKAGE = 0.05  # V <- (1 - lambda) V + lambda V(0) delta: eigenvalues >= lambda V(0)
MIN_LAG_PAIRS = 200
DEFAULT_CONFORMAL_ALPHA = 0.01  # ruling v0.2.2-R3: exact level 1 / (n_cal + 1) <= 1% at n_cal >= 99
PRIMARY_SIGMA_SYS_ESTIMATOR = "robust_median"  # the Gate #1 primary estimator, unchanged

CENTROID_SCREENS = ("centroid_locked", "centroid_fresh")
CHI2_SCREENS = {"image_scalar_chi2": "scalar", "image_gls_chi2": "gls"}
CONFORMAL_SCREENS = {"image_scalar_conformal": "scalar", "image_gls_conformal": "gls"}
SCREENS = (*CENTROID_SCREENS, *CHI2_SCREENS, *CONFORMAL_SCREENS, "naive_flux_excess")
SCREEN_ROLES = {
    "centroid_locked": "reference: Gate #1 centroid screen at the LOCKED sigma_sys",
    "centroid_fresh": "reference: centroid screen at sigma_sys re-measured on this "
    "calibration half (reported beside, never replacing, the locked value)",
    "image_scalar_chi2": "Screen A: scalar-variance Delta chi^2, median-matched chi2(3) "
    "threshold (the Gate #2 rule; control for the effect of Atlas smoothing)",
    "image_gls_chi2": "Screen B: GLS-whitened Delta chi^2 (empirical pixel covariance), "
    "median-matched chi2(3) threshold",
    "image_scalar_conformal": "Screen C: scalar-variance Delta chi^2, distribution-free "
    "leave-one-out conformal order-statistic threshold",
    "image_gls_conformal": "secondary (predeclared 2x2 corner): GLS-whitened Delta chi^2 "
    "with the conformal order-statistic threshold",
    "naive_flux_excess": "adversarial reference rule (NOT a proposed screen)",
}
STATISTICS = ("centroid_offset_arcsec", "delta_chi2_scalar", "delta_chi2_gls", "flux_ratio")
# Crowded extragalactic fields excluded from patch centres: (name, l, b, radius) in degrees.
AVOID_FIELDS = (
    ("LMC", 280.47, -32.89, 12.0),
    ("SMC", 302.81, -44.33, 8.0),
    ("M31", 121.17, -21.57, 5.0),
)
PATCH_STREAM = 0x554E57  # "UNW": patch draws never reuse the Gate #1 stream [seed, 0]
_GAIA_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")


# ------------------------------------------------------------------ small helpers
def _arcsec(q: object, name: str) -> float:
    if not isinstance(q, u.Quantity) or not q.unit.is_equivalent(u.arcsec):
        raise UnitValidationError(f"{name} must be an angle Quantity")
    return float(q.to_value(u.arcsec))


def _deg(q: object, name: str) -> float:
    return _arcsec(q, name) / 3600.0


def check_gaia_id(value: object) -> str:
    """A Gaia DR3 source id as the decimal STRING required by ruling v0.2.1-R4."""
    if not isinstance(value, str) or not _GAIA_ID_RE.match(value):
        raise SchemaValidationError(
            f"gaia_dr3_source_id must be a decimal string (ruling v0.2.1-R4), got {value!r}"
        )
    return value


def parse_utc(value: object, name: str) -> datetime:
    """An ISO-8601 timestamp as an aware UTC datetime (naive values are read as UTC)."""
    try:
        t = datetime.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} {value!r} is not an ISO-8601 timestamp") from exc
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def canonical_sha256(obj: object) -> str:
    """SHA-256 of the canonical JSON (same convention as ``inference.engine``)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=True).encode()).hexdigest()


def _id_key(star_id: str) -> int:
    return int(star_id)


# -------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    """The arcsec geometry above on one pixel grid (pixels; documented conversion)."""

    pixel_scale_arcsec: float
    fit_radius_pix: float
    sky_inner_pix: float
    stamp_half_pix: int
    primary_range_pix: float
    secondary_max_pix: float
    fine_step_pix: float
    coarse_step_pix: float

    @classmethod
    def for_pixel_scale(cls, pixel_scale: u.Quantity) -> Geometry:
        pix = _arcsec(pixel_scale, "pixel_scale")
        if not pix > 0:
            raise ValueError("pixel_scale must be positive")

        def px(q: u.Quantity) -> float:
            return float(q.to_value(u.arcsec)) / pix

        return cls(
            pix,
            px(FIT_RADIUS),
            px(SKY_INNER),
            int(np.floor(px(STAMP_HALF) + 1e-9)),
            px(PRIMARY_RANGE),
            px(SECONDARY_MAX),
            px(FINE_STEP),
            px(COARSE_STEP),
        )

    @property
    def need_pix(self) -> float:
        """Clear, finite pixels required around the star (PSF stamp and fit region)."""
        return max(self.stamp_half_pix + 3, self.fit_radius_pix + 1)

    @property
    def max_lag_pix(self) -> int:
        return int(np.ceil(self.fit_radius_pix - 1e-9))

    @property
    def n_torus(self) -> int:
        """Smallest power of two holding every lag inside the fit region without aliasing."""
        return int(2 ** np.ceil(np.log2(4 * self.max_lag_pix + 2)))

    @property
    def window_sigma_pix(self) -> float:
        return NOMINAL_W3_FWHM_ARCSEC / FWHM_PER_SIGMA / self.pixel_scale_arcsec

    def as_dict(self) -> dict:
        return {
            "pixel_scale_arcsec": self.pixel_scale_arcsec,
            "fit_radius_arcsec": float(FIT_RADIUS.to_value(u.arcsec)),
            "sky_inner_arcsec": float(SKY_INNER.to_value(u.arcsec)),
            "primary_range_arcsec": float(PRIMARY_RANGE.to_value(u.arcsec)),
            "secondary_max_arcsec": float(SECONDARY_MAX.to_value(u.arcsec)),
            "fine_step_arcsec": float(FINE_STEP.to_value(u.arcsec)),
            "coarse_step_arcsec": float(COARSE_STEP.to_value(u.arcsec)),
            "fit_radius_pix": self.fit_radius_pix,
            "sky_inner_pix": self.sky_inner_pix,
            "stamp_half_pix": self.stamp_half_pix,
            "primary_range_pix": self.primary_range_pix,
            "secondary_max_pix": self.secondary_max_pix,
            "fine_step_pix": self.fine_step_pix,
            "coarse_step_pix": self.coarse_step_pix,
            "max_lag_pix": self.max_lag_pix,
            "n_torus": self.n_torus,
            "note": "angular extents identical to Gate #2 (its pixel constants x 1.375 arcsec)",
        }


# ---------------------------------------------------------------------- cutouts
@dataclass(frozen=True, eq=False)
class UnwiseCutout:
    """One unWISE W3 intensity + uncertainty cutout of a control star.

    ``unc`` is the per-pixel standard deviation ``1 / sqrt(invvar)`` (see
    :meth:`from_invvar`); ``gaia_xy`` is the 0-based (x, y) pixel position of the Gaia DR3
    source propagated to the WISE epoch. ``fetched_at_utc`` is when the bytes left the
    archive; it is what the zero-leakage state machine orders against the freeze.
    """

    gaia_dr3_source_id: str
    split: str
    flux: np.ndarray
    unc: np.ndarray
    gaia_xy: tuple[float, float]
    pixel_scale: u.Quantity
    w3snr: float
    patch_id: str = ""
    fetched_at_utc: str | None = None

    def __post_init__(self) -> None:
        sid = check_gaia_id(self.gaia_dr3_source_id)
        flux = np.asarray(self.flux, float)
        unc = np.asarray(self.unc, float)
        if flux.ndim != 2 or flux.shape != unc.shape:
            raise ValueError(f"star {sid}: flux / unc must be 2-D and the same shape")
        if self.split not in SPLITS:
            raise ValueError(f"star {sid}: split must be one of {SPLITS}")
        if not self.w3snr > 0:
            raise ValueError(f"star {sid}: w3snr must be > 0")
        if _arcsec(self.pixel_scale, "pixel_scale") <= 0:
            raise ValueError("pixel_scale must be positive")
        if self.fetched_at_utc is not None:
            parse_utc(self.fetched_at_utc, f"star {sid}: fetched_at_utc")
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "unc", unc)
        object.__setattr__(self, "gaia_xy", (float(self.gaia_xy[0]), float(self.gaia_xy[1])))

    @classmethod
    def from_invvar(
        cls,
        gaia_dr3_source_id: str,
        split: str,
        flux: np.ndarray,
        invvar: np.ndarray,
        gaia_xy: tuple[float, float],
        pixel_scale: u.Quantity,
        w3snr: float,
        patch_id: str = "",
        fetched_at_utc: str | None = None,
    ) -> UnwiseCutout:
        """Build from an unWISE ``-invvar-m`` map: unc = 1 / sqrt(invvar), NaN where invvar <= 0."""
        iv = np.asarray(invvar, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            unc = np.where(iv > 0, 1.0 / np.sqrt(iv), np.nan)
        return cls(
            gaia_dr3_source_id, split, flux, unc, gaia_xy, pixel_scale, w3snr, patch_id,
            fetched_at_utc,
        )  # fmt: skip

    @property
    def star_id(self) -> str:
        return self.gaia_dr3_source_id

    @property
    def pix(self) -> float:
        return _arcsec(self.pixel_scale, "pixel_scale")

    def scaled(self, factor: float) -> UnwiseCutout:
        """Same cutout with flux and uncertainty multiplied by ``factor`` (unit change)."""
        return replace(self, flux=self.flux * factor, unc=self.unc * factor)


def _require_split(cutouts: Sequence[UnwiseCutout], split: str, step: str) -> list[UnwiseCutout]:
    if not cutouts:
        raise ValueError(f"{step}: no cutouts given")
    bad = [c.star_id for c in cutouts if c.split != split]
    if bad:
        raise LeakageError(
            f"{step} accepts only split == {split!r} cutouts; got {len(bad)} others "
            f"(e.g. star {bad[0]})"
        )
    ids = [c.star_id for c in cutouts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{step}: duplicate star ids")
    if len({round(c.pix, 9) for c in cutouts}) != 1:
        raise UnitValidationError(f"{step}: cutouts disagree on pixel scale")
    return sorted(cutouts, key=lambda c: _id_key(c.star_id))


def _radius(shape: tuple[int, int], x0: float, y0: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    return np.hypot(xx - x0, yy - y0)


def sky_level(img: np.ndarray, x0: float, y0: float, geometry: Geometry) -> float:
    sel = (_radius(img.shape, x0, y0) >= geometry.sky_inner_pix) & np.isfinite(img)
    if sel.sum() < 50:
        raise ValueError("too few sky pixels beyond the inner sky radius")
    return float(np.median(img[sel]))


def cutout_problem(c: UnwiseCutout, geometry: Geometry | None = None) -> str | None:
    """Reason a cutout cannot be used (truncated / non-finite near the star), else None."""
    geom = geometry or Geometry.for_pixel_scale(c.pixel_scale)
    gx, gy = c.gaia_xy
    need = geom.need_pix
    ny, nx = c.flux.shape
    if gx - need < 0 or gy - need < 0 or gx + need > nx - 1 or gy + need > ny - 1:
        return f"cutout {c.flux.shape} too small around the star (needs {need:.0f} px)"
    r = _radius(c.flux.shape, gx, gy)
    core = r <= need
    if not (np.all(np.isfinite(c.flux[core])) and np.all(np.isfinite(c.unc[core]))):
        return "non-finite flux / unc pixels near the star"
    if not np.all(c.unc[core] > 0):
        return "non-positive uncertainty pixels near the star"
    if int(((r >= geom.sky_inner_pix) & np.isfinite(c.flux)).sum()) < 50:
        return "too few finite sky pixels beyond the inner sky radius"
    return None


# ------------------------------------------------------------------ empirical PSF
def halfmax_fwhm_pix(stamp: np.ndarray, oversample: int = HALFMAX_OVERSAMPLE) -> float:
    """Model-free FWHM: diameter of the circle with the area of the >= half-peak region.

    Measured on the cubic spline of the stamp, ``oversample`` times finer than the pixels:
    at 2.75 arcsec pixels the W3 core covers ~4 native pixels, far too few to count.
    """
    st = np.asarray(stamp, float)
    h = (st.shape[0] - 1) / 2.0
    ax = np.arange(-h, h + 1.0)
    fine = np.linspace(-h, h, (st.shape[0] - 1) * oversample + 1)
    v = RectBivariateSpline(ax, ax, st, kx=3, ky=3)(fine, fine)
    area = float((v >= 0.5 * v.max()).sum()) / oversample**2
    return float(2.0 * np.sqrt(area / np.pi))


def psf_widths_arcsec(stamp: np.ndarray, pixel_scale_arcsec: float) -> dict:
    """Gaussian-fit and half-max-area FWHM of a unit-sum PSF stamp, in arcsec."""
    return {
        "fwhm_gaussian_arcsec": hf._gaussian_fwhm_pix(np.asarray(stamp, float))
        * pixel_scale_arcsec,
        "fwhm_halfmax_arcsec": halfmax_fwhm_pix(stamp) * pixel_scale_arcsec,
    }


@dataclass(frozen=True, eq=False)
class PsfStack:
    """Aligned, flux-normalised calibration stamps; the PSF is their pixel-wise median."""

    stamps: np.ndarray  # (n_star, S, S)
    star_ids: tuple[str, ...]
    excluded: tuple[dict, ...]
    pixel_scale: u.Quantity

    def psf(self, leave_out: str | None = None) -> hf.EmpiricalPSF:
        """Median PSF of every stamp, or of every stamp except ``leave_out``."""
        keep = [i for i, s in enumerate(self.star_ids) if s != leave_out]
        if len(keep) < MIN_PSF_STAMPS:
            raise ValueError(f"only {len(keep)} usable calibration stamps")
        med = np.median(self.stamps[keep], axis=0)
        ids = tuple(self.star_ids[i] for i in keep)
        return hf.EmpiricalPSF(med / med.sum(), self.pixel_scale, ids, self.excluded)


def stack_empirical_psf(
    calibration_cutouts: Sequence[UnwiseCutout], geometry: Geometry | None = None
) -> PsfStack:
    """Aligned PSF stamps from the calibration split only (raises LeakageError otherwise).

    Each stamp is divided by its least-squares amplitude against the nominal W3 Gaussian
    (a matched-filter flux with relative noise ~1 / SNR), not by its own summed flux.
    """
    cuts = _require_split(calibration_cutouts, "calibration", "stack_empirical_psf")
    geom = geometry or Geometry.for_pixel_scale(cuts[0].pixel_scale)
    ax = np.arange(-geom.stamp_half_pix, geom.stamp_half_pix + 1, dtype=float)
    window = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * geom.window_sigma_pix**2))
    window /= window.sum()
    stamps, used, excluded = [], [], []
    for c in cuts:
        why = cutout_problem(c, geom)
        if why is None:
            gx, gy = c.gaia_xy
            img = c.flux - sky_level(c.flux, gx, gy, geom)
            cx, cy = hf.windowed_centroid(c.flux, gx, gy, geom.window_sigma_pix)
            yy, xx = np.meshgrid(cy + ax, cx + ax, indexing="ij")
            st = ndimage.map_coordinates(
                np.nan_to_num(img), [yy, xx], order=3, mode="constant", cval=0.0
            )
            amplitude = float((st * window).sum() / (window * window).sum())
            if amplitude > 0:
                stamps.append(st / amplitude)
                used.append(c.star_id)
                continue
            why = "non-positive matched-filter flux"
        excluded.append({"gaia_dr3_source_id": c.star_id, "reason": why})
    if len(stamps) < MIN_PSF_STAMPS:
        raise ValueError(f"only {len(stamps)} usable calibration stamps")
    return PsfStack(np.stack(stamps), tuple(used), tuple(excluded), cuts[0].pixel_scale)


def psf_dict(psf: hf.EmpiricalPSF) -> dict:
    return {
        "stamp": psf.stamp.tolist(),
        "stamp_half_pix": psf.half,
        "pixel_scale_arcsec": psf.pix,
        "n_stars": len(psf.star_ids),
        "star_ids": list(psf.star_ids),
        "excluded": list(psf.excluded),
        **psf_widths_arcsec(psf.stamp, psf.pix),
        "halfmax_oversample": HALFMAX_OVERSAMPLE,
        "nominal_fwhm_arcsec": NOMINAL_W3_FWHM_ARCSEC,
        "construction": (
            f"sky = median(r >= {SKY_INNER.to_value(u.arcsec):g} arcsec); Gate #1 windowed "
            f'centroid ({NOMINAL_W3_FWHM_ARCSEC}" window); cubic-spline recentring; '
            f"{2 * psf.half + 1}x{2 * psf.half + 1} stamps, each divided by its least-squares "
            f'amplitude against the nominal {NOMINAL_W3_FWHM_ARCSEC}" Gaussian (NOT by its own '
            "summed flux, as Gate #2 did); pixel-wise median; renormalised to unit sum"
        ),
    }


# --------------------------------------------------------------- noise covariance
def scalar_variance(c: UnwiseCutout, geometry: Geometry) -> float:
    """Gate #2's one-number noise model: median unc^2 over the fit region."""
    gx, gy = c.gaia_xy
    sel = _radius(c.flux.shape, gx, gy) <= geometry.fit_radius_pix
    return float(np.median(c.unc[sel] ** 2))


def _noise_field(c: UnwiseCutout, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Sky-annulus pixels in units of the cutout's scalar sigma, sources clipped out."""
    gx, gy = c.gaia_xy
    z = (c.flux - sky_level(c.flux, gx, gy, geometry)) / np.sqrt(scalar_variance(c, geometry))
    m = (_radius(c.flux.shape, gx, gy) >= geometry.sky_inner_pix) & np.isfinite(z)
    med = float(np.median(z[m]))
    mad = float(np.median(np.abs(z[m] - med)))
    bad = m & (np.abs(z - med) > NOISE_CLIP_SIGMA * 1.4826 * mad)
    m &= ~ndimage.binary_dilation(bad, iterations=1)  # also drop the wings of a clipped source
    if int(m.sum()) < 50:
        raise ValueError("too few sky pixels survive the noise clip")
    return np.where(m, z - float(z[m].mean()), 0.0), m


def _lag_sums(z: np.ndarray, m: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """sum z_i z_j and the pair count for every pixel lag in [-lag, lag]^2 (masked)."""
    ny, nx = z.shape
    mf = m.astype(float)
    num = np.zeros((2 * lag + 1, 2 * lag + 1))
    den = np.zeros_like(num)
    for dy in range(-lag, lag + 1):
        ya, yb = slice(max(0, dy), ny + min(0, dy)), slice(max(0, -dy), ny + min(0, -dy))
        for dx in range(-lag, lag + 1):
            xa, xb = slice(max(0, dx), nx + min(0, dx)), slice(max(0, -dx), nx + min(0, -dx))
            num[dy + lag, dx + lag] = float(np.sum(z[ya, xa] * z[yb, xb]))
            den[dy + lag, dx + lag] = float(np.sum(mf[ya, xa] * mf[yb, xb]))
    return num, den


def regularise_lag_covariance(
    raw: np.ndarray,
    n_torus: int,
    shrinkage: float = NOISE_SHRINKAGE,
    power_floor: float = NOISE_POWER_FLOOR,
) -> np.ndarray:
    """Strictly positive-definite stationary covariance on an ``n_torus`` x ``n_torus`` torus.

    ``raw`` is an estimated autocovariance on lags [-L, L]^2. Steps (all deterministic):
    symmetrise; radial cosine taper to zero beyond L; FFT to the power spectrum and clip it
    at ``power_floor * V(0)`` (non-negative power <=> positive semi-definite); rescale to the
    measured V(0); shrink towards white noise, ``(1 - shrinkage) V + shrinkage V(0) delta``.
    Every eigenvalue of every matrix cut from the result is >= ``shrinkage * V(0)``.
    """
    raw = np.asarray(raw, float)
    lag = (raw.shape[0] - 1) // 2
    if raw.shape != (2 * lag + 1, 2 * lag + 1) or n_torus < 4 * lag + 2:
        raise ValueError("lag covariance must be (2L+1) square and fit the torus unaliased")
    if not 0 < shrinkage < 1 or not power_floor > 0:
        raise ValueError("shrinkage must be in (0, 1) and power_floor positive")
    v0 = float(raw[lag, lag])
    if not v0 > 0 or not np.all(np.isfinite(raw)):
        raise ValueError("zero-lag variance must be positive and every lag finite")
    d = np.arange(-lag, lag + 1)
    r = np.hypot(*np.meshgrid(d, d, indexing="ij"))
    flat, edge = NOISE_TAPER_FLAT * lag, lag + 1.0
    roll = 0.5 * (1.0 + np.cos(np.pi * np.clip((r - flat) / (edge - flat), 0.0, 1.0)))
    grid = np.zeros((n_torus, n_torus))
    grid[np.ix_(d % n_torus, d % n_torus)] = 0.5 * (raw + raw[::-1, ::-1]) * roll
    power = np.maximum(np.fft.fft2(grid).real, power_floor * v0)
    cov = np.fft.ifft2(power).real
    cov *= (1.0 - shrinkage) * v0 / cov[0, 0]
    cov[0, 0] += shrinkage * v0
    flip = (-np.arange(n_torus)) % n_torus
    return 0.5 * (cov + cov[np.ix_(flip, flip)])


def minimum_power(torus_covariance: np.ndarray) -> float:
    """Smallest eigenvalue of the circulant covariance = lower bound for every submatrix."""
    return float(np.fft.fft2(np.asarray(torus_covariance, float)).real.min())


@dataclass(frozen=True, eq=False)
class NoiseModel:
    """Frozen stationary pixel-noise covariance, in units of the per-cutout scalar variance."""

    torus_covariance: np.ndarray  # (N, N); V(dy, dx) at [dy % N, dx % N]
    raw_lag_covariance: np.ndarray  # (2L+1, 2L+1) pooled estimate before regularisation
    n_pairs: np.ndarray
    star_ids: tuple[str, ...]
    shrinkage: float = NOISE_SHRINKAGE
    power_floor: float = NOISE_POWER_FLOOR

    @property
    def n_torus(self) -> int:
        return int(self.torus_covariance.shape[0])

    @property
    def variance_inflation(self) -> float:
        """V(0): measured sky variance over the variance the uncertainty map claims."""
        return float(self.torus_covariance[0, 0])

    def matrix(self, py: np.ndarray, px: np.ndarray) -> np.ndarray:
        """Covariance (units of scalar variance) of the pixels at integer (py, px)."""
        n = self.n_torus
        if np.ptp(py) >= n // 2 or np.ptp(px) >= n // 2:
            raise ValueError("pixel set is wider than the unaliased half of the torus")
        return self.torus_covariance[
            (py[:, None] - py[None, :]) % n, (px[:, None] - px[None, :]) % n
        ]

    def as_dict(self) -> dict:
        lag = (self.raw_lag_covariance.shape[0] - 1) // 2
        v = self.torus_covariance
        return {
            "n_torus": self.n_torus,
            "max_lag_pix": lag,
            "torus_covariance": v.tolist(),
            "raw_lag_covariance": self.raw_lag_covariance.tolist(),
            "n_pairs": self.n_pairs.tolist(),
            "n_cutouts": len(self.star_ids),
            "star_ids": list(self.star_ids),
            "shrinkage": self.shrinkage,
            "power_floor": self.power_floor,
            "clip_sigma": NOISE_CLIP_SIGMA,
            "taper_flat_fraction": NOISE_TAPER_FLAT,
            "variance_inflation": self.variance_inflation,
            "correlation_lag1": [float(v[0, 1] / v[0, 0]), float(v[1, 0] / v[0, 0])],
            "raw_correlation_lag1": [
                float(self.raw_lag_covariance[lag, lag + 1] / self.raw_lag_covariance[lag, lag]),
                float(self.raw_lag_covariance[lag + 1, lag] / self.raw_lag_covariance[lag, lag]),
            ],
            "minimum_eigenvalue_bound": minimum_power(v),
            "construction": (
                f"pooled autocovariance of sky-annulus pixels (r >= "
                f"{SKY_INNER.to_value(u.arcsec):g} arcsec, never the PSF-subtracted core) in "
                f"units of each cutout's scalar variance; {NOISE_CLIP_SIGMA:g}-sigma MAD clip + "
                f"1 px dilation; cosine taper beyond {NOISE_TAPER_FLAT:g} max_lag; power "
                f"spectrum clipped at {self.power_floor:g} V(0) on the torus; shrinkage "
                f"{self.shrinkage:g} towards white noise. Sigma_star = var_star * V is strictly "
                "positive definite (eigenvalues >= shrinkage * V(0) * var_star)"
            ),
        }


@dataclass(frozen=True, eq=False)
class NoiseSamples:
    """Per-cutout lag sums of the calibration sky annuli (pooled into a NoiseModel)."""

    star_ids: tuple[str, ...]
    num: np.ndarray  # (n_star, 2L+1, 2L+1)
    den: np.ndarray
    n_torus: int

    def model(self, leave_out: str | None = None) -> NoiseModel:
        keep = [i for i, s in enumerate(self.star_ids) if s != leave_out]
        if not keep:
            raise ValueError("no calibration cutouts left for the noise covariance")
        num, den = self.num[keep].sum(axis=0), self.den[keep].sum(axis=0)
        if float(den.min()) < MIN_LAG_PAIRS:
            raise ValueError(f"fewer than {MIN_LAG_PAIRS} pixel pairs at some lag")
        raw = num / den
        return NoiseModel(
            regularise_lag_covariance(raw, self.n_torus),
            raw,
            den,
            tuple(self.star_ids[i] for i in keep),
        )


def collect_noise_samples(
    calibration_cutouts: Sequence[UnwiseCutout], geometry: Geometry | None = None
) -> NoiseSamples:
    """Sky-annulus lag sums from the calibration split only (LeakageError otherwise)."""
    cuts = _require_split(calibration_cutouts, "calibration", "collect_noise_samples")
    geom = geometry or Geometry.for_pixel_scale(cuts[0].pixel_scale)
    ids, nums, dens = [], [], []
    for c in cuts:
        if cutout_problem(c, geom) is not None:
            continue
        num, den = _lag_sums(*_noise_field(c, geom), geom.max_lag_pix)
        ids.append(c.star_id)
        nums.append(num)
        dens.append(den)
    if not ids:
        raise ValueError("no usable calibration cutouts for the noise covariance")
    return NoiseSamples(tuple(ids), np.stack(nums), np.stack(dens), geom.n_torus)


def scalar_projector(n: int, variance: float) -> np.ndarray:
    """Whitening + sky projection for white noise: (I - 11^T / n) / sigma (Gate #2 algebra)."""
    if not variance > 0:
        raise ValueError("scalar variance must be positive")
    return (np.eye(n) - 1.0 / n) / np.sqrt(variance)


def gls_projector(covariance: np.ndarray) -> np.ndarray:
    """Whitening + sky projection for ``Sigma = L L^T``: P_u L^-1 with u = L^-1 1.

    ``np.linalg.cholesky`` raises ``LinAlgError`` unless Sigma is strictly positive definite.
    """
    low = np.linalg.cholesky(covariance)
    w = solve_triangular(low, np.eye(low.shape[0]), lower=True)
    unit = w.sum(axis=1)
    unit /= np.linalg.norm(unit)
    return w - np.outer(unit, unit @ w)


# ---------------------------------------------------------------- image statistic
class _Templates:
    """Per-star fit region and raw (un-whitened) template banks shared by both statistics."""

    def __init__(self, psf: hf.EmpiricalPSF, c: UnwiseCutout, geom: Geometry) -> None:
        gx, gy = c.gaia_xy
        r = geom.fit_radius_pix
        x0, x1 = int(np.floor(gx - r)), int(np.ceil(gx + r))
        y0, y1 = int(np.floor(gy - r)), int(np.ceil(gy + r))
        self.sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        self.ys = np.arange(y0, y1 + 1, dtype=float)
        self.xs = np.arange(x0, x1 + 1, dtype=float)
        yy, xx = np.meshgrid(self.ys, self.xs, indexing="ij")
        self.mask = (xx - gx) ** 2 + (yy - gy) ** 2 <= r * r
        self.py = yy[self.mask].astype(int)
        self.px = xx[self.mask].astype(int)
        self.n = int(self.mask.sum())
        self.gx, self.gy, self.psf, self.geom = gx, gy, psf, geom
        pr, sm = geom.primary_range_pix, geom.secondary_max_pix
        fine = np.arange(-pr, pr + 1e-9, geom.fine_step_pix)
        coarse = np.arange(-pr, pr + 1e-9, geom.coarse_step_pix)
        rb = np.arange(-sm, sm + 1e-9, geom.coarse_step_pix)
        self.a_off = np.array([(x, y) for x in fine for y in fine])
        self.s_off = np.array([(x, y) for x in coarse for y in coarse])
        self.b_off = np.array([(x, y) for x in rb for y in rb if x * x + y * y <= sm**2])
        self.a_raw = np.stack([self.raw(*o) for o in self.a_off])
        self.s_raw = np.stack([self.raw(*o) for o in self.s_off])
        self.b_raw = np.stack([self.raw(*o) for o in self.b_off])

    def raw(self, dx: float, dy: float) -> np.ndarray:
        return self.at(self.gx + dx, self.gy + dy)

    def at(self, x: float, y: float) -> np.ndarray:
        return self.psf.evaluate(self.ys - y, self.xs - x)[self.mask]

    def vec(self, img: np.ndarray) -> np.ndarray:
        return np.asarray(img, float)[self.sl][self.mask]


class _Statistic:
    """Profile Delta chi^2 (one PSF vs two) in a whitened, sky-projected space.

    ``projector`` maps raw fit-region vectors to unit-variance residual space with the
    constant sky removed; with :func:`scalar_projector` every expression below reduces to
    ``hephaistos_fits._ScreenBank`` (data and templates minus their mean, divided by var).
    """

    def __init__(self, tm: _Templates, projector: np.ndarray, sys_pix: float) -> None:
        self.tm, self.m = tm, projector
        self.sys2 = max(sys_pix, 1e-3) ** 2
        self.sec2 = tm.geom.secondary_max_pix**2
        self.a_t = tm.a_raw @ projector.T
        self.s_t = tm.s_raw @ projector.T
        self.b_t = tm.b_raw @ projector.T
        self.a_nn = (self.a_t * self.a_t).sum(1)
        self.a_pen = (tm.a_off**2).sum(1) / self.sys2
        self.s_pen = (tm.s_off**2).sum(1) / self.sys2
        self.ss = (self.s_t * self.s_t).sum(1)
        self.bb = (self.b_t * self.b_t).sum(1)
        self.sb = self.s_t @ self.b_t.T
        det = self.ss[:, None] * self.bb[None, :] - self.sb**2
        self.det_ok = det > 1e-6 * self.ss[:, None] * self.bb[None, :]
        self.det = np.where(self.det_ok, det, 1.0)

    def tmpl(self, dx: float, dy: float) -> np.ndarray:
        return self.m @ self.tm.raw(dx, dy)

    def _chi_a(self, y: np.ndarray, yy: float, p: np.ndarray) -> float:
        t = self.tmpl(p[0], p[1])
        b = float(y @ t)
        return (yy - b * b / float(t @ t)) + float(p @ p) / self.sys2

    def _chi_b(self, y: np.ndarray, yy: float, q: np.ndarray) -> float:
        p1, p2 = q[:2], q[2:]
        if float(p2 @ p2) > self.sec2:
            return 1e30
        t1, t2 = self.tmpl(*p1), self.tmpl(*p2)
        s11, s22, s12 = float(t1 @ t1), float(t2 @ t2), float(t1 @ t2)
        b1, b2 = float(y @ t1), float(y @ t2)
        det = s11 * s22 - s12 * s12
        rss = yy - b1 * b1 / s11
        if det > 1e-6 * s11 * s22:
            f1 = (s22 * b1 - s12 * b2) / det
            f2 = (s11 * b2 - s12 * b1) / det
            if f2 >= 0:
                rss = yy - (f1 * b1 + f2 * b2)
        return rss + float(p1 @ p1) / self.sys2

    def delta_chi2(self, img: np.ndarray) -> float:
        y = self.m @ self.tm.vec(img)
        yy = float(y @ y)
        ba = self.a_t @ y
        chi_a_grid = (yy - ba**2 / self.a_nn) + self.a_pen
        ia = int(np.argmin(chi_a_grid))
        ra = minimize(
            lambda p: self._chi_a(y, yy, p),
            self.tm.a_off[ia],
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-4, "maxfev": 400},
        )
        chi_a = min(float(ra.fun), float(chi_a_grid[ia]))

        b1 = (self.s_t @ y)[:, None]
        b2 = (self.b_t @ y)[None, :]
        f1 = (self.bb[None, :] * b1 - self.sb * b2) / self.det
        f2 = (self.ss[:, None] * b2 - self.sb * b1) / self.det
        rss = yy - (f1 * b1 + f2 * b2)
        single = yy - b1**2 / self.ss[:, None]
        rss = np.where(self.det_ok & (f2 >= 0), rss, single)
        chi_b_grid = rss + self.s_pen[:, None]
        i, j = np.unravel_index(int(np.argmin(chi_b_grid)), chi_b_grid.shape)
        rb = minimize(
            lambda q: self._chi_b(y, yy, q),
            np.concatenate([self.tm.s_off[i], self.tm.b_off[j]]),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-4, "maxfev": 1200},
        )
        chi_b = min(float(rb.fun), float(chi_b_grid[i, j]), chi_a)
        return max(chi_a - chi_b, 0.0)


class _StarScreens:
    """Every screen statistic of one star under one (PSF, noise model) pair."""

    def __init__(
        self,
        c: UnwiseCutout,
        psf: hf.EmpiricalPSF,
        noise: NoiseModel,
        geom: Geometry,
        sigma_sys_arcsec: float,
    ) -> None:
        self.c, self.geom = c, geom
        self.tm = _Templates(psf, c, geom)
        var = scalar_variance(c, geom)
        sys_pix = sigma_sys_arcsec / geom.pixel_scale_arcsec
        self.scalar = _Statistic(self.tm, scalar_projector(self.tm.n, var), sys_pix)
        self.gls = _Statistic(
            self.tm, gls_projector(var * noise.matrix(self.tm.py, self.tm.px)), sys_pix
        )
        self._aperture = _radius(c.flux.shape, *c.gaia_xy) <= geom.fit_radius_pix

    def centroid(self, img: np.ndarray) -> tuple[float, float]:
        return hf.windowed_centroid(img, *self.c.gaia_xy, self.geom.window_sigma_pix)

    def offset_arcsec(self, img: np.ndarray) -> float:
        cx, cy = self.centroid(img)
        gx, gy = self.c.gaia_xy
        return float(np.hypot(cx - gx, cy - gy) * self.geom.pixel_scale_arcsec)

    def psf_flux(self, img: np.ndarray, x: float, y: float) -> float:
        """Linear (ordinary least squares) PSF-fit flux at pixel (x, y), sky projected out."""
        t = self.tm.at(x, y)
        t = t - t.mean()
        d = self.tm.vec(img)
        return float((d - d.mean()) @ t / (t @ t))

    def aperture_flux(self, img: np.ndarray) -> float:
        """Sky-subtracted aperture sum within the fit radius of the Gaia position."""
        gx, gy = self.c.gaia_xy
        return float(np.sum(img[self._aperture] - sky_level(img, gx, gy, self.geom)))

    def stats(self, img: np.ndarray, f_ref: float, ap_ref: float) -> tuple[float, ...]:
        ratio = 1.0 + (self.aperture_flux(img) - ap_ref) / f_ref
        return (
            self.offset_arcsec(img),
            self.scalar.delta_chi2(img),
            self.gls.delta_chi2(img),
            float(ratio),
        )


# --------------------------------------------------------------- frozen thresholds
def conformal_threshold(scores: Sequence[float], alpha: float) -> dict:
    """Distribution-free threshold: the (k+1)-th largest calibration null score.

    With ``n`` exchangeable calibration nulls and one new null, the new score is equally
    likely to hold any of the ``n + 1`` ranks, so it exceeds the ``(k+1)``-th largest
    calibration score with probability exactly ``(k + 1) / (n + 1)`` (continuous scores; at
    most that with ties). ``k`` is the largest integer with ``(k + 1) / (n + 1) <= alpha``:
    ``k = floor(alpha (n + 1)) - 1``. ``alpha < 1 / (n + 1)`` is unattainable and raises.
    """
    s = np.sort(np.asarray(scores, float))
    n = int(s.size)
    if n < 1 or not np.all(np.isfinite(s)) or not 0 < alpha < 1:
        raise ValueError("need finite calibration scores and alpha in (0, 1)")
    rank = int(np.floor(alpha * (n + 1) + 1e-9))
    if rank < 1:
        raise ValueError(
            f"alpha = {alpha:g} is not attainable with n_cal = {n}: the smallest finite-sample "
            f"level is 1 / (n_cal + 1) = {1.0 / (n + 1):.5f} (needs n_cal >= "
            f"{int(np.ceil(1.0 / alpha - 1e-9)) - 1})"
        )
    thr = float(s[n - rank])
    return {
        "alpha_nominal": float(alpha),
        "n_calibration": n,
        "k_exceedances": rank - 1,
        "order_statistic_rank_from_top": rank,
        "alpha_guaranteed": rank / (n + 1),
        "threshold": thr,
        "n_calibration_strictly_above": int((s > thr).sum()),
    }


def resolve_conformal_alpha(requested: float, n_cal: int, *, allow_fallback: bool = False) -> float:
    """The level the conformal screens are frozen at (ruling v0.2.2-R3).

    ``requested`` whenever ``n_cal`` calibration nulls can attain it, i.e. ``requested >=
    1 / (n_cal + 1)``. Otherwise, with ``allow_fallback``, the smallest level they can
    guarantee, ``1 / (n_cal + 1)`` (the calibration maximum, ``k = 0``); without it the
    request is returned unchanged and :func:`conformal_threshold` refuses it. The choice
    depends on the calibration half only (its size), so it is part of the freeze.
    """
    if not 0 < requested < 1 or n_cal < 1:
        raise ValueError("need conformal alpha in (0, 1) and at least one calibration null")
    if not allow_fallback or np.floor(requested * (n_cal + 1) + 1e-9) >= 1:
        return float(requested)
    return 1.0 / (n_cal + 1)


def fresh_sigma_sys(offset_arcsec: Sequence[float], w3snr: Sequence[float]) -> dict:
    """Registration floor re-measured on the calibration half (every Gate #1 estimator)."""
    r = np.asarray(offset_arcsec, float) * u.arcsec
    s = hf.centroid_sigma_stat_arcsec(w3snr) * u.arcsec
    by = {
        k: float(fn(r, s).to_value(u.arcsec)) for k, fn in hephaistos.SIGMA_SYS_ESTIMATORS.items()
    }
    return {
        "primary_estimator": PRIMARY_SIGMA_SYS_ESTIMATOR,
        "sigma_sys_arcsec": by[PRIMARY_SIGMA_SYS_ESTIMATOR],
        "sigma_sys_by_estimator_arcsec": by,
        "n_stars": int(r.size),
        "sigma_stat_rule": f'{NOMINAL_W3_FWHM_ARCSEC}" / (2.355 SNR)',
    }


def frozen_screen_rules(
    calibration: Mapping,
    locked_sigma_sys_arcsec: float,
    target_false_alarm: float,
    conformal_alpha: float,
) -> dict:
    """Every frozen threshold, derived from the stored calibration nulls alone (pure).

    Used at freeze time and again by the offline validator, so a promoted file cannot carry
    a threshold its own calibration table does not produce.
    """
    if not 0 < target_false_alarm < 1:
        raise ValueError("target_false_alarm must be in (0, 1)")
    snr = np.asarray(calibration["w3snr"], float)
    fresh = fresh_sigma_sys(calibration["centroid_offset_arcsec"], snr)
    snr_med = float(np.median(snr))
    mult = float(np.sqrt(-2.0 * np.log(target_false_alarm)))
    out: dict = {}
    for name, sys_arcsec, kind in (
        ("centroid_locked", float(locked_sigma_sys_arcsec), "locked"),
        ("centroid_fresh", fresh["sigma_sys_arcsec"], "fresh_calibration_half"),
    ):
        out[name] = {
            "role": SCREEN_ROLES[name],
            "statistic": "centroid_offset_arcsec",
            "sigma_sys_arcsec": sys_arcsec,
            "sigma_sys_source": kind,
            "sigma_stat_rule": f'{NOMINAL_W3_FWHM_ARCSEC}" / (2.355 SNR_eff)',
            "target_false_alarm": float(target_false_alarm),
            "multiplier": mult,
            "threshold_arcsec_at_calibration_median_snr": float(
                hf.centroid_threshold_arcsec(snr_med, sys_arcsec, target_false_alarm)
            ),
            "calibration_median_snr": snr_med,
        }
    for name, stat in CHI2_SCREENS.items():
        scale = hf.delta_chi2_scale(np.asarray(calibration[f"delta_chi2_{stat}"], float))
        out[name] = {
            "role": SCREEN_ROLES[name],
            "statistic": f"delta_chi2_{stat}",
            "calibration_scores": f"delta_chi2_{stat} (in-sample, as in Gate #2)",
            "chi2_dof": CHI2_DOF,
            "scale_rule": "s = median(Delta chi^2_cal) / median(chi2(3)); threshold = "
            "s * chi2.isf(FA, 3)",
            "target_false_alarm": float(target_false_alarm),
            "delta_chi2_scale": scale,
            "threshold": hf.delta_chi2_threshold(scale, target_false_alarm),
        }
    for name, stat in CONFORMAL_SCREENS.items():
        out[name] = {
            "role": SCREEN_ROLES[name],
            "statistic": f"delta_chi2_{stat}",
            "calibration_scores": f"delta_chi2_{stat}_loo (leave-one-out: PSF and covariance "
            "rebuilt without the star itself)",
            "rule": "flag when Delta chi^2 > the (k+1)-th largest calibration null; "
            "k = floor(alpha (n_cal + 1)) - 1; P(false alarm) = (k + 1) / (n_cal + 1) for "
            "exchangeable continuous scores, with no tail-shape assumption",
            **conformal_threshold(calibration[f"delta_chi2_{stat}_loo"], conformal_alpha),
        }
    out["naive_flux_excess"] = {
        "role": SCREEN_ROLES["naive_flux_excess"],
        "statistic": "flux_ratio",
        "rule": f"flux_ratio - 1 > {NAIVE_EXCESS_SIGMA:g} / SNR, flux_ratio = 1 + (aperture "
        f"flux - uninjected aperture flux) / F_star, r <= {FIT_RADIUS.to_value(u.arcsec):g} arcsec",
    }
    return {"fresh_sigma_sys": fresh, "screens": out}


def _rate(flags: np.ndarray) -> dict:
    f = np.asarray(flags, bool).ravel()
    k, n = int(f.sum()), int(f.size)
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n, "clopper_pearson_95": [lo, hi]}


def calibration_false_alarms(calibration: Mapping, screens: Mapping) -> dict:
    """In-sample false alarms of every frozen screen on the scores it was frozen from."""
    snr = np.asarray(calibration["w3snr"], float)
    off = np.asarray(calibration["centroid_offset_arcsec"], float)
    out = {}
    for name in CENTROID_SCREENS:
        sc = screens[name]
        thr = hf.centroid_threshold_arcsec(snr, sc["sigma_sys_arcsec"], sc["target_false_alarm"])
        out[name] = _rate(off > thr)
    for name, stat in CHI2_SCREENS.items():
        scores = np.asarray(calibration[f"delta_chi2_{stat}"], float)
        out[name] = _rate(scores > screens[name]["threshold"])
    for name, stat in CONFORMAL_SCREENS.items():
        scores = np.asarray(calibration[f"delta_chi2_{stat}_loo"], float)
        out[name] = _rate(scores > screens[name]["threshold"])
    return out


@dataclass(frozen=True, eq=False)
class FrozenCalibration:
    """PSF, noise covariance and every threshold, fixed on the calibration split.

    Unsealed until :meth:`seal` records the UTC time of the freeze; only a sealed state can
    evaluate withheld cutouts.
    """

    geometry: Geometry
    psf: hf.EmpiricalPSF
    noise: NoiseModel
    locked_sigma_sys_arcsec: float
    target_false_alarm: float
    conformal_alpha: float
    calibration: dict  # star_ids, patch_ids, w3snr, offsets, 4 Delta chi^2 arrays, fetch times
    frozen_at_utc: str | None = None
    # The level asked for (v0.2.2-R3). It differs from ``conformal_alpha`` only when the
    # calibration half was too small to attain it and the frozen level fell back to
    # 1 / (n_cal + 1); ``None`` means "as frozen".
    conformal_alpha_requested: float | None = None

    @property
    def star_ids(self) -> tuple[str, ...]:
        return tuple(self.calibration["star_ids"])

    @property
    def is_sealed(self) -> bool:
        return self.frozen_at_utc is not None

    @property
    def alpha_requested(self) -> float:
        req = self.conformal_alpha_requested
        return self.conformal_alpha if req is None else float(req)

    @property
    def conformal_fallback(self) -> bool:
        """True when the frozen level is the R3 fallback 1 / (n_cal + 1), not the request."""
        return self.alpha_requested != self.conformal_alpha

    def rules(self) -> dict:
        return frozen_screen_rules(
            self.calibration,
            self.locked_sigma_sys_arcsec,
            self.target_false_alarm,
            self.conformal_alpha,
        )

    def payload(self) -> dict:
        """The frozen state as JSON-ready blocks (no seal time: see ``state_sha256``)."""
        rules = self.rules()
        return {
            "rule_version": RULE_VERSION,
            "geometry": self.geometry.as_dict(),
            "empirical_psf": psf_dict(self.psf),
            "noise_model": self.noise.as_dict(),
            "frozen_screens": {
                "target_false_alarm": self.target_false_alarm,
                "conformal_alpha_nominal": self.conformal_alpha,
                "conformal_alpha_requested": self.alpha_requested,
                "conformal_alpha_fallback": self.conformal_fallback,
                "locked_sigma_sys_arcsec": self.locked_sigma_sys_arcsec,
                "fresh_sigma_sys": rules["fresh_sigma_sys"],
                "screens": rules["screens"],
                "snr_eff_rule": "SNR_eff = w3snr / (1 - f_bg) (catalog sees the blended flux)",
                "calibration": {
                    k: list(v) if isinstance(v, list | tuple) else v
                    for k, v in self.calibration.items()
                },
                "calibration_false_alarm_in_sample": calibration_false_alarms(
                    self.calibration, rules["screens"]
                ),
            },
        }

    @property
    def state_sha256(self) -> str:
        """Fingerprint of everything that is frozen (independent of when it was sealed)."""
        return canonical_sha256(self.payload())

    def seal(self, frozen_at_utc: str) -> FrozenCalibration:
        """Record the freeze time. It may not precede any calibration cutout it was built from."""
        if self.is_sealed:
            raise LeakageError("calibration state is already sealed; a freeze is never repeated")
        try:
            at = parse_utc(frozen_at_utc, "frozen_at_utc")
            times = [parse_utc(t, "calibration fetched_at_utc") for t in self._fetch_times()]
        except ValueError as exc:
            raise LeakageError(f"cannot order the freeze against its inputs: {exc}") from exc
        if times and max(times) > at:
            raise LeakageError(
                f"frozen_at_utc {frozen_at_utc} precedes a calibration cutout fetched at "
                f"{max(times).isoformat()}: that is not the time of the freeze"
            )
        return replace(self, frozen_at_utc=at.isoformat())

    def _fetch_times(self) -> list[str]:
        times = list(self.calibration["fetched_at_utc"])
        if any(t is None for t in times):
            raise ValueError("a calibration cutout carries no fetch timestamp")
        return times

    @classmethod
    def from_payload(cls, payload: Mapping, frozen_at_utc: str | None) -> FrozenCalibration:
        """Rebuild a frozen state from its stored blocks (used to resume, never to refit)."""
        p, nm, fz = payload["empirical_psf"], payload["noise_model"], payload["frozen_screens"]
        pixel_scale = float(p["pixel_scale_arcsec"]) * u.arcsec
        psf = hf.EmpiricalPSF(
            np.asarray(p["stamp"], float),
            pixel_scale,
            tuple(p["star_ids"]),
            tuple(p.get("excluded", ())),
        )
        noise = NoiseModel(
            np.asarray(nm["torus_covariance"], float),
            np.asarray(nm["raw_lag_covariance"], float),
            np.asarray(nm["n_pairs"], float),
            tuple(nm["star_ids"]),
            float(nm["shrinkage"]),
            float(nm["power_floor"]),
        )
        state = cls(
            Geometry.for_pixel_scale(pixel_scale),
            psf,
            noise,
            float(fz["locked_sigma_sys_arcsec"]),
            float(fz["target_false_alarm"]),
            float(fz["conformal_alpha_nominal"]),
            {k: list(v) if isinstance(v, list) else v for k, v in fz["calibration"].items()},
            conformal_alpha_requested=float(fz["conformal_alpha_requested"]),
        )
        if state.conformal_fallback != bool(fz["conformal_alpha_fallback"]):
            raise LeakageError("stored conformal fallback flag disagrees with its two levels")
        return replace(state, frozen_at_utc=frozen_at_utc)


def freeze_calibration(
    calibration_cutouts: Sequence[UnwiseCutout],
    locked_sigma_sys: u.Quantity,
    *,
    target_false_alarm: float = RAYLEIGH_3SIGMA_FALSE_ALARM,
    conformal_alpha: float = DEFAULT_CONFORMAL_ALPHA,
    allow_conformal_fallback: bool = False,
) -> FrozenCalibration:
    """Fit the PSF, the noise covariance and every threshold on the calibration split.

    Raises ``LeakageError`` if any cutout is not labelled ``calibration``. The returned state
    is unsealed. ``conformal_alpha`` is the requested level; with ``allow_conformal_fallback``
    a calibration half too small to attain it freezes the conformal screens at
    ``1 / (n_cal + 1)`` instead and records both levels (:func:`resolve_conformal_alpha`).
    """
    cuts = _require_split(calibration_cutouts, "calibration", "freeze_calibration")
    sys_arcsec = _arcsec(locked_sigma_sys, "locked_sigma_sys")
    geom = Geometry.for_pixel_scale(cuts[0].pixel_scale)
    stack = stack_empirical_psf(cuts, geom)
    samples = collect_noise_samples(cuts, geom)
    psf, noise = stack.psf(), samples.model()
    cal: dict = {
        k: []
        for k in (
            "star_ids",
            "patch_ids",
            "w3snr",
            "centroid_offset_arcsec",
            "delta_chi2_scalar",
            "delta_chi2_gls",
            "delta_chi2_scalar_loo",
            "delta_chi2_gls_loo",
            "fetched_at_utc",
        )
    }
    for c in cuts:
        if cutout_problem(c, geom) is not None:
            continue
        inside = _StarScreens(c, psf, noise, geom, sys_arcsec)
        psf_loo = stack.psf(leave_out=c.star_id) if c.star_id in stack.star_ids else psf
        outside = _StarScreens(c, psf_loo, samples.model(leave_out=c.star_id), geom, sys_arcsec)
        cal["star_ids"].append(c.star_id)
        cal["patch_ids"].append(c.patch_id)
        cal["w3snr"].append(float(c.w3snr))
        cal["centroid_offset_arcsec"].append(inside.offset_arcsec(c.flux))
        cal["delta_chi2_scalar"].append(inside.scalar.delta_chi2(c.flux))
        cal["delta_chi2_gls"].append(inside.gls.delta_chi2(c.flux))
        cal["delta_chi2_scalar_loo"].append(outside.scalar.delta_chi2(c.flux))
        cal["delta_chi2_gls_loo"].append(outside.gls.delta_chi2(c.flux))
        cal["fetched_at_utc"].append(c.fetched_at_utc)
    level = resolve_conformal_alpha(
        float(conformal_alpha), len(cal["star_ids"]), allow_fallback=allow_conformal_fallback
    )
    state = FrozenCalibration(
        geom,
        psf,
        noise,
        sys_arcsec,
        float(target_false_alarm),
        level,
        cal,
        conformal_alpha_requested=float(conformal_alpha),
    )
    state.rules()  # raises now if alpha is unattainable or a null median is not positive
    return state


def assert_withheld_after_freeze(frozen_at_utc: str | None, fetch_times: Iterable) -> None:
    """Every withheld fetch must be strictly later than the freeze (else ``LeakageError``)."""
    if frozen_at_utc is None:
        raise LeakageError("calibration state is not sealed: nothing may touch withheld data yet")
    try:
        at = parse_utc(frozen_at_utc, "frozen_at_utc")
        times = [parse_utc(t, "withheld fetched_at_utc") for t in fetch_times]
    except ValueError as exc:
        raise LeakageError(f"cannot prove the withheld fetch followed the freeze: {exc}") from exc
    early = [t for t in times if t <= at]
    if early:
        raise LeakageError(
            f"{len(early)} withheld cutout(s) were fetched at or before the freeze "
            f"({min(early).isoformat()} <= {at.isoformat()})"
        )


# ------------------------------------------------------------ withheld injection
@dataclass(frozen=True, eq=False)
class WithheldResult:
    """Per-star statistics of the frozen screens on the withheld two-arm injection."""

    star_ids: tuple[str, ...]
    patch_ids: tuple[str, ...]
    fetched_at_utc: tuple[str, ...]
    w3snr: np.ndarray
    reference_flux: np.ndarray
    separations_arcsec: np.ndarray
    flux_fractions: np.ndarray
    azimuth_rad: np.ndarray  # (n_star, n_sep, n_frac)
    stats: dict  # arm -> statistic -> array
    excluded: tuple[dict, ...]
    rng_seed: int
    frozen: FrozenCalibration

    def summary(self) -> dict:
        screens = self.frozen.rules()["screens"]
        flags = flags_from_statistics(self.stats, self.w3snr, self.flux_fractions, screens)
        return summarize(flags, self.separations_arcsec, self.flux_fractions)

    def as_dict(self) -> dict:
        return {
            "rng_seed": self.rng_seed,
            "rng": f"SeedSequence([{self.rng_seed}, 7]) azimuths, shape (star, sep, frac)",
            "n_stars": len(self.star_ids),
            "star_ids": list(self.star_ids),
            "patch_ids": list(self.patch_ids),
            "fetched_at_utc": list(self.fetched_at_utc),
            "w3snr": self.w3snr.tolist(),
            "reference_flux": self.reference_flux.tolist(),
            "separations_arcsec": self.separations_arcsec.tolist(),
            "flux_fractions": self.flux_fractions.tolist(),
            "injection": {
                "template": "empirical PSF (calibration stars only), cubic spline",
                "background_flux": "F_bg = F_star f_bg / (1 - f_bg); F_star = PSF-fit flux of "
                "the uninjected star at its windowed centroid",
                "arm_a": "off-center blend at separation theta from the star's measured image "
                "centroid, seeded azimuth",
                "arm_b": "matched centered excess: identical F_bg at theta = 0 (the star's "
                "measured image centroid)",
                "noise": "none added (W3 is background-limited; source shot noise neglected)",
                "azimuth_rad": self.azimuth_rad.tolist(),
            },
            "statistics": {
                arm: {k: v.tolist() for k, v in block.items()} for arm, block in self.stats.items()
            },
            "excluded": list(self.excluded),
            "summary": self.summary(),
        }


def evaluate_withheld(
    withheld_cutouts: Sequence[UnwiseCutout],
    frozen: FrozenCalibration,
    seed: int = 42,
    *,
    separations: u.Quantity | None = None,
    flux_fractions: Sequence[float] = FLUX_FRACTIONS,
) -> WithheldResult:
    """Two-arm (off-center blend vs matched centered excess) injection on withheld stars.

    Refuses (``LeakageError``): an unsealed ``frozen`` state, a non-withheld cutout, a star
    shared with the calibration set, a withheld cutout without a fetch timestamp or with one
    at or before the freeze.
    """
    cuts = _require_split(withheld_cutouts, "withheld", "evaluate_withheld")
    assert_withheld_after_freeze(frozen.frozen_at_utc, [c.fetched_at_utc for c in cuts])
    used = set(frozen.star_ids) | set(frozen.psf.star_ids) | set(frozen.noise.star_ids)
    if {c.star_id for c in cuts} & used:
        raise LeakageError("withheld stars overlap the calibration stars behind the frozen method")
    if round(cuts[0].pix, 9) != round(frozen.geometry.pixel_scale_arcsec, 9):
        raise UnitValidationError("withheld cutouts are not on the frozen pixel scale")
    if not isinstance(seed, int | np.integer):
        raise TypeError("seed must be an explicit integer")
    if separations is None:
        separations = np.array(SEPARATIONS_ARCSEC) * u.arcsec
    if not isinstance(separations, u.Quantity) or not separations.unit.is_equivalent(u.arcsec):
        raise UnitValidationError("separations must be an angle Quantity")
    seps = np.atleast_1d(np.asarray(separations.to_value(u.arcsec), float))
    fracs = np.atleast_1d(np.asarray(flux_fractions, float))
    if np.any(seps <= 0) or np.any((fracs <= 0) | (fracs >= 1)):
        raise ValueError("separations must be > 0 and flux fractions in (0, 1)")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7]))
    azim = rng.uniform(0.0, 2 * np.pi, (len(cuts), len(seps), len(fracs)))

    geom, psf = frozen.geometry, frozen.psf
    keep, excluded, snr, fref, rows = [], [], [], [], []
    for si, c in enumerate(cuts):
        why = cutout_problem(c, geom)
        if why is None:
            scr = _StarScreens(c, psf, frozen.noise, geom, frozen.locked_sigma_sys_arcsec)
            cx, cy = scr.centroid(c.flux)
            f_ref = scr.psf_flux(c.flux, cx, cy)
            ap_ref = scr.aperture_flux(c.flux)
            if not f_ref > 0:
                why = "non-positive PSF-fit flux of the uninjected star"
        if why is not None:
            excluded.append({"gaia_dr3_source_id": c.star_id, "reason": why})
            continue
        null = scr.stats(c.flux, f_ref, ap_ref)
        a = np.empty((len(seps), len(fracs), len(STATISTICS)))
        b = np.empty((len(fracs), len(STATISTICS)))
        for j, f in enumerate(fracs):
            f_bg = f_ref * f / (1.0 - f)
            b[j] = scr.stats(c.flux + f_bg * psf.render(c.flux.shape, cx, cy), f_ref, ap_ref)
            for i, th in enumerate(seps):
                tp = th / c.pix
                px = cx + tp * np.cos(azim[si, i, j])
                py = cy + tp * np.sin(azim[si, i, j])
                img = c.flux + f_bg * psf.render(c.flux.shape, px, py)
                a[i, j] = scr.stats(img, f_ref, ap_ref)
        keep.append(si)
        snr.append(float(c.w3snr))
        fref.append(f_ref)
        rows.append((null, a, b))
    if not keep:
        raise ValueError("no usable withheld cutouts")
    stats = {
        "null": {n: np.array([r[0][k] for r in rows]) for k, n in enumerate(STATISTICS)},
        "arm_a": {n: np.stack([r[1][..., k] for r in rows]) for k, n in enumerate(STATISTICS)},
        "arm_b": {n: np.stack([r[2][..., k] for r in rows]) for k, n in enumerate(STATISTICS)},
    }
    return WithheldResult(
        tuple(cuts[i].star_id for i in keep),
        tuple(cuts[i].patch_id for i in keep),
        tuple(str(cuts[i].fetched_at_utc) for i in keep),
        np.array(snr),
        np.array(fref),
        seps,
        fracs,
        azim[keep],
        stats,
        tuple(excluded),
        int(seed),
        frozen,
    )


# ------------------------------------------- flags / summaries (offline recomputable)
def flags_from_statistics(
    stats: Mapping, w3snr: Sequence[float], flux_fractions: Sequence[float], screens: Mapping
) -> dict:
    """Screen flags from stored statistics and the frozen thresholds (pure, dict inputs)."""
    snr = np.asarray(w3snr, float)
    fr = np.asarray(flux_fractions, float)
    snr_eff = {
        "null": snr,
        "arm_a": snr[:, None, None] / (1.0 - fr)[None, None, :],
        "arm_b": snr[:, None] / (1.0 - fr)[None, :],
    }
    out: dict = {s: {} for s in SCREENS}
    for arm, block in stats.items():
        off = np.asarray(block["centroid_offset_arcsec"], float)
        for name in CENTROID_SCREENS:
            sc = screens[name]
            thr = hf.centroid_threshold_arcsec(
                snr_eff[arm], float(sc["sigma_sys_arcsec"]), float(sc["target_false_alarm"])
            )
            out[name][arm] = off > thr
        for name in (*CHI2_SCREENS, *CONFORMAL_SCREENS):
            sc = screens[name]
            out[name][arm] = np.asarray(block[sc["statistic"]], float) > float(sc["threshold"])
        out["naive_flux_excess"][arm] = hf.FrozenScreens.naive_flags(
            block["flux_ratio"], snr[(...,) + (None,) * (off.ndim - 1)]
        )
    return out


def summarize(flags: Mapping, separations_arcsec: Sequence[float], flux_fractions) -> dict:
    """Recovery / false-rejection rates with exact 95% intervals for every screen."""
    seps = np.asarray(separations_arcsec, float)
    out: dict = {
        "survival_max_separation_arcsec": SURVIVAL_MAX_SEPARATION_ARCSEC,
        "survival_weighting": "p(theta) ~ theta for theta <= 3.25 arcsec, uniform f_bg",
        "flux_fractions": [float(f) for f in np.asarray(flux_fractions, float)],
        "screens": {},
    }
    for s in SCREENS:
        fa = np.asarray(flags[s]["arm_a"], bool)
        fb = np.asarray(flags[s]["arm_b"], bool)
        n = fa.shape[0]
        rec = fa.mean(axis=0)
        cp_a = [[clopper_pearson(int(k), n) for k in row] for row in fa.sum(axis=0)]
        cp_b = [clopper_pearson(int(k), n) for k in fb.sum(axis=0)]
        out["screens"][s] = {
            "role": SCREEN_ROLES[s],
            "n_stars": n,
            "null_false_alarm": _rate(flags[s]["null"]),
            "arm_a_blend_recovery": rec.tolist(),
            "arm_a_k": fa.sum(axis=0).tolist(),
            "arm_a_clopper_pearson_95": [[list(x) for x in row] for row in cp_a],
            "arm_b_false_rejection": fb.mean(axis=0).tolist(),
            "arm_b_k": fb.sum(axis=0).tolist(),
            "arm_b_clopper_pearson_95": [list(x) for x in cp_b],
            "arm_b_false_rejection_pooled": _rate(fb),
            "blend_survival_within_3p25_arcsec": hf.area_weighted_survival(
                rec, seps, SURVIVAL_MAX_SEPARATION_ARCSEC
            ),
            "centered_excess_retained_pooled": 1.0 - float(fb.mean()),
        }
    return out


# ----------------------------------------------------- promoted payload & recompute
def gate_payload(frozen: FrozenCalibration, withheld: WithheldResult) -> dict:
    """Scientific payload stored in the receipt / promoted artifact."""
    if withheld.frozen is not frozen and withheld.frozen.state_sha256 != frozen.state_sha256:
        raise LeakageError("withheld result was evaluated under a different frozen state")
    return {
        **frozen.payload(),
        "frozen_at_utc": frozen.frozen_at_utc,
        "frozen_state_sha256": frozen.state_sha256,
        "withheld": withheld.as_dict(),
    }


def frozen_state_sha256(payload: Mapping) -> str:
    """Re-derive the frozen-state fingerprint from stored blocks."""
    keys = ("rule_version", "geometry", "empirical_psf", "noise_model", "frozen_screens")
    return canonical_sha256({k: payload[k] for k in keys})


def recompute_unwise_gate(payload: Mapping) -> dict:
    """Offline re-derivation of every stored summary from the stored per-star arrays.

    Returns the recomputed PSF widths, noise-covariance diagnostics, frozen thresholds,
    calibration in-sample false alarms, withheld summary and frozen-state fingerprint;
    callers compare them with the stored values. Raises ``LeakageError`` if the stored
    timestamps or star sets break the freeze ordering.
    """
    p, nm, fz, wh = (
        payload["empirical_psf"],
        payload["noise_model"],
        payload["frozen_screens"],
        payload["withheld"],
    )
    cal = fz["calibration"]
    stamp = np.asarray(p["stamp"], float)
    pix = float(p["pixel_scale_arcsec"])
    geom = Geometry.for_pixel_scale(pix * u.arcsec)
    try:
        at = parse_utc(payload["frozen_at_utc"], "frozen_at_utc")
        cal_times = [parse_utc(t, "calibration fetched_at_utc") for t in cal["fetched_at_utc"]]
    except ValueError as exc:
        raise LeakageError(f"cannot order the freeze against its inputs: {exc}") from exc
    if cal_times and max(cal_times) > at:
        raise LeakageError("frozen_at_utc precedes a calibration cutout: not the freeze time")
    assert_withheld_after_freeze(payload["frozen_at_utc"], wh["fetched_at_utc"])
    cal_ids, wh_ids = set(cal["star_ids"]), set(wh["star_ids"])
    if cal_ids & wh_ids or not set(p["star_ids"]) <= cal_ids or not set(nm["star_ids"]) <= cal_ids:
        raise LeakageError("PSF / covariance / threshold stars are not a calibration-only set")
    for sid in (*cal["star_ids"], *wh["star_ids"], *p["star_ids"], *nm["star_ids"]):
        check_gaia_id(sid)
    torus = np.asarray(nm["torus_covariance"], float)
    regular = regularise_lag_covariance(
        np.asarray(nm["raw_lag_covariance"], float),
        int(nm["n_torus"]),
        float(nm["shrinkage"]),
        float(nm["power_floor"]),
    )
    rules = frozen_screen_rules(
        cal,
        float(fz["locked_sigma_sys_arcsec"]),
        float(fz["target_false_alarm"]),
        float(fz["conformal_alpha_nominal"]),
    )
    flags = flags_from_statistics(
        wh["statistics"], wh["w3snr"], wh["flux_fractions"], rules["screens"]
    )
    return {
        "geometry": geom.as_dict(),
        "stamp_sum": float(stamp.sum()),
        **psf_widths_arcsec(stamp, pix),
        "torus_covariance_max_abs_diff": float(np.max(np.abs(regular - torus))),
        "variance_inflation": float(torus[0, 0]),
        "minimum_eigenvalue_bound": minimum_power(torus),
        "minimum_eigenvalue_required": float(nm["shrinkage"]) * float(torus[0, 0]),
        "fresh_sigma_sys": rules["fresh_sigma_sys"],
        "screens": rules["screens"],
        "calibration_false_alarm_in_sample": calibration_false_alarms(cal, rules["screens"]),
        "summary": summarize(flags, wh["separations_arcsec"], wh["flux_fractions"]),
        "frozen_state_sha256": frozen_state_sha256(payload),
    }


# -------------------------------------------- fresh sample: patches, cuts, disjointness
def _check_angle(q: object, name: str) -> float:
    return _deg(q, name)


def patch_separation_audit(
    patches: Sequence[Mapping],
    locked_centres: SkyCoord,
    *,
    min_abs_b: u.Quantity = 30.0 * u.deg,
    min_locked_separation: u.Quantity = 15.0 * u.deg,
) -> dict:
    """Re-check fresh patches against the locked Gate #1 centres (``LeakageError`` on overlap).

    Each patch is a mapping with ``l_deg``, ``b_deg`` and ``radius_deg`` (Galactic, degrees).
    Requires the whole cone at ``|b| > min_abs_b``, the centre at least
    ``min_locked_separation`` from every locked centre, and the cone clear of LMC/SMC/M31.
    """
    b_min = _check_angle(min_abs_b, "min_abs_b")
    sep_min = _check_angle(min_locked_separation, "min_locked_separation")
    if not isinstance(locked_centres, SkyCoord) or locked_centres.isscalar:
        raise UnitValidationError("locked_centres must be a SkyCoord array")
    nearest = []
    for p in patches:
        c = SkyCoord(l=float(p["l_deg"]) * u.deg, b=float(p["b_deg"]) * u.deg, frame="galactic")
        radius = float(p["radius_deg"])
        pid = p.get("patch_id", "?")
        if abs(float(p["b_deg"])) - radius < b_min:
            raise SchemaValidationError(f"patch {pid} cone reaches |b| < {b_min:g} deg")
        sep = float(c.separation(locked_centres).deg.min())
        if sep < sep_min:
            raise LeakageError(
                f"patch {pid} lies {sep:.2f} deg from a locked Gate #1 patch (< {sep_min:g} deg)"
            )
        for name, al, ab, ar in AVOID_FIELDS:
            away = float(c.separation(SkyCoord(l=al * u.deg, b=ab * u.deg, frame="galactic")).deg)
            if away < ar + radius:
                raise SchemaValidationError(f"patch {pid} overlaps the excluded field {name}")
        nearest.append(sep)
    return {
        "n_patches": len(nearest),
        "n_locked_patches": len(locked_centres),
        "min_abs_b_deg": b_min,
        "min_locked_separation_required_deg": sep_min,
        "min_locked_separation_deg": min(nearest) if nearest else None,
        "avoided_fields": [n for n, *_ in AVOID_FIELDS],
    }


def choose_fresh_patches(
    seed: int,
    locked_centres: SkyCoord,
    *,
    n_patches: int = 16,
    radius: u.Quantity = 2.0 * u.deg,
    min_abs_b: u.Quantity = 30.0 * u.deg,
    min_locked_separation: u.Quantity = 15.0 * u.deg,
    min_patch_separation: u.Quantity = 20.0 * u.deg,
    max_draws: int = 200_000,
) -> tuple[list[dict], int]:
    """Seeded isotropic patch centres, disjoint from the locked Gate #1 patches.

    A draw is kept when its whole cone lies at ``|b| > min_abs_b``, clear of LMC/SMC/M31, at
    least ``min_locked_separation`` (centre to centre) from every locked patch and
    ``min_patch_separation`` from every patch already kept. Returns (patches, draws used).
    """
    if not isinstance(seed, int | np.integer):
        raise TypeError("seed must be an explicit integer")
    r_deg = _check_angle(radius, "radius")
    b_min = _check_angle(min_abs_b, "min_abs_b") + r_deg
    lock_min = _check_angle(min_locked_separation, "min_locked_separation")
    sep_min = _check_angle(min_patch_separation, "min_patch_separation")
    if not isinstance(locked_centres, SkyCoord) or locked_centres.isscalar:
        raise UnitValidationError("locked_centres must be a SkyCoord array")
    avoid = [
        (SkyCoord(l=al * u.deg, b=ab * u.deg, frame="galactic"), ar)
        for _, al, ab, ar in AVOID_FIELDS
    ]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), PATCH_STREAM]))
    centres: list[SkyCoord] = []
    out: list[dict] = []
    draws = 0
    while len(out) < n_patches and draws < max_draws:
        draws += 1
        l_deg = float(rng.uniform(0.0, 360.0))
        b_deg = float(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0))))
        if abs(b_deg) <= b_min:
            continue
        c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
        if any(c.separation(a).deg < ar + r_deg for a, ar in avoid):
            continue
        locked_sep = c.separation(locked_centres).deg
        if float(locked_sep.min()) < lock_min:
            continue
        if any(c.separation(o).deg < sep_min for o in centres):
            continue
        centres.append(c)
        icrs = c.icrs
        out.append(
            {
                "patch_id": f"U{len(out) + 1:02d}",
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": float(icrs.ra.deg),
                "dec_deg": float(icrs.dec.deg),
                "radius_deg": r_deg,
                "nearest_locked_patch_index": int(np.argmin(locked_sep)),
                "nearest_locked_patch_separation_deg": float(locked_sep.min()),
            }
        )
    if len(out) < n_patches:
        raise RuntimeError(f"only {len(out)} patches satisfy the separation constraints")
    return out, draws


def assert_disjoint_from_locked(fresh_ids: Iterable, locked_ids: Iterable) -> int:
    """Zero Gaia DR3 id overlap with the locked Gate #1 / Gate #2 sample (else ``LeakageError``).

    ``fresh_ids`` must be decimal strings (ruling v0.2.1-R4); ``locked_ids`` may be the
    integers the locked v0.1.x file stores. Returns the number of fresh ids checked.
    """
    fresh = [check_gaia_id(s) for s in fresh_ids]
    if len(set(fresh)) != len(fresh):
        raise SchemaValidationError("duplicate gaia_dr3_source_id in the fresh sample")
    shared = sorted(set(fresh) & {str(int(s)) for s in locked_ids}, key=int)
    if shared:
        raise LeakageError(
            f"{len(shared)} star(s) of the locked Gate #1 / Gate #2 sample appear in the fresh "
            f"split (e.g. Gaia DR3 {shared[0]})"
        )
    return len(fresh)


@dataclass(frozen=True)
class ControlCuts:
    """The Gate #1 isolation / quality cuts, with the |b| floor raised to 30 deg."""

    min_abs_b: u.Quantity = 30.0 * u.deg
    ruwe_max: float = 1.4
    w3snr_min: float = 8.0
    w3snr_max: float = 20.0
    isolation: u.Quantity = 15.0 * u.arcsec

    def __post_init__(self) -> None:
        _check_angle(self.min_abs_b, "min_abs_b")
        _arcsec(self.isolation, "isolation")

    def steps(self) -> tuple[tuple[str, str], ...]:
        """(counter key, human-readable cut) in the order the cuts are applied."""
        b = self.min_abs_b.to_value(u.deg)
        iso = self.isolation.to_value(u.arcsec)
        return (
            ("after_star_abs_b_cut", f"star-level |b| > {b:g} deg"),
            ("after_ruwe_cut", f"Gaia RUWE < {self.ruwe_max:g} (unresolved-binary rejection)"),
            (
                "after_w3_snr_band",
                f"AllWISE {self.w3snr_min:g} <= w3snr <= {self.w3snr_max:g} (drill-down regime)",
            ),
            ("after_cc_flags_0000", "AllWISE cc_flags == '0000'"),
            ("after_ext_flg_0", "AllWISE ext_flg == 0"),
            ("after_isolation", f"no other Gaia DR3 source within {iso:g} arcsec"),
        )

    def passes(self, row: Mapping) -> tuple[bool, ...]:
        """One boolean per step; the neighbour count may be None (not yet queried)."""
        nn = row.get("n_gaia_neighbours")
        return (
            abs(float(row["b_deg"])) > self.min_abs_b.to_value(u.deg),
            float(row["ruwe"]) < self.ruwe_max,
            self.w3snr_min <= float(row["w3snr"]) <= self.w3snr_max,
            str(row["cc_flags"]) == "0000",
            int(row["ext_flg"]) == 0,
            nn is not None and int(nn) == 0,
        )


def apply_control_cuts(rows: Sequence[Mapping], cuts: ControlCuts) -> tuple[list[dict], dict]:
    """Apply the cuts in order; returns (surviving rows, counts after each cut)."""
    keys = [k for k, _ in cuts.steps()]
    counts = {"candidates": len(rows), **dict.fromkeys(keys, 0)}
    kept = []
    for row in rows:
        check_gaia_id(row["gaia_dr3_source_id"])
        ok = cuts.passes(row)
        for i, key in enumerate(keys):
            if not all(ok[: i + 1]):
                break
            counts[key] += 1
        if all(ok):
            kept.append(dict(row))
    return kept, counts


def split_calibration_withheld(
    controls: Sequence[Mapping], patch_order: Sequence[str], seed: int
) -> dict[str, str]:
    """Stratified-by-patch, seeded 50/50 split (the Gate #1 rule): id -> split label.

    Within each patch (in ``patch_order``) the stars are sorted by id, permuted with
    ``SeedSequence([seed, 2])`` and labelled alternately across the global order.
    """
    if not isinstance(seed, int | np.integer):
        raise TypeError("seed must be an explicit integer")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
    out: dict[str, str] = {}
    toggle = 0
    for pid in patch_order:
        mine = sorted(
            (check_gaia_id(c["gaia_dr3_source_id"]) for c in controls if c["patch_id"] == pid),
            key=int,
        )
        for i in rng.permutation(len(mine)):
            out[mine[i]] = "calibration" if toggle % 2 == 0 else "withheld"
            toggle += 1
    if len(out) != len(controls):
        raise ValueError("a control belongs to no declared patch, or an id is duplicated")
    return out


# ------------------------------------------------- SYNTHETIC rehearsal fixtures only
def synthetic_cutout(
    rng: np.random.Generator,
    gaia_dr3_source_id: str,
    split: str,
    *,
    psf_fwhm: u.Quantity = NOMINAL_W3_FWHM_ARCSEC * u.arcsec,
    noise_correlation_fwhm: u.Quantity = 0.0 * u.arcsec,
    snr: float | None = None,
    sigma_sys: u.Quantity = 0.4356 * u.arcsec,
    shape: tuple[int, int] = (45, 45),
    pixel_scale: u.Quantity = UNWISE_PIXEL_SCALE,
    patch_id: str = "SYN",
    fetched_at_utc: str | None = None,
) -> UnwiseCutout:
    """SYNTHETIC (``synthetic_proxy``) cutout: Gaussian PSF, optionally correlated noise.

    Never archival data. The noise is white Gaussian noise smoothed with a Gaussian kernel of
    FWHM ``noise_correlation_fwhm`` and rescaled so that every pixel keeps the standard
    deviation the uncertainty map states: the map is right pixel by pixel and silent about
    the correlation, which is exactly the situation the scalar-variance screen mishandles.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    pix = _arcsec(pixel_scale, "pixel_scale")
    s = _arcsec(psf_fwhm, "psf_fwhm") / FWHM_PER_SIGMA / pix
    k = _arcsec(noise_correlation_fwhm, "noise_correlation_fwhm") / FWHM_PER_SIGMA / pix
    sys_pix = _arcsec(sigma_sys, "sigma_sys") / pix
    snr = float(rng.uniform(8.0, 20.0)) if snr is None else float(snr)
    gx = (shape[1] - 1) / 2 + rng.uniform(-0.5, 0.5)
    gy = (shape[0] - 1) / 2 + rng.uniform(-0.5, 0.5)
    tx, ty = gx + rng.normal(0, sys_pix), gy + rng.normal(0, sys_pix)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    flux = 1000.0
    sig = flux / (snr * np.sqrt(4 * np.pi) * s)
    star = flux * np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)
    noise = rng.normal(0.0, 1.0, shape)
    if k > 0:
        half = int(np.ceil(4 * k))
        kern = np.exp(-0.5 * (np.arange(-half, half + 1) / k) ** 2)
        kern /= kern.sum()
        for axis in (0, 1):
            noise = ndimage.convolve1d(noise, kern, axis=axis, mode="wrap")
        noise /= float(np.sum(kern**2))  # back to unit variance per pixel
    img = 485.0 + star + sig * noise
    return UnwiseCutout(
        gaia_dr3_source_id,
        split,
        img,
        np.full(shape, sig),
        (gx, gy),
        pixel_scale,
        snr,
        patch_id,
        fetched_at_utc,
    )
