"""Public-Launch Gate #2: frozen-method blend screens on real AllWISE W3 Atlas cutouts.

Pure kernels (no I/O, no network) operating on 2-D W3 intensity + uncertainty cutouts.
The live archive fetch lives in ``scripts/run_wise_fits_gate2.py``; the promoted
summary is re-verified offline by ``registry.load_fits_gate2``.

Method (frozen on the calibration half before any withheld cutout is evaluated):

1. **Empirical PSF** (:func:`build_empirical_psf`). Each *calibration* stamp is sky
   subtracted (median of pixels beyond ``SKY_INNER_PIX``), centred on its windowed W3
   centroid by cubic-spline resampling, normalised to unit sum, and the stack is
   median-combined. The result therefore carries the real Atlas coadd smoothing, not
   the 6.5 arcsec single-frame Gaussian of the synthetic drill-down.
2. **Frozen screens** (:func:`calibrate_frozen_screens`).
   * Centroid screen: the Gate #1 windowed-centroid offset from the Gaia position,
     flagged above ``sqrt(sigma_stat^2 + sigma_sys^2) * sqrt(-2 ln FA)`` with the
     *locked* ``sigma_sys`` and ``sigma_stat = 6.5" / (2.355 SNR)`` (Ren et al. 2026 eq. 1).
   * Image screen: profile-likelihood Delta chi^2 between a one-PSF model (position
     prior N(Gaia, sigma_sys)) and a two-PSF model (second flux >= 0), both built from
     the empirical PSF with the local sky projected out. Atlas pixels are correlated by
     the coadd kernel, so nominal chi^2(3) is not calibrated; the threshold is
     ``s * chi2.isf(FA, 3)`` with ``s = median(Delta chi^2_cal) / median(chi2(3))``
     (median matching, 50% breakdown, the same robustness choice as ``sigma_sys``).
   * ``naive_flux_excess`` (adversarial reference rule, not a proposed screen): flags
     any aperture-flux excess over the uninjected star above 3 / SNR of the star flux. It is the "reject every bright excess" rule
     the matched centered-excess control exists to expose.
3. **Withheld two-arm injection** (:func:`evaluate_withheld_matched_injection`). Per
   withheld star and (theta, f_bg) cell, the empirical PSF carrying
   ``F_bg = F_star f_bg / (1 - f_bg)`` is added to the real cutout:
   * Arm A (off-center blend): at separation theta, seeded azimuth;
   * Arm B (matched centered excess): the identical flux at theta = 0, i.e. on the
     star's own measured image centroid (where a genuine circumstellar excess lands).
   Both arms reuse the same withheld cutouts and the frozen thresholds. Injected images
   are screened with ``SNR_eff = SNR / (1 - f_bg)`` (the catalog sees the brighter blend).

Every rate carries an exact Clopper-Pearson 95% interval. Arm B false rejections are
genuine centered excesses a screen would throw away; a screen that "recovers" every
blend by also rejecting every centered excess has learned nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares, minimize
from scipy.stats import chi2

from fermi_desk.errors import LeakageError, UnitValidationError
from fermi_desk.models.hephaistos import (
    FWHM_PER_SIGMA,
    RAYLEIGH_3SIGMA_FALSE_ALARM,
    clopper_pearson,
)

RULE_VERSION = "gate2-v1"
NOMINAL_W3_FWHM_ARCSEC = 6.5  # Ren et al. 2026 Table 2 single-frame W3 FWHM
SEPARATIONS_ARCSEC = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)  # same axes as the synthetic grid
FLUX_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
SURVIVAL_MAX_SEPARATION_ARCSEC = 3.25
STAMP_HALF_PIX = 15  # 31 x 31 PSF stamp (42.6 arcsec at 1.375 arcsec / pixel)
SKY_INNER_PIX = 16.0  # sky = median of cutout pixels beyond this radius
FIT_RADIUS_PIX = 12.0  # image-screen fit region (16.5 arcsec)
PRIMARY_RANGE_PIX = 2.0  # primary position search half-width around Gaia
SECONDARY_MAX_PIX = 8.0  # secondary position search radius around Gaia
CHI2_DOF = 3
NAIVE_EXCESS_SIGMA = 3.0
SCREENS = ("centroid", "image", "naive_flux_excess")
SPLITS = ("calibration", "withheld")


def _arcsec(q: object, name: str) -> float:
    if not isinstance(q, u.Quantity) or not q.unit.is_equivalent(u.arcsec):
        raise UnitValidationError(f"{name} must be an angle Quantity")
    return float(q.to_value(u.arcsec))


def _arcsec_values(q: object, name: str) -> np.ndarray:
    if not isinstance(q, u.Quantity) or not q.unit.is_equivalent(u.arcsec):
        raise UnitValidationError(f"{name} must be an angle Quantity")
    return np.asarray(q.to_value(u.arcsec), float)


# ---------------------------------------------------------------------- cutouts
@dataclass(frozen=True, eq=False)
class W3Cutout:
    """One AllWISE W3 Atlas intensity + uncertainty cutout of a control star.

    ``gaia_xy`` is the 0-based (x, y) pixel position of the Gaia DR3 source propagated
    to the AllWISE epoch; ``flux`` and ``unc`` are in DN (Atlas BUNIT).
    """

    star_id: int
    split: str
    flux: np.ndarray
    unc: np.ndarray
    gaia_xy: tuple[float, float]
    pixel_scale: u.Quantity
    w3snr: float
    patch_id: str = ""

    def __post_init__(self) -> None:
        flux = np.asarray(self.flux, float)
        unc = np.asarray(self.unc, float)
        if flux.ndim != 2 or flux.shape != unc.shape:
            raise ValueError(f"star {self.star_id}: flux / unc must be 2-D and the same shape")
        if self.split not in SPLITS:
            raise ValueError(f"star {self.star_id}: split must be one of {SPLITS}")
        if not self.w3snr > 0:
            raise ValueError(f"star {self.star_id}: w3snr must be > 0")
        if _arcsec(self.pixel_scale, "pixel_scale") <= 0:
            raise ValueError("pixel_scale must be positive")
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "unc", unc)
        object.__setattr__(self, "gaia_xy", (float(self.gaia_xy[0]), float(self.gaia_xy[1])))

    @property
    def pix(self) -> float:
        return _arcsec(self.pixel_scale, "pixel_scale")

    def scaled(self, factor: float) -> W3Cutout:
        """Same cutout with flux and uncertainty multiplied by ``factor`` (unit change)."""
        return W3Cutout(
            self.star_id,
            self.split,
            self.flux * factor,
            self.unc * factor,
            self.gaia_xy,
            self.pixel_scale,
            self.w3snr,
            self.patch_id,
        )


def _require_split(cutouts: Sequence[W3Cutout], split: str, step: str) -> list[W3Cutout]:
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
    pix = {round(c.pix, 9) for c in cutouts}
    if len(pix) != 1:
        raise UnitValidationError(f"{step}: cutouts disagree on pixel scale")
    return sorted(cutouts, key=lambda c: c.star_id)


def _radius(shape: tuple[int, int], x0: float, y0: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    return np.hypot(xx - x0, yy - y0)


def nominal_window_sigma_pix(pixel_scale: u.Quantity) -> float:
    return NOMINAL_W3_FWHM_ARCSEC / FWHM_PER_SIGMA / _arcsec(pixel_scale, "pixel_scale")


def windowed_centroid(
    img: np.ndarray, x0: float, y0: float, s_pix: float, n_iter: int = 20
) -> tuple[float, float]:
    """Gaussian-windowed (SExtractor-style) centroid: the Gate #1 estimator, unchanged."""
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(float)
    data = np.nan_to_num(img - np.nanmedian(img))
    x, y = x0, y0
    for _ in range(n_iter):
        w = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s_pix**2)) * data
        norm = w.sum()
        if abs(norm) < 1e-12:
            break
        dx = 2 * (w * (xx - x)).sum() / norm
        dy = 2 * (w * (yy - y)).sum() / norm
        x, y = x + float(np.clip(dx, -1, 1)), y + float(np.clip(dy, -1, 1))
    return x, y


def sky_level(img: np.ndarray, x0: float, y0: float, inner_pix: float = SKY_INNER_PIX) -> float:
    sel = (_radius(img.shape, x0, y0) >= inner_pix) & np.isfinite(img)
    if sel.sum() < 50:
        raise ValueError("too few sky pixels beyond the inner sky radius")
    return float(np.median(img[sel]))


def cutout_problem(c: W3Cutout) -> str | None:
    """Reason a cutout cannot be used (truncated / non-finite near the star), else None."""
    gx, gy = c.gaia_xy
    need = max(STAMP_HALF_PIX + 3, FIT_RADIUS_PIX + 1)
    ny, nx = c.flux.shape
    if gx - need < 0 or gy - need < 0 or gx + need > nx - 1 or gy + need > ny - 1:
        return f"cutout {c.flux.shape} too small around the star (needs {need:.0f} px)"
    core = _radius(c.flux.shape, gx, gy) <= need
    if not (np.all(np.isfinite(c.flux[core])) and np.all(np.isfinite(c.unc[core]))):
        return "non-finite flux / unc pixels near the star"
    if not np.all(c.unc[core] > 0):
        return "non-positive uncertainty pixels near the star"
    return None


# ------------------------------------------------------------------ empirical PSF
def _gaussian_fwhm_pix(stamp: np.ndarray) -> float:
    h = (stamp.shape[0] - 1) / 2.0
    yy, xx = np.mgrid[0 : stamp.shape[0], 0 : stamp.shape[1]].astype(float) - h

    def resid(p):
        a, s, x0, y0 = p
        return (a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s)) - stamp).ravel()

    p0 = [float(stamp.max()), 3.0, 0.0, 0.0]
    res = least_squares(resid, p0, x_scale=[float(stamp.max()), 1.0, 1.0, 1.0])
    return float(abs(res.x[1]) * FWHM_PER_SIGMA)


def _halfmax_fwhm_pix(stamp: np.ndarray) -> float:
    """Model-free FWHM: diameter of the circle with the area of the >= half-peak pixels."""
    n = int((stamp >= 0.5 * stamp.max()).sum())
    return float(2.0 * np.sqrt(n / np.pi))


@dataclass(frozen=True, eq=False)
class EmpiricalPSF:
    """Median-combined, unit-sum W3 Atlas PSF stamp built from calibration stars only."""

    stamp: np.ndarray
    pixel_scale: u.Quantity
    star_ids: tuple[int, ...]
    excluded: tuple[dict, ...] = ()
    _spline: RectBivariateSpline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        st = np.asarray(self.stamp, float)
        if st.ndim != 2 or st.shape[0] != st.shape[1] or st.shape[0] % 2 != 1:
            raise ValueError("PSF stamp must be square with odd size")
        if not np.isclose(st.sum(), 1.0, atol=1e-9):
            raise ValueError("PSF stamp must be normalised to unit sum")
        object.__setattr__(self, "stamp", st)
        h = self.half
        ax = np.arange(-h, h + 1, dtype=float)
        object.__setattr__(self, "_spline", RectBivariateSpline(ax, ax, st, kx=3, ky=3))

    @property
    def half(self) -> int:
        return (self.stamp.shape[0] - 1) // 2

    @property
    def pix(self) -> float:
        return _arcsec(self.pixel_scale, "pixel_scale")

    @property
    def fwhm_gaussian(self) -> u.Quantity:
        return _gaussian_fwhm_pix(self.stamp) * self.pix * u.arcsec

    @property
    def fwhm_halfmax(self) -> u.Quantity:
        return _halfmax_fwhm_pix(self.stamp) * self.pix * u.arcsec

    def evaluate(self, ys_rel: np.ndarray, xs_rel: np.ndarray) -> np.ndarray:
        """PSF on the outer grid (ys_rel x xs_rel), offsets from the PSF centre in pixels."""
        v = self._spline(ys_rel, xs_rel, grid=True)
        h = self.half
        v[np.abs(ys_rel) > h, :] = 0.0
        v[:, np.abs(xs_rel) > h] = 0.0
        return v

    def render(self, shape: tuple[int, int], x: float, y: float) -> np.ndarray:
        """Unit-flux PSF centred at pixel (x, y) on an image of ``shape``."""
        return self.evaluate(np.arange(shape[0]) - y, np.arange(shape[1]) - x)

    def as_dict(self) -> dict:
        return {
            "stamp": self.stamp.tolist(),
            "stamp_half_pix": self.half,
            "pixel_scale_arcsec": self.pix,
            "n_stars": len(self.star_ids),
            "star_ids": list(self.star_ids),
            "excluded": list(self.excluded),
            "fwhm_gaussian_arcsec": float(self.fwhm_gaussian.to_value(u.arcsec)),
            "fwhm_halfmax_arcsec": float(self.fwhm_halfmax.to_value(u.arcsec)),
            "nominal_fwhm_arcsec": NOMINAL_W3_FWHM_ARCSEC,
            "construction": (
                f"sky = median(r >= {SKY_INNER_PIX:g} px); Gate #1 windowed centroid "
                f'({NOMINAL_W3_FWHM_ARCSEC}" window); cubic-spline recentring; unit-sum '
                f"{2 * self.half + 1}x{2 * self.half + 1} stamps; pixel-wise median; renormalised"
            ),
        }


def build_empirical_psf(
    calibration_cutouts: Sequence[W3Cutout], *, stamp_half: int = STAMP_HALF_PIX
) -> EmpiricalPSF:
    """Empirical W3 PSF from the calibration split only (raises LeakageError otherwise)."""
    cuts = _require_split(calibration_cutouts, "calibration", "build_empirical_psf")
    pixel_scale = cuts[0].pixel_scale
    s_pix = nominal_window_sigma_pix(pixel_scale)
    ax = np.arange(-stamp_half, stamp_half + 1, dtype=float)
    stamps, used, excluded = [], [], []
    for c in cuts:
        why = cutout_problem(c)
        if why is None:
            gx, gy = c.gaia_xy
            img = c.flux - sky_level(c.flux, gx, gy)
            cx, cy = windowed_centroid(c.flux, gx, gy, s_pix)
            yy, xx = np.meshgrid(cy + ax, cx + ax, indexing="ij")
            st = ndimage.map_coordinates(
                np.nan_to_num(img), [yy, xx], order=3, mode="constant", cval=0.0
            )
            total = float(st.sum())
            if total > 0:
                stamps.append(st / total)
                used.append(c.star_id)
                continue
            why = "non-positive sky-subtracted stamp flux"
        excluded.append({"star_id": c.star_id, "reason": why})
    if len(stamps) < 5:
        raise ValueError(f"only {len(stamps)} usable calibration stamps")
    med = np.median(np.stack(stamps), axis=0)
    return EmpiricalPSF(med / med.sum(), pixel_scale, tuple(used), tuple(excluded))


# ---------------------------------------------------------------- image screen
class _ScreenBank:
    """Per-star template banks for the profile-likelihood Delta chi^2 (sky projected out)."""

    def __init__(self, psf: EmpiricalPSF, c: W3Cutout, sys_pix: float) -> None:
        gx, gy = c.gaia_xy
        r = FIT_RADIUS_PIX
        x0, x1 = int(np.floor(gx - r)), int(np.ceil(gx + r))
        y0, y1 = int(np.floor(gy - r)), int(np.ceil(gy + r))
        self.sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        self.ys = np.arange(y0, y1 + 1, dtype=float)
        self.xs = np.arange(x0, x1 + 1, dtype=float)
        yy, xx = np.meshgrid(self.ys, self.xs, indexing="ij")
        self.mask = (xx - gx) ** 2 + (yy - gy) ** 2 <= r * r
        self.gx, self.gy, self.psf = gx, gy, psf
        self.sys2 = max(sys_pix, 1e-3) ** 2
        self.var = float(np.median(c.unc[self.sl][self.mask] ** 2))

        fine = np.arange(-PRIMARY_RANGE_PIX, PRIMARY_RANGE_PIX + 1e-9, 0.25)
        coarse = np.arange(-PRIMARY_RANGE_PIX, PRIMARY_RANGE_PIX + 1e-9, 0.5)
        rb = np.arange(-SECONDARY_MAX_PIX, SECONDARY_MAX_PIX + 1e-9, 0.5)
        self.a_off = np.array([(x, y) for x in fine for y in fine])
        self.s_off = np.array([(x, y) for x in coarse for y in coarse])
        self.b_off = np.array(
            [(x, y) for x in rb for y in rb if x * x + y * y <= SECONDARY_MAX_PIX**2]
        )
        self.a_t = np.stack([self.tmpl(*o) for o in self.a_off])
        self.s_t = np.stack([self.tmpl(*o) for o in self.s_off])
        self.b_t = np.stack([self.tmpl(*o) for o in self.b_off])
        self.a_nn = (self.a_t * self.a_t).sum(1)
        self.a_pen = (self.a_off**2).sum(1) / self.sys2
        self.s_pen = (self.s_off**2).sum(1) / self.sys2
        self.ss = (self.s_t * self.s_t).sum(1)
        self.bb = (self.b_t * self.b_t).sum(1)
        self.sb = self.s_t @ self.b_t.T
        det = self.ss[:, None] * self.bb[None, :] - self.sb**2
        self.det_ok = det > 1e-6 * self.ss[:, None] * self.bb[None, :]
        self.det = np.where(self.det_ok, det, 1.0)

    def tmpl(self, dx: float, dy: float) -> np.ndarray:
        t = self.psf.evaluate(self.ys - (self.gy + dy), self.xs - (self.gx + dx))[self.mask]
        return t - t.mean()

    def data(self, img: np.ndarray) -> np.ndarray:
        v = np.asarray(img, float)[self.sl][self.mask]
        return v - v.mean()

    def _chi_a(self, y: np.ndarray, yy: float, p: np.ndarray) -> float:
        t = self.tmpl(p[0], p[1])
        b = float(y @ t)
        return (yy - b * b / float(t @ t)) / self.var + float(p @ p) / self.sys2

    def _chi_b(self, y: np.ndarray, yy: float, q: np.ndarray) -> float:
        p1, p2 = q[:2], q[2:]
        if float(p2 @ p2) > SECONDARY_MAX_PIX**2:
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
        return rss / self.var + float(p1 @ p1) / self.sys2

    def delta_chi2(self, img: np.ndarray) -> float:
        y = self.data(img)
        yy = float(y @ y)
        ba = self.a_t @ y
        chi_a_grid = (yy - ba**2 / self.a_nn) / self.var + self.a_pen
        ia = int(np.argmin(chi_a_grid))
        ra = minimize(
            lambda p: self._chi_a(y, yy, p),
            self.a_off[ia],
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
        chi_b_grid = rss / self.var + self.s_pen[:, None]
        i, j = np.unravel_index(int(np.argmin(chi_b_grid)), chi_b_grid.shape)
        rb = minimize(
            lambda q: self._chi_b(y, yy, q),
            np.concatenate([self.s_off[i], self.b_off[j]]),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-4, "maxfev": 1200},
        )
        chi_b = min(float(rb.fun), float(chi_b_grid[i, j]), chi_a)
        return max(chi_a - chi_b, 0.0)


def psf_flux(img: np.ndarray, bank: _ScreenBank, x: float, y: float) -> float:
    """Linear PSF-fit flux (DN) at pixel (x, y), sky projected out over the fit region."""
    t = bank.psf.evaluate(bank.ys - y, bank.xs - x)[bank.mask]
    t = t - t.mean()
    return float(bank.data(img) @ t / (t @ t))


# ---------------------------------------------------------------- frozen screens
def centroid_sigma_stat_arcsec(snr: float | np.ndarray) -> np.ndarray:
    return NOMINAL_W3_FWHM_ARCSEC / (FWHM_PER_SIGMA * np.asarray(snr, float))


def centroid_threshold_arcsec(
    snr: float | np.ndarray, sigma_sys_arcsec: float, false_alarm: float
) -> np.ndarray:
    scale = np.hypot(centroid_sigma_stat_arcsec(snr), sigma_sys_arcsec)
    return scale * np.sqrt(-2.0 * np.log(false_alarm))


def delta_chi2_scale(null_dchi2: np.ndarray) -> float:
    """Median-matched chi^2(3) scale of the calibration-null Delta chi^2 distribution."""
    med = float(np.median(np.asarray(null_dchi2, float)))
    if not med > 0:
        raise ValueError("calibration-null Delta chi^2 median is not positive")
    return med / float(chi2.median(CHI2_DOF))


def delta_chi2_threshold(scale: float, false_alarm: float) -> float:
    return float(scale * chi2.isf(false_alarm, CHI2_DOF))


def _rate(flags: np.ndarray) -> dict:
    f = np.asarray(flags, bool).ravel()
    k, n = int(f.sum()), int(f.size)
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n, "clopper_pearson_95": [lo, hi]}


@dataclass(frozen=True)
class FrozenScreens:
    """Thresholds fixed on the calibration split, with their calibration null record."""

    sigma_sys_arcsec: float
    target_false_alarm: float
    dchi2_scale: float
    dchi2_threshold: float
    calibration_star_ids: tuple[int, ...]
    calibration_offset_arcsec: tuple[float, ...]
    calibration_w3snr: tuple[float, ...]
    calibration_dchi2: tuple[float, ...]
    psf_star_ids: tuple[int, ...]

    @property
    def centroid_multiplier(self) -> float:
        return float(np.sqrt(-2.0 * np.log(self.target_false_alarm)))

    def centroid_flags(self, offset_arcsec, snr) -> np.ndarray:
        thr = centroid_threshold_arcsec(snr, self.sigma_sys_arcsec, self.target_false_alarm)
        return np.asarray(offset_arcsec, float) > thr

    def image_flags(self, dchi2) -> np.ndarray:
        return np.asarray(dchi2, float) > self.dchi2_threshold

    @staticmethod
    def naive_flags(flux_ratio, snr) -> np.ndarray:
        return np.asarray(flux_ratio, float) - 1.0 > NAIVE_EXCESS_SIGMA / np.asarray(snr, float)

    def calibration_false_alarms(self) -> dict:
        snr = np.array(self.calibration_w3snr)
        return {
            "centroid": _rate(self.centroid_flags(self.calibration_offset_arcsec, snr)),
            "image": _rate(self.image_flags(self.calibration_dchi2)),
        }

    def as_dict(self) -> dict:
        snr_med = float(np.median(self.calibration_w3snr))
        return {
            "rule_version": RULE_VERSION,
            "target_false_alarm": self.target_false_alarm,
            "centroid": {
                "statistic": "Gate #1 windowed W3 centroid offset from the Gaia position",
                "sigma_sys_arcsec_locked": self.sigma_sys_arcsec,
                "sigma_stat_rule": f'{NOMINAL_W3_FWHM_ARCSEC}" / (2.355 SNR_eff)',
                "multiplier": self.centroid_multiplier,
                "threshold_arcsec_at_calibration_median_snr": float(
                    centroid_threshold_arcsec(
                        snr_med, self.sigma_sys_arcsec, self.target_false_alarm
                    )
                ),
                "calibration_median_snr": snr_med,
            },
            "image": {
                "statistic": (
                    "profile Delta chi^2, one empirical PSF (position prior N(Gaia, sigma_sys)) "
                    "vs two (second flux >= 0); sky projected out; continuous refinement"
                ),
                "chi2_dof": CHI2_DOF,
                "scale_rule": "s = median(Delta chi^2_cal) / median(chi2(3)); threshold = "
                "s * chi2.isf(FA, 3)",
                "delta_chi2_scale": self.dchi2_scale,
                "delta_chi2_threshold": self.dchi2_threshold,
                "fit_radius_pix": FIT_RADIUS_PIX,
                "primary_range_pix": PRIMARY_RANGE_PIX,
                "secondary_max_pix": SECONDARY_MAX_PIX,
            },
            "naive_flux_excess": {
                "role": "adversarial reference rule (NOT a proposed screen)",
                "rule": f"flux_ratio - 1 > {NAIVE_EXCESS_SIGMA:g} / SNR, flux_ratio = 1 + "
                f"(aperture flux - uninjected aperture flux) / F_star, r <= {FIT_RADIUS_PIX:g} px",
            },
            "snr_eff_rule": "SNR_eff = w3snr / (1 - f_bg) (catalog sees the blended flux)",
            "calibration_false_alarm_in_sample": self.calibration_false_alarms(),
            "calibration": {
                "star_ids": list(self.calibration_star_ids),
                "centroid_offset_arcsec": list(self.calibration_offset_arcsec),
                "w3snr": list(self.calibration_w3snr),
                "delta_chi2": list(self.calibration_dchi2),
            },
            "psf_star_ids": list(self.psf_star_ids),
        }


def calibrate_frozen_screens(
    calibration_cutouts: Sequence[W3Cutout],
    empirical_psf: EmpiricalPSF,
    locked_sigma_sys_arcsec: u.Quantity,
    target_false_alarm: float = RAYLEIGH_3SIGMA_FALSE_ALARM,
) -> FrozenScreens:
    """Freeze both screen thresholds on the calibration split (LeakageError otherwise)."""
    cuts = _require_split(calibration_cutouts, "calibration", "calibrate_frozen_screens")
    if not set(empirical_psf.star_ids) <= {c.star_id for c in cuts}:
        raise LeakageError("empirical PSF was built from stars outside this calibration set")
    if not 0 < target_false_alarm < 1:
        raise ValueError("target_false_alarm must be in (0, 1)")
    sys_arcsec = _arcsec(locked_sigma_sys_arcsec, "locked_sigma_sys_arcsec")
    s_pix = nominal_window_sigma_pix(cuts[0].pixel_scale)
    ids, offs, snrs, dchis = [], [], [], []
    for c in cuts:
        if cutout_problem(c) is not None:
            continue
        gx, gy = c.gaia_xy
        cx, cy = windowed_centroid(c.flux, gx, gy, s_pix)
        bank = _ScreenBank(empirical_psf, c, sys_arcsec / c.pix)
        ids.append(c.star_id)
        offs.append(float(np.hypot(cx - gx, cy - gy) * c.pix))
        snrs.append(float(c.w3snr))
        dchis.append(bank.delta_chi2(c.flux))
    scale = delta_chi2_scale(np.array(dchis))
    return FrozenScreens(
        sys_arcsec,
        float(target_false_alarm),
        scale,
        delta_chi2_threshold(scale, target_false_alarm),
        tuple(ids),
        tuple(offs),
        tuple(snrs),
        tuple(dchis),
        tuple(empirical_psf.star_ids),
    )


# ------------------------------------------------------------ withheld injection
@dataclass(frozen=True, eq=False)
class WithheldInjection:
    """Per-star statistics of the frozen screens on the withheld two-arm injection."""

    star_ids: tuple[int, ...]
    patch_ids: tuple[str, ...]
    w3snr: np.ndarray  # (n_star,)
    reference_flux_dn: np.ndarray  # (n_star,)
    separations_arcsec: np.ndarray
    flux_fractions: np.ndarray
    azimuth_rad: np.ndarray  # (n_star, n_sep, n_frac)
    stats: dict  # arm -> statistic -> array
    excluded: tuple[dict, ...]
    rng_seed: int
    frozen: FrozenScreens

    def flags(self) -> dict:
        return flags_from_statistics(
            self.stats, self.w3snr, self.flux_fractions, self.frozen.as_dict()
        )

    def summary(self) -> dict:
        return summarize(self.flags(), self.separations_arcsec, self.flux_fractions)

    def as_dict(self) -> dict:
        return {
            "rng_seed": self.rng_seed,
            "rng": f"SeedSequence([{self.rng_seed}, 7]) azimuths, shape (star, sep, frac)",
            "n_stars": len(self.star_ids),
            "star_ids": list(self.star_ids),
            "patch_ids": list(self.patch_ids),
            "w3snr": self.w3snr.tolist(),
            "reference_flux_dn": self.reference_flux_dn.tolist(),
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


def aperture_flux(img: np.ndarray, c: W3Cutout, radius_pix: float = FIT_RADIUS_PIX) -> float:
    """Sky-subtracted aperture sum (DN) within ``radius_pix`` of the Gaia position."""
    gx, gy = c.gaia_xy
    sel = _radius(img.shape, gx, gy) <= radius_pix
    return float(np.sum(img[sel] - sky_level(img, gx, gy)))


def _screen_stats(
    img: np.ndarray, c: W3Cutout, bank: _ScreenBank, f_ref: float, ap_ref: float
) -> tuple[float, float, float]:
    gx, gy = c.gaia_xy
    cx, cy = windowed_centroid(img, gx, gy, nominal_window_sigma_pix(c.pixel_scale))
    off = float(np.hypot(cx - gx, cy - gy) * c.pix)
    ratio = 1.0 + (aperture_flux(img, c) - ap_ref) / f_ref
    return off, bank.delta_chi2(img), float(ratio)


def evaluate_withheld_matched_injection(
    withheld_cutouts: Sequence[W3Cutout],
    empirical_psf: EmpiricalPSF,
    frozen_screens: FrozenScreens,
    seed: int = 42,
    *,
    separations: u.Quantity | None = None,
    flux_fractions: Sequence[float] = FLUX_FRACTIONS,
) -> WithheldInjection:
    """Two-arm (off-center blend vs matched centered excess) injection on withheld stars."""
    cuts = _require_split(withheld_cutouts, "withheld", "evaluate_withheld_matched_injection")
    wh_ids = {c.star_id for c in cuts}
    if wh_ids & set(frozen_screens.calibration_star_ids) or wh_ids & set(empirical_psf.star_ids):
        raise LeakageError("withheld stars overlap the calibration stars behind the frozen method")
    if tuple(empirical_psf.star_ids) != tuple(frozen_screens.psf_star_ids):
        raise LeakageError("frozen screens were calibrated with a different empirical PSF")
    if not isinstance(seed, int | np.integer):
        raise TypeError("seed must be an explicit integer")
    if separations is None:
        separations = np.array(SEPARATIONS_ARCSEC) * u.arcsec
    seps = np.atleast_1d(_arcsec_values(separations, "separations"))
    fracs = np.atleast_1d(np.asarray(flux_fractions, float))
    if np.any(seps <= 0) or np.any((fracs <= 0) | (fracs >= 1)):
        raise ValueError("separations must be > 0 and flux fractions in (0, 1)")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7]))
    azim = rng.uniform(0.0, 2 * np.pi, (len(cuts), len(seps), len(fracs)))

    keep, excluded, snr, fref, rows = [], [], [], [], []
    sys_arcsec = frozen_screens.sigma_sys_arcsec
    for si, c in enumerate(cuts):
        why = cutout_problem(c)
        bank = None
        if why is None:
            gx, gy = c.gaia_xy
            cx, cy = windowed_centroid(c.flux, gx, gy, nominal_window_sigma_pix(c.pixel_scale))
            bank = _ScreenBank(empirical_psf, c, sys_arcsec / c.pix)
            f_ref = psf_flux(c.flux, bank, cx, cy)
            ap_ref = aperture_flux(c.flux, c)
            if not f_ref > 0:
                why = "non-positive PSF-fit flux of the uninjected star"
        if why is not None:
            excluded.append({"star_id": c.star_id, "reason": why})
            continue
        null = _screen_stats(c.flux, c, bank, f_ref, ap_ref)
        a = np.empty((len(seps), len(fracs), 3))
        b = np.empty((len(fracs), 3))
        for j, f in enumerate(fracs):
            f_bg = f_ref * f / (1.0 - f)
            img_b = c.flux + f_bg * empirical_psf.render(c.flux.shape, cx, cy)
            b[j] = _screen_stats(img_b, c, bank, f_ref, ap_ref)
            for i, th in enumerate(seps):
                tp = th / c.pix
                px = cx + tp * np.cos(azim[si, i, j])
                py = cy + tp * np.sin(azim[si, i, j])
                img_a = c.flux + f_bg * empirical_psf.render(c.flux.shape, px, py)
                a[i, j] = _screen_stats(img_a, c, bank, f_ref, ap_ref)
        keep.append(si)
        snr.append(float(c.w3snr))
        fref.append(f_ref)
        rows.append((null, a, b))
    if not keep:
        raise ValueError("no usable withheld cutouts")
    names = ("centroid_offset_arcsec", "delta_chi2", "flux_ratio")
    stats = {
        "null": {n: np.array([r[0][k] for r in rows]) for k, n in enumerate(names)},
        "arm_a": {n: np.stack([r[1][..., k] for r in rows]) for k, n in enumerate(names)},
        "arm_b": {n: np.stack([r[2][..., k] for r in rows]) for k, n in enumerate(names)},
    }
    return WithheldInjection(
        tuple(cuts[i].star_id for i in keep),
        tuple(cuts[i].patch_id for i in keep),
        np.array(snr),
        np.array(fref),
        seps,
        fracs,
        azim[keep],
        stats,
        tuple(excluded),
        int(seed),
        frozen_screens,
    )


# ------------------------------------------- flags / summaries (offline recomputable)
def flags_from_statistics(
    stats: dict, w3snr: Sequence[float], flux_fractions: Sequence[float], frozen: dict
) -> dict:
    """Screen flags from stored statistics and the frozen thresholds (pure, dict inputs)."""
    snr = np.asarray(w3snr, float)
    fr = np.asarray(flux_fractions, float)
    fa = float(frozen["target_false_alarm"])
    sys_arcsec = float(frozen["centroid"]["sigma_sys_arcsec_locked"])
    thr_img = float(frozen["image"]["delta_chi2_threshold"])
    snr_eff = {
        "null": snr,
        "arm_a": snr[:, None, None] / (1.0 - fr)[None, None, :],
        "arm_b": snr[:, None] / (1.0 - fr)[None, :],
    }
    out: dict = {s: {} for s in SCREENS}
    for arm, block in stats.items():
        se = snr_eff[arm]
        off = np.asarray(block["centroid_offset_arcsec"], float)
        out["centroid"][arm] = off > centroid_threshold_arcsec(se, sys_arcsec, fa)
        out["image"][arm] = np.asarray(block["delta_chi2"], float) > thr_img
        out["naive_flux_excess"][arm] = FrozenScreens.naive_flags(
            block["flux_ratio"], snr[(...,) + (None,) * (off.ndim - 1)]
        )
    return out


def area_weighted_survival(
    recovery: np.ndarray, separations_arcsec: Sequence[float], max_sep: float
) -> float:
    """Blend survival 1 - P(flag): area-uniform theta <= max_sep (p ~ theta), uniform f_bg."""
    th = np.asarray(separations_arcsec, float)
    m = th <= max_sep
    if not m.any():
        raise ValueError("no separations inside max_sep")
    w = th[m] / th[m].sum()
    rec = np.asarray(recovery, float)
    return float(np.sum(w[:, None] * (1.0 - rec[m])) / rec.shape[1])


def summarize(flags: dict, separations_arcsec, flux_fractions) -> dict:
    """Recovery / false-rejection rates with exact 95% intervals for every screen."""
    seps = np.asarray(separations_arcsec, float)
    out: dict = {
        "survival_max_separation_arcsec": SURVIVAL_MAX_SEPARATION_ARCSEC,
        "survival_weighting": "p(theta) ~ theta for theta <= 3.25 arcsec, uniform f_bg",
        "screens": {},
    }
    for s in SCREENS:
        fa = np.asarray(flags[s]["arm_a"], bool)
        fb = np.asarray(flags[s]["arm_b"], bool)
        n = fa.shape[0]
        rec = fa.mean(axis=0)
        frej = fb.mean(axis=0)
        cp_a = [[clopper_pearson(int(k), n) for k in row] for row in fa.sum(axis=0)]
        cp_b = [clopper_pearson(int(k), n) for k in fb.sum(axis=0)]
        out["screens"][s] = {
            "n_stars": n,
            "null_false_alarm": _rate(flags[s]["null"]),
            "arm_a_blend_recovery": rec.tolist(),
            "arm_a_k": fa.sum(axis=0).tolist(),
            "arm_a_clopper_pearson_95": [[list(x) for x in row] for row in cp_a],
            "arm_b_false_rejection": frej.tolist(),
            "arm_b_k": fb.sum(axis=0).tolist(),
            "arm_b_clopper_pearson_95": [list(x) for x in cp_b],
            "arm_b_false_rejection_pooled": _rate(fb),
            "blend_survival_within_3p25_arcsec": area_weighted_survival(
                rec, seps, SURVIVAL_MAX_SEPARATION_ARCSEC
            ),
            "centered_excess_retained_pooled": 1.0 - float(fb.mean()),
        }
    return out


# ----------------------------------------------------- promoted payload & recompute
def gate2_payload(psf: EmpiricalPSF, frozen: FrozenScreens, withheld: WithheldInjection) -> dict:
    """Scientific payload stored in the receipt / promoted ``reproduced_archive`` file."""
    return {
        "empirical_psf": psf.as_dict(),
        "frozen_screens": frozen.as_dict(),
        "withheld": withheld.as_dict(),
    }


def recompute_gate2(payload: dict) -> dict:
    """Offline re-derivation of every stored summary from the stored per-star arrays.

    Returns the recomputed PSF widths, frozen thresholds, calibration in-sample false
    alarms and withheld summary; callers compare them with the stored values.
    """
    p, fz, wh = payload["empirical_psf"], payload["frozen_screens"], payload["withheld"]
    stamp = np.asarray(p["stamp"], float)
    pix = float(p["pixel_scale_arcsec"])
    cal = fz["calibration"]
    fa = float(fz["target_false_alarm"])
    scale = delta_chi2_scale(np.asarray(cal["delta_chi2"], float))
    sys_arcsec = float(fz["centroid"]["sigma_sys_arcsec_locked"])
    snr_c = np.asarray(cal["w3snr"], float)
    cal_fa = {
        "centroid": _rate(
            np.asarray(cal["centroid_offset_arcsec"], float)
            > centroid_threshold_arcsec(snr_c, sys_arcsec, fa)
        ),
        "image": _rate(np.asarray(cal["delta_chi2"], float) > delta_chi2_threshold(scale, fa)),
    }
    flags = flags_from_statistics(wh["statistics"], wh["w3snr"], wh["flux_fractions"], fz)
    return {
        "stamp_sum": float(stamp.sum()),
        "fwhm_gaussian_arcsec": _gaussian_fwhm_pix(stamp) * pix,
        "fwhm_halfmax_arcsec": _halfmax_fwhm_pix(stamp) * pix,
        "delta_chi2_scale": scale,
        "delta_chi2_threshold": delta_chi2_threshold(scale, fa),
        "centroid_multiplier": float(np.sqrt(-2.0 * np.log(fa))),
        "calibration_false_alarm_in_sample": cal_fa,
        "summary": summarize(flags, wh["separations_arcsec"], wh["flux_fractions"]),
    }
