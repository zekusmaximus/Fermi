"""Mid-IR blend-recovery drill-down: WISE centroid screen vs. image-level screen.

Seeded injection-recovery on simulated WISE W3 cutouts (Gaussian PSF). Each
trial places a foreground star at the Gaia position (plus an astrometric
registration error with dispersion ``sigma_sys``) and, optionally, a background
point source at separation theta carrying a fraction ``f_bg`` of the W3 flux,
then adds white pixel noise scaled so the total flux has the stated W3 SNR
(optimal PSF photometry).

Two screens decide whether the source is flagged as a blend:

* **Centroid screen** (catalog level): Gaussian-windowed W3 position vs. Gaia
  position, tested against a Rayleigh null with per-axis dispersion
  ``sqrt((FWHM / (2.355 SNR))^2 + sigma_sys^2)`` - the published convention of
  Ren et al. 2026 eq. 1 plus a registration floor.
* **Image-level screen**: profile-likelihood Delta chi^2 between a one-PSF model
  (position prior N(Gaia, sigma_sys)) and a two-PSF model (second flux >= 0),
  against chi^2 with 3 dof. Fluxes are solved linearly in closed form on PSF
  template banks, so the screen is exact up to the position-grid resolution.

Nominal thresholds are applied as published; the empirical false-alarm rate of
each screen is measured from blend-free trials and reported, never assumed.
Outputs are pure objects (no I/O) consumed by the engine / infrared model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import astropy.units as u
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import beta, binom, chi2, kstest

from fermi_desk.errors import UnitValidationError
from fermi_desk.models.infrared import IRCalibration

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.355
# Rayleigh tail probability of a 3-sigma (two-sided Gaussian equivalent) excursion.
RAYLEIGH_3SIGMA_FALSE_ALARM = 0.0027


def _arcsec(q: object, name: str) -> float:
    if not isinstance(q, u.Quantity) or not q.unit.is_equivalent(u.arcsec):
        raise UnitValidationError(f"{name} must be an angle Quantity")
    return float(q.to_value(u.arcsec))


@dataclass(frozen=True)
class CutoutSpec:
    """Simulated W3 cutout configuration (all angles as astropy Quantities)."""

    psf_fwhm: u.Quantity = field(default_factory=lambda: 6.5 * u.arcsec)
    pixel_scale: u.Quantity = field(default_factory=lambda: 1.375 * u.arcsec)
    sigma_sys: u.Quantity = field(default_factory=lambda: 0.5 * u.arcsec)
    snr: float = 12.5
    npix: int = 25
    false_alarm: float = 0.0027

    def __post_init__(self) -> None:
        for name in ("psf_fwhm", "pixel_scale", "sigma_sys"):
            _arcsec(getattr(self, name), name)
        if self.snr <= 0 or self.npix < 21 or not 0 < self.false_alarm < 1:
            raise ValueError("Need snr > 0, npix >= 21, 0 < false_alarm < 1")

    @property
    def pix(self) -> float:
        return _arcsec(self.pixel_scale, "pixel_scale")

    @property
    def s_pix(self) -> float:
        return _arcsec(self.psf_fwhm, "psf_fwhm") / FWHM_PER_SIGMA / self.pix

    @property
    def sys_pix(self) -> float:
        return _arcsec(self.sigma_sys, "sigma_sys") / self.pix

    @property
    def sigma_noise(self) -> float:
        """Per-pixel noise giving optimal-PSF-photometry SNR for unit total flux."""
        return 1.0 / (self.snr * np.sqrt(4.0 * np.pi) * self.s_pix)

    @property
    def centroid_sigma_arcsec(self) -> float:
        """Ren et al. 2026 eq. 1 statistical term combined with the registration floor."""
        stat = _arcsec(self.psf_fwhm, "psf_fwhm") / (FWHM_PER_SIGMA * self.snr)
        return float(np.hypot(stat, _arcsec(self.sigma_sys, "sigma_sys")))

    def as_dict(self) -> dict:
        return {
            "psf_fwhm_arcsec": _arcsec(self.psf_fwhm, "psf_fwhm"),
            "pixel_scale_arcsec": self.pix,
            "sigma_sys_arcsec": _arcsec(self.sigma_sys, "sigma_sys"),
            "snr": self.snr,
            "npix": self.npix,
            "false_alarm_nominal": self.false_alarm,
        }


def _grid(spec: CutoutSpec) -> tuple[np.ndarray, np.ndarray, float]:
    c = (spec.npix - 1) / 2.0
    yy, xx = np.mgrid[0 : spec.npix, 0 : spec.npix].astype(float)
    return xx, yy, c


def _psf(xx, yy, x0, y0, s):
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)


def simulate_cutouts(
    spec: CutoutSpec,
    separation: u.Quantity,
    flux_fraction: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``n`` noisy W3 cutouts, shape (n, npix, npix); the Gaia position is the centre."""
    if not 0.0 <= flux_fraction < 1.0:
        raise ValueError("flux_fraction must be in [0, 1)")
    xx, yy, c = _grid(spec)
    sep = _arcsec(separation, "separation") / spec.pix
    d = rng.normal(0.0, spec.sys_pix, (n, 2))
    pa = rng.uniform(0.0, 2 * np.pi, n)
    sx = (c + d[:, 0])[:, None, None]
    sy = (c + d[:, 1])[:, None, None]
    img = (1.0 - flux_fraction) * _psf(xx, yy, sx, sy, spec.s_pix)
    if flux_fraction > 0:
        bx = sx + (sep * np.cos(pa))[:, None, None]
        by = sy + (sep * np.sin(pa))[:, None, None]
        img = img + flux_fraction * _psf(xx, yy, bx, by, spec.s_pix)
    return img + rng.normal(0.0, spec.sigma_noise, img.shape)


@lru_cache(maxsize=8)
def _templates(npix: int, s_pix: float, sys_pix: float) -> dict:
    """PSF template banks for the profile-likelihood screens (fluxes solved linearly)."""
    yy, xx = np.mgrid[0:npix, 0:npix].astype(float)
    c = (npix - 1) / 2.0

    def bank(offsets):
        t = np.stack([_psf(xx, yy, c + ox, c + oy, s_pix).ravel() for ox, oy in offsets])
        pen = np.array([(ox * ox + oy * oy) / sys_pix**2 for ox, oy in offsets])
        return t, pen

    fine = np.arange(-2.0, 2.0 + 1e-9, 0.25)
    coarse = np.arange(-2.0, 2.0 + 1e-9, 0.5)
    a_t, a_pen = bank([(x, y) for x in fine for y in fine])
    s_t, s_pen = bank([(x, y) for x in coarse for y in coarse])
    rb = np.arange(-8.0, 8.0 + 1e-9, 0.5)
    b_t, _ = bank([(x, y) for x in rb for y in rb if x * x + y * y <= 64.0])
    ss, bb, sb = (s_t * s_t).sum(1), (b_t * b_t).sum(1), s_t @ b_t.T
    det = ss[:, None] * bb[None, :] - sb**2
    return {
        "a_t": a_t,
        "a_pen": a_pen,
        "a_nn": (a_t * a_t).sum(1),
        "s_t": s_t,
        "s_pen": s_pen,
        "b_t": b_t,
        "ss": ss,
        "bb": bb,
        "sb": sb,
        "det": det,
        "det_ok": det > 1e-6 * ss[:, None] * bb[None, :],
    }


@dataclass(frozen=True)
class ScreenOutcomes:
    centroid_offset_arcsec: np.ndarray
    centroid_flagged: np.ndarray
    delta_chi2: np.ndarray
    image_flagged: np.ndarray


def _windowed_centroid(imgs: np.ndarray, spec: CutoutSpec, n_iter: int = 10) -> np.ndarray:
    """Gaussian-windowed (SExtractor-style) position, matched to the PSF width."""
    xx, yy, c = _grid(spec)
    s = spec.s_pix
    pos = np.full((imgs.shape[0], 2), c)
    for _ in range(n_iter):
        ex = xx - pos[:, 0, None, None]
        ey = yy - pos[:, 1, None, None]
        wi = np.exp(-(ex**2 + ey**2) / (2 * s * s)) * imgs
        norm = wi.sum(axis=(1, 2))
        norm = np.where(np.abs(norm) > 1e-12, norm, 1e-12)
        step = 2.0 * np.column_stack([(wi * ex).sum(axis=(1, 2)), (wi * ey).sum(axis=(1, 2))])
        pos = pos + np.clip(step / norm[:, None], -1.0, 1.0)
    return pos


def screen_cutouts(imgs: np.ndarray, spec: CutoutSpec, batch: int = 16) -> ScreenOutcomes:
    """Apply both screens to a stack of cutouts."""
    imgs = np.asarray(imgs, float).reshape(-1, spec.npix, spec.npix)
    _, _, c = _grid(spec)
    pos = _windowed_centroid(imgs, spec)
    off = np.hypot(pos[:, 0] - c, pos[:, 1] - c) * spec.pix
    t_cent = spec.centroid_sigma_arcsec * np.sqrt(-2.0 * np.log(spec.false_alarm))

    tb = _templates(spec.npix, spec.s_pix, spec.sys_pix)
    var = spec.sigma_noise**2
    det = np.where(tb["det_ok"], tb["det"], 1.0)
    pen_b = np.repeat(tb["s_pen"], tb["b_t"].shape[0])
    dchi = np.empty(len(imgs))
    for i0 in range(0, len(imgs), batch):
        d = imgs[i0 : i0 + batch].reshape(-1, spec.npix**2)
        dd = (d * d).sum(1)
        ba = d @ tb["a_t"].T
        chi_a = np.min((dd[:, None] - ba**2 / tb["a_nn"]) / var + tb["a_pen"], axis=1)
        b1 = (d @ tb["s_t"].T)[:, :, None]  # (n, GS, 1)
        b2 = (d @ tb["b_t"].T)[:, None, :]  # (n, 1, GB)
        f1 = (tb["bb"][None, None, :] * b1 - tb["sb"][None] * b2) / det
        f2 = (tb["ss"][None, :, None] * b2 - tb["sb"][None] * b1) / det
        rss = dd[:, None, None] - (f1 * b1 + f2 * b2)
        single = dd[:, None, None] - b1**2 / tb["ss"][None, :, None]
        rss = np.where(tb["det_ok"][None] & (f2 >= 0), rss, single)
        chi_b = np.min(rss.reshape(len(d), -1) / var + pen_b, axis=1)
        dchi[i0 : i0 + batch] = np.maximum(chi_a - np.minimum(chi_b, chi_a), 0.0)
    return ScreenOutcomes(off, off > t_cent, dchi, dchi > chi2.isf(spec.false_alarm, 3))


@dataclass(frozen=True)
class BlendGrid:
    separations_arcsec: np.ndarray
    flux_fractions: np.ndarray
    centroid_recovery: np.ndarray  # (n_sep, n_frac)
    image_recovery: np.ndarray
    centroid_false_alarm: float
    image_false_alarm: float
    n_trials: int
    n_null: int
    spec: CutoutSpec

    def as_dict(self) -> dict:
        return {
            "separations_arcsec": self.separations_arcsec.tolist(),
            "flux_fractions": self.flux_fractions.tolist(),
            "centroid_recovery": self.centroid_recovery.tolist(),
            "image_recovery": self.image_recovery.tolist(),
            "centroid_false_alarm_empirical": self.centroid_false_alarm,
            "image_false_alarm_empirical": self.image_false_alarm,
            "n_trials_per_cell": self.n_trials,
            "n_null_trials": self.n_null,
            "spec": self.spec.as_dict(),
        }


def blend_recovery_grid(
    spec: CutoutSpec,
    separations: u.Quantity,
    flux_fractions: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    n_null: int | None = None,
) -> BlendGrid:
    """Recovery probability of each screen over (separation x background flux fraction)."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    seps = np.atleast_1d(separations.to_value(u.arcsec)).astype(float)
    fracs = np.atleast_1d(np.asarray(flux_fractions, float))
    n_null = n_null if n_null is not None else n_trials * len(seps)
    null = screen_cutouts(simulate_cutouts(spec, 0 * u.arcsec, 0.0, n_null, rng), spec)
    cen = np.zeros((len(seps), len(fracs)))
    img = np.zeros_like(cen)
    for i, th in enumerate(seps):
        for j, f in enumerate(fracs):
            o = screen_cutouts(simulate_cutouts(spec, th * u.arcsec, f, n_trials, rng), spec)
            cen[i, j] = o.centroid_flagged.mean()
            img[i, j] = o.image_flagged.mean()
    return BlendGrid(
        seps,
        fracs,
        cen,
        img,
        float(null.centroid_flagged.mean()),
        float(null.image_flagged.mean()),
        n_trials,
        n_null,
        spec,
    )


def blend_survival(grid: BlendGrid, method: str, max_separation: u.Quantity) -> float:
    """Fraction of blends passing the screen: area-uniform theta <= max, uniform f_bg."""
    rec = {"centroid": grid.centroid_recovery, "image": grid.image_recovery}[method]
    th = grid.separations_arcsec
    mask = th <= _arcsec(max_separation, "max_separation")
    if not mask.any():
        raise ValueError("no grid separations inside max_separation")
    w = th[mask] / th[mask].sum()  # p(theta) ~ theta (uniform in area)
    return float(np.sum(w[:, None] * (1.0 - rec[mask])) / rec.shape[1])


def screened_calibration(
    cal_id: str, completeness: float, raw_blend_rate: float, survival: float, false_alarm: float
) -> IRCalibration:
    """Blend-only false-positive rate after a screen; completeness pays the false-alarm cost."""
    return IRCalibration(cal_id, completeness * (1.0 - false_alarm), raw_blend_rate * survival)


def predicted_photocentre_shift(separation: u.Quantity, flux_fraction: float) -> u.Quantity:
    """Flux-weighted photocentre displacement of an unresolved blend."""
    return flux_fraction * separation.to(u.arcsec)


def centroid_p_value(offset: u.Quantity, offset_err: u.Quantity, spec: CutoutSpec) -> float:
    """Rayleigh-null tail probability of a measured offset given its error and sigma_sys."""
    sigma = np.hypot(_arcsec(offset_err, "offset_err"), _arcsec(spec.sigma_sys, "sigma_sys"))
    return float(np.exp(-(_arcsec(offset, "offset") ** 2) / (2 * sigma**2)))


# ------------------------------------------- empirical registration floor (sigma_sys)
# Pure estimators shared by scripts/run_wise_control_experiment.py (live archive run),
# `fermi-desk validate` (offline recomputation from the stored controls) and the IR
# drill-down. Model per control star i: the 2-D W3-centroid offset from the Gaia
# position is Rayleigh with per-axis scale s_i = sqrt(sigma_stat_i^2 + sigma_sys^2),
# sigma_stat_i = FWHM / (2.355 SNR_i) (Ren et al. 2026 eq. 1).
def _arcsec_array(q: object, name: str) -> np.ndarray:
    if not isinstance(q, u.Quantity) or not q.unit.is_equivalent(u.arcsec):
        raise UnitValidationError(f"{name} must be an angle Quantity")
    return np.atleast_1d(q.to_value(u.arcsec)).astype(float)


def _check_offsets(offsets: u.Quantity, sigma_stat: u.Quantity) -> tuple[np.ndarray, np.ndarray]:
    r = _arcsec_array(offsets, "offsets")
    s = _arcsec_array(sigma_stat, "sigma_stat")
    if r.shape != s.shape or r.size == 0:
        raise ValueError("offsets and sigma_stat must be non-empty and the same shape")
    if np.any(r < 0) or np.any(s <= 0) or not np.all(np.isfinite(r) & np.isfinite(s)):
        raise ValueError("offsets must be >= 0 and sigma_stat > 0 (finite)")
    return r, s


def rayleigh_log_likelihood(
    offsets: u.Quantity, sigma_stat: u.Quantity, sigma_sys: u.Quantity
) -> float:
    r, s = _check_offsets(offsets, sigma_stat)
    var = s**2 + _arcsec(sigma_sys, "sigma_sys") ** 2
    return float(np.sum(np.log(np.maximum(r, 1e-300) / var) - r**2 / (2.0 * var)))


def sigma_sys_mle(
    offsets: u.Quantity, sigma_stat: u.Quantity, upper: u.Quantity = 5.0 * u.arcsec
) -> u.Quantity:
    """Heteroscedastic-Rayleigh maximum-likelihood registration floor (bounded at 0)."""
    r, s = _check_offsets(offsets, sigma_stat)
    hi = _arcsec(upper, "upper")

    def nll(x: float) -> float:
        var = s**2 + x * x
        return float(-np.sum(np.log(np.maximum(r, 1e-300) / var) - r**2 / (2.0 * var)))

    res = minimize_scalar(nll, bounds=(0.0, hi), method="bounded", options={"xatol": 1e-7})
    x = float(res.x)
    if nll(0.0) <= nll(x):  # boundary optimum: statistical noise alone explains the scatter
        x = 0.0
    return x * u.arcsec


RAYLEIGH_MEDIAN = float(np.sqrt(2.0 * np.log(2.0)))  # median of the unit Rayleigh law


def sigma_sys_robust(
    offsets: u.Quantity, sigma_stat: u.Quantity, upper: u.Quantity = 10.0 * u.arcsec
) -> u.Quantity:
    """Median-matched registration floor (robust to unseen-blend outliers).

    Solves median_i[r_i / sqrt(sigma_stat_i^2 + x^2)] = sqrt(2 ln 2) for x >= 0 (the
    unit-Rayleigh median); 0 when statistical noise alone already explains the median.
    Breakdown point 50%, so a few contaminated "controls" (background W3 sources absent
    from Gaia) cannot inflate the floor and thereby blind the centroid screen.
    """
    r, s = _check_offsets(offsets, sigma_stat)
    hi = _arcsec(upper, "upper")

    def g(x: float) -> float:
        return float(np.median(r / np.sqrt(s**2 + x * x))) - RAYLEIGH_MEDIAN

    if g(0.0) <= 0.0:
        return 0.0 * u.arcsec
    if g(hi) > 0.0:
        raise ValueError(f"robust sigma_sys exceeds {hi} arcsec")
    return float(brentq(g, 0.0, hi, xtol=1e-9)) * u.arcsec


def sigma_sys_moment(offsets: u.Quantity, sigma_stat: u.Quantity) -> u.Quantity:
    """Moment estimator: sqrt(max(0, <r^2>/2 - <sigma_stat^2>))."""
    r, s = _check_offsets(offsets, sigma_stat)
    return float(np.sqrt(max(0.0, np.mean(r**2) / 2.0 - np.mean(s**2)))) * u.arcsec


SIGMA_SYS_ESTIMATORS = {
    "robust_median": sigma_sys_robust,
    "rayleigh_mle": sigma_sys_mle,
    "moment": sigma_sys_moment,
}


def sigma_sys_bootstrap(
    offsets: u.Quantity,
    sigma_stat: u.Quantity,
    rng: np.random.Generator,
    n_boot: int = 2000,
    estimator: str = "robust_median",
) -> np.ndarray:
    """Seeded nonparametric bootstrap of a sigma_sys estimator (arcsec, length n_boot)."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    fn = SIGMA_SYS_ESTIMATORS[estimator]
    r, s = _check_offsets(offsets, sigma_stat)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, r.size, r.size)
        out[b] = fn(r[i] * u.arcsec, s[i] * u.arcsec).to_value(u.arcsec)
    return out


def clopper_pearson(k: int, n: int, cl: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for k successes in n trials."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need n > 0 and 0 <= k <= n")
    a = (1.0 - cl) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def centroid_screen_false_alarms(
    offsets: u.Quantity,
    sigma_stat: u.Quantity,
    sigma_sys: u.Quantity,
    false_alarm: float = RAYLEIGH_3SIGMA_FALSE_ALARM,
) -> dict:
    """Empirical false-alarm rate of the centroid screen on blend-free control stars.

    A star is flagged when its offset exceeds s_i * sqrt(-2 ln false_alarm). Reports the
    flag count, exact 95% interval, the binomial tail P(K >= k | nominal) and a KS test
    of the normalised offsets r_i / s_i against the unit Rayleigh law.
    """
    r, s = _check_offsets(offsets, sigma_stat)
    if not 0 < false_alarm < 1:
        raise ValueError("false_alarm must be in (0, 1)")
    scale = np.sqrt(s**2 + _arcsec(sigma_sys, "sigma_sys") ** 2)
    flags = r > scale * np.sqrt(-2.0 * np.log(false_alarm))
    k, n = int(flags.sum()), int(r.size)
    lo, hi = clopper_pearson(k, n)
    ks = kstest(r / scale, lambda x: 1.0 - np.exp(-(np.asarray(x) ** 2) / 2.0))
    return {
        "sigma_sys_arcsec": _arcsec(sigma_sys, "sigma_sys"),
        "nominal_false_alarm": false_alarm,
        "n_stars": n,
        "n_flagged": k,
        "empirical_false_alarm": k / n,
        "clopper_pearson_95": [lo, hi],
        "binomial_p_at_least_k_given_nominal": float(binom.sf(k - 1, n, false_alarm)),
        "ks_rayleigh_statistic": float(ks.statistic),
        "ks_rayleigh_p_value": float(ks.pvalue),
        "flagged_index": np.flatnonzero(flags).tolist(),
    }


@dataclass(frozen=True)
class RegistrationFloor:
    """One W3 centroid registration floor fed to the drill-down, with its provenance.

    The drill-down evaluates every floor side by side (legacy baseline and measured);
    it never silently replaces one with another.
    """

    label: str
    sigma_sys: u.Quantity
    data_tier: str
    source: str
    interval_68_arcsec: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if _arcsec(self.sigma_sys, "sigma_sys") < 0:
            raise ValueError("sigma_sys must be >= 0")

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "sigma_sys_arcsec": _arcsec(self.sigma_sys, "sigma_sys"),
            "data_tier": self.data_tier,
            "source": self.source,
            "interval_68_arcsec": list(self.interval_68_arcsec)
            if self.interval_68_arcsec
            else None,
        }


def grid_shift(reference: BlendGrid, other: BlendGrid) -> dict:
    """Cell-wise change (other - reference) of two grids on the same (theta, f_bg) axes."""
    if not (
        np.array_equal(reference.separations_arcsec, other.separations_arcsec)
        and np.array_equal(reference.flux_fractions, other.flux_fractions)
    ):
        raise ValueError("grids must share separation and flux-fraction axes")
    d_cen = other.centroid_recovery - reference.centroid_recovery
    d_img = other.image_recovery - reference.image_recovery
    return {
        "centroid_recovery_delta": d_cen.tolist(),
        "image_recovery_delta": d_img.tolist(),
        "centroid_recovery_mean_delta": float(d_cen.mean()),
        "image_recovery_mean_delta": float(d_img.mean()),
        "centroid_false_alarm_delta": other.centroid_false_alarm - reference.centroid_false_alarm,
        "image_false_alarm_delta": other.image_false_alarm - reference.image_false_alarm,
    }
