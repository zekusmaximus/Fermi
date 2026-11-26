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
from scipy.stats import chi2

from fermi_desk.errors import UnitValidationError
from fermi_desk.models.infrared import IRCalibration

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.355


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
