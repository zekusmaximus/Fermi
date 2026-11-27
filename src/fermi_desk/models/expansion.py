"""Poisson-Boolean settlement-front kernel with Earth-unreached conditioning.

Model
-----
Civilization origins form a homogeneous Poisson process in space-time over the
lookback window ``[0, T]``:

* ``thin_disk_2d``: rate ``Gamma_2d`` per unit disk area per unit time; targets are
  projected onto the Galactic mid-plane.
* ``thick_disk_3d``: uniform slab ``|z_mid| <= h_z`` with volumetric rate
  ``Gamma_2d / (2 h_z)`` (column-integrated rate equals ``Gamma_2d``). Earth sits at
  ``z_mid = z_sun``.
* ``finite_disk_3d`` (v0.2.0): axisymmetric disk of finite radius ``R <= R_gal`` centred
  on the Galactic Centre, in Galactocentric cylindrical coordinates ``(R, Z)``. The Sun
  sits at ``(R_E, Z_E) = (sqrt(R_0^2 - z_sun^2), z_sun)`` where ``R_0`` is the Sun-GC
  distance. Two explicit density profiles (``disk_profile``), both normalised so that the
  column-integrated origin rate at the Sun's cylindrical radius equals ``Gamma_2d``:

  - ``uniform``:      ``rho = Gamma_2d / (2 h_z)`` for ``R <= R_gal, |Z| <= h_z``
    (identical to ``thick_disk_3d`` wherever the Galactic edge is out of causal reach);
  - ``exponential``:  ``rho = Gamma_2d * exp(-(R - R_E) / R_d) * exp(-|Z| / h_z)
    / (2 h_z (1 - exp(-z_max / h_z)))`` for ``R <= R_gal, |Z| <= z_max``, with
    ``z_max = vertical_truncation_scale_heights * h_z`` (default 5 h_z), so that
    ``int_{-z_max}^{z_max} rho(R_E, Z) dZ = Gamma_2d`` exactly.

  Origins outside the disk do not exist, so there is no local-approximation limit.

An origin of age ``s`` (founded ``s`` before the present) occupies a ball of radius
``v s``. A target at distance ``d`` is observed as it was ``tau = d / c`` ago
(light-travel delay), so it is reached iff an origin lies inside its backward cone
``|x0 - x_i| <= v (s - tau_i)``.

Earth-unreached conditioning is exact: conditioning a Poisson process on having
no points in Earth's cone ``C_E`` leaves a Poisson process on the complement, so
``P(i reached | E unreached) = 1 - exp(-Lambda(C_i minus C_E))``.

``thin_disk_2d`` / ``thick_disk_3d`` are local and translation invariant (no finite
Galactic radius); they are only self-consistent while ``v T`` is small compared with
the Sun's Galactocentric radius. ``local_approximation_valid`` reports this.
``finite_disk_3d`` is valid at any ``v T`` and reports ``valid_finite_galaxy``.

Finite-disk numerics (all seeded, no hidden I/O):

* ``P(Earth reached)``: deterministic nested Gauss-Legendre quadrature of the cone
  measure (age x height x radial arc), cross-checked by the importance sampler below.
* per-target reach measures: age-integrated (Rao-Blackwellised) kernels
  ``K_u(x) = (T - tau_i - d_i(x)/v)_+`` and
  ``K_c(x) = (min(T, d_E(x)/v) - tau_i - d_i(x)/v)_+`` integrated against the disk
  density by multiple-importance sampling (balance heuristic) over slab-clipped balls
  centred on Earth and on each target. ``K_c <= K_u`` pointwise on shared samples, so
  Earth-unreached conditioning can never increase a reach probability.
* correlated ``P0``: the thick-slab near-miss shell sampler acts as a homogeneous
  proposal process bounded strictly inside the vertical domain and thinned by
  ``rho(x) / rho_proposal`` (exact Poisson thinning), which removes every origin with
  ``R > R_gal``.

Inner loops strip units after explicit conversion to: kpc, Gyr, kpc/Gyr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, CartesianRepresentation, Galactic, Galactocentric

from fermi_desk.errors import ComputationalBudgetError, GeometryError, UnitValidationError

FINITE_DISK = "finite_disk_3d"
GEOMETRY_MODES = ("thin_disk_2d", "thick_disk_3d", FINITE_DISK)
DISK_PROFILES = ("uniform", "exponential")
LOCAL_APPROX_OK = "ok"
LOCAL_APPROX_VIOLATED = "VIOLATED"
LOCAL_APPROX_FINITE = "valid_finite_galaxy"
C_KPC_PER_GYR = float(const.c.to_value(u.kpc / u.Gyr))
# ICRS position of the Galactic Centre (Sgr A*) used to orient the Galactocentric frame
# (Reid & Brunthaler 2004, ApJ 616, 872); pinned so results never depend on astropy's
# version-dependent frame defaults.
GALCEN_ICRS_DEG = (266.4051, -28.936175)
_FD_STREAM = 0x46443344  # "FD3D": SeedSequence tag of the finite-disk importance sampler

_GL8 = np.polynomial.legendre.leggauss(8)  # exact for piecewise cubics in s
_GL16 = np.polynomial.legendre.leggauss(16)
_GLZ = np.polynomial.legendre.leggauss(24)
# Geometric sub-division of the post-contact interval: resolves the crescent onset
# (scale d / v) even when T >> d / v.
_GEOM_FRACTIONS = np.concatenate([[0.0], np.geomspace(1e-9, 1.0, 28)])
_GLR = np.polynomial.legendre.leggauss(24)  # radial-arc quadrature (finite disk)


def _as(q: object, unit: u.UnitBase, name: str) -> float:
    if not isinstance(q, u.Quantity):
        raise UnitValidationError(f"{name} must be an astropy Quantity convertible to {unit}")
    try:
        return float(q.to_value(unit))
    except u.UnitConversionError as exc:
        raise UnitValidationError(f"{name}: cannot convert {q.unit} to {unit}") from exc


def _speed_fraction_of_c(q: object) -> float:
    if not isinstance(q, u.Quantity):
        raise UnitValidationError("expansion_speed must be a Quantity (dimensionless or velocity)")
    if q.unit.physical_type == "dimensionless":
        return float(q.to_value(u.dimensionless_unscaled))
    return _as(q, u.km / u.s, "expansion_speed") / const.c.to_value(u.km / u.s)


# ------------------------------------------------------------------- geometry
def ball_slab_volume(radius: np.ndarray, z_center: np.ndarray, half_thickness: float) -> np.ndarray:
    """Volume of a ball of radius R centred at height zc inside the slab |z| <= h."""
    r = np.asarray(radius, dtype=float)
    zc = np.asarray(z_center, dtype=float)
    lo = np.clip(-half_thickness - zc, -r, r)
    hi = np.clip(half_thickness - zc, -r, r)
    prim = lambda t: r * r * t - t**3 / 3.0  # noqa: E731
    return np.pi * np.maximum(prim(hi) - prim(lo), 0.0)


def lens_area(r1: np.ndarray, r2: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Intersection area of two disks with radii r1, r2 and centre separation d."""
    r1, r2, d = np.broadcast_arrays(
        np.asarray(r1, float), np.asarray(r2, float), np.asarray(d, float)
    )
    rmin = np.minimum(r1, r2)
    contained = d <= np.abs(r1 - r2)
    disjoint = d >= r1 + r2
    partial = ~contained & ~disjoint
    with np.errstate(divide="ignore", invalid="ignore"):
        c1 = np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0)
        c2 = np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0)
        kite = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
        part = (
            r1 * r1 * np.arccos(c1) + r2 * r2 * np.arccos(c2) - 0.5 * np.sqrt(np.maximum(kite, 0))
        )
    return np.where(contained, np.pi * rmin * rmin, np.where(partial, part, 0.0))


@lru_cache(maxsize=16)
def _galactocentric_affine_kpc(r0_kpc: float, z_sun_kpc: float) -> tuple[np.ndarray, np.ndarray]:
    """(rotation A, Earth position b) with ``X_gc = A @ x_galactic + b`` [kpc].

    Derived once from ``astropy.coordinates`` (Galactic -> ICRS -> Galactocentric is a
    static rotation plus a translation: pure offline math, no IERS / network data).
    """
    frame = Galactocentric(
        galcen_coord=ICRS(ra=GALCEN_ICRS_DEG[0] * u.deg, dec=GALCEN_ICRS_DEG[1] * u.deg),
        galcen_distance=r0_kpc * u.kpc,
        z_sun=z_sun_kpc * u.kpc,
        roll=0.0 * u.deg,
    )
    probe = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cart = Galactic(CartesianRepresentation(probe.T * u.kpc)).transform_to(frame).cartesian
    g = np.column_stack([cart.x.to_value(u.kpc), cart.y.to_value(u.kpc), cart.z.to_value(u.kpc)])
    b = g[0].copy()
    a = (g[1:] - b).T.copy()
    a.setflags(write=False)
    b.setflags(write=False)
    return a, b


def galactocentric_positions(
    xyz: u.Quantity, solar_galactocentric_radius: u.Quantity, solar_height: u.Quantity
) -> u.Quantity:
    """Galactocentric Cartesian (X, Y, Z) of Earth-centred Galactic Cartesian positions.

    ``xyz`` is (N, 3), x toward the Galactic Centre, z toward the NGP (the frame of
    ``TargetTable.xyz``). Equivalent to transforming (ra, dec, distance) from ICRS with
    ``astropy.coordinates.Galactocentric(galcen_distance=R_0, z_sun=z_sun, roll=0)``.
    """
    a, b = _galactocentric_affine_kpc(
        _as(solar_galactocentric_radius, u.kpc, "solar_galactocentric_radius"),
        _as(solar_height, u.kpc, "solar_height"),
    )
    if not isinstance(xyz, u.Quantity):
        raise UnitValidationError("xyz must be an astropy Quantity convertible to kpc")
    arr = np.asarray(xyz.to_value(u.kpc), float).reshape(-1, 3)
    return (arr @ a.T + b) * u.kpc


def disc_profile_integral(
    varrho: np.ndarray, r_center: float, r_gal: float, r_d: float, r_ref: float
) -> np.ndarray:
    """``int exp(-(R - r_ref) / r_d) dA`` over ``D(varrho; r_center)`` within ``R <= r_gal``.

    ``D`` is the planar disc of radius ``varrho`` whose centre lies at cylindrical radius
    ``r_center``; ``R`` is the cylindrical radius about the Galactic Centre. With
    ``r_d = inf`` the weight is 1 and the result is the lens area of the two discs. The
    integral runs over ``R``: full circles for ``R < varrho - r_center`` (closed form)
    plus arcs of half-angle ``arccos((R^2 + r_c^2 - varrho^2) / (2 R r_c))``, with a
    cosine substitution that removes the square-root endpoint behaviour of the arc.
    """
    rho = np.asarray(varrho, float)
    finite = bool(np.isfinite(r_d))

    def radial_primitive(c: np.ndarray) -> np.ndarray:  # int_0^c weight(R) R dR
        if not finite:
            return 0.5 * c * c
        return np.exp(r_ref / r_d) * r_d * r_d * (1.0 - np.exp(-c / r_d) * (1.0 + c / r_d))

    full = 2.0 * np.pi * radial_primitive(np.clip(rho - r_center, 0.0, r_gal))
    lo = np.minimum(np.abs(r_center - rho), r_gal)
    hi = np.minimum(r_center + rho, r_gal)
    width = np.maximum(hi - lo, 0.0)[..., None]
    x, w = _GLR
    uu = (x + 1.0) / 2.0
    rr = lo[..., None] + width * (1.0 - np.cos(np.pi * uu)) / 2.0
    jac = width * (np.pi / 2.0) * np.sin(np.pi * uu) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = (rr * rr + r_center * r_center - rho[..., None] ** 2) / (2.0 * rr * r_center)
    arc = 2.0 * np.arccos(np.clip(np.nan_to_num(cosang, nan=1.0), -1.0, 1.0))
    weight = np.exp(-(rr - r_ref) / r_d) if finite else 1.0
    return full + np.sum(w * jac * weight * rr * arc, axis=-1)


def _gl_piecewise(f, lo: np.ndarray, hi: np.ndarray, edges_inner: np.ndarray, gl) -> np.ndarray:
    """Integrate f(s, row) over [lo_i, hi_i] split at per-row interior edges.

    ``edges_inner`` has shape (N, K); edges are clipped into [lo, hi] and sorted.
    ``f`` receives ``s`` with shape (N, K+1, n) and must broadcast per row.
    """
    if np.size(lo) == 0:
        return np.zeros(0)
    lo = np.asarray(lo, float)[:, None]
    hi = np.maximum(np.asarray(hi, float)[:, None], lo)
    inner = np.clip(np.asarray(edges_inner, float).reshape(len(lo), -1), lo, hi)
    edges = np.sort(np.concatenate([lo, inner, hi], axis=1), axis=1)
    a, b = edges[:, :-1, None], edges[:, 1:, None]
    x, w = gl
    s = a + (b - a) * (x + 1.0) / 2.0
    return np.sum(f(s) * w * (b - a) / 2.0, axis=(1, 2))


# --------------------------------------------------------------------- results
@dataclass(frozen=True)
class ZeroDetectionResult:
    """Correlated P(no detections) from the seeded Monte Carlo."""

    p_zero: float
    p_zero_se: float
    p_zero_targets_only: float
    mc_expected_reached: float
    mc_expected_reached_se: float
    mu_rim: float
    mu_core: float
    n_mc: int
    n_points: int
    conditioned_on_earth_unreached: bool
    # finite_disk_3d only: proposal origins surviving the density thinning, and the
    # proposal-to-local density bound used for it.
    n_points_accepted: int | None = None
    proposal_density_bound: float | None = None


@dataclass(frozen=True)
class FiniteDiskReach:
    """Importance-sampled reach measures of a target set (``finite_disk_3d``).

    ``lam_*`` are Poisson measures per target (cone / cone-minus-Earth-cone); ``*_se``
    are Monte Carlo standard errors. ``sum_*`` are over targets, with the standard error
    of the sum (targets share samples, so it is NOT the quadrature sum of the per-target
    errors). ``earth_lambda_mc`` is an independent check of the deterministic Earth cone.
    """

    lam_unconditional: np.ndarray
    lam_unconditional_se: np.ndarray
    lam_conditional: np.ndarray
    lam_conditional_se: np.ndarray
    sum_unconditional: float
    sum_unconditional_se: float
    sum_conditional: float
    sum_conditional_se: float
    earth_lambda_mc: float
    earth_lambda_mc_se: float
    n_samples: int
    seed: int


@dataclass(frozen=True)
class ExpansionModel:
    """Settlement-front kernel. All inputs are astropy Quantities; geometry is explicit.

    ``finite_disk_3d`` additionally needs ``disk_profile`` (``uniform`` | ``exponential``),
    ``disk_radius`` (R_gal), ``solar_galactocentric_radius`` (R_0) and, for the
    exponential profile, ``radial_scale_length`` (R_d). ``importance_seed`` and
    ``n_importance`` drive its seeded per-target importance sampler.
    """

    geometry: str
    origin_rate: u.Quantity
    expansion_speed: u.Quantity
    lookback: u.Quantity
    scale_height: u.Quantity | None = None
    solar_height: u.Quantity | None = None
    light_travel_delay: bool = True
    local_validity_radius: u.Quantity = field(default_factory=lambda: 8.0 * u.kpc)
    disk_profile: str | None = None
    disk_radius: u.Quantity | None = None
    solar_galactocentric_radius: u.Quantity | None = None
    radial_scale_length: u.Quantity | None = None
    vertical_truncation_scale_heights: float = 5.0
    importance_seed: int = 0
    n_importance: int = 16384

    def __post_init__(self) -> None:
        if self.geometry not in GEOMETRY_MODES:
            raise GeometryError(
                f"geometry must be one of {GEOMETRY_MODES} (explicitly), got {self.geometry!r}"
            )
        gam = _as(self.origin_rate, u.kpc**-2 / u.Gyr, "origin_rate")
        beta = _speed_fraction_of_c(self.expansion_speed)
        t = _as(self.lookback, u.Gyr, "lookback")
        if gam < 0 or not 0 <= beta < 1 or t < 0:
            raise ValueError(f"Need Gamma>=0, 0<=v/c<1, T>=0 (got {gam}, {beta}, {t})")
        if self.geometry in ("thick_disk_3d", FINITE_DISK):
            if self.scale_height is None or self.solar_height is None:
                raise GeometryError(
                    f"{self.geometry} requires explicit scale_height and solar_height"
                )
            h = _as(self.scale_height, u.kpc, "scale_height")
            zs = _as(self.solar_height, u.kpc, "solar_height")
            if h <= 0 or abs(zs) > h:
                raise GeometryError(f"Need h_z > 0 and |z_sun| <= h_z (got {h}, {zs} kpc)")
        else:
            h, zs = np.inf, 0.0
        object.__setattr__(self, "_g", gam)
        object.__setattr__(self, "_beta", beta)
        object.__setattr__(self, "_v", beta * C_KPC_PER_GYR)
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "_h", h)
        object.__setattr__(self, "_zs", zs)
        if self.geometry == FINITE_DISK:
            self._init_finite_disk(gam, h, zs)
        elif self.disk_profile is not None:
            raise GeometryError(
                f"disk_profile applies to {FINITE_DISK} only (got geometry {self.geometry!r})"
            )

    def _init_finite_disk(self, gam: float, h: float, zs: float) -> None:
        if self.disk_profile not in DISK_PROFILES:
            raise GeometryError(
                f"{FINITE_DISK} requires an explicit disk_profile, one of {DISK_PROFILES} "
                f"(got {self.disk_profile!r})"
            )
        if self.disk_radius is None or self.solar_galactocentric_radius is None:
            raise GeometryError(
                f"{FINITE_DISK} requires explicit disk_radius and solar_galactocentric_radius"
            )
        r_gal = _as(self.disk_radius, u.kpc, "disk_radius")
        r0 = _as(self.solar_galactocentric_radius, u.kpc, "solar_galactocentric_radius")
        exponential = self.disk_profile == "exponential"
        if exponential:
            if self.radial_scale_length is None:
                raise GeometryError("exponential disk_profile requires radial_scale_length")
            r_d = _as(self.radial_scale_length, u.kpc, "radial_scale_length")
            n_h = float(self.vertical_truncation_scale_heights)
            if r_d <= 0 or not n_h >= 1.0:
                raise GeometryError("Need R_d > 0 and vertical_truncation_scale_heights >= 1")
            z_lim = n_h * h
            rho0 = gam / (2.0 * h * -np.expm1(-z_lim / h))
        else:
            r_d, z_lim = np.inf, h
            rho0 = gam / (2.0 * h)
        if int(self.n_importance) < 16:
            raise ValueError("n_importance must be >= 16")
        if r0 <= abs(zs):
            raise GeometryError(f"Need R_0 > |z_sun| (got R_0={r0}, z_sun={zs} kpc)")
        rot, earth = _galactocentric_affine_kpc(r0, zs)
        r_e = float(np.hypot(earth[0], earth[1]))
        if not 0.0 < r_e < r_gal:
            raise GeometryError(
                f"Need 0 < R_sun < R_gal for {FINITE_DISK} (got R_sun={r_e}, R_gal={r_gal} kpc)"
            )
        for name, val in (
            ("_rgal", r_gal),
            ("_rd", r_d),
            ("_zlim", z_lim),
            ("_rho0", rho0),
            ("_gc_rot", rot),
            ("_gc_earth", earth),
            ("_re", r_e),
            ("_ze", float(earth[2])),
            ("_exponential", exponential),
            ("_far_e", float(np.hypot(r_e + r_gal, z_lim + abs(float(earth[2]))))),
            ("_fd_cache", {}),
        ):
            object.__setattr__(self, name, val)

    # ------------------------------------------------------------- basic props
    @property
    def is_3d(self) -> bool:
        return self.geometry == "thick_disk_3d"

    @property
    def is_finite(self) -> bool:
        return self.geometry == FINITE_DISK

    @property
    def rate_density(self) -> float:
        """Origins per kpc^2 Gyr (2D) or per kpc^3 Gyr (3D slab; finite disk: in the
        mid-plane at the Sun's cylindrical radius)."""
        if self.is_finite:
            return self._rho0
        return self._g / (2.0 * self._h) if self.is_3d else self._g

    @property
    def max_bubble_radius(self) -> u.Quantity:
        return (self._v * self._t) * u.kpc

    @property
    def local_approximation_valid(self) -> bool:
        """True while the geometry is self-consistent (always, for the finite disk)."""
        if self.is_finite:
            return True
        return bool(self.max_bubble_radius <= self.local_validity_radius)

    @property
    def local_approx_status(self) -> str:
        """``ok`` | ``VIOLATED`` (local infinite disks) | ``valid_finite_galaxy``."""
        if self.is_finite:
            return LOCAL_APPROX_FINITE
        return LOCAL_APPROX_OK if self.local_approximation_valid else LOCAL_APPROX_VIOLATED

    def _measure(self, radius: np.ndarray, z_center: np.ndarray | float) -> np.ndarray:
        if self.is_3d:
            return ball_slab_volume(radius, z_center, self._h)
        return np.pi * np.asarray(radius, float) ** 2

    def _slab_breaks(self, z_center: np.ndarray) -> np.ndarray:
        if not self.is_3d:
            return np.zeros((len(z_center), 0))
        return np.column_stack([np.abs(self._h - z_center), np.abs(self._h + z_center)])

    # ----------------------------------------------------------- target geometry
    def _positions(self, xyz: u.Quantity) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(model-frame positions relative to Earth [kpc], true distance, delay [Gyr]).

        ``thin_disk_2d`` uses a distance-preserving projection: each target keeps its
        Galactic azimuth and is placed in the plane at its TRUE distance, so geometry,
        light-travel delay and EIRP all refer to the same Earth-target distance. (A
        plain (x, y) projection would shrink high-latitude separations below v * tau
        and make fast fronts spuriously unable to reach them; found by the M2 adversary.)
        """
        arr = np.asarray(u.Quantity(xyz).to_value(u.kpc), float).reshape(-1, 3)
        d3 = np.linalg.norm(arr, axis=1)
        tau = d3 / C_KPC_PER_GYR if self.light_travel_delay else np.zeros_like(d3)
        if self.is_3d:
            return arr, d3, tau
        if self.is_finite:  # Earth-centred, axes aligned with the Galactocentric frame
            return arr @ self._gc_rot.T, d3, tau
        rho = np.linalg.norm(arr[:, :2], axis=1)
        safe = np.where(rho > 0, rho, 1.0)
        unit = np.where(rho[:, None] > 0, arr[:, :2] / safe[:, None], np.array([1.0, 0.0]))
        return unit * d3[:, None], d3, tau

    # ------------------------------------------------------- unconditional reach
    def _cone_lambda(self, z_mid: np.ndarray, t_eff: np.ndarray) -> np.ndarray:
        t_eff = np.maximum(np.asarray(t_eff, float), 0.0)
        if self._v == 0.0 or self._g == 0.0:
            return np.zeros_like(t_eff)
        if not self.is_3d:
            return self._g * np.pi * self._v**2 * t_eff**3 / 3.0
        z_mid = np.asarray(z_mid, float)
        breaks = self._slab_breaks(z_mid) / self._v
        f = lambda s: ball_slab_volume(self._v * s, z_mid[:, None, None], self._h)  # noqa: E731
        return self.rate_density * _gl_piecewise(f, np.zeros_like(t_eff), t_eff, breaks, _GL8)

    def earth_lambda(self) -> float:
        if self.is_finite:
            return self._fd_earth_cone()
        return float(self._cone_lambda(np.array([self._zs]), np.array([self._t]))[0])

    def earth_reach_probability(self) -> float:
        return float(-np.expm1(-self.earth_lambda()))

    def log_p_earth_unreached(self) -> float:
        return -self.earth_lambda()

    def target_lambda(self, xyz: u.Quantity) -> np.ndarray:
        """Unconditional cone measure for each target (light-travel delay applied)."""
        if self.is_finite:
            return self.finite_disk_reach(xyz).lam_unconditional.copy()
        pos, _, tau = self._positions(xyz)
        z_mid = pos[:, 2] + self._zs if self.is_3d else np.zeros(len(pos))
        return self._cone_lambda(z_mid, self._t - tau)

    # --------------------------------------------------------- conditional reach
    def target_lambda_given_earth_unreached(self, xyz: u.Quantity) -> np.ndarray:
        """Measure of C_i minus C_E: origins reaching target i but not Earth."""
        if self.is_finite:
            return self.finite_disk_reach(xyz).lam_conditional.copy()
        pos, d3, tau = self._positions(xyz)
        n = len(pos)
        if n == 0 or self._v == 0.0 or self._g == 0.0:
            return np.zeros(n)
        v, t = self._v, self._t
        dxy = np.linalg.norm(pos[:, :2], axis=1)
        out = np.zeros(n)
        chunk = 32 if self.is_3d else 1024
        for i0 in range(0, n, chunk):
            sl = slice(i0, i0 + chunk)
            tau_c, dxy_c = tau[sl], dxy[sl]
            lo = tau_c
            hi = np.maximum(t, lo)
            # contact epoch: r1 + r2 = d  =>  s* = (d / v + tau) / 2
            d_contact = d3[sl] if self.is_3d else dxy_c
            s_star = np.clip((d_contact / v + tau_c) / 2.0, lo, hi)
            inner = np.concatenate(
                [s_star[:, None], s_star[:, None] + (hi - s_star)[:, None] * _GEOM_FRACTIONS[1:]],
                axis=1,
            )
            if self.is_3d:
                zi = pos[sl, 2]
                f = lambda s, zi=zi, tau_c=tau_c, dxy_c=dxy_c: self._crescent_3d(  # noqa: E731
                    s, zi, tau_c, dxy_c
                )
            else:
                f = lambda s, tau_c=tau_c, dxy_c=dxy_c: self._crescent_2d(s, tau_c, dxy_c)  # noqa: E731
            out[sl] = self.rate_density * _gl_piecewise(f, lo, hi, inner, _GL16)
        return out

    def _crescent_2d(self, s: np.ndarray, tau: np.ndarray, dxy: np.ndarray) -> np.ndarray:
        r1 = np.maximum(self._v * (s - tau[:, None, None]), 0.0)
        r2 = self._v * s
        return np.pi * r1 * r1 - lens_area(r1, r2, dxy[:, None, None])

    def _crescent_3d(
        self, s: np.ndarray, zi: np.ndarray, tau: np.ndarray, dxy: np.ndarray
    ) -> np.ndarray:
        """Slab-clipped volume of B(x_i, r1) minus B(0, r2), integrated over z slices."""
        r1 = np.maximum(self._v * (s - tau[:, None, None]), 0.0)[..., None]
        r2 = (self._v * s)[..., None]
        zc = zi[:, None, None, None]
        z_lo = np.maximum(zc - r1, -self._h - self._zs)
        z_hi = np.minimum(zc + r1, self._h - self._zs)
        width = np.maximum(z_hi - z_lo, 0.0)
        xz, wz = _GLZ
        zp = z_lo + width * (xz + 1.0) / 2.0
        rho1 = np.sqrt(np.maximum(r1 * r1 - (zp - zc) ** 2, 0.0))
        rho2 = np.sqrt(np.maximum(r2 * r2 - zp * zp, 0.0))
        area = np.pi * rho1 * rho1 - lens_area(rho1, rho2, dxy[:, None, None, None])
        return np.sum(area * wz, axis=-1) * width[..., 0] / 2.0

    # ----------------------------------------------------------- expectations
    def reach_probabilities(
        self, xyz: u.Quantity, condition_on_earth_unreached: bool
    ) -> np.ndarray:
        lam = (
            self.target_lambda_given_earth_unreached(xyz)
            if condition_on_earth_unreached
            else self.target_lambda(xyz)
        )
        return -np.expm1(-lam)

    def expected_reached(self, xyz: u.Quantity, condition_on_earth_unreached: bool) -> float:
        return float(np.sum(self.reach_probabilities(xyz, condition_on_earth_unreached)))

    # ----------------------------------------------------- correlated Monte Carlo
    def _shell_radii(self, s: np.ndarray, d_max: float, core_offset: float | None):
        """Inner/outer radius of the sampled shell at age s (core_offset=None => conditional)."""
        r_out = self._v * s + d_max
        r_in = self._v * s if core_offset is None else np.maximum(self._v * s - core_offset, 0.0)
        return r_in, r_out

    def _integrate_ball(self, offset: float, sign: float) -> float:
        """rate * int_0^T M(max(0, v s + sign*offset)) ds, exact piecewise quadrature."""
        v, t, zs = self._v, self._t, self._zs
        if v == 0.0:
            radius_const = max(sign * offset, 0.0)
            return self.rate_density * t * float(self._measure(np.array([radius_const]), zs)[0])
        shift = sign * offset
        knots = [-shift / v]
        if self.is_3d:
            knots += [(abs(self._h - zs) - shift) / v, (abs(self._h + zs) - shift) / v]
        f = lambda s: self._measure(np.maximum(v * s + shift, 0.0), zs)  # noqa: E731
        val = _gl_piecewise(f, np.array([0.0]), np.array([t]), np.array([knots]), _GL8)
        return self.rate_density * float(val[0])

    def zero_detection_probability(
        self,
        xyz: u.Quantity,
        p_detect: np.ndarray,
        *,
        rng: np.random.Generator,
        condition_on_earth_unreached: bool,
        n_mc: int,
        p_earth: float = 0.0,
        max_points: float = 2.0e7,
    ) -> ZeroDetectionResult:
        """Exact (MC) P(no target detection [and no Earth evidence]) under shared fronts.

        Emission is Rao-Blackwellised: given a reach pattern, P(no detection) is
        prod_i (1 - p_i)^{reached_i}. Origins that would reach every target and
        Earth (the "core") are handled analytically; only the shell of near-miss
        fronts is sampled, with a zero-truncated Poisson count per realisation.
        ``p_earth`` (Solar System evidence) only enters without Earth conditioning.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be an explicit numpy.random.Generator")
        pos, _d3, tau = self._positions(xyz)
        p = np.asarray(p_detect, float).reshape(-1)
        if p.shape[0] != pos.shape[0]:
            raise ValueError("p_detect length must match number of targets")
        if np.any((p < 0) | (p > 1)) or not 0 <= p_earth <= 1:
            raise ValueError("detection probabilities must lie in [0, 1]")
        with np.errstate(divide="ignore"):
            log1m = np.log1p(-p)
        if self.is_finite:
            return self._fd_zero_detection(
                pos, tau, p, log1m, rng, condition_on_earth_unreached, n_mc, p_earth, max_points
            )
        dist = np.linalg.norm(pos, axis=1)
        d_max = float(dist.max()) if len(dist) else 0.0
        p_e = 0.0 if condition_on_earth_unreached else float(p_earth)

        if condition_on_earth_unreached:
            core_offset, mu_core = None, 0.0
            mu_rim = self._integrate_ball(d_max, +1.0) - self._integrate_ball(0.0, +1.0)
        else:
            core_offset = float(np.max(dist + self._v * tau)) if len(dist) else 0.0
            mu_core = self._integrate_ball(core_offset, -1.0)
            mu_rim = self._integrate_ball(d_max, +1.0) - mu_core
        mu_rim = max(mu_rim, 0.0)

        g_all_t = float(np.exp(np.sum(log1m))) if len(p) else 1.0
        g_all = g_all_t * (1.0 - p_e)
        p_core = float(-np.expm1(-mu_core))
        w_rim = float(-np.expm1(-mu_rim))

        g_joint = np.ones(0)
        g_targ = np.ones(0)
        counts = np.zeros(0)
        n_points = 0
        if w_rim > 0.0 and n_mc > 0:
            mean_pts = mu_rim / w_rim
            if n_mc * mean_pts > max_points:
                raise ComputationalBudgetError(
                    f"Expected {n_mc * mean_pts:.3g} sampled origins exceeds budget {max_points:.3g}"
                    " (reduce lookback, n_mc, or raise max_points)"
                )
            # Bound the (points x targets x dim) separation tensor to ~2e6 elements.
            per_chunk = int(max(1, 2.0e6 // max(1.0, mean_pts * max(1, len(p)))))
            gj, gt, cn = [], [], []
            done = 0
            while done < n_mc:
                k = min(per_chunk, n_mc - done)
                a, b, c, npts = self._mc_chunk(
                    k, mu_rim, pos, tau, log1m, p_e, d_max, core_offset, rng
                )
                gj.append(a)
                gt.append(b)
                cn.append(c)
                n_points += npts
                done += k
            g_joint, g_targ, counts = np.concatenate(gj), np.concatenate(gt), np.concatenate(cn)

        def combine(g_bar: float, g_core: float) -> float:
            return 1.0 - p_core * (1.0 - g_core) - (1.0 - p_core) * w_rim * (1.0 - g_bar)

        mean_or_one = lambda arr: float(arr.mean()) if arr.size else 1.0  # noqa: E731
        p_zero = combine(mean_or_one(g_joint), g_all)
        p_zero_t = combine(mean_or_one(g_targ), g_all_t)
        se = (
            (1.0 - p_core) * w_rim * float(g_joint.std(ddof=1)) / np.sqrt(g_joint.size)
            if g_joint.size > 1
            else 0.0
        )
        exp_reached = p_core * len(p) + (1.0 - p_core) * w_rim * (
            float(counts.mean()) if counts.size else 0.0
        )
        exp_reached_se = (
            (1.0 - p_core) * w_rim * float(counts.std(ddof=1)) / np.sqrt(counts.size)
            if counts.size > 1
            else 0.0
        )
        return ZeroDetectionResult(
            p_zero=float(p_zero),
            p_zero_se=float(se),
            p_zero_targets_only=float(p_zero_t),
            mc_expected_reached=float(exp_reached),
            mc_expected_reached_se=float(exp_reached_se),
            mu_rim=float(mu_rim),
            mu_core=float(mu_core),
            n_mc=int(n_mc),
            n_points=int(n_points),
            conditioned_on_earth_unreached=condition_on_earth_unreached,
        )

    def _mc_chunk(self, k, mu, pos, tau, log1m, p_e, d_max, core_offset, rng):
        counts = _zero_truncated_poisson(mu, k, rng)
        total = int(counts.sum())
        s = self._sample_ages(total, d_max, core_offset, rng)
        r_in, r_out = self._shell_radii(s, d_max, core_offset)
        pts = self._sample_shell_points(s, r_in, r_out, rng)
        owner_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        earth_hit = np.linalg.norm(pts, axis=1) <= self._v * s
        earth_any = np.logical_or.reduceat(earth_hit, owner_start)
        if len(pos):
            sep = np.linalg.norm(pts[:, None, :] - pos[None, :, :], axis=2)
            hit = sep <= self._v * (s[:, None] - tau[None, :])
            reached = np.logical_or.reduceat(hit, owner_start, axis=0)
            logg = np.where(reached, log1m[None, :], 0.0).sum(axis=1)
            n_reached = reached.sum(axis=1).astype(float)
        else:
            logg = np.zeros(k)
            n_reached = np.zeros(k)
        g_t = np.exp(logg)
        g_j = g_t * (1.0 - p_e * earth_any)
        return g_j, g_t, n_reached, total

    def _rim_measure(self, s: np.ndarray, d_max: float, core_offset: float | None) -> np.ndarray:
        r_in, r_out = self._shell_radii(s, d_max, core_offset)
        return self._measure(r_out, self._zs) - self._measure(r_in, self._zs)

    def _sample_ages(self, n: int, d_max: float, core_offset, rng) -> np.ndarray:
        """Ages with density proportional to the (non-decreasing) shell measure."""
        env = float(self._rim_measure(np.array([self._t]), d_max, core_offset)[0])
        out = np.empty(0)
        while out.size < n:
            m = max(64, 2 * (n - out.size))
            cand = rng.uniform(0.0, self._t, m)
            dens = self._rim_measure(cand, d_max, core_offset)
            if np.any(dens > env * (1 + 1e-9)):
                raise AssertionError("shell measure not monotone; rejection envelope invalid")
            keep = cand[rng.uniform(0.0, env, m) < dens]
            out = np.concatenate([out, keep])
        return out[:n]

    def _sample_shell_points(self, s, r_in, r_out, rng) -> np.ndarray:
        n = s.size
        if not self.is_3d:
            rho = np.sqrt(r_in**2 + rng.uniform(0.0, 1.0, n) * (r_out**2 - r_in**2))
            phi = rng.uniform(0.0, 2.0 * np.pi, n)
            return np.column_stack([rho * np.cos(phi), rho * np.sin(phi)])
        z_lo = np.maximum(-self._h - self._zs, -r_out)
        z_hi = np.minimum(self._h - self._zs, r_out)
        env = r_out**2 - r_in**2
        z = np.empty(n)
        todo = np.arange(n)
        while todo.size:
            zc = rng.uniform(z_lo[todo], z_hi[todo])
            area = (r_out[todo] ** 2 - zc**2) - np.maximum(r_in[todo] ** 2 - zc**2, 0.0)
            ok = rng.uniform(0.0, 1.0, todo.size) * env[todo] < area
            z[todo[ok]] = zc[ok]
            todo = todo[~ok]
        a2 = np.maximum(r_in**2 - z**2, 0.0)
        b2 = r_out**2 - z**2
        rho = np.sqrt(a2 + rng.uniform(0.0, 1.0, n) * (b2 - a2))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        return np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])

    # ================================================================ finite disk
    # Inner-loop units: kpc, Gyr. Positions are Earth-centred with Galactocentric axis
    # orientation; absolute Galactocentric position = point + self._gc_earth.
    def _fd_radial_weight(self, pts: np.ndarray) -> np.ndarray:
        """Radial density factor f_R(R) * 1[R <= R_gal] (1 at the Sun's cylindrical radius)."""
        r = np.hypot(pts[:, 0] + self._gc_earth[0], pts[:, 1] + self._gc_earth[1])
        if not self._exponential:
            return (r <= self._rgal).astype(float)
        return np.where(r <= self._rgal, np.exp(-(r - self._re) / self._rd), 0.0)

    def _fd_vertical_weight(self, z_abs: np.ndarray) -> np.ndarray:
        """Vertical density factor f_z(Z) inside |Z| <= z_lim (1 in the mid-plane)."""
        z = np.abs(z_abs)
        if not self._exponential:
            return (z <= self._zlim).astype(float)
        return np.where(z <= self._zlim, np.exp(-z / self._h), 0.0)

    def _fd_density_ratio(self, pts: np.ndarray) -> np.ndarray:
        """rho(x) / rho_0, zero outside the disk (R > R_gal or |Z| > z_lim)."""
        return self._fd_radial_weight(pts) * self._fd_vertical_weight(pts[:, 2] + self._ze)

    def _fd_slab_ball_weight(self, radius: np.ndarray, z_center: np.ndarray) -> np.ndarray:
        """``pi int f_z(Z) (r^2 - (Z - z_c)^2)_+ dZ`` over the vertical domain [kpc^3].

        The f_z-weighted volume of a ball clipped to ``|Z| <= z_lim`` (no radial cut).
        """
        r = np.asarray(radius, float)
        zc = np.asarray(z_center, float)
        if not self._exponential:
            return ball_slab_volume(r, zc, self._zlim)
        lo = np.clip(zc - r, -self._zlim, self._zlim)
        hi = np.clip(zc + r, -self._zlim, self._zlim)
        mid = np.clip(0.0, lo, hi)  # f_z has a kink in the mid-plane
        x, w = _GL16
        out = np.zeros(np.broadcast(r, zc).shape)
        for a, b in ((lo, mid), (mid, hi)):
            width = np.maximum(b - a, 0.0)[..., None]
            z = a[..., None] + width * (x + 1.0) / 2.0
            area = np.maximum(r[..., None] ** 2 - (z - zc[..., None]) ** 2, 0.0)
            piece = np.sum(w * np.exp(-np.abs(z) / self._h) * area, axis=-1)
            out = out + piece * width[..., 0] / 2.0
        return np.pi * out

    def _fd_ball_weight(self, radius: np.ndarray) -> np.ndarray:
        """``int rho / rho_0`` over the ball of given radius about Earth, clipped to the disk."""
        r = np.asarray(radius, float)
        ze = self._ze
        lo = np.clip(ze - r, -self._zlim, self._zlim)
        hi = np.clip(ze + r, -self._zlim, self._zlim)
        mid = np.clip(0.0, lo, hi)
        x, w = _GLZ
        out = np.zeros(r.shape)
        for a, b in ((lo, mid), (mid, hi)):
            width = np.maximum(b - a, 0.0)[..., None]
            z = a[..., None] + width * (x + 1.0) / 2.0
            varrho = np.sqrt(np.maximum(r[..., None] ** 2 - (z - ze) ** 2, 0.0))
            if self._exponential:
                disc = disc_profile_integral(varrho, self._re, self._rgal, self._rd, self._re)
                disc = disc * np.exp(-np.abs(z) / self._h)
            else:
                disc = lens_area(varrho, self._rgal, self._re)
            out = out + np.sum(w * disc, axis=-1) * width[..., 0] / 2.0
        return out

    def _fd_earth_cone(self, offset: float = 0.0) -> float:
        """``int_0^T int rho 1[|x - x_E| <= v s - offset] dx ds`` by deterministic quadrature."""
        v, t = self._v, self._t
        if v == 0.0 or self._g == 0.0 or t == 0.0:
            return 0.0
        edge = np.linspace(self._rgal - self._re, self._rgal + self._re, 9)
        knots_r = np.concatenate(
            [[0.0, abs(self._zlim - self._ze), self._zlim + self._ze, self._re, self._far_e], edge]
        )

        def f(s: np.ndarray) -> np.ndarray:
            r = np.maximum(v * s - offset, 0.0)
            return self._fd_ball_weight(r.ravel()).reshape(r.shape)

        knots = ((knots_r + offset) / v)[None, :]
        val = _gl_piecewise(f, np.array([0.0]), np.array([t]), knots, _GL16)
        return self._rho0 * float(val[0])

    # ---------------------------------------------------- per-target importance sampler
    def _fd_sample_fz(self, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Exact draws with density proportional to f_z on [lo, hi] (inverse CDF)."""
        uu = rng.uniform(0.0, 1.0, lo.shape)
        if not self._exponential:
            return lo + uu * (hi - lo)
        h = self._h
        prim = lambda z: h * np.sign(z) * -np.expm1(-np.abs(z) / h)  # noqa: E731
        g = prim(lo) + uu * (prim(hi) - prim(lo))
        return np.clip(-np.sign(g) * h * np.log1p(-np.abs(g) / h), lo, hi)

    def _fd_sample_balls(
        self, centre_abs: np.ndarray, radius: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """One point per row with density ``f_z(Z) 1[ball] 1[|Z| <= z_lim] / weight``.

        ``centre_abs`` are absolute Galactocentric centres; returns Earth-centred points.
        """
        m = len(radius)
        zc = centre_abs[:, 2]
        lo = np.maximum(-self._zlim, zc - radius)
        hi = np.minimum(self._zlim, zc + radius)
        env = radius**2 - (np.clip(zc, lo, hi) - zc) ** 2
        z = np.empty(m)
        todo = np.arange(m)
        while todo.size:
            cand = self._fd_sample_fz(lo[todo], hi[todo], rng)
            area = radius[todo] ** 2 - (cand - zc[todo]) ** 2
            ok = rng.uniform(0.0, 1.0, todo.size) * env[todo] < area
            z[todo[ok]] = cand[ok]
            todo = todo[~ok]
        rad = np.sqrt(rng.uniform(0.0, 1.0, m) * np.maximum(radius**2 - (z - zc) ** 2, 0.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, m)
        pts = np.column_stack(
            [centre_abs[:, 0] + rad * np.cos(phi), centre_abs[:, 1] + rad * np.sin(phi), z]
        )
        return pts - self._gc_earth

    def finite_disk_reach(self, xyz: u.Quantity) -> FiniteDiskReach:
        """Seeded importance-sampled reach measures of a target set (cached per target set)."""
        if not self.is_finite:
            raise GeometryError(f"finite_disk_reach requires geometry {FINITE_DISK!r}")
        arr = np.ascontiguousarray(u.Quantity(xyz).to_value(u.kpc), dtype=float).reshape(-1, 3)
        key = arr.tobytes()
        cache: dict = self._fd_cache
        if key not in cache:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[key] = self._fd_reach_compute(xyz)
        return cache[key]

    def _fd_reach_compute(self, xyz: u.Quantity) -> FiniteDiskReach:
        """Multiple-importance-sampled ``int rho K dx`` for every target kernel.

        Proposal components are slab-clipped balls (density ``f_z 1[ball] / weight``):
        one about Earth (radius ``v T``) and one about each target (its kernel support
        ``v (T - tau_i)``), each capped at the farthest point of the disk. Samples are
        combined with the balance heuristic, ``w(x) = rho(x) / sum_k N_k q_k(x)``; the
        vertical profile cancels exactly and points with ``R > R_gal`` carry zero weight.
        """
        pos, _d3, tau = self._positions(xyz)
        n = len(pos)
        v, t = self._v, self._t
        n_total = int(self.n_importance)
        seed = int(self.importance_seed)
        zeros = np.zeros(n)
        if v == 0.0 or self._g == 0.0 or t == 0.0:
            return FiniteDiskReach(
                zeros, zeros, zeros, zeros, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, seed
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, _FD_STREAM]))
        earth = self._gc_earth
        centre = pos + earth  # absolute Galactocentric target positions
        c_r = np.hypot(centre[:, 0], centre[:, 1])
        far = np.hypot(c_r + self._rgal, self._zlim + np.abs(centre[:, 2]))
        rc = np.minimum(v * np.maximum(t - tau, 0.0), far)  # kernel support radius, capped
        wgt = self._fd_slab_ball_weight(rc, centre[:, 2]) if n else zeros
        valid = np.flatnonzero((rc > 0.0) & (wgt > 0.0))
        rc_e = min(v * t, self._far_e)
        wgt_e = float(self._fd_slab_ball_weight(np.array([rc_e]), np.array([self._ze]))[0])

        n_e = n_total if valid.size == 0 else max(n_total // 8, 2)
        n_t = n_total - n_e if valid.size else 0
        coef = np.zeros(n)
        if n_t:
            coef[valid] = (n_t / valid.size) / wgt[valid]
        coef_e = n_e / wgt_e
        tol = 1.0 + 1e-12  # a sample sits on its own ball's boundary only up to rounding

        sum_u, sum_c = np.zeros(n), np.zeros(n)
        var_u, var_c = np.zeros(n), np.zeros(n)
        tot = {"u": 0.0, "c": 0.0, "e": 0.0}
        tot_var = {"u": 0.0, "c": 0.0, "e": 0.0}
        chunk = max(16, int(4.0e5 // max(n, 1)))
        strata = [(np.tile(earth, (n_e, 1)), np.full(n_e, rc_e))]
        if n_t:
            pick = valid[rng.integers(0, valid.size, n_t)]
            strata.append((centre[pick], rc[pick]))
        for cen, rad in strata:
            m_s = len(rad)
            pts_all = self._fd_sample_balls(cen, rad, rng)
            s1_u, s1_c = np.zeros(n), np.zeros(n)
            s2_u, s2_c = np.zeros(n), np.zeros(n)
            row1 = {"u": 0.0, "c": 0.0, "e": 0.0}
            row2 = {"u": 0.0, "c": 0.0, "e": 0.0}
            # Samples beyond the Galactic edge have zero density: they stay in the stratum
            # size m_s (as exact zeros) but need no pair distances.
            f_r_all = self._fd_radial_weight(pts_all)
            inside = f_r_all > 0.0
            pts_all, f_r_all = pts_all[inside], f_r_all[inside]
            for i0 in range(0, len(pts_all), chunk):
                pts = pts_all[i0 : i0 + chunk]
                d_e = np.linalg.norm(pts, axis=1)
                dens = coef_e * (d_e <= rc_e * tol)
                if n:
                    diff = pts[:, None, :] - pos[None, :, :]
                    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
                    dens = dens + (d <= rc[None, :] * tol) @ coef
                w = np.divide(
                    self._rho0 * f_r_all[i0 : i0 + chunk],
                    dens,
                    out=np.zeros(len(pts)),
                    where=dens > 0.0,
                )
                k_e = w * np.maximum(t - d_e / v, 0.0)
                row1["e"] += float(k_e.sum())
                row2["e"] += float((k_e * k_e).sum())
                if not n:
                    continue
                arrive = d / v + tau[None, :]
                y_u = w[:, None] * np.maximum(t - arrive, 0.0)
                y_c = w[:, None] * np.maximum(np.minimum(t, d_e / v)[:, None] - arrive, 0.0)
                s1_u += y_u.sum(axis=0)
                s1_c += y_c.sum(axis=0)
                s2_u += (y_u * y_u).sum(axis=0)
                s2_c += (y_c * y_c).sum(axis=0)
                for key, y in (("u", y_u), ("c", y_c)):
                    rows = y.sum(axis=1)
                    row1[key] += float(rows.sum())
                    row2[key] += float((rows * rows).sum())
            fac = m_s / (m_s - 1.0)  # N_k x sample variance of an i.i.d. stratum
            sum_u += s1_u
            sum_c += s1_c
            var_u += fac * np.maximum(s2_u - s1_u**2 / m_s, 0.0)
            var_c += fac * np.maximum(s2_c - s1_c**2 / m_s, 0.0)
            for key in tot:
                tot[key] += row1[key]
                tot_var[key] += fac * max(row2[key] - row1[key] ** 2 / m_s, 0.0)
        return FiniteDiskReach(
            lam_unconditional=sum_u,
            lam_unconditional_se=np.sqrt(var_u),
            lam_conditional=sum_c,
            lam_conditional_se=np.sqrt(var_c),
            sum_unconditional=tot["u"],
            sum_unconditional_se=float(np.sqrt(tot_var["u"])),
            sum_conditional=tot["c"],
            sum_conditional_se=float(np.sqrt(tot_var["c"])),
            earth_lambda_mc=tot["e"],
            earth_lambda_mc_se=float(np.sqrt(tot_var["e"])),
            n_samples=n_e + n_t,
            seed=seed,
        )

    def expected_reached_se(self, xyz: u.Quantity, condition_on_earth_unreached: bool) -> float:
        """Monte Carlo standard error of ``expected_reached`` (finite disk; 0 otherwise).

        First-order propagation: the standard error of ``sum_i Lambda_i`` scaled by
        ``E[N] / sum_i Lambda_i`` (exact while every ``Lambda_i << 1``).
        """
        if not self.is_finite:
            return 0.0
        r = self.finite_disk_reach(xyz)
        if condition_on_earth_unreached:
            lam, total, se = r.lam_conditional, r.sum_conditional, r.sum_conditional_se
        else:
            lam, total, se = r.lam_unconditional, r.sum_unconditional, r.sum_unconditional_se
        if total <= 0.0:
            return 0.0
        return float(se * np.sum(-np.expm1(-lam)) / total)

    def origin_rate_density(self, xyz: u.Quantity) -> u.Quantity:
        """Volumetric origin rate ``rho(x)`` at Earth-centred Galactic positions (finite disk).

        Zero outside the disk (``R > R_gal`` or ``|Z|`` beyond the vertical domain).
        """
        if not self.is_finite:
            raise GeometryError(f"origin_rate_density requires geometry {FINITE_DISK!r}")
        pos, _, _ = self._positions(xyz)
        return self._rho0 * self._fd_density_ratio(pos) / (u.kpc**3 * u.Gyr)

    def finite_disk_summary(self) -> dict:
        """JSON-ready description of the finite-disk geometry and its normalisation."""
        if not self.is_finite:
            raise GeometryError(f"finite_disk_summary requires geometry {FINITE_DISK!r}")
        if self._exponential:
            norm = (
                "rho(R, Z) = Gamma_2d * exp(-(R - R_E) / R_d) * exp(-|Z| / h_z) / "
                "(2 h_z (1 - exp(-z_max / h_z))) for R <= R_gal, |Z| <= z_max; "
                "int rho(R_E, Z) dZ = Gamma_2d"
            )
        else:
            norm = (
                "rho = Gamma_2d / (2 h_z) for R <= R_gal, |Z| <= h_z; int rho dZ = Gamma_2d "
                "(identical to thick_disk_3d inside the disk)"
            )
        return {
            "disk_profile": self.disk_profile,
            "galactic_disk_radius_kpc": self._rgal,
            "solar_galactocentric_distance_kpc": float(
                self.solar_galactocentric_radius.to_value(u.kpc)
            ),
            "solar_cylindrical_radius_kpc": self._re,
            "solar_height_kpc": self._ze,
            "scale_height_kpc": self._h,
            "vertical_half_extent_kpc": self._zlim,
            "radial_scale_length_kpc": self._rd if self._exponential else None,
            "midplane_rate_density_at_sun_per_kpc3_gyr": self._rho0,
            "density_normalisation": norm,
            "galcen_icrs_deg": list(GALCEN_ICRS_DEG),
            "earth_cone_method": "deterministic Gauss-Legendre quadrature (age x height x arc)",
        }

    # ------------------------------------------------------- correlated Monte Carlo
    def _fd_proposal(self, d_max: float, core_offset: float | None) -> tuple[ExpansionModel, float]:
        """Homogeneous thick-slab proposal process covering every relevant origin.

        Ages are capped where the near-miss shell has left the disk entirely; the slab
        half-thickness is the vertical domain bound; the rate is the supremum of the
        disk density within reach, so thinning by ``rho / rho_proposal`` is exact.
        """
        v, t = self._v, self._t
        reach = max(d_max, core_offset or 0.0)
        t_prop = min(t, (self._far_e + reach) / v)
        r_far = v * t_prop + d_max
        bound = 1.0
        if self._exponential:
            bound = float(np.exp((self._re - max(0.0, self._re - r_far)) / self._rd))
        prop = ExpansionModel(
            "thick_disk_3d",
            origin_rate=(self._rho0 * bound * 2.0 * self._zlim) / (u.kpc**2 * u.Gyr),
            expansion_speed=self._beta * u.dimensionless_unscaled,
            lookback=t_prop * u.Gyr,
            scale_height=self._zlim * u.kpc,
            solar_height=self._ze * u.kpc,
            light_travel_delay=self.light_travel_delay,
        )
        return prop, bound

    def _fd_zero_detection(
        self,
        pos: np.ndarray,
        tau: np.ndarray,
        p: np.ndarray,
        log1m: np.ndarray,
        rng: np.random.Generator,
        condition_on_earth_unreached: bool,
        n_mc: int,
        p_earth: float,
        max_points: float,
    ) -> ZeroDetectionResult:
        """Finite-disk analogue of the shell Monte Carlo (thinned homogeneous proposal)."""
        dist = np.linalg.norm(pos, axis=1)
        d_max = float(dist.max()) if len(dist) else 0.0
        p_e = 0.0 if condition_on_earth_unreached else float(p_earth)
        core_offset = None
        mu_core, mu_rim, bound = 0.0, 0.0, 1.0
        prop = None
        if self._v > 0.0 and self._g > 0.0 and self._t > 0.0:
            if not condition_on_earth_unreached:
                core_offset = float(np.max(dist + self._v * tau)) if len(dist) else 0.0
                mu_core = self._fd_earth_cone(core_offset)
            prop, bound = self._fd_proposal(d_max, core_offset)
            mu_rim = prop._integrate_ball(d_max, +1.0)
            if condition_on_earth_unreached:
                mu_rim -= prop._integrate_ball(0.0, +1.0)
            else:
                mu_rim -= prop._integrate_ball(core_offset, -1.0)
            mu_rim = max(mu_rim, 0.0)

        g_all_t = float(np.exp(np.sum(log1m))) if len(p) else 1.0
        g_all = g_all_t * (1.0 - p_e)
        p_core = float(-np.expm1(-mu_core))
        w_rim = float(-np.expm1(-mu_rim))

        g_joint, g_targ, counts = np.ones(0), np.ones(0), np.zeros(0)
        n_points = n_accepted = 0
        if w_rim > 0.0 and n_mc > 0 and prop is not None:
            mean_pts = mu_rim / w_rim
            if n_mc * mean_pts > max_points:
                raise ComputationalBudgetError(
                    f"Expected {n_mc * mean_pts:.3g} proposal origins exceeds budget "
                    f"{max_points:.3g} (reduce lookback, n_mc, or raise max_points)"
                )
            per_chunk = int(max(1, 2.0e6 // max(1.0, mean_pts * max(1, len(p)))))
            gj, gt, cn = [], [], []
            done = 0
            while done < n_mc:
                k = min(per_chunk, n_mc - done)
                a, b, c, npts, nacc = self._fd_mc_chunk(
                    k, mu_rim, pos, tau, log1m, p_e, d_max, core_offset, rng, prop, bound
                )
                gj.append(a)
                gt.append(b)
                cn.append(c)
                n_points += npts
                n_accepted += nacc
                done += k
            g_joint, g_targ, counts = np.concatenate(gj), np.concatenate(gt), np.concatenate(cn)

        def combine(g_bar: float, g_core: float) -> float:
            return 1.0 - p_core * (1.0 - g_core) - (1.0 - p_core) * w_rim * (1.0 - g_bar)

        mean_or_one = lambda arr: float(arr.mean()) if arr.size else 1.0  # noqa: E731
        scale = (1.0 - p_core) * w_rim
        return ZeroDetectionResult(
            p_zero=float(combine(mean_or_one(g_joint), g_all)),
            p_zero_se=float(scale * g_joint.std(ddof=1) / np.sqrt(g_joint.size))
            if g_joint.size > 1
            else 0.0,
            p_zero_targets_only=float(combine(mean_or_one(g_targ), g_all_t)),
            mc_expected_reached=float(
                p_core * len(p) + scale * (float(counts.mean()) if counts.size else 0.0)
            ),
            mc_expected_reached_se=float(scale * counts.std(ddof=1) / np.sqrt(counts.size))
            if counts.size > 1
            else 0.0,
            mu_rim=float(mu_rim),
            mu_core=float(mu_core),
            n_mc=int(n_mc),
            n_points=int(n_points),
            conditioned_on_earth_unreached=condition_on_earth_unreached,
            n_points_accepted=int(n_accepted),
            proposal_density_bound=float(bound),
        )

    def _fd_mc_chunk(self, k, mu, pos, tau, log1m, p_e, d_max, core_offset, rng, prop, bound):
        counts = _zero_truncated_poisson(mu, k, rng)
        total = int(counts.sum())
        s = prop._sample_ages(total, d_max, core_offset, rng)
        r_in, r_out = prop._shell_radii(s, d_max, core_offset)
        pts = prop._sample_shell_points(s, r_in, r_out, rng)
        ratio = self._fd_density_ratio(pts)
        if np.any(ratio > bound * (1.0 + 1e-9)):
            raise AssertionError("disk density exceeds the proposal bound; thinning invalid")
        keep = rng.uniform(0.0, 1.0, total) * bound < ratio  # exact Poisson thinning
        owner = np.repeat(np.arange(k), counts)[keep]
        pts, s = pts[keep], s[keep]
        logg = np.zeros(k)
        n_reached = np.zeros(k)
        earth_any = np.zeros(k, dtype=bool)
        if owner.size:
            real, start = np.unique(owner, return_index=True)  # owner is non-decreasing
            earth_hit = np.linalg.norm(pts, axis=1) <= self._v * s
            earth_any[real] = np.logical_or.reduceat(earth_hit, start)
            if len(pos):
                sep = np.linalg.norm(pts[:, None, :] - pos[None, :, :], axis=2)
                hit = sep <= self._v * (s[:, None] - tau[None, :])
                reached = np.logical_or.reduceat(hit, start, axis=0)
                logg[real] = np.where(reached, log1m[None, :], 0.0).sum(axis=1)
                n_reached[real] = reached.sum(axis=1)
        g_t = np.exp(logg)
        g_j = g_t * (1.0 - p_e * earth_any)
        return g_j, g_t, n_reached, total, int(owner.size)


def _zero_truncated_poisson(mu: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Exact draws from Poisson(mu) conditioned on >= 1."""
    if mu > 1.0:
        out = rng.poisson(mu, size)
        bad = out == 0
        while bad.any():
            out[bad] = rng.poisson(mu, int(bad.sum()))
            bad = out == 0
        return out
    kmax = 60  # tail mass beyond 60 is < mu^60 / 60! for mu <= 1
    k = np.arange(1, kmax + 1)
    logp = k * np.log(mu) - np.cumsum(np.log(k))
    pk = np.exp(logp - logp.max())
    pk /= pk.sum()
    return rng.choice(k, size=size, p=pk)
