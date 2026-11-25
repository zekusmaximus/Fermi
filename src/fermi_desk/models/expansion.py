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

An origin of age ``s`` (founded ``s`` before the present) occupies a ball of radius
``v s``. A target at distance ``d`` is observed as it was ``tau = d / c`` ago
(light-travel delay), so it is reached iff an origin lies inside its backward cone
``|x0 - x_i| <= v (s - tau_i)``.

Earth-unreached conditioning is exact: conditioning a Poisson process on having
no points in Earth's cone ``C_E`` leaves a Poisson process on the complement, so
``P(i reached | E unreached) = 1 - exp(-Lambda(C_i minus C_E))``.

The geometry is local and translation invariant (no finite Galactic radius); it
is only self-consistent while ``v T`` is small compared with the Sun's
Galactocentric radius. ``local_approximation_valid`` reports this.

Inner loops strip units after explicit conversion to: kpc, Gyr, kpc/Gyr.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import astropy.constants as const
import astropy.units as u
import numpy as np

from fermi_desk.errors import ComputationalBudgetError, GeometryError, UnitValidationError

GEOMETRY_MODES = ("thin_disk_2d", "thick_disk_3d")
C_KPC_PER_GYR = float(const.c.to_value(u.kpc / u.Gyr))

_GL8 = np.polynomial.legendre.leggauss(8)  # exact for piecewise cubics in s
_GL16 = np.polynomial.legendre.leggauss(16)
_GLZ = np.polynomial.legendre.leggauss(24)
# Geometric sub-division of the post-contact interval: resolves the crescent onset
# (scale d / v) even when T >> d / v.
_GEOM_FRACTIONS = np.concatenate([[0.0], np.geomspace(1e-9, 1.0, 28)])


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


@dataclass(frozen=True)
class ExpansionModel:
    """Settlement-front kernel. All inputs are astropy Quantities; geometry is explicit."""

    geometry: str
    origin_rate: u.Quantity
    expansion_speed: u.Quantity
    lookback: u.Quantity
    scale_height: u.Quantity | None = None
    solar_height: u.Quantity | None = None
    light_travel_delay: bool = True
    local_validity_radius: u.Quantity = field(default_factory=lambda: 8.0 * u.kpc)

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
        if self.geometry == "thick_disk_3d":
            if self.scale_height is None or self.solar_height is None:
                raise GeometryError("thick_disk_3d requires explicit scale_height and solar_height")
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

    # ------------------------------------------------------------- basic props
    @property
    def is_3d(self) -> bool:
        return self.geometry == "thick_disk_3d"

    @property
    def rate_density(self) -> float:
        """Origins per kpc^2 Gyr (2D) or per kpc^3 Gyr inside the slab (3D)."""
        return self._g / (2.0 * self._h) if self.is_3d else self._g

    @property
    def max_bubble_radius(self) -> u.Quantity:
        return (self._v * self._t) * u.kpc

    @property
    def local_approximation_valid(self) -> bool:
        return bool(self.max_bubble_radius <= self.local_validity_radius)

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
        """(model-frame positions relative to Earth [kpc], distance used for delay, delay [Gyr])."""
        arr = np.asarray(u.Quantity(xyz).to_value(u.kpc), float).reshape(-1, 3)
        d3 = np.linalg.norm(arr, axis=1)
        tau = d3 / C_KPC_PER_GYR if self.light_travel_delay else np.zeros_like(d3)
        pos = arr if self.is_3d else arr[:, :2]
        return pos, d3, tau

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
        return float(self._cone_lambda(np.array([self._zs]), np.array([self._t]))[0])

    def earth_reach_probability(self) -> float:
        return float(-np.expm1(-self.earth_lambda()))

    def log_p_earth_unreached(self) -> float:
        return -self.earth_lambda()

    def target_lambda(self, xyz: u.Quantity) -> np.ndarray:
        """Unconditional cone measure for each target (light-travel delay applied)."""
        pos, _, tau = self._positions(xyz)
        z_mid = pos[:, 2] + self._zs if self.is_3d else np.zeros(len(pos))
        return self._cone_lambda(z_mid, self._t - tau)

    # --------------------------------------------------------- conditional reach
    def target_lambda_given_earth_unreached(self, xyz: u.Quantity) -> np.ndarray:
        """Measure of C_i minus C_E: origins reaching target i but not Earth."""
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
