"""Model regression, deterministic replay, and IR calibration reversal (CI items 6-7)."""

from __future__ import annotations

import json

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.coordinates import ICRS, Galactocentric, SkyCoord
from typer.testing import CliRunner

from conftest import ROOT, frac, rate
from fermi_desk.cli import app
from fermi_desk.errors import GeometryError, UnitValidationError
from fermi_desk.inference.engine import (
    RunConfig,
    build_context,
    ir_reversal_sweep,
    results_digest,
    run_comparison,
)
from fermi_desk.models import infrared, radio, solar_system
from fermi_desk.models.expansion import (
    C_KPC_PER_GYR,
    GALCEN_ICRS_DEG,
    ExpansionModel,
    ball_slab_volume,
    disc_profile_integral,
    galactocentric_positions,
    lens_area,
)
from fermi_desk.queue.runners import run_lookback_window_t_sweep

H_Z, Z_SUN = 300 * u.pc, 20.8 * u.pc


def model(geometry, gamma, v, t_gyr=1e-3, **kw):
    return ExpansionModel(geometry, rate(gamma), frac(v), t_gyr * u.Gyr, H_Z, Z_SUN, **kw)


# ----------------------------------------------------------------- geometry kernels
def test_geometry_primitives():
    assert ball_slab_volume(1.0, 0.0, 10.0) == pytest.approx(4 / 3 * np.pi)
    assert ball_slab_volume(1.0, 1.0, 1.0) == pytest.approx(2 / 3 * np.pi)  # half ball
    assert ball_slab_volume(5.0, 0.0, 1.0) == pytest.approx(np.pi * (25 * 2 - 2 / 3))
    assert lens_area(1.0, 2.0, 0.5) == pytest.approx(np.pi)  # contained
    assert lens_area(1.0, 1.0, 2.5) == 0.0
    # equal unit circles at separation 1: 2 pi / 3 - sqrt(3) / 2
    assert lens_area(1.0, 1.0, 1.0) == pytest.approx(2 * np.pi / 3 - np.sqrt(3) / 2)


def test_analytic_cone_measures():
    g, beta, t = 4.0, 0.005, 1e-3
    v = beta * C_KPC_PER_GYR
    assert model("thin_disk_2d", g, beta, t).earth_lambda() == pytest.approx(
        g * np.pi * v**2 * t**3 / 3, rel=1e-12
    )
    beta = 1e-4
    v = beta * C_KPC_PER_GYR
    lam3 = g / (2 * H_Z.to_value(u.kpc)) * np.pi * v**3 * t**4 / 3
    assert model("thick_disk_3d", g, beta, t).earth_lambda() == pytest.approx(lam3, rel=1e-12)


def test_light_travel_delay_hides_targets_beyond_ct():
    m = model("thin_disk_2d", 1e3, 0.5, t_gyr=1e-9)  # T = 1 yr, c T ~ 0.31 pc
    xyz = np.array([[0.1, 0, 0], [1.0, 0, 0]]) * u.pc
    lam = m.target_lambda(xyz)
    assert lam[0] > 0 and lam[1] == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("geometry", ["thin_disk_2d", "thick_disk_3d"])
@pytest.mark.parametrize(("gamma", "v"), [(4.0, 0.005), (4e-4, 0.5)])
def test_quadrature_matches_correlated_monte_carlo(geometry, gamma, v, small_targets):
    """Two independent methods for E[N_reached | Earth unreached] must agree."""
    m = model(geometry, gamma, v)
    quad = m.expected_reached(small_targets.xyz, condition_on_earth_unreached=True)
    mc = m.zero_detection_probability(
        small_targets.xyz,
        np.zeros(3),
        rng=np.random.default_rng(2024),
        condition_on_earth_unreached=True,
        n_mc=60000,
    )
    assert abs(mc.mc_expected_reached - quad) < 4 * mc.mc_expected_reached_se + 1e-3 * quad


@pytest.mark.slow
@pytest.mark.parametrize("geometry", ["thin_disk_2d", "thick_disk_3d"])
@pytest.mark.parametrize("conditioned", [True, False])
def test_exact_p0_respects_fkg_and_markov_bounds(geometry, conditioned, bl_targets):
    m = model(geometry, 4.0, 0.005)
    p = np.full(len(bl_targets), 0.05)
    q = m.reach_probabilities(bl_targets.xyz, conditioned)
    b = radio.zero_detection_bounds(p, q)
    mc = m.zero_detection_probability(
        bl_targets.xyz,
        p,
        rng=np.random.default_rng(11),
        condition_on_earth_unreached=conditioned,
        n_mc=4000,
    )
    tol = 4 * mc.p_zero_se
    assert b.markov <= b.independent + 1e-15
    assert b.independent <= mc.p_zero_targets_only + tol
    assert mc.p_zero_targets_only <= b.frechet_upper + tol


def test_solar_system_joint_mc_matches_analytic_without_targets():
    m = model("thin_disk_2d", 4.0, 0.005)
    res = m.zero_detection_probability(
        np.zeros((0, 3)) * u.pc,
        np.zeros(0),
        rng=np.random.default_rng(0),
        condition_on_earth_unreached=False,
        n_mc=100,
        p_earth=0.5,
    )
    expect = solar_system.null_likelihood(
        m.earth_reach_probability(), 0.5, conditioned_on_earth_unreached=False
    )
    assert res.p_zero == pytest.approx(expect, abs=1e-14)
    assert solar_system.null_likelihood(0.3, 0.5, conditioned_on_earth_unreached=True) == 1.0


# --------------------------------------------------------------- radio sensitivity
def test_radiometer_reproduces_enriquez_2017_eirp(graph):
    rs = graph.surveys["bl_enriquez_2017_692"].radio_sensitivity
    rad = radio.Radiometer(
        rs.sefd, rs.snr_threshold, rs.channel_bandwidth, rs.integration_time, rs.n_pol
    )
    f_min = radio.min_detectable_flux(rad)
    # S_min = 17 Jy for a 1 Hz transmitter (Sec. 5.1)
    assert (f_min / (1 * u.Hz)).to_value(u.Jy) == pytest.approx(17.0, rel=0.05)
    eirp = radio.min_detectable_eirp(rs.reference_distance_pc * u.pc, f_min)
    assert eirp.to_value(u.W) == pytest.approx(rs.reference_eirp_w, rel=0.05)


def test_hip_1692_gaia_revision_raises_eirp_min_over_1000x(graph, bl_targets):
    df = pd.read_csv(ROOT / graph.surveys["bl_enriquez_2017_692"].catalog_path)
    row = df.loc[df.target_name == "HIP 1692"].iloc[0]
    f_min = 1e-25 * u.W / u.m**2
    ratio = radio.min_detectable_eirp(row.dist_gaia_pc * u.pc, f_min) / radio.min_detectable_eirp(
        row.dist_legacy_pc * u.pc, f_min
    )
    assert row.dist_legacy_pc == 23.0
    assert ratio.decompose().value > 1000
    claim = yaml.safe_load((ROOT / "data/adversary_claims/hip_1692.yaml").read_text())
    labels = {d["label"]: d["value_pc"] for d in claim["distances"]}
    assert labels["gaia_dr2_bailer_jones_2018_r_est"] == pytest.approx(752.61, abs=0.01)


def test_eirp_mode_is_monotone_in_distance():
    lf = radio.LogNormalEIRP(1e13 * u.W, 1.0)
    rad = radio.Radiometer(10 * u.Jy, 25.0, 3 * u.Hz, 300 * u.s, 2)
    d = np.array([5.0, 50.0, 500.0]) * u.pc
    p = radio.detection_probabilities("distance_scaled_eirp", 0.1, d, rad, lf)
    assert np.all(np.diff(p) < 0) and p[0] <= 0.1
    assert np.all(radio.detection_probabilities("flat_probability", 0.1, d) == 0.1)


# ------------------------------------------------------------ IR reversal sweep
def test_ir_calibration_sweep_reverses_preference(graph):
    ctx = build_context(
        graph, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=1)
    )
    sw = ir_reversal_sweep(ctx, "quiet_expanders", "partial_dyson_conspicuous")
    assert sw["n_calibrations"] == 15
    assert sw["min_lr"] < 1.0 < sw["max_lr"]
    assert sw["preference_reverses"] is True
    assert sw["calibration_data_tier"] == "illustrative_calibration"


def test_ir_reversal_tracks_false_positive_rate_not_ids():
    """Reversal is driven by expected false positives vs. observed count, nothing else."""
    k, n = 7, 200_000
    quiet, loud = np.array([0.0]), np.array([1e-5])
    low = [infrared.IRCalibration("lo", 0.5, 1e-7)]
    high = [infrared.IRCalibration("hi", 0.5, 1e-4)]
    assert infrared.sweep_calibrations(k, n, quiet, loud, low).max_lr < 1.0
    assert infrared.sweep_calibrations(k, n, quiet, loud, high).min_lr > 1.0


# ---------------------------------------------------------- deterministic replay
def _run_cli(tmp_path, sub, seed=42, n_mc=2000):
    out = tmp_path / sub
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--geometry",
            "thin_disk_2d",
            "--radio-mode",
            "flat_probability",
            "--seed",
            str(seed),
            "--n-mc",
            str(n_mc),
            "--out-dir",
            str(out),
            "--root",
            str(ROOT),
        ],
    )
    assert result.exit_code == 0, result.output
    (path,) = list(out.glob("*.json"))
    return json.loads(path.read_text())


def test_compare_replay_is_byte_identical(tmp_path):
    a = _run_cli(tmp_path, "a")
    b = _run_cli(tmp_path, "b")
    assert json.dumps(a["results"], sort_keys=True) == json.dumps(b["results"], sort_keys=True)
    assert a["results_sha256"] == b["results_sha256"]
    assert a["rng_seed"] == b["rng_seed"] == 42
    for key in ("run_id", "git_sha", "geometry_mode", "radio_mode", "active_surveys"):
        assert key in a
    assert "published_catalog" in a["data_tiers_used"]


@pytest.mark.slow
def test_seed_actually_drives_monte_carlo(tmp_path):
    a = _run_cli(tmp_path, "s1", seed=1)
    b = _run_cli(tmp_path, "s2", seed=2)
    ha = a["results"]["hypotheses"]["frequent_slow_expanders"]["radio"]
    hb = b["results"]["hypotheses"]["frequent_slow_expanders"]["radio"]
    assert ha["p_zero_exact_mc"] != hb["p_zero_exact_mc"]
    assert ha["expected_reached"] == hb["expected_reached"]  # quadrature: seed-free


def test_compare_requires_explicit_geometry():
    result = CliRunner().invoke(
        app, ["compare", "--radio-mode", "flat_probability", "--seed", "42"]
    )
    assert result.exit_code != 0


def test_no_hypothesis_specific_branches_in_kernels():
    """AGENTS.md law 2: benchmark ids / star names never appear in physics or engine code."""
    src = ROOT / "src" / "fermi_desk"
    forbidden = (
        "frequent_slow",
        "rare_fast",
        "quiet_expanders",
        "partial_dyson",
        "HIP 1692",
        "hip_1692",
    )
    # queue/store.py and cli.py hold job / CLI *configuration* (default pairs), not physics.
    scanned = [
        *(src / "models").glob("*.py"),
        *(src / "inference").glob("*.py"),
        src / "queue" / "runners.py",
    ]
    for path in scanned:
        text = path.read_text(encoding="utf-8")
        assert not any(f in text for f in forbidden), path


def test_unconditioned_run_includes_solar_system_jointly(graph):
    cfg = RunConfig(
        geometry="thin_disk_2d",
        radio_mode="flat_probability",
        seed=5,
        n_mc=500,
        condition_on_earth_unreached=False,
    )
    rec = run_comparison(graph, cfg, ["frequent_slow_expanders"])
    h = rec["results"]["hypotheses"]["frequent_slow_expanders"]
    # Joint radio+SS term is used for the hypothesis's own joint; marginals are kept for
    # channel-intersection Bayes factors but never double-entered.
    assert h["own_joint"]["terms"] == ["radio+solar_system", "infrared"]
    assert {"radio", "solar_system", "radio+solar_system", "infrared"} == set(h["components"])
    assert "earth_unreached" not in h["components"]


# ================================================================== v0.2.0 finite disk
R_GAL, R_0, R_D = 15.0 * u.kpc, 8.178 * u.kpc, 2.6 * u.kpc
# docs/PLANNING_LEDGER.md Sec. 3 (seed 42). Recorded under receipt schema v4; receipt schema
# v5 (v0.2.1) adds merge-policy fields OUTSIDE `results`, so the hashes must not move.
CANONICAL_SHA256 = {
    "thin_disk_2d": "4b70118aaded7861cb14d77379843dfacec2d4c517bbce5efc9654e139fbc269",
    "thick_disk_3d": "be2053912fc0a1d6970878a473dc3ee37d92d62f7904e73dfe369762634f5d41",
    "finite_disk_3d": "87a746589ac07447fdad129f335cd32f164dc11e170220f7a7a81ce0b13b9422",
}


def fd_model(gamma, v, t_gyr=1e-3, profile="uniform", n_importance=8192, seed=42):
    return ExpansionModel(
        "finite_disk_3d",
        rate(gamma),
        frac(v),
        t_gyr * u.Gyr,
        H_Z,
        Z_SUN,
        disk_profile=profile,
        disk_radius=R_GAL,
        solar_galactocentric_radius=R_0,
        radial_scale_length=R_D,
        importance_seed=seed,
        n_importance=n_importance,
    )


@pytest.mark.parametrize(
    "geometry",
    [
        "thin_disk_2d",
        pytest.param("thick_disk_3d", marks=pytest.mark.slow),
        pytest.param("finite_disk_3d", marks=pytest.mark.slow),  # v0.2.0 uniform profile
    ],
)
def test_canonical_v01x_baselines_are_byte_identical(graph, geometry):
    """Zero regression: default single-T runs reproduce the locked v0.1.x / v0.2.0 hashes."""
    cfg = RunConfig(
        geometry=geometry,
        radio_mode="flat_probability",
        seed=42,
        headline_pair=("frequent_slow_expanders", "rare_fast_expanders"),
        ir_pair=("quiet_expanders", "partial_dyson_conspicuous"),
    )
    rec = run_comparison(graph, cfg)
    assert results_digest(rec) == CANONICAL_SHA256[geometry]
    # Receipt schema v5: merge fields live beside `results`, never inside it.
    assert rec["receipt_schema_version"] == 5
    assert rec["pointing_merge_policy"] == "reject_duplicate_pointing_group"
    # The default policy never merges: a declared extension is held out and listed.
    assert rec["merge_provenance"] is None
    assert [h["survey_id"] for h in rec["held_out_extension_surveys"]] == [
        s.id for s in graph.surveys.values() if s.is_pointing_extension
    ]
    if geometry == "thin_disk_2d":
        assert rec["headline_log10_bf"] == pytest.approx(-8.04966e-05, rel=1e-6)
        assert rec["headline_policy"] == "verified_channels_only"


def test_disc_profile_integral_matches_lens_area_and_closed_form():
    rho = np.array([0.0, 0.01, 1.0, 5.0, 6.8, 6.9, 8.0, 8.2, 10.0, 20.0, 23.1, 23.2, 30.0])
    flat = disc_profile_integral(rho, 8.178, 15.0, np.inf, 8.178)
    assert np.allclose(flat, lens_area(rho, 15.0, 8.178), rtol=1e-8, atol=1e-12)
    # exponential weight over a disc that swallows the whole Galaxy: closed form
    r_d, r_ref, r_gal = 2.6, 8.178, 15.0
    a = r_gal / r_d
    total = 2 * np.pi * np.exp(r_ref / r_d) * r_d**2 * (1 - np.exp(-a) * (1 + a))
    got = disc_profile_integral(np.array([r_ref + r_gal, 40.0]), r_ref, r_gal, r_d, r_ref)
    assert np.allclose(got, total, rtol=1e-9)
    # a small disc at the reference radius sees weight ~1; the integral grows with radius
    small = disc_profile_integral(np.array([1e-3]), r_ref, r_gal, r_d, r_ref)[0]
    assert small == pytest.approx(np.pi * 1e-6, rel=1e-6)
    grid = disc_profile_integral(np.linspace(0.0, 25.0, 60), r_ref, r_gal, r_d, r_ref)
    assert np.all(np.diff(grid) >= -1e-9)


def test_galactocentric_transform_matches_astropy_from_icrs(graph, bl_targets):
    """The model-frame positions equal a direct astropy ICRS -> Galactocentric transform of
    the catalog (ra, dec, distance) with R_0 = 8.178 kpc, z_sun = 20.8 pc (offline math)."""
    df = pd.read_csv(ROOT / graph.surveys["bl_enriquez_2017_692"].catalog_path)
    frame = Galactocentric(
        galcen_coord=ICRS(ra=GALCEN_ICRS_DEG[0] * u.deg, dec=GALCEN_ICRS_DEG[1] * u.deg),
        galcen_distance=R_0,
        z_sun=Z_SUN,
        roll=0 * u.deg,
    )
    direct = SkyCoord(
        ra=df.ra_deg.to_numpy() * u.deg,
        dec=df.dec_deg.to_numpy() * u.deg,
        distance=df.dist_pc.to_numpy() * u.pc,
        frame="icrs",
    ).transform_to(frame)
    want = np.column_stack(
        [direct.x.to_value(u.kpc), direct.y.to_value(u.kpc), direct.z.to_value(u.kpc)]
    )
    got = galactocentric_positions(bl_targets.xyz, R_0, Z_SUN)
    assert got.unit == u.kpc and got.shape == (692, 3)
    assert np.max(np.abs(got.to_value(u.kpc) - want)) < 1e-9
    earth = galactocentric_positions(np.zeros((1, 3)) * u.pc, R_0, Z_SUN).to_value(u.kpc)[0]
    r_e = np.sqrt(8.178**2 - 0.0208**2)
    assert earth == pytest.approx([-r_e, 0.0, 0.0208], abs=1e-12)
    assert r_e == pytest.approx(8.17797, abs=1e-5)
    # distances are preserved (rotation + translation)
    d = np.linalg.norm(got.to_value(u.kpc) - earth, axis=1)
    assert np.allclose(d, bl_targets.distance.to_value(u.kpc), rtol=1e-12)
    with pytest.raises(UnitValidationError):
        galactocentric_positions(np.zeros((1, 3)), R_0, Z_SUN)


@pytest.mark.parametrize("profile", ["uniform", "exponential"])
def test_finite_disk_density_is_normalised_to_gamma_at_the_sun(profile):
    """int rho(R_sun, Z) dZ = Gamma_2d for both profiles; zero outside the disk."""
    gamma = 4.0
    m = fd_model(gamma, 0.005, profile=profile)
    geo = m.finite_disk_summary()
    z_lim = geo["vertical_half_extent_kpc"]
    assert z_lim == pytest.approx(0.3 if profile == "uniform" else 1.5)
    assert geo["solar_cylindrical_radius_kpc"] == pytest.approx(8.17797, abs=1e-5)
    assert geo["radial_scale_length_kpc"] == (None if profile == "uniform" else 2.6)

    # Galactocentric (X, Y, Z) -> Earth-centred Galactic, by inverting the public transform.
    probe = np.vstack([np.zeros(3), np.eye(3)]) * u.kpc
    g = galactocentric_positions(probe, R_0, Z_SUN).to_value(u.kpc)
    rot, earth = (g[1:] - g[0]).T, g[0]

    def density(gc_xyz):
        local = (np.atleast_2d(gc_xyz) - earth) @ rot  # rot^-1 = rot^T applied to rows
        return m.origin_rate_density(local * u.kpc).to_value(1 / (u.kpc**3 * u.Gyr))

    z = np.linspace(-z_lim, z_lim, 400_001)
    column = np.column_stack([np.full_like(z, earth[0]), np.zeros_like(z), z])
    assert np.trapezoid(density(column), z) == pytest.approx(gamma, rel=2e-5)
    assert m.rate_density == pytest.approx(density([earth[0], 0.0, 0.0])[0], rel=1e-12)
    # strictly inside R <= R_gal and the vertical domain
    assert density([14.999, 0.0, 0.0])[0] > 0 and density([0.0, 14.999, 0.0])[0] > 0
    assert density([15.001, 0.0, 0.0])[0] == 0.0 and density([-10.7, 10.7, 0.0])[0] == 0.0
    assert density([earth[0], 0.0, 1.001 * z_lim])[0] == 0.0
    if profile == "exponential":
        inner, outer = (
            density([-(8.17797 - 2.6), 0.0, 0.0])[0],
            density([-(8.17797 + 2.6), 0.0, 0.0])[0],
        )
        assert inner / m.rate_density == pytest.approx(np.e, rel=1e-4)  # one R_d inward
        assert outer / m.rate_density == pytest.approx(1 / np.e, rel=1e-4)
        assert density([earth[0], 0.0, 0.3])[0] / m.rate_density == pytest.approx(
            1 / np.e, rel=1e-9
        )
        # a bubble far smaller than h_z and R_d sees the local density only
        tiny = fd_model(gamma, 1e-4, 1e-3, profile="exponential")  # v T = 31 pc
        local = density([earth[0], 0.0, earth[2]])[0]
        v = 1e-4 * C_KPC_PER_GYR
        assert tiny.earth_lambda() == pytest.approx(local * np.pi * v**3 * 1e-3**4 / 3, rel=5e-3)
    with pytest.raises(GeometryError):
        model("thick_disk_3d", gamma, 0.005).origin_rate_density(np.zeros((1, 3)) * u.pc)


@pytest.mark.parametrize(
    ("gamma", "v", "t", "profile"),
    [
        (4.0, 0.005, 1e-3, "uniform"),  # pancake: bubble >> h_z, edge out of reach
        (4e-4, 0.5, 1e-3, "uniform"),  # bubble swallows the Galaxy
        (4.0, 0.005, 1e-2, "uniform"),  # edge partly in reach
        (4e-4, 0.5, 5e-5, "exponential"),
        (4.0, 0.005, 1e-2, "exponential"),
    ],
)
def test_finite_disk_earth_cone_quadrature_matches_importance_sampler(
    gamma, v, t, profile, small_targets
):
    """Two independent methods for the Earth cone measure must agree."""
    m = fd_model(gamma, v, t, profile)
    r = m.finite_disk_reach(small_targets.xyz)
    lam = m.earth_lambda()
    assert r.n_samples == 8192 and r.seed == 42
    assert 0 < r.earth_lambda_mc_se < 0.05 * lam
    assert abs(r.earth_lambda_mc - lam) <= 4 * r.earth_lambda_mc_se
    # same seed, same targets: the sampler is deterministic (and cached)
    again = fd_model(gamma, v, t, profile).finite_disk_reach(small_targets.xyz)
    assert again.earth_lambda_mc == r.earth_lambda_mc
    assert np.array_equal(again.lam_conditional, r.lam_conditional)
    other = fd_model(gamma, v, t, profile, seed=43).finite_disk_reach(small_targets.xyz)
    assert other.earth_lambda_mc != r.earth_lambda_mc


@pytest.mark.slow
@pytest.mark.parametrize(
    ("gamma", "v", "t", "profile"),
    [
        (4.0, 0.005, 1e-3, "uniform"),
        (4.0, 0.005, 1e-3, "exponential"),
        (4.0, 0.005, 1e-2, "uniform"),  # Galactic edge in reach: thinning is exercised
        (4.0, 0.005, 1e-2, "exponential"),
    ],
)
def test_finite_disk_importance_sampler_matches_correlated_monte_carlo(
    gamma, v, t, profile, small_targets
):
    """E[N_reached | Earth unreached]: Rao-Blackwellised importance sampling vs. direct
    simulation of the thinned origin process (independent estimators, 4-sigma)."""
    m = fd_model(gamma, v, t, profile, n_importance=16384)
    imp = m.expected_reached(small_targets.xyz, condition_on_earth_unreached=True)
    imp_se = m.expected_reached_se(small_targets.xyz, True)
    mc = m.zero_detection_probability(
        small_targets.xyz,
        np.zeros(3),
        rng=np.random.default_rng(2024),
        condition_on_earth_unreached=True,
        n_mc=60000,
    )
    assert mc.n_points_accepted <= mc.n_points and mc.proposal_density_bound >= 1.0
    assert mc.mc_expected_reached_se > 0
    tol = 4 * np.hypot(mc.mc_expected_reached_se, imp_se)
    assert abs(mc.mc_expected_reached - imp) < tol, (mc.mc_expected_reached, imp, tol)


@pytest.mark.slow
@pytest.mark.parametrize("profile", ["uniform", "exponential"])
@pytest.mark.parametrize("conditioned", [True, False])
def test_finite_disk_exact_p0_respects_fkg_and_markov_bounds(profile, conditioned, bl_targets):
    m = fd_model(4.0, 0.005, profile=profile, n_importance=16384)
    p = np.full(len(bl_targets), 0.05)
    q = m.reach_probabilities(bl_targets.xyz, conditioned)
    b = radio.zero_detection_bounds(p, q)
    mc = m.zero_detection_probability(
        bl_targets.xyz,
        p,
        rng=np.random.default_rng(11),
        condition_on_earth_unreached=conditioned,
        n_mc=4000,
    )
    # the bounds use importance-sampled reach probabilities: allow for their error too
    tol = 4 * mc.p_zero_se + 4 * 0.05 * m.expected_reached_se(bl_targets.xyz, conditioned)
    assert b.markov <= b.independent + 1e-15
    assert b.independent <= mc.p_zero_targets_only + tol
    assert mc.p_zero_targets_only <= b.frechet_upper + tol


def test_lookback_sweep_job_reports_finite_disk_validity(graph):
    """The registered T-sweep runner accepts the new geometry and reports its status."""
    res = run_lookback_window_t_sweep(
        graph,
        {
            "pair": ["frequent_slow_expanders", "rare_fast_expanders"],
            "t_min_gyr": 5.0e-5,
            "t_max_gyr": 1.0e-2,
            "n_points": 2,
            "geometries": ["thin_disk_2d", "finite_disk_3d"],
            "radio_mode": "flat_probability",
            "channels": ["radio"],
            "n_mc": 100,
        },
        seed=42,
    )
    status = {
        (r["geometry"], r["lookback_gyr"]): r["hypotheses"]["rare_fast_expanders"]["local_approx"]
        for r in res["rows"]
    }
    assert status[("thin_disk_2d", 5.0e-5)] == "ok"  # 50 kyr < T_valid = 52.185 kyr
    assert status[("thin_disk_2d", 1.0e-2)] == "VIOLATED"
    assert status[("finite_disk_3d", 5.0e-5)] == status[("finite_disk_3d", 1.0e-2)]
    assert status[("finite_disk_3d", 1.0e-2)] == "valid_finite_galaxy"
    fast = {r["lookback_gyr"]: r for r in res["rows"] if r["geometry"] == "finite_disk_3d"}
    p_fast = fast[1.0e-2]["hypotheses"]["rare_fast_expanders"]["p_earth_reached"]
    assert 0 < p_fast < 0.01  # the infinite disk would give ~1 at 10 Myr
