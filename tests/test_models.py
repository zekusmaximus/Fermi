"""Model regression, deterministic replay, and IR calibration reversal (CI items 6-7)."""

from __future__ import annotations

import json

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from conftest import ROOT, frac, rate
from fermi_desk.cli import app
from fermi_desk.inference.engine import RunConfig, build_context, ir_reversal_sweep, run_comparison
from fermi_desk.models import infrared, radio, solar_system
from fermi_desk.models.expansion import (
    C_KPC_PER_GYR,
    ExpansionModel,
    ball_slab_volume,
    lens_area,
)

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
