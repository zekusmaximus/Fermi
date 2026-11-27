"""Scientific CI invariants (MILESTONE_1 CI items 2-5; SPEC.md Section 2)."""

from __future__ import annotations

import itertools
import json

import astropy.units as u
import numpy as np
import pytest
from scipy.integrate import quad
from typer.testing import CliRunner

from conftest import ROOT, frac, rate
from fermi_desk.cli import app
from fermi_desk.errors import (
    DoubleCountingError,
    GeometryError,
    MergePolicyError,
    SchemaValidationError,
    Tier3ScoringError,
    UnitValidationError,
)
from fermi_desk.inference.engine import (
    HEADLINE_VERIFIED,
    HEADLINE_WITH_ILLUSTRATIVE,
    ILLUSTRATIVE_TIERS,
    RunConfig,
    build_context,
    chain_marginals,
    compare_pair,
    evaluate_hypothesis,
    evaluate_hypothesis_marginalized,
    lookback_grid,
    run_comparison,
)
from fermi_desk.models import infrared, radio
from fermi_desk.models.expansion import C_KPC_PER_GYR, ExpansionModel
from fermi_desk.registry import FactorGraph, TargetTable, check_pointing_groups
from fermi_desk.schemas import Parameter

H_Z, Z_SUN = 300 * u.pc, 20.8 * u.pc


def model(geometry, gamma, v, t_gyr=1e-3, **kw):
    return ExpansionModel(geometry, rate(gamma), frac(v), t_gyr * u.Gyr, H_Z, Z_SUN, **kw)


# ------------------------------------------------------- zero-emission invariance
SPEEDS = (1e-5, 1e-3, 0.05, 0.5)
RATES = (1e-4, 1.0, 1e3)


@pytest.mark.parametrize("geometry", ["thin_disk_2d", "thick_disk_3d"])
@pytest.mark.parametrize("mode", ["flat_probability", "distance_scaled_eirp"])
@pytest.mark.parametrize("conditioned", [True, False])
def test_zero_radio_emission_gives_unit_likelihood_ratio(
    geometry, mode, conditioned, bl_targets, graph
):
    sv = graph.surveys["bl_enriquez_2017_692"]
    rs = sv.radio_sensitivity
    rad = radio.Radiometer(
        rs.sefd, rs.snr_threshold, rs.channel_bandwidth, rs.integration_time, rs.n_pol
    )
    lf = radio.LogNormalEIRP(1e13 * u.W, 1.0)
    p = radio.detection_probabilities(mode, 0.0, bl_targets.distance, rad, lf)
    for v, g in itertools.product(SPEEDS, RATES):
        m = model(geometry, g, v)
        res = m.zero_detection_probability(
            bl_targets.xyz,
            p,
            rng=np.random.default_rng(7),
            condition_on_earth_unreached=conditioned,
            n_mc=50,
            max_points=1e9,
        )
        assert abs(res.p_zero - 1.0) <= 1e-12, (v, g)
        assert abs(res.p_zero_targets_only - 1.0) <= 1e-12
    # Analytic bounds at the most-reaching corner of the grid (they depend on p*q only).
    q = model(geometry, max(RATES), max(SPEEDS)).reach_probabilities(bl_targets.xyz, conditioned)
    b = radio.zero_detection_bounds(p, q)
    assert b.independent == 1.0 and b.markov == 1.0


def test_zero_emission_hypothesis_scores_exactly_one_in_engine(graph):
    for geometry, mode in [
        ("thin_disk_2d", "flat_probability"),
        ("thick_disk_3d", "distance_scaled_eirp"),
    ]:
        ctx = build_context(graph, RunConfig(geometry=geometry, radio_mode=mode, seed=3, n_mc=50))
        out = evaluate_hypothesis(graph.hypotheses["quiet_expanders"], ctx)
        comps = out["components"]
        assert abs(comps["radio"]["log10_lr_vs_null"]) <= 1e-12
        assert abs(comps["infrared"]["log10_lr_vs_null"]) <= 1e-12
        assert comps["solar_system"]["log10_lr_vs_null"] == 0.0
        # Normalisation: the IR raw likelihood is NOT 1 (k=7 flags have nonzero probability
        # under the silent null), and LR = raw - null exactly.
        ir = comps["infrared"]
        assert ir["log10_p_data_given_h"] == ir["log10_null"] < 0
        # Derived IR host fraction is P(star reached) * f_ir_loud = 0 when f_ir_loud = 0.
        assert out["infrared"]["signature_fraction_max"] == 0.0
        assert out["infrared"]["p_star_reached"] > 0


@pytest.mark.parametrize("host", [1e-8, 1e-5, 1e-2, 1.0])
def test_zero_ir_conspicuousness_gives_unit_likelihood_ratio(graph, host):
    sv = graph.surveys["hephaistos_wise_mid_ir"]
    cals = [
        infrared.IRCalibration(c.id, c.completeness, c.false_positive_rate)
        for c in sv.infrared.calibrations
    ]
    q = np.array([host * 0.0])
    for c in cals:
        lr = np.exp(infrared.log_lr_vs_silent(sv.observed_detections, sv.n_targets, q, c))
        assert abs(lr - 1.0) <= 1e-12
    mix = infrared.log_lr_vs_silent_calibration_mixture(
        sv.observed_detections, sv.n_targets, q, cals
    )
    assert abs(np.exp(mix) - 1.0) <= 1e-12


# ------------------------------------------------------- analytical boundary limits
@pytest.mark.parametrize(
    "geometry", ["thin_disk_2d", pytest.param("thick_disk_3d", marks=pytest.mark.slow)]
)
def test_expected_reached_vanishes_as_speed_goes_to_zero(geometry, bl_targets):
    vals = {c: [] for c in (True, False)}
    for v in (1e-2, 1e-4, 1e-6, 1e-8, 0.0):
        m = model(geometry, 4.0, v)
        for c in vals:
            vals[c].append(m.expected_reached(bl_targets.xyz, condition_on_earth_unreached=c))
    for seq in vals.values():
        assert all(a >= b for a, b in itertools.pairwise(seq))
        # Each two-decade drop in v shrinks E[N] by >= 50x (crescent ~ v, cone ~ v^2 or v^3).
        assert all(b <= a / 50 for a, b in itertools.pairwise(seq[:-1]))
        assert seq[-1] == 0.0


@pytest.mark.parametrize("geometry", ["thin_disk_2d", "thick_disk_3d"])
@pytest.mark.parametrize("conditioned", [True, False])
def test_zero_targets_gives_unit_survey_likelihood(geometry, conditioned):
    empty = TargetTable.empty()
    m = model(geometry, 4.0, 0.005)
    res = m.zero_detection_probability(
        empty.xyz,
        np.zeros(0),
        rng=np.random.default_rng(1),
        condition_on_earth_unreached=conditioned,
        n_mc=100,
    )
    assert res.p_zero == 1.0
    assert radio.zero_detection_bounds(np.zeros(0), np.zeros(0)).independent == 1.0
    assert m.expected_reached(empty.xyz, conditioned) == 0.0


@pytest.mark.parametrize("geometry", ["thin_disk_2d", "thick_disk_3d"])
@pytest.mark.parametrize(
    ("gamma", "v", "t"), [(4.0, 0.005, 1e-3), (4e-4, 0.5, 1e-3), (1.0, 1e-3, 1e-2)]
)
def test_earth_unreached_conditioning_never_increases_expected_reached(
    geometry, gamma, v, t, bl_targets
):
    m = model(geometry, gamma, v, t)
    q_c = m.reach_probabilities(bl_targets.xyz, True)
    q_u = m.reach_probabilities(bl_targets.xyz, False)
    assert np.all(q_c <= q_u + 1e-15)
    assert q_c.sum() < q_u.sum()  # strict: finite correlation radius


# --------------------------------------------------- 2D vs. 3D scaling symmetry audit
@pytest.mark.parametrize(
    ("gamma", "v", "t"), [(4.0, 0.005, 1e-3), (1e-2, 0.1, 1e-2), (10.0, 1e-4, 0.1)]
)
def test_thin_disk_symmetry_double_speed_quarter_rate(gamma, v, t):
    a = model("thin_disk_2d", gamma, v, t).earth_reach_probability()
    b = model("thin_disk_2d", 0.25 * gamma, 2.0 * v, t).earth_reach_probability()
    assert a > 0
    assert abs(a - b) <= 1e-12 * max(a, 1e-300) + 1e-15


def test_thick_disk_breaks_symmetry_when_bubbles_smaller_than_scale_height():
    v, t = 1e-4, 1e-3  # R = v T ~ 30.7 pc; 2R ~ 61 pc < h_z - z_sun = 279 pc
    assert 2 * v * C_KPC_PER_GYR * t < (H_Z - Z_SUN).to_value(u.kpc)
    m1 = model("thick_disk_3d", 4.0, v, t)
    m2 = model("thick_disk_3d", 1.0, 2 * v, t)
    # Volume-limited regime: Lambda proportional to Gamma v^3, so the transform doubles it.
    assert m2.earth_lambda() / m1.earth_lambda() == pytest.approx(2.0, rel=1e-9)
    p1, p2 = m1.earth_reach_probability(), m2.earth_reach_probability()
    assert abs(p2 - p1) / p1 > 0.5


def test_thick_disk_recovers_thin_disk_when_bubbles_exceed_scale_height():
    m3 = model("thick_disk_3d", 4.0, 0.05)  # R ~ 15 kpc >> h_z
    m2 = model("thin_disk_2d", 4.0, 0.05)
    assert m3.earth_lambda() / m2.earth_lambda() == pytest.approx(1.0, abs=2e-3)


# ---------------------------------------------------------------- geometry explicit
def test_geometry_must_be_explicit():
    with pytest.raises(GeometryError):
        ExpansionModel("disk", rate(1.0), frac(0.01), 1e-3 * u.Gyr)
    with pytest.raises(GeometryError):
        ExpansionModel("thick_disk_3d", rate(1.0), frac(0.01), 1e-3 * u.Gyr)  # no h_z / z_sun
    with pytest.raises(GeometryError):
        RunConfig(geometry="3d", radio_mode="flat_probability", seed=1)


def test_unitless_inputs_rejected_at_module_boundary():
    with pytest.raises(UnitValidationError):
        ExpansionModel("thin_disk_2d", 1.0, frac(0.01), 1e-3 * u.Gyr)
    with pytest.raises(UnitValidationError):
        ExpansionModel("thin_disk_2d", rate(1.0), frac(0.01), 1e-3 * u.pc)
    with pytest.raises(UnitValidationError):
        radio.min_detectable_eirp(np.array([10.0]), 1e-25 * u.W / u.m**2)


# --------------------------------------------------------------- double counting
def test_shared_pointing_group_raises_double_counting(graph):
    bl = graph.surveys["bl_enriquez_2017_692"]
    dup = bl.model_copy(update={"id": "bl_enriquez_2017_692_reanalysis"})
    with pytest.raises(DoubleCountingError):
        check_pointing_groups([bl, dup])
    # v0.2.1: keep_first (load-order-dependent, drops a catalog wholesale) is forbidden. The
    # pre-v0.2.1 call form must raise the explicit domain error, never silently keep one.
    with pytest.raises(MergePolicyError, match="load order"):
        check_pointing_groups([bl, dup], {bl.pointing_group_id: "keep_first"})
    # union_targets_by_pointing needs a declared parent pointing: an undeclared duplicate
    # is still a double-counting error, not something to merge.
    with pytest.raises(DoubleCountingError):
        check_pointing_groups([bl, dup], "union_targets_by_pointing")
    assert [s.id for s in check_pointing_groups([bl])] == [bl.id]


def test_engine_refuses_duplicate_pointing_groups(graph):
    bl = graph.surveys["bl_enriquez_2017_692"]
    dup = bl.model_copy(update={"id": "bl_dup"})
    g2 = FactorGraph(graph.root, graph.registry, graph.hypotheses, {**graph.surveys, "bl_dup": dup})
    with pytest.raises(DoubleCountingError):
        build_context(g2, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=1))
    with pytest.raises(DoubleCountingError):  # no policy turns an undeclared duplicate into a merge
        build_context(
            g2,
            RunConfig(
                geometry="thin_disk_2d",
                radio_mode="flat_probability",
                seed=1,
                pointing_merge_policy="union_targets_by_pointing",
            ),
        )
    with pytest.raises(MergePolicyError):
        RunConfig(
            geometry="thin_disk_2d",
            radio_mode="flat_probability",
            seed=1,
            pointing_merge_policy="keep_first",
        )


def test_conditioned_run_never_reapplies_earth_factor_via_solar_system(graph):
    ctx = build_context(
        graph, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=1, n_mc=50)
    )
    out = evaluate_hypothesis(graph.hypotheses["frequent_slow_expanders"], ctx)
    assert out["solar_system"]["subsumed_by_earth_unreached_conditioning"] is True
    assert out["components"]["solar_system"]["log10_p_data_given_h"] == 0.0
    assert "earth_unreached" in out["components"]
    assert "radio+solar_system" not in out["components"]


# ------------------------------------- illustrative channels vs. the headline BF
def _compare(tmp_path, sub, *extra):
    out = tmp_path / sub
    args = ["compare", "--geometry", "thin_disk_2d", "--radio-mode", "flat_probability"]
    args += ["--seed", "42", "--n-mc", "200", "--out-dir", str(out), "--root", str(ROOT), *extra]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    (path,) = list(out.glob("*.json"))
    return json.loads(path.read_text()), result.output


def test_default_headline_bf_excludes_illustrative_channels(tmp_path):
    """Milestone 2.5 ruling: default headline = verified channels; opt-in folds the rest in."""
    rec, text = _compare(tmp_path, "default")
    bf = rec["results"]["headline"]
    head = bf["headline"]
    assert rec["headline_policy"] == head["policy"] == HEADLINE_VERIFIED
    assert head["terms"] == ["earth_unreached", "radio"]
    assert {"infrared", "solar_system"} <= set(head["excluded_illustrative_channels"])
    assert not any(t in ILLUSTRATIVE_TIERS for t in head["data_tiers"])
    assert rec["headline_log10_bf"] == head["log10_bf"]
    assert head["log10_bf"] == pytest.approx(
        sum(bf["log10_bf_by_term"][t] for t in ("earth_unreached", "radio")), abs=1e-15
    )
    # Infrared stays active in the engine, tagged with its tier in the per-component table.
    ir = rec["results"]["hypotheses"]["frequent_slow_expanders"]["components"]["infrared"]
    assert "illustrative_calibration" in ir["data_tiers"]
    assert "infrared" in bf["terms"] and "infrared" in bf["illustrative_terms"]
    assert "ILLUSTRATIVE CHANNELS FOLDED" not in text

    inc, text_inc = _compare(tmp_path, "included", "--include-illustrative")
    bf_i = inc["results"]["headline"]
    head_i = bf_i["headline"]
    assert inc["headline_policy"] == head_i["policy"] == HEADLINE_WITH_ILLUSTRATIVE
    assert "infrared" in head_i["terms"] and "infrared" in head_i["includes_illustrative_channels"]
    assert head_i["log10_bf"] == bf_i["log10_bf"] and head_i["warning"]
    assert "ILLUSTRATIVE CHANNELS FOLDED" in text_inc
    # Same seed, same components: the flag changes only which terms the headline sums.
    assert inc["results"]["hypotheses"] == rec["results"]["hypotheses"]
    extra = sum(bf_i["log10_bf_by_term"][t] for t in bf_i["illustrative_terms"])
    assert head_i["log10_bf"] - head["log10_bf"] == pytest.approx(extra, abs=1e-12)


def test_verified_headline_uses_radio_marginal_without_earth_conditioning(graph):
    """Unconditioned runs evaluate radio+solar_system jointly; the verified headline must
    fall back to the radio marginal rather than silently dropping radio."""
    cfg = RunConfig(
        geometry="thin_disk_2d",
        radio_mode="flat_probability",
        seed=5,
        n_mc=200,
        condition_on_earth_unreached=False,
        headline_pair=("frequent_slow_expanders", "rare_fast_expanders"),
    )
    bf = run_comparison(graph, cfg, list(cfg.headline_pair))["results"]["headline"]
    assert "radio+solar_system" in bf["terms"]
    assert bf["headline"]["terms"] == ["radio"]
    assert "radio+solar_system" in bf["illustrative_terms"]


def test_distance_scaled_radio_is_illustrative_and_leaves_headline(graph):
    """The transmitter luminosity function is illustrative, so distance_scaled_eirp radio
    leaves the verified headline (only the Earth-unreached term remains)."""
    cfg = RunConfig(
        geometry="thin_disk_2d",
        radio_mode="distance_scaled_eirp",
        seed=5,
        n_mc=200,
        headline_pair=("frequent_slow_expanders", "rare_fast_expanders"),
    )
    head = run_comparison(graph, cfg, list(cfg.headline_pair))["results"]["headline"]["headline"]
    assert "radio" in head["excluded_illustrative_channels"]
    assert head["terms"] == ["earth_unreached"]


# =========================================================================== v0.2.0
# Finite Galactic disk (finite_disk_3d) and lookback-window marginalisation: every core
# invariant above must also hold for the new geometry and with marginalize_lookback=True.
R_GAL, R_0, R_D = 15.0 * u.kpc, 8.178 * u.kpc, 2.6 * u.kpc
PROFILES = ("uniform", "exponential")
PAIR = ("frequent_slow_expanders", "rare_fast_expanders")


def fd_model(gamma, v, t_gyr=1e-3, profile="uniform", n_importance=4096, seed=42, r_gal=R_GAL):
    return ExpansionModel(
        "finite_disk_3d",
        rate(gamma),
        frac(v),
        t_gyr * u.Gyr,
        H_Z,
        Z_SUN,
        disk_profile=profile,
        disk_radius=r_gal,
        solar_galactocentric_radius=R_0,
        radial_scale_length=R_D,
        importance_seed=seed,
        n_importance=n_importance,
    )


# ------------------------------------------------------------- physical limits
@pytest.mark.parametrize(("gamma", "v"), [(4.0, 0.005), (4e-4, 0.5)])
def test_finite_disk_small_bubble_converges_to_thick_disk(gamma, v, bl_targets):
    """v T << R_gal - R_sun (T = 10 kyr): the Galactic edge is out of causal reach, so the
    uniform finite disk must reproduce thick_disk_3d.

    Tolerances: P(Earth reached) is deterministic quadrature in both geometries (1e-9
    relative). E[N | Earth unreached] is importance-sampled in the finite disk and
    quadrature in the thick disk: 4 standard errors plus 0.2% (the Galactocentric frame
    tilts the mid-plane by asin(z_sun / R_0) = 0.15 deg, a < 0.1% effect here).
    """
    t = 1e-5
    assert v * C_KPC_PER_GYR * t < 0.25 * (R_GAL - R_0).to_value(u.kpc)
    thick = model("thick_disk_3d", gamma, v, t)
    fin = fd_model(gamma, v, t, n_importance=16384)
    assert fin.earth_reach_probability() == pytest.approx(thick.earth_reach_probability(), rel=1e-9)
    for cond in (True, False):
        want = thick.expected_reached(bl_targets.xyz, cond)
        got = fin.expected_reached(bl_targets.xyz, cond)
        se = fin.expected_reached_se(bl_targets.xyz, cond)
        assert 0 < se < 0.02 * want  # the Monte Carlo error itself is small
        assert abs(got - want) <= 4 * se + 2e-3 * want, (cond, got, want, se)
    # the importance sampler independently reproduces the deterministic Earth cone
    r = fin.finite_disk_reach(bl_targets.xyz)
    assert abs(r.earth_lambda_mc - fin.earth_lambda()) <= 4 * r.earth_lambda_mc_se


def test_finite_disk_edge_truncation_reduces_fast_expander_earth_reach():
    """rare/fast fronts at T = 1 Myr span 153 kpc: in the infinite disk most of the origins
    that reach Earth lie outside R_gal = 15 kpc, where no Galaxy exists."""
    gamma, v, t = 4e-4, 0.5, 1e-3
    thick = model("thick_disk_3d", gamma, v, t)
    fin = fd_model(gamma, v, t)
    assert fin.earth_reach_probability() < thick.earth_reach_probability()
    assert fin.earth_lambda() < 0.05 * thick.earth_lambda()
    # no more origins can reach Earth than the whole disk produced in the window
    assert fin.earth_lambda() < gamma * np.pi * R_GAL.to_value(u.kpc) ** 2 * t
    # slow fronts never see the edge at 1 Myr (v T = 1.5 kpc < R_gal - R_sun = 6.8 kpc)
    slow = fd_model(4.0, 0.005, t)
    assert slow.earth_lambda() == pytest.approx(
        model("thick_disk_3d", 4.0, 0.005, t).earth_lambda(), rel=1e-9
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_finite_disk_zero_speed_limit(profile, bl_targets):
    vals = {c: [] for c in (True, False)}
    lam_e = []
    for v in (1e-2, 1e-4, 1e-6, 1e-8, 0.0):
        m = fd_model(4.0, v, profile=profile, n_importance=2048)
        lam_e.append(m.earth_lambda())
        for c in vals:
            vals[c].append(m.expected_reached(bl_targets.xyz, condition_on_earth_unreached=c))
    for seq in (*vals.values(), lam_e):
        assert all(a >= b for a, b in itertools.pairwise(seq))
        assert all(b <= a / 50 for a, b in itertools.pairwise(seq[:-1]))
        assert seq[-1] == 0.0


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("conditioned", [True, False])
def test_finite_disk_zero_targets_gives_unit_survey_likelihood(profile, conditioned):
    empty = TargetTable.empty()
    m = fd_model(4.0, 0.005, profile=profile)
    res = m.zero_detection_probability(
        empty.xyz,
        np.zeros(0),
        rng=np.random.default_rng(1),
        condition_on_earth_unreached=conditioned,
        n_mc=100,
    )
    assert res.p_zero == 1.0 and res.p_zero_targets_only == 1.0
    assert m.expected_reached(empty.xyz, conditioned) == 0.0


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    ("gamma", "v", "t"), [(4.0, 0.005, 1e-3), (4e-4, 0.5, 1e-3), (1.0, 1e-3, 1e-2)]
)
def test_finite_disk_conditioning_never_increases_expected_reached(
    profile, gamma, v, t, bl_targets
):
    """Exact, not statistical: K_c <= K_u pointwise on the shared importance samples."""
    m = fd_model(gamma, v, t, profile=profile, n_importance=2048)
    q_c = m.reach_probabilities(bl_targets.xyz, True)
    q_u = m.reach_probabilities(bl_targets.xyz, False)
    assert np.all(q_c <= q_u)
    assert q_c.sum() < q_u.sum()


# ------------------------------------------------------- zero-emission invariance
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("mode", ["flat_probability", "distance_scaled_eirp"])
@pytest.mark.parametrize("conditioned", [True, False])
def test_finite_disk_zero_radio_emission_gives_unit_likelihood_ratio(
    profile, mode, conditioned, bl_targets, graph
):
    sv = graph.surveys["bl_enriquez_2017_692"]
    rs = sv.radio_sensitivity
    rad = radio.Radiometer(
        rs.sefd, rs.snr_threshold, rs.channel_bandwidth, rs.integration_time, rs.n_pol
    )
    lf = radio.LogNormalEIRP(1e13 * u.W, 1.0)
    p = radio.detection_probabilities(mode, 0.0, bl_targets.distance, rad, lf)
    for v, g in itertools.product((1e-5, 0.005, 0.5), (1e-4, 4.0)):
        m = fd_model(g, v, profile=profile, n_importance=512)
        res = m.zero_detection_probability(
            bl_targets.xyz,
            p,
            rng=np.random.default_rng(7),
            condition_on_earth_unreached=conditioned,
            n_mc=50,
            max_points=1e9,
        )
        assert abs(res.p_zero - 1.0) <= 1e-12, (v, g)
        assert abs(res.p_zero_targets_only - 1.0) <= 1e-12
    q = fd_model(4.0, 0.5, profile=profile, n_importance=512).reach_probabilities(
        bl_targets.xyz, conditioned
    )
    b = radio.zero_detection_bounds(p, q)
    assert b.independent == 1.0 and b.markov == 1.0


def _fast_cfg(geometry, **kw):
    base = dict(radio_mode="flat_probability", seed=3, n_mc=50, n_importance=512)
    return RunConfig(geometry=geometry, **{**base, **kw})


@pytest.mark.parametrize(
    "cfg",
    [
        _fast_cfg("finite_disk_3d"),
        _fast_cfg("finite_disk_3d", disk_profile="exponential", radio_mode="distance_scaled_eirp"),
        _fast_cfg("thin_disk_2d", marginalize_lookback=True, lookback_grid_points=3),
        _fast_cfg("finite_disk_3d", marginalize_lookback=True, lookback_grid_points=3),
    ],
    ids=["finite", "finite-exponential-eirp", "thin-marginalized", "finite-marginalized"],
)
def test_zero_emission_hypothesis_scores_exactly_one_in_v020_modes(graph, cfg):
    """p_radio = 0 and f_ir_loud = 0 give LR = 1 in the finite disk and with T marginalised."""
    res = run_comparison(graph, cfg, ["quiet_expanders"])["results"]
    out = res["hypotheses"]["quiet_expanders"]
    comps = out["components"]
    assert abs(comps["radio"]["log10_lr_vs_null"]) <= 1e-12
    assert abs(comps["infrared"]["log10_lr_vs_null"]) <= 1e-12
    assert abs(comps["solar_system"]["log10_lr_vs_null"]) <= 1e-12
    assert out["infrared"]["signature_fraction_max"] == 0.0
    assert out["infrared"]["p_star_reached"] > 0  # reached, but silent


# ------------------------------------------------------------ validity status
def test_local_approx_status_clears_only_under_finite_disk(graph):
    """rare_fast_expanders at T = 1 Myr: VIOLATED (T_valid = 52.185 kyr) in the local
    infinite disks, valid_finite_galaxy in the finite disk."""
    t_valid_kyr = 8.0 / (0.5 * C_KPC_PER_GYR) * 1e6
    assert t_valid_kyr == pytest.approx(52.185, abs=5e-4)
    status = {}
    for geometry in ("thin_disk_2d", "thick_disk_3d", "finite_disk_3d"):
        cfg = _fast_cfg(geometry, channels=("radio",))
        ex = run_comparison(graph, cfg, ["rare_fast_expanders"])["results"]["hypotheses"][
            "rare_fast_expanders"
        ]["expansion"]
        assert ex["max_bubble_radius_kpc"] == pytest.approx(153.3, abs=0.1)
        status[geometry] = (ex["local_approximation_valid"], ex.get("local_approx"))
    # the infinite-disk receipts are unchanged: boolean flag only, no new key
    assert status["thin_disk_2d"] == status["thick_disk_3d"] == (False, None)
    assert status["finite_disk_3d"] == (True, "valid_finite_galaxy")
    for geometry in ("thin_disk_2d", "thick_disk_3d"):
        m = model(geometry, 4e-4, 0.5)
        assert m.local_approx_status == "VIOLATED" and not m.local_approximation_valid
        assert model(geometry, 4e-4, 0.5, t_valid_kyr * 0.999e-6).local_approx_status == "ok"
    assert fd_model(4e-4, 0.5).local_approx_status == "valid_finite_galaxy"


def test_finite_disk_geometry_must_be_explicit():
    common = ("finite_disk_3d", rate(1.0), frac(0.01), 1e-3 * u.Gyr, H_Z, Z_SUN)
    radii = dict(disk_radius=R_GAL, solar_galactocentric_radius=R_0)
    with pytest.raises(GeometryError):  # no disk_profile
        ExpansionModel(*common, **radii)
    with pytest.raises(GeometryError):  # unknown profile
        ExpansionModel(*common, disk_profile="sech2", **radii)
    with pytest.raises(GeometryError):  # no disk radius / solar radius
        ExpansionModel(*common, disk_profile="uniform")
    with pytest.raises(GeometryError):  # exponential needs R_d
        ExpansionModel(*common, disk_profile="exponential", **radii)
    with pytest.raises(GeometryError):  # Sun outside the disk
        fd_model(1.0, 0.01, r_gal=5.0 * u.kpc)
    with pytest.raises(GeometryError):  # a profile on a local infinite disk is a mistake
        ExpansionModel(
            "thick_disk_3d", rate(1.0), frac(0.01), 1e-3 * u.Gyr, H_Z, Z_SUN, disk_profile="uniform"
        )
    with pytest.raises(UnitValidationError):  # bare float where a length is required
        ExpansionModel(
            *common, disk_profile="uniform", disk_radius=15.0, solar_galactocentric_radius=R_0
        )
    with pytest.raises(GeometryError):
        RunConfig(
            geometry="finite_disk_3d", radio_mode="flat_probability", seed=1, disk_profile="x"
        )


# ------------------------------------------------------- deterministic replay
def _compare_v020(tmp_path, sub, *extra, seed=42):
    out = tmp_path / sub
    args = ["compare", "--radio-mode", "flat_probability", "--seed", str(seed), "--n-mc", "200"]
    args += ["--n-importance", "512", "--channels", "radio", "--out-dir", str(out)]
    args += ["--root", str(ROOT), *extra]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    (path,) = list(out.glob("*.json"))
    return json.loads(path.read_text()), result.output


@pytest.mark.parametrize(
    "extra",
    [
        ("--geometry", "finite_disk_3d"),
        ("--geometry", "finite_disk_3d", "--disk-profile", "exponential"),
        ("--geometry", "finite_disk_3d", "--marginalize-lookback", "--lookback-grid-points", "3"),
        ("--geometry", "thin_disk_2d", "--marginalize-lookback", "--lookback-grid-points", "3"),
    ],
    ids=["finite", "finite-exponential", "finite-marginalized", "thin-marginalized"],
)
def test_v020_modes_replay_byte_identical(tmp_path, extra):
    a, text = _compare_v020(tmp_path, "a", *extra)
    b, _ = _compare_v020(tmp_path, "b", *extra)
    assert json.dumps(a["results"], sort_keys=True) == json.dumps(b["results"], sort_keys=True)
    assert a["results_sha256"] == b["results_sha256"]
    assert a["rng_seed"] == b["rng_seed"] == 42
    assert f"results_sha256={a['results_sha256']}" in text.replace("\n", "")
    if "finite_disk_3d" in extra:
        assert "valid_finite_galaxy" in text and a["geometry_mode"] == "finite_disk_3d"
        # the seed drives the importance sampler as well as the correlated Monte Carlo
        c, _ = _compare_v020(tmp_path, "c", *extra, seed=7)
        ha = a["results"]["hypotheses"]["frequent_slow_expanders"]["radio"]
        hc = c["results"]["hypotheses"]["frequent_slow_expanders"]["radio"]
        assert ha["expected_reached"] != hc["expected_reached"]
    if "--marginalize-lookback" in extra:
        assert a["config"]["marginalize_lookback"] is True
        assert a["lookback_marginalization"]["prior"]["effective_bounds_gyr"] == [5e-5, 1e-2]
        assert "Lookback T marginalised: log-uniform prior [50 kyr, 10 Myr]" in text


def test_fixed_lookback_receipts_carry_no_marginalization_keys(tmp_path):
    """marginalize_lookback=False (the default) leaves single-T receipts as they were."""
    rec, _ = _compare(tmp_path, "single")
    assert "lookback_marginalization" not in rec
    assert "lookback_marginalization" not in rec["results"]
    assert "disk_profile" not in rec  # top-level key exists for finite_disk_3d only
    assert rec["config"]["marginalize_lookback"] is False
    for h in rec["results"]["hypotheses"].values():
        assert "lookback_marginalization" not in h
        if h["expansion"]:
            assert "local_approx" not in h["expansion"] and "finite_disk" not in h["expansion"]
        # finite-disk-only registry parameters never enter an infinite-disk receipt
        assert not {"galactic_disk_radius_kpc", "solar_galactocentric_radius_kpc"} & set(
            h["parameters"]
        )


# ------------------------------------------------ lookback marginalisation exactness
def test_chain_marginals_are_likelihood_space_not_log_averages():
    w = np.array([0.2, 0.5, 0.3])
    node = {"a": [-0.1, -2.0, -6.0], "b": [-1.0, -0.5, 0.0]}
    chain, joint = chain_marginals(node, w, ["a", "b"])
    p_a = float(np.sum(w * 10.0 ** np.array(node["a"])))
    p_ab = float(np.sum(w * 10.0 ** (np.array(node["a"]) + np.array(node["b"]))))
    assert chain["a"] == pytest.approx(np.log10(p_a), abs=1e-12)
    assert chain["b"] == pytest.approx(np.log10(p_ab / p_a), abs=1e-12)  # P(b | a)
    assert joint == pytest.approx(np.log10(p_ab), abs=1e-12)
    assert chain["a"] + chain["b"] == pytest.approx(joint, abs=1e-12)
    # averaging log10 P instead would be wrong (Jensen: strictly smaller here)
    assert float(np.sum(w * np.array(node["a"]))) < chain["a"] - 0.5
    # a term that does not depend on T contributes its plain value
    flat, _ = chain_marginals({"a": node["a"], "c": [-0.25] * 3}, w, ["a", "c"])
    assert flat["c"] == pytest.approx(-0.25, abs=1e-12)


def test_lookback_grid_follows_registry_prior_and_floor_rule(graph):
    grid = lookback_grid(graph, 16)
    assert grid.effective_bounds_gyr == (5e-5, 1e-2)  # 50 kyr - 10 Myr
    assert grid.weights.sum() == pytest.approx(1.0, abs=1e-14) and np.all(grid.weights > 0)
    assert np.all(np.diff(grid.nodes_gyr) > 0)
    assert grid.nodes_gyr[0] > 5e-5 and grid.nodes_gyr[-1] < 1e-2
    # log-uniform prior: nodes are symmetric in ln T about the geometric mean
    assert np.log(grid.nodes_gyr).mean() == pytest.approx(np.log(np.sqrt(5e-5 * 1e-2)), abs=1e-12)

    base = dict(id="t", symbol="T", unit="Gyr", prior_type="log_uniform", description="x")
    floored = Parameter(
        **base, bounds=(0.0, 10.0), log_floor=1e-6, marginalization_prior_bounds=(0.0, 1e-2)
    )
    assert floored.marginalization_log_bounds == (1e-6, 1e-2)  # max(lower, log_floor)
    with pytest.raises(SchemaValidationError):  # outside the registry bounds
        Parameter(
            **base, bounds=(0.0, 10.0), log_floor=1e-6, marginalization_prior_bounds=(1e-3, 20.0)
        )
    with pytest.raises(SchemaValidationError):  # empty after the floor is applied
        Parameter(
            **base, bounds=(0.0, 10.0), log_floor=1e-1, marginalization_prior_bounds=(0.0, 1e-2)
        )
    with pytest.raises(SchemaValidationError):  # the marginalisation prior is log-uniform
        Parameter(
            **{**base, "prior_type": "uniform"},
            bounds=(1e-6, 10.0),
            marginalization_prior_bounds=(1e-3, 1e-2),
        )


def test_marginalized_earth_term_matches_analytic_integral(graph):
    """thin_disk_2d: P(Earth unreached | T) = exp(-Gamma pi v^2 T^3 / 3) is analytic, so the
    lookback marginal can be checked against adaptive quadrature of the prior integral."""
    cfg = RunConfig(
        geometry="thin_disk_2d",
        radio_mode="flat_probability",
        seed=1,
        n_mc=50,
        channels=("radio",),
        marginalize_lookback=True,
    )
    h = run_comparison(graph, cfg, ["frequent_slow_expanders"])["results"]["hypotheses"][
        "frequent_slow_expanders"
    ]
    lo, hi = np.log(5e-5), np.log(1e-2)
    k = 4.0 * np.pi * (0.005 * C_KPC_PER_GYR) ** 2 / 3.0
    exact = quad(lambda x: np.exp(-k * np.exp(3 * x)), lo, hi, epsabs=1e-13, epsrel=1e-12)[0]
    want = np.log10(exact / (hi - lo))
    got = h["components"]["earth_unreached"]["log10_p_data_given_h"]
    assert want == pytest.approx(-0.0861, abs=1e-3)  # far from the fixed-T value -0.0043
    assert got == pytest.approx(want, abs=1e-6)
    assert h["expansion"]["log10_p_earth_unreached"] == pytest.approx(want, abs=1e-6)
    chk = h["lookback_marginalization"]["earth_term_quadrature_check"]
    assert chk["n_points"] == 16 and chk["n_points_fine"] == 32
    assert chk["value"] == pytest.approx(got, abs=1e-12) and chk["abs_difference_dex"] < 1e-6
    assert h["parameters"]["lookback_window_gyr"]["value"] is None


@pytest.fixture(scope="module")
def marginalized(graph):
    cfg = RunConfig(
        geometry="finite_disk_3d",
        radio_mode="flat_probability",
        seed=42,
        n_mc=200,
        n_importance=512,
        marginalize_lookback=True,
        lookback_grid_points=4,
        headline_pair=PAIR,
    )
    return run_comparison(graph, cfg, [*PAIR, "partial_dyson_conspicuous"])


def test_marginalized_default_headline_excludes_illustrative_channels(marginalized):
    """verified_channels_only stays the default, and with T marginalised the verified
    headline is the marginal Bayes factor of the verified data alone."""
    rec = marginalized
    bf = rec["results"]["headline"]
    head = bf["headline"]
    assert rec["headline_policy"] == head["policy"] == HEADLINE_VERIFIED
    assert head["terms"] == ["earth_unreached", "radio"]
    assert {"infrared", "solar_system"} <= set(head["excluded_illustrative_channels"])
    assert not any(t in ILLUSTRATIVE_TIERS for t in head["data_tiers"])
    assert rec["headline_log10_bf"] == head["log10_bf"]
    # chain-rule terms add up: headline = sum of its terms, total = sum of all terms
    assert head["log10_bf"] == pytest.approx(
        sum(bf["log10_bf_by_term"][t] for t in head["terms"]), abs=1e-13
    )
    assert bf["log10_bf"] == pytest.approx(sum(bf["log10_bf_by_term"].values()), abs=1e-13)
    assert bf["lookback_marginalization"]["chain_order"][:2] == ["earth_unreached", "radio"]

    # Independent recomputation from the per-node VERIFIED likelihoods only.
    hyps = rec["results"]["hypotheses"]

    def verified_marginal(hid):
        lm = hyps[hid]["lookback_marginalization"]
        node = np.array(lm["node_log10_p"]["earth_unreached"]) + np.array(
            lm["node_log10_p"]["radio"]
        )
        return np.log10(np.sum(np.array(lm["weights"]) * 10.0**node))

    want = verified_marginal(PAIR[0]) - verified_marginal(PAIR[1])
    assert head["log10_bf"] == pytest.approx(want, abs=1e-12)

    # Leak test: corrupt every illustrative per-node likelihood. The verified headline
    # must not move at all; the all-terms Bayes factor must.
    a, b = (json.loads(json.dumps(hyps[h])) for h in PAIR)
    for c in ("infrared", "solar_system"):
        a["lookback_marginalization"]["node_log10_p"][c] = [-3.0, -0.1, -7.0, -0.5]
    tampered = compare_pair(a, b)
    assert tampered["verified"]["log10_bf"] == bf["verified"]["log10_bf"]
    assert tampered["verified"]["log10_bf_by_term"] == bf["verified"]["log10_bf_by_term"]
    assert tampered["log10_bf"] != bf["log10_bf"]


def test_marginalized_run_reports_per_node_diagnostics_and_constant_hypotheses(marginalized, graph):
    res = marginalized["results"]
    marg = res["lookback_marginalization"]
    assert marg["enabled"] and marg["quadrature"]["n_points"] == 4
    assert marg["prior"]["type"] == "log_uniform"
    assert "likelihood space" in marg["space"]
    assert marginalized["lookback_marginalization"]["quadrature"] == marg["quadrature"]
    fs = res["hypotheses"][PAIR[0]]
    lm = fs["lookback_marginalization"]
    assert lm["marginalized"] is True and len(lm["per_node"]) == 4
    assert sum(lm["posterior_weights"]) == pytest.approx(1.0, abs=1e-12)
    assert [n["local_approx"] for n in lm["per_node"]] == ["valid_finite_galaxy"] * 4
    assert fs["expansion"]["local_approx"] == "valid_finite_galaxy"
    # P(Earth reached) grows with T, so conditioning on Earth unreached favours short T
    p = [n["p_earth_reached"] for n in lm["per_node"]]
    assert p == sorted(p)
    ratio = np.array(lm["posterior_weights"]) / np.array(lm["weights"])
    assert ratio[-1] < ratio[0]
    # the radio term is the posterior-weighted P0 (chain rule: radio | Earth unreached)
    assert 10.0 ** fs["components"]["radio"]["log10_p_data_given_h"] == pytest.approx(
        fs["radio"]["p_zero_exact_mc"], abs=1e-12
    )
    # a hypothesis without expansion parameters cannot depend on T: constant across the grid
    pd = res["hypotheses"]["partial_dyson_conspicuous"]
    assert pd["lookback_marginalization"]["marginalized"] is False
    assert marg["hypotheses"]["partial_dyson_conspicuous"]["marginalized"] is False
    fixed = run_comparison(
        graph,
        RunConfig(geometry="finite_disk_3d", radio_mode="flat_probability", seed=42, n_mc=200),
        ["partial_dyson_conspicuous"],
    )["results"]["hypotheses"]["partial_dyson_conspicuous"]
    assert pd["components"]["infrared"]["log10_p_data_given_h"] == pytest.approx(
        fixed["components"]["infrared"]["log10_p_data_given_h"], abs=1e-12
    )


def test_marginalization_guards(graph):
    cfg = _fast_cfg("thin_disk_2d", marginalize_lookback=True, lookback_grid_points=2)
    ctx = build_context(graph, cfg)
    grid = lookback_grid(graph, 2)
    with pytest.raises(Tier3ScoringError):  # Tier-3 stays quarantined when marginalising
        evaluate_hypothesis_marginalized(graph.hypotheses["zoo_hypothesis"], ctx, grid)
    rec = run_comparison(graph, cfg)
    assert {"zoo_hypothesis", "dark_forest"}.isdisjoint(rec["results"]["hypotheses"])
    assert all(q["empirical_score"] is None for q in rec["results"]["tier3_quarantine"])
    both = _fast_cfg(
        "thin_disk_2d",
        marginalize_lookback=True,
        parameter_overrides=(("lookback_window_gyr", 1e-3),),
    )
    with pytest.raises(SchemaValidationError):  # T cannot be fixed and marginalised at once
        run_comparison(graph, both, ["frequent_slow_expanders"])
