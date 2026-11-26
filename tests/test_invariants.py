"""Scientific CI invariants (MILESTONE_1 CI items 2-5; SPEC.md Section 2)."""

from __future__ import annotations

import itertools

import astropy.units as u
import numpy as np
import pytest

from conftest import frac, rate
from fermi_desk.errors import DoubleCountingError, GeometryError, UnitValidationError
from fermi_desk.inference.engine import RunConfig, build_context, evaluate_hypothesis
from fermi_desk.models import infrared, radio
from fermi_desk.models.expansion import C_KPC_PER_GYR, ExpansionModel
from fermi_desk.registry import FactorGraph, TargetTable, check_pointing_groups

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
@pytest.mark.parametrize("geometry", ["thin_disk_2d", "thick_disk_3d"])
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
    kept = check_pointing_groups([bl, dup], {bl.pointing_group_id: "keep_first"})
    assert [s.id for s in kept] == [bl.id]


def test_engine_refuses_duplicate_pointing_groups(graph):
    bl = graph.surveys["bl_enriquez_2017_692"]
    dup = bl.model_copy(update={"id": "bl_dup"})
    g2 = FactorGraph(graph.root, graph.registry, graph.hypotheses, {**graph.surveys, "bl_dup": dup})
    with pytest.raises(DoubleCountingError):
        build_context(g2, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=1))


def test_conditioned_run_never_reapplies_earth_factor_via_solar_system(graph):
    ctx = build_context(
        graph, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=1, n_mc=50)
    )
    out = evaluate_hypothesis(graph.hypotheses["frequent_slow_expanders"], ctx)
    assert out["solar_system"]["subsumed_by_earth_unreached_conditioning"] is True
    assert out["components"]["solar_system"]["log10_p_data_given_h"] == 0.0
    assert "earth_unreached" in out["components"]
    assert "radio+solar_system" not in out["components"]
