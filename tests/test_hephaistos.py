"""Mid-IR blend-recovery drill-down (docs/MILESTONE_2.md B1)."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from fermi_desk.adversary.checks import audit_hephaistos_blend_anchors
from fermi_desk.errors import UnitValidationError
from fermi_desk.inference.ir_drilldown import run_ir_drilldown
from fermi_desk.models import hephaistos, infrared

SEPS = np.array([0.5, 3.0, 6.0]) * u.arcsec
FRACS = np.array([0.3, 0.9])


@pytest.fixture(scope="module")
def spec():
    return hephaistos.CutoutSpec()


@pytest.fixture(scope="module")
def grid(spec):
    return hephaistos.blend_recovery_grid(
        spec, SEPS, FRACS, 60, np.random.default_rng(7), n_null=400
    )


def test_centroid_sigma_follows_ren_2026_eq1(spec):
    stat = 6.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)) * 12.5)
    assert spec.centroid_sigma_arcsec == pytest.approx(np.hypot(stat, 0.5), rel=1e-12)


def test_cutout_spec_requires_angle_quantities():
    with pytest.raises(UnitValidationError):
        hephaistos.CutoutSpec(psf_fwhm=6.5)
    with pytest.raises(UnitValidationError):
        hephaistos.CutoutSpec(sigma_sys=0.5 * u.pc)


def test_recovery_grows_with_separation_and_flux_fraction(grid):
    for rec in (grid.centroid_recovery, grid.image_recovery):
        by_sep = rec.mean(axis=1)
        assert by_sep[0] < by_sep[1] < by_sep[2]
        assert rec[2, 1] >= 0.9  # wide, background-dominated blends are caught
        assert rec[0].max() <= 0.1  # sub-arcsec blends (Hephaistos D/E regime) are missed
        assert np.all(rec[:, 1] >= rec[:, 0] - 0.1)


def test_blend_free_false_alarm_rates_are_calibrated(grid, spec):
    # Empirical false-alarm rate (n = 400) stays near the nominal 0.27%.
    for fa in (grid.centroid_false_alarm, grid.image_false_alarm):
        assert fa <= 5 * spec.false_alarm + 3 * np.sqrt(spec.false_alarm / 400)


def test_grid_is_seed_deterministic(spec):
    a = hephaistos.blend_recovery_grid(spec, SEPS, FRACS, 8, np.random.default_rng(11))
    b = hephaistos.blend_recovery_grid(spec, SEPS, FRACS, 8, np.random.default_rng(11))
    assert a.as_dict() == b.as_dict()
    with pytest.raises(TypeError):
        hephaistos.blend_recovery_grid(spec, SEPS, FRACS, 8, 11)


def test_zero_blend_survival_and_screened_calibration(grid):
    s = hephaistos.blend_survival(grid, "centroid", 3.25 * u.arcsec)
    assert 0.0 <= s <= 1.0
    cal = hephaistos.screened_calibration("x", 0.5, 1.2e-5, 1.0, 0.0)
    assert cal.false_positive_rate == 1.2e-5 and cal.completeness == 0.5
    # A silent sky (q = 0) has LR exactly 1 against itself under any screened calibration.
    assert infrared.log_lr_vs_silent(7, 200_000, np.array([0.0]), cal) == 0.0


def test_photocentre_shift_is_flux_weighted():
    shift = hephaistos.predicted_photocentre_shift(1.0 * u.arcsec, 0.85)
    assert shift.to_value(u.arcsec) == pytest.approx(0.85)


def test_published_d_e_centroid_verdicts_reproduce(graph, spec):
    f = audit_hephaistos_blend_anchors(graph, spec)
    rows = (
        {r["candidate"]: r for r in f["details"]["candidates"]}
        if isinstance(f, dict)
        else {r["candidate"]: r for r in f.details["candidates"]}
    )
    assert rows["D"]["reproduced_verdict"] == "no_evidence" == rows["D"]["published_verdict"]
    assert rows["E"]["reproduced_verdict"] == "marginal" == rows["E"]["published_verdict"]
    assert rows["D"]["shift_tension_sigma"] < 1.0  # point-blend model fits D
    assert rows["E"]["shift_tension_sigma"] > 2.0  # extended galaxy (E) is a reported tension


def test_drilldown_contamination_scenarios_bracket_the_preference(graph):
    rec = run_ir_drilldown(
        graph, seed=3, pair=("quiet_expanders", "partial_dyson_conspicuous"), n_trials=6
    )
    rows = rec["results"]["likelihood_ratio"]["rows"]
    by = {(r["contamination_scenario"], r["screen"]): r for r in rows}
    low = by[("suazo_2024_subsample_convention", "none")]
    high = by[("zackrisson_2026_full_sample_critique", "none")]
    assert high["expected_false_positives"] == pytest.approx(60.0)
    assert low["expected_false_positives"] == pytest.approx(2.4)
    assert low["lr"] < 1.0 < high["lr"]  # more false positives -> quiet favoured
    assert rec["results"]["data_tier"] == "illustrative_calibration"
    again = run_ir_drilldown(
        graph, seed=3, pair=("quiet_expanders", "partial_dyson_conspicuous"), n_trials=6
    )
    assert again["results_sha256"] == rec["results_sha256"]
