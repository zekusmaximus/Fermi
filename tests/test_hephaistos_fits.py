"""Launch Gate #2 kernels (models/hephaistos_fits.py) on offline SYNTHETIC cutouts.

Every cutout here is generated in-test (Gaussian PSF + white noise); nothing is read
from the network or the FITS cache, and nothing is labelled as archival data.
"""

from __future__ import annotations

import copy

import astropy.units as u
import numpy as np
import pytest

from fermi_desk.errors import LeakageError, ProvenanceError, UnitValidationError
from fermi_desk.inference.engine import canonical_sha256
from fermi_desk.models import hephaistos_fits as hf
from fermi_desk.registry import _close
from fermi_desk.schemas import WiseFitsGate2File

PIX = 1.375 * u.arcsec
SEPS = np.array([0.5, 3.0, 6.0]) * u.arcsec
FRACS = (0.5, 0.9)
SIGMA_SYS = 0.4356 * u.arcsec


def synth_cutout(rng, sid, split, *, fwhm=9.0, snr=None, sys=0.4356, shape=(44, 45)):
    """Synthetic W3-like cutout: Gaussian PSF of ``fwhm`` arcsec, white noise, sky 485 DN."""
    pix = PIX.to_value(u.arcsec)
    s = fwhm / hf.FWHM_PER_SIGMA / pix
    snr = float(rng.uniform(8.0, 20.0)) if snr is None else snr
    gx = (shape[1] - 1) / 2 + rng.uniform(-0.5, 0.5)
    gy = (shape[0] - 1) / 2 + rng.uniform(-0.5, 0.5)
    tx, ty = gx + rng.normal(0, sys / pix), gy + rng.normal(0, sys / pix)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    flux = 1000.0
    sig = flux / (snr * np.sqrt(4 * np.pi) * s)
    star = flux * np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)
    img = 485.0 + star + rng.normal(0, sig, shape)
    return hf.W3Cutout(sid, split, img, np.full(shape, sig), (gx, gy), PIX, snr, "SYN")


@pytest.fixture(scope="module")
def cal_cuts():
    rng = np.random.default_rng(2026)
    return [synth_cutout(rng, 1000 + i, "calibration") for i in range(24)]


@pytest.fixture(scope="module")
def wh_cuts():
    rng = np.random.default_rng(2027)
    return [synth_cutout(rng, 5000 + i, "withheld") for i in range(12)]


@pytest.fixture(scope="module")
def psf(cal_cuts):
    return hf.build_empirical_psf(cal_cuts)


@pytest.fixture(scope="module")
def frozen(cal_cuts, psf):
    return hf.calibrate_frozen_screens(cal_cuts, psf, SIGMA_SYS)


def _run(wh_cuts, psf, frozen, seed=42):
    return hf.evaluate_withheld_matched_injection(
        wh_cuts, psf, frozen, seed=seed, separations=SEPS, flux_fractions=FRACS
    )


@pytest.fixture(scope="module")
def result(wh_cuts, psf, frozen):
    return _run(wh_cuts, psf, frozen)


# --------------------------------------------------------------- empirical PSF
def test_empirical_psf_is_unit_normalised_and_flux_scale_invariant(cal_cuts, psf):
    assert psf.stamp.sum() == pytest.approx(1.0, abs=1e-12)
    assert psf.stamp.shape == (2 * hf.STAMP_HALF_PIX + 1,) * 2
    assert set(psf.star_ids) == {c.star_id for c in cal_cuts}
    all_scaled = hf.build_empirical_psf([c.scaled(7.3) for c in cal_cuts])
    np.testing.assert_allclose(all_scaled.stamp, psf.stamp, rtol=0, atol=1e-12)
    one_scaled = hf.build_empirical_psf([cal_cuts[0].scaled(0.02), *cal_cuts[1:]])
    np.testing.assert_allclose(one_scaled.stamp, psf.stamp, rtol=0, atol=1e-12)


def test_empirical_psf_recovers_a_broadened_atlas_like_width():
    rng = np.random.default_rng(5)
    cuts = [synth_cutout(rng, i, "calibration", fwhm=9.0, snr=400.0) for i in range(15)]
    p = hf.build_empirical_psf(cuts)
    assert p.fwhm_gaussian.to_value(u.arcsec) == pytest.approx(9.0, abs=0.1)
    assert p.fwhm_gaussian > hf.NOMINAL_W3_FWHM_ARCSEC * u.arcsec
    rendered = p.render((44, 45), 22.3, 21.1)
    assert rendered.sum() == pytest.approx(1.0, abs=5e-3)


def test_pixel_scale_must_carry_angle_units():
    with pytest.raises(UnitValidationError):
        hf.W3Cutout(1, "calibration", np.zeros((44, 45)), np.ones((44, 45)), (22, 22), 1.375, 10.0)


# ---------------------------------------------------------- zero-leakage guards
def test_zero_leakage_guards(cal_cuts, wh_cuts, psf, frozen):
    with pytest.raises(LeakageError):
        hf.calibrate_frozen_screens([*cal_cuts, wh_cuts[0]], psf, SIGMA_SYS)
    with pytest.raises(LeakageError):
        hf.build_empirical_psf([*cal_cuts[:10], wh_cuts[0]])
    with pytest.raises(LeakageError):
        _run(cal_cuts[:3], psf, frozen)
    c = cal_cuts[0]
    impostor = hf.W3Cutout(
        c.star_id, "withheld", c.flux, c.unc, c.gaia_xy, c.pixel_scale, c.w3snr
    )  # a calibration star relabelled "withheld"
    with pytest.raises(LeakageError):
        _run([impostor], psf, frozen)
    other_psf = hf.build_empirical_psf(cal_cuts[:12])
    with pytest.raises(LeakageError):
        _run(wh_cuts[:2], other_psf, frozen)


def test_frozen_thresholds_come_from_the_calibration_nulls(frozen, cal_cuts):
    assert frozen.calibration_star_ids == tuple(sorted(c.star_id for c in cal_cuts))
    scale = hf.delta_chi2_scale(np.array(frozen.calibration_dchi2))
    assert frozen.dchi2_scale == scale
    assert frozen.dchi2_threshold == hf.delta_chi2_threshold(scale, frozen.target_false_alarm)
    assert frozen.sigma_sys_arcsec == pytest.approx(0.4356)
    assert frozen.centroid_multiplier == pytest.approx(np.sqrt(-2 * np.log(0.0027)))


# ------------------------------------------------ adversarial negative control
def test_matched_centered_excess_exposes_a_naive_bright_excess_rule(result):
    s = result.summary()["screens"]
    naive = s["naive_flux_excess"]
    # "Flag every bright excess" recovers 100% of off-center blends ...
    assert np.all(np.array(naive["arm_a_blend_recovery"]) == 1.0)
    # ... but also rejects ~100% of genuine matched centered excesses.
    assert min(naive["arm_b_false_rejection"]) >= 0.9
    # The frozen image screen keeps centered excesses while still finding wide blends.
    img = s["image"]
    assert img["arm_b_false_rejection_pooled"]["rate"] <= 0.25
    assert np.array(img["arm_a_blend_recovery"])[-1, -1] >= 0.8
    for scr in s.values():
        lo, hi = scr["arm_b_false_rejection_pooled"]["clopper_pearson_95"]
        assert lo <= scr["arm_b_false_rejection_pooled"]["rate"] <= hi


def test_recovery_rises_with_separation(result):
    n = len(result.star_ids)
    for name in ("centroid", "image"):
        rec = np.array(result.summary()["screens"][name]["arm_a_blend_recovery"])
        assert np.all(np.diff(rec, axis=0) >= -1.0 / n - 1e-12), name
        assert np.all(rec[-1] > rec[0]), name


def test_deterministic_seeded_replay(wh_cuts, psf, frozen, result):
    again = _run(wh_cuts, psf, frozen, seed=42)
    assert canonical_sha256(again.as_dict()) == canonical_sha256(result.as_dict())
    other = _run(wh_cuts[:2], psf, frozen, seed=43)
    assert not np.allclose(other.azimuth_rad, result.azimuth_rad[:2])


def test_area_weighted_survival_matches_hand_computation():
    rec = np.array([[0.0, 0.2], [0.5, 1.0], [1.0, 1.0]])
    # theta = 1, 3 inside 3.25 (weights 1/4, 3/4); theta = 6 excluded.
    want = 0.25 * (1.0 + 0.8) / 2 + 0.75 * (0.5 + 0.0) / 2
    assert hf.area_weighted_survival(rec, [1.0, 3.0, 6.0], 3.25) == pytest.approx(want)


# ------------------------------------------------ offline recompute and schema
def test_offline_recompute_reproduces_and_catches_tampering(psf, frozen, result):
    payload = hf.gate2_payload(psf, frozen, result)
    got = hf.recompute_gate2(payload)
    _close(got["summary"], payload["withheld"]["summary"], "summary")
    _close(
        got["calibration_false_alarm_in_sample"],
        payload["frozen_screens"]["calibration_false_alarm_in_sample"],
    )
    assert got["delta_chi2_threshold"] == payload["frozen_screens"]["image"]["delta_chi2_threshold"]
    bad = copy.deepcopy(payload)
    arm_b = bad["withheld"]["statistics"]["arm_b"]["delta_chi2"]
    arm_b[0][0] = 1e6  # a centered excess that now "fails" the image screen
    with pytest.raises(ProvenanceError):
        _close(hf.recompute_gate2(bad)["summary"], bad["withheld"]["summary"], "summary")


def _doc(psf, frozen, result, **over):
    payload = hf.gate2_payload(psf, frozen, result)
    n = len(result.star_ids) + len(frozen.calibration_star_ids)
    files = [{"int_sha256": "a" * 64, "unc_sha256": "b" * 64}] * n
    doc = {
        "id": "test_gate2",
        "schema_version": 1,
        "data_tier": "reproduced_archive",  # schema contract only; never written to data/
        "code_git_sha": "0" * 40,
        "rng_seed": 42,
        "query_window_utc": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
        "frozen_at_utc": "2026-01-01T00:30:00",
        "locked_inputs": {
            "sigma_sys_calibration_path": "x.json",
            "sigma_sys_calibration_sha256": "c" * 64,
            "sigma_sys_arcsec": 0.4356,
        },
        "fetch": {"n_requested": n, "n_ok": n, "n_failed": 0, "files": files},
        "source_receipt": {"path": "runs/x.json", "sha256": "d" * 64},
        "content_sha256": "e" * 64,
        **payload,
    }
    doc.update(over)
    return doc


def test_gate2_schema_contract(psf, frozen, result):
    WiseFitsGate2File.model_validate(_doc(psf, frozen, result))
    with pytest.raises(ProvenanceError):
        WiseFitsGate2File.model_validate(_doc(psf, frozen, result, code_git_sha="dirty"))
    n = len(result.star_ids) + len(frozen.calibration_star_ids)
    incomplete = {"n_requested": n, "n_ok": n - 1, "n_failed": 1, "files": [{}] * (n - 1)}
    with pytest.raises(ProvenanceError):
        WiseFitsGate2File.model_validate(_doc(psf, frozen, result, fetch=incomplete))
    late = _doc(psf, frozen, result, frozen_at_utc="2026-01-02T00:00:00")
    with pytest.raises(ProvenanceError):
        WiseFitsGate2File.model_validate(late)
    leak = _doc(psf, frozen, result)
    leak["withheld"]["star_ids"][0] = frozen.calibration_star_ids[0]
    with pytest.raises(ProvenanceError):
        WiseFitsGate2File.model_validate(leak)
