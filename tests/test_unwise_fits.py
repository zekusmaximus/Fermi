"""v0.2.2a: unsmoothed unWISE W3 gate - kernels, script, schema and validator (100% offline).

Every cutout here is SYNTHETIC (Gaussian PSF, white or correlated Gaussian noise), generated
in-test or by the script's in-memory rehearsal; nothing is read from the network (conftest
blocks it) or from an archive cache, nothing is labelled as archival data outside ``tmp_path``
copies, and no test writes inside the project tree. The only real inputs are the locked
Gate #1 / Gate #2 artifacts, which are read and never written.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json

import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.coordinates import SkyCoord
from pydantic import ValidationError

from conftest import ROOT
from fermi_desk import schemas
from fermi_desk.errors import (
    LeakageError,
    ProvenanceError,
    SchemaValidationError,
    UnitValidationError,
)
from fermi_desk.models import hephaistos_fits as hf
from fermi_desk.models import unwise_fits as uf
from fermi_desk.registry import (
    load_factor_graph,
    load_unwise_gate,
    run_validation,
    seal_unwise_gate,
    unwise_gate_bytes,
    verify_unwise_gate,
)
from fermi_desk.schemas import UNWISE_GATE_PATH, UnwiseGateWithheldFile

IR = "hephaistos_wise_mid_ir"
SEPS = np.array([0.5, 3.0, 6.0]) * u.arcsec
FRACS = (0.5, 0.9)
SIGMA_SYS = 0.4356 * u.arcsec
T_CAL = "2026-01-01T00:00:00+00:00"
T_FREEZE = "2026-01-01T01:00:00+00:00"
T_WH = "2026-01-01T02:00:00+00:00"
CLEAN_SHA = "0" * 40  # schema contract only; never written into data/
LOCKED = {
    "data/catalogs/bl_692_targets.csv": (
        "c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2"
    ),
    "data/calibrations/wise_sigma_sys_control.json": (
        "2da2a234ee0ccb07fb571431f5f178e362ff4d9d4f96d4ee878b921916127b7a"
    ),
    "data/calibrations/wise_fits_gate2_withheld.json": (
        "d4818c100b02cea98d5beafcdea66e679c072e5d47e28fe8d3a960f78c64da99"
    ),
    "data/catalogs/bl_gbt_lband_background_summary.json": (
        "eb43b118ab4d0258d9fe20474b73d187cb62a98018b20790b4112f875b0018a9"
    ),
}
slow = pytest.mark.slow


# ------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "run_unwise_gate", ROOT / "scripts" / "run_unwise_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def locked(script):
    return script.load_locked()


@pytest.fixture(scope="module")
def args(script):
    return script.build_parser().parse_args(["--dry-run"])


@pytest.fixture(scope="module")
def run(script, args, locked):
    """The script's in-memory SYNTHETIC rehearsal: 60 calibration + 60 withheld stars,
    6.5 arcsec PSF, correlated pixel noise, a reduced injection grid (synthetic_proxy)."""
    r = script.rehearse(args, locked, n_stars=120, separations=SEPS, flux_fractions=FRACS)
    assert r.receipt["status"] == "ok" and r.receipt["data_tier"] == "synthetic_proxy"
    return r


@pytest.fixture(scope="module")
def doc(script, run):
    return script.artifact_doc(run.receipt, None)


def _cutouts(n, split, seed, *, corr=0.0, fwhm=6.5, first=1000, when=T_CAL, **kw):
    rng = np.random.default_rng(seed)
    return [
        uf.synthetic_cutout(
            rng,
            str(first + i),
            split,
            psf_fwhm=fwhm * u.arcsec,
            noise_correlation_fwhm=corr * u.arcsec,
            fetched_at_utc=when,
            **kw,
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def white():
    """Calibration frozen on WHITE synthetic noise (same PSF), for the noise-model contrast."""
    return uf.freeze_calibration(_cutouts(30, "calibration", 2026), SIGMA_SYS, conformal_alpha=0.05)


def _reseal(d):
    return seal_unwise_gate(d)


def _as_archive(receipt):
    """A synthetic receipt relabelled as a clean archive run (tmp_path / in-memory use only)."""
    rec = copy.deepcopy(receipt)
    rec["data_tier"] = "reproduced_archive"
    rec["code_git_sha"] = CLEAN_SHA
    rec["frozen_state_file"]["code_git_sha"] = CLEAN_SHA
    for f in rec["fetch"]["files"]:
        f["code_git_sha"] = CLEAN_SHA
    return rec


# --------------------------------------------- fresh patches & Gate #1 disjointness
def test_fresh_patches_are_disjoint_from_all_12_gate1_patches(script, args, locked):
    patches, _, audit = script.draw_patches(args, locked)
    assert len(patches) == 16 and len(locked.patch_ids) == 12 and len(locked.star_ids) == 154
    centres = SkyCoord(
        l=[p["l_deg"] for p in patches] * u.deg,
        b=[p["b_deg"] for p in patches] * u.deg,
        frame="galactic",
    )
    for i, p in enumerate(patches):
        assert abs(p["b_deg"]) - p["radius_deg"] > 30.0  # the whole cone, not just the centre
        sep = centres[i].separation(locked.centres).deg  # independent of the kernel's own check
        assert sep.min() >= 15.0
        assert sep.min() == pytest.approx(p["nearest_locked_patch_separation_deg"], abs=1e-9)
        assert locked.patch_ids[int(np.argmin(sep))] == p["nearest_locked_patch_id"]
        others = np.delete(centres[i].separation(centres).deg, i)
        assert others.min() >= 20.0
        for _, al, ab, ar in uf.AVOID_FIELDS:
            away = centres[i].separation(SkyCoord(l=al * u.deg, b=ab * u.deg, frame="galactic"))
            assert away.deg >= ar + p["radius_deg"]
    assert audit["n_locked_patches"] == 12 and audit["min_locked_separation_deg"] >= 15.0
    again, _, _ = script.draw_patches(args, locked)
    assert again == patches  # deterministic seeded replay
    other = argparse.Namespace(**{**vars(args), "seed": 43})
    assert script.draw_patches(other, locked)[0] != patches


def test_patch_near_a_gate1_patch_or_at_low_latitude_is_rejected(locked):
    c = locked.centres[1]  # locked patch P02 (b = +72 deg)
    near = {"patch_id": "X", "l_deg": c.l.deg + 20.0, "b_deg": c.b.deg, "radius_deg": 2.0}
    sep = SkyCoord(l=near["l_deg"] * u.deg, b=near["b_deg"] * u.deg, frame="galactic").separation(c)
    assert 2.0 < sep.deg < 15.0
    with pytest.raises(LeakageError, match="locked Gate #1 patch"):
        uf.patch_separation_audit([near], locked.centres)
    low = {"patch_id": "Y", "l_deg": 40.0, "b_deg": 31.0, "radius_deg": 2.0}
    with pytest.raises(SchemaValidationError, match=r"\|b\| < 30"):
        uf.patch_separation_audit([low], locked.centres)
    lmc = {"patch_id": "Z", "l_deg": 280.47, "b_deg": -34.0, "radius_deg": 2.0}
    with pytest.raises(SchemaValidationError, match="LMC"):
        uf.patch_separation_audit([lmc], locked.centres)
    with pytest.raises(RuntimeError, match="separation constraints"):
        uf.choose_fresh_patches(42, locked.centres, n_patches=400, max_draws=400)
    with pytest.raises(UnitValidationError):
        uf.choose_fresh_patches(42, locked.centres, radius=2.0)  # bare float: no units


def test_shared_gaia_id_with_the_locked_sample_raises_leakage_error(locked, run):
    fresh = [c["gaia_dr3_source_id"] for c in run.receipt["sample"]["controls"]]
    assert uf.assert_disjoint_from_locked(fresh, locked.star_ids) == len(fresh) == 120
    stolen = str(locked.star_ids[7])  # locked ids are integers on disk; fresh ids are strings
    with pytest.raises(LeakageError, match="locked Gate #1"):
        uf.assert_disjoint_from_locked([*fresh, stolen], locked.star_ids)
    with pytest.raises(SchemaValidationError, match=r"v0.2.1-R4"):
        uf.assert_disjoint_from_locked([locked.star_ids[7]], [])  # an integer id is refused
    with pytest.raises(SchemaValidationError, match="duplicate"):
        uf.assert_disjoint_from_locked([fresh[0], fresh[0]], locked.star_ids)


def test_sample_stage_refuses_a_locked_star_returned_by_the_archive(script, args, locked):
    class Leaky(script.SyntheticArchive):
        def candidates(self, patch, a):
            rows, meta = super().candidates(patch, a)
            if patch["patch_id"] == "U03":
                rows[0]["gaia_dr3_source_id"] = str(locked.star_ids[0])
                self._neighbours[rows[0]["gaia_dr3_source_id"]] = 0
            return rows, meta

    small = argparse.Namespace(**{**vars(args), "n_controls": 32})
    with pytest.raises(LeakageError, match="locked Gate #1"):
        script.build_sample(Leaky(42, 32, 16), locked, small, script.Clock())


def test_control_cuts_match_the_gate1_isolation_protocol(run):
    cuts = uf.ControlCuts()
    good = {
        "gaia_dr3_source_id": "11",
        "b_deg": -41.0,
        "ruwe": 1.1,
        "w3snr": 12.0,
        "cc_flags": "0000",
        "ext_flg": 0,
        "n_gaia_neighbours": 0,
    }
    fails = [
        ("after_star_abs_b_cut", {"b_deg": 29.9}),
        ("after_ruwe_cut", {"ruwe": 1.4}),
        ("after_w3_snr_band", {"w3snr": 7.9}),
        ("after_w3_snr_band", {"w3snr": 20.1}),
        ("after_cc_flags_0000", {"cc_flags": "h000"}),
        ("after_ext_flg_0", {"ext_flg": 1}),
        ("after_isolation", {"n_gaia_neighbours": 1}),
        ("after_isolation", {"n_gaia_neighbours": None}),  # not yet queried is not isolated
    ]
    rows = [
        good,
        *[{**good, "gaia_dr3_source_id": str(20 + i), **f} for i, (_, f) in enumerate(fails)],
    ]
    kept, counts = uf.apply_control_cuts(rows, cuts)
    assert [r["gaia_dr3_source_id"] for r in kept] == ["11"]
    keys = [k for k, _ in cuts.steps()]
    assert [counts[k] for k in keys] == [8, 7, 5, 4, 3, 1] and counts["candidates"] == 9
    for key, change in fails:  # each decoy is stopped at exactly its own cut
        ok = cuts.passes({**good, **change})
        assert ok.index(False) == keys.index(key)
    with pytest.raises(UnitValidationError):
        uf.ControlCuts(isolation=15.0)
    # The rehearsal sample: every selected control passes every cut; the decoys were counted.
    sample = run.receipt["sample"]
    kept, _ = uf.apply_control_cuts(sample["controls"], cuts)
    assert len(kept) == len(sample["controls"]) == 120
    total = sample["cut_counts_total"]
    assert total["joined_candidates"] == 120 + 4 * 16
    assert total["after_ruwe_cut"] == total["joined_candidates"] - 16
    assert total["rejected_gaia_neighbour_within_isolation"] == 16
    assert total["selected_controls"] == 120


def test_split_is_stratified_50_50_and_seeded(run):
    controls = run.receipt["sample"]["controls"]
    order = [p["patch_id"] for p in run.receipt["sample"]["patches"]]
    split = uf.split_calibration_withheld(controls, order, 42)
    assert split == {c["gaia_dr3_source_id"]: c["split"] for c in controls}
    assert sorted(split.values()).count("calibration") == 60
    for pid in order:
        mine = [split[c["gaia_dr3_source_id"]] for c in controls if c["patch_id"] == pid]
        assert abs(mine.count("calibration") - mine.count("withheld")) <= 1
    assert uf.split_calibration_withheld(controls, order, 43) != split
    with pytest.raises(ValueError, match="no declared patch"):
        uf.split_calibration_withheld(controls, order[:-1], 42)


# ------------------------------------------------- geometry & Gate #2 equivalence
def test_geometry_is_gate2_in_arcsec():
    atlas = uf.Geometry.for_pixel_scale(1.375 * u.arcsec)
    assert (atlas.fit_radius_pix, atlas.sky_inner_pix, atlas.stamp_half_pix) == (
        hf.FIT_RADIUS_PIX,
        hf.SKY_INNER_PIX,
        hf.STAMP_HALF_PIX,
    )
    assert (atlas.primary_range_pix, atlas.secondary_max_pix) == (
        hf.PRIMARY_RANGE_PIX,
        hf.SECONDARY_MAX_PIX,
    )
    assert (atlas.fine_step_pix, atlas.coarse_step_pix) == (0.25, 0.5)
    native = uf.Geometry.for_pixel_scale(uf.UNWISE_PIXEL_SCALE)
    assert (native.fit_radius_pix, native.sky_inner_pix, native.stamp_half_pix) == (6.0, 8.0, 7)
    assert (native.max_lag_pix, native.n_torus) == (6, 32)
    with pytest.raises(UnitValidationError):
        uf.Geometry.for_pixel_scale(2.75)


def test_scalar_statistic_reproduces_the_gate2_screen_at_the_atlas_pixel_scale():
    """Screen A is the Gate #2 statistic: the same Delta chi^2 for the same PSF, to rounding."""
    pix = 1.375 * u.arcsec
    cuts = _cutouts(12, "calibration", 3, fwhm=9.0, shape=(44, 45), pixel_scale=pix)
    old = [
        hf.W3Cutout(int(c.star_id), "calibration", c.flux, c.unc, c.gaia_xy, pix, c.w3snr)
        for c in cuts
    ]
    psf = uf.stack_empirical_psf(cuts).psf()
    assert psf.stamp.shape == hf.build_empirical_psf(old).stamp.shape == (31, 31)
    geom = uf.Geometry.for_pixel_scale(pix)
    sys_pix = SIGMA_SYS.to_value(u.arcsec) / 1.375
    for c, o in zip(cuts[:4], old[:4], strict=True):
        bank = hf._ScreenBank(psf, o, sys_pix)  # Gate #2's own, untouched implementation
        tm = uf._Templates(psf, c, geom)
        stat = uf._Statistic(tm, uf.scalar_projector(tm.n, uf.scalar_variance(c, geom)), sys_pix)
        assert stat.delta_chi2(c.flux) == pytest.approx(bank.delta_chi2(o.flux), rel=1e-6)


# ------------------------------------------------------------------ empirical PSF
def test_psf_stack_recovers_6p5_arcsec_and_discriminates_the_atlas_width(script, run):
    sharp = uf.stack_empirical_psf(_cutouts(15, "calibration", 5, snr=400.0)).psf()
    w = uf.psf_widths_arcsec(sharp.stamp, sharp.pix)
    assert w["fwhm_gaussian_arcsec"] == pytest.approx(6.5, abs=0.15)
    assert w["fwhm_halfmax_arcsec"] == pytest.approx(6.5, abs=0.3)  # model-free, oversampled
    assert sharp.stamp.sum() == pytest.approx(1.0, abs=1e-12)
    assert sharp.stamp.shape == (15, 15)
    smooth = uf.stack_empirical_psf(_cutouts(15, "calibration", 6, snr=400.0, fwhm=13.69)).psf()
    ws = uf.psf_widths_arcsec(smooth.stamp, smooth.pix)
    assert ws["fwhm_gaussian_arcsec"] == pytest.approx(13.69, abs=0.5)
    assert ws["fwhm_gaussian_arcsec"] > 1.9 * w["fwhm_gaussian_arcsec"]
    assert ws["fwhm_halfmax_arcsec"] > 1.7 * w["fwhm_halfmax_arcsec"]
    # At the drill-down W3 SNR (8-20) the two widths are still far apart ...
    d = script.psf_discrimination(42)
    assert abs(d["unsmoothed"]["fwhm_gaussian_arcsec"] - 6.5) < 0.6
    assert d["atlas_like"]["fwhm_gaussian_arcsec"] > 11.5
    # ... and the rehearsal's calibration-half PSF is the unsmoothed one, not the Atlas one.
    psf = run.receipt["empirical_psf"]
    assert abs(psf["fwhm_gaussian_arcsec"] - 6.5) < 0.6 < 13.69 - psf["fwhm_gaussian_arcsec"]
    assert psf["pixel_scale_arcsec"] == 2.75 and psf["nominal_fwhm_arcsec"] == 6.5
    flux_scaled = uf.stack_empirical_psf([c.scaled(7.3) for c in _cutouts(8, "calibration", 5)])
    same = uf.stack_empirical_psf(_cutouts(8, "calibration", 5))
    np.testing.assert_allclose(flux_scaled.psf().stamp, same.psf().stamp, rtol=0, atol=1e-12)


def test_matched_filter_normalisation_avoids_the_gate2_positive_sum_selection():
    """The one deliberate change to the Gate #2 PSF recipe, and why it was made.

    Gate #2 divides every stamp by its own summed flux and drops the stamps whose sum is not
    positive. On 2.75 arcsec pixels that sum is noise-dominated: under correlated noise the
    rule discards the stars with the most negative noise, and what is left stacks to a PSF
    with part of its unit sum in a flat pedestal instead of the core.
    """
    yy, xx = np.mgrid[-7:8, -7:8]
    core = xx**2 + yy**2 <= 36  # the 16.5 arcsec fit aperture
    enclosed = {"gate2": [], "matched": []}
    for seed in (1, 2, 3):
        cuts = _cutouts(60, "calibration", seed, corr=4.5)
        old = [
            hf.W3Cutout(int(c.star_id), "calibration", c.flux, c.unc, c.gaia_xy, c.pixel_scale, 9.0)
            for c in cuts
        ]
        gate2 = hf.build_empirical_psf(old, stamp_half=7)  # the Gate #2 recipe, unmodified
        stack = uf.stack_empirical_psf(cuts)
        assert len(stack.star_ids) == 60 and not stack.excluded  # every star is kept ...
        assert len(gate2.star_ids) <= 57  # ... where the summed-flux rule drops several
        assert {e["reason"] for e in gate2.excluded} == {"non-positive sky-subtracted stamp flux"}
        enclosed["gate2"].append(float(gate2.stamp[core].sum()))
        enclosed["matched"].append(float(stack.psf().stamp[core].sum()))
    assert np.mean(enclosed["matched"]) == pytest.approx(1.0, abs=0.06)
    assert np.mean(enclosed["gate2"]) < np.mean(enclosed["matched"]) - 0.03
    # Under white noise the two recipes agree: the change matters only where it should.
    cuts = _cutouts(60, "calibration", 1)
    old = [
        hf.W3Cutout(int(c.star_id), "calibration", c.flux, c.unc, c.gaia_xy, c.pixel_scale, 9.0)
        for c in cuts
    ]
    a = hf.build_empirical_psf(old, stamp_half=7).stamp
    b = uf.stack_empirical_psf(cuts).psf().stamp
    assert abs(float(a[core].sum()) - float(b[core].sum())) < 0.03
    assert a[7, 7] == pytest.approx(b[7, 7], rel=0.05)


def test_leave_one_out_psf_excludes_exactly_one_star():
    stack = uf.stack_empirical_psf(_cutouts(9, "calibration", 8, snr=50.0))
    full, loo = stack.psf(), stack.psf(leave_out=stack.star_ids[3])
    assert len(full.star_ids) == 9 and stack.star_ids[3] not in loo.star_ids
    assert len(loo.star_ids) == 8 and not np.array_equal(full.stamp, loo.stamp)
    with pytest.raises(ValueError, match="usable calibration stamps"):
        uf.stack_empirical_psf(_cutouts(4, "calibration", 8))


# --------------------------------------------------------------- noise covariance
def _indefinite_raw(lag=6):
    """An 'autocovariance' no real process has: lag-1 = +0.9, lag-2 = -0.9 of the variance."""
    raw = np.zeros((2 * lag + 1, 2 * lag + 1))
    raw[lag, lag] = 1.0
    for d, v in ((1, 0.9), (2, -0.9)):
        for dy, dx in ((0, d), (d, 0), (0, -d), (-d, 0)):
            raw[lag + dy, lag + dx] = v
    return raw


def test_pixel_covariance_is_strictly_positive_definite_and_deterministic(run):
    noise, geom = run.frozen.noise, run.frozen.geometry
    torus = noise.torus_covariance
    v0 = noise.variance_inflation
    assert torus.shape == (32, 32) and v0 == pytest.approx(1.0, abs=0.12)
    # The circulant spectrum bounds every per-star matrix from below: >= shrinkage * V(0).
    assert uf.minimum_power(torus) >= uf.NOISE_SHRINKAGE * v0 * (1 - 1e-12)
    flip = (-np.arange(32)) % 32
    np.testing.assert_array_equal(torus, torus[np.ix_(flip, flip)])  # V(d) == V(-d) exactly
    span = np.arange(-6, 7)
    yy, xx = np.meshgrid(span, span, indexing="ij")
    disk = xx**2 + yy**2 <= geom.fit_radius_pix**2
    sigma = noise.matrix(yy[disk], xx[disk])
    assert sigma.shape == (113, 113)
    np.testing.assert_array_equal(sigma, sigma.T)
    eig = np.linalg.eigvalsh(sigma)
    assert eig.min() >= uf.NOISE_SHRINKAGE * v0 * (1 - 1e-9) > 0
    np.linalg.cholesky(sigma)  # raises LinAlgError unless strictly positive definite
    # The synthetic noise is correlated over 4.5 arcsec FWHM: lag-1 correlation ~0.6.
    rho = torus[0, 1] / v0
    assert 0.45 < rho < 0.62 and torus[1, 0] / v0 == pytest.approx(rho, abs=0.05)
    # Deterministic: the same lag table always regularises to the same bits ...
    again = uf.regularise_lag_covariance(noise.raw_lag_covariance, 32)
    np.testing.assert_array_equal(again, torus)
    # ... and even an indefinite input comes out strictly positive definite.
    bad = _indefinite_raw()
    grid = np.zeros((32, 32))
    grid[np.ix_(span % 32, span % 32)] = bad
    assert np.fft.fft2(grid).real.min() < 0  # negative power: not a covariance at all
    fixed = uf.regularise_lag_covariance(bad, 32)
    assert uf.minimum_power(fixed) >= uf.NOISE_SHRINKAGE * (1 - 1e-12)
    assert fixed[0, 0] == pytest.approx(1.0, abs=1e-12)
    np.linalg.cholesky(uf.NoiseModel(fixed, bad, bad, ()).matrix(yy[disk], xx[disk]))
    with pytest.raises(ValueError, match="positive"):
        uf.regularise_lag_covariance(np.zeros((13, 13)), 32)
    with pytest.raises(ValueError, match="torus"):
        uf.regularise_lag_covariance(bad, 16)  # lags would alias


def test_noise_samples_are_seed_deterministic_and_calibration_only(run):
    cal = run.calibration_cutouts
    a = uf.collect_noise_samples(cal).model()
    b = uf.collect_noise_samples(list(reversed(cal))).model()  # order-independent
    np.testing.assert_array_equal(a.torus_covariance, b.torus_covariance)
    np.testing.assert_array_equal(a.torus_covariance, run.frozen.noise.torus_covariance)
    loo = uf.collect_noise_samples(cal).model(leave_out=cal[0].star_id)
    assert cal[0].star_id not in loo.star_ids and len(loo.star_ids) == len(a.star_ids) - 1
    assert not np.array_equal(loo.torus_covariance, a.torus_covariance)
    white_model = uf.collect_noise_samples(_cutouts(20, "calibration", 31)).model()
    w = white_model.torus_covariance
    assert abs(w[0, 1] / w[0, 0]) < 0.05 and abs(w[1, 0] / w[0, 0]) < 0.05


def test_projectors_whiten_and_remove_the_sky():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(40, 40))
    sigma = a @ a.T + 40 * np.eye(40)
    m = uf.gls_projector(sigma)
    unit = np.linalg.solve(np.linalg.cholesky(sigma), np.ones(40))
    unit /= np.linalg.norm(unit)
    np.testing.assert_allclose(m @ sigma @ m.T, np.eye(40) - np.outer(unit, unit), atol=1e-10)
    np.testing.assert_allclose(m @ np.ones(40), 0.0, atol=1e-10)  # a constant sky vanishes
    s = uf.scalar_projector(40, 4.0)
    y = rng.normal(size=40)
    np.testing.assert_allclose(s @ y, (y - y.mean()) / 2.0, atol=1e-12)
    # White covariance: GLS and scalar projectors give the same chi^2.
    g = uf.gls_projector(4.0 * np.eye(40))
    assert float((g @ y) @ (g @ y)) == pytest.approx(float((s @ y) @ (s @ y)), rel=1e-12)
    with pytest.raises(np.linalg.LinAlgError):
        uf.gls_projector(np.ones((3, 3)))  # singular: not positive definite


@slow
def test_correlated_noise_inflates_scalar_delta_chi2_and_gls_restores_it(run, white):
    """Correlated pixel noise breaks the scalar-variance statistic; whitening repairs it.

    ``white`` was frozen on white noise, ``run`` on noise correlated over 4.5 arcsec. The
    uncertainty maps are identical pixel by pixel in both, so everything that differs is
    correlation the scalar model cannot see.
    """
    w = white.rules()["screens"]
    c = run.frozen.rules()["screens"]
    s_white = w["image_scalar_chi2"]["delta_chi2_scale"]
    # Under white noise the two statistics coincide (the estimated covariance is ~diagonal).
    assert w["image_gls_chi2"]["delta_chi2_scale"] == pytest.approx(s_white, rel=0.1)
    # Correlation inflates the scalar null statistic far more than the whitened one.
    s_scalar = c["image_scalar_chi2"]["delta_chi2_scale"]
    s_gls = c["image_gls_chi2"]["delta_chi2_scale"]
    assert s_scalar > 1.8 * s_white
    assert s_gls < 1.5 * s_white and s_gls < 0.7 * s_scalar
    # Out of sample: on the 60 WITHHELD correlated nulls, a scalar threshold frozen under the
    # white-noise assumption false-alarms far above its nominal 0.27%; the GLS statistic
    # against the equivalent threshold does not.
    null = run.withheld.stats["null"]
    n = len(run.withheld.star_ids)
    fa_scalar = int((null["delta_chi2_scalar"] > w["image_scalar_chi2"]["threshold"]).sum())
    fa_gls = int((null["delta_chi2_gls"] > w["image_gls_chi2"]["threshold"]).sum())
    assert fa_scalar / n > 10 * white.target_false_alarm
    assert fa_gls <= 1 and fa_gls < fa_scalar
    # With thresholds frozen on the matching (correlated) calibration half, every image
    # screen is back under control on the withheld nulls.
    s = run.receipt["withheld"]["summary"]["screens"]
    for name in ("image_scalar_chi2", "image_gls_chi2"):
        assert s[name]["null_false_alarm"]["k"] <= 1, name
    for name in uf.CONFORMAL_SCREENS:
        level = c[name]["alpha_guaranteed"]
        lo, _ = s[name]["null_false_alarm"]["clopper_pearson_95"]
        assert level == pytest.approx(1 / 61) and lo <= 3 * level, name


# --------------------------------------------------------------- conformal threshold
def test_conformal_threshold_finite_sample_level_is_k_plus_1_over_n_plus_1():
    scores = np.arange(1.0, 81.0)
    t = uf.conformal_threshold(scores, 0.02)  # n = 80: only the maximum attains 2%
    assert (t["k_exceedances"], t["order_statistic_rank_from_top"]) == (0, 1)
    assert t["threshold"] == 80.0 and t["alpha_guaranteed"] == pytest.approx(1 / 81)
    assert t["n_calibration_strictly_above"] == 0
    t = uf.conformal_threshold(np.arange(1.0, 101.0), 0.02)  # n = 100: second largest
    assert t["k_exceedances"] == 1 and t["threshold"] == 99.0
    assert t["alpha_guaranteed"] == pytest.approx(2 / 101)
    t = uf.conformal_threshold(np.arange(1.0, 100.0), 0.01)  # n = 99: 1/100 exactly
    assert t["k_exceedances"] == 0 and t["alpha_guaranteed"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match=r"not attainable with n_cal = 98.*n_cal >= 99"):
        uf.conformal_threshold(np.arange(1.0, 99.0), 0.01)
    with pytest.raises(ValueError, match="finite"):
        uf.conformal_threshold([1.0, np.nan], 0.5)
    for n, alpha in ((60, 0.02), (60, 0.05), (77, 0.05), (200, 0.01)):
        t = uf.conformal_threshold(np.arange(float(n)), alpha)
        k = t["k_exceedances"]
        assert (k + 1) / (n + 1) <= alpha < (k + 2) / (n + 1)  # the largest attainable k


def test_conformal_level_holds_for_heavy_tails_without_a_chi2_assumption():
    """Monte Carlo of the exchangeability argument, on scores with no finite variance."""
    rng = np.random.default_rng(np.random.SeedSequence([42, 0xC0F]))
    trials = 6000
    for n, alpha in ((60, 0.02), (60, 0.05), (100, 0.02)):
        scores = rng.standard_cauchy((trials, n + 1)) ** 2  # heavy-tailed, far from chi2(3)
        level = uf.conformal_threshold(scores[0, :n], alpha)["alpha_guaranteed"]
        hits = 0
        for row in scores:
            hits += row[n] > uf.conformal_threshold(row[:n], alpha)["threshold"]
        se = np.sqrt(level * (1 - level) / trials)
        assert abs(hits / trials - level) < 4 * se, (n, alpha, hits / trials, level)
        # The chi2(3) median-matched rule has no such guarantee on these scores.
        chi2_hits = 0
        for row in scores[:1500]:
            thr = hf.delta_chi2_threshold(hf.delta_chi2_scale(row[:n]), level)
            chi2_hits += row[n] > thr
        assert chi2_hits / 1500 > 3 * level


def test_frozen_thresholds_follow_from_the_stored_calibration_nulls(run, locked):
    fz = run.receipt["frozen_screens"]
    cal, sc = fz["calibration"], fz["screens"]
    n = len(cal["star_ids"])
    assert n == 60 and fz["conformal_alpha_nominal"] == 0.02 and fz["target_false_alarm"] == 0.0027
    for name, stat in uf.CHI2_SCREENS.items():  # Screens A and B: the Gate #2 rule, in-sample
        scale = hf.delta_chi2_scale(np.array(cal[f"delta_chi2_{stat}"]))
        assert sc[name]["delta_chi2_scale"] == scale
        assert sc[name]["threshold"] == hf.delta_chi2_threshold(scale, 0.0027)
    for name, stat in uf.CONFORMAL_SCREENS.items():  # Screen C: leave-one-out order statistic
        loo = np.sort(cal[f"delta_chi2_{stat}_loo"])
        assert (sc[name]["k_exceedances"], sc[name]["n_calibration"]) == (0, 60)
        assert sc[name]["threshold"] == loo[-1]
        assert sc[name]["alpha_guaranteed"] == pytest.approx(1 / 61)
        assert fz["calibration_false_alarm_in_sample"][name]["k"] == 0
        # Leave-one-out scores are a different (out-of-sample) set of numbers.
        assert not np.allclose(cal[f"delta_chi2_{stat}_loo"], cal[f"delta_chi2_{stat}"])
    locked_sys = float(locked.sigma_sys.to_value(u.arcsec))  # 0.4356 arcsec, Gate #1
    assert sc["centroid_locked"]["sigma_sys_arcsec"] == locked_sys
    assert locked_sys == pytest.approx(0.4356, abs=5e-5)
    fresh = fz["fresh_sigma_sys"]
    assert sc["centroid_fresh"]["sigma_sys_arcsec"] == fresh["sigma_sys_arcsec"] != locked_sys
    assert fresh["primary_estimator"] == "robust_median" and fresh["n_stars"] == 60
    assert set(fresh["sigma_sys_by_estimator_arcsec"]) == {
        "robust_median",
        "rayleigh_mle",
        "moment",
    }
    assert sc["centroid_locked"]["multiplier"] == pytest.approx(np.sqrt(-2 * np.log(0.0027)))
    with pytest.raises(ValueError, match="not attainable"):
        uf.frozen_screen_rules(cal, locked_sys, 0.0027, 0.01)  # 1% needs n_cal >= 99


# ---------------------------------------------------------- zero-leakage state machine
def test_no_fitting_step_accepts_a_withheld_cutout(run):
    cal, wh = run.calibration_cutouts, run.withheld_cutouts
    for step in (uf.stack_empirical_psf, uf.collect_noise_samples):
        with pytest.raises(LeakageError, match="calibration"):
            step([*cal[:10], wh[0]])
    with pytest.raises(LeakageError, match="calibration"):
        uf.freeze_calibration([*cal, wh[0]], SIGMA_SYS)  # PSF, covariance and thresholds
    with pytest.raises(LeakageError, match="calibration"):
        uf.freeze_calibration(wh, SIGMA_SYS)
    with pytest.raises(LeakageError, match="withheld"):
        uf.evaluate_withheld(cal[:2], run.frozen)


def test_withheld_fetch_must_follow_the_sealed_freeze(run):
    frozen, wh, cal = run.frozen, run.withheld_cutouts, run.calibration_cutouts
    assert frozen.is_sealed
    at = uf.parse_utc(frozen.frozen_at_utc, "frozen_at_utc")
    assert all(uf.parse_utc(c.fetched_at_utc, "t") <= at for c in cal)
    assert all(uf.parse_utc(c.fetched_at_utc, "t") > at for c in wh)
    one = wh[:1]
    unsealed = dataclasses.replace(frozen, frozen_at_utc=None)
    with pytest.raises(LeakageError, match="not sealed"):
        uf.evaluate_withheld(one, unsealed)
    early = dataclasses.replace(wh[0], fetched_at_utc=cal[0].fetched_at_utc)
    with pytest.raises(LeakageError, match="at or before the freeze"):
        uf.evaluate_withheld([early], frozen)
    same_instant = dataclasses.replace(wh[0], fetched_at_utc=frozen.frozen_at_utc)
    with pytest.raises(LeakageError, match="at or before the freeze"):
        uf.evaluate_withheld([same_instant], frozen)  # strictly later, not equal
    untimed = dataclasses.replace(wh[0], fetched_at_utc=None)
    with pytest.raises(LeakageError, match="cannot prove"):
        uf.evaluate_withheld([untimed], frozen)
    impostor = dataclasses.replace(cal[0], split="withheld", fetched_at_utc=wh[0].fetched_at_utc)
    with pytest.raises(LeakageError, match="overlap the calibration stars"):
        uf.evaluate_withheld([impostor], frozen)
    # A freeze cannot be back-dated before its own inputs, nor sealed twice.
    before_inputs = (uf.parse_utc(cal[0].fetched_at_utc, "t")).replace(year=2020).isoformat()
    with pytest.raises(LeakageError, match="precedes a calibration cutout"):
        unsealed.seal(before_inputs)
    with pytest.raises(LeakageError, match="already sealed"):
        frozen.seal(frozen.frozen_at_utc)
    uf.assert_withheld_after_freeze(T_FREEZE, [T_WH])
    for bad in ([T_CAL], [T_FREEZE], [None], ["yesterday"]):
        with pytest.raises(LeakageError):
            uf.assert_withheld_after_freeze(T_FREEZE, bad)
    with pytest.raises(LeakageError, match="not sealed"):
        uf.assert_withheld_after_freeze(None, [T_WH])


def test_script_stages_cannot_run_out_of_order(script, args, locked):
    small = argparse.Namespace(**{**vars(args), "n_controls": 32})
    archive = script.SyntheticArchive(42, 32, 16)
    clock = script.Clock()
    stamps = [clock() for _ in range(50)]
    assert stamps == sorted(set(stamps))  # strictly increasing: no two stamps are equal
    cache = script.Cache(None)
    sample, _ = script.load_or_build_sample(archive, cache, locked, small, clock, "dirty")
    with pytest.raises(LeakageError, match="after the calibration freeze"):
        script.fetch_split(archive, cache, sample["controls"], "withheld", small, clock, "dirty")
    # A cache that already holds a withheld cutout but no frozen state can never be used.
    star = next(c for c in sample["controls"] if c["split"] == "withheld")
    data, url = archive.cutout(star, 45)
    cache.put_cutout(
        star["gaia_dr3_source_id"],
        data,
        {"split": "withheld", "url": url, "fetched_at_utc": clock(), "code_git_sha": "dirty"},
    )
    with pytest.raises(LeakageError, match="no frozen calibration state"):
        script.run_gate(archive, cache, locked, small, ("dirty", "x"))
    # Cutouts without a sample, a redrawn protocol, and a rewritten sample are all refused.
    orphan = script.Cache(None)
    orphan.manifest["x"] = {"split": "calibration"}
    with pytest.raises(LeakageError, match="no sample"):
        script.load_or_build_sample(archive, orphan, locked, small, clock, "dirty")
    other = argparse.Namespace(**{**vars(small), "conformal_alpha": 0.05})
    with pytest.raises(LeakageError, match="different protocol"):
        script.load_or_build_sample(archive, cache, locked, other, clock, "dirty")
    with pytest.raises(LeakageError, match="immutable"):
        cache.write_once(script.SAMPLE_NAME, {"sample": {}})


@slow
def test_frozen_state_file_is_immutable_and_a_resumed_run_never_refreezes(
    script, args, locked, tmp_path
):
    small = argparse.Namespace(**{**vars(args), "conformal_alpha": 0.1})
    kw = {"n_stars": 32, "separations": SEPS[1:2], "flux_fractions": (0.5,)}
    first = script.rehearse(small, locked, script.Cache(tmp_path), **kw)
    assert first.receipt["status"] == "ok" and not first.receipt["frozen_state_file"]["resumed"]
    names = {p.name for p in tmp_path.iterdir()}
    assert {script.SAMPLE_NAME, script.FROZEN_NAME, script.MANIFEST_NAME} <= names
    assert len([n for n in names if n.endswith("_unwise_w3_cutout.tar.gz")]) == 32
    state_bytes = (tmp_path / script.FROZEN_NAME).read_bytes()
    state = json.loads(state_bytes)
    assert state["frozen_at_utc"] == first.receipt["frozen_at_utc"]
    assert state["frozen_state_sha256"] == first.receipt["frozen_state_sha256"]
    manifest = json.loads((tmp_path / script.MANIFEST_NAME).read_text("utf-8"))
    for name, entry in manifest.items():  # SHA-256 manifest of every cached reply
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == entry["sha256"]
    # Resume: nothing is fetched, nothing is refitted, the freeze time is the original one.
    again = script.rehearse(small, locked, script.Cache(tmp_path), **kw)
    r1, r2 = first.receipt, again.receipt
    assert (
        r2["frozen_state_file"]["resumed"] and r2["split_counts"]["n_cutouts_fetched_this_run"] == 0
    )
    assert all(f["cached"] for f in r2["fetch"]["files"])
    assert (r2["frozen_at_utc"], r2["frozen_state_sha256"]) == (
        r1["frozen_at_utc"],
        r1["frozen_state_sha256"],
    )
    assert r2["withheld"]["statistics"] == r1["withheld"]["statistics"]
    assert r2["frozen_screens"] == r1["frozen_screens"]
    assert (tmp_path / script.FROZEN_NAME).read_bytes() == state_bytes
    # Different settings on the same cache, or an edited state file, are hard refusals.
    changed = argparse.Namespace(**{**vars(small), "conformal_alpha": 0.2})
    with pytest.raises(LeakageError, match="different protocol"):
        script.rehearse(changed, locked, script.Cache(tmp_path), **kw)
    state["frozen"]["frozen_screens"]["screens"]["image_scalar_conformal"]["threshold"] *= 0.5
    (tmp_path / script.FROZEN_NAME).write_text(json.dumps(state), "utf-8")
    with pytest.raises(LeakageError, match="modified after it was written"):
        script.rehearse(small, locked, script.Cache(tmp_path), **kw)
    # A 16 / 16 run can never be promoted, even relabelled as a clean archive run.
    receipt = tmp_path / "small.json"
    receipt.write_text(json.dumps(_as_archive(r1)), "utf-8")
    assert script.promote(receipt, tmp_path / "out.json") == 1
    assert not (tmp_path / "out.json").exists()
    with pytest.raises(ProvenanceError, match="undersized split"):
        UnwiseGateWithheldFile.model_validate(
            script.artifact_doc(_as_archive(r1), {"path": "x", "sha256": "a" * 64})
        )


# ------------------------------------------------------- Arm A / Arm B injection
def test_matched_centered_excess_is_preserved_while_the_naive_rule_rejects_it(run):
    s = run.receipt["withheld"]["summary"]["screens"]
    assert tuple(s) == uf.SCREENS
    naive = s["naive_flux_excess"]
    # "Flag every bright excess" recovers every off-center blend ...
    assert np.all(np.array(naive["arm_a_blend_recovery"]) == 1.0)
    # ... and rejects essentially every genuine matched centered excess.
    assert min(naive["arm_b_false_rejection"]) >= 0.9
    assert naive["centered_excess_retained_pooled"] <= 0.1
    # Screens A and B keep the centered excess while still finding wide, bright blends.
    for name in ("image_scalar_chi2", "image_gls_chi2"):
        assert s[name]["arm_b_false_rejection_pooled"]["rate"] <= 0.1, name
        assert s[name]["centered_excess_retained_pooled"] >= 0.9, name
        assert np.array(s[name]["arm_a_blend_recovery"])[-1, -1] >= 0.8, name
    for scr in s.values():
        r = scr["arm_b_false_rejection_pooled"]
        assert r["clopper_pearson_95"][0] <= r["rate"] <= r["clopper_pearson_95"][1]
        assert r["n"] == 60 * len(FRACS)
    # Arm B is MATCHED: the same flux as Arm A, placed on the star instead of beside it.
    st = run.withheld.stats
    f = np.array(FRACS)
    excess_b = st["arm_b"]["flux_ratio"] - 1.0  # (star, f_bg)
    excess_a = st["arm_a"]["flux_ratio"] - 1.0  # (star, sep, f_bg)
    # The injected flux is F_star f / (1 - f) times the unit-sum PSF; the aperture sees the
    # part of that PSF inside 16.5 arcsec. The stacked wings are noisy (60 faint stars,
    # strongly correlated noise), so that part is near, not exactly, one.
    yy, xx = np.mgrid[-7:8, -7:8]
    inside = float(run.frozen.psf.stamp[xx**2 + yy**2 <= 36].sum())
    assert 0.8 < inside < 1.2
    np.testing.assert_allclose(np.median(excess_b, axis=0), inside * f / (1 - f), rtol=0.03)
    np.testing.assert_allclose(excess_a[:, 0, :], excess_b, rtol=0.02)  # 0.5 arcsec vs centered
    # A centered excess leaves the centroid where it was; an off-center blend moves it.
    off = st["arm_a"]["centroid_offset_arcsec"]
    assert np.median(off[:, -1, -1]) > 3 * np.median(st["arm_b"]["centroid_offset_arcsec"][:, -1])


def test_blend_recovery_rises_with_separation(run):
    n = len(run.withheld.star_ids)
    for name in ("centroid_locked", "centroid_fresh", "image_scalar_chi2", "image_gls_chi2"):
        rec = np.array(run.receipt["withheld"]["summary"]["screens"][name]["arm_a_blend_recovery"])
        assert rec.shape == (len(SEPS), len(FRACS))
        assert np.all(np.diff(rec, axis=0) >= -2.0 / n - 1e-12), name
        assert np.all(rec[-1] > rec[0]), name


def test_withheld_evaluation_is_a_deterministic_seeded_replay(run):
    few = run.withheld_cutouts[:3]
    kw = {"separations": SEPS[1:2], "flux_fractions": (0.5,)}
    a = uf.evaluate_withheld(few, run.frozen, seed=42, **kw)
    b = uf.evaluate_withheld(list(reversed(few)), run.frozen, seed=42, **kw)
    assert uf.canonical_sha256(a.as_dict()) == uf.canonical_sha256(b.as_dict())
    c = uf.evaluate_withheld(few, run.frozen, seed=43, **kw)
    assert not np.allclose(c.azimuth_rad, a.azimuth_rad)
    with pytest.raises(TypeError):
        uf.evaluate_withheld(few, run.frozen, seed=4.2, **kw)
    with pytest.raises(UnitValidationError):
        uf.evaluate_withheld(few, run.frozen, separations=np.array([1.0]))


def test_cutouts_carry_units_string_ids_and_a_usable_window():
    c = _cutouts(1, "calibration", 1)[0]
    with pytest.raises(UnitValidationError):
        dataclasses.replace(c, pixel_scale=2.75)
    with pytest.raises(SchemaValidationError, match=r"v0.2.1-R4"):
        dataclasses.replace(c, gaia_dr3_source_id=1234567890123456789)  # ids are strings
    with pytest.raises(SchemaValidationError):
        dataclasses.replace(c, gaia_dr3_source_id="012")
    with pytest.raises(ValueError, match="ISO-8601"):
        dataclasses.replace(c, fetched_at_utc="noon")
    assert uf.cutout_problem(c) is None
    holed = c.flux.copy()
    holed[22, 22] = np.nan
    assert "non-finite" in uf.cutout_problem(dataclasses.replace(c, flux=holed))
    tiny = dataclasses.replace(c, flux=c.flux[:15, :15], unc=c.unc[:15, :15], gaia_xy=(7.0, 7.0))
    assert "too small" in uf.cutout_problem(tiny)
    invvar = np.where(np.arange(45 * 45).reshape(45, 45) % 7 == 0, 0.0, 4.0)
    iv = uf.UnwiseCutout.from_invvar("7", "withheld", c.flux, invvar, c.gaia_xy, c.pixel_scale, 9.0)
    assert np.isnan(iv.unc[0, 0]) and iv.unc[0, 1] == 0.5  # invvar <= 0 is not a measurement


# ------------------------------------------------------ schema & offline recompute
def test_schema_constants_match_the_kernel():
    assert schemas.UNWISE_SCREENS == uf.SCREENS
    assert schemas.UNWISE_STATISTICS == uf.STATISTICS
    assert uf.UNWISE_PIXEL_SCALE.to_value(u.arcsec) == schemas.UNWISE_PIXEL_SCALE_ARCSEC
    assert set(uf.SCREEN_ROLES) == set(uf.SCREENS)


def test_rehearsal_payload_validates_and_recomputes_offline(doc, locked):
    rec = UnwiseGateWithheldFile.model_validate(doc)
    assert rec.data_tier == "synthetic_proxy" and rec.frozen_before_withheld_fetch is True
    assert rec.disjoint_from_gate1_sha256 == LOCKED["data/calibrations/wise_sigma_sys_control.json"]
    assert (rec.n_calibration, rec.n_withheld) == (60, 60)
    assert len(unwise_gate_bytes(doc)) < schemas.UNWISE_GATE_MAX_BYTES
    detail = verify_unwise_gate(doc, rec, locked.calibration)
    assert "positive definite" in detail and "(k+1)/(n_cal+1) = 1/61" in detail
    assert "0 of 120 fresh stars among the 154 locked controls" in detail
    con = rec.frozen_screens.screens.image_scalar_conformal
    assert con.alpha_guaranteed == (con.k_exceedances + 1) / (con.n_calibration + 1)
    got = uf.recompute_unwise_gate(doc)
    assert got["frozen_state_sha256"] == doc["frozen_state_sha256"]
    assert got["torus_covariance_max_abs_diff"] == 0.0
    assert got["summary"] == doc["withheld"]["summary"]


def test_every_gaia_id_in_the_artifact_is_a_string(doc):
    """Ruling v0.2.1-R4: a JSON number in place of any Gaia DR3 id is a schema error."""
    ids = [
        *[c["gaia_dr3_source_id"] for c in doc["sample"]["controls"]],
        *[f["gaia_dr3_source_id"] for f in doc["fetch"]["files"]],
        *doc["empirical_psf"]["star_ids"],
        *doc["noise_model"]["star_ids"],
        *doc["frozen_screens"]["calibration"]["star_ids"],
        *doc["withheld"]["star_ids"],
    ]
    assert ids and all(isinstance(i, str) and i.isdigit() for i in ids)
    assert all(int(i) > 2**53 for i in ids)  # the reason the ruling exists
    places = (
        lambda d: d["sample"]["controls"][0],
        lambda d: d["fetch"]["files"][0],
    )
    for where in places:
        bad = copy.deepcopy(doc)
        row = where(bad)
        row["gaia_dr3_source_id"] = int(row["gaia_dr3_source_id"])
        with pytest.raises(ValidationError, match="gaia_dr3_source_id"):
            UnwiseGateWithheldFile.model_validate(bad)
    for block, key in (
        ("empirical_psf", "star_ids"),
        ("noise_model", "star_ids"),
        ("withheld", "star_ids"),
    ):
        bad = copy.deepcopy(doc)
        bad[block][key][0] = int(bad[block][key][0])
        with pytest.raises(ValidationError, match="star_ids"):
            UnwiseGateWithheldFile.model_validate(bad)
    bad = copy.deepcopy(doc)
    cal = bad["frozen_screens"]["calibration"]["star_ids"]
    cal[0] = int(cal[0])
    with pytest.raises(ValidationError, match="star_ids"):
        UnwiseGateWithheldFile.model_validate(bad)


def _load(d):
    return UnwiseGateWithheldFile.model_validate(_reseal(d))


def test_schema_enforces_zero_leakage_ordering_and_split_integrity(doc):
    frozen_at = doc["frozen_at_utc"]
    d = copy.deepcopy(doc)  # a withheld cutout fetched before the freeze
    early = d["frozen_screens"]["calibration"]["fetched_at_utc"][0]
    d["withheld"]["fetched_at_utc"][0] = early
    with pytest.raises(LeakageError, match="at or before frozen_at_utc"):
        _load(d)
    d = copy.deepcopy(doc)  # a calibration cutout fetched after the freeze
    d["frozen_screens"]["calibration"]["fetched_at_utc"][0] = d["withheld"]["fetched_at_utc"][0]
    with pytest.raises(LeakageError, match="after frozen_at_utc"):
        _load(d)
    d = copy.deepcopy(doc)  # the flag itself
    d["frozen_before_withheld_fetch"] = False
    with pytest.raises(LeakageError, match="frozen_before_withheld_fetch"):
        _load(d)
    d = copy.deepcopy(doc)  # a withheld star behind the PSF
    d["empirical_psf"]["star_ids"][0] = d["withheld"]["star_ids"][0]
    with pytest.raises(LeakageError, match="PSF"):
        _load(d)
    d = copy.deepcopy(doc)  # ... or behind the covariance
    d["noise_model"]["star_ids"][0] = d["withheld"]["star_ids"][0]
    with pytest.raises(LeakageError, match="covariance"):
        _load(d)
    d = copy.deepcopy(doc)  # a star in both halves
    d["withheld"]["star_ids"][0] = d["frozen_screens"]["calibration"]["star_ids"][0]
    with pytest.raises(LeakageError, match="overlap"):
        _load(d)
    d = copy.deepcopy(doc)  # a withheld-labelled control evaluated as calibration
    sid = d["withheld"]["star_ids"][0]
    next(c for c in d["sample"]["controls"] if c["gaia_dr3_source_id"] == sid)["split"] = (
        "calibration"
    )
    with pytest.raises((LeakageError, SchemaValidationError)):
        _load(d)
    d = copy.deepcopy(doc)  # a patch recorded 14 deg from a locked Gate #1 patch
    d["sample"]["patches"][0]["nearest_locked_patch_separation_deg"] = 14.0
    with pytest.raises(LeakageError, match="locked Gate #1 patch"):
        _load(d)
    d = copy.deepcopy(doc)
    d["frozen_at_utc"] = "2031-01-01T00:00:00+00:00"
    with pytest.raises((LeakageError, ProvenanceError)):
        _load(d)
    assert frozen_at == doc["frozen_at_utc"]  # the fixture itself is untouched


def test_schema_pins_the_conformal_finite_sample_level(doc):
    for field, value, message in (
        ("alpha_guaranteed", 0.001, r"\(k \+ 1\) / \(n_cal \+ 1\)"),
        ("k_exceedances", 1, "rank_from_top"),
        ("n_calibration", 59, r"\(k \+ 1\) / \(n_cal \+ 1\)"),
    ):
        d = copy.deepcopy(doc)
        d["frozen_screens"]["screens"]["image_scalar_conformal"][field] = value
        with pytest.raises((ProvenanceError, SchemaValidationError), match=message):
            _load(d)
    d = copy.deepcopy(doc)  # a looser order statistic than the nominal alpha allows
    sc = d["frozen_screens"]["screens"]["image_gls_conformal"]
    sc.update(k_exceedances=1, order_statistic_rank_from_top=2, alpha_guaranteed=2 / 61)
    with pytest.raises(ProvenanceError, match="exceeds the nominal alpha"):
        _load(d)
    d = copy.deepcopy(doc)  # covariance below its positive-definiteness bound
    d["noise_model"]["minimum_eigenvalue_bound"] = 0.0
    with pytest.raises(ProvenanceError, match="strictly positive definite"):
        _load(d)
    d = copy.deepcopy(doc)
    del d["frozen_screens"]["screens"]["image_gls_chi2"]
    with pytest.raises(ValidationError, match="image_gls_chi2"):
        _load(d)
    d = copy.deepcopy(doc)
    d["withheld"]["statistics"]["arm_b"]["delta_chi2_gls"] = [[0.0]]
    with pytest.raises(SchemaValidationError, match="not shaped"):
        _load(d)
    d = copy.deepcopy(doc)
    d["unexpected_field"] = 1  # strict: unknown keys are refused
    with pytest.raises(ValidationError, match="unexpected_field"):
        _load(d)


def test_offline_recompute_catches_resealed_tampering(doc, locked):
    def verify(d):
        d = _reseal(d)
        return verify_unwise_gate(d, UnwiseGateWithheldFile.model_validate(d), locked.calibration)

    verify(copy.deepcopy(doc))
    d = copy.deepcopy(doc)  # a centered excess that now "fails" Screen B
    d["withheld"]["statistics"]["arm_b"]["delta_chi2_gls"][0][0] = 1e9
    with pytest.raises(ProvenanceError, match=r"recomputed|!="):
        verify(d)
    d = copy.deepcopy(doc)  # a convenient threshold (the frozen fingerprint no longer matches)
    d["frozen_screens"]["screens"]["image_scalar_chi2"]["threshold"] *= 0.5
    with pytest.raises(ProvenanceError, match="frozen_state_sha256"):
        verify(d)
    d = copy.deepcopy(doc)  # ... even with the fingerprint recomputed to match
    d["frozen_screens"]["screens"]["image_scalar_chi2"]["threshold"] *= 0.5
    d["frozen_state_sha256"] = uf.frozen_state_sha256(d)
    with pytest.raises(ProvenanceError, match="threshold"):
        verify(d)
    d = copy.deepcopy(doc)  # a calibration null removed from the top of the distribution
    loo = d["frozen_screens"]["calibration"]["delta_chi2_scalar_loo"]
    loo[int(np.argmax(loo))] = 0.0
    d["frozen_state_sha256"] = uf.frozen_state_sha256(d)
    with pytest.raises(ProvenanceError):
        verify(d)
    d = copy.deepcopy(doc)  # a covariance that is not the regularised raw estimate
    d["noise_model"]["torus_covariance"][0][1] *= 1.5
    d["noise_model"]["torus_covariance"][0][31] *= 1.5
    d["frozen_state_sha256"] = uf.frozen_state_sha256(d)
    with pytest.raises(ProvenanceError, match="regularised"):
        verify(d)
    d = copy.deepcopy(doc)  # a different PSF
    d["empirical_psf"]["fwhm_gaussian_arcsec"] = 13.69
    d["frozen_state_sha256"] = uf.frozen_state_sha256(d)
    with pytest.raises(ProvenanceError, match="fwhm_gaussian"):
        verify(d)
    # A locked Gate #1 star hidden in the fresh sample (renamed consistently everywhere).
    fresh = doc["withheld"]["star_ids"][0]
    stolen = str(locked.star_ids[0])
    d = json.loads(json.dumps(doc).replace(f'"{fresh}"', f'"{stolen}"'))
    d["frozen_state_sha256"] = uf.frozen_state_sha256(d)
    with pytest.raises(LeakageError, match="locked Gate #1"):
        verify(d)
    d = copy.deepcopy(doc)  # a patch moved on top of a locked patch, its record left alone
    d["sample"]["patches"][0].update(
        l_deg=float(locked.centres[0].l.deg), b_deg=float(locked.centres[0].b.deg) + 12.0
    )
    with pytest.raises((LeakageError, SchemaValidationError)):
        verify(d)
    d = copy.deepcopy(doc)  # checked against some other Gate #1 file
    d["locked_inputs"]["sigma_sys_calibration_sha256"] = "f" * 64
    d["disjoint_from_gate1_sha256"] = "f" * 64
    with pytest.raises(ProvenanceError, match="different sigma_sys calibration"):
        verify(d)
    d = copy.deepcopy(doc)
    d["disjoint_from_gate1_sha256"] = "f" * 64
    with pytest.raises(ProvenanceError, match="disjoint_from_gate1_sha256"):
        verify(d)


# ------------------------------------------------------------ validate & survey pin
def test_validate_reports_not_promoted_yet_while_unpinned():
    graph = load_factor_graph(ROOT)
    assert graph.surveys[IR].infrared.unwise_gate_path is None  # v0.2.2a pins nothing
    assert not (ROOT / UNWISE_GATE_PATH).exists()
    assert load_unwise_gate(graph.surveys[IR], ROOT) is None
    res = {r.name: r for r in run_validation(ROOT)}
    row = res["unwise_gate_promotion"]
    assert row.ok and "not promoted yet" in row.detail and "v0.2.2b" in row.detail
    assert all(r.ok for r in res.values()), [r for r in res.values() if not r.ok]
    assert len(res) == 42


def test_survey_pin_fields_are_optional_and_checked(data_copy):
    path = data_copy / "data/surveys" / f"{IR}.yaml"
    doc = yaml.safe_load(path.read_text("utf-8"))

    def write(**infrared):
        new = copy.deepcopy(doc)
        new["infrared"].update(infrared)
        path.write_text(yaml.safe_dump(new, sort_keys=False), "utf-8")

    write(unwise_gate_path=UNWISE_GATE_PATH)  # a path without its SHA-256
    with pytest.raises(ProvenanceError, match="given together"):
        load_factor_graph(data_copy)
    write(unwise_gate_path=UNWISE_GATE_PATH, unwise_gate_sha256="xyz")
    with pytest.raises(ProvenanceError, match="64-hex"):
        load_factor_graph(data_copy)
    new = copy.deepcopy(doc)
    new["infrared"].update(unwise_gate_path=UNWISE_GATE_PATH, unwise_gate_sha256="a" * 64)
    for k in ("sigma_sys_calibration_path", "sigma_sys_calibration_sha256"):
        del new["infrared"][k]
    for k in ("fits_gate2_path", "fits_gate2_sha256"):
        del new["infrared"][k]
    path.write_text(yaml.safe_dump(new, sort_keys=False), "utf-8")
    with pytest.raises(ProvenanceError, match="locked sigma_sys calibration"):
        load_factor_graph(data_copy)


@pytest.fixture(scope="module")
def promoted(script, run, tmp_path_factory):
    """The rehearsal receipt, relabelled as a clean archive run, promoted into tmp (offline)."""
    tmp = tmp_path_factory.mktemp("promoted")
    receipt = tmp / "receipt.json"
    receipt.write_text(json.dumps(_as_archive(run.receipt)), "utf-8")
    out = tmp / "unwise_w3_gate_withheld.json"
    assert script.promote(receipt, out) == 0
    return receipt, out


def _pin(root, artifact_bytes, sha=None):
    """Drop an artifact into a data copy and pin it in the IR survey's `infrared:` block."""
    (root / UNWISE_GATE_PATH).write_bytes(artifact_bytes)
    path = root / "data/surveys" / f"{IR}.yaml"
    doc = yaml.safe_load(path.read_text("utf-8"))
    doc["infrared"]["unwise_gate_path"] = UNWISE_GATE_PATH
    doc["infrared"]["unwise_gate_sha256"] = sha or hashlib.sha256(artifact_bytes).hexdigest()
    path.write_text(yaml.safe_dump(doc, sort_keys=False), "utf-8")


def test_promote_writes_a_sealed_artifact_and_prints_the_pin(script, promoted, capsys, tmp_path):
    receipt, out = promoted
    raw = out.read_bytes()
    doc = json.loads(raw)
    rec = UnwiseGateWithheldFile.model_validate(doc)
    assert rec.data_tier == "reproduced_archive" and rec.id == "unwise_w3_gate_withheld"
    assert rec.source_receipt.sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert _reseal(doc)["content_sha256"] == doc["content_sha256"]
    assert raw == unwise_gate_bytes(doc) and len(raw) < schemas.UNWISE_GATE_MAX_BYTES
    target = tmp_path / "again.json"
    assert script.main(["--promote-from", str(receipt), "--out-file", str(target)]) == 0
    printed = capsys.readouterr().out
    assert target.read_bytes() == raw  # deterministic bytes
    sha = hashlib.sha256(raw).hexdigest()
    assert f'unwise_gate_path: "{UNWISE_GATE_PATH}"' in printed
    assert f'unwise_gate_sha256: "{sha}"' in printed and "infrared" in printed
    assert not (ROOT / UNWISE_GATE_PATH).exists()  # the project artifact path is never touched


def test_validate_enforces_everything_once_the_artifact_is_pinned(promoted, data_copy):
    raw = promoted[1].read_bytes()
    # Present at the canonical path but not pinned: validate must fail.
    (data_copy / UNWISE_GATE_PATH).write_bytes(raw)
    row = {r.name: r for r in run_validation(data_copy)}["unwise_gate_promotion"]
    assert not row.ok and "no survey pins" in row.detail
    _pin(data_copy, raw)
    res = {r.name: r for r in run_validation(data_copy)}
    row = res["unwise_gate_promotion"]
    assert row.ok, row.detail
    for needle in ("pinned by", "reproduced_archive", "recomputed", "positive definite"):
        assert needle in row.detail
    assert "before every withheld fetch" in row.detail and "0 of 120 fresh stars" in row.detail
    assert all(r.ok for r in res.values()), [r for r in res.values() if not r.ok]
    gate = load_unwise_gate(load_factor_graph(data_copy).surveys[IR], data_copy)
    s = gate.summary()
    assert s["frozen_before_withheld_fetch"] and set(s["screens"]) == set(uf.SCREENS)
    assert (s["n_calibration"], s["n_withheld"]) == (60, 60)
    # The four locked artifacts in the copy are byte-identical to the pinned hashes.
    for rel, sha in LOCKED.items():
        assert hashlib.sha256((data_copy / rel).read_bytes()).hexdigest() == sha


def test_pinned_artifact_tampering_is_caught_by_validate(promoted, data_copy, doc):
    raw = promoted[1].read_bytes()
    good = json.loads(raw)

    def detail(artifact_bytes, sha=None):
        _pin(data_copy, artifact_bytes, sha)
        row = {r.name: r for r in run_validation(data_copy)}["unwise_gate_promotion"]
        assert not row.ok
        return row.detail

    assert "pinned" in detail(raw + b" ", hashlib.sha256(raw).hexdigest())  # bytes vs pin
    edited = {**good, "description": good["description"] + "!"}
    assert "content_sha256" in detail(unwise_gate_bytes(edited))  # re-pinned, not re-sealed
    moved = copy.deepcopy(good)  # re-pinned AND re-sealed: the freeze ordering still fails
    moved["withheld"]["fetched_at_utc"][0] = moved["frozen_screens"]["calibration"][
        "fetched_at_utc"
    ][0]
    assert "LeakageError" in detail(unwise_gate_bytes(_reseal(moved)))
    faked = copy.deepcopy(good)  # ... and so does a changed measurement
    faked["withheld"]["statistics"]["null"]["delta_chi2_scalar"][0] = 1e9
    assert "ProvenanceError" in detail(unwise_gate_bytes(_reseal(faked)))
    assert "can never be pinned" in detail(unwise_gate_bytes(doc))  # a synthetic_proxy payload
    dirty = {**good, "code_git_sha": "dirty"}
    assert "clean" in detail(unwise_gate_bytes(_reseal(dirty)))
    big = {**good, "caveats": [*good["caveats"], "x" * schemas.UNWISE_GATE_MAX_BYTES]}
    assert "not < 5 MB" in detail(unwise_gate_bytes(_reseal(big)))


# -------------------------------------------------------------- promote refusals
def _mutations(locked_id):
    def dirty(r):
        r["code_git_sha"] = "dirty"

    def dirty_cutout(r):
        r["fetch"]["files"][3]["code_git_sha"] = "dirty"

    def synthetic(r):
        r["data_tier"] = "synthetic_proxy"

    def undersized(r):
        r["split_counts"]["n_withheld_evaluated"] = 59

    def leaked_flag(r):
        r["frozen_before_withheld_fetch"] = False

    def leaked_timestamp(r):
        sid = r["withheld"]["star_ids"][0]
        early = r["frozen_screens"]["calibration"]["fetched_at_utc"][0]
        r["withheld"]["fetched_at_utc"][0] = early
        next(f for f in r["fetch"]["files"] if f["gaia_dr3_source_id"] == sid)["fetched_at_utc"] = (
            early
        )

    def leaked_star(r):
        text = json.dumps(r).replace(f'"{r["withheld"]["star_ids"][0]}"', f'"{locked_id}"')
        r.clear()
        r.update(json.loads(text))

    def incomplete(r):
        r["status"] = "incomplete"

    def oversized(r):
        r["caveats"] = [*r["caveats"], "x" * schemas.UNWISE_GATE_MAX_BYTES]

    def unlocked(r):
        r["locked_artifacts_verified"] = {}

    return {
        "dirty git_sha": (dirty, "clean commit"),
        "cutout fetched from a dirty tree": (dirty_cutout, "clean commit"),
        "synthetic payload": (synthetic, "synthetic rehearsal payload can never be promoted"),
        "undersized split": (undersized, "undersized split: 60 calibration / 59 withheld"),
        "freeze flag false": (leaked_flag, "leakage"),
        "withheld fetched before the freeze": (leaked_timestamp, "leakage violation: LeakageError"),
        "locked Gate #1 star in the sample": (leaked_star, "leakage violation: LeakageError"),
        "incomplete run": (incomplete, "receipt status 'incomplete'"),
        "oversized artifact": (oversized, "oversized output"),
        "locked artifacts not verified": (unlocked, "four locked artifacts"),
    }


@pytest.mark.parametrize("case", list(_mutations("1")))
def test_promote_refuses_dirty_synthetic_undersized_or_leaked_receipts(
    case, script, run, locked, tmp_path, capsys
):
    mutate, message = _mutations(str(locked.star_ids[0]))[case]
    rec = _as_archive(run.receipt)
    mutate(rec)
    path, out = tmp_path / "receipt.json", tmp_path / "promoted.json"
    path.write_text(json.dumps(rec), "utf-8")
    assert script.main(["--promote-from", str(path), "--out-file", str(out)]) == 1
    err = capsys.readouterr().err
    assert "Refusing to promote" in err and message in err, err
    assert not out.exists() and not (ROOT / UNWISE_GATE_PATH).exists()


def test_the_untouched_rehearsal_receipt_is_itself_unpromotable(script, run, tmp_path, capsys):
    path, out = tmp_path / "receipt.json", tmp_path / "promoted.json"
    path.write_text(json.dumps(run.receipt), "utf-8")  # data_tier synthetic_proxy, as written
    assert script.promote(path, out) == 1
    assert "synthetic" in capsys.readouterr().err and not out.exists()
    assert any("synthetic" in p for p in script.promotion_problems(run.receipt))


# ------------------------------------------------------- script: bytes, network, CLI
def test_cutout_reply_parser_keeps_native_pixels_and_refuses_everything_else(script):
    rng = np.random.default_rng(9)
    flux = rng.normal(100.0, 5.0, (45, 45))
    invvar = np.full((45, 45), 0.04)
    control = {
        "gaia_ra_epoch_wise_deg": 211.25,
        "gaia_dec_epoch_wise_deg": 41.5,
        "coadd_id": "2112p408_ac51",
    }
    blob = script.pack_cutout(flux, invvar, (22.31, 21.77), 211.25, 41.5, "2112p408")
    assert blob == script.pack_cutout(flux, invvar, (22.31, 21.77), 211.25, 41.5, "2112p408")
    got = script.parse_cutout(blob, control, 10.0)
    assert got.tile == "2112p408" and got.shape == (45, 45)
    assert got.gaia_xy == pytest.approx((22.31, 21.77), abs=1e-6)
    assert got.pixel_scale_measured_arcsec == pytest.approx(2.75, abs=1e-9)
    np.testing.assert_allclose(got.flux, flux, rtol=1e-6)
    np.testing.assert_allclose(got.invvar, invvar, rtol=1e-6)
    with pytest.raises(script.LiveFetchError, match="margin"):
        script.parse_cutout(blob, control, 30.0)  # the star is too close to the edge
    resampled = script.pack_cutout(
        flux, invvar, (22.3, 21.8), 211.25, 41.5, "2112p408", pixel_scale=1.375 * u.arcsec
    )
    with pytest.raises(script.LiveFetchError, match="native-pixel"):
        script.parse_cutout(resampled, control, 10.0)  # not the native 2.75 arcsec grid
    for junk in (b"<html>Service unavailable</html>", b"", flux.tobytes()[:400]):
        with pytest.raises(script.LiveFetchError):
            script.parse_cutout(junk, control, 10.0)
    assert script.cutout_url(211.25, 41.5, 45) == (
        "https://unwise.me/cutout_fits?version=allwise&ra=211.2500000&dec=41.5000000"
        "&size=45&bands=3&file_img_m=on&file_invvar_m=on"
    )
    assert script.ENDPOINTS["unwise_cutout"]["verified_live"] is False


def test_no_network_outside_fetch_and_no_fetch_from_a_dirty_tree(script, monkeypatch, capsys):
    with pytest.raises(script.LiveFetchError, match="outside --fetch"):
        script._http("https://unwise.me/cutout_fits?version=allwise")
    monkeypatch.setattr(script, "_NETWORK_ALLOWED", True)
    with pytest.raises(script.LiveFetchError, match="non-allowlisted"):
        script._http("https://example.org/cutout_fits")  # refused before any socket is opened
    monkeypatch.setattr(script, "_NETWORK_ALLOWED", False)
    monkeypatch.setattr(script, "git_state", lambda root: ("dirty", "abc"))
    assert script.main(["--fetch"]) == 2
    assert "Refusing a live fetch from a dirty git tree" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script.main([])  # a mode is required
    with pytest.raises(SystemExit):
        script.main(["--dry-run", "--fetch"])  # and only one
    assert set(script.LOCKED_ARTIFACTS.items()) == set(LOCKED.items())


def test_live_catalog_code_path_parses_canned_tap_replies_offline(script, args, monkeypatch):
    """The --fetch catalog stage on canned CSV replies: no socket is ever opened."""
    big = "6138023116601198720"  # > 2^53: must survive the CSV round trip as text
    gaia_csv = "\n".join(
        [
            "source_id,ra,dec,l,b,pmra,pmdec,ref_epoch,parallax,parallax_over_error,bp_rp,"
            "phot_g_mean_mag,ruwe,original_ext_source_id",
            f"{big},200.5,41.25,100.0,45.5,-120.0,60.0,2016.0,8.5,90.0,2.4,13.9,1.02,"
            "J132200.00+411500.0",
            "4611686018427387904,200.6,41.30,100.1,45.6,,,2016.0,5.5,40.0,2.0,14.5,1.10,"
            "J132224.00+411800.0",
            "",
        ]
    )
    wise_csv = "\n".join(
        [
            "designation,ra,dec,w3snr,w3mpro,w3sigmpro,cc_flags,ext_flg,nb,na,ph_qual,coadd_id",
            "J132200.00+411500.0,200.5001,41.2501,11.5,10.9,0.09,0000,0,1,0,AAAB,2006p408_ac51",
            "J999999.99+999999.9,10.0,10.0,9.0,11.0,0.1,0000,0,1,0,AAAB,0100p106_ac51",
            "",
        ]
    )
    neighbours_csv = "\n".join(["source_id,phot_g_mean_mag", f"{big},13.9", "77,19.5", ""])
    sent = []

    def fake_http(url, data=None, timeout=300, tries=3):
        sent.append((url, data))
        if url.startswith(script.UNWISE_CUTOUT):
            return b"tar-bytes"
        query = data.decode()
        if url == script.IRSA_TAP:
            return wise_csv.encode()
        return (neighbours_csv if "phot_g_mean_mag+FROM" in query else gaia_csv).encode()

    monkeypatch.setattr(script, "_http", fake_http)
    archive = script.LiveArchive(0.0, script.Clock())
    patch = {"patch_id": "U01", "ra_deg": 200.5, "dec_deg": 41.25, "radius_deg": 2.0}
    rows, meta = archive.candidates(patch, args)
    assert meta == {"gaia_m_dwarfs_in_cone": 2, "allwise_w3_band_rows_in_cone": 2}
    assert len(rows) == 1  # inner join on the AllWISE designation
    row = rows[0]
    assert row["gaia_dr3_source_id"] == big and row["allwise"] == "J132200.00+411500.0"
    assert (row["cc_flags"], row["ext_flg"], row["coadd_id"]) == ("0000", 0, "2006p408_ac51")
    assert all(uf.ControlCuts().passes({**row, "n_gaia_neighbours": 0}))
    assert archive.n_neighbours(row, 15.0 * u.arcsec) == 1  # one OTHER Gaia source in the cone
    ra, dec = script.propagate(row)  # back to epoch 2010.5 with the proper motion
    assert dec == pytest.approx(41.25 - 60.0 * 5.5 / 3.6e6)
    assert ra > 200.5  # negative pmra, propagated backwards
    control = {"gaia_ra_epoch_wise_deg": ra, "gaia_dec_epoch_wise_deg": dec}
    assert archive.cutout(control, 45) == (b"tar-bytes", script.cutout_url(ra, dec, 45))
    assert [q["role"] for q in archive.log] == ["gaia_m_dwarfs", "allwise_w3_band", "isolation"]
    assert sent[0][0] == script.ARI_TAP and b"allwise_best_neighbour" in sent[0][1]
    assert sent[1][0] == script.IRSA_TAP and b"coadd_id" in sent[1][1]
    assert all(u_.startswith(script.ALLOWED_URL_PREFIXES) for u_, _ in sent)


def test_locked_artifact_drift_is_a_provenance_error(script, data_copy):
    script.load_locked(data_copy)  # the copy is pristine
    target = data_copy / "data/catalogs/bl_gbt_lband_background_summary.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ProvenanceError, match="locked artifact"):
        script.load_locked(data_copy)


@slow
def test_dry_run_cli_rehearses_end_to_end_offline_and_writes_nothing(script, capsys):
    before = {p: p.stat().st_mtime_ns for p in (ROOT / "data").rglob("*") if p.is_file()}
    runs = (ROOT / "runs" / "unwise_w3_gate").exists()
    cache = (ROOT / "data/cache/unwise_w3").exists()
    code = script.main(["--dry-run", "--rehearsal-stars", "48", "--conformal-alpha", "0.05"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "DRY RUN OK: zero outbound network requests, no files written." in out
    for sha in LOCKED.values():
        assert sha in out
    assert "disjoint from all 12 Gate #1 patches" in out
    assert "https://unwise.me/cutout_fits?version=allwise" in out and "verified live: False" in out
    assert "gaiadr3.allwise_best_neighbour" in out and "allwise_p3as_psd" in out
    assert "synthetic_proxy" in out and "(k+1)/(n_cal+1) = 1/25" in out
    assert out.count("LeakageError guard") == 10 and "NOT RAISED" not in out
    for name in uf.SCREENS:
        assert f"[{name}]" in out
    assert "--promote-from would refuse this payload" in out
    after = {p: p.stat().st_mtime_ns for p in (ROOT / "data").rglob("*") if p.is_file()}
    assert after == before  # no cache, no artifact, no locked file touched
    assert (ROOT / "runs" / "unwise_w3_gate").exists() == runs
    assert (ROOT / "data/cache/unwise_w3").exists() == cache
