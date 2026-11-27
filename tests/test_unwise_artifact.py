"""Promoted unWISE W3 fresh-split gate (reproduced_archive, v0.2.2b) - offline checks only.

The unWISE FITS cutouts are never read here (they live in the gitignored
``data/cache/unwise_w3/``); only the committed summary file in ``data/calibrations/``, its
pin in the IR survey YAML and the locked Gate #1 file it must be disjoint from. Negative
paths run on ``data_copy`` trees; nothing is written inside the project tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

import pytest
import yaml

from conftest import ROOT
from fermi_desk.errors import FermiDeskError, LeakageError, ProvenanceError
from fermi_desk.models import unwise_fits as uf
from fermi_desk.registry import (
    load_factor_graph,
    load_sigma_sys_calibration,
    load_unwise_gate,
    run_validation,
    seal_unwise_gate,
    unwise_gate_bytes,
)
from fermi_desk.schemas import (
    UNWISE_GATE_MAX_BYTES,
    UNWISE_GATE_MIN_ABS_B_DEG,
    UNWISE_GATE_PATH,
)

IR = "hephaistos_wise_mid_ir"
SURVEY_YAML = "data/surveys/hephaistos_wise_mid_ir.yaml"
# The four artifacts locked before v0.2.2: byte-identical after the unWISE promotion.
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
PINNED_SHA = "b49e997054726bcfebd5b7ba5ca255a4071c3627af9ce7f871eee2799302f238"
IMAGE_SCREENS = (*uf.CHI2_SCREENS, *uf.CONFORMAL_SCREENS)


@pytest.fixture(scope="module")
def gate(graph):
    g = load_unwise_gate(graph.surveys[IR], graph.root)
    assert g is not None, "IR survey must pin the promoted unWISE W3 gate"
    return g


@pytest.fixture(scope="module")
def raw():
    return (ROOT / UNWISE_GATE_PATH).read_bytes()


def _rewrite(data_copy, mutate, *, repin=True, reseal=True):
    """Edit the artifact in a data copy; returns a loader for the edited copy."""
    path = data_copy / UNWISE_GATE_PATH
    doc = json.loads(path.read_bytes())
    mutate(doc)
    if reseal:
        doc = seal_unwise_gate(doc)
    out = unwise_gate_bytes(doc)
    path.write_bytes(out)
    if repin:
        sv_path = data_copy / SURVEY_YAML
        sv = yaml.safe_load(sv_path.read_text("utf-8"))
        sv["infrared"]["unwise_gate_sha256"] = hashlib.sha256(out).hexdigest()
        sv_path.write_text(yaml.safe_dump(sv, sort_keys=False), "utf-8")
    g = load_factor_graph(data_copy)
    return lambda: load_unwise_gate(g.surveys[IR], g.root)


# ------------------------------------------------------------------ pin & bytes
def test_all_four_pre_v022_locked_artifacts_are_byte_identical():
    for rel, sha in LOCKED.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, rel


def test_artifact_matches_its_pin_and_stays_under_5_mb(graph, gate, raw):
    ir = graph.surveys[IR].infrared
    assert ir.unwise_gate_path == UNWISE_GATE_PATH
    assert ir.unwise_gate_sha256 == PINNED_SHA == gate.file_sha256
    assert hashlib.sha256(raw).hexdigest() == PINNED_SHA
    assert len(raw) < UNWISE_GATE_MAX_BYTES == 5 * 1024 * 1024
    doc = json.loads(raw)
    assert raw == unwise_gate_bytes(doc)  # canonical serialisation, deterministic bytes
    assert seal_unwise_gate(doc)["content_sha256"] == doc["content_sha256"]
    # The pin sits inside the `infrared:` block, next to the Gate #2 pin.
    text = (ROOT / SURVEY_YAML).read_text("utf-8")
    block = text[text.index("\ninfrared:") : text.index("\ncitations:")]
    assert block.index("fits_gate2_sha256") < block.index("unwise_gate_path")
    assert f'unwise_gate_sha256: "{PINNED_SHA}"' in block


def test_artifact_is_a_clean_reproduced_archive_run(gate):
    rec = gate.record
    assert rec.data_tier == "reproduced_archive" and rec.rule_version == uf.RULE_VERSION
    assert re.fullmatch(r"[0-9a-f]{40}", rec.code_git_sha)  # clean commit, never "dirty"
    assert {f.code_git_sha for f in rec.fetch.files} == {rec.code_git_sha}
    assert rec.source_receipt is not None and len(rec.source_receipt.sha256) == 64
    assert rec.fetch.n_ok == rec.fetch.n_requested == len(rec.sample.controls)
    assert rec.fetch.n_failed == 0 and rec.rng_seed == 42
    assert rec.empirical_psf.pixel_scale_arcsec == 2.75  # native unWISE pixels, never resampled
    assert all(
        f.pixel_scale_measured_arcsec == pytest.approx(2.75, abs=1e-3) for f in rec.fetch.files
    )
    assert all(
        f.url.startswith("https://unwise.me/cutout_fits?version=allwise") for f in rec.fetch.files
    )


def test_every_gaia_id_in_the_artifact_is_a_string(raw):
    """Ruling v0.2.1-R4: no Gaia DR3 id is ever a JSON number (they exceed 2^53)."""
    doc = json.loads(raw)
    ids = [c["gaia_dr3_source_id"] for c in doc["sample"]["controls"]]
    ids += [f["gaia_dr3_source_id"] for f in doc["fetch"]["files"]]
    ids += doc["frozen_screens"]["calibration"]["star_ids"] + doc["withheld"]["star_ids"]
    ids += doc["empirical_psf"]["star_ids"] + doc["noise_model"]["star_ids"]
    assert ids and all(isinstance(i, str) and i.isdigit() for i in ids)
    assert any(int(i) > 2**53 for i in ids)
    assert re.search(rb'"gaia_dr3_source_id":\s*[0-9]', raw) is None


# ------------------------------------------------------------- sample & leakage
def test_r3_sample_size_and_exact_conformal_level(gate):
    """Ruling v0.2.2-R3: N >= 200 with >= 100 stars in each half, conformal level <= 1%."""
    rec = gate.record
    fz = rec.frozen_screens
    assert len(rec.sample.controls) >= 200 and len(rec.sample.patches) == 16
    assert rec.n_calibration >= 100 and rec.n_withheld >= 100
    assert fz.conformal_alpha_requested == 0.01 and fz.conformal_alpha_fallback is False
    assert fz.conformal_alpha_nominal == 0.01  # no fallback was needed: n_cal >= 99
    for name in uf.CONFORMAL_SCREENS:
        sc = getattr(fz.screens, name)
        assert sc.k_exceedances == 0 and sc.n_calibration == rec.n_calibration
        assert sc.alpha_guaranteed == 1 / (rec.n_calibration + 1) <= 0.01
    assert all(abs(c.b_deg) > UNWISE_GATE_MIN_ABS_B_DEG for c in rec.sample.controls)
    assert all(abs(p.b_deg) - p.radius_deg >= UNWISE_GATE_MIN_ABS_B_DEG for p in rec.sample.patches)


def test_zero_gaia_id_overlap_with_the_locked_gate1_sample(graph, gate):
    locked = load_sigma_sys_calibration(graph.surveys[IR], graph.root)
    assert locked.file_sha256 == LOCKED["data/calibrations/wise_sigma_sys_control.json"]
    assert gate.record.disjoint_from_gate1_sha256 == locked.file_sha256
    locked_ids = {str(c.gaia_dr3_source_id) for c in locked.record.controls}
    fresh_ids = {c.gaia_dr3_source_id for c in gate.record.sample.controls}
    assert len(locked_ids) == 154 and len(fresh_ids) == len(gate.record.sample.controls)
    assert fresh_ids & locked_ids == set()
    assert "0 of 252 fresh stars among the 154 locked controls" in gate.detail
    assert all(p.nearest_locked_patch_separation_deg >= 15.0 for p in gate.record.sample.patches)


def test_freeze_precedes_every_withheld_fetch(gate):
    rec = gate.record
    when = datetime.fromisoformat
    frozen = when(rec.frozen_at_utc)
    cal = [when(t) for t in rec.frozen_screens.calibration.fetched_at_utc]
    wh = [when(t) for t in rec.withheld.fetched_at_utc]
    assert max(cal) <= frozen < min(wh)  # frozen_at_utc < earliest_withheld_fetch_utc
    assert rec.frozen_before_withheld_fetch is True
    start, end = (when(t) for t in rec.query_window_utc)
    assert start <= min(cal) and max(wh) <= end
    # Every calibration / withheld star is a control of its own split, and no star is in both.
    split = {c.gaia_dr3_source_id: c.split for c in rec.sample.controls}
    cal_ids, wh_ids = set(rec.frozen_screens.calibration.star_ids), set(rec.withheld.star_ids)
    assert {split[s] for s in cal_ids} == {"calibration"} and {split[s] for s in wh_ids} == {
        "withheld"
    }
    assert not cal_ids & wh_ids
    assert set(rec.empirical_psf.star_ids) <= cal_ids and set(rec.noise_model.star_ids) <= cal_ids
    s = gate.summary()
    assert s["latest_calibration_fetch_utc"] < s["frozen_at_utc"] < s["earliest_withheld_fetch_utc"]


# ---------------------------------------------------------- offline recomputation
def test_all_screens_and_both_centroid_floors_recompute_offline(graph, gate, raw):
    doc = json.loads(raw)
    got = uf.recompute_unwise_gate(doc)
    rec = gate.record
    assert got["frozen_state_sha256"] == rec.frozen_state_sha256
    assert got["summary"] == doc["withheld"]["summary"]
    assert got["screens"] == doc["frozen_screens"]["screens"]
    assert got["torus_covariance_max_abs_diff"] <= 1e-9 * rec.noise_model.variance_inflation
    # Positive-definite pixel covariance: eigenvalues >= shrinkage * V(0) > 0.
    assert got["minimum_eigenvalue_bound"] >= got["minimum_eigenvalue_required"] * (1 - 1e-9) > 0
    assert got["fwhm_gaussian_arcsec"] == pytest.approx(
        rec.empirical_psf.fwhm_gaussian_arcsec, abs=1e-6
    )
    summary = got["summary"]["screens"]
    assert set(summary) == set(uf.SCREENS)
    for name in (*IMAGE_SCREENS, *uf.CENTROID_SCREENS):
        sc = summary[name]
        assert sc["n_stars"] == rec.n_withheld == sc["null_false_alarm"]["n"]
        lo, hi = sc["null_false_alarm"]["clopper_pearson_95"]
        assert lo <= sc["null_false_alarm"]["rate"] <= hi
        assert 0.0 <= sc["blend_survival_within_3p25_arcsec"] <= 1.0
        assert sc["arm_b_false_rejection_pooled"]["n"] == rec.n_withheld * len(
            doc["withheld"]["flux_fractions"]
        )
    # Both centroid floors: the locked Gate #1 value is carried verbatim, the fresh one is
    # re-measured on the calibration half and reported beside it.
    locked = load_sigma_sys_calibration(graph.surveys[IR], graph.root)
    fz = rec.frozen_screens
    assert fz.screens.centroid_locked.sigma_sys_arcsec == float(locked.sigma_sys.to_value("arcsec"))
    assert fz.screens.centroid_locked.sigma_sys_arcsec == pytest.approx(0.4356, abs=5e-5)
    fresh = got["fresh_sigma_sys"]
    assert fresh == fz.fresh_sigma_sys and fresh["n_stars"] == rec.n_calibration
    assert fz.screens.centroid_fresh.sigma_sys_arcsec == fresh["sigma_sys_arcsec"]
    assert fresh["sigma_sys_arcsec"] != fz.screens.centroid_locked.sigma_sys_arcsec


def test_validate_reverifies_the_promotion_end_to_end():
    res = {r.name: r for r in run_validation(ROOT)}
    row = res["unwise_gate_promotion"]
    assert row.ok, row.detail
    for needle in (
        "pinned by hephaistos_wise_mid_ir",
        PINNED_SHA[:16],
        "reproduced_archive",
        "positive definite",
        "thresholds re-derived from 126 calibration stars",
        "126 withheld summaries recomputed",
        "before every withheld fetch",
        "0 of 252 fresh stars among the 154 locked controls",
    ):
        assert needle in row.detail, needle
    assert all(r.ok for r in res.values()), [r.name for r in res.values() if not r.ok]
    assert len(res) == 42


def test_summary_reports_the_promoted_measurement_verbatim(gate, raw):
    """The summary the API / CLI expose is the stored withheld result, not a re-tuned one."""
    doc = json.loads(raw)
    s = gate.summary()
    assert (s["n_controls"], s["n_calibration"], s["n_withheld"]) == (252, 126, 126)
    assert (
        s["empirical_psf"]["fwhm_gaussian_arcsec"] == doc["empirical_psf"]["fwhm_gaussian_arcsec"]
    )
    assert s["noise_model"]["correlation_lag1"] == doc["noise_model"]["correlation_lag1"]
    # (model tuples vs JSON lists: compare after a JSON round trip)
    assert json.loads(json.dumps(s["screens"])) == doc["withheld"]["summary"]["screens"]
    assert s["unusable_cutouts"] == [] and s["n_failed"] == 0
    assert s["min_abs_b_deg"] > 30.0 and len(s["patches"]) == 16
    assert sum(p["n_calibration_controls"] + p["n_withheld_controls"] for p in s["patches"]) == 252


# ------------------------------------------------------------- tamper rejection
def test_tampered_bytes_break_the_pin(data_copy):
    load = _rewrite(data_copy, lambda d: d.update(description=d["description"] + "!"), repin=False)
    with pytest.raises(ProvenanceError, match="pinned"):
        load()


def test_content_hash_is_checked(data_copy):
    load = _rewrite(data_copy, lambda d: d.update(description=d["description"] + "?"), reseal=False)
    with pytest.raises(ProvenanceError, match="content_sha256"):
        load()


def test_withheld_summary_is_recomputed_from_per_star_statistics(data_copy):
    def flip(d):  # a consistent re-hash and re-pin cannot hide a changed measurement
        stat = d["withheld"]["statistics"]["null"]["delta_chi2_gls"]
        stat[0] = 1e9

    with pytest.raises(ProvenanceError, match=r"withheld\.summary"):
        _rewrite(data_copy, flip)()


def test_thresholds_are_rederived_from_calibration_nulls(data_copy):
    def bump(d):
        d["frozen_screens"]["screens"]["image_scalar_chi2"]["threshold"] *= 0.5

    with pytest.raises(FermiDeskError):
        _rewrite(data_copy, bump)()


def test_a_withheld_fetch_before_the_freeze_is_a_leakage_error(data_copy):
    def early(d):
        d["withheld"]["fetched_at_utc"][0] = d["frozen_screens"]["calibration"]["fetched_at_utc"][0]

    with pytest.raises(LeakageError):
        _rewrite(data_copy, early)()


def test_a_locked_gate1_star_in_the_fresh_sample_is_a_leakage_error(data_copy):
    locked = json.loads((data_copy / "data/calibrations/wise_sigma_sys_control.json").read_bytes())
    stolen = str(locked["controls"][0]["gaia_dr3_source_id"])

    def leak(d):
        old = d["sample"]["controls"][0]["gaia_dr3_source_id"]
        text = json.dumps(d).replace(f'"{old}"', f'"{stolen}"')
        d.clear()
        d.update(json.loads(text))

    with pytest.raises(LeakageError, match="locked Gate #1"):
        _rewrite(data_copy, leak)()


def test_dirty_git_sha_is_rejected(data_copy):
    with pytest.raises(ProvenanceError, match="clean"):
        _rewrite(data_copy, lambda d: d.update(code_git_sha="dirty"))()


def test_tampering_surfaces_in_validate_and_leaves_the_project_tree_alone(data_copy, raw):
    def early(d):
        d["withheld"]["fetched_at_utc"][0] = d["frozen_screens"]["calibration"]["fetched_at_utc"][0]

    _rewrite(data_copy, early)
    row = {r.name: r for r in run_validation(data_copy)}["unwise_gate_promotion"]
    assert not row.ok and "LeakageError" in row.detail
    assert (ROOT / UNWISE_GATE_PATH).read_bytes() == raw  # only the copy was edited
    for rel, sha in LOCKED.items():
        assert hashlib.sha256((data_copy / rel).read_bytes()).hexdigest() == sha, rel
