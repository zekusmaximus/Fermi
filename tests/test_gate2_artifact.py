"""Promoted Launch Gate #2 artifact (reproduced_archive) - offline checks only.

The real AllWISE FITS cutouts are never read here; only the committed summary file in
data/calibrations/ and its pin in the IR survey YAML.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from conftest import ROOT
from fermi_desk.errors import ProvenanceError
from fermi_desk.registry import (
    load_factor_graph,
    load_fits_gate2,
    load_sigma_sys_calibration,
    run_validation,
)

IR = "hephaistos_wise_mid_ir"
SURVEY_YAML = "data/surveys/hephaistos_wise_mid_ir.yaml"
LOCKED_SIGMA_SYS_SHA = "2da2a234ee0ccb07fb571431f5f178e362ff4d9d4f96d4ee878b921916127b7a"
LOCKED_CATALOG_SHA = "c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2"


@pytest.fixture(scope="module")
def gate(graph):
    g = load_fits_gate2(graph.surveys[IR], graph.root)
    assert g is not None, "IR survey must pin the promoted Gate #2 result"
    return g


def _canonical(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=True).encode()).hexdigest()


def _rewrite(data_copy, mutate, repin=True, recontent=True):
    sv_path = data_copy / SURVEY_YAML
    sv = yaml.safe_load(sv_path.read_text())
    path = data_copy / sv["infrared"]["fits_gate2_path"]
    doc = json.loads(path.read_bytes())
    mutate(doc)
    if recontent:
        doc["content_sha256"] = _canonical({k: v for k, v in doc.items() if k != "content_sha256"})
    raw = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    if repin:
        sv["infrared"]["fits_gate2_sha256"] = hashlib.sha256(raw).hexdigest()
        sv_path.write_text(yaml.safe_dump(sv, sort_keys=False))
    g = load_factor_graph(data_copy)
    return lambda: load_fits_gate2(g.surveys[IR], g.root)


def test_locked_artifacts_are_byte_identical():
    cat = (ROOT / "data/catalogs/bl_692_targets.csv").read_bytes()
    cal = (ROOT / "data/calibrations/wise_sigma_sys_control.json").read_bytes()
    assert hashlib.sha256(cat).hexdigest() == LOCKED_CATALOG_SHA
    assert hashlib.sha256(cal).hexdigest() == LOCKED_SIGMA_SYS_SHA


def test_gate2_is_complete_frozen_and_leak_free(graph, gate):
    rec = gate.record
    assert rec.data_tier == "reproduced_archive" and len(rec.code_git_sha) == 40
    assert rec.fetch.n_ok == rec.fetch.n_requested == 154 and rec.fetch.n_failed == 0
    assert rec.locked_inputs.sigma_sys_calibration_sha256 == LOCKED_SIGMA_SYS_SHA
    locked = load_sigma_sys_calibration(graph.surveys[IR], graph.root)
    split = {
        s: {c.gaia_dr3_source_id for c in locked.record.controls if c.split == s}
        for s in ("calibration", "withheld")
    }
    assert set(rec.frozen_screens.calibration["star_ids"]) == split["calibration"]
    assert set(rec.withheld.star_ids) == split["withheld"]
    assert set(rec.empirical_psf.star_ids) <= split["calibration"]
    start, end = rec.query_window_utc
    assert start <= rec.frozen_at_utc <= end
    s = gate.summary()
    assert set(s["screens"]) == {"centroid", "image", "naive_flux_excess"}
    for sc in s["screens"].values():
        lo, hi = sc["arm_b_false_rejection_pooled"]["clopper_pearson_95"]
        assert lo <= sc["arm_b_false_rejection_pooled"]["rate"] <= hi


def test_validate_recomputes_gate2_offline():
    res = {r.name: r for r in run_validation(ROOT)}
    key = next(k for k in res if k.startswith("fits_gate2:"))
    assert res[key].ok, res[key].detail
    assert "recomputed" in res[key].detail


def test_tampered_bytes_break_the_pin(data_copy):
    load = _rewrite(data_copy, lambda d: d.update(description=d["description"] + "!"), repin=False)
    with pytest.raises(ProvenanceError, match="pinned"):
        load()


def test_content_hash_is_checked(data_copy):
    load = _rewrite(
        data_copy, lambda d: d.update(description=d["description"] + "?"), recontent=False
    )
    with pytest.raises(ProvenanceError, match="content_sha256"):
        load()


def test_summaries_are_recomputed_from_per_star_statistics(data_copy):
    def flip(d):  # a consistent re-hash cannot hide a changed measurement
        arm_b = d["withheld"]["statistics"]["arm_b"]["delta_chi2"]
        arm_b[0] = [1e9 for _ in arm_b[0]]

    with pytest.raises(ProvenanceError, match=r"recomputed|stored"):
        _rewrite(data_copy, flip)()


def test_threshold_is_rederived_from_calibration_nulls(data_copy):
    def bump(d):
        d["frozen_screens"]["image"]["delta_chi2_threshold"] *= 0.5

    with pytest.raises(ProvenanceError):
        _rewrite(data_copy, bump)()


def test_withheld_star_swapped_into_calibration_is_rejected(data_copy):
    def leak(d):
        d["frozen_screens"]["calibration"]["star_ids"][0] = d["withheld"]["star_ids"][0]

    with pytest.raises(ProvenanceError):
        _rewrite(data_copy, leak)()


def test_dirty_git_sha_is_rejected(data_copy):
    with pytest.raises(ProvenanceError, match="clean"):
        _rewrite(data_copy, lambda d: d.update(code_git_sha="dirty"))()
