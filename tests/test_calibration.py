"""reproduced_archive WISE sigma_sys calibration (Milestone 2.5, Launch Gate #1).

All checks run offline against the committed artifact in data/calibrations/.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from conftest import ROOT
from fermi_desk.errors import ProvenanceError, SchemaValidationError
from fermi_desk.registry import (
    load_factor_graph,
    load_sigma_sys_calibration,
    run_validation,
)

IR = "hephaistos_wise_mid_ir"
SURVEY_YAML = "data/surveys/hephaistos_wise_mid_ir.yaml"


@pytest.fixture(scope="module")
def cal(graph):
    c = load_sigma_sys_calibration(graph.surveys[IR], graph.root)
    assert c is not None, "IR survey must pin the measured sigma_sys calibration"
    return c


def _canonical(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=True).encode()).hexdigest()


def _rewrite(data_copy, mutate, repin=True, recontent=True):
    """Mutate the calibration JSON in a data copy, optionally re-hashing it consistently."""
    sv_path = data_copy / SURVEY_YAML
    sv = yaml.safe_load(sv_path.read_text())
    path = data_copy / sv["infrared"]["sigma_sys_calibration_path"]
    doc = json.loads(path.read_bytes())
    mutate(doc)
    if recontent:
        doc["content_sha256"] = _canonical({k: v for k, v in doc.items() if k != "content_sha256"})
    raw = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    if repin:
        sv["infrared"]["sigma_sys_calibration_sha256"] = hashlib.sha256(raw).hexdigest()
        sv_path.write_text(yaml.safe_dump(sv, sort_keys=False))
    g = load_factor_graph(data_copy)
    return lambda: load_sigma_sys_calibration(g.surveys[IR], g.root)


def test_calibration_is_multi_patch_reproduced_archive(cal):
    rec = cal.record
    assert rec.data_tier == "reproduced_archive"
    assert len(rec.code_git_sha) == 40  # clean commit, never "dirty"
    assert rec.n_controls >= 100 and rec.n_patches_with_controls >= 8
    assert all(abs(p.b_deg) - p.radius_deg >= 20.0 for p in rec.patches)
    assert all(abs(c.b_deg) > 20.0 and c.ruwe < 1.4 for c in rec.controls)
    assert all(c.cc_flags == "0000" and c.ext_flg == 0 for c in rec.controls)
    assert all(p.gaia_adql.startswith("SELECT") and p.irsa_adql for p in rec.patches)
    n_cal = sum(c.split == "calibration" for c in rec.controls)
    assert abs(2 * n_cal - rec.n_controls) <= 1
    assert rec.legacy_baseline_sigma_sys_arcsec == 0.5  # baseline kept, never erased
    s = cal.summary()
    assert set(s["withheld_false_alarm"]) == {
        "out_of_sample_false_alarm_at_calibrated_floor",
        "false_alarm_at_legacy_baseline_floor",
        "false_alarm_with_no_floor",
    }
    lo, hi = s["sigma_sys_68_arcsec"]
    assert lo <= s["sigma_sys_arcsec"] <= hi


def test_validate_reports_calibration_check():
    res = {r.name: r for r in run_validation(ROOT)}
    key = next(k for k in res if k.startswith("sigma_sys_calibration:"))
    assert res[key].ok, res[key].detail
    assert "recomputed" in res[key].detail


def test_tampered_bytes_break_the_pinned_checksum(data_copy):
    load = _rewrite(data_copy, lambda d: d.update(description=d["description"] + "!"), repin=False)
    with pytest.raises(ProvenanceError, match="pinned"):
        load()


def test_content_hash_is_checked(data_copy):
    load = _rewrite(data_copy, lambda d: d.update(rng_seed=d["rng_seed"] + 1), recontent=False)
    with pytest.raises(ProvenanceError, match="content_sha256"):
        load()


def test_statistics_are_recomputed_from_the_controls(data_copy):
    def bump(d):  # a consistent re-hash cannot hide a changed measurement
        for c in d["controls"]:
            if c["split"] == "calibration":
                c["offset_arcsec"] *= 1.3

    with pytest.raises(ProvenanceError, match="recomputed"):
        _rewrite(data_copy, bump)()


def test_dirty_git_sha_calibration_is_rejected(data_copy):
    with pytest.raises(ProvenanceError, match="clean"):
        _rewrite(data_copy, lambda d: d.update(code_git_sha="dirty"))()


def test_too_few_patches_rejected(data_copy):
    def one_patch(d):
        d["n_patches_with_controls"] = 1

    with pytest.raises((SchemaValidationError, ValueError)):
        _rewrite(data_copy, one_patch)()
