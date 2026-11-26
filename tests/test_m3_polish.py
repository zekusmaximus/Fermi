"""Milestone 3 Part A: Milestone 2.5 audit polish rulings (offline)."""

from __future__ import annotations

import hashlib
import importlib.util

import pytest

from conftest import ROOT
from fermi_desk.inference.engine import RECEIPT_SCHEMA_VERSION, RunConfig, run_comparison
from fermi_desk.inference.ir_drilldown import run_ir_drilldown
from fermi_desk.queue.store import JOB_DEFINITIONS, execute_job

PAIR = ("frequent_slow_expanders", "rare_fast_expanders")
LOCKED = {
    "data/catalogs/bl_692_targets.csv": "c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2",
    "data/calibrations/wise_sigma_sys_control.json": (
        "2da2a234ee0ccb07fb571431f5f178e362ff4d9d4f96d4ee878b921916127b7a"
    ),
}


def test_outbound_network_is_blocked_during_tests():
    import socket

    from conftest import NetworkBlockedError

    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("irsa.ipac.caltech.edu", 443), timeout=1)
    with pytest.raises(NetworkBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("93.184.216.34", 80))


def test_locked_artifacts_are_byte_identical():
    for rel, sha in LOCKED.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, rel


# A.2 ------------------------------------------------------- headline exclusion notes
def _receipt(graph, mode):
    cfg = RunConfig(geometry="thin_disk_2d", radio_mode=mode, seed=7, n_mc=200, headline_pair=PAIR)
    return run_comparison(graph, cfg, list(PAIR))


def test_distance_scaled_mode_explains_radio_holdout(graph):
    rec = _receipt(graph, "distance_scaled_eirp")
    assert rec["receipt_schema_version"] == RECEIPT_SCHEMA_VERSION
    notes = {n["channel"]: n for n in rec["headline_exclusion_notes"]}
    radio = notes["radio"]
    assert radio["held_out_of_reported_headline"] and radio["active_in_run"]
    assert radio["illustrative_inputs"] == ["transmitter_eirp_luminosity_function"]
    assert "luminosity function" in radio["reasons"][0]
    assert "illustrative_calibration" in radio["reasons"][0]


def test_flat_mode_keeps_radio_in_verified_headline(graph):
    rec = _receipt(graph, "flat_probability")
    notes = {n["channel"] for n in rec["headline_exclusion_notes"]}
    assert "radio" not in notes
    assert {"infrared", "solar_system"} <= notes  # active but illustrative -> explained


# A.1 / A.3 -------------------------------------------------------- IR drill-down
@pytest.fixture(scope="module")
def drill(graph):
    return run_ir_drilldown(
        graph, seed=3, pair=("quiet_expanders", "partial_dyson_conspicuous"), n_trials=4
    )["results"]


def test_small_sample_false_alarm_bounds_are_structured(drill):
    b = drill["small_sample_false_alarm_bounds"]
    assert b["linked_archival_job"] == "wise_independent_control_fields"
    w = b["withheld_at_measured_floor"]
    assert (w["n_flagged"], w["n_stars"]) == (0, 77)
    assert w["clopper_pearson_95_upper"] == pytest.approx(0.0468, abs=5e-4)
    assert w["ks_rayleigh_p_value"] == pytest.approx(0.448, abs=0.01)
    c = b["calibration_in_sample_tail"]
    assert (c["n_flagged"], c["n_stars"]) == (2, 77)
    assert c["binomial_p_at_least_k_given_nominal"] == pytest.approx(0.0187, abs=1e-3)
    z = b["withheld_with_no_floor"]
    assert (z["n_flagged"], z["n_stars"]) == (11, 77)
    assert "4.7%" in b["statement"]


def test_atlas_smoothing_caveat_and_patch_breakdown(drill):
    cav = drill["atlas_smoothing_caveat"]
    assert "0.436" in cav and "smoothing" in cav and "unWISE" in cav
    ce = drill["control_experiment"]
    assert ce["n_controls"] == 154 and len(ce["patches"]) == 12
    assert sum(p["measured_controls"] for p in ce["patches"]) == 154
    assert sum(p["n_calibration"] + p["n_withheld"] for p in ce["patches"]) == 154
    for a in drill["anchor_benchmarks"]:
        assert set(a["by_floor"]) == {"legacy_baseline", "measured_reproduced_archive"}
    flip = drill["contamination_flip"]
    assert len(flip["rows"]) == 6 and "neither is preferred" in flip["policy"]


# A.4 ------------------------------------------------- dual-floor queue runner
def test_hephaistos_runner_evaluates_both_floors_deterministically(graph):
    job = next(j for j in JOB_DEFINITIONS if j.job_id == "hephaistos_blend_recovery_grid")
    params = {**job.parameters, "separations_arcsec": [1.0, 6.0], "flux_fractions": [0.5]}
    params["n_trials"] = 4
    a = execute_job(graph, job.job_id, params, 42)
    b = execute_job(graph, job.job_id, params, 42)
    assert a["output_sha256"] == b["output_sha256"]
    res = a["results"]
    floors = {f["label"]: f["sigma_sys_arcsec"] for f in res["registration_floors"]}
    assert floors["legacy_baseline"] == 0.5
    assert floors["measured_reproduced_archive"] == pytest.approx(0.4356, abs=1e-4)
    assert (
        res["measured_calibration_file_sha256"]
        == LOCKED["data/calibrations/wise_sigma_sys_control.json"]
    )
    assert res["floor_shift"] is not None


# A.5 ---------------------------------------------------- future latitude default
def test_control_script_default_latitude_is_30_deg():
    spec = importlib.util.spec_from_file_location(
        "wise_control_script", ROOT / "scripts" / "run_wise_control_experiment.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # import only: no network at import time
    assert mod.build_parser().parse_args(["--seed", "1"]).min_abs_b == 30.0
