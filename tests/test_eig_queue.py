"""EIG scorer and SQLite BOED queue (docs/MILESTONE_2.md B3)."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from fermi_desk.errors import QueueStateError
from fermi_desk.inference import eig
from fermi_desk.models.expansion import C_KPC_PER_GYR
from fermi_desk.queue.store import (
    JOB_DEFINITIONS,
    JobStore,
    compute_job_diagnostic,
    compute_job_eig,
    execute_job,
)

SMALL = {
    "lookback_window_t_sweep": {
        "t_min_gyr": 5.0e-5,
        "t_max_gyr": 1.0e-2,
        "n_points": 2,
        "geometries": ["thin_disk_2d", "thick_disk_3d"],
        "channels": ["radio"],
        "n_mc": 100,
    },
    "disk_scale_height_sweep": {
        "scale_heights_pc": [200.0, 300.0],
        "channels": ["radio"],
        "n_mc": 100,
    },
    "eirp_luminosity_function_sweep": {"medians_w": [1e13], "sigmas_dex": [1.0], "n_mc": 100},
    "hephaistos_blend_recovery_grid": {
        "separations_arcsec": [1.0, 6.0],
        "flux_fractions": [0.5],
        "n_trials": 6,
    },
}


def small_jobs():
    out = []
    for job in JOB_DEFINITIONS:
        if job.job_id in SMALL:
            job = replace(job, parameters={**job.parameters, **SMALL[job.job_id]})
        out.append(job)
    return out


# --------------------------------------------------------------- MI identities
def test_mutual_information_identities():
    prior = {"a": 0.5, "b": 0.5}
    same = {"a": np.array([0.3, 0.7]), "b": np.array([0.3, 0.7])}
    assert eig.mutual_information_bits(prior, same) == 0.0
    disjoint = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    assert eig.mutual_information_bits(prior, disjoint) == pytest.approx(1.0)
    skew = {"a": 0.9, "b": 0.1}
    assert eig.mutual_information_bits(skew, disjoint) == pytest.approx(eig.entropy_bits(skew))
    rnd = {"a": np.array([0.2, 0.5, 0.3]), "b": np.array([0.6, 0.1, 0.3])}
    assert 0 < eig.mutual_information_bits(prior, rnd) <= eig.entropy_bits(prior)
    with pytest.raises(ValueError):
        eig.mutual_information_bits(prior, {"a": np.array([0.5, 0.6]), "b": np.array([1.0, 0.0])})


def test_binomial_outcomes_are_normalised_on_a_common_support():
    out = eig.binomial_count_outcomes(200_000, {"lo": 1e-6, "hi": 5e-5})
    sizes = {v.size for v in out.values()}
    assert len(sizes) == 1
    for v in out.values():
        assert v.sum() == pytest.approx(1.0, abs=1e-12)


def test_utility_is_bits_per_hour():
    assert eig.utility_bits_per_hour(1.0, 3600.0, 60.0) == pytest.approx(0.5)
    assert eig.utility_bits_per_hour(1.0, 0.0, 0.0) == 0.0


def test_archival_job_eig_is_bounded(graph):
    by = {j.job_id: j for j in JOB_DEFINITIONS}
    ws = compute_job_eig(graph, by["wlodarczyk_sroka_288k_background_extension"])
    assert 0.0 <= ws.bits <= 1.0
    modes = ws.details["by_radio_mode"]
    # The distance-scaled (physical) mode is far less informative than flat at kpc distances.
    assert modes["distance_scaled_eirp"]["eig_bits"] < modes["flat_probability"]["eig_bits"]
    ctl = compute_job_eig(graph, by["wise_independent_control_fields"])
    assert 0.0 < ctl.bits <= ctl.details["max_identifiable_bits"] + 1e-12
    assert ctl.details["max_identifiable_bits"] == pytest.approx(math.log2(5))
    for j in JOB_DEFINITIONS:
        if j.tier == "0_compute":
            assert compute_job_eig(graph, j).bits == 0.0


# --------------------------------------------------------------- queue lifecycle
@pytest.fixture
def store(tmp_path, graph):
    s = JobStore(tmp_path / "queue.sqlite")
    s.seed(graph, small_jobs())
    yield s
    s.close()


def test_seed_registers_six_jobs_idempotently(store, graph):
    rows = store.list()
    ids = {r["job_id"] for r in rows}
    assert len(rows) == 6 and "lookback_window_t_sweep" in ids
    assert {r["tier"] for r in rows} == {"0_compute", "1_archival"}
    utilities = [r["utility_bits_per_hour"] for r in rows]
    assert utilities == sorted(utilities, reverse=True)
    assert store.seed(graph, small_jobs()) == []  # idempotent
    for col in (
        "expected_eig_bits",
        "eig_prior_json",
        "est_cpu_seconds",
        "est_reviewer_minutes",
        "status",
        "created_at",
        "parameters_json",
        "target_module",
    ):
        assert col in rows[0]


def test_state_machine_guards(store, graph, tmp_path):
    with pytest.raises(QueueStateError):
        store.run(graph, "lookback_window_t_sweep", 42, tmp_path)  # unclaimed
    store.claim("wise_independent_control_fields", "tester")
    with pytest.raises(QueueStateError):
        store.run(graph, "wise_independent_control_fields", 42, tmp_path)  # archival
    with pytest.raises(QueueStateError):
        store.claim("wise_independent_control_fields", "someone_else")  # already claimed
    store.reject("wise_independent_control_fields", "needs live IRSA access")
    assert store.get("wise_independent_control_fields")["status"] == "rejected"
    with pytest.raises(QueueStateError):
        store.replay(graph, "disk_scale_height_sweep", tmp_path)  # never run


SLOW_REPLAYS = {"lookback_window_t_sweep", "disk_scale_height_sweep"}


@pytest.mark.parametrize(
    "job_id",
    [pytest.param(j, marks=pytest.mark.slow) if j in SLOW_REPLAYS else j for j in sorted(SMALL)],
)
def test_every_tier0_job_replays_with_identical_hash(store, graph, tmp_path, job_id):
    store.claim(job_id, "tester")
    out = store.run(graph, job_id, 42, tmp_path)
    row = store.get(job_id)
    assert row["status"] == "completed" and row["rng_seed"] == 42
    assert row["output_sha256"] == out["output_sha256"]
    rep = store.replay(graph, job_id, tmp_path)
    assert rep["matches_recorded"] is True
    assert rep["output_sha256"] == row["output_sha256"]
    kinds = [e["kind"] for e in store.executions(job_id)]
    assert kinds == ["run", "replay"]


def test_lookback_sweep_records_exact_validity_threshold(graph):
    job = next(j for j in small_jobs() if j.job_id == "lookback_window_t_sweep")
    params = {**job.parameters, "n_points": 3, "geometries": ["thin_disk_2d"]}
    out = execute_job(graph, job.job_id, params, 42)["results"]
    assert out["t_grid_gyr"][0] == pytest.approx(5e-5) and out["t_grid_gyr"][-1] == pytest.approx(
        1e-2
    )
    for th in out["thresholds"].values():
        assert th["t_valid_gyr"] == pytest.approx(8.0 / (th["expansion_speed_c"] * C_KPC_PER_GYR))
    fs = out["thresholds"]["frequent_slow_expanders"]["t_valid_gyr"]
    rf = out["thresholds"]["rare_fast_expanders"]["t_valid_gyr"]
    assert rf == pytest.approx(fs / 100)
    for row in out["rows"]:
        for hid, h in row["hypotheses"].items():
            violated = row["lookback_gyr"] > out["thresholds"][hid]["t_valid_gyr"]
            assert (h["local_approx"] == "VIOLATED") == violated
        assert row["log10_bf_radio"] is not None
        assert set(row["bf_terms"]) >= {"earth_unreached", "radio"}


def test_seed_changes_monte_carlo_job_output(graph):
    job = next(j for j in small_jobs() if j.job_id == "hephaistos_blend_recovery_grid")
    a = execute_job(graph, job.job_id, job.parameters, 1)
    b = execute_job(graph, job.job_id, job.parameters, 2)
    assert a["output_sha256"] != b["output_sha256"]


# ------------------------------------------- Tier-0 diagnostics vs. Tier-1 EIG
def test_tier0_jobs_carry_diagnostic_impact_but_zero_eig(store, graph):
    compute, archival = store.list_compute(), store.list_archival()
    assert {r["tier"] for r in compute} == {"0_compute"} and len(compute) == 4
    assert {r["tier"] for r in archival} == {"1_archival"} and len(archival) == 2
    for r in compute:
        assert r["expected_eig_bits"] == 0.0  # never fabricate sky-photon bits
        assert r["diagnostic_impact"] and r["diagnostic_priority"] >= 1
        assert "INFERRED" not in r["source"]
    priorities = [r["diagnostic_priority"] for r in compute]
    assert priorities == sorted(priorities) and len(set(priorities)) == len(priorities)
    for r in archival:
        assert r["diagnostic_impact"] is None and r["diagnostic_priority"] is None
    u = [r["utility_bits_per_hour"] for r in archival]
    assert u == sorted(u, reverse=True)
    # The lookback statement is computed from the hypotheses: T_valid = 8 kpc / v_exp.
    lb = next(r for r in compute if r["job_id"] == "lookback_window_t_sweep")
    t_fast = 8.0 / (0.5 * C_KPC_PER_GYR) * 1e6  # kyr
    assert f"{t_fast:.3g} kyr" in lb["diagnostic_impact"]


def test_tier0_job_without_priority_is_refused(graph):
    job = next(j for j in JOB_DEFINITIONS if j.tier == "0_compute")
    with pytest.raises(QueueStateError):
        compute_job_diagnostic(graph, replace(job, diagnostic_priority=None))
    arch = next(j for j in JOB_DEFINITIONS if j.tier == "1_archival")
    with pytest.raises(QueueStateError):
        compute_job_diagnostic(graph, replace(arch, diagnostic_priority=1))


def test_pre_milestone_2p5_queue_is_migrated_in_place(tmp_path, graph):
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, tier TEXT, title TEXT, target_module TEXT, "
        "parameters_json TEXT, expected_eig_bits REAL, eig_prior_json TEXT, eig_basis TEXT, "
        "eig_details_json TEXT, est_cpu_seconds REAL, est_reviewer_minutes REAL, "
        "utility_bits_per_hour REAL, status TEXT, source TEXT, archival_script TEXT, "
        "claimed_by TEXT, claimed_at TEXT, code_git_sha TEXT, rng_seed INTEGER, "
        "output_sha256 TEXT, output_path TEXT, rejection_reason TEXT, created_at TEXT, "
        "completed_at TEXT)"
    )
    con.execute(
        "INSERT INTO jobs (job_id, tier, title, target_module, parameters_json, "
        "expected_eig_bits, eig_prior_json, eig_basis, eig_details_json, est_cpu_seconds, "
        "est_reviewer_minutes, utility_bits_per_hour, status, source, created_at, output_sha256) "
        "VALUES ('lookback_window_t_sweep', '0_compute', 't', 'm', '{}', 0, '{}', 'b', '{}', "
        "1, 1, 0, 'completed', 'old', 'now', 'abc')"
    )
    con.commit()
    con.close()
    s = JobStore(path)
    s.seed(graph, small_jobs())
    row = s.get("lookback_window_t_sweep")
    assert row["status"] == "completed" and row["output_sha256"] == "abc"  # untouched
    assert row["diagnostic_priority"] == 1 and row["diagnostic_impact"]
    s.close()
