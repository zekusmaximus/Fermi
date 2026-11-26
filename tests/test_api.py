"""FastAPI desk (docs/MILESTONE_3.md Part B) - offline TestClient tests.

Every test uses an isolated tmp_path SQLite queue, queue output directory and runs/
directory, so runs/queue.sqlite is never touched. The network guard in conftest.py
blocks any outbound connection.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from fermi_desk.api import create_app, is_loopback_host, json_safe
from fermi_desk.queue.store import JOB_DEFINITIONS

FAST_JOBS = {
    "lookback_window_t_sweep": {
        "n_points": 2,
        "geometries": ["thin_disk_2d"],
        "channels": ["radio"],
        "n_mc": 100,
    },
    "hephaistos_blend_recovery_grid": {
        "separations_arcsec": [1.0, 6.0],
        "flux_fractions": [0.5],
        "n_trials": 4,
    },
}


def _jobs():
    return [
        replace(j, parameters={**j.parameters, **FAST_JOBS[j.job_id]})
        if j.job_id in FAST_JOBS
        else j
        for j in JOB_DEFINITIONS
    ]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("desk")
    (tmp / "runs").mkdir()
    static = tmp / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Fermi Desk</title>", "utf-8")
    app = create_app(
        ROOT,
        queue_path=tmp / "queue.sqlite",
        queue_out_dir=tmp / "queue_out",
        runs_dir=tmp / "runs",
        static_dir=static,
        compare_n_mc=1000,
        ir_n_trials=4,
        adversary_n_mc=200,
        seed_jobs=_jobs(),
    )
    return TestClient(app), tmp


@pytest.fixture(scope="module")
def client(env):
    return env[0]


def _ok(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------- reads
def test_health_reports_provenance(client):
    h = _ok(client, "/api/health")
    assert h["status"] == "ok" and h["validation_passed"] is True
    assert h["validation_checks_passed"] == h["validation_checks_total"] >= 30
    assert h["catalog_sha256"].startswith("c299d61d")
    assert h["wise_calibration_sha256"].startswith("2da2a234")
    assert h["receipt_schema_version"] == 4 and h["git_sha"]
    # v0.1.2: live control-sample provenance, read from the verified calibration files.
    assert (h["n_controls"], h["n_calibration"], h["n_withheld"]) == (154, 77, 77)
    assert h["calibration_data_tier"] == "reproduced_archive"
    assert h["fits_gate2_data_tier"] == "reproduced_archive" and len(h["fits_gate2_sha256"]) == 64


def test_registry_groups_tiers_with_monoculture_audits(client):
    r = _ok(client, "/api/registry")
    t3 = {h["id"]: h for h in r["hypotheses"]["tier_3"]}
    assert set(t3) == {"zoo_hypothesis", "dark_forest"}
    for h in t3.values():
        assert h["is_quarantined"] and h["empirical_score"] is None
        assert set(h["monoculture_audit"]) >= {
            "required_compliance_fraction",
            "timescale_gyr",
            "defector_vulnerability",
        }
    assert {p["id"] for p in r["parameters"]} >= {"origin_rate_per_kpc2_gyr", "expansion_speed_c"}
    assert all("data_tier" in s for s in r["surveys"])


def test_radio_drilldown_targets_hip1692_and_beam_pairs(client):
    r = _ok(client, "/api/radio/drilldown")
    assert r["n_targets"] == len(r["targets"]) == 692
    keys = {
        "target_name",
        "ra_deg",
        "dec_deg",
        "dist_pc",
        "dist_source",
        "dist_legacy_pc",
        "eirp_min_w",
        "eirp_revision_factor",
    }
    assert keys <= set(r["targets"][0])
    hip = next(t for t in r["targets"] if t["target_name"] == "HIP 1692")
    assert hip["dist_pc"] == pytest.approx(788.7067) and hip["dist_legacy_pc"] == 23.0
    dists = {k: v["distance_pc"] for k, v in r["hip1692_audit"]["distances"].items()}
    assert sorted(round(d, 2) for d in dists.values()) == [23.0, 752.61, 788.71]
    assert len(r["beam_sharing_pairs"]) == 16


def test_infrared_drilldown_payload(client):
    r = _ok(client, "/api/infrared/drilldown?seed=42")
    res = r["results"]
    labels = {f["label"]: f["sigma_sys_arcsec"] for f in res["registration_floors"]}
    assert labels["legacy_baseline"] == 0.5
    assert labels["measured_reproduced_archive"] == pytest.approx(0.4356, abs=1e-4)
    assert res["floor_shift"]["centroid_recovery_delta"]
    b = res["small_sample_false_alarm_bounds"]
    assert b["withheld_at_measured_floor"]["n_flagged"] == 0
    assert b["calibration_in_sample_tail"]["n_flagged"] == 2
    assert b["withheld_with_no_floor"]["n_flagged"] == 11
    assert res["control_experiment"]["n_controls"] == 154
    assert "smoothing" in res["atlas_smoothing_caveat"]
    assert {a["candidate"] for a in res["anchor_benchmarks"]} == {"D", "E"}
    gate = res["real_fits_gate2"]
    assert gate["data_tier"] == "reproduced_archive" and gate["n_withheld"] == 77
    assert set(gate["screens"]) == {"centroid", "image", "naive_flux_excess"}
    side = res["blend_survival_side_by_side"]
    assert side["real_fits"]["data_tier"] == "reproduced_archive"
    assert side["synthetic"]["data_tier"] == "illustrative_calibration"
    assert res["data_tier"] == "illustrative_calibration"  # synthetic grid keeps its tier
    assert r["rng_seed"] == 42
    assert client.get("/api/infrared/drilldown?seed=-3").status_code == 422


def test_adversary_receipt(client):
    r = _ok(client, "/api/adversary?seed=42")
    checks = {f["check"]: f for f in r["results"]["findings"]}
    assert checks["local_approximation"]["status"] == "finding"
    assert len(checks["duplicate_pointings"]["details"]["beam_sharing_pairs"]) == 16
    assert all(f["repro_command"].endswith("--seed 42") for f in checks.values())
    assert checks["hip1692_distance_revision"]["citations"]
    assert len(r["results_sha256"]) == 64


def test_runs_feed_lists_receipts_newest_first(env):
    client, tmp = env
    (tmp / "runs" / "a.json").write_text('{"kind": "adversary", "rng_seed": 1}', "utf-8")
    (tmp / "runs" / "broken.json").write_text("{not json", "utf-8")
    r = _ok(client, "/api/runs")
    assert r["limit"] == 50
    assert [x["kind"] for x in r["runs"]] == ["adversary"]  # unparsable receipts skipped


# --------------------------------------------------------------------- compare
def test_compare_default_is_verified_and_opt_in_folds_illustrative(client):
    d = client.post("/api/compare", json={"seed": 42})
    assert d.status_code == 200, d.text
    d = d.json()
    assert d["headline_policy"] == "verified_channels_only"
    head = d["receipt"]["results"]["headline"]["headline"]
    assert head["terms"] == ["earth_unreached", "radio"]
    i = client.post("/api/compare", json={"seed": 42, "include_illustrative": True}).json()
    assert i["headline_policy"] == "including_illustrative_channels"
    assert i["headline_log10_bf"] != d["headline_log10_bf"]
    again = client.post("/api/compare", json={"seed": 42})
    assert again.json()["cached"] is True
    assert again.json()["receipt"] == d["receipt"]  # deterministic cache hit


def test_compare_distance_scaled_returns_exclusion_notes(client):
    r = client.post("/api/compare", json={"radio_mode": "distance_scaled_eirp"}).json()
    notes = {n["channel"]: n for n in r["headline_exclusion_notes"]}
    assert "radio" in notes and "luminosity function" in notes["radio"]["reasons"][0]


def test_compare_never_scores_tier3_and_flags_local_validity(client):
    rec = client.post("/api/compare", json={}).json()["receipt"]["results"]
    assert {"zoo_hypothesis", "dark_forest"}.isdisjoint(rec["hypotheses"])
    assert {q["hypothesis_id"] for q in rec["tier3_quarantine"]} == {
        "zoo_hypothesis",
        "dark_forest",
    }
    assert all(q["empirical_score"] is None for q in rec["tier3_quarantine"])
    fast = rec["hypotheses"]["rare_fast_expanders"]["expansion"]
    assert fast["local_approximation_valid"] is False
    assert fast["t_valid_kyr"] == pytest.approx(52.2, abs=0.05)


@pytest.mark.parametrize(
    "body",
    [
        {"geometry_mode": "disk"},
        {"seed": "42"},
        {"seed": -1},
        {"seed": 2**40},
        {"channels": []},
        {"channels": ["radio", "radio"]},
        {"channels": ["xray"]},
        {"include_illustrative": "yes"},
        {"unknown_field": 1},
    ],
)
def test_compare_rejects_invalid_requests(client, body):
    assert client.post("/api/compare", json=body).status_code == 422


# ----------------------------------------------------------------------- queue
def test_queue_lists_split_tiers(client):
    q = _ok(client, "/api/queue")
    t1, t0 = q["tier1_archival"], q["tier0_compute"]
    assert t1[0]["job_id"] == "wise_independent_control_fields"
    assert [j["utility_score"] for j in t1] == sorted(
        (j["utility_score"] for j in t1), reverse=True
    )
    assert t1[0]["linked_evidence"]["withheld_at_measured_floor"]["n_stars"] == 77
    assert t1[0]["commands"]["script"].startswith("uv run python scripts/")
    assert all(j["expected_eig_bits"] == 0.0 and j["diagnostic_impact"] for j in t0)
    assert [j["diagnostic_priority"] for j in t0] == sorted(j["diagnostic_priority"] for j in t0)
    assert "queue.sqlite" in q["queue_path"]


def test_tier1_run_is_refused_with_script_path(client):
    r = client.post("/api/queue/wise_independent_control_fields/run", json={"seed": 42})
    assert r.status_code == 400
    assert r.json()["detail"]["archival_script"] == "scripts/run_wise_control_experiment.py"


def test_tier0_run_then_replay_matches(env):
    client, tmp = env
    run = client.post(
        "/api/queue/lookback_window_t_sweep/run",
        json={"seed": 42, "contributor": "local-desk-ui"},
    )
    assert run.status_code == 200, run.text
    run = run.json()
    assert run["job"]["status"] == "completed" and run["job"]["claimed_by"] == "local-desk-ui"
    assert run["output_json"]["results"]["thresholds"]
    rep = client.post("/api/queue/lookback_window_t_sweep/replay").json()
    assert rep["match"] is True
    assert rep["recorded_sha256"] == rep["replayed_sha256"] == run["output_sha256"]
    # a completed job is not silently re-run
    assert client.post("/api/queue/lookback_window_t_sweep/run", json={}).status_code == 409
    assert not (ROOT / "runs" / "queue_out").exists()
    assert (tmp / "queue.sqlite").is_file()


@pytest.mark.parametrize("job_id", ["no_such_job", "..", "UPPER", "a" * 200])
def test_unknown_or_malformed_job_ids_are_404(client, job_id):
    # ".." is normalised away by the URL layer (405, never reaches the handler)
    assert client.post(f"/api/queue/{job_id}/run", json={}).status_code in (404, 405)


def test_bad_contributor_rejected(client):
    r = client.post("/api/queue/disk_scale_height_sweep/run", json={"contributor": "<script>"})
    assert r.status_code == 422


# -------------------------------------------------------------------- static
def test_static_bundle_and_api_share_one_app(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fermi Desk" in r.text
    assert client.get("/api/health").status_code == 200
    cors = client.options(
        "/api/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert cors.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = client.options(
        "/api/health",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in evil.headers


def test_json_safe_maps_non_finite_to_null():
    import numpy as np

    assert json_safe({"a": float("inf"), "b": [np.float64("nan"), np.int64(3)]}) == {
        "a": None,
        "b": [None, 3],
    }


# ------------------------------------------------------ v0.1.1 launch safety
def test_health_reports_read_write_mode_by_default(client):
    assert _ok(client, "/api/health")["read_only"] is False


@pytest.fixture(scope="module")
def ro_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("desk_ro")
    app = create_app(
        ROOT,
        queue_path=tmp / "queue.sqlite",
        queue_out_dir=tmp / "queue_out",
        runs_dir=tmp / "runs",
        compare_n_mc=1000,
        seed_jobs=_jobs(),
        read_only=True,
    )
    return TestClient(app), tmp


def test_read_only_health_flag_and_403_on_run_and_replay(ro_env):
    client, tmp = ro_env
    assert _ok(client, "/api/health")["read_only"] is True
    before = {j["job_id"]: j["status"] for j in _ok(client, "/api/queue")["tier0_compute"]}
    for path, body in (
        ("/api/queue/lookback_window_t_sweep/run", {"seed": 42}),
        ("/api/queue/lookback_window_t_sweep/replay", None),
        ("/api/queue/no_such_job/run", {}),  # refused before any job lookup
        ("/api/queue/wise_independent_control_fields/run", {}),
    ):
        r = client.post(path, json=body)
        assert r.status_code == 403, (path, r.text)
        assert r.json()["detail"] == (
            "Server is running in read-only mode; run or replay jobs locally via CLI."
        )
    after = {j["job_id"]: j["status"] for j in _ok(client, "/api/queue")["tier0_compute"]}
    assert after == before and set(after.values()) == {"open"}
    assert not (tmp / "queue_out").exists()
    # reads are unaffected
    assert client.post("/api/compare", json={"channels": ["radio"]}).status_code == 200


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("127.0.0.2", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.10", False),
        ("desk.example.org", False),
    ],
)
def test_loopback_host_classification(host, loopback):
    assert is_loopback_host(host) is loopback


def _serve(monkeypatch, *args):
    import uvicorn
    from typer.testing import CliRunner

    from fermi_desk.cli import app as cli_app

    launched = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: launched.append((app, kw)))
    return CliRunner().invoke(cli_app, ["serve", *args]), launched


def test_serve_refuses_public_bind_without_read_only(monkeypatch):
    result, launched = _serve(monkeypatch, "--host", "0.0.0.0")
    assert result.exit_code != 0 and launched == []
    assert "--read-only" in result.output and "Refusing" in result.output


def test_serve_public_bind_with_read_only_starts_a_read_only_app(monkeypatch):
    result, launched = _serve(monkeypatch, "--host", "0.0.0.0", "--read-only", "--port", "8799")
    assert result.exit_code == 0, result.output
    ((app, kw),) = launched
    assert kw == {"host": "0.0.0.0", "port": 8799}
    assert TestClient(app).post("/api/queue/disk_scale_height_sweep/replay").status_code == 403


def test_eirp_sweep_cpu_estimate_is_240s(client):
    (job,) = [j for j in JOB_DEFINITIONS if j.job_id == "eirp_luminosity_function_sweep"]
    assert job.est_cpu_seconds == 240.0
    t0 = {j["job_id"]: j for j in _ok(client, "/api/queue")["tier0_compute"]}
    assert t0["eirp_luminosity_function_sweep"]["est_cpu_seconds"] == 240.0


def test_reseed_refreshes_stale_cpu_estimate_without_touching_state(tmp_path):
    from fermi_desk.queue.store import JobStore
    from fermi_desk.registry import load_factor_graph

    graph = load_factor_graph(ROOT)
    s = JobStore(tmp_path / "queue.sqlite")
    try:
        s.seed(graph, _jobs())
        s.conn.execute(
            "UPDATE jobs SET est_cpu_seconds=30.0, status='claimed', claimed_by='x' "
            "WHERE job_id='eirp_luminosity_function_sweep'"
        )
        s.conn.commit()
        assert s.seed(graph, _jobs()) == []
        row = s.get("eirp_luminosity_function_sweep")
        assert row["est_cpu_seconds"] == 240.0
        assert row["status"] == "claimed" and row["claimed_by"] == "x"
    finally:
        s.close()
