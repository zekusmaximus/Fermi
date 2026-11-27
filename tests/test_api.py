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
        compare_n_importance=1024,
        lookback_grid_points=4,
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
    assert h["receipt_schema_version"] == 5 and h["git_sha"]
    assert h["version"] == "0.2.1"  # v0.2.0-R6: one version string everywhere
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
    # v0.2.1: the merge-policy audit runs with the feed and cites its sources.
    overlap = checks["pointing_overlap"]
    assert overlap["status"] == "pass" and overlap["citations"]
    assert overlap["details"]["keep_first_refused_by_registry"] is True
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
    # v0.2.1: the background-extension job points at its opt-in script, offline mode only.
    bg = next(j for j in t1 if j["job_id"] == "wlodarczyk_sroka_288k_background_extension")
    assert bg["archival_script"] == "scripts/fetch_wlodarczyk_sroka_288k.py"
    assert bg["commands"]["script"].endswith("fetch_wlodarczyk_sroka_288k.py --dry-run")
    assert bg["status"] == "open" and bg["output_sha256"] is None  # not executed in v0.2.1a
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
        # v0.2.1: a queue seeded before the fetch script existed gets its pointer refreshed.
        s.conn.execute(
            "UPDATE jobs SET archival_script='(not yet written)' "
            "WHERE job_id='wlodarczyk_sroka_288k_background_extension'"
        )
        s.conn.commit()
        assert s.seed(graph, _jobs()) == []
        row = s.get("eirp_luminosity_function_sweep")
        assert row["est_cpu_seconds"] == 240.0
        assert row["status"] == "claimed" and row["claimed_by"] == "x"
        bg = s.get("wlodarczyk_sroka_288k_background_extension")
        assert bg["archival_script"] == "scripts/fetch_wlodarczyk_sroka_288k.py"
        assert bg["status"] == "open" and bg["output_sha256"] is None
    finally:
        s.close()


# ------------------------------------------- v0.2.0: finite disk & lookback marginalisation
FINITE = {"geometry_mode": "finite_disk_3d", "channels": ["radio"]}


def _post(client, body):
    r = client.post("/api/compare", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_compare_default_response_is_intact_and_echoes_new_defaults(client):
    """Landing defaults are the v0.1.2 run: thin disk, fixed T, verified headline."""
    d = _post(client, {})
    assert d["request"] == {
        "geometry_mode": "thin_disk_2d",
        "radio_mode": "flat_probability",
        "condition_on_earth_unreached": True,
        "channels": ["radio", "infrared", "solar_system"],
        "include_illustrative": False,
        "seed": 42,
        "disk_profile": "uniform",
        "marginalize_lookback": False,
        "pointing_merge_policy": "reject_duplicate_pointing_group",
    }
    rec = d["receipt"]
    assert rec["geometry_mode"] == "thin_disk_2d" and rec["receipt_schema_version"] == 5
    assert rec["pointing_merge_policy"] == "reject_duplicate_pointing_group"
    assert rec["merge_provenance"] is None  # the default policy never merges
    assert d["headline_policy"] == "verified_channels_only"
    assert rec["results"]["headline"]["headline"]["terms"] == ["earth_unreached", "radio"]
    assert "lookback_marginalization" not in rec and "disk_profile" not in rec
    assert "lookback_marginalization" not in rec["results"]
    fast = rec["results"]["hypotheses"]["rare_fast_expanders"]["expansion"]
    assert fast["local_approximation_valid"] is False and "local_approx" not in fast
    assert fast["t_valid_kyr"] == pytest.approx(52.185, abs=5e-4)
    # explicit defaults are the same request (same cache entry, same receipt)
    again = _post(client, {"disk_profile": "uniform", "marginalize_lookback": False})
    assert again["cached"] is True and again["receipt"] == rec


def test_compare_finite_disk_clears_local_approx_violation(client):
    d = _post(client, FINITE)
    assert d["request"]["geometry_mode"] == "finite_disk_3d"
    assert d["request"]["disk_profile"] == "uniform"
    rec = d["receipt"]
    assert rec["geometry_mode"] == "finite_disk_3d" and rec["disk_profile"] == "uniform"
    assert d["headline_policy"] == "verified_channels_only"
    hyps = rec["results"]["hypotheses"]
    fast = hyps["rare_fast_expanders"]["expansion"]
    assert fast["local_approximation_valid"] is True
    assert fast["local_approx"] == "valid_finite_galaxy"
    assert (
        fast["t_valid_kyr"] is None and "finite Galactic disk" in fast["local_validity_criterion"]
    )
    assert fast["finite_disk"]["galactic_disk_radius_kpc"] == 15.0
    assert fast["finite_disk"]["solar_galactocentric_distance_kpc"] == 8.178
    # same hypothesis under the infinite thick disk: still VIOLATED, and far more reachable
    thick = _post(client, {"geometry_mode": "thick_disk_3d", "channels": ["radio"]})["receipt"]
    t_fast = thick["results"]["hypotheses"]["rare_fast_expanders"]["expansion"]
    assert t_fast["local_approximation_valid"] is False
    assert t_fast["t_valid_kyr"] == pytest.approx(52.185, abs=5e-4)
    assert fast["p_earth_reached"] < 0.05 * t_fast["p_earth_reached"]
    # finite-disk reach numbers carry their Monte Carlo error
    fd = hyps["frequent_slow_expanders"]["radio"]["finite_disk_reach"]
    assert fd["n_importance_samples"] == 1024 and fd["expected_reached_se"] > 0
    assert {"zoo_hypothesis", "dark_forest"}.isdisjoint(hyps)


def test_compare_disk_profile_is_applied_to_finite_disk_and_cache_keyed(client):
    uni = _post(client, FINITE)
    exp = _post(client, {**FINITE, "disk_profile": "exponential"})
    assert exp["cached"] is False and exp["receipt"]["disk_profile"] == "exponential"
    p = {
        k: v["receipt"]["results"]["hypotheses"]["rare_fast_expanders"]["expansion"]
        for k, v in (("uniform", uni), ("exponential", exp))
    }
    assert p["exponential"]["finite_disk"]["disk_profile"] == "exponential"
    assert p["exponential"]["finite_disk"]["radial_scale_length_kpc"] == 2.6
    assert p["exponential"]["p_earth_reached"] != p["uniform"]["p_earth_reached"]
    assert p["exponential"]["local_approx"] == "valid_finite_galaxy"
    # exact cache keying on the full request: each profile is its own entry
    assert _post(client, {**FINITE, "disk_profile": "exponential"})["cached"] is True
    assert _post(client, FINITE)["cached"] is True
    # validated but not applied outside finite_disk_3d: identical results to the default
    thin = _post(client, {"channels": ["radio"]})
    thin_exp = _post(client, {"channels": ["radio"], "disk_profile": "exponential"})
    assert thin_exp["request"]["disk_profile"] == "exponential"
    assert thin_exp["receipt"]["results"] == thin["receipt"]["results"]
    assert thin_exp["receipt"]["config"]["disk_profile"] == "uniform"


def test_compare_marginalize_lookback(client):
    fixed = _post(client, {"channels": ["radio"]})
    marg = _post(client, {"channels": ["radio"], "marginalize_lookback": True})
    assert marg["cached"] is False and marg["request"]["marginalize_lookback"] is True
    rec = marg["receipt"]
    lm = rec["lookback_marginalization"]
    assert lm["parameter"] == "lookback_window_gyr"
    assert lm["prior"]["type"] == "log_uniform"
    assert lm["prior"]["effective_bounds_gyr"] == [5.0e-5, 1.0e-2]
    assert lm["quadrature"]["n_points"] == 4 == len(lm["quadrature"]["weights"])
    assert sum(lm["quadrature"]["weights"]) == pytest.approx(1.0, abs=1e-12)
    assert rec["config"]["marginalize_lookback"] is True
    assert marg["headline_policy"] == "verified_channels_only"
    head = rec["results"]["headline"]["headline"]
    assert head["terms"] == ["earth_unreached", "radio"]
    assert marg["headline_log10_bf"] != fixed["headline_log10_bf"]
    fs = rec["results"]["hypotheses"]["frequent_slow_expanders"]
    assert len(fs["lookback_marginalization"]["per_node"]) == 4
    assert fs["parameters"]["lookback_window_gyr"]["value"] is None
    # the infinite disk is still flagged: rare/fast is VIOLATED on part of the grid
    fast = rec["results"]["hypotheses"]["rare_fast_expanders"]["expansion"]
    assert fast["local_approximation_valid"] is False and fast["local_approx"] == "VIOLATED"
    assert fast["local_approx_nodes_violated"] >= 1
    assert _post(client, {"channels": ["radio"], "marginalize_lookback": True})["cached"] is True
    # combined with the finite disk the violation clears on every node
    both = _post(client, {**FINITE, "marginalize_lookback": True})["receipt"]
    b_fast = both["results"]["hypotheses"]["rare_fast_expanders"]["expansion"]
    assert b_fast["local_approx"] == "valid_finite_galaxy" and b_fast["t_valid_kyr"] is None
    assert b_fast["local_approx_nodes_violated"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"geometry_mode": "finite_disk"},
        {"geometry_mode": "finite_disk_2d"},
        {"geometry_mode": "FINITE_DISK_3D"},
        {"geometry_mode": "finite_disk_3d", "disk_profile": "sech2"},
        {"disk_profile": "Uniform Slab"},
        {"disk_profile": None},
        {"disk_profile": 1},
        {"marginalize_lookback": "yes"},
        {"marginalize_lookback": 1},
        {"marginalize_lookback": None},
        {"geometry_mode": "finite_disk_3d", "lookback_grid_points": 64},
        {"geometry_mode": "finite_disk_3d", "n_importance": 10},
    ],
)
def test_compare_rejects_invalid_v020_requests(client, body):
    assert client.post("/api/compare", json=body).status_code == 422


def test_health_and_registry_expose_v020_parameters_offline(client):
    h = _ok(client, "/api/health")
    assert h["status"] == "ok" and h["validation_checks_passed"] == h["validation_checks_total"]
    assert h["validation_checks_total"] >= 38 and h["receipt_schema_version"] == 5
    r = _ok(client, "/api/registry")
    params = {p["id"]: p for p in r["parameters"]}
    assert params["galactic_disk_radius_kpc"]["default"] == 15.0
    assert params["solar_galactocentric_radius_kpc"]["default"] == 8.178
    assert params["galactic_radial_scale_length_kpc"]["default"] == 2.6
    for pid in ("galactic_disk_radius_kpc", "galactic_radial_scale_length_kpc"):
        assert params[pid]["unit"] == "kpc" and params[pid]["geometry_modes"] == ["finite_disk_3d"]
    assert params["lookback_window_gyr"]["marginalization_prior_bounds"] == [5.0e-5, 1.0e-2]
    assert params["disk_scale_height_pc"]["geometry_modes"] is None


# ------------------------------- v0.2.1: merge policy at the API boundary, version sync
def test_compare_refuses_keep_first_with_an_explicit_domain_error(client):
    r = client.post("/api/compare", json={"pointing_merge_policy": "keep_first"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "MergePolicyError"
    assert "keep_first" in body["detail"] and "load order" in body["detail"]
    for bad in ("keep_last", "union", "", None, 1):
        assert client.post("/api/compare", json={"pointing_merge_policy": bad}).status_code == 422


def test_compare_accepts_union_policy_and_reports_no_merge_without_an_extension(client):
    """union_targets_by_pointing is a valid request. While no survey declares a parent
    pointing (v0.2.1a) it has nothing to merge and the run is the single-survey run
    (identical results, merge_provenance null); once v0.2.1b pins a background extension
    the same request must carry a merge_provenance instead."""
    from fermi_desk.registry import load_factor_graph

    base = {"channels": ["radio"], "seed": 7}
    plain = _post(client, base)["receipt"]
    d = _post(client, {**base, "pointing_merge_policy": "union_targets_by_pointing"})
    rec = d["receipt"]
    assert d["request"]["pointing_merge_policy"] == "union_targets_by_pointing"
    assert rec["pointing_merge_policy"] == "union_targets_by_pointing"
    assert rec["held_out_extension_surveys"] == []  # union never holds an extension out
    if any(s.is_pointing_extension for s in load_factor_graph(ROOT).surveys.values()):
        assert rec["merge_provenance"]["policy"] == "union_targets_by_pointing"
    else:
        assert rec["merge_provenance"] is None
        assert rec["results"] == plain["results"]
        assert [s["id"] for s in rec["active_surveys"]] == ["bl_enriquez_2017_692"]


def test_version_is_synchronised_across_pyproject_package_and_api(client, env):
    """v0.2.0-R6: pyproject.toml is the single source; the package, the installed
    distribution metadata, the FastAPI app, /openapi.json and /api/health all agree."""
    import tomllib
    from importlib.metadata import version as installed_version

    import fermi_desk

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    declared = pyproject["project"]["version"]
    assert declared == "0.2.1"
    assert fermi_desk.__version__ == declared
    assert installed_version("fermi-desk") == declared, "stale install: run `uv sync`"
    assert client.app.version == declared
    assert _ok(client, "/openapi.json")["info"]["version"] == declared
    assert _ok(client, "/api/health")["version"] == declared
    # The version is not written anywhere else in the package sources.
    for path in (ROOT / "src" / "fermi_desk").rglob("*.py"):
        text = path.read_text("utf-8")
        assert 'version="0.' not in text and '__version__ = "' not in text, path
