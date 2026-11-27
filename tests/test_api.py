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

# v0.2.2b: SHA-256 of the promoted data/calibrations/unwise_w3_gate_withheld.json
UNWISE_GATE_SHA256 = "b49e997054726bcfebd5b7ba5ca255a4071c3627af9ce7f871eee2799302f238"

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
    assert h["version"] == "0.2.2"  # v0.2.0-R6: one version string everywhere
    # v0.1.2: live control-sample provenance, read from the verified calibration files.
    assert (h["n_controls"], h["n_calibration"], h["n_withheld"]) == (154, 77, 77)
    assert h["calibration_data_tier"] == "reproduced_archive"
    assert h["fits_gate2_data_tier"] == "reproduced_archive" and len(h["fits_gate2_sha256"]) == 64
    # v0.2.2b: the promoted unsmoothed unWISE W3 fresh-split gate, read from the verified file.
    assert h["unwise_gate_data_tier"] == "reproduced_archive"
    assert h["unwise_gate_sha256"] == UNWISE_GATE_SHA256
    assert (h["unwise_n_controls"], h["unwise_n_calibration"], h["unwise_n_withheld"]) == (
        252,
        126,
        126,
    )


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


def test_infrared_drilldown_exposes_the_promoted_unwise_w3_gate(client):
    """v0.2.2b: the unWISE W3 block and its three-source comparison, tiers never blended."""
    r = client.get("/api/infrared/drilldown?seed=42")
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    u = res["unwise_w3_gate"]
    assert u["data_tier"] == "reproduced_archive" and u["file_sha256"] == UNWISE_GATE_SHA256
    assert len(u["code_git_sha"]) == 40 and u["rule_version"] == "unwise-gate-v1"
    assert (u["n_controls"], u["n_calibration"], u["n_withheld"]) == (252, 126, 126)
    assert u["frozen_before_withheld_fetch"] is True
    assert u["latest_calibration_fetch_utc"] < u["frozen_at_utc"] < u["earliest_withheld_fetch_utc"]
    assert u["conformal_alpha_requested"] == 0.01 and u["conformal_alpha_fallback"] is False
    assert set(u["screens"]) == {
        "centroid_locked",
        "centroid_fresh",
        "image_scalar_chi2",
        "image_gls_chi2",
        "image_scalar_conformal",
        "image_gls_conformal",
        "naive_flux_excess",
    }
    assert u["locked_sigma_sys_arcsec"] == pytest.approx(0.4356, abs=1e-4)
    assert u["fresh_sigma_sys"]["sigma_sys_arcsec"] != u["locked_sigma_sys_arcsec"]

    c = u["comparison"]
    src = c["sources"]
    assert set(src) == {"unwise_w3", "allwise_atlas_gate2", "synthetic_gaussian"}
    assert (
        src["unwise_w3"]["data_tier"]
        == src["allwise_atlas_gate2"]["data_tier"]
        == ("reproduced_archive")
    )
    assert src["synthetic_gaussian"]["data_tier"] == "illustrative_calibration"
    assert c["nominal_w3_fwhm_arcsec"] == src["synthetic_gaussian"]["psf_fwhm_arcsec"] == 6.5
    gate2 = res["real_fits_gate2"]
    assert (
        src["allwise_atlas_gate2"]["psf_fwhm_arcsec"]
        == (gate2["empirical_psf"]["fwhm_gaussian_arcsec"])
    )
    assert src["allwise_atlas_gate2"]["psf_fwhm_arcsec"] == pytest.approx(13.69, abs=5e-3)
    assert src["unwise_w3"]["psf_fwhm_arcsec"] == u["empirical_psf"]["fwhm_gaussian_arcsec"]
    assert src["unwise_w3"]["pixel_scale_arcsec"] == 2.75
    assert len(src["unwise_w3"]["pixel_covariance_lag1_correlation"]) == 2
    assert src["allwise_atlas_gate2"]["pixel_covariance_lag1_correlation"] is None

    image = {row["screen"]: row for row in c["image_screens"]}
    assert list(image) == [
        "image_scalar_chi2",
        "image_gls_chi2",
        "image_scalar_conformal",
        "image_gls_conformal",
    ]
    assert [(r_["statistic"], r_["threshold_rule"]) for r_ in image.values()] == [
        ("scalar", "chi2"),
        ("gls", "chi2"),
        ("scalar", "conformal"),
        ("gls", "conformal"),
    ]
    floors = {row["screen"]: row for row in c["centroid_floors"]}
    assert list(floors) == ["centroid_locked", "centroid_fresh"]
    for name, row in {**image, **floors}.items():
        cell = row["unwise_w3"]
        assert cell == {
            "n_stars": 126,
            "null_false_alarm": u["screens"][name]["null_false_alarm"],
            "blend_survival_within_3p25_arcsec": u["screens"][name][
                "blend_survival_within_3p25_arcsec"
            ],
            "arm_b_false_rejection_pooled": u["screens"][name]["arm_b_false_rejection_pooled"],
        }
        fa = cell["null_false_alarm"]
        lo, hi = fa["clopper_pearson_95"]  # Clopper-Pearson 95% bounds on the withheld null FA
        assert fa["n"] == 126 and 0.0 <= lo <= fa["rate"] <= hi <= 1.0
        assert row["calibration_false_alarm_in_sample"]["n"] == 126
    # Gate #2 and the synthetic grid appear only where they applied the same rule.
    like = image["image_scalar_chi2"]
    assert like["counterpart"] == "image"
    assert (
        like["allwise_atlas_gate2"]["null_false_alarm"]
        == gate2["screens"]["image"]["null_false_alarm"]
    )
    assert like["allwise_atlas_gate2"]["null_false_alarm"]["k"] == 5  # Gate #2: 5/77
    assert like["allwise_atlas_gate2"]["blend_survival_within_3p25_arcsec"] == pytest.approx(
        0.934, abs=5e-4
    )
    assert like["synthetic_gaussian"]["arm_b_false_rejection_pooled"] is None
    assert (
        floors["centroid_locked"]["allwise_atlas_gate2"]["null_false_alarm"]
        == (gate2["screens"]["centroid"]["null_false_alarm"])
    )
    for name in ("image_gls_chi2", "image_scalar_conformal", "image_gls_conformal"):
        assert image[name]["allwise_atlas_gate2"] is None
        assert image[name]["synthetic_gaussian"] is None and image[name]["counterpart"] is None
    assert floors["centroid_fresh"]["allwise_atlas_gate2"] is None
    assert floors["centroid_locked"]["sigma_sys_source"] == "locked"
    assert floors["centroid_fresh"]["sigma_sys_arcsec"] == u["fresh_sigma_sys"]["sigma_sys_arcsec"]
    for name in ("image_scalar_conformal", "image_gls_conformal"):
        assert image[name]["alpha_guaranteed"] == pytest.approx(1 / 127)

    prov = c["provenance"]
    assert prov["file_sha256"] == UNWISE_GATE_SHA256 and prov["n_gate1_star_overlap"] == 0
    assert (prov["n_fresh_stars"], prov["n_gate1_stars"], prov["n_patches"]) == (252, 154, 16)
    assert prov["disjoint_from_gate1_sha256"].startswith("2da2a234")
    assert prov["min_abs_b_deg"] > 30.0
    assert prov["frozen_at_utc"] < prov["earliest_withheld_fetch_utc"]
    # Ruling v0.2.1-R4: no Gaia id anywhere in the block is a JSON number.
    for key, value in _walk(u):
        if key == "gaia_dr3_source_id":
            assert isinstance(value, str) and value.isdigit(), value
        if isinstance(value, int) and not isinstance(value, bool):
            assert abs(value) <= MAX_SAFE_JSON_INT, (key, value)
    import re

    assert re.search(r'"gaia_dr3_source_id":\s*[0-9-]', r.text) is None
    assert re.search(r'"star_ids":\s*\[\s*[0-9]', r.text) is None
    # The synthetic grid keeps its own tier; the IR drill-down tier is unchanged.
    assert res["data_tier"] == "illustrative_calibration"
    assert res["blend_survival_side_by_side"]["real_fits"]["data_tier"] == "reproduced_archive"


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
    # v0.2.1c (ruling v0.2.1-R5): the catalog declination sign-flip audit is in the feed,
    # next to HIP 1692 and the 16 beam-sharing pairs, citing its claim file.
    flip = checks["catalog_dec_sign_audit"]
    assert (flip["status"], flip["severity"]) == ("finding", "warning")
    assert any("Isaacson et al. 2017" in c for c in flip["citations"])
    d = flip["details"]
    assert d["n_affected_targets"] == 14 and len(d["affected_targets"]) == 14
    assert d["locked_catalog"]["sha256"].startswith("c299d61d")
    assert d["locked_catalog"]["modified_by_this_audit"] is False
    assert d["band_census"]["n_targets_in_mirror_band"] == 0
    ids = [t["gaia_dr3_source_id"] for t in d["affected_targets"]]
    assert sum(i is None for i in ids) == 3  # three targets have no Gaia DR3 id in the catalog
    assert all(i is None or (isinstance(i, str) and i.isdigit()) for i in ids)
    assert {c["target_name"] for c in d["northern_controls"]} == {"HIP 59608", "HIP 92283"}
    (sec,) = d["secondary_discrepancies"]
    assert len(sec["records"]) == 3 and sec["inherited_by_locked_catalog"] is False


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
    # v0.2.1b: executed live and promoted, so the job is completed BY its pinned artifact;
    # the registered synthetic-proxy prediction stays in expected_eig_bits.
    pin = _ok(client, "/api/health")["background_summary_sha256"]
    assert bg["status"] == "completed" and bg["output_sha256"] == pin and len(pin) == 64
    assert len(bg["code_git_sha"]) == 40 and bg["commands"]["report"].endswith("background-field")
    assert bg["expected_eig_bits"] == pytest.approx(0.0145226, rel=1e-4)
    ev = bg["linked_evidence"]
    assert ev["kind"] == "background_field_eig" and ev["summary_sha256"] == pin
    assert ev["summary_data_tier"] == "reproduced_archive" and ev["n_background_stars"] == 119947
    assert {r["radio_mode"] for r in ev["comparison"]} == {
        "distance_scaled_eirp",
        "flat_probability",
    }
    assert "692" in ev["statement"] and "1,327" in ev["statement"] and "288,315" in ev["statement"]
    assert t1[0]["commands"]["report"] is None  # the WISE job has no promoted large-N artifact
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
        # v0.2.1b: the promoted, pinned artifact completes the job; its registered EIG
        # prediction is untouched (tests/test_background_field.py covers the transitions).
        assert bg["status"] == "completed" and len(bg["output_sha256"]) == 64
        assert bg["expected_eig_bits"] == pytest.approx(0.0145226, rel=1e-4)
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
    assert declared == "0.2.2"
    assert fermi_desk.__version__ == declared
    assert installed_version("fermi-desk") == declared, "stale install: run `uv sync`"
    assert client.app.version == declared
    assert _ok(client, "/openapi.json")["info"]["version"] == declared
    assert _ok(client, "/api/health")["version"] == declared
    # The version is not written anywhere else in the package sources.
    for path in (ROOT / "src" / "fermi_desk").rglob("*.py"):
        text = path.read_text("utf-8")
        assert 'version="0.' not in text and '__version__ = "' not in text, path


# ----------------------------- v0.2.1b: promoted background field, R2 guard, string Gaia ids
MAX_SAFE_JSON_INT = 2**53 - 1
BG_JOB = "wlodarczyk_sroka_288k_background_extension"


def _walk(obj, key=None):
    """Yield (key, scalar) for every scalar in a JSON payload (lists inherit their key)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)
    else:
        yield key, obj


def test_health_exposes_the_promoted_background_sha_and_tier(client):
    from fermi_desk.registry import load_factor_graph

    h = _ok(client, "/api/health")
    survey = load_factor_graph(ROOT).surveys["bl_gbt_lband_background_gaia_dr3"]
    assert h["background_survey_id"] == survey.id
    assert h["background_summary_sha256"] == survey.background_summary_sha256
    assert h["background_summary_data_tier"] == "reproduced_archive"
    assert len(h["background_summary_code_git_sha"]) == 40
    assert (h["n_background_stars"], h["n_suspected_unmatched_primary"]) == (119947, 49)
    # v0.2.1c (ruling v0.2.1-R5): the pinned pointing-correction overlay and its state.
    assert h["pointing_corrections_path"] == survey.pointing_corrections_path
    assert h["pointing_corrections_sha256"] == survey.pointing_corrections_sha256
    assert (h["n_corrected_pointings"], h["n_uncorrected_flipped_cones"]) == (14, 0)
    assert h["status"] == "ok" and h["validation_checks_total"] >= 41
    assert h["catalog_sha256"].startswith("c299d61d")  # the locked primaries are untouched


def test_radio_drilldown_exposes_the_background_field_extension(client):
    r = client.get("/api/radio/drilldown")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_targets"] == 692  # the primary drill-down is unchanged
    bf = d["background_field"]
    assert bf["promoted"] is True
    prov = bf["provenance"]
    health = _ok(client, "/api/health")
    assert prov["file_sha256"] == health["background_summary_sha256"]
    assert prov["data_tier"] == "reproduced_archive" and len(prov["code_git_sha"]) == 40
    assert all(len(s) == 40 for s in prov["cone_cache_code_git_shas"])
    assert (prov["n_pointings"], prov["beam_fwhm_arcmin"], prov["cone_radius_arcmin"]) == (
        692,
        8.4,
        4.2,
    )
    assert (prov["n_background_stars"], prov["n_cells"]) == (119947, 46888)
    assert prov["query_window_utc"][0] < prov["query_window_utc"][1] < prov["assembled_at_utc"]
    # Distance cuts and de-duplication counters, including the reported suspected primaries.
    cuts, mp, da = prov["distance_cuts"], prov["merge_provenance"], prov["dedup_audit"]
    assert (cuts["n_cone_sightings_raw"], cuts["n_dropped_beyond_d_max"]) == (258927, 132)
    assert (mp["n_dedup_gaia_id"], mp["n_dedup_angular_fallback"]) == (4281, 106)
    assert (mp["n_multi_beam_resolved"], mp["n_background_absorbed_by_primary"]) == (3764, 647)
    assert da["n_duplicate_gaia_ids_vs_primary"] == 0
    assert (da["n_suspected_unmatched_primary"], da["n_suspected_unmatched_primary_stars"]) == (
        49,
        47,
    )
    assert "ruling v0.2.1-R3" in da["suspected_unmatched_primary_policy"]
    assert len(bf["suspected_unmatched_primary"]) == 49
    # Scope contrast against the published catalogue.
    pub, got = prov["scope"]["published"], prov["scope"]["retrieved"]
    assert (pub["n_pointings"], pub["n_stars_total"], pub["gaia_release"]) == (
        1327,
        288315,
        "Gaia DR2",
    )
    assert pub["n_stars_gbt_l_band"]["verified_against_paper"] is False
    assert (got["n_pointings"], got["n_background_stars"], got["gaia_release"]) == (
        692,
        119947,
        "Gaia DR3",
    )
    # Realized vs predicted EIG, both radio modes, never labelled as verified.
    rows = {x["radio_mode"]: x for x in bf["eig"]["comparison"]}
    assert rows["distance_scaled_eirp"]["label"] == "physical"
    assert rows["flat_probability"]["label"] == "unphysical"
    assert rows["distance_scaled_eirp"]["predicted_eig_bits"] == pytest.approx(0.01452, abs=5e-6)
    assert rows["distance_scaled_eirp"]["realized_eig_bits"] == pytest.approx(0.00234721, rel=1e-5)
    assert rows["flat_probability"]["predicted_eig_bits"] == pytest.approx(0.827, abs=5e-4)
    assert rows["flat_probability"]["realized_eig_bits"] == pytest.approx(0.650681, rel=1e-5)
    for row in rows.values():
        assert row["data_tiers"] == ["illustrative_calibration", "reproduced_archive"]
    # Harris-FKG / Frechet bounds, shell summary, off-axis attenuation, pointing-centre audit.
    assert len(bf["bounds"]) == 4
    for b in bf["bounds"]:
        assert b["p_zero_markov_lower"] <= b["p_zero_fkg_lower"] <= b["p_zero_frechet_upper"]
    assert "LOWER bound" in bf["correlation_assumption"]
    assert bf["shells"]["n_stars"] == 119947 and bf["shells"]["eirp_min_cumulative_by_decade"]
    assert bf["attenuation"]["median_eirp_min_ratio_background_over_primary"] > 1e3
    # Ruling v0.2.1-R5: the 14 Isaacson -10 deg erratum cones are corrected by the pinned
    # overlay; none is left uncorrected; the locked catalog is untouched.
    audit, pc = bf["pointing_centre_audit"], prov["pointing_corrections"]
    assert audit["uncorrected_flipped_cones"]["n"] == 0
    assert audit["corrected_via_overlay"]["n"] == 14
    assert audit["corrected_via_overlay"]["overlay_sha256"] == health["pointing_corrections_sha256"]
    assert audit["corrected_via_overlay"]["locked_catalog_modified"] is False
    assert (pc["n_corrected_pointings"], pc["n_uncorrected_pointings"]) == (14, 0)
    assert pc["pointing_corrections_sha256"] == health["pointing_corrections_sha256"]
    assert pc["locked_catalog_sha256"] == health["catalog_sha256"]
    assert len(pc["corrected_pointings"]) == 14
    assert all(
        c["cone_dec_deg"] < -10.0 < 10.0 <= c["catalog_dec_deg"] for c in pc["corrected_pointings"]
    )
    dense = audit["densest_band_of_unfound_primaries"]
    assert dense["n_no_counterpart"] == 3 and dense["sign_flip_signature"] is False
    assert bf["commands"]["report"] == "uv run fermi-desk background-field"
    # Ruling v0.2.1-R4: every Gaia id in the response is a string, never a JSON number.
    ids = [v for k, v in _walk(bf) if k == "gaia_dr3_source_id"]
    assert len(ids) == 49 and all(isinstance(v, str) and v.isdigit() for v in ids)
    assert any(int(v) > MAX_SAFE_JSON_INT for v in ids)
    import re

    assert re.search(r'"gaia_dr3_source_id":\s*[0-9-]', r.text) is None


def test_no_api_payload_carries_a_gaia_id_or_integer_javascript_would_round(client):
    """Ruling v0.2.1-R4 across the API, including ids read from the locked v0.1.x
    artifacts (which keep their integer form on disk and are stringified at the boundary)."""
    n_ids = 0
    for path in (
        "/api/health",
        "/api/registry",
        "/api/radio/drilldown",
        "/api/infrared/drilldown?seed=42",
        "/api/adversary?seed=42",
        "/api/queue",
        "/api/runs",
    ):
        for key, value in _walk(_ok(client, path)):
            if isinstance(value, bool) or not isinstance(value, int):
                if key == "gaia_dr3_source_id":
                    # null = the catalog has no Gaia DR3 id for that target (ruling v0.2.1-R5:
                    # "a quoted string or null"); an id that exists is always a decimal string.
                    assert value is None or (isinstance(value, str) and value.isdigit()), (
                        path,
                        value,
                    )
                    n_ids += value is not None
                continue
            assert abs(value) <= MAX_SAFE_JSON_INT, (path, key, value)
            assert key != "gaia_dr3_source_id", (path, value)
    assert n_ids >= 49 + 2  # background suspects + the Hephaistos D / E anchors


def test_json_safe_stringifies_source_ids_and_unsafe_integers():
    import numpy as np

    big = 6898272528885931234
    out = json_safe(
        {
            "gaia_dr3_source_id": big,
            "nested": [{"gaia_dr3_source_id": np.int64(5)}],  # ALWAYS a string under an id key
            "star_ids": [big, 7],
            "psf_star_ids": np.array([big]),
            "rng_seed": 42,
            "n_stars": np.int64(119947),
            "survey_id": "bl",
            "flag": True,
            "unsafe": 2**53,
            "safe": 2**53 - 1,
            "negative_unsafe": -(2**60),
        }
    )
    assert out == {
        "gaia_dr3_source_id": str(big),
        "nested": [{"gaia_dr3_source_id": "5"}],
        "star_ids": [str(big), "7"],
        "psf_star_ids": [str(big)],
        "rng_seed": 42,
        "n_stars": 119947,
        "survey_id": "bl",
        "flag": True,
        "unsafe": str(2**53),
        "safe": 2**53 - 1,
        "negative_unsafe": str(-(2**60)),
    }
    assert int(float(big)) != big  # what a JavaScript number would have done to it


def test_compare_union_flat_holds_radio_out_and_carries_background_provenance(client):
    base = {
        "channels": ["radio"],
        "seed": 7,
        "pointing_merge_policy": "union_targets_by_pointing",
        "radio_mode": "flat_probability",
    }
    d = _post(client, base)
    rec = d["receipt"]
    assert d["headline_policy"] == "verified_channels_only"
    (note,) = d["headline_exclusion_notes"]
    assert note["channel"] == "radio" and note["rulings"] == ["v0.2.1-R2"]
    assert note["held_out_of_reported_headline"] is True
    assert "d^2 / B(theta)" in note["reasons"][0]
    assert note["merged_background"]["n_background_stars"] == 119947
    head = rec["results"]["headline"]["headline"]
    assert head["terms"] == ["earth_unreached"]
    assert "radio" in rec["results"]["headline"]["log10_bf_by_term"]  # still in the table
    assert abs(d["headline_log10_bf"]) < 1e-9 and note["log10_bf_term"] < -0.5
    bf = rec["background_field"]
    assert bf["data_tier"] == "reproduced_archive" and bf["n_background_stars"] == 119947
    assert bf["dedup_audit"]["n_suspected_unmatched_primary"] == 49
    assert rec["merge_provenance"]["n_merged_targets"] == 692 + 119947
    assert rec["held_out_extension_surveys"] == []
    # Opt-in: the term enters the headline under the loud policy.
    loud = _post(client, {**base, "include_illustrative": True})
    assert loud["headline_policy"] == "including_illustrative_channels"
    assert loud["headline_log10_bf"] == pytest.approx(note["log10_bf_term"], abs=1e-6)
    assert loud["headline_exclusion_notes"][0]["held_out_of_reported_headline"] is False
    # Default policy: 692 primaries only, the extension listed as held out.
    plain = _post(client, {"channels": ["radio"], "seed": 7})["receipt"]
    assert plain["background_field"] is None and plain["merge_provenance"] is None
    assert [h["survey_id"] for h in plain["held_out_extension_surveys"]] == [
        "bl_gbt_lband_background_gaia_dr3"
    ]
    assert plain["results"]["headline"]["headline"]["terms"] == ["earth_unreached", "radio"]


def test_read_only_server_shows_the_promoted_evidence_without_writing(ro_env):
    client, _ = ro_env
    q = _ok(client, "/api/queue")
    bg = next(j for j in q["tier1_archival"] if j["job_id"] == BG_JOB)
    assert bg["linked_evidence"]["kind"] == "background_field_eig"
    assert bg["status"] == "completed"  # seeded by this server from the pinned artifact
