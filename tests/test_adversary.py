"""Adversary runner (docs/MILESTONE_2.md B2)."""

from __future__ import annotations

import json

import astropy.units as u
import numpy as np
import pytest
import yaml

from fermi_desk.adversary.checks import (
    HIP1692_CLAIM,
    audit_distance_revision,
    audit_duplicate_pointings,
    audit_hip1692_distance_revision,
    audit_illustrative_tiers,
    audit_local_approximation,
    audit_scaling_symmetry,
    run_adversary,
)
from fermi_desk.inference.engine import (
    RunConfig,
    build_context,
    expansion_model,
    resolve_parameters,
)
from fermi_desk.registry import load_factor_graph

PAIR = ("frequent_slow_expanders", "rare_fast_expanders")


@pytest.fixture(scope="module")
def hip(graph):
    return audit_hip1692_distance_revision(graph, PAIR, seed=42, n_mc=300)


def test_hip1692_dual_verification_exceeds_1000x_under_both_distances(hip):
    assert hip.status == "pass"
    d = hip.details["distances"]
    legacy = d["isaacson_2017_legacy"]["distance_pc"]
    for label in ("gaia_dr3_inverse_parallax", "gaia_dr2_bailer_jones_2018_r_est"):
        v = d[label]
        # factor is computed from EIRP_min in watts, and equals (d / d_legacy)^2
        assert v["eirp_factor_vs_legacy"] == pytest.approx((v["distance_pc"] / legacy) ** 2)
        assert v["eirp_factor_vs_legacy"] > hip.details["threshold"] == 1000.0
        assert v["exceeds_threshold"]
    assert d["gaia_dr3_inverse_parallax"]["eirp_factor_vs_legacy"] == pytest.approx(1176, abs=1)
    assert d["gaia_dr2_bailer_jones_2018_r_est"]["eirp_factor_vs_legacy"] == pytest.approx(
        1071, abs=1
    )
    assert hip.details["canonical_label"] == "gaia_dr3_inverse_parallax"
    assert hip.details["catalog_dist_pc"] == pytest.approx(788.7067)


def test_hip1692_eirp_in_watts(hip):
    d = hip.details["distances"]["gaia_dr3_inverse_parallax"]
    f_min = hip.details["f_min_w_m2"] * u.W / u.m**2
    expect = (4 * np.pi * (d["distance_pc"] * u.pc) ** 2 * f_min).to(u.W)
    assert d["eirp_min_w"] == pytest.approx(expect.value)


def test_legacy_distance_is_more_constraining_for_both_hypotheses(hip):
    for hid in PAIR:
        by = hip.details["radio_likelihood_by_distance"][hid]
        assert by["isaacson_2017_legacy"]["delta_log10_lr_fkg_vs_canonical"] < 0
        assert by["gaia_dr3_inverse_parallax"]["delta_log10_lr_fkg_vs_canonical"] == 0.0
        assert (
            by["isaacson_2017_legacy"]["target_p_detect_given_reached"]
            > by["gaia_dr2_bailer_jones_2018_r_est"]["target_p_detect_given_reached"]
            > by["gaia_dr3_inverse_parallax"]["target_p_detect_given_reached"]
        )


def test_distance_audit_fails_when_threshold_is_not_met(data_copy):
    path = data_copy / HIP1692_CLAIM
    doc = yaml.safe_load(path.read_text())
    doc["min_eirp_factor_threshold"] = 1100.0  # DR2 anchor (1071x) no longer clears it
    path.write_text(yaml.safe_dump(doc))
    f = audit_distance_revision(
        load_factor_graph(data_copy), HIP1692_CLAIM, PAIR[:1], seed=1, n_mc=50
    )
    assert f.status == "fail"
    assert not f.details["distances"]["gaia_dr2_bailer_jones_2018_r_est"]["exceeds_threshold"]


def test_duplicate_pointing_detector(graph, data_copy):
    f = audit_duplicate_pointings(graph)
    assert f.details["detector_self_test_passed"]
    assert f.details["collisions"] == [] and f.details["duplicate_target_names"] == []
    assert f.status == "finding"  # beam-sharing pairs are reported, not hidden
    names = {(p["a"], p["b"]) for p in f.details["beam_sharing_pairs"]}
    assert ("GJ 1245AC", "GJ 1245B") in names
    # Inject a second survey on the BL pointing group: the audit must fail.
    src = data_copy / "data" / "surveys" / "bl_enriquez_2017_692.yaml"
    doc = yaml.safe_load(src.read_text())
    doc["id"] = "bl_duplicate"
    (src.parent / "bl_duplicate.yaml").write_text(yaml.safe_dump(doc))
    bad = audit_duplicate_pointings(load_factor_graph(data_copy))
    assert bad.status == "fail" and bad.details["collisions"]


def test_symmetry_validity_and_tier_audits(graph):
    sym = audit_scaling_symmetry(graph, PAIR)
    assert sym.status == "pass"
    assert sym.details["thin_disk_2d"]["relative_difference"] < 1e-12
    loc = audit_local_approximation(graph, sorted(graph.hypotheses))
    bad = {r["hypothesis_id"] for r in loc.details["rows"] if not r["valid"]}
    assert "rare_fast_expanders" in bad and "frequent_slow_expanders" not in bad
    assert audit_illustrative_tiers(graph).status == "pass"


def test_high_latitude_target_reachable_by_fast_fronts_in_2d(graph):
    """Regression for the M2 adversary finding: the 2D projection preserves true distance."""
    cfg = RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=0)
    m = expansion_model(resolve_parameters(graph, graph.hypotheses["rare_fast_expanders"], 33), cfg)
    polar = np.array([[0.0, 1.0, 22.9]]) * u.pc  # nearly at the Galactic pole, d ~ 23 pc
    inplane = np.array([[0.0, np.hypot(1.0, 22.9), 0.0]]) * u.pc  # same true distance
    q_polar = m.reach_probabilities(polar, True)[0]
    q_plane = m.reach_probabilities(inplane, True)[0]
    assert q_polar > 0
    assert q_polar == pytest.approx(q_plane, rel=1e-9)


def test_adversary_receipt_is_seed_deterministic(graph):
    a = run_adversary(graph, seed=9, pair=PAIR, n_mc=100)
    b = run_adversary(graph, seed=9, pair=PAIR, n_mc=100)
    assert a["results_sha256"] == b["results_sha256"]
    assert not any(f["status"] == "fail" for f in a["results"]["findings"])


def test_build_context_targets_carry_coordinates(graph):
    ctx = build_context(
        graph, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=0)
    )
    assert ctx.targets.ra_deg is not None and len(ctx.targets.ra_deg) == 692


def test_beam_sharing_pairs_are_structured_receipt_records(graph):
    """Milestone 3's UI reads these records straight from the adversary receipt."""
    rec = run_adversary(graph, seed=9, pair=PAIR, n_mc=100)
    dup = next(f for f in rec["results"]["findings"] if f["check"] == "duplicate_pointings")
    pairs = json.loads(json.dumps(dup["details"]["beam_sharing_pairs"]))  # JSON round trip
    assert dup["details"]["n_beam_sharing_pairs"] == len(pairs) == 16
    keys = set(dup["details"]["beam_sharing_pair_schema"])
    assert keys >= {"pair_id", "a", "b", "separation_arcmin", "ra_a_deg", "dec_b_deg"}
    assert all(set(p) == keys for p in pairs)
    assert len({p["pair_id"] for p in pairs}) == 16
    assert ("GJ 702A", "GJ 702B") in {(p["a"], p["b"]) for p in pairs}
    assert all(p["separation_over_beam_fwhm"] < 1.0 and p["counted_once_each"] for p in pairs)
