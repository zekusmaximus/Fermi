"""Adversary runner (docs/MILESTONE_2.md B2)."""

from __future__ import annotations

import hashlib
import json

import astropy.units as u
import numpy as np
import pytest
import yaml

from fermi_desk.adversary.checks import (
    DEC_SIGN_FLIP_CLAIM,
    HIP1692_CLAIM,
    audit_catalog_dec_sign_flip,
    audit_distance_revision,
    audit_duplicate_pointings,
    audit_hip1692_distance_revision,
    audit_illustrative_tiers,
    audit_local_approximation,
    audit_pointing_overlap,
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
    # v0.2.1: the merge-policy audit is part of every adversary run.
    overlap = next(f for f in a["results"]["findings"] if f["check"] == "pointing_overlap")
    assert overlap["status"] == "pass" and overlap["details"]["merges"] == []
    assert overlap["details"]["policy_requested"] == "reject_duplicate_pointing_group"


def test_pointing_overlap_audit_fails_loudly_on_keep_first_and_unmerged_collisions(
    graph, data_copy
):
    """keep_first is a critical failure; so is any pointing-group collision left unmerged."""
    ok = audit_pointing_overlap(graph)
    assert (ok.status, ok.severity) == ("pass", "info")
    assert ok.details["keep_first_refused_by_registry"] is True
    assert set(ok.details["allowed_policies"]) == {
        "reject_duplicate_pointing_group",
        "union_targets_by_pointing",
    }
    bad = audit_pointing_overlap(graph, policy="keep_first")
    assert (bad.status, bad.severity) == ("fail", "critical")
    assert "load order" in bad.details["error"]
    # A second, undeclared survey on the BL pointing group: unmerged collision under both
    # the default policy and union_targets_by_pointing (no declared parent pointing).
    src = data_copy / "data" / "surveys" / "bl_enriquez_2017_692.yaml"
    doc = yaml.safe_load(src.read_text())
    doc["id"] = "bl_duplicate"
    (src.parent / "bl_duplicate.yaml").write_text(yaml.safe_dump(doc))
    dup_graph = load_factor_graph(data_copy)
    for policy in ("reject_duplicate_pointing_group", "union_targets_by_pointing"):
        f = audit_pointing_overlap(dup_graph, policy=policy)
        assert (f.status, f.severity) == ("fail", "critical") and f.details["unmerged_collisions"]


def test_build_context_targets_carry_coordinates(graph):
    ctx = build_context(
        graph, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=0)
    )
    assert ctx.targets.ra_deg is not None and len(ctx.targets.ra_deg) == 692


# ------------------------------------- catalog declination sign flip (v0.2.1-R5)
LOCKED_CATALOG_SHA256 = "c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2"
FLIPPED = {
    "HIP 22288", "HIP 77349", "HIP 11029", "HIP 115445", "HIP 8497", "HIP 75253", "HIP 3909",
    "HIP 34104", "HIP 103388", "HIP 58576", "HIP 4845", "HIP 9094", "HIP 5364", "HIP 76219",
}  # fmt: skip
NO_GAIA_ID = {"HIP 34104", "HIP 58576", "HIP 77349"}
NORTHERN_CONTROLS = {"HIP 59608", "HIP 92283"}
MINUS_ZERO = {"HIP 99711", "HIP 12530", "HIP 82588"}


def _catalog_sha(root) -> str:
    return hashlib.sha256((root / "data/catalogs/bl_692_targets.csv").read_bytes()).hexdigest()


def test_catalog_dec_sign_audit_reproduces_the_14_flips_offline(graph):
    before = _catalog_sha(graph.root)
    f = audit_catalog_dec_sign_flip(graph)
    # A real catalog erratum is a first-class finding, never a pass and never hidden.
    assert (f.check, f.status, f.severity) == ("catalog_dec_sign_audit", "finding", "warning")
    d = f.details
    assert d["problems"] == [] and d["ruling"] == "v0.2.1-R5"
    # The mirror band of the locked catalog is empty; the flipped band holds 14 + 2.
    assert d["band_census"]["mirror_band_deg"] == [-11.0, -10.0]
    assert d["band_census"]["n_targets_in_mirror_band"] == 0
    assert d["band_census"]["flipped_band_deg"] == [10.0, 11.0]
    assert d["band_census"]["n_targets_in_flipped_band"] == 16
    rows = {a["target_name"]: a for a in d["affected_targets"]}
    assert set(rows) == FLIPPED and d["n_affected_targets"] == 14 and d["n_catalog_targets"] == 692
    for name, a in rows.items():
        assert all(a["checks"].values()), (name, a["checks"])
        assert 10.0 <= a["catalog_dec_deg"] < 11.0 and -11.0 <= a["true_dec_deg"] < -10.0
        assert a["sign_restored_separation_arcsec"] <= d["match_tolerance_arcsec"] == 2.0
        assert 20.1 <= a["offset_deg"] <= 21.9
        assert a["truth_data_tier"] == "reproduced_archive"
        # ruling v0.2.1-R4: an id is a decimal string; a target without one is null
        gid = a["gaia_dr3_source_id"]
        assert (gid is None) == (name in NO_GAIA_ID)
        assert gid is None or (isinstance(gid, str) and gid.isdigit())
        assert a["truth_source"] == ("hip2" if name in NO_GAIA_ID else "gaia_dr3")
    assert any(
        int(a["gaia_dr3_source_id"]) > 2**53 for a in rows.values() if a["gaia_dr3_source_id"]
    )
    # The two genuinely northern targets of the band are controls: reported, never flipped.
    controls = {c["target_name"]: c for c in d["northern_controls"]}
    assert set(controls) == NORTHERN_CONTROLS and not NORTHERN_CONTROLS & FLIPPED
    assert all(c["in_flipped_band"] and not c["corrected"] for c in controls.values())
    # 692stars.csv -00 deg rows: an upstream table discrepancy the catalog does not inherit,
    # with no claim about where the telescope pointed.
    (sec,) = d["secondary_discrepancies"]
    assert {r["target_name"] for r in sec["records"]} == MINUS_ZERO
    assert sec["inherited_by_locked_catalog"] is False
    assert all(
        r["catalog_dec_deg"] < 0 and r["raw_692stars_dec"].startswith("+00") for r in sec["records"]
    )
    assert sec["telescope_pointing_claim"].startswith("none")
    # The exact immutable catalog hash is in the finding, and the audit wrote nothing.
    assert d["locked_catalog"] == {
        "path": "data/catalogs/bl_692_targets.csv",
        "sha256": LOCKED_CATALOG_SHA256,
        "checksum_verified": True,
        "modified_by_this_audit": False,
    }
    assert _catalog_sha(graph.root) == before == LOCKED_CATALOG_SHA256
    assert {e["data_tier"] for e in d["evidence"]} == {"published_catalog"}
    json.dumps(d)  # receipt-serialisable


def test_catalog_dec_sign_audit_is_recomputed_from_the_catalog_rows(graph):
    """Independent recount in the test: negate the catalog declination, compare with truth."""
    import pandas as pd
    from astropy.coordinates import SkyCoord

    df = pd.read_csv(graph.root / "data/catalogs/bl_692_targets.csv").set_index("target_name")
    dec = df["dec_deg"]
    assert int(((dec >= -11.0) & (dec < -10.0)).sum()) == 0
    assert set(df.index[(dec >= 10.0) & (dec < 11.0)]) == FLIPPED | NORTHERN_CONTROLS
    claim = yaml.safe_load((graph.root / DEC_SIGN_FLIP_CLAIM).read_text("utf-8"))
    assert [c["target_name"] for c in claim["pointing_corrections"]] != []
    for c in claim["pointing_corrections"]:
        row = df.loc[c["target_name"]]
        truth = SkyCoord(c["true_ra_deg"] * u.deg, c["true_dec_deg"] * u.deg)
        restored = SkyCoord(row.ra_deg * u.deg, -row.dec_deg * u.deg)
        assert restored.separation(truth) < 2.0 * u.arcsec
        assert SkyCoord(row.ra_deg * u.deg, row.dec_deg * u.deg).separation(truth) > 20 * u.deg
        # quoted string or null in the file itself (never a YAML integer)
        assert c["gaia_dr3_source_id"] is None or isinstance(c["gaia_dr3_source_id"], str)
    f = audit_catalog_dec_sign_flip(graph)
    by = {a["target_name"]: a for a in f.details["affected_targets"]}
    for c in claim["pointing_corrections"]:
        assert by[c["target_name"]]["offset_deg"] == pytest.approx(c["offset_deg"], abs=5e-4)


def test_catalog_dec_sign_audit_is_part_of_every_adversary_run(graph):
    rec = run_adversary(graph, seed=9, pair=PAIR, n_mc=100)
    checks = [f["check"] for f in rec["results"]["findings"]]
    assert checks.count("catalog_dec_sign_audit") == 1
    assert checks.index("catalog_dec_sign_audit") == checks.index("hip1692_distance_revision") + 1
    f = next(x for x in rec["results"]["findings"] if x["check"] == "catalog_dec_sign_audit")
    assert f["status"] == "finding" and f["details"]["n_affected_targets"] == 14
    assert f["details"]["locked_catalog"]["sha256"] == LOCKED_CATALOG_SHA256


def _edit_claim(root, edit) -> None:
    path = root / DEC_SIGN_FLIP_CLAIM
    doc = yaml.safe_load(path.read_text("utf-8"))
    edit(doc)
    path.write_text(yaml.safe_dump(doc, sort_keys=False), "utf-8")


def _first(doc, name):
    return next(c for c in doc["pointing_corrections"] if c["target_name"] == name)


@pytest.mark.parametrize(
    ("label", "edit", "message"),
    [
        (
            "truth_left_in_the_north",
            lambda d: _first(d, "HIP 22288").update(true_dec_deg=10.933646),
            "HIP 22288",
        ),
        (
            "catalog_value_edited",
            lambda d: _first(d, "HIP 5364").update(catalog_dec_deg=-10.182222),
            "catalog_coordinates_match_claim",
        ),
        (
            "wrong_gaia_id",
            lambda d: _first(d, "HIP 9094").update(gaia_dr3_source_id="2462426800883134337"),
            "gaia_dr3_source_id_matches_claim",
        ),
        (
            "numeric_gaia_id",
            lambda d: _first(d, "HIP 9094").update(gaia_dr3_source_id=2462426800883134336),
            "gaia_dr3_source_id",
        ),
        (
            "correction_dropped",
            lambda d: d["pointing_corrections"].pop(0),
            "neither corrected nor declared controls",
        ),
        (
            "control_flipped",
            lambda d: d["pointing_corrections"].append(
                {
                    **_first(d, "HIP 22288"),
                    "target_name": "HIP 59608",
                    "gaia_dr3_source_id": "3906361133255335296",
                }
            ),
            "both as a correction and as a control",
        ),
        (
            "other_catalog",
            lambda d: d["locked_catalog"].update(sha256="0" * 64),
            "different catalog",
        ),
        (
            "wrong_census",
            lambda d: d["catalog_band_census"].update(n_targets_in_mirror_band=14),
            "claim expects 14",
        ),
        (
            "wrong_offset",
            lambda d: _first(d, "HIP 76219").update(offset_deg=2.0),
            "offset_matches_claim",
        ),
    ],
)
def test_catalog_dec_sign_audit_detects_a_tampered_claim(data_copy, label, edit, message):
    _edit_claim(data_copy, edit)
    f = audit_catalog_dec_sign_flip(load_factor_graph(data_copy))
    assert (f.status, f.severity) == ("fail", "critical"), label
    assert message in f.summary, (label, f.summary)


def test_catalog_dec_sign_audit_detects_a_modified_catalog(data_copy):
    """Editing the locked catalog (even to 'fix' it) breaks its checksum: the audit fails."""
    path = data_copy / "data/catalogs/bl_692_targets.csv"
    raw = path.read_bytes()
    old, new = b"HIP 22288,72.004583,10.933611,", b"HIP 22288,72.004583,-10.933611,"
    assert raw.count(old) == 1
    path.write_bytes(raw.replace(old, new))  # one byte added, nothing else touched
    f = audit_catalog_dec_sign_flip(load_factor_graph(data_copy))
    assert (f.status, f.severity) == ("fail", "critical")
    assert "checksum mismatch" in f.summary


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
