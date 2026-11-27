"""v0.2.1b: the promoted Gaia DR3 background field, ruling v0.2.1-R2, realized EIG (offline).

Everything here reads the REAL promoted artifact
(``data/catalogs/bl_gbt_lband_background_summary.json``, ``reproduced_archive``) and the
locked primary catalog. No test touches the network (``conftest.py`` blocks it) and no test
writes inside the project tree: tamper cases work on copies under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from conftest import ROOT
from fermi_desk.cli import app
from fermi_desk.errors import DoubleCountingError, ProvenanceError, SchemaValidationError
from fermi_desk.inference import background_field as bgf
from fermi_desk.inference import eig
from fermi_desk.inference.engine import (
    CHANNELS,
    FLAT_BACKGROUND_INPUT,
    HEADLINE_VERIFIED,
    HEADLINE_WITH_ILLUSTRATIVE,
    RunConfig,
    results_digest,
    run_comparison,
)
from fermi_desk.models import radio
from fermi_desk.queue import runners
from fermi_desk.queue.store import JOB_DEFINITIONS, JobStore
from fermi_desk.registry import (
    PUBLISHED_BACKGROUND_SCOPE,
    background_summary_bytes,
    load_background_summary,
    load_factor_graph,
    run_validation,
    seal_background_summary,
)
from fermi_desk.schemas import (
    BACKGROUND_SUMMARY_MAX_BYTES,
    BACKGROUND_SUMMARY_PATH,
    REJECT_DUPLICATE_POINTING_GROUP,
    UNION_TARGETS_BY_POINTING,
    BackgroundSummaryFile,
)

PARENT = "bl_enriquez_2017_692"
EXT = "bl_gbt_lband_background_gaia_dr3"
PAIR = ("frequent_slow_expanders", "rare_fast_expanders")
IR_PAIR = ("quiet_expanders", "partial_dyson_conspicuous")
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
}
CANONICAL_THIN_DISK_FLAT = "4b70118aaded7861cb14d77379843dfacec2d4c517bbce5efc9654e139fbc269"
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
JOB = next(j for j in JOB_DEFINITIONS if j.job_id == runners.BACKGROUND_JOB_ID)


# ------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def field(graph):
    f = bgf.load_background_field(graph)
    assert f is not None, "v0.2.1b promotes a background field; none is pinned"
    return f


@pytest.fixture(scope="module")
def raw_doc():
    return json.loads((ROOT / BACKGROUND_SUMMARY_PATH).read_bytes())


@pytest.fixture(scope="module")
def evidence(graph):
    ev = runners.background_extension_evidence(graph, JOB.parameters)
    assert ev is not None
    return ev


def _copy_data(dst) -> None:
    shutil.copytree(ROOT / "data", dst / "data", ignore=shutil.ignore_patterns("cache", "fits"))


def _root_with(tmp_path, mutate, *, repin=True, reseal=True):
    """A data/ copy whose artifact was edited; optionally re-sealed and re-pinned."""
    root = tmp_path / f"case{len(list(tmp_path.iterdir()))}"
    _copy_data(root)
    path = root / BACKGROUND_SUMMARY_PATH
    old_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    doc = mutate(json.loads(path.read_bytes()))
    if reseal:
        doc = seal_background_summary(doc)
    raw = background_summary_bytes(doc)
    path.write_bytes(raw)
    if repin:
        survey = root / "data" / "surveys" / f"{EXT}.yaml"
        text = survey.read_text("utf-8")
        assert old_sha in text
        survey.write_text(text.replace(old_sha, hashlib.sha256(raw).hexdigest()), "utf-8")
    return root


def _load(root):
    g = load_factor_graph(root)
    return load_background_summary(g.surveys[EXT], g.surveys[PARENT], root)


def _stripped_graph(tmp_path):
    """The project graph without any pointing extension (the pre-promotion state)."""
    root = tmp_path / "stripped"
    _copy_data(root)
    for path in (root / "data" / "surveys").glob("*.yaml"):
        if yaml.safe_load(path.read_text("utf-8")).get("parent_pointing_survey_id"):
            path.unlink()
    (root / BACKGROUND_SUMMARY_PATH).unlink()
    return load_factor_graph(root)


# ====================================================== 1. promoted artifact integrity
def test_locked_v01x_artifacts_are_byte_identical():
    for rel, sha in LOCKED.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, rel


def test_promoted_artifact_is_pinned_compact_clean_and_complete(graph, field, raw_doc):
    survey = graph.surveys[EXT]
    raw = (ROOT / BACKGROUND_SUMMARY_PATH).read_bytes()
    # Declared pointing extension of the locked BL 692 pointings, pinned by SHA-256.
    assert (
        survey.pointing_group_id == graph.surveys[PARENT].pointing_group_id == "bl_gbt_lband_2017"
    )
    assert survey.parent_pointing_survey_id == PARENT and survey.is_pointing_extension
    assert survey.pointing_merge_policy == UNION_TARGETS_BY_POINTING
    assert survey.data_tier == "reproduced_archive" and survey.observed_detections == 0
    assert survey.background_summary_path == BACKGROUND_SUMMARY_PATH
    assert survey.background_summary_sha256 == hashlib.sha256(raw).hexdigest() == field.file_sha256
    # Strictly below the 5 MB git budget, in the canonical serialisation (byte-exact).
    assert len(raw) < BACKGROUND_SUMMARY_MAX_BYTES == 5 * 1024 * 1024
    assert background_summary_bytes(raw_doc) == raw
    assert seal_background_summary(raw_doc)["content_sha256"] == raw_doc["content_sha256"]
    rec = field.record
    # Clean provenance: the assembling commit and every commit that queried the archive.
    assert rec.data_tier == "reproduced_archive" and GIT_SHA.match(rec.code_git_sha)
    assert rec.cone_cache_code_git_shas and all(
        GIT_SHA.match(s) for s in rec.cone_cache_code_git_shas
    )
    assert rec.cone_cache_manifest_sha256 == rec.merge_provenance.input_sha256s[1]
    assert rec.query_window_utc[0] < rec.query_window_utc[1] < rec.assembled_at_utc
    assert rec.fetched_at_utc == rec.query_window_utc[1]
    assert (
        "gaiadr3.gaia_source" in rec.source_service
        and "gaia.ari.uni-heidelberg.de" in rec.source_service
    )
    assert rec.source_receipt is not None and rec.source_receipt.path.startswith("runs/")
    # Frozen protocol: all 692 locked pointings, 8.4' beam / 4.2' cone, d <= 10 kpc.
    assert (rec.n_pointings, rec.n_pointings_completed) == (692, 692)
    assert (rec.beam_fwhm_arcmin, rec.cone_radius_arcmin, rec.distance_cuts.d_max_pc) == (
        8.4,
        4.2,
        1e4,
    )
    assert rec.primary_catalog_sha256 == LOCKED["data/catalogs/bl_692_targets.csv"]
    assert "NOT the published catalogue" in rec.methodology_note
    assert "Gaia DR3" in rec.methodology_note and "288,315" in rec.methodology_note


def test_promoted_artifact_counts_are_the_exact_live_fetch_counts(field):
    """The numbers quoted in the handoff, pinned (they are fixed by the artifact's SHA-256)."""
    rec = field.record
    cuts, mp, da = rec.distance_cuts, rec.merge_provenance, rec.dedup_audit
    assert (rec.shells.n_stars, rec.shells.n_cells) == (120176, 46913)
    assert cuts.n_cone_sightings_raw == 259285
    assert cuts.n_dropped_invalid_parallax + cuts.n_dropped_beyond_d_max == 134601 + 133
    assert (cuts.n_dropped_no_bailer_jones, cuts.n_dropped_outside_cone_recheck) == (0, 0)
    assert cuts.n_sightings_with_distance == 124551 == 37888 + 86663
    assert (cuts.n_background_inverse_parallax, cuts.n_background_bailer_jones_geo) == (
        36069,
        84107,
    )
    assert (mp.n_raw_targets, mp.n_merged_targets) == (692 + 124551, 692 + 120176)
    assert (mp.n_dedup_gaia_id, mp.n_dedup_angular_fallback) == (4270, 105)
    assert (mp.n_multi_beam_resolved, mp.n_background_absorbed_by_primary) == (3764, 635)
    assert mp.n_merged_by_role == {"primary": 692, "background": 120176}
    assert da.n_background_sightings_matched_to_primary_gaia_id == 523
    assert da.n_background_sightings_matched_to_primary_angular == 112
    counts = field.dedup_counts()
    assert counts["n_suspected_unmatched_primary"] == 49
    assert counts["n_suspected_unmatched_primary_stars"] == 47
    assert counts["n_primaries_with_a_suspected_counterpart"] == 45
    # 32 of those 45 primaries had already absorbed a Gaia sighting (the suspected star is a
    # second nearby source); 13 absorbed nothing (the likelier double counts).
    assert counts["n_suspected_primaries_that_absorbed_a_sighting"] == 32
    assert counts["n_suspected_primaries_that_absorbed_nothing"] == 13
    assert counts["n_primary_without_gaia_id"] == 170
    assert "never removed (ruling v0.2.1-R3)" in counts["suspected_unmatched_primary_policy"]
    assert (
        "0 duplicated Gaia ids" in field.detail and "49 suspected_unmatched_primary" in field.detail
    )


def test_zero_duplicate_gaia_ids_against_the_locked_primary_catalog(field, raw_doc):
    cat = pd.read_csv(ROOT / "data/catalogs/bl_692_targets.csv", dtype={"gaia_dr3_source_id": str})
    primary_ids = {x for x in cat["gaia_dr3_source_id"].dropna() if x.strip()}
    assert len(primary_ids) == 522
    da, mp = field.record.dedup_audit, field.record.merge_provenance
    assert da.n_duplicate_gaia_ids_vs_primary == 0 == da.n_duplicate_gaia_ids_within_background
    assert mp.n_duplicate_gaia_ids_post_merge == 0
    sample = raw_doc["validation_sample"]
    kept = [s["gaia_dr3_source_id"] for s in sample["background"]]
    assert len(kept) == len(set(kept)) == 1000 and not primary_ids & set(kept)
    # Every primary with a Gaia id was sighted and absorbed (never left in a shell) ...
    by_id = {
        m["gaia_dr3_source_id"]
        for m in sample["primary_matched"]
        if m["match"] == "gaia_dr3_source_id"
    }
    assert by_id <= primary_ids and len(by_id) == 506  # 16 were never sighted: see the audit
    # ... and the suspected stars carry ids no primary has.
    suspects = {
        r["gaia_dr3_source_id"] for r in raw_doc["dedup_audit"]["suspected_unmatched_primary"]
    }
    assert len(suspects) == 47 and not suspects & primary_ids


def test_artifact_serialises_every_gaia_id_as_a_string_never_a_json_number(raw_doc):
    """Ruling v0.2.1-R4, on the promoted bytes themselves."""
    raw = (ROOT / BACKGROUND_SUMMARY_PATH).read_bytes()
    assert re.search(rb'"gaia_dr3_source_id":\s*[0-9-]', raw) is None
    ids = re.findall(rb'"gaia_dr3_source_id":"([1-9][0-9]*)"', raw)
    assert len(ids) == 1000 + 635 + 49
    over = [int(x) for x in ids if int(x) > 2**53]
    assert len(over) > 1600 and any(int(float(x)) != x for x in over)  # float64 would corrupt
    star = raw_doc["validation_sample"]["background"][0]
    numeric = {**star, "gaia_dr3_source_id": int(star["gaia_dr3_source_id"])}
    sample = {**raw_doc["validation_sample"]}
    sample["background"] = [numeric, *sample["background"][1:]]
    with pytest.raises(ValueError, match="gaia_dr3_source_id"):
        BackgroundSummaryFile.model_validate({**raw_doc, "validation_sample": sample})


def test_validate_checks_the_promoted_artifact_automatically(field):
    res = {r.name: r for r in run_validation(ROOT)}
    assert all(r.ok for r in res.values()), [r for r in res.values() if not r.ok]
    row = res[f"background_summary:{BACKGROUND_SUMMARY_PATH}"]
    assert field.file_sha256[:16] in row.detail and "reproduced_archive" in row.detail
    assert "0 duplicated Gaia ids" in row.detail and "1000 validation stars" in row.detail
    assert "suspected_unmatched_primary" in row.detail
    assert f"pinned by {EXT}" in res["background_field_promotion"].detail
    assert EXT in res["pointing_merge_policy"].detail
    assert res[f"survey:{EXT}"].ok and len(res) >= 40


def test_tampered_promoted_artifact_is_rejected(tmp_path, graph):
    """Every way of editing the artifact is caught: by the pin, the seal, the schema or the
    offline recomputation against the locked primary catalog."""
    assert (
        _load(_root_with(tmp_path, lambda d: d)).file_sha256
        == graph.surveys[EXT].background_summary_sha256
    )

    def edit(**over):
        return lambda d: {**d, **over}

    def star(**over):
        def mutate(d):
            vs = d["validation_sample"]
            first = {**vs["background"][0], **over}
            return {**d, "validation_sample": {**vs, "background": [first, *vs["background"][1:]]}}

        return mutate

    def shells(d):
        sh = d["shells"]
        return {**d, "shells": {**sh, "count": [sh["count"][0] + 1, *sh["count"][1:]]}}

    primary_id = next(
        m["gaia_dr3_source_id"]
        for m in json.loads((ROOT / BACKGROUND_SUMMARY_PATH).read_bytes())["validation_sample"][
            "primary_matched"
        ]
        if m["match"] == "gaia_dr3_source_id"
    )
    with pytest.raises(ProvenanceError, match="sha256"):  # edited, pin not updated
        _load(_root_with(tmp_path, edit(description="edited"), repin=False))
    with pytest.raises(ProvenanceError, match="content_sha256"):  # re-pinned, not re-sealed
        _load(_root_with(tmp_path, edit(description="edited"), reseal=False))
    cases = (  # re-sealed AND re-pinned: schema / recomputation must still catch it
        (edit(code_git_sha="dirty"), ProvenanceError, "clean 40-hex commit"),
        (edit(n_pointings_completed=691), ProvenanceError, "incomplete pointing coverage"),
        (edit(cone_cache_code_git_shas=["dirty"]), ProvenanceError, "non-clean tree"),
        (edit(data_tier="synthetic_proxy"), ProvenanceError, "data_tier"),
        (edit(primary_catalog_sha256="0" * 64), ProvenanceError, "parent catalog"),
        (star(eirp_min_w=1.0e12), ProvenanceError, "EIRP_min"),
        (star(theta_arcmin=0.01), ProvenanceError, "theta"),
        (star(gaia_dr3_source_id=primary_id), DoubleCountingError, "duplicates a primary Gaia id"),
        (shells, (ProvenanceError, SchemaValidationError), None),
        (edit(description="x" * BACKGROUND_SUMMARY_MAX_BYTES), ProvenanceError, "not < 5 MB"),
    )
    for mutate, error, message in cases:
        with pytest.raises(error, match=message):
            _load(_root_with(tmp_path, mutate))
    # validate reports the failure instead of passing silently.
    bad = _root_with(tmp_path, star(distance_pc=3.0))
    row = {r.name: r for r in run_validation(bad)}[f"background_summary:{BACKGROUND_SUMMARY_PATH}"]
    assert not row.ok and "ProvenanceError" in row.detail


def test_crlf_checkout_cannot_change_the_pinned_bytes():
    attrs = (ROOT / ".gitattributes").read_text("utf-8")
    assert "data/catalogs/*.json -text" in attrs and "data/catalogs/*.csv -text" in attrs
    assert b"\r" not in (ROOT / BACKGROUND_SUMMARY_PATH).read_bytes()


# ============================================================ 2. ruling v0.2.1-R2
def _cfg(mode, policy=UNION_TARGETS_BY_POINTING, **kw):
    kw.setdefault("channels", ("radio",))
    kw.setdefault("n_mc", 200)
    return RunConfig(
        geometry="thin_disk_2d",
        radio_mode=mode,
        seed=3,
        headline_pair=PAIR,
        pointing_merge_policy=policy,
        **kw,
    )


@pytest.fixture(scope="module")
def flat_union(graph):
    return run_comparison(graph, _cfg("flat_probability"), list(PAIR))


def test_default_policy_still_reproduces_the_canonical_hash_and_holds_the_extension_out(graph):
    cfg = RunConfig(
        geometry="thin_disk_2d",
        radio_mode="flat_probability",
        seed=42,
        channels=CHANNELS,
        headline_pair=PAIR,
        ir_pair=IR_PAIR,
    )
    rec = run_comparison(graph, cfg)
    assert results_digest(rec) == CANONICAL_THIN_DISK_FLAT
    assert rec["pointing_merge_policy"] == REJECT_DUPLICATE_POINTING_GROUP
    assert rec["merge_provenance"] is None and rec["background_field"] is None
    (held,) = rec["held_out_extension_surveys"]
    assert held["survey_id"] == EXT and held["parent_pointing_survey_id"] == PARENT
    assert "NOT in this run" in held["reason"]
    head = rec["results"]["headline"]["headline"]
    assert head["policy"] == HEADLINE_VERIFIED and head["terms"] == ["earth_unreached", "radio"]
    assert rec["headline_log10_bf"] == pytest.approx(-8.04966e-05, rel=1e-5)
    assert [s["id"] for s in rec["active_surveys"] if s["channel"] == "radio"] == [PARENT]


def test_flat_mode_merged_background_is_held_out_of_the_verified_headline(flat_union, field):
    rec = flat_union
    bf = rec["results"]["headline"]
    head = bf["headline"]
    # Radio leaves the verified headline although the star catalogue is reproduced_archive.
    assert head["policy"] == HEADLINE_VERIFIED == rec["headline_policy"]
    assert head["terms"] == ["earth_unreached"] and head["channels"] == []
    assert head["excluded_illustrative_channels"] == ["radio"]
    assert rec["headline_log10_bf"] == head["log10_bf"] == bf["verified"]["log10_bf"]
    assert abs(head["log10_bf"]) < 1e-12  # the 2D Earth term cancels: a null headline
    # ... and stays visible in the component table, tagged with its tiers.
    assert bf["terms"] == ["earth_unreached", "radio"]
    assert bf["term_data_tiers"]["radio"] == [
        "illustrative_calibration",
        "published_catalog",
        "reproduced_archive",
    ]
    assert bf["illustrative_terms"] == ["radio"]
    assert bf["log10_bf_by_term"]["radio"] < -0.5  # what it WOULD have added to a headline
    # One structured exclusion note names the ruling, the reason and where the term comes from.
    (note,) = head["headline_exclusion_notes"]
    assert note == rec["headline_exclusion_notes"][0]
    assert note["channel"] == "radio" and note["held_out_of_verified_headline"] is True
    assert note["held_out_of_reported_headline"] is True and note["active_in_run"] is True
    assert note["illustrative_inputs"] == [FLAT_BACKGROUND_INPUT] and note["rulings"] == [
        "v0.2.1-R2"
    ]
    (reason,) = note["reasons"]
    for phrase in ("d^2 / B(theta)", "10 kpc", "illustrative sensitivity check", "v0.2.1-R2"):
        assert phrase in reason
    assert note["log10_bf_term"] == bf["log10_bf_by_term"]["radio"]
    mb = note["merged_background"]
    assert mb["survey_id"] == EXT and mb["data_tier"] == "reproduced_archive"
    assert mb["summary_sha256"] == field.file_sha256
    assert (mb["n_primary_targets"], mb["n_background_stars"]) == (692, 120176)
    assert mb["log10_bf_primary_targets_exact_mc"] + mb[
        "log10_bf_background_shells_fkg_product"
    ] == pytest.approx(note["log10_bf_term"], abs=1e-12)
    assert abs(mb["log10_bf_background_shells_fkg_product"]) > 100 * abs(
        mb["log10_bf_primary_targets_exact_mc"]
    )
    # Receipt-level provenance of the merged field (ruling v0.2.1-R3 count included).
    assert "illustrative_calibration" in rec["data_tiers_used"]
    assert rec["merge_provenance"] == field.record.merge_provenance.model_dump()
    assert rec["background_field"]["file_sha256"] == field.file_sha256
    assert rec["background_field"]["dedup_audit"]["n_suspected_unmatched_primary"] == 49
    assert rec["background_field"]["scope"]["retrieved"]["n_background_stars"] == 120176
    assert rec["held_out_extension_surveys"] == []
    for hid in PAIR:
        comp = rec["results"]["hypotheses"][hid]["components"]["radio"]
        assert comp["illustrative_inputs"] == [FLAT_BACKGROUND_INPUT]
        assert rec["results"]["hypotheses"][hid]["radio"]["n_targets_merged"] == 692 + 120176


def test_include_illustrative_folds_the_flat_background_in_with_the_loud_warning(graph, flat_union):
    rec = run_comparison(
        graph, _cfg("flat_probability", include_illustrative_in_headline=True), list(PAIR)
    )
    head = rec["results"]["headline"]["headline"]
    assert head["policy"] == HEADLINE_WITH_ILLUSTRATIVE == rec["headline_policy"]
    assert head["terms"] == ["earth_unreached", "radio"]
    assert head["includes_illustrative_channels"] == ["radio"]
    assert (
        "ILLUSTRATIVE CHANNELS INCLUDED" in head["warning"]
        and "NOT an observational bound" in head["warning"]
    )
    (note,) = head["headline_exclusion_notes"]
    assert note["held_out_of_reported_headline"] is False and note["rulings"] == ["v0.2.1-R2"]
    # Opt-in changes the headline number only; every likelihood is identical.
    assert rec["headline_log10_bf"] == pytest.approx(flat_union["results"]["headline"]["log10_bf"])
    assert rec["headline_log10_bf"] < -0.5
    assert rec["results"]["hypotheses"] == flat_union["results"]["hypotheses"]


def test_distance_scaled_merge_stays_excluded_for_its_own_reason(graph):
    """Under distance_scaled_eirp radio is already held out (M2-R2: illustrative luminosity
    function); the R2 flat-background input must not appear there."""
    rec = run_comparison(graph, _cfg("distance_scaled_eirp"), list(PAIR))
    head = rec["results"]["headline"]["headline"]
    assert head["terms"] == ["earth_unreached"]
    (note,) = head["headline_exclusion_notes"]
    assert note["illustrative_inputs"] == ["transmitter_eirp_luminosity_function"]
    assert "rulings" not in note and note["merged_background"]["n_background_stars"] == 120176
    # Off-axis d^2 / B(theta) attenuation: the physical background term is tiny.
    assert abs(note["log10_bf_term"]) < 0.01


def test_single_survey_notes_keep_their_pre_r2_shape(graph):
    """Notes of earlier releases gain no keys (their results_sha256 depends on it)."""
    rec = run_comparison(
        graph, _cfg("distance_scaled_eirp", REJECT_DUPLICATE_POINTING_GROUP), list(PAIR)
    )
    (note,) = rec["headline_exclusion_notes"]
    assert set(note) == {
        "channel",
        "active_in_run",
        "held_out_of_verified_headline",
        "held_out_of_reported_headline",
        "illustrative_inputs",
        "reasons",
        "data_tiers",
        "log10_bf_term",
    }
    flat = run_comparison(
        graph, _cfg("flat_probability", REJECT_DUPLICATE_POINTING_GROUP), list(PAIR)
    )
    assert flat["headline_exclusion_notes"] == []
    assert flat["results"]["headline"]["headline"]["terms"] == ["earth_unreached", "radio"]
    assert "illustrative_calibration" not in flat["data_tiers_used"]


def test_cli_reports_the_r2_note_the_banner_and_the_suspected_count(tmp_path):
    base = [
        "compare",
        "--geometry",
        "thin_disk_2d",
        "--radio-mode",
        "flat_probability",
        "--pointing-merge-policy",
        "union_targets_by_pointing",
        "--seed",
        "42",
        "--channels",
        "radio",
        "--n-mc",
        "200",
        "--out-dir",
        str(tmp_path),
    ]
    res = CliRunner().invoke(app, base)
    out = " ".join(res.output.split())
    assert res.exit_code == 0, res.output
    assert "headline_exclusion_note radio (held out of this headline" in out
    assert "ruling v0.2.1-R2" in out and "d^2 / B(theta)" in out
    assert "terms: earth_unreached;" in out and "illustrative channels excluded: ['radio']" in out
    assert "suspected_unmatched_primary = 49 row(s), 47 distinct stars" in out
    assert "N_bg = 120176 stars in 46913 shell cells over 692 pointings" in out
    assert "1,327 Price et al. (2020)" in out and "288,315" in out
    loud = " ".join(CliRunner().invoke(app, [*base, "--include-illustrative"]).output.split())
    assert "ILLUSTRATIVE CHANNELS FOLDED INTO THE HEADLINE BF - NOT AN OBSERVATIONAL BOUND" in loud
    assert "headline_exclusion_note radio (INCLUDED in this headline" in loud
    held = " ".join(
        CliRunner()
        .invoke(
            app,
            [a for a in base if a not in ("--pointing-merge-policy", "union_targets_by_pointing")],
        )
        .output.split()
    )
    assert (
        f"pointing extension held out {EXT}" in held and "suspected_unmatched_primary" not in held
    )


# ================================================ 3. realized vs predicted EIG
def test_information_helpers_are_consistent():
    prior = {"a": 0.5, "b": 0.5}
    p0 = {"a": 0.2, "b": 0.9}
    post = eig.posterior(prior, p0)
    assert post == pytest.approx({"a": 0.2 / 1.1, "b": 0.9 / 1.1}) and sum(
        post.values()
    ) == pytest.approx(1)
    # The EIG is the outcome-averaged realized information.
    mi = eig.mutual_information_bits(prior, {h: eig.binary_outcome(v) for h, v in p0.items()})
    p_null = 0.5 * 0.2 + 0.5 * 0.9
    avg = p_null * eig.realized_information_bits(prior, p0) + (1 - p_null) * (
        eig.realized_information_bits(prior, {h: 1 - v for h, v in p0.items()})
    )
    assert avg == pytest.approx(mi, rel=1e-12) and mi > 0
    assert eig.realized_information_bits(prior, {"a": 0.4, "b": 0.4}) == pytest.approx(
        0.0, abs=1e-15
    )
    with pytest.raises(ValueError, match="zero probability"):
        eig.posterior(prior, {"a": 0.0, "b": 0.0})
    # Range over dependence bounds: 0 when the intervals overlap, a corner value otherwise.
    lo, hi = eig.binary_eig_range(prior, {"a": (0.2, 0.95), "b": (0.9, 0.99)})
    assert lo == 0.0 and hi == pytest.approx(
        eig.mutual_information_bits(
            prior, {"a": eig.binary_outcome(0.2), "b": eig.binary_outcome(0.99)}
        )
    )
    lo, hi = eig.binary_eig_range(prior, {"a": (0.1, 0.3), "b": (0.6, 0.8)})
    near = eig.mutual_information_bits(
        prior, {"a": eig.binary_outcome(0.3), "b": eig.binary_outcome(0.6)}
    )
    assert lo == pytest.approx(near) and 0 < lo < hi
    with pytest.raises(ValueError):
        eig.binary_eig_range(prior, {"a": (0.5, 0.4), "b": (0.1, 0.2)})


def test_realized_eig_is_recorded_beside_the_prediction_in_both_radio_modes(evidence, field):
    ev = evidence
    assert (
        ev["summary_sha256"] == field.file_sha256
        and ev["summary_data_tier"] == "reproduced_archive"
    )
    assert ev["n_background_stars"] == ev["realized"]["n_stars"] == 120176
    d = ev["definition"]
    assert d["prior"] == {"type": "uniform", "over": list(PAIR)} and d["geometry"] == "thin_disk_2d"
    assert d["condition_on_earth_unreached"] is True and "Harris-FKG lower bound" in d["outcome"]
    rows = {r["radio_mode"]: r for r in ev["comparison"]}
    phys, flat = rows["distance_scaled_eirp"], rows["flat_probability"]
    assert (phys["label"], phys["physical"]) == ("physical", True)
    assert (flat["label"], flat["physical"]) == ("unphysical", False)
    # Milestone 2 synthetic-proxy prediction, unchanged (PLANNING_LEDGER Sec. 3).
    assert phys["predicted_eig_bits"] == pytest.approx(0.01452, abs=5e-6)
    assert flat["predicted_eig_bits"] == pytest.approx(0.827, abs=5e-4)
    assert (
        ev["predicted"]["data_tier"] == "synthetic_proxy" and ev["predicted"]["n_stars"] == 249990
    )
    # Realized with the pinned Gaia DR3 shells (deterministic: no RNG).
    assert phys["realized_eig_bits"] == pytest.approx(0.00235113, rel=1e-5)
    assert flat["realized_eig_bits"] == pytest.approx(0.651404, rel=1e-5)
    assert phys["realized_over_predicted"] == pytest.approx(0.16189, rel=1e-4)
    assert flat["realized_over_predicted"] == pytest.approx(0.78735, rel=1e-4)
    # The real, attenuated population is LESS informative than predicted in both modes: a
    # null result, reported as it is.
    assert phys["realized_eig_bits"] < phys["predicted_eig_bits"] < 0.02
    assert flat["realized_eig_bits"] < flat["predicted_eig_bits"]
    assert flat["realized_eig_bits"] > 250 * phys["realized_eig_bits"]  # d^2 / B(theta)
    for r in (phys, flat):
        lo, hi = r["realized_eig_bits_range_fkg_to_frechet"]
        assert lo == 0.0 and hi >= r["realized_eig_bits"] > 0
        # Neither value is a verified bound: both modes carry an illustrative input.
        assert r["data_tiers"] == ["illustrative_calibration", "reproduced_archive"]
        assert 0 < r["realized_information_bits_from_observed_null"] < 1
    assert "illustrative_calibration" in ev["tier_note"] and "v0.2.1-R2" in ev["tier_note"]
    for mode, m in ev["realized"]["by_radio_mode"].items():
        assert "v0.2.1-R2" in m["illustrative_reason"] or mode == "distance_scaled_eirp"
        post = m["posterior_after_observed_null"]
        assert (
            sum(post.values()) == pytest.approx(1.0) and post[PAIR[1]] > 0.5
        )  # null favours rare/fast


def test_realized_eig_is_reported_with_and_without_suspected_unmatched_primaries(evidence, field):
    """Ruling v0.2.1-R3: the 47 suspected stars stay in the shells; the diagnostic shows how
    little they move the result."""
    without = evidence["realized_excluding_suspected_unmatched_primary"]
    assert without["diagnostic_only"] is True and without["n_stars_removed"] == 47
    assert without["n_stars"] == 120176 - 47 and "never removed" in without["policy"]
    for r in evidence["comparison"]:
        full = r["realized_eig_bits"]
        excl = r["realized_eig_bits_excluding_suspected_unmatched_primary"]
        assert excl < full and (full - excl) / full < 0.005
    trimmed = field.shells_excluding_suspected()
    assert trimmed.n_stars == field.shells.n_stars - 47 and field.shells.n_stars == 120176
    assert trimmed.n_cells <= field.shells.n_cells and trimmed.count.min() >= 1
    cells = field.suspected_star_cells()
    assert len(cells) == 47
    stored = set(
        zip(
            field.shells.pointing_index.tolist(),
            field.shells.distance_bin.tolist(),
            field.shells.eirp_min_bin.tolist(),
            strict=True,
        )
    )
    assert set(cells.values()) <= stored


def test_harris_fkg_and_frechet_bounds_bracket_every_zero_detection_probability(evidence):
    rows = bgf.bounds_table(evidence["realized"])
    assert len(rows) == 4
    for b in rows:
        assert (
            0.0
            <= b["p_zero_markov_lower"]
            <= b["p_zero_fkg_lower"]
            <= b["p_zero_frechet_upper"]
            <= 1.0
        )
        assert b["expected_detections"] > 0 and b["expected_reached"] > b["expected_detections"]
        assert b["log10_p_zero_fkg_lower"] == pytest.approx(
            np.log10(b["p_zero_fkg_lower"]), abs=1e-12
        )
    by = {(b["radio_mode"], b["hypothesis_id"]): b for b in rows}
    fs_flat = by[("flat_probability", PAIR[0])]
    fs_phys = by[("distance_scaled_eirp", PAIR[0])]
    # Same stars reached in both modes; what differs is how detectable they are.
    assert fs_flat["expected_reached"] == pytest.approx(fs_phys["expected_reached"])
    assert fs_flat["expected_detections"] == pytest.approx(2.0004, rel=1e-4)
    assert fs_phys["expected_detections"] == pytest.approx(0.0048679, rel=1e-4)
    assert fs_flat["p_zero_fkg_lower"] == pytest.approx(0.135274, rel=1e-5)
    assert fs_phys["p_zero_fkg_lower"] == pytest.approx(0.995144, rel=1e-6)
    assert "LOWER bound" in radio.SHELL_CORRELATION_ASSUMPTION


def test_scope_states_what_was_retrieved_and_what_the_paper_published(field, evidence):
    scope = field.scope()
    assert scope == evidence["scope"] and scope["comparable"] is False
    got, pub = scope["retrieved"], scope["published"]
    assert (got["n_pointings"], got["n_background_stars"], got["gaia_release"]) == (
        692,
        120176,
        "Gaia DR3",
    )
    assert "Enriquez et al. (2017) GBT L-band pointings" in got["pointings"]
    assert got["data_tier"] == "reproduced_archive"
    assert (pub["n_pointings"], pub["n_stars_total"], pub["gaia_release"]) == (
        1327,
        288315,
        "Gaia DR2",
    )
    assert pub is not PUBLISHED_BACKGROUND_SCOPE and pub == PUBLISHED_BACKGROUND_SCOPE
    assert pub["n_stars_gbt_l_plus_s_band"] == {
        "value": 249990,
        "verified_against_paper": True,
        "source": "Table 2, WG&S GBT cumulative N at EIRP_min <= 1e18 W",
    }
    # The L-band-only count comes from the planning brief and is flagged, not asserted.
    lband = pub["n_stars_gbt_l_band"]
    assert lband["value"] == 251983 and lband["verified_against_paper"] is False
    for text in (scope["statement"], evidence["statement"]):
        assert "692" in text and "1,327" in text and "288,315" in text and "Gaia DR2" in text
    assert "120,176" in evidence["statement"] and "not expected to agree" in evidence["statement"]


def test_pointing_centre_audit_reports_primaries_missing_from_their_own_cone(graph, field):
    """Offline audit of the cone centres (artifact + locked catalog). It surfaces, without
    correcting, the declination band where the locked catalog lists 16 targets and none in
    the mirror band, 14 of which have no Gaia counterpart in their own cone."""
    a = bgf.pointing_centre_audit(graph, field)
    assert (a["n_primaries"], a["n_with_gaia_id"], a["n_without_gaia_id"]) == (692, 522, 170)
    assert a["n_with_gaia_id_sighted_in_own_cone"] == 506
    assert len(a["with_gaia_id_not_sighted_in_own_cone"]) == 16
    assert a["n_without_gaia_id_absorbed_by_angular_fallback"] == 105
    assert a["n_without_gaia_id_suspected_only"] == 13
    assert a["n_no_counterpart_in_own_cone"] == 68
    assert "immutable (ruling v0.2.1-R3)" in a["policy"]
    d = a["densest_band_of_unfound_primaries"]
    assert d["declination_band_deg"] == [10, 11] and d["mirror_band_deg"] == [-11, -10]
    assert (d["n_targets"], d["n_targets_in_mirror_band"], d["n_no_counterpart"]) == (16, 0, 14)
    assert len(d["targets_no_counterpart"]) == 14 and "HIP 5364" in d["targets_no_counterpart"]
    assert d["n_background_stars_in_those_pointings"] == 1055
    assert d["fraction_of_background_stars"] == pytest.approx(1055 / 120176)
    assert "declination sign error" in d["reading"] and "consistent with" in d["reading"]
    # Independent recount from the locked catalog: nothing below the mirror band is empty
    # by coverage (the catalog reaches Dec -29.6 deg).
    cat = pd.read_csv(ROOT / "data/catalogs/bl_692_targets.csv")
    dec = cat["dec_deg"]
    assert (
        int(((dec >= 10) & (dec < 11)).sum()) == 16 and int(((dec > -11) & (dec <= -10)).sum()) == 0
    )
    assert dec.min() < -29 and int(((dec > -12) & (dec <= -11)).sum()) > 0
    # The locked catalog itself is untouched by the audit.
    sha = hashlib.sha256((ROOT / "data/catalogs/bl_692_targets.csv").read_bytes()).hexdigest()
    assert sha == LOCKED["data/catalogs/bl_692_targets.csv"]


def test_shell_and_attenuation_summaries(graph, field):
    sh = bgf.shell_summary(field)
    assert (sh["n_stars"], sh["n_cells"]) == (120176, 46913)
    assert sum(sh["distance_histogram"]["counts"]) == 120176
    assert (
        len(sh["distance_histogram"]["bin_edges_pc"]) == len(sh["distance_histogram"]["counts"]) + 1
    )
    cum = [c["n_stars"] for c in sh["eirp_min_cumulative_by_decade"]]
    assert cum == sorted(cum) and cum[-1] == 120176
    assert sh["distance_cumulative"][-1]["n_stars"] == 120176
    assert sh["stars_per_pointing"]["n_pointings"] == 692
    att = bgf.attenuation_summary(graph, field)
    v = att["validation_sample"]
    assert att["beam_response_at_cone_edge"] == 0.5 and v["n_stars"] == 1000
    assert 0.5 <= v["beam_response"]["min"] <= v["beam_response"]["median"] <= 1.0
    assert (
        1.0
        <= v["beam_attenuation_factor"]["median"]
        <= v["beam_attenuation_factor"]["max"]
        <= 2.0 + 1e-9
    )
    assert v["theta_arcmin"]["max"] <= 4.2
    # The background needs orders of magnitude more EIRP than the on-axis primaries; the
    # d^2 term, not the beam, is why.
    assert att["median_eirp_min_ratio_background_over_primary"] > 1e3
    dec = att["decomposition_of_the_median_ratio"]
    assert dec["distance_term"] > 1e3 > 2.0 >= dec["beam_term"] >= 1.0
    assert att["background_eirp_min_w"]["p05"] < att["background_eirp_min_w"]["p95"]


# ========================================================== 4. BOED queue wiring
def _job_store(tmp_path, graph, name="queue.sqlite"):
    s = JobStore(tmp_path / name)
    s.seed(graph, [JOB])
    return s


def test_seed_completes_the_job_from_the_pinned_artifact_and_keeps_the_prediction(
    tmp_path, graph, field
):
    s = _job_store(tmp_path, graph)
    try:
        row = s.get(JOB.job_id)
        assert row["status"] == "completed" and row["output_sha256"] == field.file_sha256
        assert row["output_path"] == BACKGROUND_SUMMARY_PATH
        assert row["code_git_sha"] == field.record.code_git_sha and GIT_SHA.match(
            row["code_git_sha"]
        )
        assert row["completed_at"] == field.record.assembled_at_utc
        # The registered EIG stays the Milestone 2 synthetic-proxy PREDICTION.
        assert row["expected_eig_bits"] == pytest.approx(0.0145226, rel=1e-4)
        details = json.loads(row["eig_details_json"])
        assert details["proxy_data_tier"] == "synthetic_proxy"
        assert details["by_radio_mode"]["flat_probability"]["eig_bits"] == pytest.approx(
            0.82734, rel=1e-4
        )
        assert "Wlodarczyk-Sroka 2020 method" in row["title"] and "288,315" not in row["title"]
        assert s.executions(JOB.job_id) == []  # a Tier-1 job is never run by the queue
        assert s.seed(graph, [JOB]) == [] and s.sync_archival_completions(graph, [JOB]) == []
        assert s.get(JOB.job_id) == row  # idempotent
    finally:
        s.close()


def test_completion_sync_is_non_destructive(tmp_path, graph, field):
    bare = _stripped_graph(tmp_path)
    assert runners.background_extension_completion(bare) is None
    assert runners.background_extension_evidence(bare, JOB.parameters) is None
    s = _job_store(tmp_path, bare)  # seeded BEFORE promotion: the job is open
    try:
        assert s.get(JOB.job_id)["status"] == "open" and s.get(JOB.job_id)["output_sha256"] is None
        predicted = s.get(JOB.job_id)["expected_eig_bits"]
        s.claim(JOB.job_id, "x")
        assert s.seed(graph, [JOB]) == []  # re-seed after promotion: nothing inserted ...
        row = s.get(JOB.job_id)
        assert row["status"] == "completed" and row["claimed_by"] == "x"  # ... claim kept
        assert row["output_sha256"] == field.file_sha256 and row["expected_eig_bits"] == predicted
        # An existing completion and a rejection are never rewritten.
        s.conn.execute("UPDATE jobs SET output_sha256='kept' WHERE job_id=?", (JOB.job_id,))
        s.conn.commit()
        s.seed(graph, [JOB])
        assert s.get(JOB.job_id)["output_sha256"] == "kept"
    finally:
        s.close()
    r = _job_store(tmp_path, bare, "rejected.sqlite")
    try:
        r.reject(JOB.job_id, "out of scope")
        r.seed(graph, [JOB])
        row = r.get(JOB.job_id)
        assert row["status"] == "rejected" and row["output_sha256"] is None
    finally:
        r.close()


def test_cli_background_field_report_and_receipt(tmp_path):
    res = CliRunner().invoke(app, ["background-field", "--out-dir", str(tmp_path)])
    out = " ".join(res.output.replace("│", "|").split())  # box-drawing column rule
    assert res.exit_code == 0, res.output
    for text in (
        "tier=reproduced_archive | N_bg=120176",
        "raw Gaia DR3 cone sightings | 259285",
        "suspected_unmatched_primary rows | 49",
        "duplicate Gaia ids vs bl_692_targets.csv | 0",
        "distance_scaled_eirp (physical)",
        "flat_probability (unphysical)",
        "Harris-FKG lower (used)",
        "UNVERIFIED",
        "not a verified bound",
    ):
        assert text in out, text
    (path,) = tmp_path.glob("background_field-*.json")
    doc = json.loads(path.read_text("utf-8"))
    assert doc["kind"] == "background_field" and doc["data_tier"] == "reproduced_archive"
    assert doc["results"]["provenance"]["n_background_stars"] == 120176
    assert doc["results"]["eig"]["scope"]["published"]["n_stars_total"] == 288315
    assert re.search(r'"gaia_dr3_source_id":\s*[0-9-]', path.read_text("utf-8")) is None
    again = CliRunner().invoke(app, ["background-field", "--out-dir", str(tmp_path)])
    assert doc["results_sha256"] in again.output.replace("\n", "")  # deterministic replay
