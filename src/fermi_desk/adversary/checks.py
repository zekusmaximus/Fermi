"""Adversary runner: automated reproductions of catalog and methodology failure modes.

Each audit returns a :class:`Finding` whose ``status`` is

* ``pass``    - the audited claim / invariant reproduces as stated;
* ``finding`` - a real issue or tension is surfaced (first-class output);
* ``fail``    - the audited claim does NOT reproduce (blocks promotion).

Audits read data contracts through ``registry`` and recompute physics through
the same kernels as the macro engine; they never patch numbers in.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from fermi_desk.errors import DoubleCountingError, FermiDeskError, MergePolicyError
from fermi_desk.inference.engine import (
    RunConfig,
    build_context,
    canonical_sha256,
    evaluate_hypothesis,
    expansion_model,
    git_state,
    radiometer,
    resolve_parameters,
)
from fermi_desk.models import hephaistos, radio
from fermi_desk.models.expansion import C_KPC_PER_GYR
from fermi_desk.registry import (
    MISSING_GAIA_ID,
    FactorGraph,
    MergedTargets,
    check_pointing_groups,
    load_background_summary,
    load_pointing_overlay,
    load_targets,
    resolve_pointing_groups,
    verify_catalog_checksum,
)
from fermi_desk.schemas import (
    POINTING_MERGE_POLICIES,
    REJECT_DUPLICATE_POINTING_GROUP,
    validate_pointing_merge_policy,
)

HIP1692_CLAIM = "data/adversary_claims/hip_1692.yaml"
HEPHAISTOS_DE_CLAIM = "data/adversary_claims/hephaistos_candidates_d_e.yaml"
DEC_SIGN_FLIP_CLAIM = "data/adversary_claims/isaacson_2017_dec_sign_flip.yaml"
LOCAL_VALIDITY_KPC = 8.0


@dataclass
class Finding:
    check: str
    status: str  # pass | finding | fail
    severity: str  # info | warning | critical
    summary: str
    details: dict = field(default_factory=dict)


def _yaml(graph: FactorGraph, rel: str) -> dict:
    return yaml.safe_load((graph.root / rel).read_text(encoding="utf-8"))


def _radio_survey(graph: FactorGraph):
    """The primary radio survey (a pointing extension is never a stand-alone survey)."""
    return next(
        s for s in graph.surveys.values() if s.channel == "radio" and not s.is_pointing_extension
    )


# ---------------------------------------------------------- distance revisions
def audit_distance_revision(
    graph: FactorGraph,
    claim_path: str,
    hypothesis_ids: Sequence[str],
    *,
    seed: int,
    n_mc: int = 4000,
    geometry: str = "thin_disk_2d",
) -> Finding:
    """Verify a target's distance revision under every recorded distance (data-driven).

    * EIRP_min(d) = 4 pi d^2 F_min in watts for each recorded distance;
    * every non-legacy distance must raise EIRP_min by more than the claim's
      ``min_eirp_factor_threshold``;
    * the canonical distance must be the one the catalog actually uses;
    * reports the resulting change in the distance_scaled_eirp radio likelihood.
    """
    claim = _yaml(graph, claim_path)
    sv = _radio_survey(graph)
    df = pd.read_csv(graph.root / sv.catalog_path)
    row = df.loc[df["target_name"] == claim["target_name"]]
    if len(row) != 1:
        return Finding("distance_revision", "fail", "critical", "target not in catalog", {})
    row = row.iloc[0]
    f_min = radio.min_detectable_flux(radiometer(sv))
    dists = {d["label"]: d for d in claim["distances"]}
    legacy_label = next(
        lb for lb, d in dists.items() if d.get("catalog_column") == "dist_legacy_pc"
    )
    legacy = dists[legacy_label]["value_pc"] * u.pc
    eirp_legacy = radio.min_detectable_eirp(legacy, f_min)
    threshold = float(claim["min_eirp_factor_threshold"])

    per = {}
    ok = True
    for label, d in dists.items():
        eirp = radio.min_detectable_eirp(d["value_pc"] * u.pc, f_min)
        factor = float((eirp / eirp_legacy).to_value(u.dimensionless_unscaled))
        exceeds = label == legacy_label or factor > threshold
        ok &= exceeds
        per[label] = {
            "distance_pc": float(d["value_pc"]),
            "data_tier": d["data_tier"],
            "eirp_min_w": float(eirp.to_value(u.W)),
            "eirp_factor_vs_legacy": factor,
            "exceeds_threshold": bool(exceeds),
        }
    canonical = claim["canonical_distance_label"]
    used_pc = float(row["dist_pc"])
    canon_matches = math.isclose(used_pc, dists[canonical]["value_pc"], rel_tol=1e-6)
    ok &= canon_matches

    # Likelihood delta: move the target to each recorded distance (line of sight fixed).
    cfg = RunConfig(
        geometry=geometry,
        radio_mode="distance_scaled_eirp",
        seed=seed,
        n_mc=n_mc,
        channels=("radio",),
    )
    ctx = build_context(graph, cfg)
    idx = int(np.flatnonzero(ctx.targets.names == claim["target_name"])[0])
    deltas: dict[str, dict] = {}
    for hid in hypothesis_ids:
        hyp = graph.hypotheses[hid]
        hp = resolve_parameters(graph, hyp, cfg.range_grid_points)
        lf = radio.LogNormalEIRP(
            hp.quantity("transmitter_eirp_median_w"), hp.value("transmitter_eirp_sigma_dex")
        )
        by_label = {}
        for label, d in dists.items():
            tt = ctx.targets.with_distance(idx, d["value_pc"] * u.pc, label)
            res = evaluate_hypothesis(hyp, replace(ctx, targets=tt))["radio"]
            by_label[label] = {
                "log10_lr_radio_mc": math.log10(res["p_zero_exact_mc"]),
                "mc_se_log10": res["p_zero_exact_mc_se"] / (res["p_zero_exact_mc"] * math.log(10)),
                "log10_lr_radio_fkg_bound": math.log10(res["p_zero_independent_fkg_lower"]),
                "expected_detections": res["expected_detections"],
                "target_p_detect_given_reached": float(
                    radio.detection_probabilities(
                        "distance_scaled_eirp",
                        hp.value("radio_detect_prob"),
                        tt.distance[idx : idx + 1],
                        radiometer(sv),
                        lf,
                    )[0]
                ),
            }
        base = by_label[canonical]
        for v in by_label.values():
            v["delta_log10_lr_fkg_vs_canonical"] = (
                v["log10_lr_radio_fkg_bound"] - base["log10_lr_radio_fkg_bound"]
            )
            v["delta_expected_detections_vs_canonical"] = (
                v["expected_detections"] - base["expected_detections"]
            )
        deltas[hid] = by_label
    status = "pass" if ok else "fail"
    return Finding(
        "distance_revision",
        status,
        "info" if ok else "critical",
        (
            f"{claim['target_name']}: EIRP_min raised "
            + ", ".join(
                f"{v['eirp_factor_vs_legacy']:.0f}x ({lb})"
                for lb, v in per.items()
                if lb != legacy_label
            )
            + f"; all > {threshold:.0f}x: {ok}; catalog uses canonical {canonical}: {canon_matches}"
        ),
        {
            "claim_path": claim_path,
            "target_name": claim["target_name"],
            "threshold": threshold,
            "f_min_w_m2": float(f_min.to_value(u.W / u.m**2)),
            "catalog_dist_pc": used_pc,
            "canonical_label": canonical,
            "historical_anchor_label": claim.get("historical_anchor_label"),
            "distances": per,
            "radio_likelihood_by_distance": deltas,
            "likelihood_note": (
                "distance_scaled_eirp radio channel, Earth-unreached conditioning on; deltas use "
                "the deterministic FKG (independent-product) bound because MC noise exceeds a "
                "single-target change; MC values reported with standard errors"
            ),
        },
    )


def audit_hip1692_distance_revision(
    graph: FactorGraph, hypothesis_ids: Sequence[str], *, seed: int, n_mc: int = 4000
) -> Finding:
    """Planning-specified HIP 1692 dual verification (canonical DR3 and DR2 anchor)."""
    f = audit_distance_revision(graph, HIP1692_CLAIM, hypothesis_ids, seed=seed, n_mc=n_mc)
    f.check = "hip1692_distance_revision"
    return f


# ------------------------------------------------- catalog declination sign flip
def audit_catalog_dec_sign_flip(
    graph: FactorGraph, claim_path: str = DEC_SIGN_FLIP_CLAIM
) -> Finding:
    """Reproduce the Isaacson et al. (2017) declination sign flip offline (ruling v0.2.1-R5).

    Reads only the locked primary catalog (checksum-verified, never written) and the claim
    file. Recounts the flipped declination band and its mirror, and for every recorded
    correction recomputes from the catalog row: that it sits in the flipped band, that its
    sign-restored position ``(ra, -dec)`` lies within the claim's tolerance of the recorded
    J2000 truth, and its angular offset from that truth. The declared northern controls must
    be in the band and uncorrected, and the band must hold nothing else. The secondary
    692stars.csv discrepancy is checked NOT to be inherited by the catalog.

    ``finding`` when everything reproduces (a real catalog erratum is surfaced); ``fail``
    when the claim does not reproduce against the locked catalog.
    """
    check = "catalog_dec_sign_audit"
    sv = _radio_survey(graph)
    try:
        overlay = load_pointing_overlay(graph.root, claim_path)
        path = verify_catalog_checksum(sv, graph.root)
    except (FermiDeskError, ValueError, OSError) as exc:
        return Finding(
            check,
            "fail",
            "critical",
            f"claim or locked catalog unreadable: {type(exc).__name__}: {exc}",
            {"claim_path": claim_path, "error": f"{type(exc).__name__}: {exc}"},
        )
    rec = overlay.record
    df = pd.read_csv(path, dtype={"gaia_dr3_source_id": str}).set_index("target_name")
    problems: list[str] = []
    if (rec.locked_catalog.path, rec.locked_catalog.sha256) != (sv.catalog_path, sv.catalog_sha256):
        problems.append("claim is tied to a different catalog than the locked primary list")

    census = rec.catalog_band_census
    (lo, hi), (mlo, mhi) = census.flipped_band_deg, census.mirror_band_deg
    dec = df["dec_deg"].to_numpy(dtype=float)
    band_names = sorted(df.index[(dec >= lo) & (dec < hi)])
    n_mirror = int(np.sum((dec >= mlo) & (dec < mhi)))
    if len(band_names) != census.n_targets_in_flipped_band:
        problems.append(
            f"{len(band_names)} catalog targets in [{lo:g}, {hi:g}) deg, claim expects "
            f"{census.n_targets_in_flipped_band}"
        )
    if n_mirror != census.n_targets_in_mirror_band:
        problems.append(
            f"{n_mirror} catalog targets in [{mlo:g}, {mhi:g}) deg, claim expects "
            f"{census.n_targets_in_mirror_band}"
        )

    def catalog_id(name: str) -> str | None:
        raw = df.at[name, "gaia_dr3_source_id"]
        return raw.strip() if isinstance(raw, str) and raw.strip() else None

    tol = rec.match_tolerance_arcsec * u.arcsec
    affected = []
    for c in rec.pointing_corrections:
        if c.target_name not in df.index:
            problems.append(f"{c.target_name}: not in the locked catalog")
            continue
        ra, de = float(df.at[c.target_name, "ra_deg"]), float(df.at[c.target_name, "dec_deg"])
        truth = SkyCoord(c.true_ra_deg * u.deg, c.true_dec_deg * u.deg)
        restored_sep = SkyCoord(ra * u.deg, -de * u.deg).separation(truth).to(u.arcsec)
        offset = SkyCoord(ra * u.deg, de * u.deg).separation(truth).to(u.deg)
        checks = {
            "catalog_dec_in_flipped_band": bool(lo <= de < hi),
            "true_dec_in_mirror_band": bool(mlo <= c.true_dec_deg < mhi),
            "catalog_coordinates_match_claim": bool(
                math.isclose(ra, c.catalog_ra_deg, abs_tol=1e-9)
                and math.isclose(de, c.catalog_dec_deg, abs_tol=1e-9)
            ),
            "gaia_dr3_source_id_matches_claim": catalog_id(c.target_name) == c.gaia_dr3_source_id,
            "sign_restored_matches_truth": bool(restored_sep <= tol),
            "offset_matches_claim": bool(abs(offset.to_value(u.deg) - c.offset_deg) <= 5e-4),
        }
        bad = sorted(k for k, ok in checks.items() if not ok)
        if bad:
            problems.append(f"{c.target_name}: {', '.join(bad)} failed")
        affected.append(
            {
                "target_name": c.target_name,
                "gaia_dr3_source_id": c.gaia_dr3_source_id,  # decimal string or None (R4)
                "catalog_ra_deg": ra,
                "catalog_dec_deg": de,
                "true_ra_deg": c.true_ra_deg,
                "true_dec_deg": c.true_dec_deg,
                "truth_source": c.truth_source,
                "truth_data_tier": rec.truth_sources[c.truth_source].data_tier,
                "offset_deg": float(offset.to_value(u.deg)),
                "sign_restored_separation_arcsec": float(restored_sep.to_value(u.arcsec)),
                "raw_isaacson_dec": c.raw_isaacson_dec,
                "raw_692stars_dec": c.raw_692stars_dec,
                "checks": checks,
            }
        )

    corrected = {c.target_name for c in rec.pointing_corrections}
    controls = []
    for k in rec.northern_controls:
        present = k.target_name in df.index
        de = float(df.at[k.target_name, "dec_deg"]) if present else float("nan")
        ok = (
            present
            and lo <= de < hi
            and math.isclose(de, k.catalog_dec_deg, abs_tol=1e-9)
            and catalog_id(k.target_name) == k.gaia_dr3_source_id
            and k.target_name not in corrected
        )
        if not ok:
            problems.append(f"northern control {k.target_name} does not reproduce")
        controls.append(
            {
                "target_name": k.target_name,
                "gaia_dr3_source_id": k.gaia_dr3_source_id,
                "catalog_dec_deg": de,
                "raw_isaacson_dec": k.raw_isaacson_dec,
                "raw_692stars_dec": k.raw_692stars_dec,
                "in_flipped_band": bool(present and lo <= de < hi),
                "corrected": k.target_name in corrected,
            }
        )
    unexplained = sorted(
        set(band_names) - corrected - {k.target_name for k in rec.northern_controls}
    )
    if unexplained:
        problems.append(f"band targets neither corrected nor declared controls: {unexplained}")

    secondary = []
    for d in rec.secondary_discrepancies:
        rows = []
        for r in d.records:
            present = r.target_name in df.index
            de = float(df.at[r.target_name, "dec_deg"]) if present else float("nan")
            agrees = present and abs(de - r.true_dec_deg) * u.deg <= tol
            if not (agrees and math.isclose(de, r.catalog_dec_deg, abs_tol=1e-9)):
                problems.append(
                    f"{d.id}: {r.target_name} is inherited by, or absent from, the catalog"
                )
            rows.append(
                {
                    "target_name": r.target_name,
                    "gaia_dr3_source_id": r.gaia_dr3_source_id,
                    "catalog_dec_deg": de,
                    "true_dec_deg": r.true_dec_deg,
                    "truth_source": r.truth_source,
                    "raw_isaacson_dec": r.raw_isaacson_dec,
                    "raw_692stars_dec": r.raw_692stars_dec,
                    "inherited_by_locked_catalog": not agrees,
                }
            )
        secondary.append(
            {
                "id": d.id,
                "data_tier": d.data_tier,
                "source": d.source,
                "description": " ".join(d.description.split()),
                "inherited_by_locked_catalog": any(r["inherited_by_locked_catalog"] for r in rows),
                "telescope_pointing_claim": d.telescope_pointing_claim,
                "records": rows,
            }
        )

    offsets = [a["offset_deg"] for a in affected] or [float("nan")]
    worst = max((a["sign_restored_separation_arcsec"] for a in affected), default=float("nan"))
    n_sec = sum(len(s["records"]) for s in secondary)
    status = "fail" if problems else "finding"
    summary = (
        "; ".join(problems)
        if problems
        else (
            f"{len(affected)} of {len(df)} primaries carry a sign-flipped declination (Isaacson et "
            f"al. 2017 Table 1): {n_mirror} catalog targets in [{mlo:g}, {mhi:g}) deg, "
            f"{len(band_names)} in [{lo:g}, {hi:g}) deg; offsets {min(offsets):.2f} - "
            f"{max(offsets):.2f} deg; sign-restored positions match the J2000 truth within "
            f"{worst:.2f} arcsec (tolerance {rec.match_tolerance_arcsec:g}); {len(controls)} "
            f"northern controls untouched; {n_sec} 692stars.csv -00 deg discrepancy record(s) not "
            f"inherited; locked catalog {str(sv.catalog_sha256)[:12]}... read-only"
        )
    )
    return Finding(
        check,
        status,
        "critical" if problems else "warning",
        summary,
        {
            "claim_path": claim_path,
            "claim_id": rec.id,
            "claim_sha256": overlay.sha256,
            "ruling": rec.ruling,
            "claim": " ".join(rec.claim.split()),
            "locked_catalog": {
                "path": sv.catalog_path,
                "sha256": sv.catalog_sha256,
                "checksum_verified": True,
                "modified_by_this_audit": False,
            },
            "band_census": {
                "flipped_band_deg": [lo, hi],
                "n_targets_in_flipped_band": len(band_names),
                "expected_in_flipped_band": census.n_targets_in_flipped_band,
                "mirror_band_deg": [mlo, mhi],
                "n_targets_in_mirror_band": n_mirror,
                "expected_in_mirror_band": census.n_targets_in_mirror_band,
            },
            "match_tolerance_arcsec": rec.match_tolerance_arcsec,
            "n_affected_targets": len(affected),
            "n_catalog_targets": len(df),
            "affected_targets": affected,
            "northern_controls": controls,
            "secondary_discrepancies": secondary,
            "evidence": [e.model_dump() for e in rec.evidence],
            "correction_policy": " ".join(
                rec.locked_catalog.model_dump().get("policy", "").split()
            ),
            "overlay_rule": " ".join(str(rec.model_dump().get("overlay_rule", "")).split()),
            "problems": problems,
        },
    )


# ------------------------------------------------------------- double counting
def audit_duplicate_pointings(graph: FactorGraph) -> Finding:
    surveys = list(graph.surveys.values())
    collisions = []
    try:
        check_pointing_groups(surveys)
    except DoubleCountingError as exc:
        collisions.append(str(exc))
    sv = _radio_survey(graph)
    probe = sv.model_copy(update={"id": f"{sv.id}__adversary_injected_duplicate"})
    try:
        check_pointing_groups([*surveys, probe])
        self_test = False
    except DoubleCountingError:
        self_test = True

    targets = load_targets(sv, graph.root)
    dup_names = sorted(pd.Series(targets.names).value_counts().loc[lambda s: s > 1].index)
    fwhm = (sv.radio_sensitivity.beam_fwhm_arcmin or 8.4) * u.arcmin
    sky = SkyCoord(targets.ra_deg * u.deg, targets.dec_deg * u.deg)
    i1, i2, sep, _ = sky.search_around_sky(sky, fwhm)
    pairs = [
        {
            "a": str(targets.names[a]),
            "b": str(targets.names[b]),
            "separation_arcmin": float(s.to_value(u.arcmin)),
            "separation_over_beam_fwhm": float((s / fwhm).to_value(u.dimensionless_unscaled)),
            "dist_a_pc": float(targets.distance[a].to_value(u.pc)),
            "dist_b_pc": float(targets.distance[b].to_value(u.pc)),
            "ra_a_deg": float(targets.ra_deg[a]),
            "dec_a_deg": float(targets.dec_deg[a]),
            "ra_b_deg": float(targets.ra_deg[b]),
            "dec_b_deg": float(targets.dec_deg[b]),
            "pointing_group_id": sv.pointing_group_id,
            "counted_once_each": True,
        }
        for a, b, s in zip(i1, i2, sep, strict=True)
        if a < b
    ]
    pairs.sort(key=lambda p: (p["a"], p["b"]))
    for i, p in enumerate(pairs, 1):
        p["pair_id"] = f"beam_pair_{i:02d}"
    status = (
        "fail" if collisions or dup_names or not self_test else ("finding" if pairs else "pass")
    )
    return Finding(
        "duplicate_pointings",
        status,
        "critical" if status == "fail" else "info",
        (
            f"cross-survey pointing collisions: {len(collisions)}; detector self-test "
            f"{'passed' if self_test else 'FAILED'}; duplicate target names: {len(dup_names)}; "
            f"target pairs sharing one GBT L-band beam (< {fwhm.to_value(u.arcmin):.1f} arcmin): "
            f"{len(pairs)}"
        ),
        {
            "collisions": collisions,
            "detector_self_test_passed": self_test,
            "duplicate_target_names": dup_names,
            "beam_fwhm_arcmin": float(fwhm.to_value(u.arcmin)),
            "beam_sharing_pairs": pairs,
            "n_beam_sharing_pairs": len(pairs),
            "beam_sharing_pair_schema": sorted(pairs[0]) if pairs else [],
            "interpretation": (
                "beam-sharing stars each enter the likelihood exactly once (no double counting); "
                "the extra coverage each gets from its neighbour's pointing is ignored, which is "
                "conservative (it can only understate the null's constraining power)"
            ),
        },
    )


def _recount_merge(merged: MergedTargets) -> dict:
    """Independent recount of a star-level merge (never trusts its own provenance)."""
    win, cl, gaia = merged.winner, merged.cluster, merged.gaia_dr3_source_id
    dup = merged.duplicate_gaia_ids()
    theta = merged.theta.to_value(u.arcmin)
    theta_min = np.full(len(win), np.inf)
    np.minimum.at(theta_min, cl, theta)
    not_min = int(np.sum(theta[win] != theta_min))
    # Unmerged collisions: a star with no Gaia id in any of its sightings lying closer than
    # the fallback radius to a different star - unless both are distinct targets of the
    # same primary list (published separate targets are never merged with each other).
    tol = merged.provenance.angular_fallback_arcsec
    star_has_id = np.zeros(len(win), bool)
    star_has_id[cl[gaia != MISSING_GAIA_ID]] = True
    primary_lists: dict[int, set[int]] = {}
    for r in np.flatnonzero(merged.row_role == "primary"):
        primary_lists.setdefault(int(cl[r]), set()).add(int(merged.survey_index[r]))
    unmerged: dict[tuple[int, int], dict] = {}
    rows = np.flatnonzero(~star_has_id[cl])
    if rows.size and len(cl) > 1:
        sky = SkyCoord(merged.ra_deg * u.deg, merged.dec_deg * u.deg)
        i1, i2, sep, _ = sky.search_around_sky(sky[rows], tol * u.arcsec)
        for a, b, s in zip(rows[i1], i2, sep.to_value(u.arcsec), strict=True):
            ka, kb = int(cl[a]), int(cl[b])
            if (
                ka == kb
                or s >= tol
                or (primary_lists.get(ka, set()) & primary_lists.get(kb, set()))
            ):
                continue
            unmerged.setdefault(
                (min(ka, kb), max(ka, kb)),
                {"a": str(merged.names[a]), "b": str(merged.names[b]), "separation_arcsec": s},
            )
    unmerged = list(unmerged.values())
    p = merged.provenance
    counts_ok = (
        p.n_raw_targets == len(merged.cluster)
        and p.n_merged_targets == len(win)
        and p.n_raw_targets - p.n_merged_targets == p.n_dedup_gaia_id + p.n_dedup_angular_fallback
    )
    return {
        "n_raw_targets_recounted": len(merged.cluster),
        "n_merged_targets_recounted": len(win),
        "n_duplicate_gaia_ids_post_merge": len(dup),
        "duplicate_gaia_ids": [int(x) for x in dup[:20]],
        "n_unmerged_angular_collisions": len(unmerged),
        "unmerged_angular_collisions": unmerged[:20],
        "n_kept_sightings_not_minimum_offset": not_min,
        "provenance_counts_consistent": bool(counts_ok),
        "merge_provenance": p.model_dump(),
    }


def audit_pointing_overlap(
    graph: FactorGraph,
    *,
    policy: str = REJECT_DUPLICATE_POINTING_GROUP,
    merged: MergedTargets | None = None,
) -> Finding:
    """Audit how surveys that share a pointing group are combined under ``policy``.

    Fails loudly when ``keep_first`` (or any unknown policy) is requested, when the registry
    would accept ``keep_first``, when surveys share a pointing group without a declared
    parent-pointing relationship (unmerged collision), or when a
    ``union_targets_by_pointing`` merge leaves a duplicated Gaia DR3 id, an unmerged
    angular collision, or a kept sighting that is not the minimum-offset one. ``merged``
    optionally audits a star-level merge directly (independent recount).
    """
    check = "pointing_overlap"
    details: dict = {"policy_requested": policy, "allowed_policies": list(POINTING_MERGE_POLICIES)}
    try:
        validate_pointing_merge_policy(policy)
    except MergePolicyError as exc:
        details["error"] = str(exc)
        return Finding(
            check, "fail", "critical", f"forbidden merge policy {policy!r}: {exc}", details
        )
    surveys = list(graph.surveys.values())
    try:
        resolve_pointing_groups(surveys, "keep_first")
        keep_first_refused = False
    except MergePolicyError:
        keep_first_refused = True
    details["keep_first_refused_by_registry"] = keep_first_refused
    problems = [] if keep_first_refused else ["registry accepted keep_first"]
    try:
        res = resolve_pointing_groups(surveys, policy)
    except DoubleCountingError as exc:
        details["unmerged_collisions"] = [str(exc)]
        return Finding(check, "fail", "critical", f"unmerged pointing collision: {exc}", details)
    details["held_out_extension_surveys"] = list(res.held_out)
    merges = []
    for m in res.merges:
        for ext in m.extensions:
            row: dict = {"pointing_group_id": m.pointing_group_id, "parent": m.parent.id}
            row["extension"] = ext.id
            try:
                bf = load_background_summary(ext, m.parent, graph.root)
            except (FermiDeskError, ValueError, OSError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                problems.append(f"{ext.id}: {row['error']}")
            else:
                da = bf.record.dedup_audit
                row.update(
                    data_tier=bf.record.data_tier,
                    summary_sha256=bf.file_sha256,
                    merge_provenance=bf.merge_provenance.model_dump(),
                    n_duplicate_gaia_ids_vs_primary=da.n_duplicate_gaia_ids_vs_primary,
                    n_duplicate_gaia_ids_within_background=da.n_duplicate_gaia_ids_within_background,
                    n_suspected_unmatched_primary=da.n_suspected_unmatched_primary,
                    offline_recomputation=bf.detail,
                )
            merges.append(row)
    details["merges"] = merges
    if merged is not None:
        rc = _recount_merge(merged)
        details["star_level_recount"] = rc
        if rc["n_duplicate_gaia_ids_post_merge"]:
            problems.append(
                f"{rc['n_duplicate_gaia_ids_post_merge']} duplicated Gaia ids post-merge"
            )
        if rc["n_unmerged_angular_collisions"]:
            problems.append(f"{rc['n_unmerged_angular_collisions']} unmerged angular collisions")
        if rc["n_kept_sightings_not_minimum_offset"]:
            problems.append("a kept sighting is not the minimum-offset one")
        if not rc["provenance_counts_consistent"]:
            problems.append("MergeProvenance counts disagree with the recount")
    details["problems"] = problems
    n_merged = len(merges) + (merged is not None)
    return Finding(
        check,
        "fail" if problems else "pass",
        "critical" if problems else "info",
        "; ".join(problems)
        if problems
        else (
            f"policy {res.policy}: keep_first refused; {n_merged} union merge(s) audited with 0 "
            f"duplicated Gaia ids; declared extensions held out of this run: "
            f"{[h['survey_id'] for h in res.held_out] or 'none'}"
        ),
        details,
    )


# ------------------------------------------------------- Hephaistos D/E anchors
def _verdict(p: float, false_alarm: float) -> str:
    if p < false_alarm:
        return "significant"
    return "marginal" if p < 0.05 else "no_evidence"


def audit_hephaistos_blend_anchors(graph: FactorGraph, spec: hephaistos.CutoutSpec) -> Finding:
    """Reproduce the published centroid verdicts for candidates D/E and test the blend model."""
    claim = _yaml(graph, HEPHAISTOS_DE_CLAIM)
    rows, ok, tension = [], True, []
    for c in claim["candidates"]:
        sep = c["background_separation_arcsec"] * u.arcsec
        lo, hi = c["w3_flux_fraction_bounds"]
        pred_lo = hephaistos.predicted_photocentre_shift(sep, lo).to_value(u.arcsec)
        pred_hi = hephaistos.predicted_photocentre_shift(sep, hi).to_value(u.arcsec)
        meas, err = c["w3_centroid_offset_arcsec"], c["w3_centroid_offset_err_arcsec"]
        z = (
            0.0
            if pred_lo <= meas <= pred_hi
            else min(abs(meas - pred_lo), abs(meas - pred_hi)) / err
        )
        p = hephaistos.centroid_p_value(meas * u.arcsec, err * u.arcsec, spec)
        verdict = _verdict(p, spec.false_alarm)
        match = verdict == c["published_centroid_verdict"]
        ok &= match
        if z > 2.0:
            tension.append(c["label"])
        rows.append(
            {
                "candidate": c["label"],
                "gaia_dr3_source_id": c["gaia_dr3_source_id"],
                "predicted_photocentre_shift_arcsec": [pred_lo, pred_hi],
                "measured_w3_offset_arcsec": meas,
                "measured_err_arcsec": err,
                "shift_tension_sigma": z,
                "centroid_screen_p_value": p,
                "reproduced_verdict": verdict,
                "published_verdict": c["published_centroid_verdict"],
                "verdict_matches": match,
            }
        )
    status = "fail" if not ok else ("finding" if tension else "pass")
    return Finding(
        "hephaistos_blend_anchors",
        status,
        "warning" if status != "pass" else "info",
        "centroid-screen verdicts reproduce published D/E outcomes: "
        f"{ok}; point-blend photocentre tension (>2 sigma) for: {tension or 'none'}",
        {
            "spec": spec.as_dict(),
            "candidates": rows,
            "note": (
                "Both candidates' contaminating galaxies sit ~1 arcsec away (JWST/MIRI), far "
                "below the 6.5 arcsec W3 FWHM: a centroid screen at WISE resolution is predicted "
                "to miss them, as published. Tension for an extended galaxy (E) is expected from "
                "a point-source blend model."
            ),
        },
    )


# --------------------------------------------------------- symmetry & validity
def audit_scaling_symmetry(graph: FactorGraph, pair: Sequence[str]) -> Finding:
    vals = {}
    for geom in ("thin_disk_2d", "thick_disk_3d"):
        cfg = RunConfig(geometry=geom, radio_mode="flat_probability", seed=0)
        ps = []
        for hid in pair:
            m = expansion_model(resolve_parameters(graph, graph.hypotheses[hid], 33), cfg)
            ps.append(m.earth_reach_probability())
        vals[geom] = {
            "p_earth_reached": dict(zip(pair, ps, strict=True)),
            "relative_difference": abs(ps[0] - ps[1]) / max(ps),
        }
    deg2d = vals["thin_disk_2d"]["relative_difference"] < 1e-12
    broken3d = vals["thick_disk_3d"]["relative_difference"] > 1e-6
    return Finding(
        "scaling_symmetry_2d_vs_3d",
        "pass" if deg2d and broken3d else "finding",
        "info",
        f"2D P(Earth reached) degenerate for {list(pair)}: {deg2d}; 3D relative split "
        f"{vals['thick_disk_3d']['relative_difference']:.3e}",
        vals,
    )


def audit_local_approximation(graph: FactorGraph, hypothesis_ids: Sequence[str]) -> Finding:
    cfg = RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=0)
    rows = []
    for hid in hypothesis_ids:
        params = resolve_parameters(graph, graph.hypotheses[hid], 33)
        if not params.has("expansion_speed_c"):
            continue
        m = expansion_model(params, cfg)
        v = params.value("expansion_speed_c") * C_KPC_PER_GYR
        rows.append(
            {
                "hypothesis_id": hid,
                "max_bubble_radius_kpc": float(m.max_bubble_radius.to_value(u.kpc)),
                "t_valid_gyr": LOCAL_VALIDITY_KPC / v if v > 0 else math.inf,
                "lookback_gyr": params.value("lookback_window_gyr"),
                "valid": m.local_approximation_valid,
            }
        )
    bad = [r["hypothesis_id"] for r in rows if not r["valid"]]
    return Finding(
        "local_approximation",
        "finding" if bad else "pass",
        "warning" if bad else "info",
        f"hypotheses outside the local infinite-disk validity (v*T > {LOCAL_VALIDITY_KPC} kpc): "
        f"{bad or 'none'}",
        {"rows": rows},
    )


def audit_illustrative_tiers(graph: FactorGraph) -> Finding:
    bad = []
    for sv in graph.surveys.values():
        if sv.infrared and sv.infrared.calibration_data_tier != "illustrative_calibration":
            bad.append(f"{sv.id}: calibrations tagged {sv.infrared.calibration_data_tier}")
        if sv.infrared and sv.data_tier == "published_catalog":
            bad.append(f"{sv.id}: survey tier published_catalog over illustrative calibrations")
    return Finding(
        "illustrative_tier_quarantine",
        "fail" if bad else "pass",
        "critical" if bad else "info",
        "illustrative IR calibrations are never labelled as published bounds"
        if not bad
        else "; ".join(bad),
        {"violations": bad},
    )


# ------------------------------------------------------------------- runner
def run_adversary(
    graph: FactorGraph,
    *,
    seed: int,
    pair: Sequence[str],
    n_mc: int = 4000,
) -> dict:
    spec = hephaistos.CutoutSpec()
    findings = [
        audit_hip1692_distance_revision(graph, pair, seed=seed, n_mc=n_mc),
        audit_catalog_dec_sign_flip(graph),
        audit_duplicate_pointings(graph),
        audit_pointing_overlap(graph),
        audit_hephaistos_blend_anchors(graph, spec),
        audit_scaling_symmetry(graph, pair),
        audit_local_approximation(graph, sorted(graph.hypotheses)),
        audit_illustrative_tiers(graph),
    ]
    results = {"findings": [asdict(f) for f in findings]}
    git_sha, git_head = git_state(graph.root)
    created = datetime.now(UTC)
    return {
        "receipt_schema_version": 1,
        "kind": "adversary",
        "run_id": f"adversary-{created.strftime('%Y%m%dT%H%M%S%fZ')}",
        "created_at_utc": created.isoformat(),
        "git_sha": git_sha,
        "git_head": git_head,
        "rng_seed": seed,
        "config": {"pair": list(pair), "n_mc": n_mc},
        "results": results,
        "results_sha256": canonical_sha256(results),
    }


def write_adversary_receipt(receipt: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{receipt['run_id']}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path
