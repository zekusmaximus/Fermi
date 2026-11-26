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

from fermi_desk.errors import DoubleCountingError
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
from fermi_desk.registry import FactorGraph, check_pointing_groups, load_targets

HIP1692_CLAIM = "data/adversary_claims/hip_1692.yaml"
HEPHAISTOS_DE_CLAIM = "data/adversary_claims/hephaistos_candidates_d_e.yaml"
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
    return next(s for s in graph.surveys.values() if s.channel == "radio")


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
            "dist_a_pc": float(targets.distance[a].to_value(u.pc)),
            "dist_b_pc": float(targets.distance[b].to_value(u.pc)),
        }
        for a, b, s in zip(i1, i2, sep, strict=True)
        if a < b
    ]
    pairs.sort(key=lambda p: (p["a"], p["b"]))
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
            "interpretation": (
                "beam-sharing stars each enter the likelihood exactly once (no double counting); "
                "the extra coverage each gets from its neighbour's pointing is ignored, which is "
                "conservative (it can only understate the null's constraining power)"
            ),
        },
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
        audit_duplicate_pointings(graph),
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
