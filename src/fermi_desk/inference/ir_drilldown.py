"""Hephaistos IR drill-down: blend-recovery grid -> screened calibrations -> IR likelihoods.

Macro/micro decoupling (AGENTS.md law 6): the image-level kernel
(``models/hephaistos.py``) produces clean false-positive / completeness objects;
this module only composes them with the survey counts and hypothesis signature
grids through ``models/infrared.py``.

Milestone 2.5: the drill-down is evaluated under every available W3 centroid
registration floor side by side - the legacy 0.50 arcsec baseline and, when the
survey pins one, the measured ``reproduced_archive`` floor from the live control
experiment. Both floors use the same RNG seed (common random numbers), so the
reported shifts are due to ``sigma_sys`` alone, not Monte Carlo noise. Both
published contamination conventions stay side by side; no winner is picked.
"""

from __future__ import annotations

from datetime import UTC, datetime

import astropy.units as u
import numpy as np
import yaml

from fermi_desk.adversary.checks import HEPHAISTOS_DE_CLAIM, audit_hephaistos_blend_anchors
from fermi_desk.inference.engine import (
    EXPANSION_PARAMS,
    RunConfig,
    build_context,
    canonical_sha256,
    expansion_model,
    git_state,
    ir_signature_grid,
    resolve_parameters,
)
from fermi_desk.models import hephaistos, infrared
from fermi_desk.registry import FactorGraph, load_sigma_sys_calibration

SEPARATIONS_ARCSEC = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
FLUX_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
BASE_COMPLETENESS = 0.5  # illustrative: middle of the Milestone-1 completeness bracket
LEGACY_SIGMA_SYS_ARCSEC = 0.5
LINKED_ARCHIVAL_JOB = "wise_independent_control_fields"
LEGACY = "legacy_baseline"
MEASURED = "measured_reproduced_archive"


def registration_floors(graph: FactorGraph, legacy_sigma_sys_arcsec: float) -> tuple[list, dict]:
    """Legacy baseline floor plus the survey's pinned measured floor (if any)."""
    floors = [
        hephaistos.RegistrationFloor(
            LEGACY,
            legacy_sigma_sys_arcsec * u.arcsec,
            "illustrative_calibration",
            "Suazo et al. 2024 Sec. 4: 'WISE astrometric precision of 0.5 arcsec' (WISE All-Sky "
            "Explanatory Supplement Sec. VI.4); Milestone 2 drill-down default; adopted, not "
            "measured on controls",
        )
    ]
    sv = next(s for s in graph.surveys.values() if s.channel == "infrared")
    cal = load_sigma_sys_calibration(sv, graph.root)
    info: dict = {"survey_id": sv.id, "measured_calibration": None}
    if cal is not None:
        floors.append(
            hephaistos.RegistrationFloor(
                MEASURED,
                cal.sigma_sys,
                cal.record.data_tier,
                f"{cal.path} ({cal.stats['primary_estimator']}, {cal.record.n_controls} "
                f"controls / {cal.record.n_patches_with_controls} patches, calibration half)",
                cal.interval_68,
            )
        )
        info["measured_calibration"] = cal.summary()
    return floors, info


def _fa(block: dict) -> dict:
    lo, hi = block["clopper_pearson_95"]
    return {
        "n_flagged": block["n_flagged"],
        "n_stars": block["n_stars"],
        "rate": block.get("rate", block.get("empirical_false_alarm")),
        "clopper_pearson_95": [lo, hi],
        "clopper_pearson_95_upper": hi,
        "ks_rayleigh_p_value": block.get("ks_rayleigh_p_value"),
        "binomial_p_at_least_k_given_nominal": block.get("binomial_p_at_least_k_given_nominal"),
        "nominal_false_alarm": block.get("nominal_false_alarm"),
        "sigma_sys_arcsec": block.get("sigma_sys_arcsec"),
    }


def small_sample_bounds(meas: dict | None) -> dict | None:
    """Structured small-sample false-alarm statements (Milestone 3 Part A.1)."""
    if meas is None:
        return None
    wh = meas["withheld_false_alarm"]
    out = {
        "linked_archival_job": LINKED_ARCHIVAL_JOB,
        "withheld_at_measured_floor": _fa(wh["out_of_sample_false_alarm_at_calibrated_floor"]),
        "withheld_at_legacy_floor": _fa(wh["false_alarm_at_legacy_baseline_floor"]),
        "withheld_with_no_floor": _fa(wh["false_alarm_with_no_floor"]),
        "calibration_in_sample_tail": _fa(meas["calibration_in_sample_false_alarm"]),
    }
    w = out["withheld_at_measured_floor"]
    c = out["calibration_in_sample_tail"]
    z = out["withheld_with_no_floor"]
    out["statement"] = (
        f"Withheld {w['n_flagged']}/{w['n_stars']} 3-sigma centroid false alarms at the measured "
        f"floor (Clopper-Pearson 95% upper bound {w['clopper_pearson_95_upper']:.3f}; Rayleigh KS "
        f"p = {w['ks_rayleigh_p_value']:.2g}); calibration-half in-sample tail "
        f"{c['n_flagged']}/{c['n_stars']} (binomial p = "
        f"{c['binomial_p_at_least_k_given_nominal']:.2g} vs nominal); no floor: "
        f"{z['n_flagged']}/{z['n_stars']}. n = {w['n_stars']} only bounds the rate below "
        f"{w['clopper_pearson_95_upper']:.1%}; the {LINKED_ARCHIVAL_JOB} job is the larger test."
    )
    return out


def control_experiment(meas: dict | None) -> dict | None:
    if meas is None:
        return None
    return {
        "n_controls": meas["n_controls"],
        "n_patches_with_controls": meas["n_patches_with_controls"],
        "n_calibration": meas["n_calibration"],
        "n_withheld": meas["n_withheld"],
        "min_abs_b_deg": meas["min_abs_b_deg"],
        "patches": meas["patches"],
        "cut_counts_total": meas["cut_counts_total"],
        "query_window_utc": meas["query_window_utc"],
        "code_git_sha": meas["code_git_sha"],
        "file_sha256": meas["file_sha256"],
    }


def atlas_smoothing_caveat(meas: dict | None) -> str:
    if meas is None:
        return "No measured sigma_sys calibration is pinned; the legacy floor is an adopted value."
    return (
        f"sigma_sys = {meas['sigma_sys_arcsec']:.3f} arcsec was measured on AllWISE Atlas W3 "
        "coadds, which are resampled and smoothed by the coadd interpolation kernel; the floor "
        "therefore absorbs Atlas image smoothing together with astrometric registration (and any "
        "departure of the real centroid noise from the FWHM / (2.355 SNR) model). It is not a pure "
        "registration error. Follow-up (SPEC.md v0.2): re-measure on unWISE coadds, which preserve "
        "the native PSF, to separate the two contributions."
    )


def anchor_benchmarks(by_floor: dict) -> list[dict]:
    """Candidates D / E under each floor: offset significance, p-value, verdict."""
    rows: dict[str, dict] = {}
    for label, block in by_floor.items():
        sys_arcsec = block["registration_floor"]["sigma_sys_arcsec"]
        for c in block["anchors"]["candidates"]:
            r = rows.setdefault(
                c["candidate"],
                {
                    "candidate": c["candidate"],
                    "gaia_dr3_source_id": c["gaia_dr3_source_id"],
                    "measured_w3_offset_arcsec": c["measured_w3_offset_arcsec"],
                    "measured_err_arcsec": c["measured_err_arcsec"],
                    "predicted_photocentre_shift_arcsec": c["predicted_photocentre_shift_arcsec"],
                    "shift_tension_sigma": c["shift_tension_sigma"],
                    "published_verdict": c["published_verdict"],
                    "by_floor": {},
                },
            )
            sigma = float(np.hypot(c["measured_err_arcsec"], sys_arcsec))
            r["by_floor"][label] = {
                "sigma_sys_arcsec": sys_arcsec,
                "combined_sigma_arcsec": sigma,
                "offset_significance_sigma": c["measured_w3_offset_arcsec"] / sigma,
                "centroid_screen_p_value": c["centroid_screen_p_value"],
                "reproduced_verdict": c["reproduced_verdict"],
                "verdict_matches_published": c["verdict_matches"],
            }
    return list(rows.values())


def contamination_flip(by_floor: dict, scenarios: dict) -> dict:
    """LR under both published contamination conventions; no convention is preferred."""
    table = []
    for label, block in by_floor.items():
        for screen in ("none", "centroid", "image"):
            row = {"registration_floor": label, "screen": screen}
            for r in block["likelihood_ratio_rows"]:
                if r["screen"] == screen:
                    row[r["contamination_scenario"]] = {"lr": r["lr"], "log10_lr": r["log10_lr"]}
            table.append(row)
    return {
        "scenarios": scenarios,
        "rows": table,
        "policy": "both published conventions side by side; neither is preferred",
        "lr_reading": "LR < 1 favours the denominator (conspicuous Dyson), LR > 1 the numerator",
    }


def _evaluate_floor(graph, floor, seed, n_trials, snr, scenarios, grids, k, n) -> tuple:
    spec = hephaistos.CutoutSpec(snr=snr, sigma_sys=floor.sigma_sys)
    grid = hephaistos.blend_recovery_grid(
        spec,
        np.array(SEPARATIONS_ARCSEC) * u.arcsec,
        np.array(FLUX_FRACTIONS),
        n_trials,
        np.random.default_rng(np.random.SeedSequence(seed)),  # common random numbers
    )
    max_sep = 3.25 * u.arcsec
    screens = {
        "none": {"survival": 1.0, "false_alarm": 0.0},
        "centroid": {
            "survival": hephaistos.blend_survival(grid, "centroid", max_sep),
            "false_alarm": grid.centroid_false_alarm,
        },
        "image": {
            "survival": hephaistos.blend_survival(grid, "image", max_sep),
            "false_alarm": grid.image_false_alarm,
        },
    }
    lr_rows = []
    for scen, raw in scenarios.items():
        for method, sc in screens.items():
            cal = hephaistos.screened_calibration(
                f"{scen}:{method}", BASE_COMPLETENESS, raw, sc["survival"], sc["false_alarm"]
            )
            row = infrared.sweep_calibrations(k, n, grids[0], grids[1], [cal]).per_calibration[0]
            lr_rows.append(
                {
                    "contamination_scenario": scen,
                    "raw_blend_rate": raw,
                    "screen": method,
                    "blend_survival": sc["survival"],
                    "screen_false_alarm": sc["false_alarm"],
                    "effective_false_positive_rate": cal.false_positive_rate,
                    "expected_false_positives": cal.false_positive_rate * n,
                    "completeness": cal.completeness,
                    "lr": row["lr"],
                    "log10_lr": row["log10_lr"],
                }
            )
    anchors = audit_hephaistos_blend_anchors(graph, spec)
    block = {
        "registration_floor": floor.as_dict(),
        "grid": grid.as_dict(),
        "screens_within_3p25_arcsec": screens,
        "anchors": {"status": anchors.status, "summary": anchors.summary, **anchors.details},
        "likelihood_ratio_rows": lr_rows,
    }
    return block, grid


def run_ir_drilldown(
    graph: FactorGraph,
    *,
    seed: int,
    pair: tuple[str, str],
    n_trials: int = 40,
    snr: float = 12.5,
    legacy_sigma_sys_arcsec: float = LEGACY_SIGMA_SYS_ARCSEC,
) -> dict:
    claim = yaml.safe_load((graph.root / HEPHAISTOS_DE_CLAIM).read_text("utf-8"))
    cf = claim["contamination_fractions"]
    rate = float(cf["within_3p25_arcsec"])
    scenarios = {
        "suazo_2024_subsample_convention": rate,
        "zackrisson_2026_full_sample_critique": rate
        * float(cf["zackrisson_2026_full_sample"])
        / float(cf["suazo_2024_applied_sample"]),
    }

    cfg = RunConfig(
        geometry="thin_disk_2d", radio_mode="flat_probability", seed=seed, channels=("infrared",)
    )
    ctx = build_context(graph, cfg)
    sv = ctx.surveys["infrared"]
    k, n = sv.observed_detections, sv.n_targets
    grids = []
    for hid in pair:
        params = resolve_parameters(graph, graph.hypotheses[hid], cfg.range_grid_points)
        has_exp = all(params.has(p) for p in EXPANSION_PARAMS)
        grids.append(
            ir_signature_grid(params, expansion_model(params, cfg) if has_exp else None, ctx)[0]
        )

    floors, floor_info = registration_floors(graph, legacy_sigma_sys_arcsec)
    by_floor: dict[str, dict] = {}
    raw_grids: dict[str, hephaistos.BlendGrid] = {}
    for floor in floors:
        by_floor[floor.label], raw_grids[floor.label] = _evaluate_floor(
            graph, floor, seed, n_trials, snr, scenarios, grids, k, n
        )

    shift = None
    if MEASURED in by_floor:
        base, meas = by_floor[LEGACY], by_floor[MEASURED]
        shift = {
            "from": LEGACY,
            "to": MEASURED,
            "sigma_sys_arcsec": [
                base["registration_floor"]["sigma_sys_arcsec"],
                meas["registration_floor"]["sigma_sys_arcsec"],
            ],
            **hephaistos.grid_shift(raw_grids[LEGACY], raw_grids[MEASURED]),
            "survival_delta": {
                m: meas["screens_within_3p25_arcsec"][m]["survival"]
                - base["screens_within_3p25_arcsec"][m]["survival"]
                for m in ("centroid", "image")
            },
            "anchor_verdicts": {
                c0["candidate"]: {
                    LEGACY: c0["reproduced_verdict"],
                    MEASURED: c1["reproduced_verdict"],
                    "published": c0["published_verdict"],
                    "p_value": [c0["centroid_screen_p_value"], c1["centroid_screen_p_value"]],
                }
                for c0, c1 in zip(
                    base["anchors"]["candidates"], meas["anchors"]["candidates"], strict=True
                )
            },
            "log10_lr_delta": [
                {
                    "contamination_scenario": r0["contamination_scenario"],
                    "screen": r0["screen"],
                    "delta": r1["log10_lr"] - r0["log10_lr"],
                }
                for r0, r1 in zip(
                    base["likelihood_ratio_rows"], meas["likelihood_ratio_rows"], strict=True
                )
            ],
        }

    meas = floor_info["measured_calibration"]
    results = {
        "registration_floors": [f.as_dict() for f in floors],
        "small_sample_false_alarm_bounds": small_sample_bounds(meas),
        "control_experiment": control_experiment(meas),
        "atlas_smoothing_caveat": atlas_smoothing_caveat(meas),
        "anchor_benchmarks": anchor_benchmarks(by_floor),
        "contamination_flip": contamination_flip(by_floor, scenarios),
        "registration_floor_sources": floor_info,
        "by_floor": by_floor,
        "floor_shift": shift,
        "likelihood_ratio": {
            "numerator": pair[0],
            "denominator": pair[1],
            "observed_candidates": k,
            "n_targets": n,
            "base_completeness_illustrative": BASE_COMPLETENESS,
            "contamination_scenarios": scenarios,
            "scenario_policy": "both published conventions reported side by side; none preferred",
        },
        "data_tier": "illustrative_calibration",  # synthetic injection-recovery images
        "anchor_data_tier": claim["data_tier"],
    }
    git_sha, git_head = git_state(graph.root)
    created = datetime.now(UTC)
    return {
        "receipt_schema_version": 2,
        "kind": "ir_drilldown",
        "run_id": f"ir-drilldown-{created.strftime('%Y%m%dT%H%M%S%fZ')}",
        "created_at_utc": created.isoformat(),
        "git_sha": git_sha,
        "git_head": git_head,
        "rng_seed": seed,
        "config": {
            "pair": list(pair),
            "n_trials": n_trials,
            "snr": snr,
            "legacy_sigma_sys_arcsec": legacy_sigma_sys_arcsec,
            "registration_floors": [f.label for f in floors],
        },
        "results": results,
        "results_sha256": canonical_sha256(results),
    }
