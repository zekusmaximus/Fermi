"""Hephaistos IR drill-down: blend-recovery grid -> screened calibrations -> IR likelihoods.

Macro/micro decoupling (AGENTS.md law 6): the image-level kernel
(``models/hephaistos.py``) produces clean false-positive / completeness objects;
this module only composes them with the survey counts and hypothesis signature
grids through ``models/infrared.py``.
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
from fermi_desk.registry import FactorGraph

SEPARATIONS_ARCSEC = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
FLUX_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
BASE_COMPLETENESS = 0.5  # illustrative: middle of the Milestone-1 completeness bracket


def run_ir_drilldown(
    graph: FactorGraph,
    *,
    seed: int,
    pair: tuple[str, str],
    n_trials: int = 40,
    snr: float = 12.5,
    sigma_sys_arcsec: float = 0.5,
) -> dict:
    spec = hephaistos.CutoutSpec(snr=snr, sigma_sys=sigma_sys_arcsec * u.arcsec)
    grid = hephaistos.blend_recovery_grid(
        spec,
        np.array(SEPARATIONS_ARCSEC) * u.arcsec,
        np.array(FLUX_FRACTIONS),
        n_trials,
        np.random.default_rng(np.random.SeedSequence(seed)),
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

    lr_rows = []
    for scen, raw in scenarios.items():
        for method, sc in screens.items():
            cal = hephaistos.screened_calibration(
                f"{scen}:{method}", BASE_COMPLETENESS, raw, sc["survival"], sc["false_alarm"]
            )
            sw = infrared.sweep_calibrations(k, n, grids[0], grids[1], [cal])
            row = sw.per_calibration[0]
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
    results = {
        "grid": grid.as_dict(),
        "screens_within_3p25_arcsec": screens,
        "anchors": {"status": anchors.status, "summary": anchors.summary, **anchors.details},
        "likelihood_ratio": {
            "numerator": pair[0],
            "denominator": pair[1],
            "observed_candidates": k,
            "n_targets": n,
            "base_completeness_illustrative": BASE_COMPLETENESS,
            "rows": lr_rows,
        },
        "data_tier": "illustrative_calibration",
        "anchor_data_tier": claim["data_tier"],
    }
    git_sha, git_head = git_state(graph.root)
    created = datetime.now(UTC)
    return {
        "receipt_schema_version": 1,
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
            "sigma_sys_arcsec": sigma_sys_arcsec,
        },
        "results": results,
        "results_sha256": canonical_sha256(results),
    }
