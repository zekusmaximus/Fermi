"""Deterministic job runners (Tier-0 compute) and EIG predictions for queue jobs.

Runner outputs never contain timestamps, paths, or git state, so that
``output_sha256`` depends only on (job_id, parameters, rng_seed, code).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import astropy.units as u
import numpy as np
import yaml

from fermi_desk.inference import eig
from fermi_desk.inference.engine import (
    RunConfig,
    build_context,
    expansion_model,
    ir_calibrations,
    resolve_parameters,
    score_hypotheses,
)
from fermi_desk.models import hephaistos, radio
from fermi_desk.models.expansion import C_KPC_PER_GYR
from fermi_desk.registry import FactorGraph

LOCAL_VALIDITY_KPC = 8.0


def _uniform(keys) -> dict[str, float]:
    keys = list(keys)
    return {k: 1.0 / len(keys) for k in keys}


# --------------------------------------------------------------- Tier-0 runners
def run_lookback_window_t_sweep(graph: FactorGraph, p: Mapping, seed: int) -> dict:
    """Sweep T for the benchmark pair in each geometry; flag the local-validity threshold."""
    a, b = p["pair"]
    t_grid = np.geomspace(p["t_min_gyr"], p["t_max_gyr"], int(p["n_points"]))
    thresholds = {}
    for hid in (a, b):
        v = resolve_parameters(graph, graph.hypotheses[hid], 33).value("expansion_speed_c")
        thresholds[hid] = {
            "expansion_speed_c": v,
            "t_valid_gyr": LOCAL_VALIDITY_KPC / (v * C_KPC_PER_GYR),
            "criterion": f"v_exp * T > {LOCAL_VALIDITY_KPC} kpc => local_approx VIOLATED",
        }
    rows = []
    for geom in p["geometries"]:
        for t in t_grid:
            cfg = RunConfig(
                geometry=geom,
                radio_mode=p["radio_mode"],
                seed=seed,
                n_mc=int(p["n_mc"]),
                channels=tuple(p["channels"]),
                parameter_overrides=(("lookback_window_gyr", float(t)),),
                headline_pair=(a, b),
            )
            _, res = score_hypotheses(graph, cfg, [a, b])
            hl = res["headline"]
            per = {}
            for hid in (a, b):
                h = res["hypotheses"][hid]
                per[hid] = {
                    "p_earth_reached": h["expansion"]["p_earth_reached"],
                    "expected_reached_given_earth_unreached": h["radio"]["expected_reached"],
                    "max_bubble_radius_kpc": h["expansion"]["max_bubble_radius_kpc"],
                    "local_approx": "ok"
                    if h["expansion"]["local_approximation_valid"]
                    else "VIOLATED",
                }
            rows.append(
                {
                    "geometry": geom,
                    "lookback_gyr": float(t),
                    "hypotheses": per,
                    "log10_bf_radio": hl["log10_bf_by_term"].get("radio"),
                    "log10_bf_joint": hl["log10_bf"],
                    "log10_bf_joint_excluding_illustrative": hl["log10_bf_excluding_illustrative"],
                    "log10_bf_headline_verified": hl["verified"]["log10_bf"],
                    "bf_terms": hl["terms"],
                    "headline_verified_terms": hl["verified"]["terms"],
                }
            )
    return {"pair": [a, b], "t_grid_gyr": t_grid.tolist(), "thresholds": thresholds, "rows": rows}


def run_disk_scale_height_sweep(graph: FactorGraph, p: Mapping, seed: int) -> dict:
    a, b = p["pair"]
    rows = []
    for h in p["scale_heights_pc"]:
        cfg = RunConfig(
            geometry="thick_disk_3d",
            radio_mode=p["radio_mode"],
            seed=seed,
            n_mc=int(p["n_mc"]),
            channels=tuple(p["channels"]),
            parameter_overrides=(("disk_scale_height_pc", float(h)),),
            headline_pair=(a, b),
        )
        _, res = score_hypotheses(graph, cfg, [a, b])
        rows.append(
            {
                "scale_height_pc": float(h),
                "p_earth_reached": {
                    k: res["hypotheses"][k]["expansion"]["p_earth_reached"] for k in (a, b)
                },
                "log10_bf_by_term": res["headline"]["log10_bf_by_term"],
                "log10_bf_joint": res["headline"]["log10_bf"],
                "log10_bf_headline_verified": res["headline"]["verified"]["log10_bf"],
            }
        )
    return {"pair": [a, b], "rows": rows}


def run_eirp_luminosity_function_sweep(graph: FactorGraph, p: Mapping, seed: int) -> dict:
    a, b = p["pair"]
    rows = []
    for med in p["medians_w"]:
        for sig in p["sigmas_dex"]:
            cfg = RunConfig(
                geometry=p["geometry"],
                radio_mode="distance_scaled_eirp",
                seed=seed,
                n_mc=int(p["n_mc"]),
                channels=("radio",),
                parameter_overrides=(
                    ("transmitter_eirp_median_w", float(med)),
                    ("transmitter_eirp_sigma_dex", float(sig)),
                ),
                headline_pair=(a, b),
            )
            _, res = score_hypotheses(graph, cfg, [a, b])
            rows.append(
                {
                    "median_w": float(med),
                    "sigma_dex": float(sig),
                    "log10_lr_radio": {
                        k: res["hypotheses"][k]["components"]["radio"]["log10_lr_vs_null"]
                        for k in (a, b)
                    },
                    "mean_p_detect_given_reached": {
                        k: res["hypotheses"][k]["radio"]["mean_p_detect_given_reached"]
                        for k in (a, b)
                    },
                    "log10_bf_radio": res["headline"]["log10_bf"],
                }
            )
    return {"pair": [a, b], "rows": rows}


def run_hephaistos_blend_recovery_grid(graph: FactorGraph, p: Mapping, seed: int) -> dict:
    spec = hephaistos.CutoutSpec(
        psf_fwhm=p["psf_fwhm_arcsec"] * u.arcsec,
        sigma_sys=p["sigma_sys_arcsec"] * u.arcsec,
        snr=p["snr"],
    )
    grid = hephaistos.blend_recovery_grid(
        spec,
        np.array(p["separations_arcsec"]) * u.arcsec,
        np.array(p["flux_fractions"]),
        int(p["n_trials"]),
        np.random.default_rng(np.random.SeedSequence(seed)),
    )
    return {
        "grid": grid.as_dict(),
        "survival_within_3p25_arcsec": {
            m: hephaistos.blend_survival(grid, m, 3.25 * u.arcsec) for m in ("centroid", "image")
        },
    }


TIER0_RUNNERS: dict[str, Callable[[FactorGraph, Mapping, int], dict]] = {
    "lookback_window_t_sweep": run_lookback_window_t_sweep,
    "disk_scale_height_sweep": run_disk_scale_height_sweep,
    "eirp_luminosity_function_sweep": run_eirp_luminosity_function_sweep,
    "hephaistos_blend_recovery_grid": run_hephaistos_blend_recovery_grid,
}


# ------------------------------------------- Tier-0 diagnostic-impact statements
# Each statement is COMPUTED from the job parameters and the registry, so it cannot
# drift from what the job actually sweeps. Tier-0 EIG stays exactly 0 bits.
def _span(values, fmt: str = ".3g") -> str:
    lo, hi = min(values), max(values)
    return f"{lo:{fmt}}-{hi:{fmt}}"


def _duration(t_gyr: float) -> str:
    return f"{t_gyr * 1e6:.3g} kyr" if t_gyr < 1e-3 else f"{t_gyr * 1e3:.3g} Myr"


def diagnose_lookback_window_t_sweep(graph: FactorGraph, p: Mapping) -> str:
    cut = []
    for hid in p["pair"]:
        v = resolve_parameters(graph, graph.hypotheses[hid], 33).value("expansion_speed_c")
        cut.append(f"{_duration(LOCAL_VALIDITY_KPC / (v * C_KPC_PER_GYR))} ({hid}, v_exp={v:g} c)")
    return (
        f"Exposes the local infinite-disk validity cutoff T_valid = {LOCAL_VALIDITY_KPC:g} kpc / "
        f"v_exp: {'; '.join(cut)}; and the headline BF sensitivity across T = "
        f"{_duration(float(p['t_min_gyr']))}-{_duration(float(p['t_max_gyr']))} in "
        f"{' and '.join(p['geometries'])}"
    )


def diagnose_hephaistos_blend_recovery_grid(graph: FactorGraph, p: Mapping) -> str:
    return (
        f"Maps centroid vs. image-level blend recovery over theta = "
        f"{_span(p['separations_arcsec'])} arcsec x f_bg = {_span(p['flux_fractions'])} at W3 "
        f"SNR {p['snr']:g}, sigma_sys {p['sigma_sys_arcsec']:g} arcsec; fixes the blend-survival "
        "and false-alarm rates that turn the illustrative IR false-positive bracket into "
        "screened calibrations"
    )


def diagnose_disk_scale_height_sweep(graph: FactorGraph, p: Mapping) -> str:
    h0 = graph.registry.get("disk_scale_height_pc").default
    return (
        f"Measures thick_disk_3d headline BF sensitivity to the scale height h_z = "
        f"{_span(p['scale_heights_pc'], '.0f')} pc (registry default {h0:g} pc), the scale at "
        "which the 2D Gamma*v^2 degeneracy of the benchmark pair breaks"
    )


def diagnose_eirp_luminosity_function_sweep(graph: FactorGraph, p: Mapping) -> str:
    m0 = graph.registry.get("transmitter_eirp_median_w").default
    s0 = graph.registry.get("transmitter_eirp_sigma_dex").default
    return (
        f"Bounds how far the illustrative log-normal transmitter EIRP luminosity function "
        f"(median {_span(p['medians_w'], '.0e')} W, sigma {_span(p['sigmas_dex'], 'g')} dex; "
        f"registry default {m0:.0e} W, {s0:g} dex) moves the distance_scaled_eirp radio BF"
    )


TIER0_DIAGNOSTICS: dict[str, Callable[[FactorGraph, Mapping], str]] = {
    "lookback_window_t_sweep": diagnose_lookback_window_t_sweep,
    "disk_scale_height_sweep": diagnose_disk_scale_height_sweep,
    "eirp_luminosity_function_sweep": diagnose_eirp_luminosity_function_sweep,
    "hephaistos_blend_recovery_grid": diagnose_hephaistos_blend_recovery_grid,
}


# ------------------------------------------------------------- EIG predictions
def eig_background_extension(graph: FactorGraph, p: Mapping) -> eig.EIGResult:
    """EIG of adding the published 288,315-star background null (binary outcome).

    Proxy population: quantile points of the published GBT EIRP_min shell CDF
    (Wlodarczyk-Sroka 2020 Table 2), placed along BL pointing directions at the
    on-axis-equivalent distance d = 50 pc * sqrt(EIRP_min / 2.1e12 W).
    P(0 detections | H) uses the independent-product (FKG lower bound) form.
    """
    shells = yaml.safe_load((graph.root / p["shell_table"]).read_text("utf-8"))
    edges = np.array([float(s[0]) for s in shells["shells"]])
    cum = np.array([float(s[1]) for s in shells["shells"]])
    log_e = np.log10(np.concatenate([[float(p["lowest_eirp_w"])], edges]))
    cum = np.concatenate([[0.0], cum])
    n_total = cum[-1]
    k = int(p["n_quantiles"])
    eirp = 10 ** np.interp((np.arange(k) + 0.5) / k * n_total, cum, log_e)
    d_eff = 50.0 * np.sqrt(eirp / float(shells["reference_eirp_w_at_50pc"]))

    a, b = p["pair"]
    cfg = RunConfig(geometry=p["geometry"], radio_mode=p["radio_mode"], seed=0, channels=("radio",))
    ctx = build_context(graph, cfg)
    order = np.argsort(ctx.targets.names.astype(str))
    pick = order[:: max(1, len(order) // int(p["n_directions"]))][: int(p["n_directions"])]
    unit = ctx.targets.xyz.to_value(u.pc)[pick]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    pts = (d_eff[:, None, None] * unit[None, :, :]).reshape(-1, 3) * u.pc
    eirp_pts = np.repeat(eirp, len(pick))
    w_star = n_total / len(eirp_pts)

    preds = {}
    per_mode = {}
    for mode in ("flat_probability", "distance_scaled_eirp"):
        p0 = {}
        for hid in (a, b):
            params = resolve_parameters(graph, graph.hypotheses[hid], 33)
            model = expansion_model(params, cfg)
            q = model.reach_probabilities(pts, cfg.condition_on_earth_unreached)
            p_r = params.value("radio_detect_prob")
            if mode == "flat_probability":
                pd_ = np.full(len(q), p_r)
            else:
                lf = radio.LogNormalEIRP(
                    params.quantity("transmitter_eirp_median_w"),
                    params.value("transmitter_eirp_sigma_dex"),
                )
                pd_ = p_r * lf.survival(eirp_pts * u.W)
            p0[hid] = float(math.exp(w_star * np.sum(np.log1p(-pd_ * q))))
        prior = _uniform((a, b))
        bits = eig.mutual_information_bits(prior, {h: eig.binary_outcome(v) for h, v in p0.items()})
        per_mode[mode] = {"p_zero_detections": p0, "eig_bits": bits}
        if mode == p["radio_mode"]:
            preds = per_mode[mode]
    return eig.EIGResult(
        bits=preds["eig_bits"],
        prior={"type": "uniform", "over": [a, b]},
        basis=(
            "I(H;Y), Y = {0 detections, >=1} among the published GBT background stars; proxy "
            "population from Wlodarczyk-Sroka 2020 Table 2 shells; independent-product P0"
        ),
        details={
            "n_stars_gbt": n_total,
            "proxy_points": len(eirp_pts),
            "radio_mode_primary": p["radio_mode"],
            "by_radio_mode": per_mode,
            "proxy_data_tier": "synthetic_proxy",
        },
    )


def eig_control_fields(graph: FactorGraph, p: Mapping) -> eig.EIGResult:
    """EIG about which IR calibration holds, from false-positive counts in control stars."""
    sv = graph.surveys[p["ir_survey"]]
    cals = ir_calibrations(sv)
    rates = {c.id: c.false_positive_rate for c in cals}
    prior = _uniform(rates)
    outcomes = eig.binomial_count_outcomes(int(p["n_control_stars"]), rates)
    bits = eig.mutual_information_bits(prior, outcomes)
    distinct = len(set(rates.values()))
    return eig.EIGResult(
        bits=bits,
        prior={"type": "uniform", "over": sorted(rates)},
        basis=(
            "I(C;K), K ~ Binomial(n_control, alpha_C) false-positive flags among blend-free "
            "control stars; completeness is unidentifiable from controls"
        ),
        details={
            "n_control_stars": int(p["n_control_stars"]),
            "prior_entropy_bits": eig.entropy_bits(prior),
            "max_identifiable_bits": math.log2(distinct),
            "distinct_false_positive_rates": distinct,
            "support_size": len(next(iter(outcomes.values()))),
        },
    )


ARCHIVAL_EIG: dict[str, Callable[[FactorGraph, Mapping], eig.EIGResult]] = {
    "wlodarczyk_sroka_288k_background_extension": eig_background_extension,
    "wise_independent_control_fields": eig_control_fields,
}


def tier0_eig() -> eig.EIGResult:
    return eig.EIGResult(
        bits=0.0,
        prior={"type": "none"},
        basis=(
            "Tier-0 compute job: no new observational data enter, so I(H; output) = 0 exactly; "
            "value is diagnostic (see diagnostic_impact / diagnostic_priority), not information "
            "about hypotheses"
        ),
        details={},
    )
