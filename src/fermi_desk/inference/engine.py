"""Joint likelihood engine with separate Earth-unreached conditioning.

Likelihood normalisation (SPEC.md Sec. 4.3 ruling): every component reports

* ``log10_p_data_given_h``  - raw log10 P(observed data | H);
* ``log10_null``            - the same quantity under the component's
  hypothesis-independent null model (no settlement / zero emission / silent sky);
* ``log10_lr_vs_null``      - their difference.

For the zero-detection radio, Solar System and Earth-unreached terms the null
probability is exactly 1, so raw and LR coincide; for the mid-IR candidate count
they do not. Joint values are always labelled with their channel set, and
pairwise Bayes factors are computed only over the intersection of the two
hypotheses' active channels.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np

from fermi_desk.errors import GeometryError, SchemaValidationError, Tier3ScoringError
from fermi_desk.models import infrared, radio, solar_system
from fermi_desk.models.expansion import GEOMETRY_MODES, ExpansionModel
from fermi_desk.registry import FactorGraph, TargetTable, check_pointing_groups, load_targets
from fermi_desk.schemas import Hypothesis, Survey

LN10 = math.log(10.0)
CHANNELS = ("radio", "infrared", "solar_system")
# Data tiers that are NOT verified observational inputs (SPEC.md Sec. 2 invariant 6).
ILLUSTRATIVE_TIERS = frozenset(
    {"illustrative_calibration", "synthetic_proxy", "synthetic_proxy_692"}
)
RECEIPT_SCHEMA_VERSION = 4
# Why each illustrative input is illustrative (surfaced as headline_exclusion_notes).
ILLUSTRATIVE_INPUT_REASONS = {
    "transmitter_eirp_luminosity_function": (
        "distance_scaled_eirp mode: the log-normal transmitter EIRP luminosity function "
        "(median / sigma_dex) is illustrative_calibration, not a measured population, so the "
        "radio term is held out of the verified headline"
    ),
    "ir_calibration_bracket": (
        "the mid-IR completeness / false-positive calibration bracket is "
        "illustrative_calibration (a sensitivity bracket, not a measured calibration)"
    ),
    "solar_system_null": (
        "the Solar System artifact null is a qualitative literature summary paired with an "
        "illustrative_calibration detection probability"
    ),
}
HEADLINE_VERIFIED = "verified_channels_only"
HEADLINE_WITH_ILLUSTRATIVE = "including_illustrative_channels"
EXPANSION_PARAMS = ("origin_rate_per_kpc2_gyr", "expansion_speed_c")
EARTH = "earth_unreached"
RADIO_SS = "radio+solar_system"
NULL_MODELS = {
    EARTH: "no settlement (Gamma_2d = 0): P(Earth unreached) = 1",
    "radio": "zero radio emission (p_radio = 0): P(0 detections) = 1",
    "solar_system": "zero Solar System evidence (p_ss = 0): P(null) = 1",
    RADIO_SS: "zero radio emission and zero Solar System evidence: P = 1",
    "infrared": "silent sky (signature fraction q = 0): k flags are all false positives",
}


# ---------------------------------------------------------------------- config
@dataclass(frozen=True)
class RunConfig:
    geometry: str
    radio_mode: str
    seed: int
    channels: tuple[str, ...] = CHANNELS
    condition_on_earth_unreached: bool = True
    n_mc: int = 20000
    range_grid_points: int = 33
    light_travel_delay: bool = True
    headline_pair: tuple[str, str] | None = None
    ir_pair: tuple[str, str] | None = None
    merge_policies: dict[str, str] = field(default_factory=dict)
    # Overrides of registry DEFAULTS only (e.g. lookback sweeps); never of a value
    # a hypothesis constrains itself.
    parameter_overrides: tuple[tuple[str, float], ...] = ()
    # Milestone 2.5 policy ruling: the headline Bayes factor covers verified channels
    # only unless the caller deliberately folds illustrative channels in.
    include_illustrative_in_headline: bool = False

    def __post_init__(self) -> None:
        if self.geometry not in GEOMETRY_MODES:
            raise GeometryError(f"geometry must be explicit, one of {GEOMETRY_MODES}")
        if self.radio_mode not in radio.RADIO_MODES:
            raise ValueError(f"radio_mode must be one of {radio.RADIO_MODES}")
        bad = set(self.channels) - set(CHANNELS)
        if bad:
            raise ValueError(f"Unknown channels {sorted(bad)}")
        if self.n_mc < 2:
            raise ValueError("n_mc must be >= 2")


# --------------------------------------------------------- parameter resolution
@dataclass(frozen=True)
class ResolvedParameters:
    hypothesis_id: str
    fixed: dict[str, float]
    ranges: dict[str, np.ndarray]
    graph: FactorGraph

    def value(self, pid: str) -> float:
        if pid in self.fixed:
            return self.fixed[pid]
        if pid in self.ranges:
            raise SchemaValidationError(
                f"{self.hypothesis_id}: {pid} is a range; this channel needs a fixed value"
            )
        raise SchemaValidationError(f"{self.hypothesis_id}: {pid} unspecified and has no default")

    def quantity(self, pid: str) -> u.Quantity:
        return self.graph.registry.get(pid).quantity(self.value(pid))

    def grid(self, pid: str) -> np.ndarray:
        if pid in self.ranges:
            return self.ranges[pid]
        return np.array([self.value(pid)])

    def has(self, pid: str) -> bool:
        return pid in self.fixed or pid in self.ranges


def resolve_parameters(
    graph: FactorGraph,
    hyp: Hypothesis,
    n_grid: int,
    overrides: Sequence[tuple[str, float]] = (),
) -> ResolvedParameters:
    """Fixed constraints, then overrides of registry defaults, then registry defaults."""
    over = dict(overrides)
    for pid, val in over.items():
        if pid in hyp.parameter_constraints:
            raise SchemaValidationError(
                f"{hyp.id}: override of {pid} conflicts with the hypothesis's own constraint"
            )
        lo, hi = graph.registry.get(pid).bounds
        if not lo <= val <= hi:
            raise SchemaValidationError(f"override {pid}={val} outside registry bounds")
    fixed: dict[str, float] = {}
    ranges: dict[str, np.ndarray] = {}
    for p in graph.registry.parameters:
        c = hyp.parameter_constraints.get(p.id)
        if c is None:
            if p.id in over:
                fixed[p.id] = float(over[p.id])
            elif p.default is not None:
                fixed[p.id] = float(p.default)
            continue
        if c.mode == "fixed":
            fixed[p.id] = float(c.value)
            continue
        lo, hi = float(c.min), float(c.max)
        mid = (np.arange(n_grid) + 0.5) / n_grid
        if p.prior_type == "log_uniform":
            lo = max(lo, p.log_floor or lo)
            ranges[p.id] = np.exp(np.log(lo) + mid * (np.log(hi) - np.log(lo)))
        else:
            ranges[p.id] = lo + mid * (hi - lo)
    return ResolvedParameters(hyp.id, fixed, ranges, graph)


def expansion_model(params: ResolvedParameters, cfg: RunConfig) -> ExpansionModel:
    return ExpansionModel(
        geometry=cfg.geometry,
        origin_rate=params.quantity("origin_rate_per_kpc2_gyr"),
        expansion_speed=params.quantity("expansion_speed_c"),
        lookback=params.quantity("lookback_window_gyr"),
        scale_height=params.quantity("disk_scale_height_pc"),
        solar_height=params.quantity("solar_height_pc"),
        light_travel_delay=cfg.light_travel_delay,
    )


def radiometer(sv: Survey) -> radio.Radiometer:
    rs = sv.radio_sensitivity
    if rs is None:
        raise SchemaValidationError(f"{sv.id} has no radio_sensitivity")
    return radio.Radiometer(
        rs.sefd, rs.snr_threshold, rs.channel_bandwidth, rs.integration_time, rs.n_pol
    )


def ir_calibrations(sv: Survey) -> list[infrared.IRCalibration]:
    if sv.infrared is None:
        raise SchemaValidationError(f"{sv.id} has no infrared block")
    return [
        infrared.IRCalibration(c.id, c.completeness, c.false_positive_rate)
        for c in sv.infrared.calibrations
    ]


def footprint_lattice(radius: u.Quantity, n_radial: int = 8, n_dir: int = 32) -> u.Quantity:
    """Deterministic volume-uniform lattice filling a ball (radial quantiles x Fibonacci sphere)."""
    r = radius.to_value(u.pc) * ((np.arange(n_radial) + 0.5) / n_radial) ** (1.0 / 3.0)
    k = np.arange(n_dir) + 0.5
    cos_t = 1.0 - 2.0 * k / n_dir
    phi = np.pi * (1.0 + 5.0**0.5) * k
    sin_t = np.sqrt(1.0 - cos_t**2)
    dirs = np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
    return (r[:, None, None] * dirs[None, :, :]).reshape(-1, 3) * u.pc


def ir_signature_grid(
    params: ResolvedParameters, model: ExpansionModel | None, ctx: RunContext
) -> tuple[np.ndarray, dict]:
    """Per-star IR signature fraction q, plus provenance of the host fraction.

    Expansion hypotheses: q = P(star reached) * f_ir_loud (planning ruling); the
    reach probability is averaged over the IR footprint lattice. Others: q =
    f_ir_host * f_ir_loud with an explicitly declared host fraction.
    """
    loud = params.grid("ir_conspicuous_fraction")
    hyp = ctx.graph.hypotheses[params.hypothesis_id]
    if model is not None:
        if "ir_host_fraction" in hyp.parameter_constraints:
            raise SchemaValidationError(
                f"{hyp.id}: expansion hypotheses must not declare ir_host_fraction; "
                "it is derived as P(star reached) (SPEC.md Sec. 4.2 coupling ruling)"
            )
        assert ctx.ir_footprint is not None
        p_reach = float(
            np.mean(
                model.reach_probabilities(ctx.ir_footprint, ctx.config.condition_on_earth_unreached)
            )
        )
        return p_reach * loud, {
            "host_fraction_source": "derived: mean P(star reached) over IR footprint lattice",
            "p_star_reached": p_reach,
            "footprint_points": len(ctx.ir_footprint),
        }
    host = params.grid("ir_host_fraction")
    return np.outer(host, loud).ravel(), {
        "host_fraction_source": "declared ir_host_fraction",
        "ir_host_fraction_min": float(host.min()),
        "ir_host_fraction_max": float(host.max()),
    }


# -------------------------------------------------------------- run context
@dataclass(frozen=True)
class RunContext:
    graph: FactorGraph
    config: RunConfig
    surveys: dict[str, Survey]  # channel -> survey
    targets: TargetTable | None
    ir_footprint: u.Quantity | None
    catalog_column_tiers: tuple[str, ...]


def build_context(graph: FactorGraph, cfg: RunConfig) -> RunContext:
    selected = check_pointing_groups(
        graph.surveys_for_channels(cfg.channels),
        cfg.merge_policies,  # type: ignore[arg-type]
    )
    by_channel: dict[str, Survey] = {}
    for sv in selected:
        if sv.channel in by_channel:
            raise SchemaValidationError(
                f"Multiple {sv.channel} surveys selected ({by_channel[sv.channel].id}, {sv.id}); "
                "joint runs take one survey per channel"
            )
        by_channel[sv.channel] = sv
    targets = load_targets(by_channel["radio"], graph.root) if "radio" in by_channel else None
    footprint = None
    if "infrared" in by_channel and by_channel["infrared"].infrared is not None:
        footprint = footprint_lattice(
            by_channel["infrared"].infrared.footprint_max_distance_pc * u.pc
        )
    col_tiers: set[str] = set()
    for sv in by_channel.values():
        if sv.catalog_provenance_path:
            prov = json.loads((graph.root / sv.catalog_provenance_path).read_text("utf-8"))
            col_tiers |= {t for t in prov.get("column_data_tiers", {}) if t != "derived"}
    return RunContext(graph, cfg, by_channel, targets, footprint, tuple(sorted(col_tiers)))


def _component(
    raw_ln: float,
    null_ln: float,
    name: str,
    channels: list[str],
    tiers,
    illustrative_inputs: Sequence[str] = (),
) -> dict:
    return {
        "channels": channels,
        "log10_p_data_given_h": raw_ln / LN10,
        "log10_null": null_ln / LN10,
        "log10_lr_vs_null": (raw_ln - null_ln) / LN10,
        "null_model": NULL_MODELS[name],
        "data_tiers": sorted(set(tiers)),
        "illustrative_inputs": sorted(set(illustrative_inputs)),
    }


# ------------------------------------------------------------- per hypothesis
def evaluate_hypothesis(hyp: Hypothesis, ctx: RunContext) -> dict:
    """Score one Tier-1/2 hypothesis. Tier-3 hypotheses raise ``Tier3ScoringError``."""
    if hyp.is_quarantined:
        raise Tier3ScoringError(
            f"{hyp.id!r} is Tier-3 (quarantined): it cannot receive empirical exclusion scores"
        )
    cfg = ctx.config
    params = resolve_parameters(ctx.graph, hyp, cfg.range_grid_points, cfg.parameter_overrides)
    active = [c for c in cfg.channels if c in hyp.channels_activated and c in ctx.surveys]
    has_expansion = all(params.has(p) for p in EXPANSION_PARAMS)
    if ("radio" in active or "solar_system" in active) and not has_expansion:
        raise SchemaValidationError(f"{hyp.id}: radio/solar_system channels need expansion params")
    cond = cfg.condition_on_earth_unreached

    out: dict = {
        "hypothesis_id": hyp.id,
        "tier": hyp.tier,
        "channels_evaluated": active,
        "parameters": {
            k: {"value": v, "unit": ctx.graph.registry.get(k).unit}
            for k, v in sorted(params.fixed.items())
        },
        "parameter_ranges": {
            k: {"min": float(g.min()), "max": float(g.max()), "grid_points": int(g.size)}
            for k, g in sorted(params.ranges.items())
        },
    }
    comps: dict[str, dict] = {}

    model = expansion_model(params, cfg) if has_expansion else None
    if model is not None:
        out["expansion"] = {
            "geometry_mode": cfg.geometry,
            "earth_lambda": model.earth_lambda(),
            "p_earth_reached": model.earth_reach_probability(),
            "log10_p_earth_unreached": model.log_p_earth_unreached() / LN10,
            "max_bubble_radius_kpc": float(model.max_bubble_radius.to_value(u.kpc)),
            "local_approximation_valid": model.local_approximation_valid,
        }
        if cond:
            comps[EARTH] = _component(model.log_p_earth_unreached(), 0.0, EARTH, [], [])
    else:
        out["expansion"] = None

    ss_active = "solar_system" in active
    p_ss = params.value("solar_system_artifact_detect_prob") if ss_active else 0.0
    ss_tiers = [ctx.surveys["solar_system"].data_tier] if ss_active else []

    if "radio" in active:
        assert model is not None and ctx.targets is not None
        sv = ctx.surveys["radio"]
        if sv.observed_detections != 0:
            raise NotImplementedError("radio likelihood implements the zero-detection case only")
        lf = None
        if cfg.radio_mode == "distance_scaled_eirp":
            lf = radio.LogNormalEIRP(
                params.quantity("transmitter_eirp_median_w"),
                params.value("transmitter_eirp_sigma_dex"),
            )
        rad = radiometer(sv)
        p_det = radio.detection_probabilities(
            cfg.radio_mode, params.value("radio_detect_prob"), ctx.targets.distance, rad, lf
        )
        q = model.reach_probabilities(ctx.targets.xyz, cond)
        q_unc = q if not cond else model.reach_probabilities(ctx.targets.xyz, False)
        joint_ss = ss_active and not cond
        mc = model.zero_detection_probability(
            ctx.targets.xyz,
            p_det,
            rng=np.random.default_rng(np.random.SeedSequence(cfg.seed)),
            condition_on_earth_unreached=cond,
            n_mc=cfg.n_mc,
            p_earth=p_ss if joint_ss else 0.0,
        )
        bounds = radio.zero_detection_bounds(p_det, q)
        eirp_min = radio.min_detectable_eirp(
            ctx.targets.distance, radio.min_detectable_flux(rad)
        ).to_value(u.W)
        i_max = int(np.argmax(eirp_min))
        radio_tiers = [sv.data_tier, *ctx.catalog_column_tiers]
        if sv.radio_sensitivity:
            radio_tiers.append(sv.radio_sensitivity.data_tier)
        radio_illustrative = []
        if lf is not None:
            radio_tiers.append("illustrative_calibration")  # transmitter luminosity function
            radio_illustrative.append("transmitter_eirp_luminosity_function")
        out["radio"] = {
            "survey_id": sv.id,
            "radio_mode": cfg.radio_mode,
            "n_targets": len(ctx.targets),
            "observed_detections": sv.observed_detections,
            "f_min_w_m2": float(radio.min_detectable_flux(rad).to_value(u.W / u.m**2)),
            "mean_p_detect_given_reached": float(p_det.mean()) if p_det.size else 0.0,
            "expected_reached_unconditional": float(q_unc.sum()),
            "expected_reached": float(q.sum()),
            "expected_reached_conditioned_on_earth_unreached": cond,
            "mc_expected_reached": mc.mc_expected_reached,
            "mc_expected_reached_se": mc.mc_expected_reached_se,
            "expected_detections": bounds.expected_detections,
            "p_zero_exact_mc": mc.p_zero_targets_only,
            "p_zero_exact_mc_se": mc.p_zero_se,
            "p_zero_independent_fkg_lower": bounds.independent,
            "p_zero_markov_lower": bounds.markov,
            "p_zero_frechet_upper": bounds.frechet_upper,
            "mc_mu_rim": mc.mu_rim,
            "mc_mu_core": mc.mu_core,
            "mc_points_sampled": mc.n_points,
            "max_eirp_min_target": {
                "target_name": str(ctx.targets.names[i_max]),
                "dist_pc": float(ctx.targets.distance[i_max].to_value(u.pc)),
                "dist_source": str(ctx.targets.dist_source[i_max]),
                "eirp_min_w": float(eirp_min[i_max]),
                "p_detect_given_reached": float(p_det[i_max]),
            },
        }
        comps["radio"] = _component(
            math.log(mc.p_zero_targets_only),
            0.0,
            "radio",
            ["radio"],
            radio_tiers,
            radio_illustrative,
        )
        if joint_ss:
            comps[RADIO_SS] = _component(
                math.log(mc.p_zero),
                0.0,
                RADIO_SS,
                ["radio", "solar_system"],
                radio_tiers + ss_tiers,
                [*radio_illustrative, "solar_system_null"],
            )

    if ss_active:
        assert model is not None
        factor = solar_system.null_likelihood(
            model.earth_reach_probability(), p_ss, conditioned_on_earth_unreached=cond
        )
        out["solar_system"] = {
            "survey_id": ctx.surveys["solar_system"].id,
            "p_artifact_detect": p_ss,
            "marginal_null_likelihood": factor,
            "subsumed_by_earth_unreached_conditioning": cond,
            "evaluated_jointly_with_radio": "radio" in active and not cond,
        }
        comps["solar_system"] = _component(
            math.log(factor) if factor > 0 else -math.inf,
            0.0,
            "solar_system",
            ["solar_system"],
            ss_tiers,
            ["solar_system_null"] if any(t in ILLUSTRATIVE_TIERS for t in ss_tiers) else [],
        )

    if "infrared" in active:
        sv = ctx.surveys["infrared"]
        grid, host_info = ir_signature_grid(params, model, ctx)
        cals = ir_calibrations(sv)
        k, n = sv.observed_detections, sv.n_targets
        raw = infrared.log_marginal_likelihood_calibration_mixture(k, n, grid, cals)
        null = infrared.log_marginal_likelihood_calibration_mixture(k, n, np.array([0.0]), cals)
        assert sv.infrared is not None
        out["infrared"] = {
            "survey_id": sv.id,
            "n_targets": n,
            "observed_candidates": k,
            "signature_fraction_min": float(grid.min()),
            "signature_fraction_max": float(grid.max()),
            **host_info,
            "calibration_marginalisation": "equal-weight mixture over all declared calibrations",
            "calibration_data_tier": sv.infrared.calibration_data_tier,
        }
        comps["infrared"] = _component(
            raw,
            null,
            "infrared",
            ["infrared"],
            [sv.data_tier, sv.infrared.counts_data_tier, sv.infrared.calibration_data_tier],
            ["ir_calibration_bracket"]
            if sv.infrared.calibration_data_tier in ILLUSTRATIVE_TIERS
            else [],
        )

    out["components"] = comps
    terms = select_terms(comps, set(active), include_earth=EARTH in comps)
    out["own_joint"] = _joint(comps, terms, active)
    return out


def select_terms(comps: dict, channels: set[str], include_earth: bool) -> list[str]:
    """Non-overlapping components covering ``channels`` exactly once each."""
    terms = [EARTH] if include_earth and EARTH in comps else []
    if {"radio", "solar_system"} <= channels and RADIO_SS in comps:
        terms.append(RADIO_SS)
    else:
        terms += [c for c in ("radio", "solar_system") if c in channels and c in comps]
    if "infrared" in channels and "infrared" in comps:
        terms.append("infrared")
    return terms


def _joint(comps: dict, terms: list[str], channels: Sequence[str]) -> dict:
    return {
        "channels": sorted(channels),
        "includes_earth_unreached": EARTH in terms,
        "terms": terms,
        "log10_p_data_given_h": float(sum(comps[t]["log10_p_data_given_h"] for t in terms)),
        "log10_lr_vs_null": float(sum(comps[t]["log10_lr_vs_null"] for t in terms)),
    }


# ----------------------------------------------------------------- comparisons
def _is_illustrative(tiers: Sequence[str]) -> bool:
    return any(t in ILLUSTRATIVE_TIERS for t in tiers)


def _pair_terms(a: dict, b: dict, channels: set[str], earth: bool) -> list[str]:
    ta = select_terms(a["components"], channels, earth)
    tb = select_terms(b["components"], channels, earth)
    if ta != tb:
        raise SchemaValidationError(f"Incompatible component decomposition: {ta} vs {tb}")
    return ta


def compare_pair(a: dict, b: dict) -> dict:
    """log10 Bayes factor of a over b, strictly over their shared active channels.

    Two sums are reported. ``log10_bf`` covers every shared channel. The
    ``verified`` block covers only the shared channels whose component carries no
    illustrative / synthetic data tier in either hypothesis (Milestone 2.5 ruling).
    Channel exclusion happens *before* term selection, so a joint term such as
    ``radio+solar_system`` is never dropped wholesale: the verified radio marginal
    is used in its place.
    """
    shared = sorted(set(a["channels_evaluated"]) & set(b["channels_evaluated"]))
    earth = EARTH in a["components"] and EARTH in b["components"]
    ta = _pair_terms(a, b, set(shared), earth)

    def tiers_of(t: str) -> list[str]:
        return sorted({x for h in (a, b) for x in h["components"][t]["data_tiers"]})

    def bf(t: str) -> float:
        return (
            a["components"][t]["log10_p_data_given_h"] - b["components"][t]["log10_p_data_given_h"]
        )

    per = {t: bf(t) for t in ta}
    tiers = sorted({x for t in ta for x in tiers_of(t)})
    illustrative = [t for t in ta if _is_illustrative(tiers_of(t))]
    illustrative_channels = sorted(
        c
        for c in shared
        if _is_illustrative(
            [x for h in (a, b) for x in h["components"].get(c, {}).get("data_tiers", [])]
        )
    )
    verified_channels = sorted(set(shared) - set(illustrative_channels))
    notes = []
    for c in illustrative_channels:
        comp = [h["components"].get(c, {}) for h in (a, b)]
        inputs = sorted({x for k in comp for x in k.get("illustrative_inputs", [])})
        notes.append(
            {
                "channel": c,
                "active_in_run": True,
                "held_out_of_verified_headline": True,
                "illustrative_inputs": inputs,
                "reasons": [ILLUSTRATIVE_INPUT_REASONS[x] for x in inputs]
                or ["component carries an illustrative / synthetic data tier"],
                "data_tiers": sorted({x for k in comp for x in k.get("data_tiers", [])}),
                "log10_bf_term": bf(c) if c in a["components"] and c in b["components"] else None,
            }
        )
    tv = _pair_terms(a, b, set(verified_channels), earth)
    per_v = {t: bf(t) for t in tv}
    verified = {
        "policy": HEADLINE_VERIFIED,
        "channels": verified_channels,
        "terms": tv,
        "log10_bf_by_term": per_v,
        "log10_bf": float(sum(per_v.values())) if tv else None,
        "excluded_illustrative_channels": illustrative_channels,
        "data_tiers": sorted({x for t in tv for x in tiers_of(t)}),
    }
    return {
        "numerator": a["hypothesis_id"],
        "denominator": b["hypothesis_id"],
        "channels_in_bf": shared,
        "earth_unreached_in_bf": earth,
        "terms": ta,
        "log10_bf_by_term": per,
        "term_data_tiers": {t: tiers_of(t) for t in ta},
        "log10_bf": float(sum(per.values())) if ta else None,
        "illustrative_terms": illustrative,
        "illustrative_channels": illustrative_channels,
        "log10_bf_excluding_illustrative": verified["log10_bf"],
        "verified": verified,
        "headline_exclusion_notes": notes,
        "data_tiers_in_bf": tiers,
        "excluded_channels": {
            "numerator_only": sorted(set(a["channels_evaluated"]) - set(shared)),
            "denominator_only": sorted(set(b["channels_evaluated"]) - set(shared)),
        },
    }


def headline_summary(bf: dict, include_illustrative: bool) -> dict:
    """The single headline number of a run, with its policy stated explicitly."""
    if include_illustrative:
        return {
            "policy": HEADLINE_WITH_ILLUSTRATIVE,
            "log10_bf": bf["log10_bf"],
            "terms": bf["terms"],
            "channels": bf["channels_in_bf"],
            "includes_illustrative_channels": bf["illustrative_channels"],
            "data_tiers": bf["data_tiers_in_bf"],
            "headline_exclusion_notes": [
                {**n, "held_out_of_reported_headline": False}
                for n in bf["headline_exclusion_notes"]
            ],
            "warning": (
                "ILLUSTRATIVE CHANNELS INCLUDED: this headline folds illustrative_calibration "
                "terms into the joint Bayes factor; it is a sensitivity statement, NOT an "
                "observational bound"
            )
            if bf["illustrative_channels"]
            else None,
        }
    v = bf["verified"]
    return {
        "policy": HEADLINE_VERIFIED,
        "log10_bf": v["log10_bf"],
        "terms": v["terms"],
        "channels": v["channels"],
        "includes_illustrative_channels": [],
        "excluded_illustrative_channels": v["excluded_illustrative_channels"],
        "headline_exclusion_notes": [
            {**n, "held_out_of_reported_headline": True} for n in bf["headline_exclusion_notes"]
        ],
        "data_tiers": v["data_tiers"],
        "warning": None,
    }


def ir_reversal_sweep(ctx: RunContext, quiet_id: str, loud_id: str) -> dict:
    sv = ctx.surveys["infrared"]
    graph, cfg = ctx.graph, ctx.config
    grids = []
    for hid in (quiet_id, loud_id):
        params = resolve_parameters(
            graph, graph.hypotheses[hid], cfg.range_grid_points, cfg.parameter_overrides
        )
        has_exp = all(params.has(p) for p in EXPANSION_PARAMS)
        model = expansion_model(params, cfg) if has_exp else None
        grids.append(ir_signature_grid(params, model, ctx)[0])
    sw = infrared.sweep_calibrations(
        sv.observed_detections, sv.n_targets, grids[0], grids[1], ir_calibrations(sv)
    )
    return {
        "numerator": quiet_id,
        "denominator": loud_id,
        "survey_id": sv.id,
        "calibration_data_tier": sv.infrared.calibration_data_tier if sv.infrared else None,
        "n_calibrations": len(sw.per_calibration),
        "min_lr": sw.min_lr,
        "min_lr_calibration": sw.min_calibration,
        "max_lr": sw.max_lr,
        "max_lr_calibration": sw.max_calibration,
        "preference_reverses": sw.reverses,
        "per_calibration": sw.per_calibration,
    }


# --------------------------------------------------------------------- receipts
def git_state(root: Path) -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return ("dirty" if dirty else head), head


def _data_tiers(ctx: RunContext) -> list[str]:
    tiers = set(ctx.catalog_column_tiers)
    for sv in ctx.surveys.values():
        tiers.add(sv.data_tier)
        if sv.radio_sensitivity:
            tiers.add(sv.radio_sensitivity.data_tier)
        if sv.infrared:
            tiers |= {sv.infrared.counts_data_tier, sv.infrared.calibration_data_tier}
    if "radio" in ctx.surveys and ctx.config.radio_mode == "distance_scaled_eirp":
        tiers.add("illustrative_calibration")  # transmitter luminosity function
    return sorted(tiers)


def score_hypotheses(
    graph: FactorGraph, cfg: RunConfig, hypothesis_ids: Sequence[str] | None = None
) -> tuple[RunContext, dict]:
    """Score requested hypotheses; returns the context and the ``results`` block."""
    ctx = build_context(graph, cfg)
    ids = list(hypothesis_ids) if hypothesis_ids is not None else sorted(graph.hypotheses)
    scored: dict[str, dict] = {}
    quarantine: list[dict] = []
    for hid in ids:
        hyp = graph.hypotheses[hid]
        if hyp.is_quarantined:
            assert hyp.monoculture_audit is not None
            quarantine.append(
                {
                    "hypothesis_id": hid,
                    "quarantine_reason": hyp.quarantine_reason,
                    "monoculture_audit": hyp.monoculture_audit.model_dump(),
                    "empirical_score": None,
                }
            )
            continue
        scored[hid] = evaluate_hypothesis(hyp, ctx)
    results: dict = {
        "likelihood_conventions": {
            "log10_p_data_given_h": "raw log10 probability of the observed data under H",
            "log10_lr_vs_null": "log10 [P(data|H) / P(data|component null)]; nulls in NULL_MODELS",
            "null_models": NULL_MODELS,
            "bayes_factors": "over the intersection of shared active channels only",
        },
        "hypotheses": scored,
        "tier3_quarantine": quarantine,
        "pairwise_bayes_factors": [
            compare_pair(scored[a], scored[b]) for a, b in itertools.combinations(sorted(scored), 2)
        ],
    }
    if cfg.headline_pair:
        a, b = cfg.headline_pair
        hl = compare_pair(scored[a], scored[b])
        hl["headline"] = headline_summary(hl, cfg.include_illustrative_in_headline)
        results["headline"] = hl
    if cfg.ir_pair and "infrared" in ctx.surveys:
        results["ir_calibration_sweep"] = ir_reversal_sweep(ctx, *cfg.ir_pair)
    return ctx, results


def run_comparison(
    graph: FactorGraph, cfg: RunConfig, hypothesis_ids: Sequence[str] | None = None
) -> dict:
    """Score hypotheses and assemble a RunReceipt (dict, JSON-ready)."""
    ctx, results = score_hypotheses(graph, cfg, hypothesis_ids)
    git_sha, git_head = git_state(graph.root)
    cfg_dict = asdict(cfg)
    created = datetime.now(UTC)
    digest = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:10]
    return {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "run_id": f"{created.strftime('%Y%m%dT%H%M%S%fZ')}-{digest}",
        "created_at_utc": created.isoformat(),
        "git_sha": git_sha,
        "git_head": git_head,
        "rng_seed": cfg.seed,
        "geometry_mode": cfg.geometry,
        "radio_mode": cfg.radio_mode,
        "condition_on_earth_unreached": cfg.condition_on_earth_unreached,
        "config": cfg_dict,
        "active_surveys": [
            {
                "id": s.id,
                "channel": s.channel,
                "pointing_group_id": s.pointing_group_id,
                "data_tier": s.data_tier,
                "catalog_sha256": s.catalog_sha256,
            }
            for s in ctx.surveys.values()
        ],
        "data_tiers_used": _data_tiers(ctx),
        "headline_policy": HEADLINE_WITH_ILLUSTRATIVE
        if cfg.include_illustrative_in_headline
        else HEADLINE_VERIFIED,
        "headline_log10_bf": results["headline"]["headline"]["log10_bf"]
        if "headline" in results
        else None,
        "headline_exclusion_notes": results["headline"]["headline"]["headline_exclusion_notes"]
        if "headline" in results
        else [],
        "results": results,
    }


def canonical_sha256(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=True).encode()).hexdigest()


def results_digest(receipt: dict) -> str:
    """SHA-256 of the canonical numeric results block (deterministic-replay fingerprint)."""
    return canonical_sha256(receipt["results"])


def write_receipt(receipt: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {**receipt, "results_sha256": results_digest(receipt)}
    path = out_dir / f"{receipt['run_id']}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
