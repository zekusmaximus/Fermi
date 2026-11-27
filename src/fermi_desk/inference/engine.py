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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np

from fermi_desk.errors import GeometryError, SchemaValidationError, Tier3ScoringError
from fermi_desk.models import infrared, radio, solar_system
from fermi_desk.models.expansion import (
    DISK_PROFILES,
    FINITE_DISK,
    GEOMETRY_MODES,
    LOCAL_APPROX_FINITE,
    LOCAL_APPROX_OK,
    LOCAL_APPROX_VIOLATED,
    ExpansionModel,
)
from fermi_desk.registry import (
    BackgroundField,
    FactorGraph,
    TargetTable,
    load_background_summary,
    load_targets,
    resolve_pointing_groups,
)
from fermi_desk.schemas import (
    REJECT_DUPLICATE_POINTING_GROUP,
    UNION_TARGETS_BY_POINTING,
    Hypothesis,
    MergeProvenance,
    Survey,
    validate_pointing_merge_policy,
)

LN10 = math.log(10.0)
LOOKBACK = "lookback_window_gyr"
LOOKBACK_MARGINALIZATION = "lookback_marginalization"
MARGINAL_SPACE = (
    "likelihood space: P(data | H) = sum_k w_k * prod_c P_c(data_c | H, T_k), evaluated as "
    "logsumexp_k(ln w_k + sum_c ln P_c); log-probabilities are never averaged"
)
MARGINAL_TERM_SEMANTICS = (
    "sequential conditional marginals (chain rule) in the listed term order: each term is "
    "log10 P(term data | H, data of the terms before it), marginalised over the lookback "
    "prior; the terms add up exactly to the joint marginal over the listed set. Verified "
    "terms always precede illustrative ones, so no illustrative likelihood re-weights the "
    "lookback posterior behind a verified term"
)
CHANNELS = ("radio", "infrared", "solar_system")
# Data tiers that are NOT verified observational inputs (SPEC.md Sec. 2 invariant 6).
ILLUSTRATIVE_TIERS = frozenset(
    {"illustrative_calibration", "synthetic_proxy", "synthetic_proxy_692"}
)
# v5 (v0.2.1): receipts carry `pointing_merge_policy` and `merge_provenance`, and
# `config.merge_policies` (per-group dict that admitted keep_first) is replaced by
# `config.pointing_merge_policy`. `results` - the payload behind `results_sha256` - is
# unchanged for single-survey runs.
RECEIPT_SCHEMA_VERSION = 5
# Reach nodes of the background shells are evaluated in blocks: the finite-disk importance
# sampler shares its sample budget across one target set, so each block keeps its own.
BACKGROUND_REACH_CHUNK = 512
# Ruling v0.2.1-R2: a flat detection probability on merged background stars is illustrative
# whatever the tier of the star catalogue behind it.
FLAT_BACKGROUND_INPUT = "flat_probability_on_merged_background_field"
# Planning rulings behind an illustrative input. A note gains a `rulings` key only when one
# of its inputs is listed here, so notes of earlier releases keep their exact shape (and
# every single-survey results_sha256 is unchanged).
ILLUSTRATIVE_INPUT_RULINGS = {FLAT_BACKGROUND_INPUT: "v0.2.1-R2"}
# Why each illustrative input is illustrative (surfaced as headline_exclusion_notes).
ILLUSTRATIVE_INPUT_REASONS = {
    "synthetic_background_field": (
        "the merged background-field shells are a synthetic_proxy population (offline test "
        "or dry-run fixture), not archive stars, so the radio term is held out of the "
        "verified headline"
    ),
    FLAT_BACKGROUND_INPUT: (
        "flat_probability mode with union_targets_by_pointing: one distance- and "
        "beam-invariant detection probability p_radio is applied to every merged "
        "background-field star, off-axis and out to 10 kpc, which ignores the d^2 / B(theta) "
        "attenuation of EIRP_min. The merged radio term is therefore an illustrative "
        "sensitivity check, not an observational bound, and is held out of the verified "
        "headline (ruling v0.2.1-R2). distance_scaled_eirp carries the physical off-axis "
        "sensitivity; the default reject_duplicate_pointing_group policy keeps the verified "
        "692-primary radio term"
    ),
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
    # v0.2.1: how surveys sharing a pointing_group_id are combined. The default refuses
    # every undeclared collision and leaves declared extensions out of the run;
    # "union_targets_by_pointing" merges a declared extension into its parent star by star.
    # "keep_first" raises MergePolicyError.
    pointing_merge_policy: str = REJECT_DUPLICATE_POINTING_GROUP
    # Overrides of registry DEFAULTS only (e.g. lookback sweeps); never of a value
    # a hypothesis constrains itself.
    parameter_overrides: tuple[tuple[str, float], ...] = ()
    # Milestone 2.5 policy ruling: the headline Bayes factor covers verified channels
    # only unless the caller deliberately folds illustrative channels in.
    include_illustrative_in_headline: bool = False
    # v0.2.0 -------------------------------------------------------------------
    # Density profile of finite_disk_3d ("uniform" | "exponential"); ignored otherwise.
    disk_profile: str = "uniform"
    # Integrate the lookback window T over its registry log-uniform prior (likelihood
    # space, Gauss-Legendre in ln T) instead of fixing it. Default off: single-T runs
    # and their results_sha256 are unchanged.
    marginalize_lookback: bool = False
    lookback_grid_points: int = 16
    # Samples of the seeded per-target importance sampler (finite_disk_3d only).
    n_importance: int = 16384

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
        if self.disk_profile not in DISK_PROFILES:
            raise GeometryError(f"disk_profile must be one of {DISK_PROFILES}")
        if self.lookback_grid_points < 2:
            raise ValueError("lookback_grid_points must be >= 2")
        if self.n_importance < 16:
            raise ValueError("n_importance must be >= 16")
        validate_pointing_merge_policy(self.pointing_merge_policy)  # MergePolicyError


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
    finite: dict = {}
    if cfg.geometry == FINITE_DISK:
        finite = {
            "disk_profile": cfg.disk_profile,
            "disk_radius": params.quantity("galactic_disk_radius_kpc"),
            "solar_galactocentric_radius": params.quantity("solar_galactocentric_radius_kpc"),
            "radial_scale_length": params.quantity("galactic_radial_scale_length_kpc"),
            "importance_seed": cfg.seed,
            "n_importance": cfg.n_importance,
        }
    return ExpansionModel(
        geometry=cfg.geometry,
        origin_rate=params.quantity("origin_rate_per_kpc2_gyr"),
        expansion_speed=params.quantity("expansion_speed_c"),
        lookback=params.quantity("lookback_window_gyr"),
        scale_height=params.quantity("disk_scale_height_pc"),
        solar_height=params.quantity("solar_height_pc"),
        light_travel_delay=cfg.light_travel_delay,
        **finite,
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
        info = {
            "host_fraction_source": "derived: mean P(star reached) over IR footprint lattice",
            "p_star_reached": p_reach,
            "footprint_points": len(ctx.ir_footprint),
        }
        if model.is_finite:  # importance-sampled reach: report its Monte Carlo error
            info["p_star_reached_se"] = model.expected_reached_se(
                ctx.ir_footprint, ctx.config.condition_on_earth_unreached
            ) / len(ctx.ir_footprint)
        return p_reach * loud, info
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
    # v0.2.1 union_targets_by_pointing: background-field shells merged with the radio
    # primaries (None for single-survey runs), their verified source and merge audit.
    background: radio.BackgroundShells | None = None
    background_field: BackgroundField | None = None
    merge_provenance: MergeProvenance | None = None
    held_out_surveys: tuple[dict, ...] = ()


def build_context(graph: FactorGraph, cfg: RunConfig) -> RunContext:
    resolution = resolve_pointing_groups(
        graph.surveys_for_channels(cfg.channels), cfg.pointing_merge_policy
    )
    background_field = None
    for merge in resolution.merges:
        ext = merge.extensions
        if (
            merge.parent.channel != "radio"
            or len(ext) != 1
            or ext[0].background_summary_path is None
        ):
            raise SchemaValidationError(
                f"{UNION_TARGETS_BY_POINTING} is implemented for exactly one radio "
                f"background-summary extension per pointing group (got {[e.id for e in ext]} "
                f"on {merge.parent.id})"
            )
        background_field = load_background_summary(ext[0], merge.parent, graph.root)
    by_channel: dict[str, Survey] = {}
    for sv in resolution.active:
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
    return RunContext(
        graph,
        cfg,
        by_channel,
        targets,
        footprint,
        tuple(sorted(col_tiers)),
        background=background_field.shells if background_field else None,
        background_field=background_field,
        merge_provenance=background_field.merge_provenance if background_field else None,
        held_out_surveys=resolution.held_out,
    )


def background_radio_term(
    model: ExpansionModel,
    shells: radio.BackgroundShells,
    radio_mode: str,
    p_radio: float,
    luminosity_function: radio.LogNormalEIRP | None,
    condition_on_earth_unreached: bool,
) -> dict:
    """Zero-detection term of the background-field shells (merged primary + background runs).

    Spatial-correlation assumption (``radio.SHELL_CORRELATION_ASSUMPTION``): the primaries
    keep the exact correlated Monte Carlo; the distant background shells are evaluated as
    the independent product of marginal conditional-reach probabilities,

        ln P0_bg = sum_cells N_c * log(1 - p_c * q_c),

    with ``q_c`` the reach probability at the cell's beam-centre direction and distance-bin
    centre (given Earth unreached when the run conditions on it) and ``p_c`` the detection
    probability at the cell's off-axis EIRP_min. By Harris-FKG this is a LOWER bound on the
    exact P0; the Frechet upper bound is returned beside it. ``p_radio = 0`` gives
    ``ln P0_bg = 0`` exactly, so zero-emission invariance holds for merged runs in every
    geometry and radio mode.
    """
    xyz, node_of_cell = shells.reach_nodes()
    q_nodes = (
        np.concatenate(
            [
                model.reach_probabilities(
                    xyz[i : i + BACKGROUND_REACH_CHUNK], condition_on_earth_unreached
                )
                for i in range(0, len(xyz), BACKGROUND_REACH_CHUNK)
            ]
        )
        if len(xyz)
        else np.zeros(0)
    )
    q = q_nodes[node_of_cell]
    p_det = radio.shell_detection_probabilities(radio_mode, p_radio, shells, luminosity_function)
    z = radio.shell_zero_detection(p_det, q, shells.count)
    n_stars = shells.n_stars
    return {
        "survey_id": shells.survey_id,
        "data_tier": shells.data_tier,
        "summary_sha256": shells.source_sha256,
        "n_stars": n_stars,
        "n_cells": shells.n_cells,
        "n_reach_nodes": len(xyz),
        "evaluation": "independent conditional-reach shell product (Harris-FKG lower bound)",
        "spatial_correlation_assumption": radio.SHELL_CORRELATION_ASSUMPTION,
        "mean_p_detect_given_reached": float(np.sum(shells.count * p_det) / n_stars)
        if n_stars
        else 0.0,
        "expected_reached": z.expected_reached,
        "expected_reached_conditioned_on_earth_unreached": condition_on_earth_unreached,
        "expected_detections": z.expected_detections,
        "log10_p_zero_independent_fkg_lower": z.log_p_zero_independent / LN10,
        "p_zero_independent_fkg_lower": math.exp(z.log_p_zero_independent),
        "p_zero_markov_lower": z.markov,
        "p_zero_frechet_upper": z.frechet_upper,
    }


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
        # Parameters scoped to another geometry (Parameter.geometry_modes) are not part
        # of this run's model and are not reported.
        "parameters": {
            k: {"value": v, "unit": ctx.graph.registry.get(k).unit}
            for k, v in sorted(params.fixed.items())
            if ctx.graph.registry.get(k).applies_to(cfg.geometry)
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
        if model.is_finite:
            # Structured validity status + geometry (keys added for finite_disk_3d only,
            # so thin_disk_2d / thick_disk_3d results stay byte-identical).
            out["expansion"]["local_approx"] = model.local_approx_status
            out["expansion"]["finite_disk"] = model.finite_disk_summary()
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
        if model.is_finite:
            fd = model.finite_disk_reach(ctx.targets.xyz)
            out["radio"]["finite_disk_reach"] = {
                "estimator": (
                    "multiple-importance sampling (balance heuristic) of age-integrated "
                    "reach kernels over slab-clipped balls about Earth and each target"
                ),
                "n_importance_samples": fd.n_samples,
                "importance_seed": fd.seed,
                "expected_reached_se": model.expected_reached_se(ctx.targets.xyz, cond),
                "expected_reached_unconditional_se": model.expected_reached_se(
                    ctx.targets.xyz, False
                ),
                "sum_lambda_conditional": fd.sum_conditional,
                "sum_lambda_conditional_se": fd.sum_conditional_se,
                "sum_lambda_unconditional": fd.sum_unconditional,
                "sum_lambda_unconditional_se": fd.sum_unconditional_se,
                "earth_lambda_importance_crosscheck": fd.earth_lambda_mc,
                "earth_lambda_importance_crosscheck_se": fd.earth_lambda_mc_se,
                "mc_proposal_points_accepted": mc.n_points_accepted,
                "mc_proposal_density_bound": mc.proposal_density_bound,
                "bounds_note": (
                    "FKG / Markov / Frechet bounds use the importance-sampled reach "
                    "probabilities, so they carry the Monte Carlo error reported here"
                ),
            }
        ln_radio = math.log(mc.p_zero_targets_only)
        ln_radio_ss = math.log(mc.p_zero) if joint_ss else 0.0
        if ctx.background is not None:
            # union_targets_by_pointing: the shells multiply onto the primaries' exact MC term
            # (keys below exist only in merged runs; single-survey results are unchanged).
            bg = background_radio_term(
                model, ctx.background, cfg.radio_mode, params.value("radio_detect_prob"), lf, cond
            )
            ln_bg = bg["log10_p_zero_independent_fkg_lower"] * LN10
            bg["log10_p_zero_primary_mc"] = ln_radio / LN10
            bg["log10_p_zero_merged"] = (ln_radio + ln_bg) / LN10
            ln_radio += ln_bg
            ln_radio_ss += ln_bg
            out["radio"]["background"] = bg
            out["radio"]["n_targets_merged"] = len(ctx.targets) + bg["n_stars"]
            radio_tiers.append(ctx.background.data_tier)
            if ctx.background.data_tier in ILLUSTRATIVE_TIERS:
                radio_illustrative.append("synthetic_background_field")
            if cfg.radio_mode == "flat_probability":
                # Ruling v0.2.1-R2: a sensitivity-blind p_radio on off-axis stars out to
                # 10 kpc is an illustrative model input, exactly like the transmitter
                # luminosity function above. The term stays in the component tables.
                radio_tiers.append("illustrative_calibration")
                radio_illustrative.append(FLAT_BACKGROUND_INPUT)
        comps["radio"] = _component(
            ln_radio,
            0.0,
            "radio",
            ["radio"],
            radio_tiers,
            radio_illustrative,
        )
        if joint_ss:
            comps[RADIO_SS] = _component(
                ln_radio_ss,
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


# ------------------------------------------------ lookback-window marginalisation
@dataclass(frozen=True)
class LookbackGrid:
    """Deterministic Gauss-Legendre grid in ln T for the registry's log-uniform prior on T.

    Bounds come from ``lookback_window_gyr.marginalization_prior_bounds`` with the SPEC.md
    Sec. 4.1 prior-floor rule applied (effective lower edge = max(lower, log_floor)).
    ``weights`` sum to 1: they are the prior mass assigned to each node.
    """

    declared_bounds_gyr: tuple[float, float]
    effective_bounds_gyr: tuple[float, float]
    log_floor: float | None
    nodes_gyr: np.ndarray
    weights: np.ndarray

    def as_dict(self) -> dict:
        return {
            "parameter": LOOKBACK,
            "prior": {
                "type": "log_uniform",
                "declared_bounds_gyr": list(self.declared_bounds_gyr),
                "effective_bounds_gyr": list(self.effective_bounds_gyr),
                "log_floor": self.log_floor,
                "floor_rule": "effective lower edge = max(lower bound, log_floor)",
            },
            "quadrature": {
                "rule": "Gauss-Legendre in ln T",
                "n_points": int(self.nodes_gyr.size),
                "lookback_gyr": self.nodes_gyr.tolist(),
                "weights": self.weights.tolist(),
            },
        }


def lookback_grid(graph: FactorGraph, n_points: int) -> LookbackGrid:
    p = graph.registry.get(LOOKBACK)
    if p.marginalization_prior_bounds is None:
        raise SchemaValidationError(
            f"{LOOKBACK} declares no marginalization_prior_bounds; cannot marginalise the lookback"
        )
    lo, hi = p.marginalization_log_bounds
    x, w = np.polynomial.legendre.leggauss(int(n_points))
    nodes = np.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * (x + 1.0) / 2.0)
    return LookbackGrid(
        tuple(p.marginalization_prior_bounds), (lo, hi), p.log_floor, nodes, w / w.sum()
    )


def _logsumexp(ln: np.ndarray) -> float:
    m = float(np.max(ln))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(ln - m))))


def chain_marginals(
    node_log10_p: Mapping[str, Sequence[float]], weights: Sequence[float], terms: Sequence[str]
) -> tuple[dict[str, float], float]:
    """Chain-rule decomposition of the joint marginal over ``terms`` (log10 units).

    ``J_m = log sum_k w_k prod_{c in terms[:m]} P_c(T_k)``; term ``m`` is ``J_m - J_{m-1}``,
    i.e. ``log P(term m | H, terms before it)`` with T integrated out. Returns the
    per-term values and the joint ``J_len(terms)``. Marginalisation is in likelihood
    space (logsumexp); a zero-probability prefix yields ``-inf`` then ``nan``.
    """
    acc = np.log(np.asarray(weights, float))
    prev = _logsumexp(acc)  # ln sum_k w_k = 0 up to rounding
    out: dict[str, float] = {}
    for t in terms:
        acc = acc + np.asarray(node_log10_p[t], float) * LN10
        cur = _logsumexp(acc)
        out[t] = (cur - prev) / LN10 if math.isfinite(prev) else math.nan
        prev = cur
    return out, prev / LN10


def _weighted(values: Sequence[float], weights: np.ndarray) -> float:
    return float(np.sum(weights * np.asarray(values, float)))


def evaluate_hypothesis_marginalized(hyp: Hypothesis, ctx: RunContext, grid: LookbackGrid) -> dict:
    """Score one hypothesis with the lookback window integrated over its prior.

    Every grid node is a full ``evaluate_hypothesis`` run at fixed ``T_k`` (same seed:
    common random numbers across nodes). A hypothesis whose likelihood cannot depend on
    T (no expansion parameters), or which constrains T itself, is evaluated once and
    carried as a constant across the grid.
    """
    if hyp.is_quarantined:
        raise Tier3ScoringError(
            f"{hyp.id!r} is Tier-3 (quarantined): it cannot receive empirical exclusion scores"
        )
    cfg = ctx.config
    if any(k == LOOKBACK for k, _ in cfg.parameter_overrides):
        raise SchemaValidationError(
            f"{LOOKBACK} cannot be both overridden and marginalised in one run"
        )
    params = resolve_parameters(ctx.graph, hyp, cfg.range_grid_points, cfg.parameter_overrides)
    reason = None
    if not all(params.has(p) for p in EXPANSION_PARAMS):
        reason = "no expansion parameters: likelihood does not depend on the lookback window"
    elif LOOKBACK in hyp.parameter_constraints:
        reason = "hypothesis constrains the lookback window itself; its own value is used"
    base = replace(cfg, marginalize_lookback=False)

    def at(t: float | None) -> RunContext:
        over = cfg.parameter_overrides if t is None else (*cfg.parameter_overrides, (LOOKBACK, t))
        return replace(ctx, config=replace(base, parameter_overrides=over))

    n = int(grid.nodes_gyr.size)
    if reason is not None:
        nodes = [evaluate_hypothesis(hyp, at(None))] * n
    else:
        nodes = [evaluate_hypothesis(hyp, at(float(t))) for t in grid.nodes_gyr]
    out = _marginalize_nodes(nodes, grid, reason)
    if reason is None:  # cheap deterministic convergence check of the grid itself
        fine = lookback_grid(ctx.graph, 2 * n)
        lam = [
            expansion_model(
                resolve_parameters(
                    ctx.graph, hyp, cfg.range_grid_points, at(float(t)).config.parameter_overrides
                ),
                base,
            ).earth_lambda()
            for t in fine.nodes_gyr
        ]
        coarse = _logsumexp(
            np.log(grid.weights) - np.array([nd["expansion"]["earth_lambda"] for nd in nodes])
        )
        dense = _logsumexp(np.log(fine.weights) - np.asarray(lam))
        out[LOOKBACK_MARGINALIZATION]["earth_term_quadrature_check"] = {
            "quantity": "log10 P(Earth unreached | H), lookback-marginalised (seed-free)",
            "n_points": n,
            "value": coarse / LN10,
            "n_points_fine": 2 * n,
            "value_fine": dense / LN10,
            "abs_difference_dex": abs(coarse - dense) / LN10,
        }
    return out


def _marginalize_nodes(nodes: Sequence[dict], grid: LookbackGrid, reason: str | None) -> dict:
    """Collapse per-node hypothesis results into one lookback-marginalised result."""
    first = nodes[0]
    w = grid.weights
    comps0 = first["components"]
    node_log10 = {c: [nd["components"][c]["log10_p_data_given_h"] for nd in nodes] for c in comps0}
    own_terms = list(first["own_joint"]["terms"])
    chain, joint = chain_marginals(node_log10, w, own_terms)
    comps: dict[str, dict] = {}
    for c, comp in comps0.items():
        if c in chain:
            raw = chain[c]
            before = own_terms[: own_terms.index(c)]
            how = f"conditional on {before}" if before else "first term: prior marginal"
        else:  # alternate decomposition (e.g. radio marginal beside radio+solar_system)
            raw = chain_marginals(node_log10, w, [c])[0][c]
            how = "standalone prior marginal (not part of the own-joint chain)"
        comps[c] = {
            **comp,
            "log10_p_data_given_h": raw,
            "log10_lr_vs_null": raw - comp["log10_null"],
            "lookback_marginal": how,
        }
    # Lookback posterior given Earth unreached (the conditioning the radio / IR terms carry).
    post = w
    if EARTH in comps0:
        ln_post = np.log(w) + np.asarray(node_log10[EARTH], float) * LN10
        post = np.exp(ln_post - _logsumexp(ln_post))

    out: dict = {
        "hypothesis_id": first["hypothesis_id"],
        "tier": first["tier"],
        "channels_evaluated": first["channels_evaluated"],
        "parameters": dict(first["parameters"]),
        "parameter_ranges": first["parameter_ranges"],
    }
    if reason is None:
        out["parameters"][LOOKBACK] = {
            "value": None,
            "unit": first["parameters"][LOOKBACK]["unit"],
            "marginalized_log_uniform_bounds": list(grid.effective_bounds_gyr),
        }

    ex = [nd["expansion"] for nd in nodes]
    if first["expansion"] is None:
        out["expansion"] = None
    else:
        ln_unreached = _logsumexp(np.log(w) - np.array([e["earth_lambda"] for e in ex]))
        violated = sum(not e["local_approximation_valid"] for e in ex)
        finite = "finite_disk" in first["expansion"]
        out["expansion"] = {
            "geometry_mode": first["expansion"]["geometry_mode"],
            "lookback_marginalized": reason is None,
            "earth_lambda": -ln_unreached,
            "p_earth_reached": float(-np.expm1(ln_unreached)),
            "log10_p_earth_unreached": ln_unreached / LN10,
            "max_bubble_radius_kpc": max(e["max_bubble_radius_kpc"] for e in ex),
            "local_approximation_valid": violated == 0,
            "local_approx": LOCAL_APPROX_FINITE
            if finite
            else (LOCAL_APPROX_VIOLATED if violated else LOCAL_APPROX_OK),
            "local_approx_nodes_violated": int(violated) if reason is None else int(violated > 0),
            "summary_semantics": (
                "prior-weighted over the lookback grid: p_earth_reached = 1 - sum_k w_k "
                "exp(-Lambda_E(T_k)); earth_lambda = -ln sum_k w_k exp(-Lambda_E(T_k))"
            ),
        }
        if finite:
            out["expansion"]["finite_disk"] = first["expansion"]["finite_disk"]

    if "radio" in first:
        rd = [nd["radio"] for nd in nodes]
        static = (
            "survey_id",
            "radio_mode",
            "n_targets",
            "observed_detections",
            "f_min_w_m2",
            "mean_p_detect_given_reached",
            "expected_reached_conditioned_on_earth_unreached",
            "max_eirp_min_target",
        )
        averaged = (
            "expected_reached",
            "mc_expected_reached",
            "expected_detections",
            "p_zero_exact_mc",
            "p_zero_independent_fkg_lower",
            "p_zero_markov_lower",
            "p_zero_frechet_upper",
        )
        out["radio"] = {k: rd[0][k] for k in static}
        out["radio"]["expected_reached_unconditional"] = _weighted(
            [r["expected_reached_unconditional"] for r in rd], w
        )
        for k in averaged:
            out["radio"][k] = _weighted([r[k] for r in rd], post)
        for k in ("mc_expected_reached_se", "p_zero_exact_mc_se"):
            out["radio"][k] = _weighted([r[k] for r in rd], post)
        out["radio"]["mc_points_sampled"] = int(sum(r["mc_points_sampled"] for r in rd))
        out["radio"]["lookback_averaging"] = (
            "expected_reached_unconditional is prior-weighted; every other number is "
            "weighted by the lookback posterior given Earth unreached (the prior when "
            "Earth conditioning is off); *_se are weighted linear sums of the per-node "
            "standard errors (an upper bound: nodes share one seed)"
        )
        if "background" in rd[0]:  # merged primary + background runs only
            bgs = [r["background"] for r in rd]
            avg = {
                k: _weighted([b[k] for b in bgs], post)
                for k in (
                    "expected_reached",
                    "expected_detections",
                    "p_zero_independent_fkg_lower",
                    "p_zero_markov_lower",
                    "p_zero_frechet_upper",
                )
            }
            p0 = avg["p_zero_independent_fkg_lower"]
            drop = ("log10_p_zero_primary_mc", "log10_p_zero_merged")
            out["radio"]["background"] = {
                **{k: v for k, v in bgs[0].items() if k not in drop},
                **avg,
                "log10_p_zero_independent_fkg_lower": math.log10(p0) if p0 > 0 else -math.inf,
            }
            out["radio"]["n_targets_merged"] = rd[0]["n_targets_merged"]
    if "solar_system" in first:
        ss = [nd["solar_system"] for nd in nodes]
        out["solar_system"] = {
            **ss[0],
            "marginal_null_likelihood": _weighted(
                [s["marginal_null_likelihood"] for s in ss], post
            ),
        }
    if "infrared" in first:
        ir = [nd["infrared"] for nd in nodes]
        out["infrared"] = {
            **ir[0],
            "signature_fraction_min": min(i["signature_fraction_min"] for i in ir),
            "signature_fraction_max": max(i["signature_fraction_max"] for i in ir),
        }
        for k in ("p_star_reached", "p_star_reached_se"):
            if k in ir[0]:
                out["infrared"][k] = _weighted([i[k] for i in ir], post)

    out["components"] = comps
    out["own_joint"] = {
        "channels": first["own_joint"]["channels"],
        "includes_earth_unreached": first["own_joint"]["includes_earth_unreached"],
        "terms": own_terms,
        "log10_p_data_given_h": joint,
        "log10_lr_vs_null": joint - float(sum(comps0[t]["log10_null"] for t in own_terms)),
    }
    per_node = []
    for k, nd in enumerate(nodes):
        row = {
            "lookback_gyr": float(grid.nodes_gyr[k]),
            "prior_weight": float(w[k]),
            "posterior_weight": float(post[k]),
            "log10_p_by_component": {c: node_log10[c][k] for c in comps0},
        }
        if nd["expansion"] is not None:
            e = nd["expansion"]
            row["p_earth_reached"] = e["p_earth_reached"]
            row["local_approx"] = e.get(
                "local_approx",
                LOCAL_APPROX_OK if e["local_approximation_valid"] else LOCAL_APPROX_VIOLATED,
            )
        if "radio" in nd:
            for key in ("expected_reached", "p_zero_exact_mc", "p_zero_exact_mc_se"):
                row[key] = nd["radio"][key]
        if "p_star_reached" in nd.get("infrared", {}):
            row["ir_p_star_reached"] = nd["infrared"]["p_star_reached"]
        per_node.append(row)
    out[LOOKBACK_MARGINALIZATION] = {
        "marginalized": reason is None,
        "not_marginalized_reason": reason,
        "weights": w.tolist(),
        "posterior_weights": np.asarray(post).tolist(),
        "posterior_weight_semantics": "lookback posterior given Earth unreached"
        if EARTH in comps0
        else "lookback prior (no Earth-unreached conditioning)",
        "node_log10_p": node_log10,
        "per_node": per_node,
    }
    return out


def _is_marginalized(h: Mapping) -> bool:
    return LOOKBACK_MARGINALIZATION in h


# ----------------------------------------------------------------- comparisons
def _is_illustrative(tiers: Sequence[str]) -> bool:
    return any(t in ILLUSTRATIVE_TIERS for t in tiers)


def _merged_background_breakdown(a: dict, b: dict) -> dict | None:
    """Where a merged radio Bayes-factor term comes from: primaries vs background shells.

    Present only when both hypotheses carry a merged background. The split into the
    primaries' exact Monte Carlo term and the shells' independent-product term exists for
    fixed-T runs; a lookback-marginalised run reports the star counts and tier only.
    """
    bgs = [(h.get("radio") or {}).get("background") for h in (a, b)]
    if not all(bgs):
        return None
    out = {
        "survey_id": bgs[0]["survey_id"],
        "data_tier": bgs[0]["data_tier"],
        "summary_sha256": bgs[0]["summary_sha256"],
        "n_background_stars": bgs[0]["n_stars"],
        "n_primary_targets": a["radio"]["n_targets"],
    }
    keys = ("log10_p_zero_primary_mc", "log10_p_zero_independent_fkg_lower")
    if all(k in bg for bg in bgs for k in keys):
        out["log10_bf_primary_targets_exact_mc"] = bgs[0][keys[0]] - bgs[1][keys[0]]
        out["log10_bf_background_shells_fkg_product"] = bgs[0][keys[1]] - bgs[1][keys[1]]
    return out


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

    Lookback-marginalised results (v0.2.0): the likelihood no longer factorises over
    terms, so each sum is the joint marginal over its own term set, decomposed by the
    chain rule with verified terms first (``chain_marginals``). The verified sum is
    therefore the exact marginal Bayes factor of the verified data alone: no
    illustrative likelihood ever re-weights the lookback posterior behind it.
    """
    shared = sorted(set(a["channels_evaluated"]) & set(b["channels_evaluated"]))
    earth = EARTH in a["components"] and EARTH in b["components"]
    ta = _pair_terms(a, b, set(shared), earth)
    marginal = _is_marginalized(a)
    if marginal != _is_marginalized(b):
        raise SchemaValidationError(
            "cannot compare a lookback-marginalised result with a fixed-T one"
        )

    def tiers_of(t: str) -> list[str]:
        return sorted({x for h in (a, b) for x in h["components"][t]["data_tiers"]})

    def chain_bf(terms: Sequence[str]) -> dict[str, float]:
        la, lb = a[LOOKBACK_MARGINALIZATION], b[LOOKBACK_MARGINALIZATION]
        ca = chain_marginals(la["node_log10_p"], la["weights"], terms)[0]
        cb = chain_marginals(lb["node_log10_p"], lb["weights"], terms)[0]
        return {t: ca[t] - cb[t] for t in terms}

    if marginal:
        ta = [t for t in ta if not _is_illustrative(tiers_of(t))] + [
            t for t in ta if _is_illustrative(tiers_of(t))
        ]
        per = chain_bf(ta)

        def bf(t: str) -> float:
            return per[t] if t in per else chain_bf([t])[t]

    else:

        def bf(t: str) -> float:
            return (
                a["components"][t]["log10_p_data_given_h"]
                - b["components"][t]["log10_p_data_given_h"]
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
        note = {
            "channel": c,
            "active_in_run": True,
            "held_out_of_verified_headline": True,
            "illustrative_inputs": inputs,
            "reasons": [ILLUSTRATIVE_INPUT_REASONS[x] for x in inputs]
            or ["component carries an illustrative / synthetic data tier"],
            "data_tiers": sorted({x for k in comp for x in k.get("data_tiers", [])}),
            "log10_bf_term": bf(c) if c in a["components"] and c in b["components"] else None,
        }
        # Keys below exist only in merged (union_targets_by_pointing) runs.
        rulings = sorted(
            {ILLUSTRATIVE_INPUT_RULINGS[x] for x in inputs if x in ILLUSTRATIVE_INPUT_RULINGS}
        )
        if rulings:
            note["rulings"] = rulings
        merged = _merged_background_breakdown(a, b) if c == "radio" else None
        if merged:
            note["merged_background"] = merged
        notes.append(note)
    tv = _pair_terms(a, b, set(verified_channels), earth)
    per_v = chain_bf(tv) if marginal else {t: bf(t) for t in tv}
    verified = {
        "policy": HEADLINE_VERIFIED,
        "channels": verified_channels,
        "terms": tv,
        "log10_bf_by_term": per_v,
        "log10_bf": float(sum(per_v.values())) if tv else None,
        "excluded_illustrative_channels": illustrative_channels,
        "data_tiers": sorted({x for t in tv for x in tiers_of(t)}),
    }
    extra: dict = {}
    if marginal:
        extra[LOOKBACK_MARGINALIZATION] = {
            "space": MARGINAL_SPACE,
            "term_semantics": MARGINAL_TERM_SEMANTICS,
            "chain_order": list(ta),
            "verified_chain_order": list(tv),
        }
    return {
        **extra,
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
    if ctx.background is not None:
        tiers.add(ctx.background.data_tier)
        if "radio" in ctx.surveys and ctx.config.radio_mode == "flat_probability":
            tiers.add("illustrative_calibration")  # v0.2.1-R2: flat p_radio on background stars
    return sorted(tiers)


def _active_surveys(ctx: RunContext) -> list[dict]:
    rows = [
        {
            "id": s.id,
            "channel": s.channel,
            "pointing_group_id": s.pointing_group_id,
            "data_tier": s.data_tier,
            "catalog_sha256": s.catalog_sha256,
        }
        for s in ctx.surveys.values()
    ]
    if ctx.background is not None:  # merged extension: listed after the surveys it joins
        parent = ctx.surveys["radio"]
        rows.append(
            {
                "id": ctx.background.survey_id,
                "channel": "radio",
                "pointing_group_id": parent.pointing_group_id,
                "data_tier": ctx.background.data_tier,
                "catalog_sha256": None,
                "background_summary_sha256": ctx.background.source_sha256,
                "merged_into": parent.id,
                "pointing_merge_policy": UNION_TARGETS_BY_POINTING,
            }
        )
    return rows


def score_hypotheses(
    graph: FactorGraph, cfg: RunConfig, hypothesis_ids: Sequence[str] | None = None
) -> tuple[RunContext, dict]:
    """Score requested hypotheses; returns the context and the ``results`` block."""
    ctx = build_context(graph, cfg)
    ids = list(hypothesis_ids) if hypothesis_ids is not None else sorted(graph.hypotheses)
    grid = lookback_grid(graph, cfg.lookback_grid_points) if cfg.marginalize_lookback else None
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
        scored[hid] = (
            evaluate_hypothesis(hyp, ctx)
            if grid is None
            else evaluate_hypothesis_marginalized(hyp, ctx, grid)
        )
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
    if grid is not None:  # keys added only when marginalising: single-T results unchanged
        results[LOOKBACK_MARGINALIZATION] = {
            "enabled": True,
            **grid.as_dict(),
            "space": MARGINAL_SPACE,
            "term_semantics": MARGINAL_TERM_SEMANTICS,
            "hypotheses": {
                hid: {
                    "marginalized": h[LOOKBACK_MARGINALIZATION]["marginalized"],
                    "not_marginalized_reason": h[LOOKBACK_MARGINALIZATION][
                        "not_marginalized_reason"
                    ],
                }
                for hid, h in scored.items()
            },
        }
        if "ir_calibration_sweep" in results:
            results["ir_calibration_sweep"]["lookback_note"] = (
                "calibration sweep evaluated at the fixed registry lookback default, "
                "not lookback-marginalised"
            )
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
    extra: dict = {}
    if cfg.geometry == FINITE_DISK:
        extra["disk_profile"] = cfg.disk_profile
    if cfg.marginalize_lookback:  # prior bounds, grid and weights (also inside results)
        marg = results[LOOKBACK_MARGINALIZATION]
        extra[LOOKBACK_MARGINALIZATION] = {k: marg[k] for k in ("parameter", "prior", "quadrature")}
    return {
        **extra,
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
        "active_surveys": _active_surveys(ctx),
        # Receipt schema v5: the policy is always stated; merge_provenance is None unless
        # union_targets_by_pointing actually merged an extension into its parent.
        "pointing_merge_policy": cfg.pointing_merge_policy,
        "merge_provenance": ctx.merge_provenance.model_dump() if ctx.merge_provenance else None,
        # v0.2.1b: provenance of the merged background field (None unless merged): pinned
        # SHA-256, tier, clean code_git_sha, exact N_bg, de-duplication counters including
        # suspected_unmatched_primary (ruling v0.2.1-R3) and the retrieved-vs-published scope.
        "background_field": ctx.background_field.summary() if ctx.background_field else None,
        "held_out_extension_surveys": list(ctx.held_out_surveys),
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
