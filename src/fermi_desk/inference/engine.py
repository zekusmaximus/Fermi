"""Joint likelihood engine with separate Earth-unreached conditioning.

Every hypothesis likelihood is reported as natural-log components:

* ``earth_unreached``: log P(Earth unreached | H)   (only if conditioning is on)
* ``radio`` (+ ``solar_system`` when evaluated jointly): log P(0 detections | H, ...)
* ``infrared``: log [P(k | H) / P(k | silent sky)]

All non-Earth components are ratios against a hypothesis-independent silent-sky
reference, so differences between hypotheses are log Bayes factors over the
channels they share.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np

from fermi_desk.errors import SchemaValidationError, Tier3ScoringError
from fermi_desk.models import infrared, radio, solar_system
from fermi_desk.models.expansion import GEOMETRY_MODES, ExpansionModel
from fermi_desk.registry import (
    FactorGraph,
    TargetTable,
    check_pointing_groups,
    load_targets,
)
from fermi_desk.schemas import Hypothesis, Survey

LN10 = math.log(10.0)
CHANNELS = ("radio", "infrared", "solar_system")
EXPANSION_PARAMS = ("origin_rate_per_kpc2_gyr", "expansion_speed_c")


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

    def __post_init__(self) -> None:
        from fermi_desk.errors import GeometryError

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


def resolve_parameters(graph: FactorGraph, hyp: Hypothesis, n_grid: int) -> ResolvedParameters:
    """Fixed constraints, then registry defaults; ranges become prior-midpoint grids."""
    fixed: dict[str, float] = {}
    ranges: dict[str, np.ndarray] = {}
    for p in graph.registry.parameters:
        c = hyp.parameter_constraints.get(p.id)
        if c is None:
            if p.default is not None:
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


def _expansion_model(params: ResolvedParameters, cfg: RunConfig) -> ExpansionModel:
    return ExpansionModel(
        geometry=cfg.geometry,
        origin_rate=params.quantity("origin_rate_per_kpc2_gyr"),
        expansion_speed=params.quantity("expansion_speed_c"),
        lookback=params.quantity("lookback_window_gyr"),
        scale_height=params.quantity("disk_scale_height_pc"),
        solar_height=params.quantity("solar_height_pc"),
        light_travel_delay=cfg.light_travel_delay,
    )


def _radiometer(sv: Survey) -> radio.Radiometer:
    rs = sv.radio_sensitivity
    assert rs is not None
    return radio.Radiometer(
        rs.sefd, rs.snr_threshold, rs.channel_bandwidth, rs.integration_time, rs.n_pol
    )


def _ir_calibrations(sv: Survey) -> list[infrared.IRCalibration]:
    assert sv.infrared is not None
    return [
        infrared.IRCalibration(c.id, c.completeness, c.false_positive_rate)
        for c in sv.infrared.calibrations
    ]


def ir_signature_grid(params: ResolvedParameters) -> np.ndarray:
    host = params.grid("ir_host_fraction")
    loud = params.grid("ir_conspicuous_fraction")
    return np.outer(host, loud).ravel()


# -------------------------------------------------------------- run context
@dataclass(frozen=True)
class RunContext:
    graph: FactorGraph
    config: RunConfig
    surveys: dict[str, Survey]  # channel -> survey
    targets: TargetTable | None


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
                "M1 joint runs take one survey per channel"
            )
        by_channel[sv.channel] = sv
    targets = load_targets(by_channel["radio"], graph.root) if "radio" in by_channel else None
    return RunContext(graph, cfg, by_channel, targets)


# ------------------------------------------------------------- per hypothesis
def evaluate_hypothesis(hyp: Hypothesis, ctx: RunContext) -> dict:
    """Score one Tier-1/2 hypothesis. Tier-3 hypotheses raise ``Tier3ScoringError``."""
    if hyp.is_quarantined:
        raise Tier3ScoringError(
            f"{hyp.id!r} is Tier-3 (quarantined): it cannot receive empirical exclusion scores"
        )
    cfg = ctx.config
    params = resolve_parameters(ctx.graph, hyp, cfg.range_grid_points)
    active = [c for c in cfg.channels if c in hyp.channels_activated and c in ctx.surveys]
    has_expansion = all(params.has(p) for p in EXPANSION_PARAMS)
    if ("radio" in active or "solar_system" in active) and not has_expansion:
        raise SchemaValidationError(f"{hyp.id}: radio/solar_system channels need expansion params")

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
        "log_likelihood_components": {},
    }
    comps: dict[str, float] = out["log_likelihood_components"]

    model = _expansion_model(params, cfg) if has_expansion else None
    if model is not None:
        p_e = model.earth_reach_probability()
        out["expansion"] = {
            "geometry_mode": cfg.geometry,
            "earth_lambda": model.earth_lambda(),
            "p_earth_reached": p_e,
            "log10_p_earth_unreached": model.log_p_earth_unreached() / LN10,
            "max_bubble_radius_kpc": float(model.max_bubble_radius.to_value(u.kpc)),
            "local_approximation_valid": model.local_approximation_valid,
        }
        if cfg.condition_on_earth_unreached:
            comps["earth_unreached"] = model.log_p_earth_unreached()
    else:
        out["expansion"] = None

    ss_active = "solar_system" in active
    p_ss = params.value("solar_system_artifact_detect_prob") if ss_active else 0.0

    if "radio" in active:
        assert model is not None and ctx.targets is not None
        sv = ctx.surveys["radio"]
        lf = None
        if cfg.radio_mode == "distance_scaled_eirp":
            lf = radio.LogNormalEIRP(
                params.quantity("transmitter_eirp_median_w"),
                params.value("transmitter_eirp_sigma_dex"),
            )
        p_det = radio.detection_probabilities(
            cfg.radio_mode,
            params.value("radio_detect_prob"),
            ctx.targets.distance,
            _radiometer(sv),
            lf,
        )
        cond = cfg.condition_on_earth_unreached
        q = model.reach_probabilities(ctx.targets.xyz, cond)
        q_unc = q if not cond else model.reach_probabilities(ctx.targets.xyz, False)
        joint_ss = ss_active and not cond
        rng = np.random.default_rng(np.random.SeedSequence(cfg.seed))
        mc = model.zero_detection_probability(
            ctx.targets.xyz,
            p_det,
            rng=rng,
            condition_on_earth_unreached=cond,
            n_mc=cfg.n_mc,
            p_earth=p_ss if joint_ss else 0.0,
        )
        bounds = radio.zero_detection_bounds(p_det, q)
        eirp_min = radio.min_detectable_eirp(
            ctx.targets.distance, radio.min_detectable_flux(_radiometer(sv))
        ).to_value(u.W)
        i_max = int(np.argmax(eirp_min))
        out["radio"] = {
            "survey_id": sv.id,
            "radio_mode": cfg.radio_mode,
            "n_targets": len(ctx.targets),
            "observed_detections": sv.observed_detections,
            "f_min_w_m2": float(radio.min_detectable_flux(_radiometer(sv)).to_value(u.W / u.m**2)),
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
        if sv.observed_detections != 0:
            raise NotImplementedError("M1 radio likelihood implements the zero-detection case only")
        if joint_ss:
            comps["radio+solar_system"] = math.log(mc.p_zero)
        else:
            comps["radio"] = math.log(mc.p_zero_targets_only)

    if ss_active:
        assert model is not None
        factor = solar_system.null_likelihood(
            model.earth_reach_probability(),
            p_ss,
            conditioned_on_earth_unreached=cfg.condition_on_earth_unreached,
        )
        out["solar_system"] = {
            "survey_id": ctx.surveys["solar_system"].id,
            "p_artifact_detect": p_ss,
            "marginal_null_likelihood": factor,
            "subsumed_by_earth_unreached_conditioning": cfg.condition_on_earth_unreached,
            "evaluated_jointly_with_radio": "radio" in active
            and not cfg.condition_on_earth_unreached,
        }
        if "radio" not in active or cfg.condition_on_earth_unreached:
            comps["solar_system"] = math.log(factor) if factor > 0 else -math.inf

    if "infrared" in active:
        sv = ctx.surveys["infrared"]
        grid = ir_signature_grid(params)
        cals = _ir_calibrations(sv)
        log_lr = infrared.log_lr_vs_silent_calibration_mixture(
            sv.observed_detections, sv.n_targets, grid, cals
        )
        out["infrared"] = {
            "survey_id": sv.id,
            "n_targets": sv.n_targets,
            "observed_candidates": sv.observed_detections,
            "signature_fraction_min": float(grid.min()),
            "signature_fraction_max": float(grid.max()),
            "calibration_marginalisation": "equal-weight mixture over all declared calibrations",
            "calibration_data_tier": sv.infrared.calibration_data_tier if sv.infrared else None,
            "log10_lr_vs_silent": log_lr / LN10,
        }
        comps["infrared"] = log_lr

    total = sum(comps.values())
    out["joint_log10_likelihood"] = total / LN10
    out["log10_likelihood_components"] = {k: v / LN10 for k, v in comps.items()}
    return out


# ----------------------------------------------------------------- comparisons
def compare_pair(a: dict, b: dict) -> dict:
    ca, cb = a["log10_likelihood_components"], b["log10_likelihood_components"]
    shared = [k for k in ca if k in cb]
    per = {k: ca[k] - cb[k] for k in shared}
    return {
        "numerator": a["hypothesis_id"],
        "denominator": b["hypothesis_id"],
        "shared_components": shared,
        "log10_bayes_factor_by_component": per,
        "log10_bayes_factor_joint": float(sum(per.values())),
        "unshared_components": sorted(set(ca) ^ set(cb)),
    }


def ir_reversal_sweep(ctx: RunContext, quiet_id: str, loud_id: str) -> dict:
    sv = ctx.surveys["infrared"]
    graph, n = ctx.graph, ctx.config.range_grid_points
    ga = ir_signature_grid(resolve_parameters(graph, graph.hypotheses[quiet_id], n))
    gb = ir_signature_grid(resolve_parameters(graph, graph.hypotheses[loud_id], n))
    sw = infrared.sweep_calibrations(
        sv.observed_detections, sv.n_targets, ga, gb, _ir_calibrations(sv)
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
def _git_state(root: Path) -> tuple[str, str]:
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
    tiers = set()
    for sv in ctx.surveys.values():
        tiers.add(sv.data_tier)
        if sv.radio_sensitivity:
            tiers.add(sv.radio_sensitivity.data_tier)
        if sv.infrared:
            tiers |= {sv.infrared.counts_data_tier, sv.infrared.calibration_data_tier}
        if sv.catalog_provenance_path:
            prov = json.loads((ctx.graph.root / sv.catalog_provenance_path).read_text("utf-8"))
            tiers |= {t for t in prov.get("column_data_tiers", {}) if t != "derived"}
    if "radio" in ctx.surveys and ctx.config.radio_mode == "distance_scaled_eirp":
        tiers.add("illustrative_calibration")  # transmitter luminosity function
    return sorted(tiers)


def run_comparison(
    graph: FactorGraph, cfg: RunConfig, hypothesis_ids: Sequence[str] | None = None
) -> dict:
    """Score all requested hypotheses and assemble a RunReceipt (dict, JSON-ready)."""
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

    results: dict = {"hypotheses": scored, "tier3_quarantine": quarantine}
    if cfg.headline_pair:
        a, b = cfg.headline_pair
        results["headline"] = compare_pair(scored[a], scored[b])
    if cfg.ir_pair and "infrared" in ctx.surveys:
        results["ir_calibration_sweep"] = ir_reversal_sweep(ctx, *cfg.ir_pair)

    git_sha, git_head = _git_state(graph.root)
    cfg_dict = asdict(cfg)
    created = datetime.now(UTC)
    digest = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:10]
    return {
        "receipt_schema_version": 1,
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
        "results": results,
    }


def results_digest(receipt: dict) -> str:
    """SHA-256 of the canonical numeric results block (deterministic-replay fingerprint)."""
    return hashlib.sha256(
        json.dumps(receipt["results"], sort_keys=True, allow_nan=True).encode()
    ).hexdigest()


def write_receipt(receipt: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {**receipt, "results_sha256": results_digest(receipt)}
    path = out_dir / f"{receipt['run_id']}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
