"""Background-field extension report (v0.2.1b): the promoted Gaia DR3 stars inside the BL beams.

Everything here is computed offline from the SHA-256-pinned, validated summary artifact
(``data/catalogs/bl_gbt_lband_background_summary.json``) and the locked primary catalog:

* ``shell_summary``       - where the stars are (distance / EIRP_min distributions);
* ``attenuation_summary`` - how much the d^2 / B(theta) factor raises EIRP_min off-axis;
* ``realized_eig``        - the expected information gain of the background-field null with
  the REAL star population, in both radio modes, with Harris-FKG / Frechet bounds, the
  information actually gained from the observed null, and the same numbers without the
  suspected unmatched primaries (ruling v0.2.1-R3: reported, never removed).

Data tiers are kept apart: star counts and de-duplication are ``reproduced_archive``; every
EIG and P(no detection) additionally depends on an ``illustrative_calibration`` input (the
transmitter luminosity function in ``distance_scaled_eirp``, the sensitivity-blind flat
probability in ``flat_probability``; ruling v0.2.1-R2), and is labelled so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import astropy.units as u
import numpy as np

from fermi_desk.inference import eig
from fermi_desk.inference.engine import (
    FLAT_BACKGROUND_INPUT,
    ILLUSTRATIVE_INPUT_REASONS,
    RunConfig,
    background_radio_term,
    expansion_model,
    radiometer,
    resolve_parameters,
)
from fermi_desk.models import radio
from fermi_desk.registry import (
    MISSING_GAIA_ID,
    BackgroundField,
    FactorGraph,
    load_background_summary,
    load_targets,
    primary_sightings,
)
from fermi_desk.schemas import Survey

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
# Honest labels of the two radio modes when applied to off-axis background stars.
RADIO_MODE_LABELS = {
    "distance_scaled_eirp": {
        "label": "physical",
        "physical": True,
        "description": (
            "p_detect = p_radio * P(EIRP >= EIRP_min(d, theta)): each star is as detectable as "
            "its distance and beam offset allow"
        ),
        "illustrative_input": "transmitter_eirp_luminosity_function",
    },
    "flat_probability": {
        "label": "unphysical",
        "physical": False,
        "description": (
            "p_detect = p_radio for every star, on-axis at 5 pc or off-axis at 10 kpc alike "
            "(ruling v0.2.1-R2: an illustrative sensitivity check, never a verified bound)"
        ),
        "illustrative_input": FLAT_BACKGROUND_INPUT,
    },
}
EIG_OUTCOME = (
    "Y = {0 detections, >= 1 detection} among the background-field stars; P(0 | H) is the "
    "independent conditional-reach shell product (Harris-FKG lower bound), given Earth unreached"
)


def find_background_extension(graph: FactorGraph) -> tuple[Survey, Survey] | None:
    """(extension survey, parent survey) of the pinned radio background field, if any."""
    for sv in graph.surveys.values():
        if sv.channel == "radio" and sv.background_summary_path is not None:
            return sv, graph.surveys[sv.parent_pointing_survey_id or ""]
    return None


def load_background_field(graph: FactorGraph) -> BackgroundField | None:
    """The verified background field (SHA-256 pin, schema, offline recomputation) or None."""
    found = find_background_extension(graph)
    if found is None:
        return None
    return load_background_summary(found[0], found[1], graph.root)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Quantiles of a weighted sample (lower-value convention: smallest v with CDF >= q)."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order].astype(float)
    cdf = np.cumsum(w) / w.sum()
    return {
        f"p{round(q * 100):02d}": float(v[min(np.searchsorted(cdf, q), len(v) - 1)])
        for q in QUANTILES
    }


def _is_half_decade(value: float) -> bool:
    x = 2.0 * math.log10(value)
    return abs(x - round(x)) < 1e-9


def shell_summary(field: BackgroundField) -> dict:
    """Distance and EIRP_min distributions of the background stars (bin-centre values)."""
    sh = field.shells
    b = sh.binning
    count = sh.count
    d_counts = np.bincount(sh.distance_bin, weights=count, minlength=b.n_distance_bins)
    e_counts = np.bincount(sh.eirp_min_bin, weights=count, minlength=b.n_eirp_bins)
    d_edges = 10.0 ** (b.distance_log10_pc_min + b.distance_bin_dex * np.arange(len(d_counts) + 1))
    per_decade = round(1.0 / b.eirp_bin_dex)
    decades = []
    for k in range(per_decade, b.n_eirp_bins + 1, per_decade):
        n = int(e_counts[:k].sum())
        if n == 0 and not decades:
            continue
        decades.append(
            {
                "eirp_min_max_w": float(10.0 ** (b.eirp_log10_w_min + k * b.eirp_bin_dex)),
                "n_stars": n,
            }
        )
        if n == sh.n_stars:
            break
    per_pointing = np.bincount(
        sh.pointing_index, weights=count, minlength=len(sh.pointing_unit_xyz)
    )
    xyz, _ = sh.reach_nodes()
    cum = np.cumsum(d_counts)
    return {
        "binning": b.as_dict(),
        "n_cells": sh.n_cells,
        "n_stars": sh.n_stars,
        "n_reach_nodes": len(xyz),
        "representative": "every star sits at its beam-centre direction and geometric bin centres",
        "stars_per_pointing": {
            "min": int(per_pointing.min()),
            "median": float(np.median(per_pointing)),
            "max": int(per_pointing.max()),
            "n_pointings": len(per_pointing),
            "n_pointings_with_stars": int(np.sum(per_pointing > 0)),
        },
        "distance_histogram": {
            "bin_edges_pc": d_edges.tolist(),
            "counts": [int(x) for x in d_counts],
        },
        "distance_cumulative": [
            {"d_max_pc": float(d_edges[k]), "n_stars": int(cum[k - 1])}
            for k in range(1, len(d_edges))
            if _is_half_decade(d_edges[k]) or k == len(d_edges) - 1
        ],
        "distance_quantiles_pc": _weighted_quantiles(sh.cell_distance().to_value(u.pc), count),
        "eirp_min_cumulative_by_decade": decades,
        "eirp_min_quantiles_w": _weighted_quantiles(sh.cell_eirp_min().to_value(u.W), count),
    }


def attenuation_summary(graph: FactorGraph, field: BackgroundField) -> dict:
    """How far off-axis and distant the background stars are, against the on-axis primaries."""
    rec = field.record
    parent = graph.surveys[field.parent_survey_id]
    primaries = load_targets(parent, graph.root)
    f_min = radio.min_detectable_flux(radiometer(parent))
    fwhm = rec.beam_fwhm_arcmin * u.arcmin
    prim_eirp = radio.min_detectable_eirp(primaries.distance, f_min).to_value(u.W)
    sample = rec.validation_sample.background
    theta = np.array([s.theta_arcmin for s in sample]) * u.arcmin
    beam = radio.beam_response(theta, fwhm)
    sh = field.shells
    bg_eirp = _weighted_quantiles(sh.cell_eirp_min().to_value(u.W), sh.count)
    bg_dist = _weighted_quantiles(sh.cell_distance().to_value(u.pc), sh.count)
    prim_q = {f"p{round(q * 100):02d}": float(np.quantile(prim_eirp, q)) for q in QUANTILES}
    d_prim = float(np.median(primaries.distance.to_value(u.pc)))
    return {
        "formula": "EIRP_min(d, theta) = 4 pi d^2 F_min / B(theta), B(theta) = 2^(-4 (theta / FWHM)^2)",
        "f_min_w_m2": rec.f_min_w_m2,
        "beam_fwhm_arcmin": rec.beam_fwhm_arcmin,
        "cone_radius_arcmin": rec.cone_radius_arcmin,
        "beam_response_at_cone_edge": float(
            radio.beam_response(rec.cone_radius_arcmin * u.arcmin, fwhm)
        ),
        "validation_sample": {
            "n_stars": len(sample),
            "exact": True,
            "theta_arcmin": {
                "median": float(np.median(theta.value)),
                "max": float(theta.value.max()),
            },
            "beam_response": {
                "min": float(beam.min()),
                "median": float(np.median(beam)),
                "mean": float(beam.mean()),
            },
            "beam_attenuation_factor": {
                "median": float(np.median(1.0 / beam)),
                "mean": float(np.mean(1.0 / beam)),
                "max": float(np.max(1.0 / beam)),
            },
        },
        "background_eirp_min_w": bg_eirp,
        "background_distance_pc": bg_dist,
        "primary_eirp_min_w": {
            **prim_q,
            "max": float(prim_eirp.max()),
            "n_targets": len(primaries),
        },
        "primary_median_distance_pc": d_prim,
        "median_eirp_min_ratio_background_over_primary": bg_eirp["p50"] / prim_q["p50"],
        "decomposition_of_the_median_ratio": {
            "distance_term": (bg_dist["p50"] / d_prim) ** 2,
            "beam_term": float(np.median(1.0 / beam)),
            "note": (
                "the d^2 term dominates: the beam costs at most a factor of 2 inside the FWHM "
                "cone, the distance costs orders of magnitude"
            ),
        },
    }


def pointing_centre_audit(graph: FactorGraph, field: BackgroundField) -> dict:
    """Was each primary found inside its own cone? (offline: artifact + locked catalog)

    Every cone is centred on the primary's catalog coordinates, except the pointings moved
    by the pinned pointing-correction overlay (ruling v0.2.1-R5). A primary with no Gaia
    counterpart in its own cone either is absent from Gaia DR3 with a usable parallax
    (very bright or very high proper-motion stars) or does not lie where the catalog says.
    The audit reports the declination band holding most unfound primaries beside its mirror
    band: many targets on one side and none on the other is the signature of a declination
    sign error in the primary catalog, in which case those cones sample the wrong field.

    Two blocks separate what the overlay has fixed from what is still wrong in this
    artifact: ``corrected_via_overlay`` (cones fetched at the overlay's J2000 centre) and
    ``uncorrected_flipped_cones`` (overlay entries this artifact still centres on the
    catalog declination). The primary catalog itself is immutable (rulings v0.2.1-R3 / R5)
    and declination bands below are always the catalog's.
    """
    prim = primary_sightings(graph.surveys[field.parent_survey_id], graph.root)
    names = [str(x) for x in prim.names]
    has_id = prim.gaia_dr3_source_id != MISSING_GAIA_ID
    rec = field.record
    by_id = {
        m.primary_target_name
        for m in rec.validation_sample.primary_matched
        if m.match == "gaia_dr3_source_id"
    }
    absorbed = field.absorbing_primaries()
    suspected = {r.primary_target_name for r in rec.dedup_audit.suspected_unmatched_primary}
    found = np.array([n in absorbed or n in suspected for n in names])
    band = np.floor(np.abs(prim.dec_deg)).astype(int)
    north = prim.dec_deg >= 0
    per_pointing = np.bincount(
        field.shells.pointing_index, weights=field.shells.count, minlength=len(names)
    )
    no_id = [n for n, ok in zip(names, has_id, strict=True) if not ok]
    # Ruling v0.2.1-R5: cones the pinned overlay moved, against cones it should have moved
    # but that this artifact still centres on the catalog's (wrong) coordinates.
    pc = field.pointing_corrections()
    index = {n: i for i, n in enumerate(names)}
    how = {m.primary_target_name: m.match for m in rec.validation_sample.primary_matched}
    fixed, stale = pc["corrected_target_names"], pc["uncorrected_target_names"]
    stale_idx = np.array([index[n] for n in stale], dtype=int)
    overlay_ref = {
        "overlay_path": pc["pointing_corrections_path"],
        "overlay_sha256": pc["pointing_corrections_sha256"],
        "claim_id": pc["claim_id"],
        "ruling": pc["ruling"],
        "locked_catalog_path": pc["locked_catalog_path"],
        "locked_catalog_sha256": pc["locked_catalog_sha256"],
        "locked_catalog_modified": False,
    }
    out = {
        "uncorrected_flipped_cones": {
            "n": len(stale),
            "targets": sorted(stale),
            "n_background_stars_in_those_pointings": int(per_pointing[stale_idx].sum()),
            "fraction_of_background_stars": float(
                per_pointing[stale_idx].sum() / per_pointing.sum()
            ),
            "reading": (
                "cones still centred on a catalog declination that the pinned overlay identifies "
                "as sign-flipped: their stars are ~21 deg from the observed beam. Re-fetch and "
                "re-promote with the overlay to clear them."
                if stale
                else "none: every cone the pinned overlay corrects was fetched at its corrected "
                "centre"
            ),
            **overlay_ref,
        },
        "corrected_via_overlay": {
            "n": len(fixed),
            "n_pinned_corrections": pc["n_pinned_corrections"],
            "targets": sorted(fixed),
            "n_absorbed_by_gaia_id": sum(how.get(n) == "gaia_dr3_source_id" for n in fixed),
            "n_absorbed_by_angular_fallback": sum(how.get(n) == "angular_fallback" for n in fixed),
            "not_absorbed": sorted(n for n in fixed if n not in how),
            "n_listed_as_suspected_unmatched_primary": sum(n in suspected for n in fixed),
            "cones": pc["corrected_pointings"],
            "policy": (
                "non-destructive overlay: cone centres and de-duplication positions come from "
                "the SHA-256-pinned claim file; the locked primary catalog is read, never written"
            ),
            **overlay_ref,
        },
        "n_primaries": len(names),
        "n_with_gaia_id": int(has_id.sum()),
        "n_with_gaia_id_sighted_in_own_cone": sum(n in by_id for n in names),
        "with_gaia_id_not_sighted_in_own_cone": sorted(
            n for n, ok in zip(names, has_id, strict=True) if ok and n not in by_id
        ),
        "n_without_gaia_id": len(no_id),
        "n_without_gaia_id_absorbed_by_angular_fallback": sum(n in absorbed for n in no_id),
        "n_without_gaia_id_suspected_only": sum(
            n in suspected and n not in absorbed for n in no_id
        ),
        "n_no_counterpart_in_own_cone": int((~found).sum()),
        "densest_band_of_unfound_primaries": None,
        "policy": (
            "reported, not corrected: the primary catalog is immutable (ruling v0.2.1-R3); a "
            "catalog correction needs its own provenance run and a new pinned SHA-256"
        ),
    }
    if (~found).any():
        keys = band[~found] * 2 + north[~found]
        vals, counts = np.unique(keys, return_counts=True)
        top = int(vals[np.argmax(counts)])  # ties: lowest declination band
        b, is_north = divmod(top, 2)
        here = (band == b) & (north == bool(is_north))
        mirror = (band == b) & (north != bool(is_north))
        missing = here & ~found
        lo, hi = (b, b + 1) if is_north else (-(b + 1), -b)
        out["densest_band_of_unfound_primaries"] = {
            "declination_band_deg": [lo, hi],
            "mirror_band_deg": [-hi, -lo],
            "n_targets": int(here.sum()),
            "n_targets_in_mirror_band": int(mirror.sum()),
            "n_no_counterpart": int(missing.sum()),
            "targets_no_counterpart": sorted(n for n, m in zip(names, missing, strict=True) if m),
            "n_background_stars_in_those_pointings": int(per_pointing[missing].sum()),
            "fraction_of_background_stars": float(per_pointing[missing].sum() / per_pointing.sum()),
            "reading": (
                "many catalog targets in one declination band, none in its mirror band, and "
                "most of them missing from their own cone: consistent with a declination sign "
                "error in the primary catalog for that band. If so, these cones were fetched "
                "around the mirrored position and their stars are not the stars in the beam."
            ),
        }
    return out


def _mode_terms(
    graph: FactorGraph,
    shells: radio.BackgroundShells,
    pair: Sequence[str],
    geometry: str,
    mode: str,
) -> dict[str, dict]:
    """Per-hypothesis zero-detection term of the shells in one radio mode (seed-free)."""
    cfg = RunConfig(geometry=geometry, radio_mode=mode, seed=0, channels=("radio",))
    out = {}
    for hid in pair:
        params = resolve_parameters(graph, graph.hypotheses[hid], cfg.range_grid_points)
        lf = None
        if mode == "distance_scaled_eirp":
            lf = radio.LogNormalEIRP(
                params.quantity("transmitter_eirp_median_w"),
                params.value("transmitter_eirp_sigma_dex"),
            )
        out[hid] = background_radio_term(
            expansion_model(params, cfg),
            shells,
            mode,
            params.value("radio_detect_prob"),
            lf,
            cfg.condition_on_earth_unreached,
        )
    return out


def realized_eig(
    graph: FactorGraph,
    field: BackgroundField,
    pair: Sequence[str],
    geometry: str = "thin_disk_2d",
    shells: radio.BackgroundShells | None = None,
) -> dict:
    """EIG of the background-field null with the real star population, both radio modes.

    Same definition as the Milestone 2 prediction (``queue.runners.eig_background_extension``):
    uniform prior over ``pair``, binary outcome, independent-product P(0 | H), the job's
    geometry, Earth unreached. Deterministic: analytic reach probabilities, no RNG.
    """
    shells = field.shells if shells is None else shells
    prior = {h: 1.0 / len(pair) for h in pair}
    by_mode = {}
    for mode, meta in RADIO_MODE_LABELS.items():
        terms = _mode_terms(graph, shells, pair, geometry, mode)
        p0 = {h: t["p_zero_independent_fkg_lower"] for h, t in terms.items()}
        upper = {h: t["p_zero_frechet_upper"] for h, t in terms.items()}
        lo, hi = eig.binary_eig_range(prior, {h: (p0[h], upper[h]) for h in pair})
        a, b = pair[0], pair[1]
        by_mode[mode] = {
            **meta,
            "data_tiers": sorted({shells.data_tier, "illustrative_calibration"}),
            "illustrative_reason": ILLUSTRATIVE_INPUT_REASONS[meta["illustrative_input"]],
            "p_zero_fkg_lower": p0,
            "p_zero_frechet_upper": upper,
            "p_zero_markov_lower": {h: t["p_zero_markov_lower"] for h, t in terms.items()},
            "log10_p_zero_fkg_lower": {
                h: t["log10_p_zero_independent_fkg_lower"] for h, t in terms.items()
            },
            "expected_reached": {h: t["expected_reached"] for h, t in terms.items()},
            "expected_detections": {h: t["expected_detections"] for h, t in terms.items()},
            "mean_p_detect_given_reached": {
                h: t["mean_p_detect_given_reached"] for h, t in terms.items()
            },
            "eig_bits": eig.mutual_information_bits(
                prior, {h: eig.binary_outcome(v) for h, v in p0.items()}
            ),
            "eig_bits_range_fkg_to_frechet": [lo, hi],
            "observed_outcome": "0 detections",
            "realized_information_bits": eig.realized_information_bits(prior, p0),
            "posterior_after_observed_null": eig.posterior(prior, p0),
            "log10_bf_fkg": terms[a]["log10_p_zero_independent_fkg_lower"]
            - terms[b]["log10_p_zero_independent_fkg_lower"],
        }
    return {
        "prior": {"type": "uniform", "over": list(pair)},
        "geometry": geometry,
        "condition_on_earth_unreached": True,
        "outcome": EIG_OUTCOME,
        "n_stars": shells.n_stars,
        "n_cells": shells.n_cells,
        "population_data_tier": shells.data_tier,
        "summary_sha256": shells.source_sha256,
        "by_radio_mode": by_mode,
    }


def bounds_table(realized: dict) -> list[dict]:
    """Harris-FKG / Frechet / Markov bounds on P(no detection), one row per (mode, H)."""
    rows = []
    for mode, m in realized["by_radio_mode"].items():
        for hid in realized["prior"]["over"]:
            rows.append(
                {
                    "radio_mode": mode,
                    "radio_mode_label": m["label"],
                    "hypothesis_id": hid,
                    "expected_reached": m["expected_reached"][hid],
                    "expected_detections": m["expected_detections"][hid],
                    "p_zero_markov_lower": m["p_zero_markov_lower"][hid],
                    "p_zero_fkg_lower": m["p_zero_fkg_lower"][hid],
                    "p_zero_frechet_upper": m["p_zero_frechet_upper"][hid],
                    "log10_p_zero_fkg_lower": m["log10_p_zero_fkg_lower"][hid],
                    "data_tiers": m["data_tiers"],
                }
            )
    return rows
