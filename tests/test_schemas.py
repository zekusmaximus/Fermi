"""Unit & schema validation (MILESTONE_1 CI item 1)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from fermi_desk.errors import (
    ProvenanceError,
    SchemaValidationError,
    Tier3SchemaError,
    Tier3ScoringError,
    UnitValidationError,
)
from fermi_desk.inference.engine import RunConfig, build_context, evaluate_hypothesis
from fermi_desk.registry import load_factor_graph, run_validation
from fermi_desk.schemas import Hypothesis, Parameter, Survey, parse_unit


def _param(**kw):
    base = dict(id="x", symbol="x", unit="pc", prior_type="uniform", bounds=[0, 1], description="d")
    base.update(kw)
    return base


@pytest.mark.parametrize("bad", ["furlongs_per_banana", "", "kpc^^2", "1 / (parsnip * Gyr)"])
def test_rejects_bad_astropy_units(bad):
    with pytest.raises(UnitValidationError):
        Parameter.model_validate(_param(unit=bad))


@pytest.mark.parametrize("good", ["1 / (kpc2 * Gyr)", "pc", "Gyr", "W", "dimensionless"])
def test_accepts_valid_units(good):
    assert Parameter.model_validate(_param(unit=good)).astropy_unit == parse_unit(good)


def test_log_uniform_zero_lower_bound_requires_floor():
    with pytest.raises(SchemaValidationError):
        Parameter.model_validate(_param(unit="dimensionless", prior_type="log_uniform"))
    p = Parameter.model_validate(
        _param(unit="dimensionless", prior_type="log_uniform", log_floor=1e-8)
    )
    assert p.log_floor == 1e-8


def _tier3(**kw):
    base = dict(
        id="z",
        name="Z",
        tier=3,
        summary="s",
        quarantine_reason="r",
        monoculture_audit=dict(
            required_compliance_fraction=1.0, timescale_gyr=1.0, defector_vulnerability="v"
        ),
    )
    base.update(kw)
    return base


def test_tier3_requires_monoculture_audit():
    with pytest.raises(Tier3SchemaError):
        Hypothesis.model_validate(_tier3(monoculture_audit=None))
    with pytest.raises(Tier3SchemaError):
        Hypothesis.model_validate(_tier3(quarantine_reason=None))
    assert Hypothesis.model_validate(_tier3()).is_quarantined


def test_tier3_cannot_declare_empirical_channels():
    with pytest.raises(Tier3SchemaError):
        Hypothesis.model_validate(_tier3(channels_activated=["radio"]))


def test_tier3_blocked_from_empirical_scoring(graph):
    ctx = build_context(
        graph, RunConfig(geometry="thin_disk_2d", radio_mode="flat_probability", seed=1)
    )
    for hid in ("zoo_hypothesis", "dark_forest"):
        with pytest.raises(Tier3ScoringError):
            evaluate_hypothesis(graph.hypotheses[hid], ctx)


def test_repository_data_validates(graph):
    assert len(graph.registry.parameters) >= 8
    tiers = {h.tier for h in graph.hypotheses.values()}
    assert tiers == {1, 2, 3}
    assert {"radio", "infrared", "solar_system"} == {s.channel for s in graph.surveys.values()}
    failures = [r for r in run_validation(graph.root) if not r.ok]
    assert failures == []


def test_illustrative_ir_calibrations_never_labelled_published(graph):
    ir = graph.surveys["hephaistos_wise_mid_ir"]
    assert len(ir.infrared.calibrations) == 15
    assert ir.infrared.calibration_data_tier == "illustrative_calibration"
    assert ir.data_tier == "illustrative_calibration"


def test_unknown_parameter_in_hypothesis_rejected(data_copy):
    path = data_copy / "data" / "hypotheses" / "frequent_slow_expanders.yaml"
    doc = yaml.safe_load(path.read_text())
    doc["parameter_constraints"]["warp_factor"] = {"mode": "fixed", "value": 9.0}
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(SchemaValidationError):
        load_factor_graph(data_copy)


def test_catalog_checksum_mismatch_is_provenance_error(data_copy):
    csv = data_copy / "data" / "catalogs" / "bl_692_targets.csv"
    csv.write_bytes(csv.read_bytes() + b"\n")
    results = {r.name: r for r in run_validation(data_copy)}
    bad = results["checksum:data/catalogs/bl_692_targets.csv"]
    assert not bad.ok and "ProvenanceError" in bad.detail


def test_catalog_without_checksum_rejected(graph):
    doc = graph.surveys["bl_enriquez_2017_692"].model_dump()
    doc["catalog_sha256"] = None
    with pytest.raises(ProvenanceError):
        Survey.model_validate(doc)


def test_published_catalog_requires_citations(graph):
    doc = graph.surveys["bl_enriquez_2017_692"].model_dump()
    doc["citations"] = []
    with pytest.raises(ProvenanceError):
        Survey.model_validate(doc)


# ------------------------------------------------------- v0.2.0 registry additions
def test_geometry_mode_literal_matches_expansion_kernel():
    from typing import get_args

    from fermi_desk.models.expansion import DISK_PROFILES, GEOMETRY_MODES
    from fermi_desk.schemas import GeometryMode

    assert get_args(GeometryMode) == GEOMETRY_MODES
    assert GEOMETRY_MODES == ("thin_disk_2d", "thick_disk_3d", "finite_disk_3d")
    assert DISK_PROFILES == ("uniform", "exponential")


def test_galactic_structure_parameters_are_documented_with_units(graph):
    import astropy.units as u

    want = {
        "galactic_disk_radius_kpc": 15.0,
        "solar_galactocentric_radius_kpc": 8.178,
        "galactic_radial_scale_length_kpc": 2.6,
    }
    for pid, default in want.items():
        p = graph.registry.get(pid)
        assert p.default == default and p.astropy_unit == u.kpc
        assert p.quantity(p.default).to_value(u.pc) == pytest.approx(default * 1e3)
        assert p.bounds[0] < default < p.bounds[1]
        # scoped to the finite disk: never part of a thin_disk_2d / thick_disk_3d model
        assert p.geometry_modes == ["finite_disk_3d"]
        assert p.applies_to("finite_disk_3d")
        assert not p.applies_to("thin_disk_2d") and not p.applies_to("thick_disk_3d")
        assert p.description
    # measured values carry a citation; R_gal is declared a convention instead
    assert graph.registry.get("solar_galactocentric_radius_kpc").citations
    assert graph.registry.get("galactic_radial_scale_length_kpc").citations
    assert "convention" in graph.registry.get("galactic_disk_radius_kpc").description
    # reused, unscoped parameters (reported by every geometry, as before)
    for pid, default in (("disk_scale_height_pc", 300.0), ("solar_height_pc", 20.8)):
        p = graph.registry.get(pid)
        assert p.default == default and p.astropy_unit == u.pc and p.geometry_modes is None
    # the lookback marginalisation prior lives in the registry: 50 kyr - 10 Myr
    t = graph.registry.get("lookback_window_gyr")
    assert t.prior_type == "log_uniform" and t.default == 1.0e-3
    assert t.marginalization_prior_bounds == (5.0e-5, 1.0e-2)
    assert t.marginalization_log_bounds == (5.0e-5, 1.0e-2)
    names = {r.name: r for r in run_validation(graph.root)}
    for pid in want:
        assert names[f"unit:{pid}"].ok and "kpc" in names[f"unit:{pid}"].detail


def test_parameter_report_is_scoped_to_the_run_geometry(graph):
    reported = {}
    for geometry in ("thin_disk_2d", "thick_disk_3d", "finite_disk_3d"):
        cfg = RunConfig(
            geometry=geometry,
            radio_mode="flat_probability",
            seed=1,
            n_mc=50,
            n_importance=256,
            channels=("radio",),
        )
        ctx = build_context(graph, cfg)
        out = evaluate_hypothesis(graph.hypotheses["frequent_slow_expanders"], ctx)
        reported[geometry] = out["parameters"]
    finite_only = {
        "galactic_disk_radius_kpc",
        "solar_galactocentric_radius_kpc",
        "galactic_radial_scale_length_kpc",
    }
    assert reported["thin_disk_2d"] == reported["thick_disk_3d"]
    assert not finite_only & set(reported["thin_disk_2d"])
    assert set(reported["finite_disk_3d"]) == set(reported["thin_disk_2d"]) | finite_only
    assert reported["finite_disk_3d"]["galactic_disk_radius_kpc"] == {"value": 15.0, "unit": "kpc"}


def test_bad_geometry_scope_or_unit_is_rejected(data_copy):
    path = data_copy / "data" / "parameters.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in doc["parameters"]}
    by_id["galactic_disk_radius_kpc"]["geometry_modes"] = ["spherical_halo_3d"]
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_factor_graph(data_copy)
    by_id["galactic_disk_radius_kpc"]["geometry_modes"] = ["finite_disk_3d"]
    by_id["galactic_disk_radius_kpc"]["unit"] = "kiloparsecs_ish"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(UnitValidationError):
        load_factor_graph(data_copy)
