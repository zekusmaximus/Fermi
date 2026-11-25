"""Unit & schema validation (MILESTONE_1 CI item 1)."""

from __future__ import annotations

import pytest
import yaml

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
