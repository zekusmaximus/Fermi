"""Factor-graph loader, provenance verification, and double-counting guard.

All file I/O for data contracts lives here (never inside ``models/``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from fermi_desk.errors import (
    DoubleCountingError,
    FermiDeskError,
    ProvenanceError,
    SchemaValidationError,
)
from fermi_desk.schemas import Hypothesis, ParameterRegistry, Survey

MergePolicy = Literal["keep_first"]
CATALOG_REQUIRED_COLUMNS = (
    "target_name",
    "ra_deg",
    "dec_deg",
    "dist_pc",
    "dist_source",
    "pointing_group_id",
    "data_tier",
)


# ------------------------------------------------------------------------ discovery
def find_project_root(start: Path | None = None) -> Path:
    """Locate the directory holding ``data/parameters.yaml``.

    Order: ``$FERMI_DESK_ROOT``, then walking up from ``start`` (default: cwd),
    then the source checkout containing this package.
    """
    env = os.environ.get("FERMI_DESK_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for cand in (here, *here.parents):
        if (cand / "data" / "parameters.yaml").is_file():
            return cand
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "data" / "parameters.yaml").is_file():
        return pkg_root
    raise FileNotFoundError("Could not locate data/parameters.yaml; set FERMI_DESK_ROOT")


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise SchemaValidationError(f"{path}: expected a YAML mapping")
    return doc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------- factor graph
@dataclass(frozen=True)
class FactorGraph:
    root: Path
    registry: ParameterRegistry
    hypotheses: dict[str, Hypothesis]
    surveys: dict[str, Survey]

    def surveys_for_channels(self, channels: Iterable[str]) -> list[Survey]:
        chans = set(channels)
        return [s for s in self.surveys.values() if s.channel in chans]


def _check_constraints(registry: ParameterRegistry, hyp: Hypothesis) -> None:
    for pid, c in hyp.parameter_constraints.items():
        if pid not in registry.ids:
            raise SchemaValidationError(f"{hyp.id}: unknown parameter {pid!r}")
        lo, hi = registry.get(pid).bounds
        vals = [c.value] if c.mode == "fixed" else [c.min, c.max]
        for v in vals:
            if not lo <= v <= hi:
                raise SchemaValidationError(
                    f"{hyp.id}: {pid}={v} outside registry bounds [{lo}, {hi}]"
                )


def load_factor_graph(root: Path | None = None) -> FactorGraph:
    root = (root or find_project_root()).resolve()
    data = root / "data"
    registry = ParameterRegistry.model_validate(_read_yaml(data / "parameters.yaml"))

    hypotheses: dict[str, Hypothesis] = {}
    for path in sorted((data / "hypotheses").glob("*.yaml")):
        hyp = Hypothesis.model_validate(_read_yaml(path))
        if hyp.id != path.stem:
            raise SchemaValidationError(f"{path}: id {hyp.id!r} must match file stem")
        if hyp.id in hypotheses:
            raise SchemaValidationError(f"Duplicate hypothesis id {hyp.id!r}")
        _check_constraints(registry, hyp)
        hypotheses[hyp.id] = hyp

    surveys: dict[str, Survey] = {}
    for path in sorted((data / "surveys").glob("*.yaml")):
        sv = Survey.model_validate(_read_yaml(path))
        if sv.id != path.stem:
            raise SchemaValidationError(f"{path}: id {sv.id!r} must match file stem")
        surveys[sv.id] = sv
    return FactorGraph(root=root, registry=registry, hypotheses=hypotheses, surveys=surveys)


# ------------------------------------------------------------------ double counting
def check_pointing_groups(
    surveys: Sequence[Survey],
    merge_policies: Mapping[str, MergePolicy] | None = None,
) -> list[Survey]:
    """Return the surveys to condition on, refusing silent double-counting.

    Two surveys sharing a ``pointing_group_id`` raise ``DoubleCountingError``
    unless that group has an explicit merge policy (``keep_first`` keeps the
    first survey in the given order and drops the rest).
    """
    policies = dict(merge_policies or {})
    kept: list[Survey] = []
    seen: dict[str, Survey] = {}
    for sv in surveys:
        prior = seen.get(sv.pointing_group_id)
        if prior is None:
            seen[sv.pointing_group_id] = sv
            kept.append(sv)
            continue
        policy = policies.get(sv.pointing_group_id)
        if policy == "keep_first":
            continue
        raise DoubleCountingError(
            f"Surveys {prior.id!r} and {sv.id!r} share pointing_group_id "
            f"{sv.pointing_group_id!r}; declare a merge policy to resolve explicitly"
        )
    return kept


# --------------------------------------------------------------------- catalogs
@dataclass(frozen=True)
class TargetTable:
    """Survey targets with Galactic Cartesian positions relative to Earth."""

    names: np.ndarray
    distance: u.Quantity  # (N,) pc
    xyz: u.Quantity  # (N, 3) pc; x toward Galactic centre, z toward NGP
    dist_source: np.ndarray
    pointing_group_id: str
    data_tier: str
    catalog_sha256: str
    ra_deg: np.ndarray | None = None
    dec_deg: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.names)

    @classmethod
    def from_arrays(
        cls,
        names: Sequence[str],
        xyz: u.Quantity,
        pointing_group_id: str = "synthetic",
        data_tier: str = "synthetic_proxy",
    ) -> TargetTable:
        """Build a table directly from Cartesian positions (tests / synthetic studies)."""
        xyz = u.Quantity(xyz).to(u.pc).reshape(-1, 3)
        return cls(
            names=np.asarray(list(names), dtype=object),
            distance=np.linalg.norm(xyz.value, axis=1) * u.pc,
            xyz=xyz,
            dist_source=np.full(len(xyz), "synthetic", dtype=object),
            pointing_group_id=pointing_group_id,
            data_tier=data_tier,
            catalog_sha256="",
        )

    def with_distance(self, index: int, distance: u.Quantity, source: str) -> TargetTable:
        """Copy with one target moved along its line of sight to ``distance`` (adversary toggles)."""
        d_old = float(self.distance[index].to_value(u.pc))
        d_new = float(u.Quantity(distance).to_value(u.pc))
        if d_old <= 0 or d_new <= 0:
            raise ValueError("distances must be positive")
        xyz = self.xyz.to_value(u.pc).copy()
        xyz[index] *= d_new / d_old
        dist = self.distance.to_value(u.pc).copy()
        dist[index] = d_new
        src = self.dist_source.copy()
        src[index] = source
        return replace(self, xyz=xyz * u.pc, distance=dist * u.pc, dist_source=src)

    @classmethod
    def empty(cls) -> TargetTable:
        return cls.from_arrays([], np.zeros((0, 3)) * u.pc)


def verify_catalog_checksum(survey: Survey, root: Path) -> Path:
    if survey.catalog_path is None:
        raise ProvenanceError(f"Survey {survey.id!r} has no catalog")
    path = root / survey.catalog_path
    if not path.is_file():
        raise ProvenanceError(f"Catalog {path} missing (run scripts/fetch_bl_692.py)")
    actual = sha256_file(path)
    if actual != survey.catalog_sha256:
        raise ProvenanceError(
            f"Catalog checksum mismatch for {survey.id!r}: pinned {survey.catalog_sha256}, "
            f"file {actual}"
        )
    return path


def load_targets(survey: Survey, root: Path) -> TargetTable:
    path = verify_catalog_checksum(survey, root)
    df = pd.read_csv(path)
    missing = [c for c in CATALOG_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"{path}: missing columns {missing}")
    if len(df) != survey.n_targets:
        raise ProvenanceError(f"{path}: {len(df)} rows but survey declares {survey.n_targets}")
    if df["target_name"].duplicated().any():
        raise SchemaValidationError(f"{path}: duplicate target_name rows")
    groups = set(df["pointing_group_id"].astype(str))
    if groups != {survey.pointing_group_id}:
        raise ProvenanceError(f"{path}: pointing_group_id {groups} != {survey.pointing_group_id}")
    tiers = set(df["data_tier"].astype(str))
    if tiers != {survey.data_tier}:
        raise ProvenanceError(f"{path}: row data_tier {tiers} != survey tier {survey.data_tier}")
    dist = pd.to_numeric(df["dist_pc"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(dist) & (dist > 0)):
        raise SchemaValidationError(f"{path}: dist_pc must be finite and positive for all rows")

    coord = SkyCoord(
        ra=df["ra_deg"].to_numpy(dtype=float) * u.deg,
        dec=df["dec_deg"].to_numpy(dtype=float) * u.deg,
        distance=dist * u.pc,
        frame="icrs",
    ).galactic
    cart = coord.cartesian
    xyz = np.column_stack([cart.x.to_value(u.pc), cart.y.to_value(u.pc), cart.z.to_value(u.pc)])
    return TargetTable(
        names=df["target_name"].to_numpy(dtype=object),
        distance=dist * u.pc,
        xyz=xyz * u.pc,
        dist_source=df["dist_source"].to_numpy(dtype=object),
        pointing_group_id=survey.pointing_group_id,
        data_tier=survey.data_tier,
        catalog_sha256=survey.catalog_sha256 or "",
        ra_deg=df["ra_deg"].to_numpy(dtype=float),
        dec_deg=df["dec_deg"].to_numpy(dtype=float),
    )


# --------------------------------------------------------------------- validation
@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_provenance_json(survey: Survey, root: Path) -> str:
    if survey.catalog_provenance_path is None:
        raise ProvenanceError(f"{survey.id}: no catalog_provenance_path")
    prov = json.loads((root / survey.catalog_provenance_path).read_text(encoding="utf-8"))
    if prov.get("output_sha256") != survey.catalog_sha256:
        raise ProvenanceError(f"{survey.id}: provenance output_sha256 != pinned catalog_sha256")
    if prov.get("data_tier") != survey.data_tier:
        raise ProvenanceError(f"{survey.id}: provenance data_tier != survey data_tier")
    if prov.get("row_count") != survey.n_targets:
        raise ProvenanceError(f"{survey.id}: provenance row_count != n_targets")
    return f"sources={sorted((prov.get('sources') or {}).values(), key=str)}"


def _check_flux_limit(survey: Survey) -> str:
    """minimum_flux_limit_jy must equal the radiometer-equation S_min for a 1 Hz transmitter."""
    rs = survey.radio_sensitivity
    s_min = (
        rs.snr_threshold
        * rs.sefd
        * np.sqrt(rs.channel_bandwidth / (rs.n_pol * rs.integration_time))
        / (1 * u.Hz)
    ).to(u.Jy)
    if not math.isclose(s_min.value, survey.minimum_flux_limit_jy, rel_tol=0.05):
        raise ProvenanceError(
            f"{survey.id}: minimum_flux_limit_jy={survey.minimum_flux_limit_jy} but radiometer "
            f"equation gives {s_min:.3f} (1 Hz transmitter)"
        )
    return (
        f"declared {survey.minimum_flux_limit_jy} Jy; radiometer equation {s_min:.3f} (within 5%)"
    )


def _check_anchor_claim(survey: Survey, root: Path, claim_path: str) -> str:
    claim = _read_yaml(root / claim_path)
    df = pd.read_csv(verify_catalog_checksum(survey, root))
    rows = df[df["target_name"] == claim["target_name"]]
    if len(rows) != 1:
        raise ProvenanceError(f"{claim_path}: target {claim['target_name']!r} not in catalog")
    row = rows.iloc[0]
    checked = []
    for d in claim.get("distances", []):
        if "data_tier" not in d:
            raise ProvenanceError(f"{claim_path}: distance {d.get('label')} lacks data_tier")
        col = d.get("catalog_column")
        if col:
            if not math.isclose(float(row[col]), float(d["value_pc"]), rel_tol=1e-6):
                raise ProvenanceError(
                    f"{claim_path}: {d['label']}={d['value_pc']} disagrees with catalog "
                    f"{col}={row[col]}"
                )
            checked.append(f"{d['label']}={d['value_pc']} pc")
    return f"{claim['target_name']}: " + ", ".join(checked)


def run_validation(root: Path | None = None) -> list[CheckResult]:
    """Run every schema / unit / checksum check, collecting failures instead of stopping."""
    results: list[CheckResult] = []

    def record(name: str, fn: Callable[[], str]) -> None:
        try:
            results.append(CheckResult(name, True, fn()))
        except (FermiDeskError, FileNotFoundError, KeyError, ValueError) as exc:
            results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}"))

    try:
        graph = load_factor_graph(root)
    except (FermiDeskError, FileNotFoundError, ValueError) as exc:
        return [CheckResult("load_factor_graph", False, f"{type(exc).__name__}: {exc}")]

    reg = graph.registry
    results.append(
        CheckResult("parameters.yaml", True, f"{len(reg.parameters)} parameters, all units parsed")
    )
    for p in reg.parameters:
        results.append(
            CheckResult(
                f"unit:{p.id}",
                True,
                f"{p.unit!r} -> {p.astropy_unit.to_string() or 'dimensionless_unscaled'}",
            )
        )
    for h in graph.hypotheses.values():
        label = "QUARANTINED (monoculture_audit present)" if h.is_quarantined else "scorable"
        results.append(CheckResult(f"hypothesis:{h.id}", True, f"tier {h.tier}, {label}"))
    for s in graph.surveys.values():
        results.append(
            CheckResult(
                f"survey:{s.id}", True, f"{s.channel}, {s.pointing_group_id}, {s.data_tier}"
            )
        )
    record(
        "pointing_group_ids unique",
        lambda: (
            check_pointing_groups(list(graph.surveys.values()))
            and f"{len(graph.surveys)} distinct groups"
        ),
    )
    for s in graph.surveys.values():
        if s.catalog_path:
            record(
                f"checksum:{s.catalog_path}",
                lambda s=s: f"sha256 {verify_catalog_checksum(s, graph.root) and s.catalog_sha256}",
            )
            record(
                f"catalog_schema:{s.id}",
                lambda s=s: f"{len(load_targets(s, graph.root))} targets, tier {s.data_tier}",
            )
            record(f"provenance:{s.id}", lambda s=s: _check_provenance_json(s, graph.root))
            if s.radio_sensitivity is not None and s.minimum_flux_limit_jy is not None:
                record(f"flux_limit:{s.id}", lambda s=s: _check_flux_limit(s))
            for cp in s.anchor_claims:
                record(
                    f"anchor_claim:{cp}", lambda s=s, cp=cp: _check_anchor_claim(s, graph.root, cp)
                )
    return results
