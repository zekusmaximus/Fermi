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
    MergePolicyError,
    ProvenanceError,
    SchemaValidationError,
    UnitValidationError,
)
from fermi_desk.models import hephaistos, hephaistos_fits, radio
from fermi_desk.models.expansion import DISK_PROFILES, FINITE_DISK, ExpansionModel
from fermi_desk.schemas import (
    BACKGROUND_SUMMARY_MAX_BYTES,
    BACKGROUND_SUMMARY_PATH,
    POINTING_MERGE_POLICIES,
    REJECT_DUPLICATE_POINTING_GROUP,
    UNION_TARGETS_BY_POINTING,
    BackgroundSummaryFile,
    Hypothesis,
    MergeProvenance,
    ParameterRegistry,
    PointingCorrectionsFile,
    SigmaSysCalibrationFile,
    Survey,
    WiseFitsGate2File,
    validate_pointing_merge_policy,
)

ANGULAR_FALLBACK = 3.0 * u.arcsec  # de-duplication fallback when a Gaia DR3 id is missing
MISSING_GAIA_ID = -1  # sentinel in SightingTable.gaia_dr3_source_id (real ids are positive)
COLLISION_RULE = (
    "one record per physical star: the sighting with the minimum radial offset from its "
    "beam centre (maximum beam gain, lowest EIRP_min) is kept; ties go to the primary "
    "(on-axis) observation, then to input order"
)
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
    for sv in surveys.values():
        _check_parent_pointing(sv, surveys)
    return FactorGraph(root=root, registry=registry, hypotheses=hypotheses, surveys=surveys)


def _check_parent_pointing(sv: Survey, surveys: Mapping[str, Survey]) -> None:
    """A declared parent pointing must exist and be compatible (same group and channel)."""
    if sv.parent_pointing_survey_id is None:
        return
    parent = surveys.get(sv.parent_pointing_survey_id)
    if parent is None:
        raise SchemaValidationError(
            f"{sv.id}: parent_pointing_survey_id {sv.parent_pointing_survey_id!r} is not a survey"
        )
    if parent.is_pointing_extension:
        raise SchemaValidationError(f"{sv.id}: its parent {parent.id!r} is itself an extension")
    if (parent.pointing_group_id, parent.channel) != (sv.pointing_group_id, sv.channel):
        raise SchemaValidationError(
            f"{sv.id}: parent {parent.id!r} has a different pointing_group_id or channel; a "
            "union_targets_by_pointing extension must share both with its parent"
        )


# ------------------------------------------------------------------ double counting
@dataclass(frozen=True)
class PointingMerge:
    """One ``union_targets_by_pointing`` merge: a parent survey and its extensions."""

    pointing_group_id: str
    parent: Survey
    extensions: tuple[Survey, ...]


@dataclass(frozen=True)
class PointingResolution:
    """Which surveys a run conditions on, and how shared pointing groups were resolved.

    ``active``: surveys evaluated as their own likelihood terms (parents and stand-alone
    surveys). ``merges``: extensions folded into their parent, star by star.
    ``held_out``: declared extensions left out because the run kept the default
    ``reject_duplicate_pointing_group`` policy; they are reported, never dropped silently.
    """

    policy: str
    active: tuple[Survey, ...]
    merges: tuple[PointingMerge, ...]
    held_out: tuple[dict, ...]


def resolve_pointing_groups(
    surveys: Sequence[Survey],
    policy: str | Mapping[str, str] = REJECT_DUPLICATE_POINTING_GROUP,
) -> PointingResolution:
    """Resolve surveys sharing a ``pointing_group_id`` without ever double-counting.

    * ``keep_first`` (or any unknown policy) raises ``MergePolicyError``.
    * Surveys sharing a group WITHOUT a declared compatible parent-pointing relationship
      raise ``DoubleCountingError`` under every policy: nothing is dropped or merged on
      the strength of load order.
    * A survey that declares ``parent_pointing_survey_id`` (and therefore
      ``pointing_merge_policy: union_targets_by_pointing``) is an extension of its parent:
      under ``union_targets_by_pointing`` it is merged into the parent star by star; under
      the default ``reject_duplicate_pointing_group`` it stays out of the run and is listed
      in ``held_out`` (the parent then runs exactly as a single-survey run).

    ``policy`` is one policy name, or a mapping ``pointing_group_id -> policy`` (groups
    not listed use the default).
    """
    if isinstance(policy, Mapping):
        per_group = {g: validate_pointing_merge_policy(p) for g, p in policy.items()}
        default = REJECT_DUPLICATE_POINTING_GROUP
        label = (
            UNION_TARGETS_BY_POINTING
            if UNION_TARGETS_BY_POINTING in per_group.values()
            else default
        )
    else:
        per_group, default = {}, validate_pointing_merge_policy(policy)
        label = default
    groups: dict[str, list[Survey]] = {}
    for sv in surveys:
        groups.setdefault(sv.pointing_group_id, []).append(sv)
    active: list[Survey] = []
    merges: list[PointingMerge] = []
    held_out: list[dict] = []
    for gid, members in groups.items():
        parents = [s for s in members if not s.is_pointing_extension]
        extensions = [s for s in members if s.is_pointing_extension]
        related = len(parents) == 1 and all(
            e.parent_pointing_survey_id == parents[0].id and e.channel == parents[0].channel
            for e in extensions
        )
        if (len(members) > 1 and not related) or not parents:
            a, b = (members + members)[:2]
            raise DoubleCountingError(
                f"Surveys {a.id!r} and {b.id!r} share pointing_group_id {gid!r} without a "
                "declared compatible parent-pointing relationship "
                f"({REJECT_DUPLICATE_POINTING_GROUP}); a survey observed inside another "
                "survey's pointings must declare parent_pointing_survey_id and "
                f"pointing_merge_policy: {UNION_TARGETS_BY_POINTING} (keep_first is forbidden)"
                if len(members) > 1
                else f"Extension survey {a.id!r} was selected without its parent pointing "
                f"{a.parent_pointing_survey_id!r}"
            )
        parent = parents[0]
        active.append(parent)
        if not extensions:
            continue
        if per_group.get(gid, default) == UNION_TARGETS_BY_POINTING:
            merges.append(PointingMerge(gid, parent, tuple(extensions)))
        else:
            held_out += [
                {
                    "survey_id": e.id,
                    "pointing_group_id": gid,
                    "parent_pointing_survey_id": parent.id,
                    "reason": (
                        f"run policy {REJECT_DUPLICATE_POINTING_GROUP}: a declared extension "
                        "of a parent pointing only enters through "
                        f"{UNION_TARGETS_BY_POINTING}; its stars are NOT in this run"
                    ),
                }
                for e in extensions
            ]
    return PointingResolution(label, tuple(active), tuple(merges), tuple(held_out))


def check_pointing_groups(
    surveys: Sequence[Survey],
    policy: str | Mapping[str, str] = REJECT_DUPLICATE_POINTING_GROUP,
) -> list[Survey]:
    """Surveys to condition on directly (see :func:`resolve_pointing_groups`)."""
    return list(resolve_pointing_groups(surveys, policy).active)


# ------------------------------------------------ union_targets_by_pointing kernel
@dataclass(frozen=True)
class SightingTable:
    """Star sightings of one survey inside a shared pointing group.

    One row per (star, beam) sighting. ``role`` is ``primary`` for a published on-axis
    target list (rows are distinct targets, ``theta`` is exactly 0) or ``background`` for
    stars found inside the beams. ``ra_deg`` / ``dec_deg`` must be given at one common
    epoch across all tables of a merge (they drive the angular fallback only).
    ``gaia_dr3_source_id`` is int64 with ``MISSING_GAIA_ID`` where unknown.
    """

    survey_id: str
    role: Literal["primary", "background"]
    sha256: str
    names: np.ndarray
    gaia_dr3_source_id: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    distance: u.Quantity
    pointing_index: np.ndarray
    theta: u.Quantity

    def __post_init__(self) -> None:
        if not isinstance(self.distance, u.Quantity) or not self.distance.unit.is_equivalent(u.pc):
            raise UnitValidationError("SightingTable.distance must be a length Quantity")
        if not isinstance(self.theta, u.Quantity) or not self.theta.unit.is_equivalent(u.arcmin):
            raise UnitValidationError("SightingTable.theta must be an angle Quantity")
        if self.role not in ("primary", "background"):
            raise SchemaValidationError(f"{self.survey_id}: unknown sighting role {self.role!r}")
        n = len(self.names)
        ids = np.asarray(self.gaia_dr3_source_id)
        if ids.dtype.kind not in "iu":
            raise SchemaValidationError(
                f"{self.survey_id}: gaia_dr3_source_id must be integer (float64 corrupts 64-bit ids)"
            )
        cols = (ids, self.ra_deg, self.dec_deg, self.distance, self.pointing_index, self.theta)
        if any(np.size(c) != n for c in cols):
            raise SchemaValidationError(f"{self.survey_id}: sighting columns differ in length")
        d, th = self.distance.to_value(u.pc), self.theta.to_value(u.arcmin)
        if n and not (np.all(np.isfinite(d) & (d > 0)) and np.all(np.isfinite(th) & (th >= 0))):
            raise SchemaValidationError(f"{self.survey_id}: need distance > 0 and theta >= 0")
        if n and not np.all(np.isfinite(self.ra_deg) & np.isfinite(self.dec_deg)):
            raise SchemaValidationError(f"{self.survey_id}: non-finite coordinates")
        if self.role == "primary" and n and np.any(th != 0.0):
            raise SchemaValidationError(
                f"{self.survey_id}: primary targets are on-axis (theta = 0)"
            )

    def __len__(self) -> int:
        return len(self.names)


def sightings_sha256(
    names: Sequence[str], gaia_ids: np.ndarray, pointing_index: np.ndarray, theta_arcmin: np.ndarray
) -> str:
    """Canonical content hash of an in-memory sighting list (no file behind it)."""
    return _canonical_sha256(
        {
            "names": [str(x) for x in names],
            "gaia_dr3_source_id": [int(x) for x in np.asarray(gaia_ids)],
            "pointing_index": [int(x) for x in np.asarray(pointing_index)],
            "theta_arcmin": [float(x) for x in np.asarray(theta_arcmin)],
        }
    )


@dataclass(frozen=True)
class MergedTargets:
    """Result of :func:`union_targets_by_pointing` over the concatenated input rows.

    Raw arrays have one entry per input row (tables concatenated in the given order).
    ``cluster[r]`` is the merged-star index of raw row ``r``; ``winner[k]`` is the raw row
    kept for merged star ``k`` (minimum offset). Merged stars are ordered by their winner
    row, so primaries keep catalog order and precede background stars.
    """

    surveys: tuple[str, ...]
    roles: tuple[str, ...]
    survey_index: np.ndarray
    names: np.ndarray
    gaia_dr3_source_id: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    distance: u.Quantity
    pointing_index: np.ndarray
    theta: u.Quantity
    cluster: np.ndarray
    winner: np.ndarray
    provenance: MergeProvenance

    def __len__(self) -> int:
        return len(self.winner)

    @property
    def row_role(self) -> np.ndarray:
        return np.asarray(self.roles, dtype=object)[self.survey_index]

    @property
    def winner_role(self) -> np.ndarray:
        return self.row_role[self.winner]

    @property
    def n_sightings(self) -> np.ndarray:
        return np.bincount(self.cluster, minlength=len(self.winner))

    def source_surveys(self, k: int) -> tuple[str, ...]:
        """Merged source provenance of star ``k``: every survey that sighted it."""
        idx = np.unique(self.survey_index[self.cluster == k])
        return tuple(self.surveys[i] for i in idx)

    def absorbed_rows(self) -> np.ndarray:
        """Raw rows that are not the kept sighting of their star."""
        keep = np.zeros(len(self.cluster), bool)
        keep[self.winner] = True
        return np.flatnonzero(~keep)

    def duplicate_gaia_ids(self) -> np.ndarray:
        """Independent recount: Gaia ids carried by more than one merged star (must be empty)."""
        ids = self.gaia_dr3_source_id[self.winner]
        ids = ids[ids != MISSING_GAIA_ID]
        uniq, counts = np.unique(ids, return_counts=True)
        return uniq[counts > 1]


def union_targets_by_pointing(
    pointing_group_id: str,
    tables: Sequence[SightingTable],
    *,
    angular_fallback: u.Quantity = ANGULAR_FALLBACK,
) -> MergedTargets:
    """Merge the star lists of surveys sharing one pointing group (pure, deterministic).

    Each physical star enters the result exactly once:

    1. **Gaia DR3 ``source_id``** (exact integer match) is the primary key: all sightings
       with the same id are one star, within and across surveys.
    2. **Angular fallback** (separation ``< angular_fallback``, default 3 arcsec) applies
       only to pairs in which at least one record has no Gaia id. Links are taken in
       order of increasing separation; a link is refused if it would join two different
       Gaia ids or two rows of the same primary target list (published distinct targets
       such as GJ 702A / GJ 702B are never merged with each other). Links are transitive
       among id-less records (A-B and B-C within the radius make one star).
    3. **Collision rule**: of all sightings of one star (several overlapping beams, or a
       primary plus background sightings) the one with the minimum radial offset theta is
       kept; ties go to the primary observation, then to input order.

    The result does not depend on anything but the input rows and their order. Raises
    ``DoubleCountingError`` if a primary list repeats a Gaia id or if any Gaia id survives
    twice (it cannot, by construction; the recount is an independent guard).
    """
    if not isinstance(angular_fallback, u.Quantity) or not angular_fallback.unit.is_equivalent(
        u.arcsec
    ):
        raise UnitValidationError("angular_fallback must be an angle Quantity")
    tol_arcsec = float(angular_fallback.to_value(u.arcsec))
    if not tol_arcsec > 0:
        raise ValueError("angular_fallback must be positive")
    if len(tables) < 2:
        raise SchemaValidationError("union_targets_by_pointing merges at least two surveys")
    surveys = tuple(t.survey_id for t in tables)
    if len(set(surveys)) != len(surveys):
        raise DoubleCountingError(f"the same survey was passed twice into a merge: {surveys}")
    for t in tables:
        ids = np.asarray(t.gaia_dr3_source_id)
        ids = ids[ids != MISSING_GAIA_ID]
        if t.role == "primary" and len(np.unique(ids)) != len(ids):
            raise DoubleCountingError(
                f"{t.survey_id}: a primary target list repeats a gaia_dr3_source_id"
            )

    def cat(get, dtype=None) -> np.ndarray:
        return np.concatenate([np.asarray(get(t), dtype=dtype) for t in tables])

    survey_index = cat(lambda t: np.full(len(t), surveys.index(t.survey_id)), np.int64)
    gaia = cat(lambda t: t.gaia_dr3_source_id, np.int64)
    ra, dec = cat(lambda t: t.ra_deg, float), cat(lambda t: t.dec_deg, float)
    dist = cat(lambda t: t.distance.to_value(u.pc), float)
    theta = cat(lambda t: t.theta.to_value(u.arcmin), float)
    pointing = cat(lambda t: t.pointing_index, np.int64)
    names = cat(lambda t: t.names, object)
    is_primary = np.asarray([t.role == "primary" for t in tables])[survey_index]
    n = len(gaia)
    rows = np.arange(n)

    # Stage 1: exact Gaia id. label[r] = first row carrying the same id.
    label = rows.copy()
    has_id = gaia != MISSING_GAIA_ID
    if has_id.any():
        _, first, inverse = np.unique(gaia[has_id], return_index=True, return_inverse=True)
        label[has_id] = rows[has_id][first][inverse.reshape(-1)]
    n_dedup_gaia = int(n - len(np.unique(label)))

    # Stage 2: angular fallback, only for pairs with a record that lacks a Gaia id.
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    n_dedup_angular = n_refused = 0
    no_id = np.flatnonzero(~has_id)
    if no_id.size and n > 1:
        sky = SkyCoord(ra * u.deg, dec * u.deg)
        i1, i2, sep, _ = sky.search_around_sky(sky[no_id], angular_fallback)
        a, b = no_id[i1], np.asarray(i2)
        sep_as = sep.to_value(u.arcsec)
        ok = (a != b) & (sep_as < tol_arcsec)
        lo, hi = np.minimum(a, b)[ok], np.maximum(a, b)[ok]
        pairs = sorted(set(zip(sep_as[ok].tolist(), lo.tolist(), hi.tolist(), strict=True)))
        # Per-cluster state: the single Gaia id (or missing) and the primary tables present.
        gid: dict[int, int] = {}
        prim: dict[int, frozenset[int]] = {}

        def state(root: int) -> tuple[int, frozenset[int]]:
            if root not in gid:
                members = np.flatnonzero(label == root)
                ids = gaia[members]
                ids = ids[ids != MISSING_GAIA_ID]
                gid[root] = int(ids[0]) if ids.size else MISSING_GAIA_ID
                prim[root] = frozenset(int(s) for s in survey_index[members][is_primary[members]])
            return gid[root], prim[root]

        seen: set[tuple[int, int]] = set()
        for _sep, x, y in pairs:
            rx, ry = find(int(label[x])), find(int(label[y]))
            if rx == ry or (rx, ry) in seen:
                continue
            (gx, px), (gy, py) = state(rx), state(ry)
            if (gx != MISSING_GAIA_ID and gy != MISSING_GAIA_ID and gx != gy) or (px & py):
                seen.add((rx, ry))
                n_refused += 1
                continue
            keep, drop = min(rx, ry), max(rx, ry)
            parent[drop] = keep
            gid[keep] = gx if gx != MISSING_GAIA_ID else gy
            prim[keep] = px | py
            n_dedup_angular += 1
    root_of = np.asarray([find(int(x)) for x in np.unique(label)])
    lookup = dict(zip(np.unique(label).tolist(), root_of.tolist(), strict=True))
    roots = np.asarray([lookup[int(x)] for x in label]) if parent else label

    # Collision rule: minimum theta, then primary, then input order.
    order = np.lexsort((rows, ~is_primary, theta))
    _, first = np.unique(roots[order], return_index=True)
    winner = np.sort(order[first])
    cluster_of_root = {int(roots[w]): k for k, w in enumerate(winner)}
    cluster = np.asarray([cluster_of_root[int(r)] for r in roots], np.int64)
    n_merged = len(winner)

    spans = np.zeros(n_merged, np.int64)
    if n:
        pairs_cp = np.unique(np.column_stack([cluster, pointing]), axis=0)
        spans = np.bincount(pairs_cp[:, 0], minlength=n_merged)
    has_primary = np.zeros(n_merged, bool)
    has_primary[cluster[is_primary]] = True
    n_absorbed = int(np.sum(~is_primary & has_primary[cluster]))
    win_ids = gaia[winner]
    win_ids = win_ids[win_ids != MISSING_GAIA_ID]
    n_dup = int(len(win_ids) - len(np.unique(win_ids)))
    if n_dup:
        raise DoubleCountingError(
            f"{n_dup} gaia_dr3_source_id value(s) survive more than once after "
            f"{UNION_TARGETS_BY_POINTING} of {surveys}"
        )
    win_primary = is_primary[winner]
    provenance = MergeProvenance(
        pointing_group_id=pointing_group_id,
        policy=UNION_TARGETS_BY_POINTING,
        input_surveys=list(surveys),
        input_sha256s=[t.sha256 for t in tables],
        n_raw_targets=n,
        n_merged_targets=n_merged,
        n_dedup_gaia_id=n_dedup_gaia,
        n_dedup_angular_fallback=n_dedup_angular,
        n_multi_beam_resolved=int(np.sum(spans > 1)),
        angular_fallback_arcsec=tol_arcsec,
        n_raw_by_survey={t.survey_id: len(t) for t in tables},
        n_merged_by_role={
            "primary": int(win_primary.sum()),
            "background": int((~win_primary).sum()),
        },
        n_background_absorbed_by_primary=n_absorbed,
        n_angular_links_refused=n_refused,
        n_duplicate_gaia_ids_post_merge=0,
        collision_rule=COLLISION_RULE,
    )
    return MergedTargets(
        surveys=surveys,
        roles=tuple(t.role for t in tables),
        survey_index=survey_index,
        names=names,
        gaia_dr3_source_id=gaia,
        ra_deg=ra,
        dec_deg=dec,
        distance=dist * u.pc,
        pointing_index=pointing,
        theta=theta * u.arcmin,
        cluster=cluster,
        winner=winner,
        provenance=provenance,
    )


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


def _canonical_sha256(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=True).encode()).hexdigest()


# ------------------------------------------- background-field summary (v0.2.1)
def primary_sightings(survey: Survey, root: Path) -> SightingTable:
    """The checksum-verified primary target list as on-axis sightings (one beam per target).

    Gaia DR3 ids are read as text and converted to exact integers: parsing the column as
    float64 would corrupt 64-bit ids above 2^53.
    """
    path = verify_catalog_checksum(survey, root)
    df = pd.read_csv(path, dtype={"gaia_dr3_source_id": str})
    missing = [c for c in CATALOG_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"{path}: missing columns {missing}")
    raw = df["gaia_dr3_source_id"] if "gaia_dr3_source_id" in df.columns else [None] * len(df)
    ids = np.array(
        [int(x) if isinstance(x, str) and x.strip() else MISSING_GAIA_ID for x in raw], np.int64
    )
    return SightingTable(
        survey_id=survey.id,
        role="primary",
        sha256=survey.catalog_sha256 or "",
        names=df["target_name"].to_numpy(dtype=object),
        gaia_dr3_source_id=ids,
        ra_deg=df["ra_deg"].to_numpy(dtype=float),
        dec_deg=df["dec_deg"].to_numpy(dtype=float),
        distance=pd.to_numeric(df["dist_pc"], errors="coerce").to_numpy(dtype=float) * u.pc,
        pointing_index=np.arange(len(df), dtype=np.int64),
        theta=np.zeros(len(df)) * u.arcmin,
    )


# ------------------------------------ pointing-correction overlay (ruling v0.2.1-R5)
CATALOG_COORD_TOLERANCE_DEG = 1e-9  # overlay catalog_* values are verbatim catalog values


@dataclass(frozen=True)
class PointingOverlay:
    """SHA-256-verified pointing corrections for one locked primary catalog.

    Non-destructive: :meth:`apply` returns a copy of the primary sightings in which the
    corrected targets carry their J2000 truth position. The catalog file is never written,
    and the returned table keeps the catalog's SHA-256 (the overlay has its own).
    """

    path: str
    sha256: str
    record: PointingCorrectionsFile

    @property
    def target_names(self) -> list[str]:
        return [c.target_name for c in self.record.pointing_corrections]

    def _rows(self, primary: SightingTable) -> list[tuple[int, object]]:
        """(pointing index, correction) pairs, after checking the overlay fits ``primary``."""
        rec = self.record
        if primary.role != "primary":
            raise SchemaValidationError("pointing corrections apply to a primary target list")
        if primary.sha256 != rec.locked_catalog.sha256:
            raise ProvenanceError(
                f"{self.path} corrects catalog {rec.locked_catalog.sha256[:12]}..., not the "
                f"primary list {primary.sha256[:12]}... it was applied to"
            )
        index = {str(n): i for i, n in enumerate(primary.names)}
        rows = []
        for c in rec.pointing_corrections:
            i = index.get(c.target_name)
            if i is None:
                raise ProvenanceError(f"{self.path}: {c.target_name!r} is not a catalog target")
            if (
                abs(float(primary.ra_deg[i]) - c.catalog_ra_deg) > CATALOG_COORD_TOLERANCE_DEG
                or abs(float(primary.dec_deg[i]) - c.catalog_dec_deg) > CATALOG_COORD_TOLERANCE_DEG
            ):
                raise ProvenanceError(
                    f"{self.path}: {c.target_name} catalog coordinates "
                    f"({primary.ra_deg[i]}, {primary.dec_deg[i]}) differ from the recorded "
                    f"({c.catalog_ra_deg}, {c.catalog_dec_deg}); was the overlay already applied?"
                )
            want = MISSING_GAIA_ID if c.gaia_dr3_source_id is None else int(c.gaia_dr3_source_id)
            if int(primary.gaia_dr3_source_id[i]) != want:
                raise ProvenanceError(
                    f"{self.path}: {c.target_name} gaia_dr3_source_id disagrees with the catalog"
                )
            rows.append((i, c))
        return rows

    def apply(self, primary: SightingTable) -> SightingTable:
        """Primary sightings with the corrected J2000 positions (every other row unchanged)."""
        ra, dec = np.array(primary.ra_deg, float), np.array(primary.dec_deg, float)
        for i, c in self._rows(primary):
            ra[i], dec[i] = c.true_ra_deg, c.true_dec_deg
        return replace(primary, ra_deg=ra, dec_deg=dec)

    def corrected_pointings(
        self, primary: SightingTable, cone_receipts: Mapping[int, Mapping] | None = None
    ) -> list[dict]:
        """``CorrectedPointing`` rows of the artifact, in catalog (pointing-index) order.

        ``primary`` is the UNCORRECTED catalog table. ``cone_receipts`` maps a pointing index
        to its cone cache manifest entry (live runs); rehearsals have none.
        """
        out = []
        for i, c in sorted(self._rows(primary), key=lambda r: r[0]):
            row = {
                "target_name": c.target_name,
                "pointing_index": int(i),
                "catalog_ra_deg": c.catalog_ra_deg,
                "catalog_dec_deg": c.catalog_dec_deg,
                "cone_ra_deg": c.true_ra_deg,
                "cone_dec_deg": c.true_dec_deg,
                "truth_source": c.truth_source,
            }
            entry = (cone_receipts or {}).get(int(i))
            if entry is not None:
                row.update({k: entry.get(k) for k in ("adql_sha256", "fetched_at_utc")})
                row["code_git_sha"] = entry.get("code_git_sha")
            out.append(row)
        return out

    def summary_meta(
        self, primary: SightingTable, cone_receipts: Mapping[int, Mapping] | None = None
    ) -> dict:
        """The ``BackgroundSummaryFile`` keys that record this overlay."""
        rows = self.corrected_pointings(primary, cone_receipts)
        return {
            "pointing_corrections_path": self.path,
            "pointing_corrections_sha256": self.sha256,
            "n_corrected_pointings": len(rows),
            "corrected_target_names": [r["target_name"] for r in rows],
            "corrected_pointings": rows,
        }


def load_pointing_overlay(
    root: Path, rel_path: str, expected_sha256: str | None = None
) -> PointingOverlay:
    """Load a pointing-correction overlay; a pinned SHA-256 that does not match is an error."""
    path = root / rel_path
    if not path.is_file():
        raise ProvenanceError(f"pointing corrections {rel_path} missing")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ProvenanceError(f"{rel_path}: sha256 {sha} != pinned {expected_sha256}")
    doc = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(doc, dict):
        raise SchemaValidationError(f"{rel_path}: expected a YAML mapping")
    return PointingOverlay(rel_path, sha, PointingCorrectionsFile.model_validate(doc))


def survey_pointing_overlay(survey: Survey, root: Path) -> PointingOverlay | None:
    """The overlay an extension survey pins (verified against the pin), or None."""
    if survey.pointing_corrections_path is None:
        return None
    return load_pointing_overlay(
        root, survey.pointing_corrections_path, survey.pointing_corrections_sha256
    )


def pointing_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Galactic Cartesian unit vectors of ICRS directions (same frame as ``TargetTable.xyz``)."""
    cart = SkyCoord(
        ra=np.asarray(ra_deg, float) * u.deg, dec=np.asarray(dec_deg, float) * u.deg, frame="icrs"
    ).galactic.cartesian
    return np.column_stack([cart.x.value, cart.y.value, cart.z.value]).astype(float)


def survey_f_min(survey: Survey) -> u.Quantity:
    """Narrowband minimum detectable flux of a radio survey's declared radiometer."""
    rs = survey.radio_sensitivity
    if rs is None:
        raise SchemaValidationError(f"{survey.id} has no radio_sensitivity")
    return radio.min_detectable_flux(
        radio.Radiometer(
            rs.sefd, rs.snr_threshold, rs.channel_bandwidth, rs.integration_time, rs.n_pol
        )
    )


def build_background_summary(
    merged: MergedTargets,
    *,
    primary: SightingTable,
    sky_ra_deg: np.ndarray,
    sky_dec_deg: np.ndarray,
    distance_method: np.ndarray,
    f_min: u.Quantity,
    beam_fwhm: u.Quantity,
    cuts: Mapping[str, float],
    meta: Mapping[str, object],
    rng_seed: int,
    n_validation: int = 1000,
    binning: radio.ShellBinning = radio.DEFAULT_SHELL_BINNING,
    suspected_unmatched_primary: Sequence[dict] = (),
) -> dict:
    """Assemble the compact background-field artifact from a star-level merge (pure).

    ``merged`` must be ``union_targets_by_pointing([primary, background])``. Per-raw-row
    inputs (same order as the merge's concatenated rows): ``sky_ra_deg`` / ``sky_dec_deg``
    are the source-catalog-epoch coordinates that define each sighting's ``theta``;
    ``distance_method`` names how its distance was obtained. Primary-matched background
    sightings are removed by the merge BEFORE shell binning, so no primary star is counted
    in a shell. The validation sample is drawn with ``SeedSequence([rng_seed, 0x42474B])``.
    The returned dict is sealed (``content_sha256``) but not schema-validated: callers
    validate with ``BackgroundSummaryFile`` once provenance is final (a live run's receipt
    may legitimately be dirty or incomplete, and is then simply not promotable).
    """
    if len(merged.surveys) != 2 or merged.roles != ("primary", "background"):
        raise SchemaValidationError("background summary needs a [primary, background] merge")
    if merged.surveys[0] != primary.survey_id:
        raise ProvenanceError("merge was not computed against the given primary catalog")
    n_raw = len(merged.cluster)
    if any(np.size(c) != n_raw for c in (sky_ra_deg, sky_dec_deg, distance_method)):
        raise SchemaValidationError("per-row inputs must match the merge's raw rows")
    win = merged.winner
    bg_rows = win[merged.winner_role == "background"]
    theta = merged.theta.to_value(u.arcmin)
    dist = merged.distance.to_value(u.pc)
    gaia = merged.gaia_dr3_source_id
    if np.any(gaia[bg_rows] == MISSING_GAIA_ID):
        raise ProvenanceError("background stars must carry a canonical gaia_dr3_source_id")
    binned = radio.bin_background_stars(
        merged.pointing_index[bg_rows],
        dist[bg_rows] * u.pc,
        theta[bg_rows] * u.arcmin,
        f_min,
        binning,
        beam_fwhm,
    )
    cells = {
        k: binned.cells[:, i].tolist()
        for i, k in enumerate(("pointing_index", "distance_bin", "eirp_min_bin", "count"))
    }
    eirp = binned.eirp_min.to_value(u.W)
    method = np.asarray(distance_method, dtype=object)

    n_req = int(n_validation)
    rng = np.random.default_rng(np.random.SeedSequence([int(rng_seed), 0x42474B]))
    pick = np.sort(rng.choice(len(bg_rows), size=min(n_req, len(bg_rows)), replace=False))
    sample = [
        {
            "gaia_dr3_source_id": str(int(gaia[r])),  # v0.2.1-R4: string, never a number
            "ra_deg": float(sky_ra_deg[r]),
            "dec_deg": float(sky_dec_deg[r]),
            "ra_match_deg": float(merged.ra_deg[r]),
            "dec_match_deg": float(merged.dec_deg[r]),
            "pointing_index": int(merged.pointing_index[r]),
            "theta_arcmin": float(theta[r]),
            "distance_pc": float(dist[r]),
            "distance_method": str(method[r]),
            "eirp_min_w": float(eirp[i]),
            "distance_bin": int(binned.distance_bin[i]),
            "eirp_min_bin": int(binned.eirp_min_bin[i]),
        }
        for i, r in ((int(i), int(bg_rows[i])) for i in pick)
    ]

    role = merged.row_role
    absorbed = [
        int(r)
        for r in merged.absorbed_rows()
        if role[r] == "background" and role[win[merged.cluster[r]]] == "primary"
    ]
    matched = []
    if absorbed:
        a = np.asarray(absorbed)
        w = win[merged.cluster[a]]
        sep = (
            SkyCoord(merged.ra_deg[a] * u.deg, merged.dec_deg[a] * u.deg)
            .separation(SkyCoord(merged.ra_deg[w] * u.deg, merged.dec_deg[w] * u.deg))
            .to_value(u.arcsec)
        )
        for r, wr, s in zip(a.tolist(), w.tolist(), sep.tolist(), strict=True):
            by_id = gaia[r] != MISSING_GAIA_ID and gaia[r] == gaia[wr]
            matched.append(
                {
                    "gaia_dr3_source_id": str(int(gaia[r])),
                    "primary_target_name": str(merged.names[wr]),
                    "match": "gaia_dr3_source_id" if by_id else "angular_fallback",
                    "pointing_index": int(merged.pointing_index[r]),
                    "theta_arcmin": float(theta[r]),
                    "separation_arcsec": float(s),
                }
            )
    prov = merged.provenance
    by_method = {
        m: int(np.sum(method[bg_rows] == m))
        for m in ("gaia_dr3_inverse_parallax", "bailer_jones_2021_geo")
    }
    doc: dict = {
        "schema_version": 1,
        **dict(meta),
        "n_pointings": len(primary),
        "beam_fwhm_arcmin": float(beam_fwhm.to_value(u.arcmin)),
        "cone_radius_arcmin": float(beam_fwhm.to_value(u.arcmin)) / 2.0,
        "f_min_w_m2": float(f_min.to_value(u.W / u.m**2)),
        "primary_catalog_sha256": primary.sha256,
        "parent_survey_id": primary.survey_id,
        "pointing_group_id": prov.pointing_group_id,
        "distance_cuts": {
            **dict(cuts),
            "n_background_stars": len(bg_rows),
            "n_background_inverse_parallax": by_method["gaia_dr3_inverse_parallax"],
            "n_background_bailer_jones_geo": by_method["bailer_jones_2021_geo"],
        },
        "merge_provenance": prov.model_dump(),
        "dedup_audit": {
            "n_primary_targets": len(primary),
            "n_primary_with_gaia_id": int(np.sum(primary.gaia_dr3_source_id != MISSING_GAIA_ID)),
            "n_background_sightings_matched_to_primary_gaia_id": sum(
                m["match"] == "gaia_dr3_source_id" for m in matched
            ),
            "n_background_sightings_matched_to_primary_angular": sum(
                m["match"] == "angular_fallback" for m in matched
            ),
            "n_duplicate_gaia_ids_vs_primary": len(
                np.intersect1d(
                    gaia[bg_rows],
                    primary.gaia_dr3_source_id[primary.gaia_dr3_source_id != MISSING_GAIA_ID],
                )
            ),
            "n_duplicate_gaia_ids_within_background": len(bg_rows) - len(np.unique(gaia[bg_rows])),
            "angular_fallback_arcsec": prov.angular_fallback_arcsec,
            "angular_match_epoch": str(meta.get("angular_match_epoch", "J2000.0")),
            "n_suspected_unmatched_primary": len(suspected_unmatched_primary),
            "suspected_unmatched_primary": list(suspected_unmatched_primary),
        },
        "shells": {
            "binning": binning.as_dict(),
            "pointing_names": [str(x) for x in primary.names],
            **cells,
            "n_cells": len(cells["count"]),
            "n_stars": len(bg_rows),
            "cells_sha256": _canonical_sha256(cells),
        },
        "validation_sample": {
            "rng_seed": int(rng_seed),
            "n_background_requested": n_req,
            "selection": (
                "numpy Generator(SeedSequence([rng_seed, 0x42474B])).choice without "
                "replacement over the de-duplicated background stars in merge order, plus "
                "every background sighting absorbed by a primary target"
            ),
            "background": sample,
            "primary_matched": matched,
        },
        "spatial_correlation_assumption": radio.SHELL_CORRELATION_ASSUMPTION,
    }
    doc.pop("angular_match_epoch", None)
    return seal_background_summary(doc)


def seal_background_summary(doc: Mapping) -> dict:
    """Return ``doc`` with ``content_sha256`` (re)computed over everything else."""
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    return {**body, "content_sha256": _canonical_sha256(body)}


def background_summary_bytes(doc: Mapping) -> bytes:
    """Canonical on-disk serialisation (compact separators; the 5 MB budget is for these bytes)."""
    return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_background_summary(doc: Mapping, path: Path) -> tuple[str, int]:
    """Write a summary artifact; refuses a file at or above the 5 MB compactness budget."""
    raw = background_summary_bytes(doc)
    if len(raw) >= BACKGROUND_SUMMARY_MAX_BYTES:
        raise ProvenanceError(
            f"background summary is {len(raw)} bytes (>= {BACKGROUND_SUMMARY_MAX_BYTES}); "
            "coarsen the shell bins or shrink the validation sample"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), len(raw)


@dataclass(frozen=True)
class BackgroundField:
    """Verified background-field summary of one extension survey, ready for the engine."""

    survey_id: str
    parent_survey_id: str
    path: str
    file_sha256: str
    record: BackgroundSummaryFile
    shells: radio.BackgroundShells
    detail: str
    # Ruling v0.2.1-R5: the overlay the survey pins (None when it pins none). Whether the
    # ARTIFACT was built with it is `record.pointing_corrections_sha256`.
    pointing_overlay: PointingOverlay | None = None

    @property
    def merge_provenance(self) -> MergeProvenance:
        return self.record.merge_provenance

    def pointing_corrections(self) -> dict:
        """Overlay state: what the survey pins against what this artifact was built with.

        ``uncorrected`` pointings are overlay entries whose cone in this artifact is still
        centred on the locked catalog's (wrong) coordinates.
        """
        r, ov = self.record, self.pointing_overlay
        pinned = ov.target_names if ov else []
        applied = set(r.corrected_target_names)
        uncorrected = [n for n in pinned if n not in applied]
        return {
            "pointing_corrections_path": ov.path if ov else None,
            "pointing_corrections_sha256": ov.sha256 if ov else None,
            "claim_id": ov.record.id if ov else None,
            "ruling": ov.record.ruling if ov else None,
            "locked_catalog_path": r.primary_catalog_path,
            "locked_catalog_sha256": r.primary_catalog_sha256,
            "locked_catalog_modified": False,
            "n_pinned_corrections": len(pinned),
            "artifact_built_with_overlay": r.pointing_corrections_sha256 is not None,
            "n_corrected_pointings": r.n_corrected_pointings,
            "corrected_target_names": list(r.corrected_target_names),
            "corrected_pointings": [c.model_dump() for c in r.corrected_pointings],
            "n_uncorrected_pointings": len(uncorrected),
            "uncorrected_target_names": uncorrected,
        }

    def summary(self) -> dict:
        r = self.record
        return {
            "id": r.id,
            "survey_id": self.survey_id,
            "parent_survey_id": self.parent_survey_id,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "content_sha256": r.content_sha256,
            "data_tier": r.data_tier,
            "code_git_sha": r.code_git_sha,
            "fetched_at_utc": r.fetched_at_utc,
            "query_window_utc": list(r.query_window_utc),
            "assembled_at_utc": r.assembled_at_utc,
            "cone_cache_manifest_sha256": r.cone_cache_manifest_sha256,
            "cone_cache_code_git_shas": list(r.cone_cache_code_git_shas),
            "source_service": r.source_service,
            "n_pointings": r.n_pointings,
            "n_background_stars": r.shells.n_stars,
            "n_cells": r.shells.n_cells,
            "beam_fwhm_arcmin": r.beam_fwhm_arcmin,
            "cone_radius_arcmin": r.cone_radius_arcmin,
            "distance_cuts": r.distance_cuts.model_dump(),
            "merge_provenance": r.merge_provenance.model_dump(),
            "dedup_audit": self.dedup_counts(),
            "pointing_corrections": self.pointing_corrections(),
            "scope": self.scope(),
            "methodology_note": r.methodology_note,
        }

    def dedup_counts(self) -> dict:
        """De-duplication counters of the artifact (ruling v0.2.1-R3: reported, never hidden).

        ``n_suspected_unmatched_primary`` counts review ROWS; one star beside two primaries
        (e.g. a visual binary published as two targets) is listed twice, so the number of
        distinct stars still sitting in the shells is reported separately.
        """
        da = self.record.dedup_audit
        rows = da.suspected_unmatched_primary
        return {
            **{k: v for k, v in da.model_dump().items() if k != "suspected_unmatched_primary"},
            "n_primary_without_gaia_id": da.n_primary_targets - da.n_primary_with_gaia_id,
            "n_suspected_unmatched_primary_stars": len({r.gaia_dr3_source_id for r in rows}),
            "n_primaries_with_a_suspected_counterpart": len({r.primary_target_name for r in rows}),
            # A primary that already absorbed a Gaia sighting and still has a suspected star
            # beside it has a SECOND nearby Gaia source (typically a resolved companion);
            # one that absorbed nothing is the likelier case of a primary counted twice.
            "n_suspected_primaries_that_absorbed_a_sighting": len(
                {r.primary_target_name for r in rows} & self.absorbing_primaries()
            ),
            "n_suspected_primaries_that_absorbed_nothing": len(
                {r.primary_target_name for r in rows} - self.absorbing_primaries()
            ),
            "suspected_unmatched_primary_policy": SUSPECTED_PRIMARY_POLICY,
        }

    def absorbing_primaries(self) -> set[str]:
        """Primaries that absorbed at least one background sighting (Gaia id or 3 arcsec)."""
        return {m.primary_target_name for m in self.record.validation_sample.primary_matched}

    def scope(self) -> dict:
        """What was retrieved here, beside the published catalogue whose method it follows."""
        r = self.record
        return {
            "retrieved": {
                "survey_id": self.survey_id,
                "data_tier": r.data_tier,
                "pointings": (
                    f"{r.n_pointings} Enriquez et al. (2017) GBT L-band pointings "
                    f"({r.primary_catalog_path})"
                ),
                "n_pointings": r.n_pointings,
                "n_background_stars": r.shells.n_stars,
                "gaia_release": "Gaia DR3",
                "distances": (
                    f"inverse parallax for parallax_over_error >= "
                    f"{r.distance_cuts.inverse_parallax_min_snr:g} "
                    f"({r.distance_cuts.n_background_inverse_parallax} stars); Bailer-Jones et al. "
                    f"(2021) geometric distances above "
                    f"{r.distance_cuts.bailer_jones_min_snr_exclusive:g} "
                    f"({r.distance_cuts.n_background_bailer_jones_geo} stars); d <= "
                    f"{r.distance_cuts.d_max_pc:g} pc"
                ),
                "selection": (
                    f"inside the {r.beam_fwhm_arcmin:g} arcmin FWHM beam (cone radius "
                    f"{r.cone_radius_arcmin:g} arcmin = FWHM / 2)"
                ),
                "source_service": r.source_service,
            },
            "published": dict(PUBLISHED_BACKGROUND_SCOPE),
            "comparable": False,
            "statement": (
                f"N_bg = {r.shells.n_stars:,} background stars retrieved over {r.n_pointings} "
                "Enriquez et al. (2017) GBT L-band pointings with Gaia DR3. Wlodarczyk-Sroka et al. "
                f"(2020) report {PUBLISHED_BACKGROUND_SCOPE['n_stars_total']:,} stars over "
                f"{PUBLISHED_BACKGROUND_SCOPE['n_pointings']:,} Price et al. (2020) multi-band "
                "pointings with Gaia DR2: a different pointing set, receiver set and Gaia release, "
                "so the two counts are not expected to agree. This artifact reproduces the method, "
                "not the catalogue."
            ),
        }

    def suspected_star_cells(self) -> dict[str, tuple[int, int, int]]:
        """Shell cell ``(pointing, distance_bin, eirp_min_bin)`` of each suspected star."""
        r = self.record
        f_q = r.f_min_w_m2 * u.W / u.m**2
        fwhm = r.beam_fwhm_arcmin * u.arcmin
        b = self.shells.binning
        cells = {}
        for sp in r.dedup_audit.suspected_unmatched_primary:
            eirp = radio.min_detectable_eirp_off_axis(
                sp.distance_pc * u.pc, sp.theta_arcmin * u.arcmin, f_q, fwhm
            )
            cells[sp.gaia_dr3_source_id] = (
                sp.pointing_index,
                int(b.distance_bin(sp.distance_pc * u.pc)[0]),
                int(b.eirp_bin(eirp)[0]),
            )
        return cells

    def shells_excluding_suspected(self) -> radio.BackgroundShells:
        """DIAGNOSTIC view: the shells without the suspected unmatched primaries.

        Never used for a likelihood that enters a headline: ruling v0.2.1-R3 keeps every
        suspected star in the promoted shells. This view exists only so a result can be
        reported with and without them, bounding the effect of a primary being counted
        twice (once on-axis, once here).
        """
        sh = self.shells
        drop: dict[tuple[int, int, int], int] = {}
        for cell in self.suspected_star_cells().values():
            drop[cell] = drop.get(cell, 0) + 1
        count = sh.count.copy()
        keys = zip(
            sh.pointing_index.tolist(),
            sh.distance_bin.tolist(),
            sh.eirp_min_bin.tolist(),
            strict=True,
        )
        for i, key in enumerate(keys):
            if key in drop:
                count[i] -= drop.pop(key)
        if drop or np.any(count < 0):
            raise ProvenanceError("suspected unmatched primaries do not address stored shell cells")
        keep = count > 0
        return replace(
            sh,
            pointing_index=sh.pointing_index[keep],
            distance_bin=sh.distance_bin[keep],
            eirp_min_bin=sh.eirp_min_bin[keep],
            count=count[keep],
        )


# Ruling v0.2.1-R3, stated once and surfaced wherever the count is shown.
SUSPECTED_PRIMARY_POLICY = (
    "listed for review and kept in the shells, never removed (ruling v0.2.1-R3): the only "
    "de-duplication keys are the Gaia DR3 source_id and, when an id is missing, a 3 arcsec "
    "match at J2000. A suspected star that really is its primary is counted twice in a merged "
    "run (once on-axis as the primary, once off-axis in a shell)"
)
# Scope of the published catalogue whose METHOD the background field reproduces. Every
# entry was checked against the paper text (arXiv:2006.09756v2) on 2026-11-27 except the
# one flagged `verified_against_paper: False`.
PUBLISHED_BACKGROUND_SCOPE: dict = {
    "citation": "Wlodarczyk-Sroka, Garrett & Siemion 2020, MNRAS 498, 5720 (arXiv:2006.09756)",
    "data_tier": "published_catalog",
    "pointings": (
        "1,327 Price et al. (2020) primary targets: 882 with GBT L-band, 1,005 with GBT S-band, "
        "189 with Parkes 10 cm"
    ),
    "n_pointings": 1327,
    "n_stars_total": 288315,
    "n_stars_gbt_l_plus_s_band": {
        "value": 249990,
        "verified_against_paper": True,
        "source": "Table 2, WG&S GBT cumulative N at EIRP_min <= 1e18 W",
    },
    "n_stars_gbt_l_band": {
        "value": 251983,
        "verified_against_paper": False,
        "source": (
            "v0.2.1b planning brief; not stated in the paper text, whose Table 2 gives 249,990 "
            "stars for GBT L- and S-band together"
        ),
    },
    "gaia_release": "Gaia DR2",
    "distances": "Bailer-Jones et al. (2018) geometric distances, d < 10 kpc",
    "selection": (
        "inside the primary-beam FWHM of each receiver (GBT L-band 8.4 arcmin, GBT S-band "
        "5.5 arcmin, Parkes 10 cm 6.4 arcmin)"
    ),
}


def _overlay_primaries(
    rec: BackgroundSummaryFile, prim: SightingTable, overlay: PointingOverlay | None
) -> SightingTable:
    """Beam centres the summary was built with: the catalog, plus the overlay it records.

    A summary that records corrected pointings must record exactly the overlay the survey
    pins, applied to exactly its entries, with the cone centres the overlay states and (for
    cones that came from the archive) the ADQL of a query at those centres.
    """
    if overlay is not None:
        overlay.corrected_pointings(prim)  # the pinned overlay must fit the locked catalog
    if rec.pointing_corrections_sha256 is None:
        return prim
    if overlay is None:
        raise ProvenanceError(
            "background summary records pointing corrections but its survey pins no overlay"
        )
    if (rec.pointing_corrections_path, rec.pointing_corrections_sha256) != (
        overlay.path,
        overlay.sha256,
    ):
        raise ProvenanceError(
            f"background summary was built with pointing corrections "
            f"{rec.pointing_corrections_sha256[:12]}..., the survey pins {overlay.sha256[:12]}..."
        )
    want = overlay.corrected_pointings(prim)
    got = rec.corrected_pointings
    if [c.target_name for c in got] != [w["target_name"] for w in want]:
        raise ProvenanceError(
            "background summary did not correct exactly the pointings of the pinned overlay"
        )
    radius = f"{rec.cone_radius_arcmin / 60.0:.8f}"
    for c, w in zip(got, want, strict=True):
        same = c.pointing_index == w["pointing_index"] and all(
            math.isclose(getattr(c, k), w[k], rel_tol=0.0, abs_tol=CATALOG_COORD_TOLERANCE_DEG)
            for k in ("catalog_ra_deg", "catalog_dec_deg", "cone_ra_deg", "cone_dec_deg")
        )
        if not same or c.truth_source != w["truth_source"]:
            raise ProvenanceError(
                f"{c.target_name}: cone centre ({c.cone_ra_deg}, {c.cone_dec_deg}) is not the "
                f"overlay's corrected position ({w['cone_ra_deg']}, {w['cone_dec_deg']})"
            )
        if c.adql_sha256 is not None:
            adql = (
                rec.adql_template.replace("{ra_deg}", f"{c.cone_ra_deg:.6f}")
                .replace("{dec_deg}", f"{c.cone_dec_deg:.6f}")
                .replace("{radius_deg}", radius)
            )
            if hashlib.sha256(adql.encode()).hexdigest() != c.adql_sha256:
                raise ProvenanceError(
                    f"{c.target_name}: the cached cone was not queried at the corrected "
                    f"coordinates ({c.cone_ra_deg:.6f}, {c.cone_dec_deg:.6f})"
                )
    return overlay.apply(prim)


def _recompute_background_summary(
    rec: BackgroundSummaryFile,
    survey: Survey,
    parent: Survey,
    root: Path,
    file_sha: str,
    overlay: PointingOverlay | None = None,
) -> tuple[str, radio.BackgroundShells]:
    """Offline re-derivation of everything the summary asserts about itself.

    Locked-primary checksum, merge provenance against that catalog, radiometer and beam
    constants, shell checksum, and per-star recomputation on the validation sample:
    theta from the beam centre, off-axis EIRP_min, both bin indices, zero Gaia ids shared
    with the primaries and zero unmatched ID-less primaries inside the fallback radius.
    Beam centres are the locked catalog's coordinates with the summary's recorded pointing
    corrections applied (ruling v0.2.1-R5); the catalog itself is only ever read.
    """
    if survey.parent_pointing_survey_id != parent.id:
        raise ProvenanceError(f"{survey.id} does not declare {parent.id!r} as its parent pointing")
    if rec.parent_survey_id != parent.id or rec.pointing_group_id != parent.pointing_group_id:
        raise ProvenanceError("background summary was built for a different parent pointing")
    if rec.data_tier != survey.data_tier:
        raise ProvenanceError(
            f"summary data_tier {rec.data_tier!r} != survey data_tier {survey.data_tier!r}"
        )
    prim = primary_sightings(parent, root)  # verifies the parent catalog checksum first
    if rec.primary_catalog_sha256 != prim.sha256 or rec.primary_catalog_path != parent.catalog_path:
        raise ProvenanceError("background summary was de-duplicated against a different catalog")
    centres = _overlay_primaries(rec, prim, overlay)  # beam centres / dedup positions
    mp, da, sh = rec.merge_provenance, rec.dedup_audit, rec.shells
    if mp.input_surveys != [parent.id, survey.id]:
        raise ProvenanceError(
            f"merge provenance covers {mp.input_surveys}, expected {[parent.id, survey.id]}"
        )
    for sv in (survey, parent):
        f_w = float(survey_f_min(sv).to_value(u.W / u.m**2))
        if not math.isclose(rec.f_min_w_m2, f_w, rel_tol=1e-12, abs_tol=0.0):
            raise ProvenanceError(
                f"summary F_min {rec.f_min_w_m2} != radiometer F_min {f_w} of {sv.id}"
            )
        fwhm = sv.radio_sensitivity.beam_fwhm_arcmin if sv.radio_sensitivity else None
        if fwhm is not None and fwhm != rec.beam_fwhm_arcmin:
            raise ProvenanceError(f"summary beam FWHM != {sv.id} beam_fwhm_arcmin {fwhm}")
    if sh.pointing_names != [str(x) for x in prim.names]:
        raise ProvenanceError("summary pointings are not the locked primary targets, in order")
    has_id = prim.gaia_dr3_source_id != MISSING_GAIA_ID
    if (da.n_primary_targets, da.n_primary_with_gaia_id) != (len(prim), int(has_id.sum())):
        raise ProvenanceError("dedup audit primary counts disagree with the locked catalog")
    if not math.isclose(da.angular_fallback_arcsec, mp.angular_fallback_arcsec, abs_tol=1e-12):
        raise ProvenanceError("dedup audit and merge provenance disagree on the fallback radius")
    cols = ("pointing_index", "distance_bin", "eirp_min_bin", "count")
    if _canonical_sha256({k: getattr(sh, k) for k in cols}) != sh.cells_sha256:
        raise ProvenanceError("shell table cells_sha256 mismatch")
    binning = radio.ShellBinning.from_dict(sh.binning.model_dump())
    fwhm_q = rec.beam_fwhm_arcmin * u.arcmin
    f_q = rec.f_min_w_m2 * u.W / u.m**2
    cone, tol = rec.cone_radius_arcmin, mp.angular_fallback_arcsec
    cell_count = dict(
        zip(
            zip(sh.pointing_index, sh.distance_bin, sh.eirp_min_bin, strict=True),
            sh.count,
            strict=True,
        )
    )

    bg = rec.validation_sample.background
    if bg:
        pidx = np.array([s.pointing_index for s in bg])
        if pidx.max() >= len(prim):
            raise ProvenanceError("validation star points at an unknown pointing")
        theta = np.array([s.theta_arcmin for s in bg])
        d = np.array([s.distance_pc for s in bg])
        centre = SkyCoord(centres.ra_deg[pidx] * u.deg, centres.dec_deg[pidx] * u.deg)
        star = SkyCoord([s.ra_deg for s in bg] * u.deg, [s.dec_deg for s in bg] * u.deg)
        theta_re = centre.separation(star).to_value(u.arcmin)
        if np.max(np.abs(theta_re - theta)) > 1e-6:
            raise ProvenanceError("recomputed beam offset theta disagrees with the stored value")
        if theta.max() > cone + 1e-9:
            raise ProvenanceError("validation star lies outside the cone radius")
        if d.max() > rec.distance_cuts.d_max_pc:
            raise ProvenanceError("validation star lies beyond d_max")
        eirp = radio.min_detectable_eirp_off_axis(d * u.pc, theta * u.arcmin, f_q, fwhm_q)
        stored = np.array([s.eirp_min_w for s in bg])
        if not np.allclose(eirp.to_value(u.W), stored, rtol=1e-12, atol=0.0):
            raise ProvenanceError("recomputed off-axis EIRP_min disagrees with the stored value")
        if np.any(eirp < radio.min_detectable_eirp(d * u.pc, f_q)):
            raise ProvenanceError("off-axis EIRP_min below the on-axis value")
        d_bin, e_bin = binning.distance_bin(d * u.pc), binning.eirp_bin(eirp)
        if not (
            np.array_equal(d_bin, [s.distance_bin for s in bg])
            and np.array_equal(e_bin, [s.eirp_min_bin for s in bg])
        ):
            raise ProvenanceError("recomputed shell bins disagree with the stored bins")
        keys, n_in = np.unique(np.column_stack([pidx, d_bin, e_bin]), axis=0, return_counts=True)
        for key, k in zip(keys.tolist(), n_in.tolist(), strict=True):
            if cell_count.get(tuple(key), 0) < k:
                raise ProvenanceError(f"validation stars in cell {key} exceed the stored count")
        ids = np.array([int(s.gaia_dr3_source_id) for s in bg], np.int64)  # exact
        if np.intersect1d(ids, prim.gaia_dr3_source_id[has_id]).size:
            raise DoubleCountingError("a validation background star duplicates a primary Gaia id")
        if (~has_id).any():
            match = SkyCoord(
                [s.ra_match_deg for s in bg] * u.deg, [s.dec_match_deg for s in bg] * u.deg
            )
            no_id = SkyCoord(centres.ra_deg[~has_id] * u.deg, centres.dec_deg[~has_id] * u.deg)
            _, _, sep, _ = match.search_around_sky(no_id, tol * u.arcsec)
            if np.any(sep.to_value(u.arcsec) < tol):
                raise DoubleCountingError(
                    "a validation background star lies inside the angular fallback radius of a "
                    "primary without a Gaia id (unmerged collision)"
                )
    index = {str(x): i for i, x in enumerate(prim.names)}
    # Suspected unmatched primaries (ruling v0.2.1-R3: listed, never removed): every one
    # must be a star that really is in the shells, next to an ID-less primary.
    suspects: dict[str, tuple[int, int, int]] = {}
    primary_ids = set(prim.gaia_dr3_source_id[has_id].tolist())
    for sp in da.suspected_unmatched_primary:
        j = index.get(sp.primary_target_name)
        if j is None or has_id[j]:
            raise ProvenanceError(
                f"suspected unmatched primary names {sp.primary_target_name!r}, which is not "
                "a primary without a Gaia id"
            )
        if int(sp.gaia_dr3_source_id) in primary_ids:
            raise DoubleCountingError("a suspected star carries a primary Gaia id")
        if sp.pointing_index >= len(prim) or sp.theta_arcmin > cone + 1e-9:
            raise ProvenanceError("suspected star lies outside every recorded beam")
        e_sp = radio.min_detectable_eirp_off_axis(
            sp.distance_pc * u.pc, sp.theta_arcmin * u.arcmin, f_q, fwhm_q
        )
        cell = (
            sp.pointing_index,
            int(binning.distance_bin(sp.distance_pc * u.pc)[0]),
            int(binning.eirp_bin(e_sp)[0]),
        )
        suspects[sp.gaia_dr3_source_id] = cell  # one star may neighbour two primaries
    per_cell: dict[tuple[int, int, int], int] = {}
    for cell in suspects.values():
        per_cell[cell] = per_cell.get(cell, 0) + 1
    for cell, k in per_cell.items():
        if cell_count.get(cell, 0) < k:
            raise ProvenanceError(
                f"suspected unmatched primaries in cell {cell} exceed the stored shell count"
            )
    for m in rec.validation_sample.primary_matched:
        j = index.get(m.primary_target_name)
        if j is None:
            raise ProvenanceError(
                f"primary-matched sighting names unknown target {m.primary_target_name!r}"
            )
        pid = int(prim.gaia_dr3_source_id[j])
        if m.match == "gaia_dr3_source_id" and pid != int(m.gaia_dr3_source_id):
            raise ProvenanceError(f"{m.primary_target_name}: Gaia-id match with a different id")
        if m.match == "angular_fallback" and (pid != MISSING_GAIA_ID or m.separation_arcsec >= tol):
            raise ProvenanceError(
                f"{m.primary_target_name}: angular fallback used although both records carry "
                "a Gaia id, or beyond the fallback radius"
            )
        if m.theta_arcmin > cone + 1e-9:
            raise ProvenanceError("primary-matched sighting lies outside the cone radius")
    shells = radio.BackgroundShells(
        pointing_unit_xyz=pointing_unit_vectors(centres.ra_deg, centres.dec_deg),
        pointing_index=np.asarray(sh.pointing_index, np.int64),
        distance_bin=np.asarray(sh.distance_bin, np.int64),
        eirp_min_bin=np.asarray(sh.eirp_min_bin, np.int64),
        count=np.asarray(sh.count, np.int64),
        binning=binning,
        data_tier=rec.data_tier,
        survey_id=survey.id,
        source_sha256=file_sha,
    )
    detail = (
        f"{sh.n_stars} background stars in {sh.n_cells} cells over {rec.n_pointings_completed}/"
        f"{rec.n_pointings} pointings; de-duplicated against {parent.catalog_path} "
        f"({prim.sha256[:8]}...): {mp.n_dedup_gaia_id} Gaia-id + {mp.n_dedup_angular_fallback} "
        f"angular removals, {mp.n_multi_beam_resolved} multi-beam stars, 0 duplicated Gaia ids; "
        f"{len(bg)} validation stars: theta, off-axis EIRP_min [W] and bins recomputed; "
        f"{len(rec.validation_sample.primary_matched)} primary-matched sightings verified; "
        f"{da.n_suspected_unmatched_primary} suspected_unmatched_primary row(s) "
        f"({len(suspects)} distinct stars) kept in the shells and reported (v0.2.1-R3)"
    )
    if rec.pointing_corrections_sha256 is not None:
        n_adql = sum(c.adql_sha256 is not None for c in rec.corrected_pointings)
        detail += (
            f"; {rec.n_corrected_pointings} cone centre(s) corrected by the pinned overlay "
            f"{rec.pointing_corrections_path} ({rec.pointing_corrections_sha256[:8]}...), "
            f"{n_adql} cone query hash(es) recomputed at the corrected coordinates (v0.2.1-R5)"
        )
    return detail, shells


def load_background_summary(survey: Survey, parent: Survey, root: Path) -> BackgroundField:
    """Load and fully verify an extension survey's pinned background summary.

    Checks: file size < 5 MB; file SHA-256 == survey pin; schema; content_sha256; then
    everything in :func:`_recompute_background_summary`.
    """
    if survey.background_summary_path is None:
        raise ProvenanceError(f"{survey.id} pins no background summary")
    path = root / survey.background_summary_path
    if not path.is_file():
        raise ProvenanceError(
            f"background summary {path} missing (scripts/fetch_wlodarczyk_sroka_288k.py)"
        )
    raw = path.read_bytes()
    if len(raw) >= BACKGROUND_SUMMARY_MAX_BYTES:
        raise ProvenanceError(f"{survey.background_summary_path}: {len(raw)} bytes is not < 5 MB")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != survey.background_summary_sha256:
        raise ProvenanceError(
            f"{survey.background_summary_path}: sha256 {sha} != pinned "
            f"{survey.background_summary_sha256}"
        )
    doc = json.loads(raw)
    rec = BackgroundSummaryFile.model_validate(doc)
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    if _canonical_sha256(body) != rec.content_sha256:
        raise ProvenanceError(f"{survey.background_summary_path}: content_sha256 mismatch")
    overlay = survey_pointing_overlay(survey, root)  # ProvenanceError on a pin mismatch
    detail, shells = _recompute_background_summary(rec, survey, parent, root, sha, overlay)
    return BackgroundField(
        survey.id, parent.id, survey.background_summary_path, sha, rec, shells, detail, overlay
    )


# ------------------------------------------- reproduced_archive sigma_sys calibration


@dataclass(frozen=True)
class SigmaSysCalibration:
    """Verified, measured W3 centroid registration floor (``reproduced_archive``)."""

    survey_id: str
    path: str
    file_sha256: str
    record: SigmaSysCalibrationFile

    @property
    def stats(self) -> dict:
        return self.record.statistics

    @property
    def sigma_sys(self) -> u.Quantity:
        return float(self.stats["calibration"]["sigma_sys_arcsec"]) * u.arcsec

    @property
    def interval_68(self) -> tuple[float, float]:
        lo, hi = self.stats["calibration"]["bootstrap"]["p16_p84_arcsec"]
        return float(lo), float(hi)

    @property
    def legacy_baseline(self) -> u.Quantity:
        return self.record.legacy_baseline_sigma_sys_arcsec * u.arcsec

    def summary(self) -> dict:
        cal, wh = self.stats["calibration"], self.stats["withheld"]
        return {
            "id": self.record.id,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "content_sha256": self.record.content_sha256,
            "data_tier": self.record.data_tier,
            "code_git_sha": self.record.code_git_sha,
            "rng_seed": self.record.rng_seed,
            "query_window_utc": list(self.record.query_window_utc),
            "n_controls": self.record.n_controls,
            "n_patches_with_controls": self.record.n_patches_with_controls,
            "primary_estimator": self.stats["primary_estimator"],
            "sigma_sys_arcsec": float(self.sigma_sys.to_value(u.arcsec)),
            "sigma_sys_68_arcsec": list(self.interval_68),
            "sigma_sys_by_estimator_arcsec": cal["sigma_sys_by_estimator_arcsec"],
            "n_calibration": cal["n_stars"],
            "n_withheld": wh["n_stars"],
            "withheld_false_alarm": {
                k: {
                    "n_flagged": wh[k]["n_flagged"],
                    "n_stars": wh[k]["n_stars"],
                    "rate": wh[k]["empirical_false_alarm"],
                    "clopper_pearson_95": wh[k]["clopper_pearson_95"],
                    "sigma_sys_arcsec": wh[k]["sigma_sys_arcsec"],
                    "ks_rayleigh_p_value": wh[k]["ks_rayleigh_p_value"],
                    "binomial_p_at_least_k_given_nominal": wh[k][
                        "binomial_p_at_least_k_given_nominal"
                    ],
                    "nominal_false_alarm": wh[k]["nominal_false_alarm"],
                }
                for k in (
                    "out_of_sample_false_alarm_at_calibrated_floor",
                    "false_alarm_at_legacy_baseline_floor",
                    "false_alarm_with_no_floor",
                )
            },
            "calibration_in_sample_false_alarm": {
                k: cal["in_sample_false_alarm_at_calibrated_floor"][k]
                for k in (
                    "n_flagged",
                    "n_stars",
                    "empirical_false_alarm",
                    "clopper_pearson_95",
                    "binomial_p_at_least_k_given_nominal",
                    "ks_rayleigh_p_value",
                    "nominal_false_alarm",
                    "sigma_sys_arcsec",
                )
            },
            "patches": [
                {
                    "patch_id": p.patch_id,
                    "l_deg": p.l_deg,
                    "b_deg": p.b_deg,
                    "ra_deg": p.ra_deg,
                    "dec_deg": p.dec_deg,
                    "radius_deg": p.radius_deg,
                    "measured_controls": p.cut_counts.get("measured_controls"),
                    "rejected_gaia_neighbour_within_15arcsec": p.cut_counts.get(
                        "rejected_gaia_neighbour_within_15arcsec"
                    ),
                    "n_calibration": sum(
                        c.patch_id == p.patch_id and c.split == "calibration"
                        for c in self.record.controls
                    ),
                    "n_withheld": sum(
                        c.patch_id == p.patch_id and c.split == "withheld"
                        for c in self.record.controls
                    ),
                }
                for p in self.record.patches
            ],
            "cut_counts_total": self.record.model_extra.get("cut_counts_total", {}),
            "min_abs_b_deg": float(
                self.record.model_extra.get("config", {}).get("min_abs_b", 20.0)
            ),
        }


def _recompute_sigma_sys(rec: SigmaSysCalibrationFile) -> str:
    """Offline re-derivation of the stored statistics from the stored controls."""
    st = rec.statistics
    est = hephaistos.SIGMA_SYS_ESTIMATORS[st["primary_estimator"]]

    def arrays(split: str) -> tuple[u.Quantity, u.Quantity]:
        rows = [c for c in rec.controls if c.split == split]
        return (
            np.array([c.offset_arcsec for c in rows]) * u.arcsec,
            np.array([c.sigma_stat_arcsec for c in rows]) * u.arcsec,
        )

    r_c, s_c = arrays("calibration")
    r_w, s_w = arrays("withheld")
    sigma = float(est(r_c, s_c).to_value(u.arcsec))
    stored = float(st["calibration"]["sigma_sys_arcsec"])
    if not math.isclose(sigma, stored, rel_tol=0, abs_tol=1e-9):
        raise ProvenanceError(f"recomputed sigma_sys {sigma} != stored {stored}")
    wh = st["withheld"]
    for key, floor in (
        ("out_of_sample_false_alarm_at_calibrated_floor", sigma),
        ("false_alarm_at_legacy_baseline_floor", rec.legacy_baseline_sigma_sys_arcsec),
        ("false_alarm_with_no_floor", 0.0),
    ):
        fa = hephaistos.centroid_screen_false_alarms(
            r_w, s_w, floor * u.arcsec, float(wh[key]["nominal_false_alarm"])
        )
        if fa["n_flagged"] != wh[key]["n_flagged"] or fa["n_stars"] != wh[key]["n_stars"]:
            raise ProvenanceError(f"recomputed withheld {key} disagrees with stored value")
    k = wh["out_of_sample_false_alarm_at_calibrated_floor"]
    return (
        f"sigma_sys[{st['primary_estimator']}] = {sigma:.4f} arcsec recomputed from "
        f"{len(r_c)} calibration stars; withheld false alarms {k['n_flagged']}/{k['n_stars']} "
        "recomputed"
    )


def load_sigma_sys_calibration(survey: Survey, root: Path) -> SigmaSysCalibration | None:
    """Load and fully verify a survey's pinned sigma_sys calibration (None if absent).

    Checks: file-bytes SHA-256 == pin in the survey YAML; schema; content_sha256 over
    the canonical JSON; offline recomputation of sigma_sys and the withheld false-alarm
    counts from the stored control table.
    """
    ir = survey.infrared
    if ir is None or ir.sigma_sys_calibration_path is None:
        return None
    path = root / ir.sigma_sys_calibration_path
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ir.sigma_sys_calibration_sha256:
        raise ProvenanceError(
            f"{ir.sigma_sys_calibration_path}: sha256 {sha} != pinned "
            f"{ir.sigma_sys_calibration_sha256}"
        )
    doc = json.loads(raw)
    rec = SigmaSysCalibrationFile.model_validate(doc)
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    if _canonical_sha256(body) != rec.content_sha256:
        raise ProvenanceError(f"{ir.sigma_sys_calibration_path}: content_sha256 mismatch")
    _recompute_sigma_sys(rec)
    return SigmaSysCalibration(survey.id, ir.sigma_sys_calibration_path, sha, rec)


# ------------------------------------- reproduced_archive Gate #2 (real AllWISE FITS)
def _close(a: object, b: object, path: str = "", tol: float = 1e-9) -> None:
    """Recursive equality with an absolute float tolerance (raises ProvenanceError)."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise ProvenanceError(f"{path}: keys differ ({sorted(set(a) ^ set(b))})")
        for k in a:
            _close(a[k], b[k], f"{path}.{k}", tol)
    elif isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            raise ProvenanceError(f"{path}: lengths differ")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, bool | str) or isinstance(b, bool | str) or a is None or b is None:
        if a != b:
            raise ProvenanceError(f"{path}: {a!r} != {b!r}")
    elif not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol):
        raise ProvenanceError(f"{path}: recomputed {a!r} != stored {b!r}")


@dataclass(frozen=True)
class FitsGate2Result:
    """Verified real-FITS withheld blend-recovery result (``reproduced_archive``)."""

    survey_id: str
    path: str
    file_sha256: str
    record: WiseFitsGate2File

    def summary(self) -> dict:
        r = self.record
        fz, wh, psf = r.frozen_screens, r.withheld, r.empirical_psf
        return {
            "id": r.id,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "content_sha256": r.content_sha256,
            "data_tier": r.data_tier,
            "code_git_sha": r.code_git_sha,
            "rng_seed": r.rng_seed,
            "query_window_utc": list(r.query_window_utc),
            "frozen_at_utc": r.frozen_at_utc,
            "n_requested": r.fetch.n_requested,
            "n_fetched_ok": r.fetch.n_ok,
            "n_calibration": len(fz.calibration["star_ids"]),
            "n_withheld": wh.n_stars,
            "locked_sigma_sys_arcsec": r.locked_inputs.sigma_sys_arcsec,
            "empirical_psf": {
                "fwhm_gaussian_arcsec": psf.fwhm_gaussian_arcsec,
                "fwhm_halfmax_arcsec": psf.fwhm_halfmax_arcsec,
                "nominal_fwhm_arcsec": psf.nominal_fwhm_arcsec,
                "n_stars": psf.n_stars,
                "pixel_scale_arcsec": psf.pixel_scale_arcsec,
            },
            "frozen_screens": {
                "rule_version": fz.rule_version,
                "target_false_alarm": fz.target_false_alarm,
                "centroid_multiplier": fz.centroid["multiplier"],
                "centroid_threshold_arcsec_at_calibration_median_snr": fz.centroid[
                    "threshold_arcsec_at_calibration_median_snr"
                ],
                "delta_chi2_scale": fz.image["delta_chi2_scale"],
                "delta_chi2_threshold": fz.image["delta_chi2_threshold"],
                "calibration_false_alarm_in_sample": fz.calibration_false_alarm_in_sample,
                "snr_eff_rule": fz.model_extra.get("snr_eff_rule"),
            },
            "separations_arcsec": wh.separations_arcsec,
            "flux_fractions": wh.flux_fractions,
            "screens": wh.summary["screens"],
            "survival_weighting": wh.summary["survival_weighting"],
            "caveats": r.model_extra.get("caveats", []),
        }


def _recompute_fits_gate2(rec: WiseFitsGate2File, locked: SigmaSysCalibration) -> str:
    """Offline re-derivation of the Gate #2 summaries + split-integrity audit."""
    li = rec.locked_inputs
    if li.sigma_sys_calibration_path != locked.path or (
        li.sigma_sys_calibration_sha256 != locked.file_sha256
    ):
        raise ProvenanceError("Gate #2 was frozen against a different sigma_sys calibration")
    sys_locked = float(locked.sigma_sys.to_value(u.arcsec))
    for v in (li.sigma_sys_arcsec, rec.frozen_screens.centroid["sigma_sys_arcsec_locked"]):
        if not math.isclose(float(v), sys_locked, rel_tol=0, abs_tol=1e-12):
            raise ProvenanceError(f"Gate #2 sigma_sys {v} != locked {sys_locked}")
    split = {
        s: {c.gaia_dr3_source_id for c in locked.record.controls if c.split == s}
        for s in ("calibration", "withheld")
    }
    if set(rec.frozen_screens.calibration["star_ids"]) != split["calibration"]:
        raise ProvenanceError("Gate #2 calibration stars != locked calibration split")
    if set(rec.withheld.star_ids) != split["withheld"]:
        raise ProvenanceError("Gate #2 withheld stars != locked withheld split (incomplete?)")
    payload = {
        "empirical_psf": rec.empirical_psf.model_dump(),
        "frozen_screens": rec.frozen_screens.model_dump(),
        "withheld": rec.withheld.model_dump(),
    }
    got = hephaistos_fits.recompute_gate2(payload)
    psf, fz = rec.empirical_psf, rec.frozen_screens
    _close(got["stamp_sum"], 1.0, "empirical_psf.stamp_sum")
    _close(got["fwhm_gaussian_arcsec"], psf.fwhm_gaussian_arcsec, "fwhm_gaussian", 1e-6)
    _close(got["fwhm_halfmax_arcsec"], psf.fwhm_halfmax_arcsec, "fwhm_halfmax", 1e-9)
    _close(got["delta_chi2_scale"], fz.image["delta_chi2_scale"], "delta_chi2_scale")
    _close(got["delta_chi2_threshold"], fz.image["delta_chi2_threshold"], "delta_chi2_threshold")
    _close(got["centroid_multiplier"], fz.centroid["multiplier"], "centroid_multiplier")
    _close(
        got["calibration_false_alarm_in_sample"],
        fz.calibration_false_alarm_in_sample,
        "calibration_false_alarm_in_sample",
    )
    _close(got["summary"], rec.withheld.summary, "withheld.summary")
    img = rec.withheld.summary["screens"]["image"]
    cen = rec.withheld.summary["screens"]["centroid"]
    return (
        f"PSF FWHM {psf.fwhm_gaussian_arcsec:.3f} arcsec (nominal {psf.nominal_fwhm_arcsec}); "
        f"thresholds re-derived from {len(split['calibration'])} calibration stars; "
        f"{rec.withheld.n_stars} withheld summaries recomputed (survival <= 3.25 arcsec: "
        f"centroid {cen['blend_survival_within_3p25_arcsec']:.3f}, image "
        f"{img['blend_survival_within_3p25_arcsec']:.3f})"
    )


def load_fits_gate2(survey: Survey, root: Path) -> FitsGate2Result | None:
    """Load and fully verify a survey's pinned Gate #2 result (None if absent).

    Checks: file SHA-256 == survey pin; schema; content_sha256; locked sigma_sys input
    (file hash and value); calibration / withheld star sets == the locked split; and
    offline recomputation of thresholds, PSF width and every withheld summary.
    """
    ir = survey.infrared
    if ir is None or ir.fits_gate2_path is None:
        return None
    locked = load_sigma_sys_calibration(survey, root)
    if locked is None:
        raise ProvenanceError("Gate #2 result pinned without the locked sigma_sys calibration")
    raw = (root / ir.fits_gate2_path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ir.fits_gate2_sha256:
        raise ProvenanceError(
            f"{ir.fits_gate2_path}: sha256 {sha} != pinned {ir.fits_gate2_sha256}"
        )
    doc = json.loads(raw)
    rec = WiseFitsGate2File.model_validate(doc)
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    if _canonical_sha256(body) != rec.content_sha256:
        raise ProvenanceError(f"{ir.fits_gate2_path}: content_sha256 mismatch")
    _recompute_fits_gate2(rec, locked)
    return FitsGate2Result(survey.id, ir.fits_gate2_path, sha, rec)


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


def _check_finite_disk_geometry(reg: ParameterRegistry) -> str:
    """Registry Galactic-structure defaults must build a valid finite_disk_3d model (offline)."""

    def q(pid: str) -> u.Quantity:
        p = reg.get(pid)
        if p.default is None:
            raise SchemaValidationError(f"{pid}: finite_disk_3d needs a registry default")
        return p.quantity(p.default)

    geo = {}
    for profile in DISK_PROFILES:
        model = ExpansionModel(
            FINITE_DISK,
            origin_rate=1.0 / (u.kpc**2 * u.Gyr),
            expansion_speed=0.01 * u.dimensionless_unscaled,
            lookback=1.0e-3 * u.Gyr,
            scale_height=q("disk_scale_height_pc"),
            solar_height=q("solar_height_pc"),
            disk_profile=profile,
            disk_radius=q("galactic_disk_radius_kpc"),
            solar_galactocentric_radius=q("solar_galactocentric_radius_kpc"),
            radial_scale_length=q("galactic_radial_scale_length_kpc"),
        )
        geo[profile] = model.finite_disk_summary()
    g = geo["exponential"]
    return (
        f"R_gal = {g['galactic_disk_radius_kpc']:g} kpc, Sun at (R, Z) = "
        f"({g['solar_cylindrical_radius_kpc']:.5f}, {g['solar_height_kpc']:.4f}) kpc "
        f"(R_0 = {g['solar_galactocentric_distance_kpc']:g} kpc), h_z = "
        f"{g['scale_height_kpc']:g} kpc, R_d = {g['radial_scale_length_kpc']:g} kpc; profiles "
        f"{list(geo)} build offline (astropy Galactocentric frame, no network)"
    )


def _check_merge_policy_contract(graph: FactorGraph) -> str:
    """Self-test of the pointing merge policy contract (offline, no data needed)."""
    for call in (
        lambda: validate_pointing_merge_policy("keep_first"),
        lambda: resolve_pointing_groups(list(graph.surveys.values()), "keep_first"),
    ):
        try:
            call()
        except MergePolicyError:
            continue
        raise DoubleCountingError("keep_first was accepted as a pointing merge policy")
    res = resolve_pointing_groups(list(graph.surveys.values()))
    ext = [s.id for s in graph.surveys.values() if s.is_pointing_extension]
    return (
        f"default {res.policy}; allowed {list(POINTING_MERGE_POLICIES)}; keep_first refused "
        f"(MergePolicyError); declared pointing extensions: {ext or 'none'}"
    )


def _check_background_promotion(graph: FactorGraph) -> str:
    """The canonical background summary must be either absent or pinned by a survey."""
    pinned = [s for s in graph.surveys.values() if s.background_summary_path is not None]
    present = (graph.root / BACKGROUND_SUMMARY_PATH).is_file()
    if present and BACKGROUND_SUMMARY_PATH not in {s.background_summary_path for s in pinned}:
        raise ProvenanceError(
            f"{BACKGROUND_SUMMARY_PATH} exists but no survey pins its SHA-256 "
            "(background_summary_path / background_summary_sha256 in data/surveys/)"
        )
    if not pinned:
        return (
            f"not promoted yet: no survey pins a background summary and {BACKGROUND_SUMMARY_PATH} "
            "is absent (scripts/fetch_wlodarczyk_sroka_288k.py --fetch, then --promote-from); "
            "once pinned, SHA-256, "
            "schema and offline recomputation are enforced automatically"
        )
    return "pinned by " + ", ".join(f"{s.id} -> {s.background_summary_path}" for s in pinned)


def _check_pointing_corrections(graph: FactorGraph, survey: Survey) -> str:
    """The pinned pointing-correction overlay (ruling v0.2.1-R5), end to end and offline.

    SHA-256 pin, schema, fit to the locked catalog (which is only read), and the state of
    the promoted summary: every corrected cone must have been queried at the overlay's
    coordinates (re-derived in ``_overlay_primaries``).
    """
    parent = graph.surveys[survey.parent_pointing_survey_id or ""]
    overlay = survey_pointing_overlay(survey, graph.root)
    if overlay is None:
        raise ProvenanceError(f"{survey.id} pins no pointing corrections")
    prim = primary_sightings(parent, graph.root)
    rows = overlay.corrected_pointings(prim)
    pc = load_background_summary(survey, parent, graph.root).pointing_corrections()
    head = (
        f"sha256 {overlay.sha256[:16]}... pinned by {survey.id} ({overlay.record.ruling}); "
        f"{len(rows)} corrections fit the locked {parent.catalog_path} ({prim.sha256[:8]}..., "
        "read-only)"
    )
    if not pc["artifact_built_with_overlay"]:
        return (
            f"{head}; promoted summary PREDATES the overlay: 0/{len(rows)} cones corrected, "
            f"{pc['n_uncorrected_pointings']} uncorrected flipped cone(s) - re-run "
            "scripts/fetch_wlodarczyk_sroka_288k.py --fetch, then --promote-from"
        )
    decs = [c["cone_dec_deg"] for c in pc["corrected_pointings"]]
    return (
        f"{head}; promoted summary: {pc['n_corrected_pointings']}/{len(rows)} cones fetched at "
        f"the corrected coordinates (cone dec {min(decs):+.4f} .. {max(decs):+.4f} deg), "
        f"{pc['n_uncorrected_pointings']} uncorrected"
    )


def _check_lookback_prior(reg: ParameterRegistry) -> str:
    p = reg.get("lookback_window_gyr")
    lo, hi = p.marginalization_log_bounds  # raises SchemaValidationError if undeclared
    if p.default is not None and not lo <= p.default <= hi:
        raise SchemaValidationError(
            f"lookback default {p.default} lies outside its marginalisation prior [{lo}, {hi}]"
        )
    return (
        f"log-uniform T in [{lo * 1e6:g} kyr, {hi * 1e3:g} Myr] (effective bounds after the "
        f"log_floor rule; unit {p.unit}); fixed default {p.default:g} {p.unit} lies inside"
    )


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
    record("finite_disk_geometry", lambda: _check_finite_disk_geometry(reg))
    record("lookback_marginalization_prior", lambda: _check_lookback_prior(reg))
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
            and f"{len({s.pointing_group_id for s in graph.surveys.values()})} distinct groups"
        ),
    )
    record("pointing_merge_policy", lambda: _check_merge_policy_contract(graph))
    record("background_field_promotion", lambda: _check_background_promotion(graph))
    for s in graph.surveys.values():
        if s.background_summary_path is not None:

            def _background(s: Survey = s) -> str:
                parent = graph.surveys[s.parent_pointing_survey_id or ""]
                bf = load_background_summary(s, parent, graph.root)
                return (
                    f"sha256 {bf.file_sha256[:16]}..., {bf.record.data_tier}, git "
                    f"{bf.record.code_git_sha[:12]}; {bf.detail}"
                )

            record(f"background_summary:{s.background_summary_path}", _background)
        if s.pointing_corrections_path is not None:
            record(
                f"pointing_corrections:{s.pointing_corrections_path}",
                lambda s=s: _check_pointing_corrections(graph, s),
            )
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
        if s.infrared is not None and s.infrared.sigma_sys_calibration_path is not None:

            def _calib(s: Survey = s) -> str:
                cal = load_sigma_sys_calibration(s, graph.root)
                assert cal is not None
                return (
                    f"sha256 {cal.file_sha256[:16]}..., {cal.record.data_tier}, "
                    f"git {cal.record.code_git_sha[:12]}, {cal.record.n_controls} controls / "
                    f"{cal.record.n_patches_with_controls} patches; "
                    + _recompute_sigma_sys(cal.record)
                )

            record(f"sigma_sys_calibration:{s.infrared.sigma_sys_calibration_path}", _calib)
        if s.infrared is not None and s.infrared.fits_gate2_path is not None:

            def _gate2(s: Survey = s) -> str:
                locked = load_sigma_sys_calibration(s, graph.root)
                g = load_fits_gate2(s, graph.root)
                assert g is not None and locked is not None
                return (
                    f"sha256 {g.file_sha256[:16]}..., {g.record.data_tier}, git "
                    f"{g.record.code_git_sha[:12]}, {g.record.fetch.n_ok}/"
                    f"{g.record.fetch.n_requested} cutouts; "
                    + _recompute_fits_gate2(g.record, locked)
                )

            record(f"fits_gate2:{s.infrared.fits_gate2_path}", _gate2)
    return results
