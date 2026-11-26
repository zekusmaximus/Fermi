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
from fermi_desk.models import hephaistos, hephaistos_fits
from fermi_desk.schemas import (
    Hypothesis,
    ParameterRegistry,
    SigmaSysCalibrationFile,
    Survey,
    WiseFitsGate2File,
)

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


# ------------------------------------------- reproduced_archive sigma_sys calibration
def _canonical_sha256(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=True).encode()).hexdigest()


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
