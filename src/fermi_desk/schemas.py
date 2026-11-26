"""Pydantic v2 data contracts (SPEC.md Section 4) with astropy unit validation."""

from __future__ import annotations

import re
from typing import Literal

import astropy.units as u
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fermi_desk.errors import (
    ProvenanceError,
    SchemaValidationError,
    Tier3SchemaError,
    UnitValidationError,
)

DataTier = Literal[
    "published_catalog",
    "reproduced_archive",
    "illustrative_calibration",
    "synthetic_proxy",
    "synthetic_proxy_692",
]
Channel = Literal["radio", "infrared", "solar_system"]
PriorType = Literal["log_uniform", "uniform", "fixed", "fixed_or_bounded"]

# astropy does not parse the word "dimensionless"; SPEC.md uses it, so it is the
# single documented alias. Every other unit string must parse via u.Unit().
_UNIT_ALIASES: dict[str, u.UnitBase] = {"dimensionless": u.dimensionless_unscaled}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_unit(unit: str) -> u.UnitBase:
    """Parse a registry unit string, raising ``UnitValidationError`` on failure."""
    if not isinstance(unit, str) or not unit.strip():
        raise UnitValidationError(f"Unit must be a non-empty string, got {unit!r}")
    key = unit.strip()
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    try:
        parsed = u.Unit(key, parse_strict="raise")
    except (ValueError, TypeError) as exc:
        raise UnitValidationError(f"Invalid astropy unit string {unit!r}: {exc}") from exc
    if isinstance(parsed, u.UnrecognizedUnit):
        raise UnitValidationError(f"Unrecognized astropy unit string {unit!r}")
    return parsed


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- params
class Parameter(_Frozen):
    id: str
    symbol: str
    unit: str
    prior_type: PriorType
    bounds: tuple[float, float]
    default: float | None = None
    log_floor: float | None = None
    description: str
    citations: list[str] = Field(default_factory=list)

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, v: str) -> str:
        parse_unit(v)  # raises UnitValidationError (propagates unwrapped)
        return v

    @model_validator(mode="after")
    def _check_bounds(self) -> Parameter:
        lo, hi = self.bounds
        if not lo <= hi:
            raise SchemaValidationError(f"{self.id}: bounds {self.bounds} not ordered")
        floor_ok = self.log_floor is not None and 0 < self.log_floor < hi
        if self.prior_type == "log_uniform" and lo <= 0 and not floor_ok:
            raise SchemaValidationError(
                f"{self.id}: log_uniform prior with lower bound {lo} <= 0 requires a "
                "positive log_floor below the upper bound"
            )
        if self.default is not None and not lo <= self.default <= hi:
            raise SchemaValidationError(f"{self.id}: default {self.default} outside {self.bounds}")
        return self

    @property
    def astropy_unit(self) -> u.UnitBase:
        return parse_unit(self.unit)

    def quantity(self, value: float) -> u.Quantity:
        return value * self.astropy_unit


class ParameterRegistry(_Frozen):
    schema_version: int = 1
    parameters: list[Parameter]

    @model_validator(mode="after")
    def _unique(self) -> ParameterRegistry:
        ids = [p.id for p in self.parameters]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise SchemaValidationError(f"Duplicate parameter ids: {dupes}")
        return self

    def get(self, pid: str) -> Parameter:
        for p in self.parameters:
            if p.id == pid:
                return p
        raise KeyError(pid)

    @property
    def ids(self) -> set[str]:
        return {p.id for p in self.parameters}


# ----------------------------------------------------------------------- hypotheses
class ParameterConstraint(_Frozen):
    mode: Literal["fixed", "range"]
    value: float | None = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _shape(self) -> ParameterConstraint:
        if self.mode == "fixed":
            if self.value is None or self.min is not None or self.max is not None:
                raise SchemaValidationError("fixed constraint needs exactly `value`")
        else:
            if self.min is None or self.max is None or self.value is not None:
                raise SchemaValidationError("range constraint needs exactly `min` and `max`")
            if not self.min < self.max:
                raise SchemaValidationError(f"range min {self.min} must be < max {self.max}")
        return self


class MonocultureAudit(_Frozen):
    required_compliance_fraction: float = Field(gt=0.0, le=1.0)
    timescale_gyr: float = Field(gt=0.0)
    defector_vulnerability: str = Field(min_length=1)


class Hypothesis(_Frozen):
    id: str
    name: str
    tier: Literal[1, 2, 3]
    summary: str
    parameter_constraints: dict[str, ParameterConstraint] = Field(default_factory=dict)
    channels_activated: list[Channel] = Field(default_factory=list)
    falsification_condition: str | None = None
    quarantine_reason: str | None = None
    monoculture_audit: MonocultureAudit | None = None
    benchmark_notes: str | None = None
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _tier_rules(self) -> Hypothesis:
        if self.tier == 3:
            missing = [
                k for k in ("quarantine_reason", "monoculture_audit") if getattr(self, k) is None
            ]
            if missing:
                raise Tier3SchemaError(f"Tier-3 hypothesis {self.id!r} missing {missing}")
            if self.channels_activated or self.parameter_constraints:
                raise Tier3SchemaError(
                    f"Tier-3 hypothesis {self.id!r} must not declare empirical channels or "
                    "parameter constraints (quarantined from exclusion races)"
                )
        elif not self.channels_activated:
            raise SchemaValidationError(f"Tier-{self.tier} hypothesis {self.id!r} has no channels")
        return self

    @property
    def is_quarantined(self) -> bool:
        return self.tier == 3


# -------------------------------------------------------------------------- surveys
class RadioSensitivity(_Frozen):
    data_tier: DataTier
    sefd_jy: float = Field(gt=0)
    snr_threshold: float = Field(gt=0)
    channel_bandwidth_hz: float = Field(gt=0)
    integration_time_s: float = Field(gt=0)
    n_pol: int = Field(ge=1, le=2)
    reference_eirp_w: float | None = Field(default=None, gt=0)
    reference_distance_pc: float | None = Field(default=None, gt=0)
    beam_fwhm_arcmin: float | None = Field(default=None, gt=0)
    citation: str

    @property
    def sefd(self) -> u.Quantity:
        return self.sefd_jy * u.Jy

    @property
    def channel_bandwidth(self) -> u.Quantity:
        return self.channel_bandwidth_hz * u.Hz

    @property
    def integration_time(self) -> u.Quantity:
        return self.integration_time_s * u.s


class IRCalibrationPoint(_Frozen):
    id: str
    completeness: float = Field(gt=0.0, le=1.0)
    false_positive_rate: float = Field(ge=0.0, lt=1.0)


class InfraredSpec(_Frozen):
    counts_data_tier: DataTier
    counts_citation: str
    footprint_max_distance_pc: float = Field(gt=0)
    anchor_false_positive_rate: float | None = None
    anchor_false_positive_citation: str | None = None
    calibration_data_tier: DataTier
    calibration_construction: str
    calibrations: list[IRCalibrationPoint] = Field(min_length=1)
    # Milestone 2.5: measured W3 centroid registration floor (reproduced_archive),
    # pinned by the SHA-256 of the file bytes like a catalog.
    sigma_sys_calibration_path: str | None = None
    sigma_sys_calibration_sha256: str | None = None
    # v0.1.2 / Launch Gate #2: frozen-method real-FITS withheld blend-recovery result
    # (reproduced_archive), pinned by file SHA-256 the same way.
    fits_gate2_path: str | None = None
    fits_gate2_sha256: str | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> InfraredSpec:
        ids = [c.id for c in self.calibrations]
        if len(ids) != len(set(ids)):
            raise SchemaValidationError("Duplicate IR calibration ids")
        if (self.sigma_sys_calibration_path is None) != (self.sigma_sys_calibration_sha256 is None):
            raise ProvenanceError("sigma_sys_calibration_path and _sha256 must be given together")
        if self.sigma_sys_calibration_sha256 is not None and not _SHA256_RE.match(
            self.sigma_sys_calibration_sha256
        ):
            raise ProvenanceError("sigma_sys_calibration_sha256 must be a 64-hex SHA-256")
        if (self.fits_gate2_path is None) != (self.fits_gate2_sha256 is None):
            raise ProvenanceError("fits_gate2_path and fits_gate2_sha256 must be given together")
        if self.fits_gate2_sha256 is not None and not _SHA256_RE.match(self.fits_gate2_sha256):
            raise ProvenanceError("fits_gate2_sha256 must be a 64-hex SHA-256")
        if self.fits_gate2_path is not None and self.sigma_sys_calibration_path is None:
            raise ProvenanceError("fits_gate2 requires the locked sigma_sys calibration it froze")
        return self


class SolarSystemSpec(_Frozen):
    description: str


class Survey(_Frozen):
    id: str
    name: str
    channel: Channel
    pointing_group_id: str = Field(min_length=1)
    data_tier: DataTier
    n_targets: int = Field(ge=0)
    observed_detections: int = Field(ge=0)
    frequency_ghz: tuple[float, float] | None = None
    minimum_flux_limit_jy: float | None = Field(default=None, gt=0)
    sefd_jy: float | None = Field(default=None, gt=0)
    catalog_path: str | None = None
    catalog_sha256: str | None = None
    catalog_provenance_path: str | None = None
    radio_sensitivity: RadioSensitivity | None = None
    infrared: InfraredSpec | None = None
    solar_system: SolarSystemSpec | None = None
    anchor_claims: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _channel_contract(self) -> Survey:
        if self.channel == "radio" and (
            self.radio_sensitivity is None or self.catalog_path is None
        ):
            raise SchemaValidationError(
                f"Radio survey {self.id!r} needs radio_sensitivity and catalog_path"
            )
        if (
            self.sefd_jy is not None
            and self.radio_sensitivity is not None
            and self.sefd_jy != self.radio_sensitivity.sefd_jy
        ):
            raise SchemaValidationError(f"{self.id}: sefd_jy disagrees with radio_sensitivity")
        if self.channel == "infrared" and self.infrared is None:
            raise SchemaValidationError(f"Infrared survey {self.id!r} needs an `infrared` block")
        if self.channel == "solar_system" and self.solar_system is None:
            raise SchemaValidationError(f"Solar-system survey {self.id!r} needs `solar_system`")
        if self.catalog_path is not None and (
            self.catalog_sha256 is None or not _SHA256_RE.match(self.catalog_sha256)
        ):
            raise ProvenanceError(
                f"Survey {self.id!r} references a catalog without a valid catalog_sha256"
            )
        if self.data_tier == "published_catalog" and not self.citations:
            raise ProvenanceError(f"published_catalog survey {self.id!r} must carry citations")
        return self


# ------------------------------------------------- reproduced_archive calibrations
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class _Open(BaseModel):
    """Calibration records: required fields are strict, extra provenance is kept."""

    model_config = ConfigDict(extra="allow", frozen=True)


class ControlStar(_Open):
    gaia_dr3_source_id: int
    patch_id: str
    b_deg: float
    ruwe: float = Field(lt=1.4)
    cc_flags: Literal["0000"]
    ext_flg: Literal[0]
    w3snr: float = Field(gt=0)
    offset_arcsec: float = Field(ge=0)
    sigma_stat_arcsec: float = Field(gt=0)
    split: Literal["calibration", "withheld"]


class ControlPatch(_Open):
    patch_id: str
    l_deg: float
    b_deg: float
    ra_deg: float
    dec_deg: float
    radius_deg: float = Field(gt=0)
    gaia_adql: str
    irsa_adql: str
    cut_counts: dict[str, int]


class SourceReceipt(_Frozen):
    path: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ProvenanceError("source_receipt.sha256 must be a 64-hex SHA-256")
        return v


class SigmaSysCalibrationFile(_Open):
    """``data/calibrations/wise_sigma_sys_control.json`` (Milestone 2.5, Launch Gate #1)."""

    id: str
    schema_version: Literal[1]
    data_tier: Literal["reproduced_archive"]
    code_git_sha: str
    rng_seed: int
    query_window_utc: tuple[str, str]
    legacy_baseline_sigma_sys_arcsec: float = Field(gt=0)
    n_controls: int = Field(ge=100)
    n_patches_with_controls: int = Field(ge=8)
    patches: list[ControlPatch] = Field(min_length=8)
    controls: list[ControlStar] = Field(min_length=100)
    statistics: dict
    source_receipt: SourceReceipt
    content_sha256: str

    @model_validator(mode="after")
    def _contract(self) -> SigmaSysCalibrationFile:
        if not _GIT_SHA_RE.match(self.code_git_sha):
            raise ProvenanceError(
                f"code_git_sha {self.code_git_sha!r} is not a clean 40-hex commit (dirty run?)"
            )
        if not _SHA256_RE.match(self.content_sha256):
            raise ProvenanceError("content_sha256 must be a 64-hex SHA-256")
        if len(self.controls) != self.n_controls:
            raise SchemaValidationError("n_controls disagrees with the controls table")
        min_b = float(self.model_extra.get("config", {}).get("min_abs_b", 20.0))
        for p in self.patches:
            if abs(p.b_deg) - p.radius_deg < min_b:
                raise SchemaValidationError(f"patch {p.patch_id} cone reaches |b| < {min_b}")
        ids = {p.patch_id for p in self.patches}
        if any(c.patch_id not in ids or abs(c.b_deg) <= min_b for c in self.controls):
            raise SchemaValidationError("control outside a declared high-latitude patch")
        if len({c.gaia_dr3_source_id for c in self.controls}) != len(self.controls):
            raise SchemaValidationError("duplicate control star")
        if len({c.patch_id for c in self.controls}) != self.n_patches_with_controls:
            raise SchemaValidationError("n_patches_with_controls disagrees with controls")
        n_cal = sum(c.split == "calibration" for c in self.controls)
        if abs(2 * n_cal - len(self.controls)) > 1:
            raise SchemaValidationError("calibration / withheld split is not 50/50")
        return self


# ------------------------------------- reproduced_archive Gate #2 (real AllWISE FITS)
class Gate2LockedInputs(_Frozen):
    sigma_sys_calibration_path: str
    sigma_sys_calibration_sha256: str
    sigma_sys_arcsec: float = Field(gt=0)

    @field_validator("sigma_sys_calibration_sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ProvenanceError("locked sigma_sys_calibration_sha256 must be a 64-hex SHA-256")
        return v


class Gate2Fetch(_Open):
    n_requested: int = Field(ge=1)
    n_ok: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    files: list[dict]

    @model_validator(mode="after")
    def _complete(self) -> Gate2Fetch:
        if self.n_ok + self.n_failed != self.n_requested:
            raise SchemaValidationError("fetch counts do not add up")
        if self.n_failed or self.n_ok != self.n_requested:
            raise ProvenanceError("Gate #2 file built from an incomplete cutout fetch")
        if any(
            not _SHA256_RE.match(str(f.get(k, "")))
            for f in self.files
            for k in ("int_sha256", "unc_sha256")
        ):
            raise ProvenanceError("every fetched cutout needs int_sha256 and unc_sha256")
        if len(self.files) != self.n_ok:
            raise SchemaValidationError("fetch.files disagrees with n_ok")
        return self


class Gate2PSF(_Open):
    stamp: list[list[float]] = Field(min_length=3)
    stamp_half_pix: int = Field(ge=1)
    pixel_scale_arcsec: float = Field(gt=0)
    n_stars: int = Field(ge=5)
    star_ids: list[int]
    fwhm_gaussian_arcsec: float = Field(gt=0)
    fwhm_halfmax_arcsec: float = Field(gt=0)
    nominal_fwhm_arcsec: float = Field(gt=0)

    @model_validator(mode="after")
    def _shape(self) -> Gate2PSF:
        n = 2 * self.stamp_half_pix + 1
        if len(self.stamp) != n or any(len(r) != n for r in self.stamp):
            raise SchemaValidationError("PSF stamp shape disagrees with stamp_half_pix")
        if len(self.star_ids) != self.n_stars:
            raise SchemaValidationError("PSF n_stars disagrees with star_ids")
        return self


class Gate2Frozen(_Open):
    rule_version: str
    target_false_alarm: float = Field(gt=0, lt=1)
    centroid: dict
    image: dict
    calibration: dict
    psf_star_ids: list[int]
    calibration_false_alarm_in_sample: dict


class Gate2Withheld(_Open):
    rng_seed: int
    n_stars: int = Field(ge=1)
    star_ids: list[int]
    w3snr: list[float]
    separations_arcsec: list[float] = Field(min_length=1)
    flux_fractions: list[float] = Field(min_length=1)
    statistics: dict
    summary: dict

    @model_validator(mode="after")
    def _shapes(self) -> Gate2Withheld:
        n, ns, nf = self.n_stars, len(self.separations_arcsec), len(self.flux_fractions)
        if len(self.star_ids) != n or len(self.w3snr) != n:
            raise SchemaValidationError("withheld per-star arrays disagree with n_stars")
        want = {"null": (n,), "arm_a": (n, ns, nf), "arm_b": (n, nf)}
        for arm, shape in want.items():
            block = self.statistics.get(arm)
            if not isinstance(block, dict):
                raise SchemaValidationError(f"withheld statistics missing arm {arm!r}")
            for key in ("centroid_offset_arcsec", "delta_chi2", "flux_ratio"):
                if _shape_of(block.get(key)) != shape:
                    raise SchemaValidationError(f"withheld {arm}.{key} is not shaped {shape}")
        return self


def _shape_of(v: object) -> tuple[int, ...]:
    shape: list[int] = []
    while isinstance(v, list):
        shape.append(len(v))
        v = v[0] if v else None
    return tuple(shape)


class WiseFitsGate2File(_Open):
    """``data/calibrations/wise_fits_gate2_withheld.json`` (v0.1.2, Launch Gate #2)."""

    id: str
    schema_version: Literal[1]
    data_tier: Literal["reproduced_archive"]
    code_git_sha: str
    rng_seed: int
    query_window_utc: tuple[str, str]
    frozen_at_utc: str
    locked_inputs: Gate2LockedInputs
    fetch: Gate2Fetch
    empirical_psf: Gate2PSF
    frozen_screens: Gate2Frozen
    withheld: Gate2Withheld
    source_receipt: SourceReceipt
    content_sha256: str

    @model_validator(mode="after")
    def _contract(self) -> WiseFitsGate2File:
        if not _GIT_SHA_RE.match(self.code_git_sha):
            raise ProvenanceError(
                f"code_git_sha {self.code_git_sha!r} is not a clean 40-hex commit (dirty run?)"
            )
        if not _SHA256_RE.match(self.content_sha256):
            raise ProvenanceError("content_sha256 must be a 64-hex SHA-256")
        cal = set(self.frozen_screens.calibration.get("star_ids", []))
        wh = set(self.withheld.star_ids)
        if not cal or cal & wh:
            raise ProvenanceError("calibration and withheld star sets must be non-empty, disjoint")
        if not set(self.empirical_psf.star_ids) <= cal:
            raise ProvenanceError("empirical PSF uses stars outside the calibration split")
        if list(self.frozen_screens.psf_star_ids) != list(self.empirical_psf.star_ids):
            raise ProvenanceError("frozen screens were calibrated with a different PSF")
        start, end = self.query_window_utc
        if not start <= self.frozen_at_utc <= end:
            raise ProvenanceError("frozen_at_utc lies outside the run window")
        if self.withheld.rng_seed != self.rng_seed:
            raise SchemaValidationError("withheld rng_seed disagrees with rng_seed")
        return self
