"""Pydantic v2 data contracts (SPEC.md Section 4) with astropy unit validation."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Literal

import astropy.units as u
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from fermi_desk.errors import (
    LeakageError,
    MergePolicyError,
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
# Must equal fermi_desk.models.expansion.GEOMETRY_MODES (asserted in tests/test_schemas.py).
GeometryMode = Literal["thin_disk_2d", "thick_disk_3d", "finite_disk_3d"]

# astropy does not parse the word "dimensionless"; SPEC.md uses it, so it is the
# single documented alias. Every other unit string must parse via u.Unit().
_UNIT_ALIASES: dict[str, u.UnitBase] = {"dimensionless": u.dimensionless_unscaled}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# ------------------------------------------------- pointing merge policy (v0.2.1)
# How two surveys sharing a `pointing_group_id` are combined in one run. There are
# exactly two policies; `keep_first` existed before v0.2.1 and is now refused everywhere
# (schema, registry, engine, CLI, API) with `MergePolicyError`.
PointingMergePolicy = Literal["reject_duplicate_pointing_group", "union_targets_by_pointing"]
REJECT_DUPLICATE_POINTING_GROUP = "reject_duplicate_pointing_group"
UNION_TARGETS_BY_POINTING = "union_targets_by_pointing"
POINTING_MERGE_POLICIES: tuple[str, ...] = (
    REJECT_DUPLICATE_POINTING_GROUP,
    UNION_TARGETS_BY_POINTING,
)
KEEP_FIRST_FORBIDDEN_REASON = (
    "pointing merge policy 'keep_first' is forbidden: it resolves a pointing-group "
    "collision by keeping whichever survey happens to load first and discarding the other "
    "catalog wholesale, so the result depends on file load order and silently loses data "
    "(primary targets and background-field stars inside a shared beam are complementary "
    "catalogs, not duplicates). Use 'union_targets_by_pointing' (per-star Gaia DR3 "
    "de-duplication, minimum-offset sighting kept) for a survey that declares its parent "
    "pointing, or keep the default 'reject_duplicate_pointing_group'."
)


def validate_pointing_merge_policy(value: object) -> str:
    """Return a valid policy name; ``keep_first`` and unknown values raise ``MergePolicyError``.

    Never aliases or silently downgrades: a forbidden policy is an error at every boundary.
    """
    if value == "keep_first":
        raise MergePolicyError(KEEP_FIRST_FORBIDDEN_REASON)
    if not isinstance(value, str) or value not in POINTING_MERGE_POLICIES:
        raise MergePolicyError(
            f"unknown pointing merge policy {value!r}; allowed: {POINTING_MERGE_POLICIES}"
        )
    return value


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
    # v0.2.0: geometry modes this parameter applies to. None = every geometry. A
    # parameter scoped to other geometries is never reported in a run's per-hypothesis
    # `parameters` block, so adding finite-disk structure parameters leaves existing
    # thin_disk_2d / thick_disk_3d receipts (and their results_sha256) untouched.
    geometry_modes: list[GeometryMode] | None = None
    # v0.2.0: bounds of the log-uniform prior used when this parameter is marginalised
    # instead of fixed at `default` (lookback-window marginalisation). Subject to the
    # prior-floor rule: the effective lower edge is max(lower, log_floor) and must be > 0.
    marginalization_prior_bounds: tuple[float, float] | None = None

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
        if self.geometry_modes is not None and not self.geometry_modes:
            raise SchemaValidationError(f"{self.id}: geometry_modes must be omitted or non-empty")
        if self.marginalization_prior_bounds is not None:
            if self.prior_type != "log_uniform":
                raise SchemaValidationError(
                    f"{self.id}: marginalization_prior_bounds requires a log_uniform prior"
                )
            m_lo, m_hi = self.marginalization_log_bounds
            if not (lo <= self.marginalization_prior_bounds[0] and m_hi <= hi):
                raise SchemaValidationError(
                    f"{self.id}: marginalization_prior_bounds "
                    f"{self.marginalization_prior_bounds} outside bounds {self.bounds}"
                )
            if not 0.0 < m_lo < m_hi:
                raise SchemaValidationError(
                    f"{self.id}: marginalization prior needs 0 < lower < upper after applying "
                    f"log_floor (got effective [{m_lo}, {m_hi}])"
                )
        return self

    @property
    def astropy_unit(self) -> u.UnitBase:
        return parse_unit(self.unit)

    @property
    def marginalization_log_bounds(self) -> tuple[float, float]:
        """Effective (lower, upper) of the marginalisation prior: max(lower, log_floor)."""
        if self.marginalization_prior_bounds is None:
            raise SchemaValidationError(f"{self.id}: no marginalization_prior_bounds declared")
        lo, hi = self.marginalization_prior_bounds
        return max(lo, self.log_floor) if self.log_floor is not None else lo, hi

    def applies_to(self, geometry: str) -> bool:
        return self.geometry_modes is None or geometry in self.geometry_modes

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
    # v0.2.2: frozen-method blend screens on unsmoothed unWISE W3 coadds of a FRESH control
    # sample (UnwiseGateWithheldFile, reproduced_archive), pinned by file SHA-256. Unset
    # until scripts/run_unwise_gate.py --promote-from has produced the artifact (v0.2.2b).
    unwise_gate_path: str | None = None
    unwise_gate_sha256: str | None = None

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
        if (self.unwise_gate_path is None) != (self.unwise_gate_sha256 is None):
            raise ProvenanceError("unwise_gate_path and unwise_gate_sha256 must be given together")
        if self.unwise_gate_sha256 is not None and not _SHA256_RE.match(self.unwise_gate_sha256):
            raise ProvenanceError("unwise_gate_sha256 must be a 64-hex SHA-256")
        if self.unwise_gate_path is not None and self.sigma_sys_calibration_path is None:
            raise ProvenanceError(
                "unwise_gate requires the locked sigma_sys calibration its fresh sample is "
                "disjoint from"
            )
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
    # v0.2.1: a survey observed inside another survey's pointings (e.g. background-field
    # stars in the BL 692 beams) declares its parent and the union policy. It is never an
    # independent survey: it only enters a run through `union_targets_by_pointing`.
    pointing_merge_policy: PointingMergePolicy = REJECT_DUPLICATE_POINTING_GROUP
    parent_pointing_survey_id: str | None = None
    # Compact promoted background-field summary (BackgroundSummaryFile), pinned by SHA-256.
    background_summary_path: str | None = None
    background_summary_sha256: str | None = None
    # Ruling v0.2.1-R5: SHA-256-pinned pointing corrections applied as a non-destructive
    # overlay when the background cones are fetched and de-duplicated (PointingCorrectionsFile).
    pointing_corrections_path: str | None = None
    pointing_corrections_sha256: str | None = None

    @field_validator("pointing_merge_policy", mode="before")
    @classmethod
    def _check_merge_policy(cls, v: object) -> str:
        return validate_pointing_merge_policy(v)  # MergePolicyError propagates unwrapped

    @property
    def is_pointing_extension(self) -> bool:
        return self.parent_pointing_survey_id is not None

    @model_validator(mode="after")
    def _channel_contract(self) -> Survey:
        if (self.pointing_merge_policy == UNION_TARGETS_BY_POINTING) != self.is_pointing_extension:
            raise SchemaValidationError(
                f"{self.id}: pointing_merge_policy {UNION_TARGETS_BY_POINTING!r} and "
                "parent_pointing_survey_id must be declared together (a survey either is an "
                "extension of a parent pointing or it is not)"
            )
        if self.parent_pointing_survey_id == self.id:
            raise SchemaValidationError(f"{self.id}: a survey cannot be its own parent pointing")
        if (self.background_summary_path is None) != (self.background_summary_sha256 is None):
            raise ProvenanceError(
                f"{self.id}: background_summary_path and background_summary_sha256 must be "
                "given together"
            )
        if self.background_summary_sha256 is not None and not _SHA256_RE.match(
            self.background_summary_sha256
        ):
            raise ProvenanceError(f"{self.id}: background_summary_sha256 must be a 64-hex SHA-256")
        if self.background_summary_path is not None and not self.is_pointing_extension:
            raise SchemaValidationError(
                f"{self.id}: a background summary is only valid on a survey that declares "
                "parent_pointing_survey_id"
            )
        if (self.pointing_corrections_path is None) != (self.pointing_corrections_sha256 is None):
            raise ProvenanceError(
                f"{self.id}: pointing_corrections_path and pointing_corrections_sha256 must be "
                "given together"
            )
        if self.pointing_corrections_sha256 is not None and not _SHA256_RE.match(
            self.pointing_corrections_sha256
        ):
            raise ProvenanceError(
                f"{self.id}: pointing_corrections_sha256 must be a 64-hex SHA-256"
            )
        if self.pointing_corrections_path is not None and self.background_summary_path is None:
            raise SchemaValidationError(
                f"{self.id}: pointing corrections are an overlay on background cones and are only "
                "valid on a survey that pins a background summary"
            )
        if self.channel == "radio" and (
            self.radio_sensitivity is None
            or (self.catalog_path is None and self.background_summary_path is None)
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


# ------------------------------------------ union_targets_by_pointing (v0.2.1)
BL_LBAND_POINTING_GROUP = "bl_gbt_lband_2017"
BACKGROUND_SUMMARY_MAX_BYTES = 5 * 1024 * 1024  # promoted artifact must stay strictly below
BACKGROUND_SUMMARY_PATH = "data/catalogs/bl_gbt_lband_background_summary.json"
# Frozen v0.2.1 protocol of a reproduced_archive background summary.
BL_LBAND_N_POINTINGS = 692
BL_LBAND_BEAM_FWHM_ARCMIN = 8.4
BACKGROUND_D_MAX_PC = 10000.0
BACKGROUND_VALIDATION_SAMPLE = 1000
DistanceMethod = Literal["gaia_dr3_inverse_parallax", "bailer_jones_2021_geo"]
# Ruling v0.2.1-R4: Gaia DR3 source ids are 64-bit integers (up to ~6.9e18, far above
# 2^53), and a JSON number that large is silently rounded by JavaScript consumers. Every
# id written into an artifact from v0.2.1 on is therefore a decimal STRING; a JSON
# number in its place is a schema error (pydantic never coerces int -> str here).
# Locked v0.1.x artifacts keep their integer ids and are stringified at the API boundary.
GaiaSourceIdStr = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]{0,18}$")]


def _iso_utc(value: str, name: str) -> str:
    try:
        datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"{name} {value!r} is not an ISO-8601 timestamp") from exc
    return value


# ------------------------------------ pointing-correction overlay (ruling v0.2.1-R5)
TruthSource = Literal["gaia_dr3", "hip2"]


class PointingCorrection(_Frozen):
    """One corrected pointing: the locked catalog's coordinates and the J2000 truth."""

    target_name: str = Field(min_length=1)
    gaia_dr3_source_id: GaiaSourceIdStr | None
    catalog_ra_deg: float = Field(ge=0, lt=360)
    catalog_dec_deg: float = Field(ge=-90, le=90)
    true_ra_deg: float = Field(ge=0, lt=360)
    true_dec_deg: float = Field(ge=-90, le=90)
    truth_source: TruthSource
    offset_deg: float = Field(gt=0)
    raw_isaacson_dec: str = Field(min_length=1)
    raw_692stars_dec: str = Field(min_length=1)


class PointingControl(_Frozen):
    """A target of the audited band that is genuinely where the catalog says (never corrected)."""

    target_name: str = Field(min_length=1)
    gaia_dr3_source_id: GaiaSourceIdStr | None
    catalog_ra_deg: float = Field(ge=0, lt=360)
    catalog_dec_deg: float = Field(ge=-90, le=90)
    raw_isaacson_dec: str = Field(min_length=1)
    raw_692stars_dec: str = Field(min_length=1)


class SecondaryDiscrepancyRecord(_Frozen):
    target_name: str = Field(min_length=1)
    gaia_dr3_source_id: GaiaSourceIdStr | None
    catalog_dec_deg: float = Field(ge=-90, le=90)
    true_dec_deg: float = Field(ge=-90, le=90)
    truth_source: TruthSource
    raw_isaacson_dec: str = Field(min_length=1)
    raw_692stars_dec: str = Field(min_length=1)


class SecondaryDiscrepancy(_Open):
    """An upstream published-table discrepancy the locked catalog does NOT inherit."""

    id: str = Field(min_length=1)
    data_tier: DataTier
    source: str = Field(min_length=1)
    description: str = Field(min_length=1)
    inherited_by_locked_catalog: Literal[False]
    telescope_pointing_claim: str = Field(min_length=1)
    records: list[SecondaryDiscrepancyRecord] = Field(min_length=1)


class ClaimEvidence(_Open):
    label: str = Field(min_length=1)
    data_tier: DataTier
    source: str = Field(min_length=1)


class TruthSourceSpec(_Open):
    data_tier: DataTier
    source: str = Field(min_length=1)
    queried_at_utc: str = Field(min_length=1)
    epoch: str = Field(min_length=1)


class LockedCatalogRef(_Open):
    path: str = Field(min_length=1)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ProvenanceError("locked_catalog.sha256 must be a 64-hex SHA-256")
        return v


class CatalogBandCensus(_Frozen):
    """Expected target counts of the locked catalog in the flipped band and its mirror."""

    flipped_band_deg: tuple[float, float]  # [lo, hi)
    n_targets_in_flipped_band: int = Field(ge=0)
    mirror_band_deg: tuple[float, float]  # [lo, hi)
    n_targets_in_mirror_band: int = Field(ge=0)

    @model_validator(mode="after")
    def _mirrored(self) -> CatalogBandCensus:
        (lo, hi), (mlo, mhi) = self.flipped_band_deg, self.mirror_band_deg
        if not lo < hi or (mlo, mhi) != (-hi, -lo):
            raise SchemaValidationError(
                "catalog_band_census: mirror_band_deg must be the sign-mirrored flipped band"
            )
        return self


class PointingCorrectionsFile(_Open):
    """``data/adversary_claims/isaacson_2017_dec_sign_flip.yaml`` (ruling v0.2.1-R5).

    The single canonical table of pointing corrections. It never edits the locked primary
    catalog: it is applied as an overlay when background cones are fetched, de-duplicated
    and re-verified, and it is pinned by SHA-256 in the extension survey that uses it.
    """

    id: str = Field(min_length=1)
    ruling: str = Field(min_length=1)
    pointing_group_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    locked_catalog: LockedCatalogRef
    catalog_band_census: CatalogBandCensus
    match_tolerance_arcsec: float = Field(gt=0)
    evidence: list[ClaimEvidence] = Field(min_length=1)
    truth_sources: dict[TruthSource, TruthSourceSpec]
    pointing_corrections: list[PointingCorrection] = Field(min_length=1)
    northern_controls: list[PointingControl] = Field(default_factory=list)
    secondary_discrepancies: list[SecondaryDiscrepancy] = Field(default_factory=list)
    citations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _contract(self) -> PointingCorrectionsFile:
        names = [c.target_name for c in self.pointing_corrections]
        if len(set(names)) != len(names):
            raise SchemaValidationError(f"{self.id}: a target is corrected twice")
        both = set(names) & {c.target_name for c in self.northern_controls}
        if both:
            raise SchemaValidationError(
                f"{self.id}: {sorted(both)} listed both as a correction and as a control"
            )
        ids = [c.gaia_dr3_source_id for c in self.pointing_corrections if c.gaia_dr3_source_id]
        if len(set(ids)) != len(ids):
            raise SchemaValidationError(f"{self.id}: a gaia_dr3_source_id is corrected twice")
        used = {c.truth_source for c in self.pointing_corrections}
        used |= {r.truth_source for d in self.secondary_discrepancies for r in d.records}
        missing = sorted(used - set(self.truth_sources))
        if missing:
            raise ProvenanceError(f"{self.id}: truth sources {missing} are used but not declared")
        return self


class CorrectedPointing(_Frozen):
    """A pointing whose cone was centred on the overlay's J2000 truth, not the catalog."""

    target_name: str = Field(min_length=1)
    pointing_index: int = Field(ge=0)
    catalog_ra_deg: float
    catalog_dec_deg: float
    cone_ra_deg: float  # queried cone centre == primary position in the de-duplication
    cone_dec_deg: float
    truth_source: TruthSource
    # From the cone cache manifest (absent in synthetic rehearsals, required when promoted).
    adql_sha256: str | None = None
    fetched_at_utc: str | None = None
    code_git_sha: str | None = None


class MergeProvenance(_Frozen):
    """Receipt record of one ``union_targets_by_pointing`` merge (star-level audit).

    ``n_raw_targets - n_merged_targets == n_dedup_gaia_id + n_dedup_angular_fallback``
    always holds: every removed row is attributed to exactly one de-duplication key.
    """

    pointing_group_id: str = Field(min_length=1)
    policy: Literal["union_targets_by_pointing"]
    input_surveys: list[str] = Field(min_length=2)
    input_sha256s: list[str] = Field(min_length=2)
    n_raw_targets: int = Field(ge=0)
    n_merged_targets: int = Field(ge=0)
    n_dedup_gaia_id: int = Field(ge=0)
    n_dedup_angular_fallback: int = Field(ge=0)
    n_multi_beam_resolved: int = Field(ge=0)
    # Audit detail beyond the required counters.
    angular_fallback_arcsec: float = Field(gt=0)
    n_raw_by_survey: dict[str, int]
    n_merged_by_role: dict[str, int]
    n_background_absorbed_by_primary: int = Field(ge=0)
    n_angular_links_refused: int = Field(ge=0)
    n_duplicate_gaia_ids_post_merge: Literal[0]
    collision_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def _contract(self) -> MergeProvenance:
        if len(self.input_surveys) != len(self.input_sha256s):
            raise ProvenanceError("merge provenance: one sha256 per input survey is required")
        if len(set(self.input_surveys)) != len(self.input_surveys):
            raise ProvenanceError("merge provenance: input surveys must be distinct")
        if any(not _SHA256_RE.match(s) for s in self.input_sha256s):
            raise ProvenanceError("merge provenance: input_sha256s must be 64-hex SHA-256")
        removed = self.n_raw_targets - self.n_merged_targets
        if removed != self.n_dedup_gaia_id + self.n_dedup_angular_fallback:
            raise ProvenanceError(
                f"merge provenance: {removed} rows removed but {self.n_dedup_gaia_id} + "
                f"{self.n_dedup_angular_fallback} attributed to de-duplication keys"
            )
        if set(self.n_raw_by_survey) != set(self.input_surveys) or (
            sum(self.n_raw_by_survey.values()) != self.n_raw_targets
        ):
            raise ProvenanceError("merge provenance: n_raw_by_survey disagrees with the inputs")
        if sum(self.n_merged_by_role.values()) != self.n_merged_targets:
            raise ProvenanceError("merge provenance: n_merged_by_role disagrees with the total")
        return self


class BackgroundDistanceCuts(_Frozen):
    """Every cut between raw cone sightings and sightings that carry a distance."""

    d_max_pc: float = Field(gt=0)
    inverse_parallax_min_snr: float = Field(gt=0)
    bailer_jones_min_snr_exclusive: float = Field(ge=0)
    n_cone_sightings_raw: int = Field(ge=0)
    n_dropped_invalid_parallax: int = Field(ge=0)
    n_dropped_no_bailer_jones: int = Field(ge=0)
    n_dropped_beyond_d_max: int = Field(ge=0)
    n_dropped_outside_cone_recheck: int = Field(ge=0)
    n_sightings_with_distance: int = Field(ge=0)
    n_sightings_inverse_parallax: int = Field(ge=0)
    n_sightings_bailer_jones_geo: int = Field(ge=0)
    # Unique background stars after union_targets_by_pointing, by distance method.
    n_background_stars: int = Field(ge=0)
    n_background_inverse_parallax: int = Field(ge=0)
    n_background_bailer_jones_geo: int = Field(ge=0)

    @model_validator(mode="after")
    def _arithmetic(self) -> BackgroundDistanceCuts:
        dropped = (
            self.n_dropped_invalid_parallax
            + self.n_dropped_no_bailer_jones
            + self.n_dropped_beyond_d_max
            + self.n_dropped_outside_cone_recheck
        )
        if self.n_cone_sightings_raw - dropped != self.n_sightings_with_distance:
            raise ProvenanceError("distance cuts: raw - dropped != sightings with a distance")
        if (
            self.n_sightings_inverse_parallax + self.n_sightings_bailer_jones_geo
            != self.n_sightings_with_distance
        ):
            raise ProvenanceError("distance cuts: per-method sighting counts do not add up")
        if (
            self.n_background_inverse_parallax + self.n_background_bailer_jones_geo
            != self.n_background_stars
        ):
            raise ProvenanceError("distance cuts: per-method background star counts do not add up")
        if not self.bailer_jones_min_snr_exclusive < self.inverse_parallax_min_snr:
            raise SchemaValidationError(
                "distance cuts: the Bailer-Jones SNR floor must lie below the inverse-parallax one"
            )
        return self


class SuspectedUnmatchedPrimary(_Frozen):
    """A surviving background star that may be an ID-less primary (review only).

    Within 120 arcsec (J2000) of a primary without a Gaia id and within 30% of its
    distance. Ruling v0.2.1-R3: listed, counted and reported - never removed. Each one
    that really is the primary is counted twice (once on-axis, once in a shell).
    """

    primary_target_name: str = Field(min_length=1)
    gaia_dr3_source_id: GaiaSourceIdStr
    separation_arcsec: float = Field(ge=0)
    distance_pc: float = Field(gt=0)
    primary_distance_pc: float = Field(gt=0)
    pointing_index: int = Field(ge=0)  # beam of the star's kept (minimum-offset) sighting
    theta_arcmin: float = Field(ge=0)


class BackgroundDedupAudit(_Frozen):
    """Exact de-duplication counts of the background field against the locked primaries."""

    n_primary_targets: int = Field(ge=1)
    n_primary_with_gaia_id: int = Field(ge=0)
    n_background_sightings_matched_to_primary_gaia_id: int = Field(ge=0)
    n_background_sightings_matched_to_primary_angular: int = Field(ge=0)
    n_duplicate_gaia_ids_vs_primary: Literal[0]
    n_duplicate_gaia_ids_within_background: Literal[0]
    angular_fallback_arcsec: float = Field(gt=0)
    angular_match_epoch: str = Field(min_length=1)
    # Surviving background stars close to an ID-less primary at a compatible distance:
    # reported for human review, never silently removed (they stay in the shells).
    n_suspected_unmatched_primary: int = Field(ge=0)
    suspected_unmatched_primary: list[SuspectedUnmatchedPrimary] = Field(default_factory=list)

    @model_validator(mode="after")
    def _counts(self) -> BackgroundDedupAudit:
        if self.n_primary_with_gaia_id > self.n_primary_targets:
            raise SchemaValidationError("dedup audit: more primary Gaia IDs than primaries")
        if len(self.suspected_unmatched_primary) != self.n_suspected_unmatched_primary:
            raise SchemaValidationError("dedup audit: suspected_unmatched_primary count mismatch")
        return self


class ShellBinningSpec(_Frozen):
    distance_bin_dex: float = Field(gt=0)
    distance_log10_pc_min: float
    n_distance_bins: int = Field(ge=1)
    eirp_bin_dex: float = Field(gt=0)
    eirp_log10_w_min: float
    n_eirp_bins: int = Field(ge=1)
    representative: Literal["geometric_bin_centre"]


class BackgroundShellTable(_Frozen):
    """Per-pointing ``(distance_bin, eirp_min_bin)`` star counts (sparse, sorted cells).

    A cell is addressed by ``(pointing_index, distance_bin, eirp_min_bin)``; its stars are
    evaluated at the beam-centre direction of the pointing and the geometric bin centres.
    Keeping the pointing index makes the table sufficient for all three geometry modes
    (2D needs distance only, 3D the height ``d sin b``, the finite disk the full direction).
    """

    binning: ShellBinningSpec
    pointing_names: list[str] = Field(min_length=1)
    pointing_index: list[int]
    distance_bin: list[int]
    eirp_min_bin: list[int]
    count: list[int]
    n_cells: int = Field(ge=0)
    n_stars: int = Field(ge=0)
    cells_sha256: str

    @model_validator(mode="after")
    def _shape(self) -> BackgroundShellTable:
        cols = (self.pointing_index, self.distance_bin, self.eirp_min_bin, self.count)
        if any(len(c) != self.n_cells for c in cols):
            raise SchemaValidationError("shell table: column lengths disagree with n_cells")
        if len(set(self.pointing_names)) != len(self.pointing_names):
            raise SchemaValidationError("shell table: duplicate pointing names")
        if not _SHA256_RE.match(self.cells_sha256):
            raise ProvenanceError("shell table: cells_sha256 must be a 64-hex SHA-256")
        if sum(self.count) != self.n_stars:
            raise SchemaValidationError("shell table: counts do not sum to n_stars")
        b = self.binning
        limits = (len(self.pointing_names), b.n_distance_bins, b.n_eirp_bins)
        prev: tuple[int, int, int] | None = None
        for key in zip(self.pointing_index, self.distance_bin, self.eirp_min_bin, strict=True):
            if any(not 0 <= k < lim for k, lim in zip(key, limits, strict=True)):
                raise SchemaValidationError(f"shell table: cell {key} outside the declared grid")
            if prev is not None and key <= prev:
                raise SchemaValidationError("shell table: cells must be strictly sorted, unique")
            prev = key
        if any(c < 1 for c in self.count):
            raise SchemaValidationError("shell table: every stored cell needs count >= 1")
        return self


class BackgroundSampleStar(_Frozen):
    gaia_dr3_source_id: GaiaSourceIdStr
    ra_deg: float  # catalog epoch of the source service (Gaia DR3: 2016.0); defines theta
    dec_deg: float
    ra_match_deg: float  # epoch of the primary catalog (J2000.0); used by the angular fallback
    dec_match_deg: float
    pointing_index: int = Field(ge=0)
    theta_arcmin: float = Field(ge=0)
    distance_pc: float = Field(gt=0)
    distance_method: DistanceMethod
    eirp_min_w: float = Field(gt=0)
    distance_bin: int = Field(ge=0)
    eirp_min_bin: int = Field(ge=0)


class PrimaryMatchedSighting(_Frozen):
    """A background sighting absorbed by a primary target (removed before shell binning)."""

    gaia_dr3_source_id: GaiaSourceIdStr
    primary_target_name: str
    match: Literal["gaia_dr3_source_id", "angular_fallback"]
    pointing_index: int = Field(ge=0)
    theta_arcmin: float = Field(ge=0)
    separation_arcsec: float = Field(ge=0)


class BackgroundValidationSample(_Frozen):
    rng_seed: int
    n_background_requested: int = Field(ge=1)
    selection: str = Field(min_length=1)
    background: list[BackgroundSampleStar]
    primary_matched: list[PrimaryMatchedSighting]

    @model_validator(mode="after")
    def _unique(self) -> BackgroundValidationSample:
        ids = [s.gaia_dr3_source_id for s in self.background]
        if len(set(ids)) != len(ids):
            raise SchemaValidationError("validation sample: duplicated gaia_dr3_source_id")
        if len(ids) > self.n_background_requested:
            raise SchemaValidationError("validation sample larger than requested")
        return self


class BackgroundSummaryFile(_Frozen):
    """Compact promoted background-field artifact (``BACKGROUND_SUMMARY_PATH``, v0.2.1).

    ``reproduced_archive`` files come from ``scripts/fetch_wlodarczyk_sroka_288k.py
    --promote-from`` only and must carry a clean commit, complete pointing coverage and
    their source receipt. ``synthetic_proxy`` files are offline test and dry-run fixtures;
    the engine treats that tier as illustrative.
    """

    id: str = Field(min_length=1)
    schema_version: Literal[1]
    data_tier: Literal["reproduced_archive", "synthetic_proxy"]
    description: str = Field(min_length=1)
    methodology_note: str = Field(min_length=1)
    code_git_sha: str
    fetched_at_utc: str
    query_window_utc: tuple[str, str]
    source_service: str = Field(min_length=1)
    adql_template: str = Field(min_length=1)
    pointing_group_id: Literal["bl_gbt_lband_2017"]
    parent_survey_id: str = Field(min_length=1)
    primary_catalog_path: str = Field(min_length=1)
    primary_catalog_sha256: str
    n_pointings: int = Field(ge=1)
    n_pointings_completed: int = Field(ge=0)
    beam_fwhm_arcmin: float = Field(gt=0)
    cone_radius_arcmin: float = Field(gt=0)
    f_min_w_m2: float = Field(gt=0)
    distance_cuts: BackgroundDistanceCuts
    merge_provenance: MergeProvenance
    dedup_audit: BackgroundDedupAudit
    shells: BackgroundShellTable
    validation_sample: BackgroundValidationSample
    spatial_correlation_assumption: str = Field(min_length=1)
    source_receipt: SourceReceipt | None = None
    # Self-description without the (gitignored) receipt. `query_window_utc` /
    # `fetched_at_utc` are when cones were actually fetched from the archive;
    # `assembled_at_utc` is when this summary was built from the cone cache (later, for
    # a resumed or cache-only run). The manifest digest pins the raw cone bytes and the
    # commit list names every code version that queried the archive.
    assembled_at_utc: str | None = None
    cone_cache_manifest_sha256: str | None = None
    cone_cache_code_git_shas: list[str] = Field(default_factory=list)
    # Ruling v0.2.1-R5: the pointing-correction overlay this summary was built with (absent
    # in summaries that predate it). The locked primary catalog is never edited; these
    # pointings had their cone centre and their de-duplication position replaced.
    pointing_corrections_path: str | None = None
    pointing_corrections_sha256: str | None = None
    n_corrected_pointings: int = Field(default=0, ge=0)
    corrected_target_names: list[str] = Field(default_factory=list)
    corrected_pointings: list[CorrectedPointing] = Field(default_factory=list)
    content_sha256: str

    @model_validator(mode="after")
    def _overlay_contract(self) -> BackgroundSummaryFile:
        cp = self.corrected_pointings
        if (self.pointing_corrections_path is None) != (self.pointing_corrections_sha256 is None):
            raise ProvenanceError(
                "pointing_corrections_path and pointing_corrections_sha256 must be given together"
            )
        if self.pointing_corrections_sha256 is None:
            if self.n_corrected_pointings or self.corrected_target_names or cp:
                raise ProvenanceError("corrected pointings recorded without a pinned overlay")
            return self
        if not _SHA256_RE.match(self.pointing_corrections_sha256):
            raise ProvenanceError("pointing_corrections_sha256 must be a 64-hex SHA-256")
        names = [c.target_name for c in cp]
        if self.n_corrected_pointings != len(cp) or self.corrected_target_names != names:
            raise ProvenanceError(
                "n_corrected_pointings / corrected_target_names disagree with corrected_pointings"
            )
        if len(set(names)) != len(names):
            raise ProvenanceError("a pointing is listed as corrected twice")
        pointing_names = self.shells.pointing_names
        for c in cp:
            if c.pointing_index >= len(pointing_names) or (
                pointing_names[c.pointing_index] != c.target_name
            ):
                raise ProvenanceError(
                    f"corrected pointing {c.target_name!r} does not address its own beam"
                )
            if self.data_tier == "reproduced_archive":
                if c.adql_sha256 is None or not _SHA256_RE.match(c.adql_sha256):
                    raise ProvenanceError(
                        f"corrected pointing {c.target_name!r} lacks the adql_sha256 of its cone"
                    )
                if c.code_git_sha is None or not _GIT_SHA_RE.match(c.code_git_sha):
                    raise ProvenanceError(
                        f"corrected cone of {c.target_name!r} was not fetched from a clean commit"
                    )
                _iso_utc(c.fetched_at_utc or "", f"corrected_pointings[{c.target_name}]")
        return self

    @model_validator(mode="after")
    def _contract(self) -> BackgroundSummaryFile:
        if not _SHA256_RE.match(self.content_sha256):
            raise ProvenanceError("content_sha256 must be a 64-hex SHA-256")
        if not _SHA256_RE.match(self.primary_catalog_sha256):
            raise ProvenanceError("primary_catalog_sha256 must be a 64-hex SHA-256")
        _iso_utc(self.fetched_at_utc, "fetched_at_utc")
        for t in self.query_window_utc:
            _iso_utc(t, "query_window_utc")
        if abs(self.cone_radius_arcmin - self.beam_fwhm_arcmin / 2.0) > 1e-9:
            raise SchemaValidationError("cone_radius_arcmin must equal beam_fwhm_arcmin / 2")
        sh, mp, dc, da = self.shells, self.merge_provenance, self.distance_cuts, self.dedup_audit
        if len(sh.pointing_names) != self.n_pointings:
            raise SchemaValidationError("n_pointings disagrees with shells.pointing_names")
        if self.n_pointings_completed > self.n_pointings:
            raise SchemaValidationError("n_pointings_completed exceeds n_pointings")
        if mp.pointing_group_id != self.pointing_group_id:
            raise ProvenanceError("merge provenance covers a different pointing group")
        if mp.input_surveys[0] != self.parent_survey_id or (
            mp.input_sha256s[0] != self.primary_catalog_sha256
        ):
            raise ProvenanceError("merge provenance was not computed against the parent catalog")
        if sh.n_stars != dc.n_background_stars or (
            mp.n_merged_by_role.get("background", 0) != sh.n_stars
        ):
            raise ProvenanceError("shell star count disagrees with the merge / distance-cut counts")
        if mp.n_merged_by_role.get("primary", 0) != da.n_primary_targets:
            raise ProvenanceError("merge kept a different number of primaries than the catalog")
        vs = self.validation_sample
        matched = {
            k: sum(m.match == k for m in vs.primary_matched)
            for k in ("gaia_dr3_source_id", "angular_fallback")
        }
        if matched["gaia_dr3_source_id"] != da.n_background_sightings_matched_to_primary_gaia_id:
            raise ProvenanceError("primary-matched Gaia-ID sightings disagree with the dedup audit")
        if matched["angular_fallback"] != da.n_background_sightings_matched_to_primary_angular:
            raise ProvenanceError("primary-matched angular sightings disagree with the dedup audit")
        if len(vs.primary_matched) != mp.n_background_absorbed_by_primary:
            raise ProvenanceError("validation sample must list every primary-matched sighting")
        if len(vs.background) != min(vs.n_background_requested, sh.n_stars):
            raise ProvenanceError("validation sample size != min(requested, background stars)")
        if self.data_tier == "reproduced_archive":
            if not _GIT_SHA_RE.match(self.code_git_sha):
                raise ProvenanceError(
                    f"code_git_sha {self.code_git_sha!r} is not a clean 40-hex commit (dirty run?)"
                )
            if self.n_pointings_completed != self.n_pointings:
                raise ProvenanceError(
                    f"incomplete pointing coverage: {self.n_pointings_completed} of "
                    f"{self.n_pointings} pointings"
                )
            if self.source_receipt is None:
                raise ProvenanceError("reproduced_archive summary needs its source_receipt")
            if self.assembled_at_utc is None:
                raise ProvenanceError("reproduced_archive summary needs assembled_at_utc")
            _iso_utc(self.assembled_at_utc, "assembled_at_utc")
            first, last = (datetime.fromisoformat(t) for t in self.query_window_utc)
            if not first <= last <= datetime.fromisoformat(self.assembled_at_utc):
                raise ProvenanceError(
                    "archive query window must be ordered and precede assembled_at_utc"
                )
            if self.fetched_at_utc != self.query_window_utc[1]:
                raise ProvenanceError("fetched_at_utc must be the last archive query time")
            if self.cone_cache_manifest_sha256 is None or not _SHA256_RE.match(
                self.cone_cache_manifest_sha256
            ):
                raise ProvenanceError(
                    "reproduced_archive summary needs cone_cache_manifest_sha256 (64-hex)"
                )
            if mp.input_sha256s[1] != self.cone_cache_manifest_sha256:
                raise ProvenanceError(
                    "merge provenance was not computed from the recorded cone cache"
                )
            if not self.cone_cache_code_git_shas or any(
                not _GIT_SHA_RE.match(s) for s in self.cone_cache_code_git_shas
            ):
                raise ProvenanceError(
                    "cone cache holds pointings fetched from a non-clean tree: "
                    f"{self.cone_cache_code_git_shas}"
                )
            canonical = {
                "n_pointings": (self.n_pointings, BL_LBAND_N_POINTINGS),
                "beam_fwhm_arcmin": (self.beam_fwhm_arcmin, BL_LBAND_BEAM_FWHM_ARCMIN),
                "distance_cuts.d_max_pc": (dc.d_max_pc, BACKGROUND_D_MAX_PC),
                "distance_cuts.inverse_parallax_min_snr": (dc.inverse_parallax_min_snr, 5.0),
                "distance_cuts.bailer_jones_min_snr_exclusive": (
                    dc.bailer_jones_min_snr_exclusive,
                    1.0,
                ),
            }
            off = {k: got for k, (got, want) in canonical.items() if got != want}
            if off:
                raise ProvenanceError(
                    f"reproduced_archive summary departs from the frozen v0.2.1 protocol: {off}"
                )
            if vs.n_background_requested < BACKGROUND_VALIDATION_SAMPLE:
                raise ProvenanceError(
                    f"reproduced_archive summary needs a validation sample of at least "
                    f"{BACKGROUND_VALIDATION_SAMPLE} background stars"
                )
        return self


# --------------------------- unsmoothed unWISE W3 gate on a fresh split (v0.2.2)
UNWISE_GATE_PATH = "data/calibrations/unwise_w3_gate_withheld.json"
UNWISE_GATE_MAX_BYTES = 5 * 1024 * 1024  # promoted artifact must stay strictly below
UNWISE_GATE_MIN_SPLIT = 60  # evaluated stars required in EACH half of a promoted run
UNWISE_GATE_MIN_ABS_B_DEG = 30.0
UNWISE_GATE_MIN_LOCKED_SEPARATION_DEG = 15.0
UNWISE_PIXEL_SCALE_ARCSEC = 2.75
# Must equal fermi_desk.models.unwise_fits.SCREENS / STATISTICS (asserted in the tests).
UNWISE_SCREENS: tuple[str, ...] = (
    "centroid_locked",
    "centroid_fresh",
    "image_scalar_chi2",
    "image_gls_chi2",
    "image_scalar_conformal",
    "image_gls_conformal",
    "naive_flux_excess",
)
UNWISE_STATISTICS: tuple[str, ...] = (
    "centroid_offset_arcsec",
    "delta_chi2_scalar",
    "delta_chi2_gls",
    "flux_ratio",
)
Split = Literal["calibration", "withheld"]
Pair = tuple[float, float]


def _utc(value: str, name: str) -> datetime:
    """An ISO-8601 timestamp as an aware UTC datetime (naive values are read as UTC)."""
    t = datetime.fromisoformat(_iso_utc(value, name))
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


class UnwiseLockedInputs(_Frozen):
    """The locked Gate #1 file the fresh sample was checked against (never modified)."""

    sigma_sys_calibration_path: str = Field(min_length=1)
    sigma_sys_calibration_sha256: str
    sigma_sys_arcsec: float = Field(gt=0)
    n_locked_stars: int = Field(ge=1)
    n_locked_patches: int = Field(ge=1)

    @field_validator("sigma_sys_calibration_sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if not _SHA256_RE.match(v):
            raise ProvenanceError("locked sigma_sys_calibration_sha256 must be a 64-hex SHA-256")
        return v


class UnwisePatch(_Frozen):
    patch_id: str = Field(min_length=1)
    l_deg: float
    b_deg: float
    ra_deg: float
    dec_deg: float
    radius_deg: float = Field(gt=0)
    nearest_locked_patch_id: str = Field(min_length=1)
    nearest_locked_patch_separation_deg: float = Field(ge=0)
    gaia_adql: str = Field(min_length=1)
    allwise_adql: str = Field(min_length=1)
    cut_counts: dict[str, int]


class UnwiseControl(_Frozen):
    """One fresh control star. Its Gaia id is a decimal string (ruling v0.2.1-R4)."""

    gaia_dr3_source_id: GaiaSourceIdStr
    allwise: str = Field(min_length=1)
    coadd_id: str | None = None
    patch_id: str = Field(min_length=1)
    split: Split
    l_deg: float
    b_deg: float
    gaia_ra_epoch_wise_deg: float
    gaia_dec_epoch_wise_deg: float
    phot_g_mean_mag: float
    bp_rp: float
    parallax_mas: float = Field(gt=0)
    ruwe: float = Field(lt=1.4)
    w3snr: float = Field(ge=8.0, le=20.0)
    w3mpro: float
    cc_flags: Literal["0000"]
    ext_flg: Literal[0]
    n_gaia_neighbours: Literal[0]


class UnwiseSample(_Frozen):
    patches: list[UnwisePatch] = Field(min_length=1)
    controls: list[UnwiseControl] = Field(min_length=2)
    selection: dict
    cut_counts_total: dict[str, int]


class UnwiseCutoutFile(_Frozen):
    gaia_dr3_source_id: GaiaSourceIdStr
    split: Split
    url: str = Field(min_length=1)
    sha256: str
    bytes: int = Field(ge=1)
    fetched_at_utc: str
    cached: bool
    tile: str = Field(min_length=1)
    shape: tuple[int, int]
    gaia_xy: Pair
    pixel_scale_measured_arcsec: float = Field(gt=0)
    code_git_sha: str

    @model_validator(mode="after")
    def _contract(self) -> UnwiseCutoutFile:
        if not _SHA256_RE.match(self.sha256):
            raise ProvenanceError("every fetched cutout needs a 64-hex sha256")
        _iso_utc(self.fetched_at_utc, f"fetch.files[{self.gaia_dr3_source_id}].fetched_at_utc")
        return self


class UnwiseFetch(_Frozen):
    service: str = Field(min_length=1)
    url_template: str = Field(min_length=1)
    cutout_size_pix: int = Field(ge=1)
    n_requested: int = Field(ge=1)
    n_ok: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    files: list[UnwiseCutoutFile]
    failures: list[dict]
    cache_manifest_sha256: str

    @model_validator(mode="after")
    def _counts(self) -> UnwiseFetch:
        if self.n_ok + self.n_failed != self.n_requested:
            raise SchemaValidationError("fetch counts do not add up")
        if len(self.files) != self.n_ok or len(self.failures) != self.n_failed:
            raise SchemaValidationError("fetch.files / fetch.failures disagree with the counts")
        if not _SHA256_RE.match(self.cache_manifest_sha256):
            raise ProvenanceError("fetch.cache_manifest_sha256 must be a 64-hex SHA-256")
        return self


class UnwisePSF(_Frozen):
    stamp: list[list[float]] = Field(min_length=3)
    stamp_half_pix: int = Field(ge=1)
    pixel_scale_arcsec: float = Field(gt=0)
    n_stars: int = Field(ge=5)
    star_ids: list[GaiaSourceIdStr]
    excluded: list[dict]
    fwhm_gaussian_arcsec: float = Field(gt=0)
    fwhm_halfmax_arcsec: float = Field(gt=0)
    halfmax_oversample: int = Field(ge=1)
    nominal_fwhm_arcsec: float = Field(gt=0)
    construction: str

    @model_validator(mode="after")
    def _shape(self) -> UnwisePSF:
        n = 2 * self.stamp_half_pix + 1
        if len(self.stamp) != n or any(len(r) != n for r in self.stamp):
            raise SchemaValidationError("PSF stamp shape disagrees with stamp_half_pix")
        if len(self.star_ids) != self.n_stars:
            raise SchemaValidationError("PSF n_stars disagrees with star_ids")
        return self


class UnwiseNoiseModel(_Frozen):
    """Frozen stationary pixel-noise covariance (units of the per-cutout scalar variance)."""

    n_torus: int = Field(ge=4)
    max_lag_pix: int = Field(ge=1)
    torus_covariance: list[list[float]]
    raw_lag_covariance: list[list[float]]
    n_pairs: list[list[float]]
    n_cutouts: int = Field(ge=1)
    star_ids: list[GaiaSourceIdStr]
    shrinkage: float = Field(gt=0, lt=1)
    power_floor: float = Field(gt=0)
    clip_sigma: float = Field(gt=0)
    taper_flat_fraction: float = Field(ge=0, le=1)
    variance_inflation: float = Field(gt=0)
    correlation_lag1: Pair
    raw_correlation_lag1: Pair
    minimum_eigenvalue_bound: float
    construction: str

    @model_validator(mode="after")
    def _contract(self) -> UnwiseNoiseModel:
        n, m = self.n_torus, 2 * self.max_lag_pix + 1
        if _shape_of(self.torus_covariance) != (n, n):
            raise SchemaValidationError("torus_covariance is not n_torus x n_torus")
        if _shape_of(self.raw_lag_covariance) != (m, m) or _shape_of(self.n_pairs) != (m, m):
            raise SchemaValidationError("lag tables are not (2 max_lag + 1) square")
        if n < 4 * self.max_lag_pix + 2:
            raise SchemaValidationError("torus is too small to hold every lag unaliased")
        if len(self.star_ids) != self.n_cutouts:
            raise SchemaValidationError("noise-model n_cutouts disagrees with star_ids")
        if self.variance_inflation != self.torus_covariance[0][0]:
            raise SchemaValidationError("variance_inflation must equal torus_covariance[0][0]")
        # Strict positive definiteness: the smallest eigenvalue of the circulant covariance
        # bounds every per-star matrix from below, and shrinkage guarantees shrinkage * V(0).
        floor = self.shrinkage * self.variance_inflation
        if not self.minimum_eigenvalue_bound >= floor * (1.0 - 1e-9):
            raise ProvenanceError(
                f"pixel covariance is not strictly positive definite: minimum eigenvalue "
                f"{self.minimum_eigenvalue_bound} < shrinkage * V(0) = {floor}"
            )
        return self


class UnwiseCentroidScreen(_Frozen):
    role: str
    statistic: Literal["centroid_offset_arcsec"]
    sigma_sys_arcsec: float = Field(ge=0)
    sigma_sys_source: Literal["locked", "fresh_calibration_half"]
    sigma_stat_rule: str
    target_false_alarm: float = Field(gt=0, lt=1)
    multiplier: float = Field(gt=0)
    threshold_arcsec_at_calibration_median_snr: float = Field(gt=0)
    calibration_median_snr: float = Field(gt=0)


class UnwiseChi2Screen(_Frozen):
    role: str
    statistic: Literal["delta_chi2_scalar", "delta_chi2_gls"]
    calibration_scores: str
    chi2_dof: Literal[3]
    scale_rule: str
    target_false_alarm: float = Field(gt=0, lt=1)
    delta_chi2_scale: float = Field(gt=0)
    threshold: float = Field(gt=0)


class UnwiseConformalScreen(_Frozen):
    """Distribution-free order-statistic threshold with its exact finite-sample level.

    The threshold is the ``(k+1)``-th largest of ``n_calibration`` leave-one-out calibration
    nulls. For exchangeable continuous scores a new null exceeds it with probability
    ``alpha_guaranteed = (k + 1) / (n_calibration + 1)`` exactly, and ``k`` is the largest
    integer keeping that at or below ``alpha_nominal``.
    """

    role: str
    statistic: Literal["delta_chi2_scalar", "delta_chi2_gls"]
    calibration_scores: str
    rule: str
    alpha_nominal: float = Field(gt=0, lt=1)
    n_calibration: int = Field(ge=1)
    k_exceedances: int = Field(ge=0)
    order_statistic_rank_from_top: int = Field(ge=1)
    alpha_guaranteed: float = Field(gt=0, lt=1)
    threshold: float
    n_calibration_strictly_above: int = Field(ge=0)

    @model_validator(mode="after")
    def _finite_sample_level(self) -> UnwiseConformalScreen:
        k, n = self.k_exceedances, self.n_calibration
        if self.order_statistic_rank_from_top != k + 1 or k + 1 > n:
            raise SchemaValidationError("order_statistic_rank_from_top must equal k + 1 <= n_cal")
        if abs(self.alpha_guaranteed - (k + 1) / (n + 1)) > 1e-12:
            raise ProvenanceError(
                f"alpha_guaranteed {self.alpha_guaranteed} != (k + 1) / (n_cal + 1) = "
                f"{(k + 1) / (n + 1)}"
            )
        if self.alpha_guaranteed > self.alpha_nominal + 1e-9:
            raise ProvenanceError("conformal level exceeds the nominal alpha it was frozen for")
        if (k + 2) / (n + 1) <= self.alpha_nominal + 1e-9:
            raise ProvenanceError("k is not the largest order statistic attaining alpha_nominal")
        if self.n_calibration_strictly_above > k:
            raise ProvenanceError("more calibration nulls exceed the threshold than k allows")
        return self


class UnwiseNaiveScreen(_Frozen):
    role: str
    statistic: Literal["flux_ratio"]
    rule: str


class UnwiseScreens(_Frozen):
    """Every frozen screen of rule version ``unwise-gate-v1`` (all seven are required)."""

    centroid_locked: UnwiseCentroidScreen
    centroid_fresh: UnwiseCentroidScreen
    image_scalar_chi2: UnwiseChi2Screen
    image_gls_chi2: UnwiseChi2Screen
    image_scalar_conformal: UnwiseConformalScreen
    image_gls_conformal: UnwiseConformalScreen
    naive_flux_excess: UnwiseNaiveScreen

    @model_validator(mode="after")
    def _statistics(self) -> UnwiseScreens:
        want = {
            "image_scalar_chi2": "delta_chi2_scalar",
            "image_gls_chi2": "delta_chi2_gls",
            "image_scalar_conformal": "delta_chi2_scalar",
            "image_gls_conformal": "delta_chi2_gls",
        }
        for name, stat in want.items():
            if getattr(self, name).statistic != stat:
                raise SchemaValidationError(f"screen {name} must threshold {stat}")
        if self.centroid_locked.sigma_sys_source != "locked" or (
            self.centroid_fresh.sigma_sys_source != "fresh_calibration_half"
        ):
            raise SchemaValidationError("centroid screens carry the wrong sigma_sys source")
        return self


class UnwiseCalibrationNulls(_Frozen):
    """Per-star calibration statistics every threshold is re-derived from."""

    star_ids: list[GaiaSourceIdStr] = Field(min_length=1)
    patch_ids: list[str]
    w3snr: list[float]
    centroid_offset_arcsec: list[float]
    delta_chi2_scalar: list[float]
    delta_chi2_gls: list[float]
    delta_chi2_scalar_loo: list[float]
    delta_chi2_gls_loo: list[float]
    fetched_at_utc: list[str]

    @model_validator(mode="after")
    def _lengths(self) -> UnwiseCalibrationNulls:
        n = len(self.star_ids)
        for name in type(self).model_fields:
            if len(getattr(self, name)) != n:
                raise SchemaValidationError(f"calibration.{name} disagrees with star_ids")
        if len(set(self.star_ids)) != n:
            raise SchemaValidationError("duplicate calibration star")
        return self


class UnwiseFrozenScreens(_Frozen):
    target_false_alarm: float = Field(gt=0, lt=1)
    conformal_alpha_nominal: float = Field(gt=0, lt=1)
    locked_sigma_sys_arcsec: float = Field(gt=0)
    fresh_sigma_sys: dict
    screens: UnwiseScreens
    snr_eff_rule: str
    calibration: UnwiseCalibrationNulls
    calibration_false_alarm_in_sample: dict


class UnwiseRate(_Frozen):
    k: int = Field(ge=0)
    n: int = Field(ge=1)
    rate: float = Field(ge=0, le=1)
    clopper_pearson_95: Pair


class UnwiseScreenSummary(_Frozen):
    """Withheld result of one screen: null false alarms, Arm A and Arm B tables."""

    role: str
    n_stars: int = Field(ge=1)
    null_false_alarm: UnwiseRate
    arm_a_blend_recovery: list[list[float]]
    arm_a_k: list[list[int]]
    arm_a_clopper_pearson_95: list[list[Pair]]
    arm_b_false_rejection: list[float]
    arm_b_k: list[int]
    arm_b_clopper_pearson_95: list[Pair]
    arm_b_false_rejection_pooled: UnwiseRate
    blend_survival_within_3p25_arcsec: float = Field(ge=0, le=1)
    centered_excess_retained_pooled: float = Field(ge=0, le=1)


class UnwiseSummary(_Frozen):
    survival_max_separation_arcsec: float = Field(gt=0)
    survival_weighting: str
    flux_fractions: list[float]
    screens: dict[str, UnwiseScreenSummary]

    @model_validator(mode="after")
    def _all_screens(self) -> UnwiseSummary:
        if set(self.screens) != set(UNWISE_SCREENS):
            raise SchemaValidationError(f"summary must cover exactly {UNWISE_SCREENS}")
        return self


class UnwiseWithheld(_Frozen):
    rng_seed: int
    rng: str
    n_stars: int = Field(ge=1)
    star_ids: list[GaiaSourceIdStr]
    patch_ids: list[str]
    fetched_at_utc: list[str]
    w3snr: list[float]
    reference_flux: list[float]
    separations_arcsec: list[float] = Field(min_length=1)
    flux_fractions: list[float] = Field(min_length=1)
    injection: dict
    statistics: dict[str, dict[str, list]]
    excluded: list[dict]
    summary: UnwiseSummary

    @model_validator(mode="after")
    def _shapes(self) -> UnwiseWithheld:
        n, ns, nf = self.n_stars, len(self.separations_arcsec), len(self.flux_fractions)
        for name in ("star_ids", "patch_ids", "fetched_at_utc", "w3snr", "reference_flux"):
            if len(getattr(self, name)) != n:
                raise SchemaValidationError(f"withheld.{name} disagrees with n_stars")
        if len(set(self.star_ids)) != n:
            raise SchemaValidationError("duplicate withheld star")
        want = {"null": (n,), "arm_a": (n, ns, nf), "arm_b": (n, nf)}
        if set(self.statistics) != set(want):
            raise SchemaValidationError(f"withheld statistics must hold exactly {sorted(want)}")
        for arm, shape in want.items():
            block = self.statistics[arm]
            if set(block) != set(UNWISE_STATISTICS):
                raise SchemaValidationError(f"withheld {arm} must hold {UNWISE_STATISTICS}")
            for key in UNWISE_STATISTICS:
                if _shape_of(block[key]) != shape:
                    raise SchemaValidationError(f"withheld {arm}.{key} is not shaped {shape}")
        if self.summary.flux_fractions != self.flux_fractions:
            raise SchemaValidationError("summary flux fractions disagree with the injection grid")
        for name, sc in self.summary.screens.items():
            if sc.n_stars != n or sc.null_false_alarm.n != n:
                raise SchemaValidationError(f"summary of {name} covers a different star count")
            if _shape_of(sc.arm_a_blend_recovery) != (ns, nf) or len(sc.arm_b_k) != nf:
                raise SchemaValidationError(f"summary tables of {name} are mis-shaped")
        return self


class UnwiseGateWithheldFile(_Frozen):
    """``UNWISE_GATE_PATH``: frozen-method blend screens on unsmoothed unWISE W3 (v0.2.2).

    ``reproduced_archive`` files come from ``scripts/run_unwise_gate.py --promote-from``
    only: clean commit, source receipt, at least ``UNWISE_GATE_MIN_SPLIT`` evaluated stars
    in each half, native 2.75 arcsec pixels. ``synthetic_proxy`` files are the in-memory
    dry-run rehearsal and offline test fixtures; they can never be promoted or pinned.

    Zero leakage is part of the schema: every calibration cutout was fetched at or before
    ``frozen_at_utc`` and every withheld cutout strictly after it (``LeakageError``
    otherwise), the PSF and covariance stars are calibration stars, and the two halves share
    no star. ``disjoint_from_gate1_sha256`` is the SHA-256 of the locked Gate #1 control file
    the fresh sample was checked against (re-checked star by star by ``fermi-desk validate``).
    """

    id: str = Field(min_length=1)
    schema_version: Literal[1]
    data_tier: Literal["reproduced_archive", "synthetic_proxy"]
    description: str = Field(min_length=1)
    rule_version: str = Field(min_length=1)
    code_git_sha: str
    rng_seed: int
    query_window_utc: tuple[str, str]
    frozen_at_utc: str
    frozen_state_sha256: str
    frozen_before_withheld_fetch: bool
    disjoint_from_gate1_sha256: str
    locked_inputs: UnwiseLockedInputs
    sample: UnwiseSample
    endpoints: dict
    fetch: UnwiseFetch
    geometry: dict
    empirical_psf: UnwisePSF
    noise_model: UnwiseNoiseModel
    frozen_screens: UnwiseFrozenScreens
    withheld: UnwiseWithheld
    config: dict
    caveats: list[str]
    source_receipt: SourceReceipt | None = None
    content_sha256: str

    @property
    def n_calibration(self) -> int:
        return len(self.frozen_screens.calibration.star_ids)

    @property
    def n_withheld(self) -> int:
        return self.withheld.n_stars

    @model_validator(mode="after")
    def _zero_leakage(self) -> UnwiseGateWithheldFile:
        cal, wh = self.frozen_screens.calibration, self.withheld
        cal_ids, wh_ids = set(cal.star_ids), set(wh.star_ids)
        if cal_ids & wh_ids:
            raise LeakageError("calibration and withheld star sets overlap")
        if not set(self.empirical_psf.star_ids) <= cal_ids:
            raise LeakageError("empirical PSF uses stars outside the calibration split")
        if not set(self.noise_model.star_ids) <= cal_ids:
            raise LeakageError("pixel covariance uses stars outside the calibration split")
        frozen = _utc(self.frozen_at_utc, "frozen_at_utc")
        start, end = (_utc(t, "query_window_utc") for t in self.query_window_utc)
        if not start <= frozen <= end:
            raise ProvenanceError("frozen_at_utc lies outside the run window")
        late = [t for t in cal.fetched_at_utc if _utc(t, "calibration fetched_at_utc") > frozen]
        if late:
            raise LeakageError(
                f"{len(late)} calibration cutout(s) were fetched after frozen_at_utc: the "
                "recorded freeze is not the freeze"
            )
        early = [t for t in wh.fetched_at_utc if _utc(t, "withheld fetched_at_utc") <= frozen]
        if early:
            raise LeakageError(
                f"{len(early)} withheld cutout(s) were fetched at or before frozen_at_utc "
                f"({min(early)} <= {self.frozen_at_utc})"
            )
        if not self.frozen_before_withheld_fetch:
            raise LeakageError("frozen_before_withheld_fetch is false: the freeze did not hold")
        return self

    @model_validator(mode="after")
    def _contract(self) -> UnwiseGateWithheldFile:
        for name in ("content_sha256", "frozen_state_sha256", "disjoint_from_gate1_sha256"):
            if not _SHA256_RE.match(getattr(self, name)):
                raise ProvenanceError(f"{name} must be a 64-hex SHA-256")
        li = self.locked_inputs
        if self.disjoint_from_gate1_sha256 != li.sigma_sys_calibration_sha256:
            raise ProvenanceError(
                "disjoint_from_gate1_sha256 must be the SHA-256 of the locked sigma_sys "
                "calibration named in locked_inputs"
            )
        fz, wh, sample = self.frozen_screens, self.withheld, self.sample
        if fz.locked_sigma_sys_arcsec != li.sigma_sys_arcsec or (
            fz.screens.centroid_locked.sigma_sys_arcsec != li.sigma_sys_arcsec
        ):
            raise ProvenanceError("the locked sigma_sys was not carried into the frozen screens")
        if wh.rng_seed != self.rng_seed:
            raise SchemaValidationError("withheld rng_seed disagrees with rng_seed")
        n_cal = self.n_calibration
        for name in ("image_scalar_conformal", "image_gls_conformal"):
            sc = getattr(fz.screens, name)
            if sc.n_calibration != n_cal or sc.alpha_nominal != fz.conformal_alpha_nominal:
                raise ProvenanceError(f"{name} was not frozen on the stored calibration nulls")
        if self.empirical_psf.pixel_scale_arcsec != self.geometry.get("pixel_scale_arcsec"):
            raise SchemaValidationError("geometry and PSF disagree on the pixel scale")
        # The fresh sample: declared high-latitude patches, a 50/50 split, every evaluated
        # star a member with the right label.
        patches = {p.patch_id: p for p in sample.patches}
        if len(patches) != len(sample.patches):
            raise SchemaValidationError("duplicate patch id")
        for p in sample.patches:
            if abs(p.b_deg) - p.radius_deg < UNWISE_GATE_MIN_ABS_B_DEG:
                raise SchemaValidationError(
                    f"patch {p.patch_id} cone reaches |b| < {UNWISE_GATE_MIN_ABS_B_DEG:g} deg"
                )
            if p.nearest_locked_patch_separation_deg < UNWISE_GATE_MIN_LOCKED_SEPARATION_DEG:
                raise LeakageError(
                    f"patch {p.patch_id} lies {p.nearest_locked_patch_separation_deg:.2f} deg "
                    f"from a locked Gate #1 patch (< {UNWISE_GATE_MIN_LOCKED_SEPARATION_DEG:g})"
                )
        split = {c.gaia_dr3_source_id: c.split for c in sample.controls}
        if len(split) != len(sample.controls):
            raise SchemaValidationError("duplicate control star")
        if any(
            c.patch_id not in patches or abs(c.b_deg) <= UNWISE_GATE_MIN_ABS_B_DEG
            for c in sample.controls
        ):
            raise SchemaValidationError("control outside a declared high-latitude patch")
        n_first = sum(s == "calibration" for s in split.values())
        if abs(2 * n_first - len(split)) > 1:
            raise SchemaValidationError("calibration / withheld split is not 50/50")
        for label, ids in (
            ("calibration", fz.calibration.star_ids),
            ("withheld", wh.star_ids),
        ):
            wrong = [s for s in ids if split.get(s) != label]
            if wrong:
                raise LeakageError(
                    f"{len(wrong)} {label} star(s) are not {label} controls of the sample "
                    f"(e.g. {wrong[0]})"
                )
        # Every evaluated star has its cutout record, with the same split and fetch time.
        files = {f.gaia_dr3_source_id: f for f in self.fetch.files}
        if len(files) != len(self.fetch.files) or self.fetch.n_requested != len(split):
            raise SchemaValidationError("fetch records disagree with the sample")
        for ids, times in (
            (fz.calibration.star_ids, fz.calibration.fetched_at_utc),
            (wh.star_ids, wh.fetched_at_utc),
        ):
            for sid, t in zip(ids, times, strict=True):
                f = files.get(sid)
                if f is None or f.split != split[sid]:
                    raise ProvenanceError(f"star {sid} has no cutout record of its own split")
                if _utc(f.fetched_at_utc, "fetch") != _utc(t, "fetched_at_utc"):
                    raise ProvenanceError(f"star {sid}: fetch time disagrees with its record")
        if self.data_tier == "reproduced_archive":
            if not _GIT_SHA_RE.match(self.code_git_sha):
                raise ProvenanceError(
                    f"code_git_sha {self.code_git_sha!r} is not a clean 40-hex commit (dirty run?)"
                )
            shas = sorted({f.code_git_sha for f in self.fetch.files})
            if any(not _GIT_SHA_RE.match(s) for s in shas):
                raise ProvenanceError(f"cutouts were fetched from a non-clean tree: {shas}")
            if self.source_receipt is None:
                raise ProvenanceError("reproduced_archive unWISE gate needs its source_receipt")
            if n_cal < UNWISE_GATE_MIN_SPLIT or wh.n_stars < UNWISE_GATE_MIN_SPLIT:
                raise ProvenanceError(
                    f"undersized split: {n_cal} calibration / {wh.n_stars} withheld stars "
                    f"evaluated, at least {UNWISE_GATE_MIN_SPLIT} required in each half"
                )
            if abs(self.empirical_psf.pixel_scale_arcsec - UNWISE_PIXEL_SCALE_ARCSEC) > 1e-6:
                raise ProvenanceError(
                    f"reproduced_archive unWISE cutouts must be on the native "
                    f"{UNWISE_PIXEL_SCALE_ARCSEC} arcsec pixels (no resampling)"
                )
        return self
