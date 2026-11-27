"""FastAPI read / replay layer for the Fermi desk (Milestone 3, ``docs/MILESTONE_3.md`` Part B).

Design rules:

* **Offline.** No route performs outbound network I/O; every payload is computed from
  the verified local ``data/`` tree, the SQLite queue and ``runs/`` receipts.
* **Same engine, same invariants.** Routes call the exact functions the CLI calls
  (``run_comparison``, ``run_ir_drilldown``, ``run_adversary``, ``JobStore``); nothing
  is re-implemented here, so double-counting rejection, Tier-3 quarantine and the
  ``verified_channels_only`` headline policy hold by construction.
* **Strict inputs.** Pydantic v2 models with literal enums, strict types and bounded
  seeds; job ids must be registered queue ids; no path parameters reach the
  filesystem; no shell execution.
* **JSON-safe.** Non-finite floats (``inf`` / ``nan``) are emitted as ``null``.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from fermi_desk import __version__
from fermi_desk.errors import FermiDeskError, QueueStateError
from fermi_desk.inference.engine import (
    CHANNELS,
    RECEIPT_SCHEMA_VERSION,
    RunConfig,
    git_state,
    radiometer,
    run_comparison,
)
from fermi_desk.models import radio
from fermi_desk.models.expansion import C_KPC_PER_GYR, FINITE_DISK
from fermi_desk.queue.store import JOB_DEFINITIONS, JobDefinition, JobStore
from fermi_desk.registry import (
    FactorGraph,
    load_factor_graph,
    load_fits_gate2,
    load_sigma_sys_calibration,
    run_validation,
)
from fermi_desk.schemas import (
    REJECT_DUPLICATE_POINTING_GROUP,
    PointingMergePolicy,
    validate_pointing_merge_policy,
)

PAIR = ("frequent_slow_expanders", "rare_fast_expanders")
IR_PAIR = ("quiet_expanders", "partial_dyson_conspicuous")
LOCAL_VALIDITY_KPC = 8.0
MAX_SEED = 2**32 - 1
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
JOB_ID_RE = re.compile(r"^[a-z0-9_]{1,80}$")
RUNS_FEED_LIMIT = 50
READ_ONLY_DETAIL = "Server is running in read-only mode; run or replay jobs locally via CLI."

Seed = Annotated[StrictInt, Field(ge=0, le=MAX_SEED)]
ChannelName = Literal["radio", "infrared", "solar_system"]


def is_loopback_host(host: str) -> bool:
    """True for bind addresses reachable only from this machine (localhost, 127/8, ::1)."""
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False  # any other hostname may resolve to a routable interface


# ------------------------------------------------------------------- JSON safety
def json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays and map non-finite floats to None."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


# --------------------------------------------------------------- request models
class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geometry_mode: Literal["thin_disk_2d", "thick_disk_3d", "finite_disk_3d"] = "thin_disk_2d"
    radio_mode: Literal["flat_probability", "distance_scaled_eirp"] = "flat_probability"
    condition_on_earth_unreached: StrictBool = True
    channels: list[ChannelName] = Field(
        default_factory=lambda: list(CHANNELS), min_length=1, max_length=3
    )
    include_illustrative: StrictBool = False
    seed: Seed = 42
    # v0.2.0. disk_profile is validated for every request and applied to finite_disk_3d
    # only; marginalize_lookback integrates T over its registry log-uniform prior.
    disk_profile: Literal["uniform", "exponential"] = "uniform"
    marginalize_lookback: StrictBool = False
    # v0.2.1. "keep_first" is refused with an explicit MergePolicyError (HTTP 422), never
    # aliased; the default leaves declared pointing extensions out of the run.
    pointing_merge_policy: PointingMergePolicy = REJECT_DUPLICATE_POINTING_GROUP

    @field_validator("pointing_merge_policy", mode="before")
    @classmethod
    def _merge_policy(cls, v: object) -> str:
        return validate_pointing_merge_policy(v)

    @field_validator("channels")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("channels must be unique")
        return sorted(v, key=CHANNELS.index)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: Seed = 42
    contributor: str = Field("local-desk-ui", pattern=r"^[A-Za-z0-9._-]{1,64}$")


# -------------------------------------------------------------- response models
class _Out(BaseModel):
    model_config = ConfigDict(extra="allow")


class HealthOut(_Out):
    status: Literal["ok", "degraded"]
    git_sha: str
    git_head: str
    validation_passed: bool
    validation_checks_passed: int
    validation_checks_total: int
    validation_failures: list[str]
    catalog_sha256: str | None
    wise_calibration_sha256: str | None
    # v0.1.2: live provenance of the AllWISE control sample (read from the verified files)
    n_controls: int | None
    n_calibration: int | None
    n_withheld: int | None
    calibration_data_tier: str | None
    fits_gate2_sha256: str | None
    fits_gate2_data_tier: str | None
    receipt_schema_version: int
    read_only: bool
    version: str  # fermi_desk.__version__ (single source: pyproject.toml)


class RegistryOut(_Out):
    parameters: list[dict]
    hypotheses: dict[str, list[dict]]
    surveys: list[dict]


class CompareOut(_Out):
    request: dict
    cached: bool
    headline_policy: str
    headline_log10_bf: float | None
    headline_exclusion_notes: list[dict]
    receipt: dict


class QueueOut(_Out):
    queue_path: str
    tier1_archival: list[dict]
    tier0_compute: list[dict]


class RunOut(_Out):
    job: dict
    output_json: dict
    output_sha256: str
    output_path: str


class ReplayOut(_Out):
    job_id: str
    rng_seed: int
    recorded_sha256: str
    replayed_sha256: str
    match: bool
    code_git_sha: str


class RunsOut(_Out):
    runs: list[dict]
    limit: int


# ----------------------------------------------------------------- app factory
ADVERSARY_CITATIONS = {
    "duplicate_pointings": [
        "Enriquez et al. 2017, ApJ 849, 104 (BL 692 pointings)",
        "Wlodarczyk-Sroka, Garrett & Siemion 2020, MNRAS 498, 5720 (GBT L-band FWHM 8.4')",
    ],
    "pointing_overlap": [
        "SPEC.md Sec. 2 invariant 4 (strict non-double-counting) and v0.2 roadmap item 1 "
        "(union_targets_by_pointing; keep_first forbidden)",
        "Wlodarczyk-Sroka, Garrett & Siemion 2020, MNRAS 498, 5720 (background stars in BL beams)",
    ],
    "scaling_symmetry_2d_vs_3d": ["SPEC.md Sec. 2 invariant 5 (explicit 2D vs 3D geometry)"],
    "local_approximation": ["SPEC.md v0.2 roadmap item 2 (finite-Galaxy disk, R_gal = 15 kpc)"],
    "illustrative_tier_quarantine": ["SPEC.md Sec. 2 invariant 6 (data provenance tagging)"],
}
ADVERSARY_CLAIM_FILES = {
    "hip1692_distance_revision": "data/adversary_claims/hip_1692.yaml",
    "hephaistos_blend_anchors": "data/adversary_claims/hephaistos_candidates_d_e.yaml",
}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def create_app(
    root: Path | None = None,
    *,
    queue_path: Path | None = None,
    queue_out_dir: Path | None = None,
    runs_dir: Path | None = None,
    static_dir: Path | None = None,
    compare_n_mc: int = 20000,
    compare_n_importance: int = 16384,
    lookback_grid_points: int = 16,
    ir_n_trials: int = 40,
    adversary_n_mc: int = 4000,
    seed_jobs: Sequence[JobDefinition] | None = None,
    read_only: bool = False,
) -> FastAPI:
    """Build the desk app. Every path defaults inside the project root.

    ``read_only=True`` (for any non-loopback deployment) refuses the two endpoints that
    execute engine code synchronously, /api/queue/{job_id}/run and /replay, with HTTP 403;
    all read endpoints are unchanged.
    """
    graph: FactorGraph = load_factor_graph(root)
    root = graph.root
    queue_path = queue_path or root / "runs" / "queue.sqlite"
    queue_out_dir = queue_out_dir or queue_path.parent / "queue"
    runs_dir = runs_dir or root / "runs"
    jobs = tuple(seed_jobs) if seed_jobs is not None else JOB_DEFINITIONS
    ir_survey = next(s for s in graph.surveys.values() if s.channel == "infrared")
    radio_survey = next(
        s for s in graph.surveys.values() if s.channel == "radio" and not s.is_pointing_extension
    )

    app = FastAPI(
        title="Fermi Paradox Constraint Desk",
        version=__version__,
        description="Offline read / replay API over the verified fermi-desk engine.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(FermiDeskError)
    async def _domain_error(_: Request, exc: FermiDeskError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"error": type(exc).__name__, "detail": str(exc)}
        )

    # ------------------------------------------------------------- cached cores
    @lru_cache(maxsize=4)
    def validation(sha: str) -> dict:
        res = run_validation(root)
        return {
            "passed": all(r.ok for r in res),
            "n_ok": sum(r.ok for r in res),
            "n": len(res),
            "failures": [f"{r.name}: {r.detail}" for r in res if not r.ok],
        }

    @lru_cache(maxsize=4)
    def provenance(sha: str) -> dict:
        """Control-sample provenance from the verified files (None fields if unavailable)."""
        out: dict = dict.fromkeys(
            (
                "n_controls",
                "n_calibration",
                "n_withheld",
                "calibration_data_tier",
                "fits_gate2_sha256",
                "fits_gate2_data_tier",
            )
        )
        try:
            cal = load_sigma_sys_calibration(ir_survey, root)
            gate = load_fits_gate2(ir_survey, root)
        except (FermiDeskError, FileNotFoundError, KeyError, ValueError):
            return out  # validation (health.validation_failures) reports the reason
        if cal is not None:
            ctl = cal.record.controls
            out.update(
                n_controls=cal.record.n_controls,
                n_calibration=sum(c.split == "calibration" for c in ctl),
                n_withheld=sum(c.split == "withheld" for c in ctl),
                calibration_data_tier=cal.record.data_tier,
            )
        if gate is not None:
            out.update(
                fits_gate2_sha256=gate.file_sha256, fits_gate2_data_tier=gate.record.data_tier
            )
        return out

    @lru_cache(maxsize=128)
    def compare_cached(sha: str, key: str) -> dict:
        req = CompareRequest.model_validate_json(key)
        cfg = RunConfig(
            geometry=req.geometry_mode,
            radio_mode=req.radio_mode,
            seed=req.seed,
            channels=tuple(req.channels),
            condition_on_earth_unreached=req.condition_on_earth_unreached,
            n_mc=compare_n_mc,
            headline_pair=PAIR,
            ir_pair=IR_PAIR if "infrared" in req.channels else None,
            include_illustrative_in_headline=req.include_illustrative,
            disk_profile=req.disk_profile if req.geometry_mode == FINITE_DISK else "uniform",
            marginalize_lookback=req.marginalize_lookback,
            lookback_grid_points=lookback_grid_points,
            n_importance=compare_n_importance,
            pointing_merge_policy=req.pointing_merge_policy,
        )
        rec = run_comparison(graph, cfg)
        for h in rec["results"]["hypotheses"].values():
            v = h["parameters"].get("expansion_speed_c", {}).get("value")
            if h.get("expansion") and v:
                if req.geometry_mode == FINITE_DISK:
                    # No infinite-disk limit: origins are confined to R <= R_gal.
                    h["expansion"]["t_valid_gyr"] = None
                    h["expansion"]["t_valid_kyr"] = None
                    h["expansion"]["local_validity_criterion"] = (
                        "finite Galactic disk (origins confined to R <= R_gal): valid at any "
                        "v_exp * T"
                    )
                    continue
                t = LOCAL_VALIDITY_KPC / (v * C_KPC_PER_GYR)
                h["expansion"]["t_valid_gyr"] = t
                h["expansion"]["t_valid_kyr"] = t * 1e6
                h["expansion"]["local_validity_criterion"] = (
                    f"v_exp * T <= {LOCAL_VALIDITY_KPC:g} kpc (infinite-disk approximation)"
                )
        return json_safe(rec)

    @lru_cache(maxsize=8)
    def ir_cached(sha: str, seed: int) -> dict:
        from fermi_desk.inference.ir_drilldown import run_ir_drilldown

        return json_safe(run_ir_drilldown(graph, seed=seed, pair=IR_PAIR, n_trials=ir_n_trials))

    @lru_cache(maxsize=8)
    def adversary_cached(sha: str, seed: int) -> dict:
        from fermi_desk.adversary.checks import run_adversary

        rec = run_adversary(graph, seed=seed, pair=PAIR, n_mc=adversary_n_mc)
        for f in rec["results"]["findings"]:
            cites = list(ADVERSARY_CITATIONS.get(f["check"], []))
            claim = ADVERSARY_CLAIM_FILES.get(f["check"])
            if claim:
                cites += yaml.safe_load((root / claim).read_text("utf-8")).get("citations", [])
            f["citations"] = cites
            f["repro_command"] = f"uv run fermi-desk adversary run --seed {seed}"
        return json_safe(rec)

    @lru_cache(maxsize=2)
    def radio_cached(sha: str) -> dict:
        from fermi_desk.adversary.checks import (
            audit_duplicate_pointings,
            audit_hip1692_distance_revision,
        )

        df = pd.read_csv(root / radio_survey.catalog_path)
        f_min = radio.min_detectable_flux(radiometer(radio_survey))
        eirp = radio.min_detectable_eirp(df["dist_pc"].to_numpy(float) * u.pc, f_min)
        cols = ["target_name", "ra_deg", "dec_deg", "dist_pc", "dist_source", "dist_legacy_pc"]
        targets = df[[*cols, "eirp_revision_factor"]].copy()
        targets["eirp_min_w"] = eirp.to_value(u.W)
        hip = audit_hip1692_distance_revision(graph, PAIR, seed=42, n_mc=adversary_n_mc)
        dup = audit_duplicate_pointings(graph)
        return json_safe(
            {
                "survey_id": radio_survey.id,
                "data_tier": radio_survey.data_tier,
                "catalog_sha256": radio_survey.catalog_sha256,
                "f_min_w_m2": float(f_min.to_value(u.W / u.m**2)),
                "beam_fwhm_arcmin": dup.details["beam_fwhm_arcmin"],
                "n_targets": len(targets),
                "targets": targets.to_dict(orient="records"),
                "hip1692_audit": {"status": hip.status, "summary": hip.summary, **hip.details},
                "hip1692_citations": yaml.safe_load(
                    (root / ADVERSARY_CLAIM_FILES["hip1692_distance_revision"]).read_text("utf-8")
                ).get("citations", []),
                "beam_sharing_pairs": dup.details["beam_sharing_pairs"],
                "beam_sharing_interpretation": dup.details["interpretation"],
            }
        )

    def sha() -> str:
        return git_state(root)[0]

    def writable() -> None:
        # Runs before the job lookup, so a read-only server reveals nothing about job state.
        if read_only:
            raise HTTPException(status_code=403, detail=READ_ONLY_DETAIL)

    def store() -> JobStore:
        s = JobStore(queue_path)
        if not s.list():
            s.seed(graph, jobs)
        return s

    # ------------------------------------------------------------------ routes
    @app.get("/api/health", response_model=HealthOut)
    def health() -> dict:
        git_sha, git_head = git_state(root)
        v = validation(git_sha)
        cal_sha = (ir_survey.infrared.sigma_sys_calibration_sha256) if ir_survey.infrared else None
        return {
            "status": "ok" if v["passed"] else "degraded",
            "git_sha": git_sha,
            "git_head": git_head,
            "validation_passed": v["passed"],
            "validation_checks_passed": v["n_ok"],
            "validation_checks_total": v["n"],
            "validation_failures": v["failures"],
            "catalog_sha256": radio_survey.catalog_sha256,
            "wise_calibration_sha256": cal_sha,
            **provenance(git_sha),
            "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
            "read_only": read_only,
            "version": __version__,
        }

    @app.get("/api/registry", response_model=RegistryOut)
    def registry() -> dict:
        groups: dict[str, list[dict]] = {"tier_1": [], "tier_2": [], "tier_3": []}
        for h in sorted(graph.hypotheses.values(), key=lambda h: h.id):
            d = h.model_dump()
            d["is_quarantined"] = h.is_quarantined
            d["empirical_score"] = None if h.is_quarantined else "scored via /api/compare"
            groups[f"tier_{h.tier}"].append(d)
        surveys = []
        for s in graph.surveys.values():
            d = s.model_dump()
            if s.infrared is not None:  # the 15-point bracket is large and illustrative
                d["infrared"]["n_calibrations"] = len(s.infrared.calibrations)
            surveys.append(d)
        return json_safe(
            {
                "parameters": [p.model_dump() for p in graph.registry.parameters],
                "hypotheses": groups,
                "surveys": surveys,
            }
        )

    @app.post("/api/compare", response_model=CompareOut)
    def compare(req: CompareRequest) -> dict:
        key = req.model_dump_json()
        before = compare_cached.cache_info().hits
        rec = compare_cached(sha(), key)
        return {
            "request": req.model_dump(),
            "cached": compare_cached.cache_info().hits > before,
            "headline_policy": rec["headline_policy"],
            "headline_log10_bf": rec["headline_log10_bf"],
            "headline_exclusion_notes": rec["headline_exclusion_notes"],
            "receipt": rec,
        }

    @app.get("/api/radio/drilldown")
    def radio_drilldown() -> dict:
        return radio_cached(sha())

    @app.get("/api/infrared/drilldown")
    def infrared_drilldown(seed: Annotated[int, Query(ge=0, le=MAX_SEED)] = 42) -> dict:
        return ir_cached(sha(), seed)

    @app.get("/api/adversary")
    def adversary(seed: Annotated[int, Query(ge=0, le=MAX_SEED)] = 42) -> dict:
        return adversary_cached(sha(), seed)

    @app.get("/api/queue", response_model=QueueOut)
    def queue() -> dict:
        s = store()
        try:
            arch, comp = s.list_archival(), s.list_compute()
        finally:
            s.close()
        cal = load_sigma_sys_calibration(ir_survey, root)
        evidence = None
        if cal is not None:
            from fermi_desk.inference.ir_drilldown import LINKED_ARCHIVAL_JOB, small_sample_bounds

            evidence = {LINKED_ARCHIVAL_JOB: small_sample_bounds(cal.summary())}
        t1 = []
        for r in arch:
            script = r["archival_script"] or ""
            runnable = script.startswith("scripts/")
            t1.append(
                {
                    **_job_core(r),
                    "utility_score": r["utility_bits_per_hour"],
                    "eig_prior": json.loads(r["eig_prior_json"]),
                    "eig_basis": r["eig_basis"],
                    "eig_details": json.loads(r["eig_details_json"]),
                    "archival_script": r["archival_script"],
                    "commands": {
                        "claim": f"uv run fermi-desk queue claim {r['job_id']} --by <name>",
                        "script": (
                            f"uv run python {script} --n-controls 120 --seed 42"
                            if r["job_id"] == "wise_independent_control_fields"
                            # Offline rehearsal only: the live fetch needs an explicit --fetch.
                            else f"uv run python {script} --dry-run"
                            if r["job_id"] == "wlodarczyk_sroka_288k_background_extension"
                            else f"uv run python {script}"
                        )
                        if runnable
                        else None,
                    },
                    "linked_evidence": (evidence or {}).get(r["job_id"]),
                }
            )
        t0 = [
            {
                **_job_core(r),
                "diagnostic_priority": r["diagnostic_priority"],
                "diagnostic_impact": r["diagnostic_impact"],
                "eig_note": "Tier-0 compute adds no sky photons: EIG = 0.0 bits by construction",
                "commands": {
                    "claim": f"uv run fermi-desk queue claim {r['job_id']} --by <name>",
                    "run": f"uv run fermi-desk queue run {r['job_id']} --seed 42",
                    "replay": f"uv run fermi-desk queue replay {r['job_id']}",
                },
            }
            for r in comp
        ]
        return json_safe(
            {"queue_path": _rel(queue_path, root), "tier1_archival": t1, "tier0_compute": t0}
        )

    def _require_job(s: JobStore, job_id: str) -> dict:
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(status_code=404, detail="unknown job id")
        row = s.get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return row

    @app.post("/api/queue/{job_id}/run", response_model=RunOut, dependencies=[Depends(writable)])
    def queue_run(job_id: str, req: RunRequest | None = None) -> dict:
        req = req or RunRequest()
        s = store()
        try:
            row = _require_job(s, job_id)
            if row["tier"] != "0_compute":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "tier1_archival_refused",
                        "message": (
                            f"{job_id} is a Tier-1 archival job (live archive data); the desk "
                            "never runs it. Run its opt-in script from a terminal instead."
                        ),
                        "archival_script": row["archival_script"],
                    },
                )
            if row["status"] in ("completed", "rejected"):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": f"job_{row['status']}",
                        "message": f"{job_id} is {row['status']}; use /replay to verify it",
                        "output_sha256": row["output_sha256"],
                    },
                )
            if row["status"] == "open":
                s.claim(job_id, req.contributor)
            try:
                out = s.run(graph, job_id, req.seed, queue_out_dir)
            except QueueStateError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            job = s.require(job_id)
        finally:
            s.close()
        return json_safe(
            {
                "job": _job_core(job) | {"claimed_by": job["claimed_by"]},
                "output_json": {k: out[k] for k in ("job_id", "parameters", "rng_seed", "results")},
                "output_sha256": out["output_sha256"],
                "output_path": _rel(Path(out["output_path"]), root),
            }
        )

    @app.post(
        "/api/queue/{job_id}/replay", response_model=ReplayOut, dependencies=[Depends(writable)]
    )
    def queue_replay(job_id: str) -> dict:
        s = store()
        try:
            row = _require_job(s, job_id)
            if row["tier"] != "0_compute":
                raise HTTPException(status_code=400, detail="Tier-1 archival jobs have no replay")
            if row["status"] != "completed":
                raise HTTPException(status_code=409, detail=f"{job_id} has no completed run")
            out = s.replay(graph, job_id, queue_out_dir, raise_on_mismatch=False)
        finally:
            s.close()
        return {
            "job_id": job_id,
            "rng_seed": out["rng_seed"],
            "recorded_sha256": out["recorded_output_sha256"],
            "replayed_sha256": out["output_sha256"],
            "match": bool(out["matches_recorded"]),
            "code_git_sha": sha(),
        }

    @app.get("/api/runs", response_model=RunsOut)
    def runs() -> dict:
        return json_safe({"runs": list_runs(runs_dir, root), "limit": RUNS_FEED_LIMIT})

    # ------------------------------------------------------------ static bundle
    if static_dir is not None and (static_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        def index() -> dict:
            return {
                "message": "fermi-desk API is running; build the UI with `npm run build` in web/ "
                "and serve with --static-dir web/dist",
                "docs": "/docs",
            }

    return app


def _job_core(r: dict) -> dict:
    return {
        "job_id": r["job_id"],
        "tier": r["tier"],
        "title": r["title"],
        "status": r["status"],
        "expected_eig_bits": r["expected_eig_bits"],
        "est_cpu_seconds": r["est_cpu_seconds"],
        "est_reviewer_minutes": r["est_reviewer_minutes"],
        "source": r["source"],
        "rng_seed": r["rng_seed"],
        "code_git_sha": r["code_git_sha"],
        "output_sha256": r["output_sha256"],
        "completed_at": r["completed_at"],
        "parameters": json.loads(r["parameters_json"]),
    }


def list_runs(runs_dir: Path, root: Path, limit: int = RUNS_FEED_LIMIT) -> list[dict]:
    """Newest-first summaries of JSON receipts under ``runs_dir`` (never follows symlinks)."""
    if not runs_dir.is_dir():
        return []
    files = [p for p in runs_dir.rglob("*.json") if p.is_file() and not p.is_symlink()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files[:limit]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        kind = d.get("kind") or (
            "compare" if "headline_policy" in d or "geometry_mode" in d else None
        )
        if kind in ("run", "replay"):
            kind = f"queue_{kind}"
        out.append(
            {
                "path": _rel(p, root),
                "kind": kind or "unknown",
                "run_id": d.get("run_id") or d.get("job_id") or p.stem,
                "created_at": d.get("created_at_utc")
                or d.get("written_at")
                or d.get("generated_at_utc")
                or (d.get("query_window_utc") or [None])[-1],
                "git_sha": d.get("git_sha") or d.get("code_git_sha"),
                "rng_seed": d.get("rng_seed"),
                "sha256": d.get("results_sha256") or d.get("output_sha256"),
                "sha256_kind": "results_sha256" if d.get("results_sha256") else "output_sha256",
                "headline_policy": d.get("headline_policy"),
                "headline_log10_bf": d.get("headline_log10_bf"),
                "data_tier": d.get("data_tier"),
                "geometry_mode": d.get("geometry_mode"),
                "radio_mode": d.get("radio_mode"),
            }
        )
    return out
