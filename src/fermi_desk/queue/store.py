"""SQLite BOED job queue: registered jobs, EIG-ranked listing, claims, runs, replays.

Schema follows SPEC.md Sec. 4.4 (job_id, tier, title, target_module,
parameters_json, expected_eig_bits + prior, est_cpu_seconds,
est_reviewer_minutes, status, code_git_sha, rng_seed, output_sha256,
created_at) plus claim / replay bookkeeping in an ``executions`` table.

Milestone 2.5: Tier-0 compute jobs keep ``expected_eig_bits = 0`` exactly (they add
no sky photons) and instead carry ``diagnostic_impact`` (what model uncertainty or
validity boundary the job resolves, computed from its parameters) and
``diagnostic_priority`` (rank among compute jobs, 1 = first). Tier-1 archival jobs
are ranked by EIG bits per hour and carry neither diagnostic field.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fermi_desk.errors import QueueStateError, ReplayMismatchError
from fermi_desk.inference.eig import utility_bits_per_hour
from fermi_desk.inference.engine import canonical_sha256, git_state
from fermi_desk.queue import runners
from fermi_desk.registry import FactorGraph

PAIR = ["frequent_slow_expanders", "rare_fast_expanders"]
ALL_CHANNELS = ["radio", "infrared", "solar_system"]


@dataclass(frozen=True)
class JobDefinition:
    job_id: str
    tier: str  # 0_compute | 1_archival
    title: str
    target_module: str
    parameters: dict
    est_cpu_seconds: float
    est_reviewer_minutes: float
    source: str  # where the job comes from (SPEC section / milestone brief)
    archival_script: str | None = None
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    # Tier-0 only: rank among compute jobs (1 = resolve first). Tier-0 jobs carry no
    # observational EIG, so they are ranked by what model-validity question they settle.
    diagnostic_priority: int | None = None


JOB_DEFINITIONS: tuple[JobDefinition, ...] = (
    JobDefinition(
        job_id="wlodarczyk_sroka_288k_background_extension",
        tier="1_archival",
        title=(
            "Condition on the background-field null inside the BL L-band beams "
            "(Wlodarczyk-Sroka 2020 method)"
        ),
        target_module="fermi_desk.models.radio",
        parameters={
            "pair": PAIR,
            "geometry": "thin_disk_2d",
            "radio_mode": "distance_scaled_eirp",
            "shell_table": "data/queue/wlodarczyk_sroka_2020_gbt_shells.yaml",
            "lowest_eirp_w": 1.0e10,
            "n_quantiles": 64,
            "n_directions": 8,
        },
        est_cpu_seconds=600.0,
        est_reviewer_minutes=180.0,
        source="SPEC.md Sec. 5 (Milestone 2)",
        archival_script="scripts/fetch_wlodarczyk_sroka_288k.py",
        notes=(
            "Shares pointing_group_id bl_gbt_lband_2017 with BL 692: enters only through "
            "union_targets_by_pointing (v0.2.1). The script reproduces the Wlodarczyk-Sroka "
            "2020 method with Gaia DR3 cones on the 692 Enriquez 2017 pointings (not their "
            "288,315-star catalogue, which covers 1,327 Price 2020 pointings with Gaia DR2). "
            "Executed live in v0.2.1b: the job is completed by the SHA-256-pinned artifact "
            "data/catalogs/bl_gbt_lband_background_summary.json; expected_eig_bits stays the "
            "registered synthetic-proxy prediction and the realized value is linked evidence."
        ),
    ),
    JobDefinition(
        job_id="wise_independent_control_fields",
        tier="1_archival",
        title="Measure WISE W3 false-positive / centroid floor on independent control stars",
        target_module="fermi_desk.models.hephaistos",
        parameters={"ir_survey": "hephaistos_wise_mid_ir", "n_control_stars": 200000},
        est_cpu_seconds=3600.0,
        est_reviewer_minutes=240.0,
        source="SPEC.md Sec. 5 (Milestone 2)",
        archival_script="scripts/run_wise_control_experiment.py",
    ),
    JobDefinition(
        job_id="hephaistos_blend_recovery_grid",
        tier="0_compute",
        title="WISE centroid vs. image-level blend-recovery grid (separation x flux fraction)",
        target_module="fermi_desk.models.hephaistos",
        parameters={
            "separations_arcsec": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
            "flux_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
            "n_trials": 40,
            "snr": 12.5,
            "psf_fwhm_arcsec": 6.5,
            "sigma_sys_arcsec": 0.5,
        },
        est_cpu_seconds=10.0,
        est_reviewer_minutes=20.0,
        source="docs/MILESTONE_2.md B3 (registered jobs table)",
        diagnostic_priority=2,
    ),
    JobDefinition(
        job_id="disk_scale_height_sweep",
        tier="0_compute",
        title="Thick-disk scale-height sweep of the benchmark Bayes factor",
        target_module="fermi_desk.models.expansion",
        parameters={
            "pair": PAIR,
            "scale_heights_pc": [100.0, 200.0, 300.0, 500.0, 1000.0],
            "radio_mode": "flat_probability",
            "channels": ALL_CHANNELS,
            "n_mc": 4000,
        },
        est_cpu_seconds=30.0,
        est_reviewer_minutes=15.0,
        source="docs/MILESTONE_2.md B3 (registered jobs table)",
        diagnostic_priority=3,
    ),
    JobDefinition(
        job_id="eirp_luminosity_function_sweep",
        tier="0_compute",
        title="Transmitter EIRP luminosity-function sweep (distance_scaled_eirp radio)",
        target_module="fermi_desk.models.radio",
        parameters={
            "pair": PAIR,
            "geometry": "thin_disk_2d",
            "medians_w": [1e11, 1e12, 1e13, 1e14, 1e15],
            "sigmas_dex": [0.5, 1.0, 2.0],
            "n_mc": 2000,
        },
        # v0.1.1: ~4 min per full-parameter run (and again per replay) was measured
        # during Milestone 3 verification; the original 30 s registration was stale.
        est_cpu_seconds=240.0,
        est_reviewer_minutes=15.0,
        source="docs/MILESTONE_2.md B3 (registered jobs table)",
        diagnostic_priority=4,
    ),
    JobDefinition(
        job_id="lookback_window_t_sweep",
        tier="0_compute",
        title="Lookback-window T sweep (50 kyr - 10 Myr), 2D and 3D, benchmark pair",
        target_module="fermi_desk.models.expansion",
        parameters={
            "pair": PAIR,
            "t_min_gyr": 5.0e-5,
            "t_max_gyr": 1.0e-2,
            "n_points": 12,
            "geometries": ["thin_disk_2d", "thick_disk_3d"],
            "radio_mode": "flat_probability",
            "channels": ALL_CHANNELS,
            "n_mc": 4000,
        },
        est_cpu_seconds=120.0,
        est_reviewer_minutes=15.0,
        source="docs/MILESTONE_2.md B3 (planning-specified 6th job)",
        diagnostic_priority=1,
    ),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    tier TEXT NOT NULL CHECK (tier IN ('0_compute', '1_archival')),
    title TEXT NOT NULL,
    target_module TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    expected_eig_bits REAL NOT NULL,
    eig_prior_json TEXT NOT NULL,
    eig_basis TEXT NOT NULL,
    eig_details_json TEXT NOT NULL,
    est_cpu_seconds REAL NOT NULL,
    est_reviewer_minutes REAL NOT NULL,
    utility_bits_per_hour REAL NOT NULL,
    diagnostic_impact TEXT,
    diagnostic_priority INTEGER,
    status TEXT NOT NULL CHECK (status IN ('open', 'claimed', 'completed', 'rejected')),
    source TEXT NOT NULL,
    archival_script TEXT,
    claimed_by TEXT,
    claimed_at TEXT,
    code_git_sha TEXT,
    rng_seed INTEGER,
    output_sha256 TEXT,
    output_path TEXT,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS executions (
    exec_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    kind TEXT NOT NULL CHECK (kind IN ('run', 'replay')),
    rng_seed INTEGER NOT NULL,
    code_git_sha TEXT NOT NULL,
    output_sha256 TEXT NOT NULL,
    matches_recorded INTEGER,
    created_at TEXT NOT NULL
);
"""
# Columns added after the first released schema: (name, SQL type). Old databases are
# migrated in place with ALTER TABLE so existing rows keep their status and hashes.
MIGRATED_COLUMNS = (("diagnostic_impact", "TEXT"), ("diagnostic_priority", "INTEGER"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def compute_job_eig(graph: FactorGraph, job: JobDefinition):
    if job.tier == "0_compute":
        return runners.tier0_eig()
    return runners.ARCHIVAL_EIG[job.job_id](graph, job.parameters)


def compute_job_diagnostic(graph: FactorGraph, job: JobDefinition) -> tuple[str | None, int | None]:
    """(diagnostic_impact, diagnostic_priority); both None for Tier-1 archival jobs."""
    if job.tier != "0_compute":
        if job.diagnostic_priority is not None:
            raise QueueStateError(f"{job.job_id}: Tier-1 jobs are ranked by EIG, not priority")
        return None, None
    if job.diagnostic_priority is None or job.diagnostic_priority < 1:
        raise QueueStateError(f"{job.job_id}: Tier-0 job needs diagnostic_priority >= 1")
    if job.job_id not in runners.TIER0_DIAGNOSTICS:
        raise QueueStateError(f"{job.job_id}: Tier-0 job has no diagnostic_impact statement")
    return runners.TIER0_DIAGNOSTICS[job.job_id](graph, job.parameters), job.diagnostic_priority


def execute_job(graph: FactorGraph, job_id: str, parameters: Mapping, seed: int) -> dict:
    """Run a Tier-0 job and return its deterministic output payload (with sha256)."""
    if job_id not in runners.TIER0_RUNNERS:
        raise QueueStateError(f"{job_id} has no offline runner (Tier-1 archival job)")
    results = runners.TIER0_RUNNERS[job_id](graph, parameters, seed)
    payload = {
        "job_id": job_id,
        "parameters": dict(parameters),
        "rng_seed": seed,
        "results": results,
    }
    return {**payload, "output_sha256": canonical_sha256(payload)}


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(jobs)")}
        for name, sql_type in MIGRATED_COLUMNS:
            if name not in have:
                self.conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ seed
    def seed(
        self, graph: FactorGraph, jobs: Iterable[JobDefinition] = JOB_DEFINITIONS
    ) -> list[str]:
        """Register jobs that are not yet present (existing rows keep their status)."""
        inserted = []
        for job in jobs:
            if self.get(job.job_id) is not None:
                # Refresh registration metadata on existing rows (diagnostic fields added in
                # Milestone 2.5; cost estimates revised in v0.1.1; archival script pointer
                # in v0.2.1; title wording in v0.2.1b). The stored EIG is reused for the
                # utility score; status, seeds and output hashes are never touched.
                impact, priority = compute_job_diagnostic(graph, job)
                self.conn.execute(
                    "UPDATE jobs SET diagnostic_impact=?, diagnostic_priority=?, title=?, source=?, "
                    "archival_script=?, est_cpu_seconds=?, est_reviewer_minutes=?, "
                    "utility_bits_per_hour=? WHERE job_id=?",
                    (
                        impact,
                        priority,
                        job.title,
                        job.source,
                        job.archival_script,
                        job.est_cpu_seconds,
                        job.est_reviewer_minutes,
                        utility_bits_per_hour(
                            self.get(job.job_id)["expected_eig_bits"],
                            job.est_cpu_seconds,
                            job.est_reviewer_minutes,
                        ),
                        job.job_id,
                    ),
                )
                continue
            res = compute_job_eig(graph, job)
            impact, priority = compute_job_diagnostic(graph, job)
            self.conn.execute(
                """INSERT INTO jobs (job_id, tier, title, target_module, parameters_json,
                expected_eig_bits, eig_prior_json, eig_basis, eig_details_json, est_cpu_seconds,
                est_reviewer_minutes, utility_bits_per_hour, diagnostic_impact,
                diagnostic_priority, status, source, archival_script, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?)""",
                (
                    job.job_id,
                    job.tier,
                    job.title,
                    job.target_module,
                    json.dumps(job.parameters, sort_keys=True),
                    res.bits,
                    json.dumps(res.prior, sort_keys=True),
                    res.basis,
                    json.dumps(res.details, sort_keys=True),
                    job.est_cpu_seconds,
                    job.est_reviewer_minutes,
                    utility_bits_per_hour(res.bits, job.est_cpu_seconds, job.est_reviewer_minutes),
                    impact,
                    priority,
                    job.source,
                    job.archival_script,
                    _now(),
                ),
            )
            inserted.append(job.job_id)
        self.conn.commit()
        self.sync_archival_completions(graph, jobs)
        return inserted

    def sync_archival_completions(
        self, graph: FactorGraph, jobs: Iterable[JobDefinition] = JOB_DEFINITIONS
    ) -> list[str]:
        """Mark Tier-1 archival jobs completed by their promoted, SHA-256-pinned artifact.

        A Tier-1 job is never run by the queue: its opt-in script fetches live data and
        promotes a pinned artifact. When that artifact exists and passes the full offline
        verification, the job's row records it (``output_sha256`` = artifact file hash,
        ``code_git_sha`` = the commit that built it). Non-destructive: only ``open`` or
        ``claimed`` rows move to ``completed`` (a claim keeps its ``claimed_by``); rejected
        rows, existing completions, the registered EIG prediction, seeds of other jobs and
        the ``executions`` log are never touched.
        """
        done = []
        for job in jobs:
            resolver = runners.ARCHIVAL_COMPLETIONS.get(job.job_id)
            row = self.get(job.job_id)
            if resolver is None or row is None or row["status"] not in ("open", "claimed"):
                continue
            evidence = resolver(graph)
            if evidence is None:
                continue
            self.conn.execute(
                """UPDATE jobs SET status='completed', code_git_sha=?, rng_seed=?,
                output_sha256=?, output_path=?, completed_at=? WHERE job_id=?""",
                (
                    evidence["code_git_sha"],
                    evidence["rng_seed"],
                    evidence["output_sha256"],
                    evidence["output_path"],
                    evidence["completed_at"],
                    job.job_id,
                ),
            )
            done.append(job.job_id)
        self.conn.commit()
        return done

    # ---------------------------------------------------------------- reads
    def get(self, job_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def require(self, job_id: str) -> dict:
        row = self.get(job_id)
        if row is None:
            raise QueueStateError(f"unknown job {job_id!r} (run `fermi-desk queue seed`)")
        return row

    def list(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM jobs ORDER BY utility_bits_per_hour DESC, job_id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_archival(self) -> list[dict]:
        """Tier-1 archival queue, ranked by EIG bits per hour of compute + review."""
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE tier = '1_archival' "
            "ORDER BY utility_bits_per_hour DESC, job_id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_compute(self) -> list[dict]:
        """Tier-0 compute queue, ranked by diagnostic priority (1 first)."""
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE tier = '0_compute' "
            "ORDER BY diagnostic_priority IS NULL, diagnostic_priority ASC, job_id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def executions(self, job_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM executions WHERE job_id = ? ORDER BY exec_id", (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------- transitions
    def claim(self, job_id: str, by: str) -> None:
        row = self.require(job_id)
        if row["status"] != "open":
            raise QueueStateError(f"{job_id} is {row['status']}, only open jobs can be claimed")
        self.conn.execute(
            "UPDATE jobs SET status='claimed', claimed_by=?, claimed_at=? WHERE job_id=?",
            (by, _now(), job_id),
        )
        self.conn.commit()

    def reject(self, job_id: str, reason: str) -> None:
        row = self.require(job_id)
        if row["status"] not in ("open", "claimed"):
            raise QueueStateError(f"{job_id} is {row['status']}; cannot reject")
        self.conn.execute(
            "UPDATE jobs SET status='rejected', rejection_reason=? WHERE job_id=?", (reason, job_id)
        )
        self.conn.commit()

    def run(self, graph: FactorGraph, job_id: str, seed: int, out_dir: Path) -> dict:
        row = self.require(job_id)
        if row["tier"] != "0_compute":
            raise QueueStateError(
                f"{job_id} is Tier-1 archival (live data); run its script instead: "
                f"{row['archival_script']}"
            )
        if row["status"] != "claimed":
            raise QueueStateError(f"{job_id} is {row['status']}; claim it before running")
        out = execute_job(graph, job_id, json.loads(row["parameters_json"]), seed)
        git_sha, git_head = git_state(graph.root)
        path = self._write(out, out_dir, git_sha, git_head, "run")
        self.conn.execute(
            """UPDATE jobs SET status='completed', code_git_sha=?, rng_seed=?, output_sha256=?,
            output_path=?, completed_at=? WHERE job_id=?""",
            (git_sha, seed, out["output_sha256"], path.as_posix(), _now(), job_id),
        )
        self._record(job_id, "run", seed, git_sha, out["output_sha256"], None)
        return {**out, "output_path": path.as_posix(), "code_git_sha": git_sha}

    def replay(
        self, graph: FactorGraph, job_id: str, out_dir: Path, raise_on_mismatch: bool = True
    ) -> dict:
        row = self.require(job_id)
        if row["status"] != "completed" or row["output_sha256"] is None:
            raise QueueStateError(f"{job_id} has no completed run to replay")
        seed = int(row["rng_seed"])
        out = execute_job(graph, job_id, json.loads(row["parameters_json"]), seed)
        git_sha, git_head = git_state(graph.root)
        path = self._write(out, out_dir, git_sha, git_head, "replay")
        match = out["output_sha256"] == row["output_sha256"]
        self._record(job_id, "replay", seed, git_sha, out["output_sha256"], match)
        result = {
            **out,
            "output_path": path.as_posix(),
            "recorded_output_sha256": row["output_sha256"],
            "matches_recorded": match,
        }
        if not match and raise_on_mismatch:
            raise ReplayMismatchError(
                f"{job_id}: replay sha {out['output_sha256']} != recorded {row['output_sha256']}"
            )
        return result

    # --------------------------------------------------------------- helpers
    def _record(self, job_id, kind, seed, git_sha, sha, match) -> None:
        self.conn.execute(
            """INSERT INTO executions (job_id, kind, rng_seed, code_git_sha, output_sha256,
            matches_recorded, created_at) VALUES (?,?,?,?,?,?,?)""",
            (job_id, kind, seed, git_sha, sha, None if match is None else int(match), _now()),
        )
        self.conn.commit()

    @staticmethod
    def _write(out: dict, out_dir: Path, git_sha: str, git_head: str, kind: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = out_dir / f"{out['job_id']}-{kind}-seed{out['rng_seed']}-{stamp}.json"
        doc = {
            **out,
            "kind": kind,
            "code_git_sha": git_sha,
            "git_head": git_head,
            "written_at": _now(),
        }
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        return path
