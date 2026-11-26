# Milestone 2: Drill-Down Verticals, Adversary Runner & BOED Queue

> **Provenance of this brief.** No `docs/MILESTONE_2.md` existed when Milestone 2
> was assigned. This file was drafted on 2026-11-25 by the implementing agent, with
> the user's approval, from (a) `SPEC.md` §4.4 and §5 and (b) the planning
> agent's Milestone 2 assignment message. Items not literally present in those two
> sources were originally tagged `INFERRED`.
>
> **Status (Milestone 2.5 audit closeout, 2026-11-26): FINAL.** The planning agent
> approved every formerly-`INFERRED` item; the tags have been removed and this brief
> is now an authoritative source (cited as `docs/MILESTONE_2.md` by
> `src/fermi_desk/queue/store.py`). Later policy rulings are recorded in
> `docs/MILESTONE_2_5.md` and `SPEC.md`.

## Objective
Add the three Milestone 2 verticals of `SPEC.md` §5: an image-level mid-IR
blend-recovery drill-down (Project Hephaistos), an adversary runner that
reproduces real catalog / methodology failure modes, and a SQLite BOED job queue
with an exact expected-information-gain (EIG) scorer and deterministic replay.
Also land the approved Milestone 1 spec rulings (Part A).

## Explicit Scope Boundaries (DO NOT BUILD in Milestone 2)
- No `web/` (React/Vite) and no `api.py` (FastAPI): Milestone 3.
- No autonomous LLM agent loops or external task daemons.
- `uv run pytest -v` stays 100% offline (passes with sockets blocked). Live
  archive access happens only in explicit `scripts/` entry points.
- `scripts/fetch_bl_692.py` and the pinned BL 692 catalog checksum stay unchanged.

---

## Part A — Approved Milestone 1 Rulings (from the planning agent)
1. `SPEC.md` §4.1: `"dimensionless"` → `astropy.units.dimensionless_unscaled` is
   the single canonical unit alias.
2. `SPEC.md` §4.1: a log-uniform prior with lower bound `0.0` requires an explicit
   positive `log_floor`.
3. `SPEC.md` §4.3: the BL survey uses `minimum_flux_limit_jy: 17.0` (Enriquez et al.
   2017 §5.1 S_min), with optional `sefd_jy: 10.0`.
4. IR macro→micro coupling: `models/infrared.py` stays independent of expansion
   physics. When an expansion hypothesis activates `infrared`, `inference/engine.py`
   derives `f_ir_host = P(star reached) × f_ir_loud` and passes it into
   `models/infrared.py` (zero-emission invariance preserved at `f_ir_loud = 0`).
   - Approved detail: `P(star reached)` is the mean reach probability (under
     the run's Earth-unreached setting) over a deterministic lattice filling the
     IR survey footprint (Hephaistos: stars within 300 pc, Suazo et al. 2024 §2.1).
     Expansion hypotheses must not also declare `ir_host_fraction`; hypotheses
     without expansion parameters (e.g. `partial_dyson_conspicuous`) still declare it.
   - Approved: `frequent_slow_expanders` / `rare_fast_expanders` activate
     `infrared` with `ir_conspicuous_fraction` range [0.01, 0.5], taken from the
     `SPEC.md` §4.2 example.
5. HIP 1692: Gaia DR3 (788.71 pc, ~1,176× EIRP_min) stays canonical in
   `data/catalogs/bl_692_targets.csv`. Bailer-Jones DR2 (752.61 pc, ~1,071×)
   stays in `data/adversary_claims/hip_1692.yaml` as the Wlodarczyk-Sroka et al.
   (2020) historical anchor.
6. Likelihood normalisation: receipts and `compare` tables separate raw
   `log10 P(data | H)` from `log10 LR vs null`, never show raw joint comparisons
   across mismatched channel sets, and compute every pairwise Bayes factor over
   the intersection of shared channels, listing the channels that entered it.

---

## Part B — Deliverables

### B1. Infrared drill-down (`src/fermi_desk/models/hephaistos.py`, `fermi-desk ir-drilldown`)
SPEC: "AllWISE centroid vs. image-level blend-recovery grid (angular separation
vs. background flux fraction), anchored on Project Hephaistos Candidates D and E
(Ren et al. 2026; Zackrisson et al. 2026)."
- Seeded injection–recovery on simulated WISE W3 cutouts with Gaussian PSFs.
  Each trial injects star + background point source (separation θ, W3 flux
  fraction f_bg), Gaussian pixel noise at a stated W3 SNR, and an astrometric
  registration error with dispersion `sigma_sys`. Approved: Gaussian PSF
  (FWHM 6.5″ for W3) and 1.375″ pixels, from Ren et al. 2026 Table 2 and the AllWISE
  Atlas pixel scale.
- Two screens, each with a stated false-alarm rate:
  1. **Centroid screen**: W3 flux-weighted centroid offset from the Gaia
     position, tested against a Rayleigh null with
     `sigma = sqrt((FWHM / (2.355 SNR))^2 + sigma_sys^2)` (Ren et al. 2026 eq. 1).
  2. **Image-level screen**: Δχ² between a one-PSF model (position prior
     centred on Gaia with `sigma_sys`) and a two-PSF model.
- Outputs:
  - recovery-probability grids for both screens;
  - false-alarm rates;
  - blend survival fractions;
  - screened `IRCalibration` objects consumed through `inference/engine.py` /
    `models/infrared.py` (AGENTS.md law 6).
  All carry `data_tier: illustrative_calibration` (synthetic images).
- Anchors (`data/adversary_claims/hephaistos_candidates_d_e.yaml`,
  `published_catalog`):
  - Gaia DR3 IDs and positions;
  - ≈1″ background-galaxy separations (Zackrisson et al. 2026);
  - W3 centroid offsets (Ren et al. 2026 Table 5);
  - published contamination fractions (Zackrisson et al. 2026 §5.1).
  The drill-down reports whether the centroid screen is predicted to catch each
  anchor and compares that with the published outcome (D: no centroid evidence;
  E: marginal).
- Approved: the drill-down recomputes LR[quiet_expanders : partial_dyson_conspicuous]
  under unscreened / centroid-screened / image-screened false-positive rates for
  two published contamination scenarios (the Suazo 2024 2×10⁵-subsample
  convention and the Zackrisson 2026 full-sample critique).

### B2. Adversary runner (`src/fermi_desk/adversary/checks.py`, `fermi-desk adversary run`)
- `audit_hip1692_distance_revision` (planning-specified):
  - verify BOTH the canonical DR3 distance (catalog) and the Bailer-Jones DR2
    anchor (claim file) using `astropy.units` (W);
  - assert a >1,000× EIRP_min increase under both (threshold, not a hard-coded
    multiplier);
  - report the change in the `distance_scaled_eirp` radio likelihood.
- `audit_duplicate_pointings` (SPEC): cross-survey `pointing_group_id`
  collisions (with a detector self-test on an injected duplicate), duplicate
  target names, and beam-sharing target pairs closer than the GBT
  L-band FWHM.
- Approved additional audits:
  - Hephaistos D/E centroid-screen reproduction;
  - 2D-degeneracy / 3D-breaking check on the benchmark pair at current defaults;
  - local-approximation (v·T ≤ 8 kpc) violations;
  - IR calibration-tier check (illustrative calibrations never reported as bounds).
- Seeded; writes a JSON receipt under `runs/adversary/`.

### B3. BOED queue & EIG (`src/fermi_desk/queue/store.py`, `src/fermi_desk/inference/eig.py`)
- SQLite store with every `SPEC.md` §4.4 field, plus claim / replay bookkeeping.
- Lifecycle CLI: `fermi-desk queue seed | list | claim | run | replay | reject`.
  `run` executes Tier-0 compute jobs offline with an explicit seed and records
  `output_sha256`. `replay` re-executes with the recorded parameters and seed and
  must reproduce the hash exactly. Tier-1 archival jobs are refused by `run` and
  point to their opt-in script.
- EIG: exact mutual information (bits) between a stated prior over hypotheses
  (or calibrations) and a job's discrete outcome. Count outcomes are evaluated
  exactly on a support truncated at tail mass < 1e-15, with the tail lumped into
  one outcome. Utility = bits / (cpu_hours + reviewer_hours).
  - Approved: Tier-0 compute jobs add no observational data, so their EIG
    about hypotheses is reported as exactly 0 bits; their value is diagnostic
    (Milestone 2.5: stated in the `diagnostic_impact` / `diagnostic_priority` columns).
- Registered jobs:

| job_id | tier | source |
|---|---|---|
| `wlodarczyk_sroka_288k_background_extension` | 1_archival | SPEC §5 |
| `wise_independent_control_fields` | 1_archival | SPEC §5 |
| `hephaistos_blend_recovery_grid` | 0_compute | this brief (approved) |
| `disk_scale_height_sweep` | 0_compute | this brief (approved) |
| `eirp_luminosity_function_sweep` | 0_compute | this brief (approved) |
| `lookback_window_t_sweep` | 0_compute | planning-specified (6th job) |

- `lookback_window_t_sweep`: log-spaced T from 5e-5 Gyr (50 kyr) to 1e-2 Gyr
  (10 Myr), `frequent_slow_expanders` vs `rare_fast_expanders`, both geometries.
  At each T it records:
  - P(Earth reached);
  - E[N_reached | Earth unreached];
  - pairwise radio and joint log10 BF;
  - per-hypothesis `local_approx` status.
  It also records the exact threshold `T_valid = 8 kpc / v_exp` per hypothesis.

### B4. Opt-in live script (`scripts/run_wise_control_experiment.py`)
Fetches AllWISE W3 cutouts from NASA/IPAC IRSA for control positions and
measures the empirical centroid-offset floor (`sigma_sys`). It is never imported
or run by `pytest`, and writes a `reproduced_archive` receipt with query receipts.
Approved: control positions are SNR-matched M dwarfs drawn from Gaia DR3 within
300 pc via the ARI TAP mirror.

## Mandatory CI additions
- `tests/test_hephaistos.py`: grid monotonicity (recovery grows with θ and f_bg),
  calibrated false-alarm rates, seeded determinism, zero-blend invariance,
  anchor-consistency.
- `tests/test_adversary.py`: HIP 1692 dual verification (>1,000× under both
  distances), likelihood delta sign, duplicate-pointing detector self-test,
  receipt determinism.
- `tests/test_eig_queue.py`: MI identities (0 for identical predictions,
  ≤ H(prior), log2 2 for disjoint binary predictions), queue lifecycle,
  replay hash equality for every Tier-0 job, archival-run refusal, the T-sweep's
  `local_approx` threshold.
- Every Milestone 1 test stays in place, updated only where the Part A
  normalisation fix renames fields.

## Acceptance & Handoff
`AGENTS.md` §5 format, with verbatim output of: `pytest -v`, `ruff check`,
`ruff format --check`, `validate`, `compare` (thin_disk_2d / flat, seed 42),
`ir-drilldown --seed 42`, `adversary run --seed 42`, and the queue lifecycle
(seed, list, claim, run, replay with exact `output_sha256` match).

---

## Canonical baseline (locked at Milestone 2.5)
The distance-preserving 2D projection fix (Milestone 2 adversary finding: fast fronts
must reach high-latitude stars at their true distance) changed the thin-disk numbers.
The post-fix values below are the **canonical baseline**; the pre-fix values are
superseded. Command: `fermi-desk compare --geometry <g> --radio-mode flat_probability
--seed 42` (n_mc = 20000, Earth-unreached conditioning on).

| quantity | thin_disk_2d | thick_disk_3d |
|---|---|---|
| P(Earth reached), frequent_slow / rare_fast | 0.00979582 / 0.00979582 | 0.00945466 / 0.00979579 |
| E[N reached \| Earth unreached] (BL 692), frequent_slow | 0.10073 | 0.0725544 |
| E[N reached \| Earth unreached] (BL 692), rare_fast | 0.000348837 | 0.000179262 |
| log10 BF term `earth_unreached` | -8.67e-19 (exact 2D degeneracy) | +1.49587e-04 |
| log10 BF term `radio` | -8.04966e-05 | -5.3355e-05 |
| log10 BF term `infrared` (illustrative, excluded from headline) | +0.128843 | +0.16936 |
| **Headline log10 BF, verified channels only** (`earth_unreached + radio`) | **-8.04966e-05** | **+9.62322e-05** |
| all shared terms incl. illustrative (`--include-illustrative`) | +0.128762 | +0.169457 |
| `results_sha256` (seed 42) | `3bbdbc8e89f27fd4…` | `cd8743def6e73eed…` |

Superseded pre-fix thin_disk_2d values (for audit only): radio term -5.62041e-05,
E[N] frequent_slow 0.0760993 / rare_fast 0.000179312.
