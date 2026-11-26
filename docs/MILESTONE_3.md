# Milestone 3: FastAPI Backend & Public Interactive Desk UI

## Objective
Turn the validated Milestone 1–2.5 Python core into a fast, inspectable **interactive research desk** (`src/fermi_desk/api.py` + `web/` React/TypeScript/Vite frontend) that a contributor or astronomer can launch locally with one command (`uv run fermi-desk serve`) and immediately explore, challenge, and replay.

Milestone 3 also closes the five small polish rulings from the Milestone 2.5 audit before exposing those payloads through the API.

## Explicit Scope Boundaries (DO NOT BUILD in Milestone 3)
- Do **NOT** build autonomous LLM agent loops, background worker daemons, or user authentication/OAuth accounts.
- Do **NOT** trigger live external network calls (IRSA, Gaia, VizieR) from any FastAPI endpoint or React page load. The entire web app and test suite must run **100% offline** against the verified local repository data and SQLite queue.
- Do **NOT** modify `data/catalogs/bl_692_targets.csv` (SHA-256 `c299d61d...16ee2`) or `data/calibrations/wise_sigma_sys_control.json` (SHA-256 `2da2a234...`). Both verified artifacts remain strictly locked.
- Do **NOT** implement the v0.2 finite-Galaxy disk or 288k-star `union_targets_by_pointing` merge policy yet (documented in `SPEC.md` v0.2).

---

## Part A — Milestone 2.5 Audit Polish (Complete First)

1. **Expose Small-Sample False-Alarm Bounds in `ir_drilldown`:**
   - Ensure `inference/ir_drilldown.py` emits structured fields for the withheld false-alarm rate (`0/77`, Clopper-Pearson 95% upper bound `0.047`, Rayleigh KS `p = 0.45`), the in-sample calibration tail (`2/77`, `p = 0.019`), and the zero-floor failure (`11/77`) so the UI displays them directly alongside the Tier-1 `wise_independent_control_fields` job.
2. **Inline Reason for `distance_scaled_eirp` Verified Headline Holdout:**
   - In `inference/engine.py` and `cli.py`, add a structured `headline_exclusion_notes` list on receipts whenever a channel is active in the run but held out of the default `verified_channels_only` headline BF (specifically explaining in `distance_scaled_eirp` mode that Radio is held out because the transmitter EIRP luminosity function is `illustrative_calibration`).
3. **AllWISE Atlas Smoothing Caveat:**
   - Attach an explicit `atlas_smoothing_caveat` string to the `ir_drilldown` output noting that `sigma_sys = 0.436"` absorbs AllWISE Atlas image smoothing together with astrometric registration, and document an unWISE coadd follow-up in `SPEC.md` under v0.2.
4. **Upgrade `hephaistos_blend_recovery_grid` Queue Runner:**
   - Update the Tier-0 queue runner in `queue/runners.py` so it evaluates **both** the legacy Suazo `0.500"` baseline and the verified `reproduced_archive` `0.436"` floor from `data/calibrations/wise_sigma_sys_control.json` side-by-side (matching `fermi-desk ir-drilldown` and preserving deterministic SHA-256 replay).
5. **Raise Future Default Latitude Floor (`scripts/run_wise_control_experiment.py`):**
   - Change the default Galactic latitude cut for future runs from `|b| > 20 deg` to `|b| > 30 deg` (avoiding inner-Galaxy crowding like Patch P01) without re-running or altering the locked 154-star artifact in `data/calibrations/wise_sigma_sys_control.json`.

---

## Part B — FastAPI Backend (`src/fermi_desk/api.py`)

Add `fastapi` and `uvicorn` to `pyproject.toml` (plus `httpx` in the `dev` dependency group for offline `TestClient` tests). Add a `fermi-desk serve` CLI command (`--host 127.0.0.1 --port 8000 --static-dir web/dist`) that serves both `/api/*` and the compiled Vite bundle if `web/dist` exists.

### Required Endpoints
All request/response models must use Pydantic v2 with strict input validation and safe bounds (e.g., valid enum literals, bounded `seed`, no shell execution, no arbitrary path traversal):

1. `GET /api/health`
   - Returns `status`, `git_sha`, `validation_passed` (bool + check count), catalog SHA-256, WISE calibration SHA-256, and receipt schema version.
2. `GET /api/registry`
   - Returns canonical parameters (`data/parameters.yaml`), all hypotheses grouped by Tier 1 / Tier 2 / Tier 3 (including full `monoculture_audit` blocks for Tier 3), and survey metadata with explicit `data_tier` tags.
3. `POST /api/compare`
   - Request body: `geometry_mode` (`thin_disk_2d` | `thick_disk_3d`), `radio_mode` (`flat_probability` | `distance_scaled_eirp`), `condition_on_earth_unreached` (`bool`), `channels` (subset of `["radio", "infrared", "solar_system"]`), `include_illustrative` (`bool`, default `False`), `seed` (`int`, default `42`).
   - Enforces every engine invariant (double-counting rejection, Tier-3 quarantine, `verified_channels_only` default headline, `headline_exclusion_notes`).
   - Caches identical `(git_sha, request_params)` responses in memory (LRU) so repeated UI toggles respond instantaneously while remaining deterministic.
4. `GET /api/radio/drilldown`
   - Returns the 692 Breakthrough Listen targets (`target_name`, `ra_deg`, `dec_deg`, `dist_pc`, `dist_source`, `dist_legacy_pc`, `eirp_min_w`, `eirp_revision_factor`), the full HIP 1692 triple-distance audit (`23.0 pc` legacy, `752.61 pc` DR2 Bailer-Jones, `788.71 pc` DR3 canonical), and the 16 structured GBT beam-sharing pairs.
5. `GET /api/infrared/drilldown?seed=42`
   - Returns the full `ir_drilldown` payload (LRU-cached by seed): side-by-side `0.500"` vs `0.436"` Centroid and Image-level blend-recovery grids + cell deltas, 154-star / 12-patch breakdown, withheld `0/77` (`95% UL < 4.7%`) and calibration `2/77` stats, zero-floor `11/77` failure, Candidate D & E anchor benchmarks, Suazo vs Zackrisson contamination flip table, and the `atlas_smoothing_caveat`.
6. `GET /api/adversary?seed=42`
   - Returns the full structured Adversary audit receipt (all checks, HIP 1692 table, 16 beam-sharing pairs, local validity violation, `results_sha256`).
7. `GET /api/queue`
   - Auto-seeds the SQLite queue (`runs/queue.sqlite`) if empty and returns two explicitly separated lists:
     - `tier1_archival`: ranked by `utility_score` (EIG bits/hr), with exact `expected_eig_bits`, CPU/review estimates, and copyable script/claim commands.
     - `tier0_compute`: ranked by `diagnostic_priority`, with `expected_eig_bits: 0.0`, `diagnostic_impact`, current status, `output_sha256`, and copyable CLI replay commands.
8. `POST /api/queue/{job_id}/run`
   - Allowed **strictly for Tier-0 compute jobs**. Returns HTTP `400` with a clear message + script path if called on a Tier-1 archival job.
   - Accepts `{ "seed": 42, "contributor": "local-desk-ui" }`, claims (if open) and executes the registered runner deterministically, persists to SQLite, and returns the completed job record + `output_json` + `output_sha256`.
9. `POST /api/queue/{job_id}/replay`
   - Replays a completed Tier-0 job using its stored `rng_seed` and `parameters_json`, returning `recorded_sha256`, `replayed_sha256`, and `match: bool`.
10. `GET /api/runs`
    - Lists the most recent JSON run receipts found in `runs/` (newest first, capped at 50) for the live desk activity feed.

### CORS & Security
- Allow localhost dev origins (`http://localhost:5173`, `http://127.0.0.1:5173`) via `CORSMiddleware`.
- Never render raw unsanitized HTML from receipts or catalog strings.

---

## Part C — Interactive Public Desk UI (`web/`)

Build a crisp, technical observatory/instrument frontend in `web/` using **React + TypeScript + Vite**. Keep dependencies lean, stable, and verified against npm (no beta/hallucinated package versions; prefer lightweight native SVG visualizations or one verified charting library such as `recharts`—avoid heavy multi-megabyte bundles).

### Visual & Epistemic Design Rules
- **Instrument aesthetic, not a marketing site:** High-contrast dark or technical slate palette, monospace numerals (`tabular-nums`) for all probabilities, log10 BFs, SHA-256 hashes, and coordinates.
- **Data-tier badges everywhere:** Visually distinguish `published_catalog` (emerald), `reproduced_archive` (cyan), and `illustrative_calibration` (amber) on every table, chart, and toggle. Never let illustrative data visually blend into verified bounds.
- **Persistent Provenance Header:** Displays active `git_sha`, validation badge (`30/30 OK`), locked catalog & WISE calibration short hashes, and active `rng_seed`.

### Required Five Views (Single-Page Navigation)

1. **Macro Desk (Default Homepage)**
   - **Flagship Question Header:** *"Can Breakthrough Listen's 692-star survey separate frequent/slow expanders from rare/fast expanders—once we condition on Earth remaining unreached?"*
   - **Interactive Control Strip:**
     - Geometry toggle: `thin_disk_2d` vs `thick_disk_3d`
     - Radio Mode toggle: `flat_probability` vs `distance_scaled_eirp`
     - Independent checkboxes: `Earth unreached`, `Radio`, `Infrared`, `Solar System`
     - Explicit warning-styled toggle: **`Include Illustrative Channels in Headline BF`** (OFF by default)
     - Seed selector (default `42`)
   - **Headline Verdict Card:** Prominent `HEADLINE log10 BF` readout, active policy badge (`verified_channels_only` vs loud red/amber `ILLUSTRATIVE CHANNELS INCLUDED — NOT AN OBSERVATIONAL BOUND` banner), exact channels entered, and inline `headline_exclusion_notes` (explaining immediately why Radio steps out of verified headline when `distance_scaled_eirp` is active).
   - **Hypothesis Comparison Table:** Shows `P(Earth reached)`, `E[N | Earth unreached]`, `P0 (MC +/- se)`, red `LOCAL APPROX VIOLATED` badge (`T_valid = 52.2 kyr`) on `rare_fast_expanders`, and cleanly separated `log10 LR vs null` and `raw log10 P(data | H)` tabs/columns.
   - **Quick Drill-Down Links:** Direct buttons jumping into Radio Drill-Down, Infrared Drill-Down, and Tier-3 Quarantine.

2. **Radio Drill-Down**
   - **HIP 1692 Spotlight Card:** Side-by-side comparison of legacy (`23.0 pc`, `1.12e12 W`), Gaia DR2 Bailer-Jones (`752.61 pc`, `1,071x`), and Gaia DR3 canonical (`788.71 pc`, `1,176x`), with citation to Wlodarczyk-Sroka et al. (2020).
   - **692-Target Scatter Plot (Log-Log):** Distance (`pc`) vs Minimum Detectable EIRP (`W`), colored by `dist_source` (`gaia_dr3_parallax` vs `isaacson_2017_vizier`), with interactive hover tooltips and HIP 1692 highlighted.
   - **16 GBT Beam-Sharing Pairs Table:** Searchable/sortable table showing pair names (`GJ 702A/B`, etc.), separation in arcmin, fraction of `8.4'` FWHM beam, and `counted_once_each` verification badge.

3. **Infrared Drill-Down (One Click Down)**
   - **Live Empirical Calibration Card (`reproduced_archive`):** Highlights the 154-star / 12-patch AllWISE measurement (`sigma_sys = 0.436"` vs `0.500"` Suazo legacy), withheld `0/77` false alarms (`95% UL < 4.7%`, Rayleigh KS `p = 0.45`), calibration `2/77` tail hint (`p = 0.019`), zero-floor `11/77` breakdown, and the explicit **Atlas smoothing caveat**.
   - **Interactive Blend-Recovery Heatmap Matrix:** Angular separation (`0.5"`–`6.0"`) × background flux fraction (`0.1`–`0.9`), with toggles for:
     - Screen: `Centroid` vs `Image-level (Δχ²)`
     - Floor: `0.436" (Measured)` vs `0.500" (Legacy Suazo)` vs `Delta (+gain)`
   - **Candidates D & E JWST Anchor Card:** Explicit visual comparison explaining why Candidate D (`0.59σ`, `p = 0.307`) slips past catalog centroid screens while JWST/MIRI resolved a `~1"` background galaxy.
   - **Contamination Convention Flip Panel:** Side-by-side Suazo (2024) vs Zackrisson (2026) table showing the likelihood ratio flip (`0.33–0.49` vs `1.37`) with zero editorial bias toward either convention.

4. **Adversary Feed & Tier-3 Quarantine Shelf**
   - **Adversary Checks Table/Cards:** All structured checks from `/api/adversary` with status pills (`pass` / `finding` / `warning`), quantitative impact, citations, and copyable `repro_command` buttons.
   - **Tier-3 Quarantine Shelf:** Physically separated section rendering `zoo_hypothesis` and `dark_forest` with a `QUARANTINED — ZERO EMPIRICAL SCORE` badge and full **Monoculture Fragility Audit** metrics (`required_compliance_fraction`, `timescale_gyr`, `defector_vulnerability`).

5. **Split BOED Action Queue & Live Receipts**
   - **Tab 1 — Tier-1 Archival Queue (EIG bits/hr):** Ranked table putting `wise_independent_control_fields` (`1.058 bits`) and `wlodarczyk_sroka_288k_background_extension` (`0.0145 bits` physical vs `0.86` flat) front and center, linking the `0/77 < 4.7%` small-sample bound directly to Job #1, with copyable CLI claim/script snippets.
   - **Tab 2 — Tier-0 Compute Queue (Diagnostic Priority):** Shows `diagnostic_impact`, `EIG = 0.0 bits` note, status, `output_sha256`, and interactive **`Run (seed 42)`** and **`Verify Replay`** buttons wired to `/api/queue/{job_id}/run` and `/replay`—including an inline results viewer for `lookback_window_t_sweep` showing the exact `T_valid = 52.2 kyr` boundary.
   - **Recent Receipts Feed:** Compact timeline of local run receipts from `/api/runs` with `git_sha`, `rng_seed`, and hash badges.

---

## Mandatory Verification Suite (100% Offline)

1. **Backend & Invariant Tests (`tests/test_api.py` + existing 123 tests):**
   - `uv run pytest -v` passes **100% offline** (verified with sockets blocked).
   - `tests/test_api.py` uses FastAPI `TestClient` (with `tmp_path` SQLite override so it never pollutes `runs/queue.sqlite`) and verifies:
     - `/api/health`, `/api/registry`, `/api/radio/drilldown`, `/api/infrared/drilldown`, `/api/adversary`, `/api/queue`, `/api/runs` all return `200` and valid JSON schemas.
     - Default `POST /api/compare` enforces `headline_policy == "verified_channels_only"`; enabling `include_illustrative=True` updates the policy and BF; `distance_scaled_eirp` returns a non-empty `headline_exclusion_notes`.
     - `POST /api/queue/wise_independent_control_fields/run` returns HTTP `400` (Tier-1 refusal).
     - `POST /api/queue/lookback_window_t_sweep/run` followed by `/replay` returns `match == True` with identical `output_sha256`.
     - Tier-3 hypotheses appear in `/api/registry` with `monoculture_audit` and never receive empirical scores in `/api/compare`.
2. **Python Lint & Schema Checks:**
   - `uv run ruff check .` and `uv run ruff format --check .` pass cleanly.
   - `uv run fermi-desk validate` passes all checks (including untouched catalog and WISE calibration SHA-256 hashes).
3. **Frontend Build & Typecheck (`web/`):**
   - `npm ci` (or `npm install`), `npm run typecheck` (`tsc --noEmit`), and `npm run build` succeed cleanly on Windows/PowerShell with zero TypeScript or bundler errors.
   - At least one lightweight frontend unit test suite (`vitest`) verifying data-tier badge classification, illustrative-warning banner gating, and SHA-256 truncation/display helpers.
4. **End-to-End Local Serve Smoke Test:**
   - Confirm `uv run fermi-desk serve --static-dir web/dist` serves `GET /` (HTML from built Vite bundle) and `GET /api/health` cleanly via local test/inspection without hanging.
