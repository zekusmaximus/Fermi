# Milestone 1: Core Factor Graph, Macro Inference Engine & Scientific CI

## Objective
Build the reproducible Python core (`src/fermi_desk/`) and seed data (`data/`) for the Fermi Paradox Empirical Constraint Desk. At the end of Milestone 1, a single CLI command must reproduce the headline multi-channel Fermi comparison, expose 2D-vs-3D geometry and distance-scaled EIRP sensitivity, enforce Tier-3 quarantine, and pass all Scientific CI invariants.

## Explicit Scope Boundaries (DO NOT BUILD in Milestone 1)
- Do **NOT** build the React/Vite frontend (`web/`).
- Do **NOT** build FastAPI endpoints (`api.py`).
- Do **NOT** write live NASA/IRSA HTTP scrapers or autonomous LLM agent loops yet.
- Do **NOT** build the full image-level AllWISE FITS injection pipeline yet (Milestone 2).

---

## Required Deliverables

### 1. Project Setup (`pyproject.toml`)
- Python 3.11+, managed with `uv`.
- Core dependencies: `pydantic>=2.7`, `pyyaml`, `numpy`, `scipy`, `astropy`, `pandas`, `typer`, `rich`.
- Dev dependencies: `pytest`, `ruff`.

### 2. Seeded Data Layer (`data/`)
Create validated YAML/CSV files:
- `data/parameters.yaml`: Canonical parameter registry matching `SPEC.md` Section 4.1 (at least 8 physical parameters with valid `astropy.units` strings and prior bounds).
- `data/hypotheses/`: At least 6 hypothesis YAML files:
  - Tier 1: `frequent_slow_expanders.yaml`, `rare_fast_expanders.yaml`, `partial_dyson_conspicuous.yaml`, `quiet_expanders.yaml`
  - Tier 2: `hard_steps_rare_intelligence.yaml`
  - Tier 3 (Quarantined): `zoo_hypothesis.yaml`, `dark_forest.yaml` (must include complete `monoculture_audit` blocks).
- `data/surveys/`:
  - `bl_enriquez_2017_692.yaml` (`channel: radio`, `pointing_group_id: bl_gbt_lband_2017`)
  - `hephaistos_wise_mid_ir.yaml` (`channel: infrared`, `pointing_group_id: wise_gaia_hephaistos`, containing a structured table of 15 `illustrative_calibration` completeness/false-positive pairs plus citation hooks to Suazo et al. 2024, Ren et al. 2026, and Zackrisson et al. 2026)
  - `solar_system_null.yaml` (`channel: solar_system`, `pointing_group_id: local_solar_system`)
- `data/catalogs/bl_692_targets.csv`:
  - Include a clean, deterministic schema (`target_name, ra_deg, dec_deg, dist_pc, dist_source, pointing_group_id`).
  - Include real published entries for key anchor stars (explicitly including `HIP 1692` with both its legacy 23.0 pc value and Gaia-revised 752.61 pc metadata field so Milestone 2's Adversary can toggle it) plus either the full real Enriquez et al. (2017) 692-star table if fetched/verified from a public canonical source, OR a clearly tagged `data_tier: synthetic_proxy_692` fixture with a manifest note and fetch script `scripts/fetch_bl_692.py` to pull the exact published table from Berkeley SETI / VizieR. **Never mislabel synthetic coordinates as `published_catalog`.**

### 3. Schemas & Factor Graph Loader (`schemas.py`, `registry.py`)
- Validate every parameter's `unit` string by instantiating `astropy.units.Unit(unit)`. Reject invalid units at load time.
- Enforce Tier-3 schema rules: `tier == 3` MUST provide `quarantine_reason` and `monoculture_audit`, and MUST raise a validation error if passed directly into Tier-1 empirical likelihood scoring.
- Enforce `pointing_group_id` deduplication in `registry.py`: passing two surveys with the same `pointing_group_id` into a joint inference run MUST raise `DoubleCountingError` unless explicitly resolved via a declared merge policy.

### 4. Physics & Forward Models (`src/fermi_desk/models/`)
- **`expansion.py` (Spatial Expansion & Earth-Unreached Conditioning):**
  - Implement a seeded Monte Carlo / semi-analytic Poisson-Boolean bubble model in both:
    - `thin_disk_2d`: Galactic disk origins with rate `Gamma_2d` and expansion speed `v_exp` over lookback window `T`.
    - `thick_disk_3d`: 3D slab with vertical scale height `h_z` (default 300 pc) and volumetric rate derived consistently from `Gamma_2d / (2 * h_z)`.
  - Given target 3D Cartesian coordinates relative to Earth `(x, y, z)` in pc (converted via `astropy.units`):
    - Compute unconditional reach probability `P(reached)` at Earth.
    - Compute expected number of reached target stars conditioned on Earth being unreached: `E[N_reached | Earth unreached]`.
    - Compute exact/correlated survival probability `P(0 detections | Earth unreached)` and the Fréchet/Markov lower bound under arbitrary positive spatial correlation across shared expansion fronts.
- **`radio.py` (Breakthrough Listen Forward Model):**
  - Support two explicit response modes:
    - `flat_probability`: uniform detection probability `p_radio` per reached star (enables direct replication of the baseline benchmark).
    - `distance_scaled_eirp`: computes minimum detectable EIRP per star as `EIRP_min = 4 * pi * d^2 * F_min` using `astropy.units` (Watts), evaluating detection probability against a parameterized transmitter luminosity function.
- **`infrared.py` (Mid-IR Calibration Sensitivity Model):**
  - Given a hypothesis's `f_ir_loud` and a survey calibration pair `(completeness, false_positive_rate)`, compute the likelihood ratio comparing `quiet_expanders` vs. `partial_dyson_conspicuous`.
  - Provide a sweep utility across the 15 calibration points in `hephaistos_wise_mid_ir.yaml` that outputs the min/max likelihood ratio and explicitly flags whether the preference reverses across the calibration bracket.
- **`solar_system.py`:**
  - Simple, transparent local-reach null constraint conditioned cleanly on Earth-unreached status so it never double-counts the anthropic condition.

### 5. Joint Inference Engine & CLI (`inference/engine.py`, `cli.py`)
- `engine.py` must allow toggling any subset of `channels` (`radio`, `infrared`, `solar_system`) AND independently toggling `condition_on_earth_unreached: bool`.
- Emit a structured JSON `RunReceipt` for every run containing: `run_id`, `git_sha` (or `"dirty"`), `rng_seed`, `geometry_mode`, `radio_mode`, `active_surveys`, `data_tiers_used`, and full numeric results.
- `cli.py` (using `typer` + `rich`) must provide:
  - `fermi-desk validate` — validates all YAML schemas, units, and checksums.
  - `fermi-desk compare --geometry thin_disk_2d --radio-mode flat_probability --seed 42` — prints the headline comparison table (`frequent_slow_expanders` vs. `rare_fast_expanders`), Tier-3 quarantine notice, and IR calibration reversal summary, and writes a JSON receipt to `runs/`.
  - `fermi-desk compare --geometry thick_disk_3d --radio-mode distance_scaled_eirp --seed 42` — runs the upgraded physical mode for direct side-by-side comparison.

### 6. Mandatory Scientific CI Suite (`tests/`)
All of the following tests MUST exist and pass via `uv run pytest`:
1. **Unit & Schema Validation (`test_schemas.py`):** Rejects bad `astropy` units; rejects Tier-3 hypotheses missing `monoculture_audit`; blocks Tier-3 hypotheses from empirical scoring.
2. **Zero-Emission Invariance (`test_invariants.py`):** Setting `p_radio = 0.0` yields `likelihood_ratio == 1.0` (within `1e-12`) across all expansion speeds and origin rates. Same for `f_ir_loud = 0.0` in the infrared channel.
3. **Analytical Boundary Limits (`test_invariants.py`):**
   - As `v_exp -> 0`, expected reached targets approaches `0`.
   - With `n_targets == 0`, survey likelihood equals `1.0`.
   - Conditioning on Earth unreached strictly reduces or preserves expected reached nearby targets relative to unconditional expectation when bubbles have finite spatial correlation radius.
4. **2D vs. 3D Scaling Symmetry Audit (`test_invariants.py`):**
   - In `thin_disk_2d`, doubling `v_exp` and multiplying `Gamma_2d` by `0.25` preserves single-point reach probability `P(reached)` within tolerance.
   - In `thick_disk_3d` (with bubble radius `v_exp * T < h_z`), the same `2x / 0.25x` transform DOES NOT preserve `P(reached)` (explicitly verifying the 3D volume scaling difference).
5. **Double-Counting Guard (`test_invariants.py`):** Passing two surveys sharing `pointing_group_id` raises `DoubleCountingError`.
6. **Deterministic Replay (`test_models.py`):** Running `compare` twice with `--seed 42` yields byte-identical numerical metrics in the output JSON receipts.
7. **IR Calibration Reversal (`test_models.py`):** Sweeping the 15 IR calibrations demonstrates preference reversal (`min_lr < 1.0 < max_lr`).

---

## Acceptance & Handoff Output
When Milestone 1 is complete, produce a short handoff report containing:
1. Full file tree of created files.
2. Verbatim output of `uv run pytest -v` and `uv run ruff check .`.
3. Verbatim output of `uv run fermi-desk validate` and both `fermi-desk compare` commands (`thin_disk_2d` vs `thick_disk_3d`).
4. Explicit list of any `data_tier: synthetic_proxy_692` or `illustrative_calibration` placeholders still awaiting live archive fetch.