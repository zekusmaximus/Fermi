# v0.2 Roadmap: Finite Galaxy, Unsmoothed unWISE, and Background-Field Scaling

## Objective
`v0.1.2` cleared all four Public-Launch Exit Gates and exposed three concrete scientific bottlenecks plus one deployment gap. `v0.2` resolves them in four strictly scoped, independently verifiable sprints without modifying any locked `v0.1.x` empirical artifacts or weakening offline reproducibility.

## Universal v0.2 Guardrails (Apply to Every Sprint)
1. **Locked v0.1.x Artifacts Remain Immutable:** Never overwrite or alter `data/catalogs/bl_692_targets.csv` (`c299d61d...16ee2`), `data/calibrations/wise_sigma_sys_control.json` (`2da2a234...`), or `data/calibrations/wise_fits_gate2_withheld.json` (`d4818c10...`). New archive campaigns write new versioned artifacts with independent SHA-256 pins and provenance receipts.
2. **Strict Data-Tier Isolation:** `verified_channels_only` remains the default Headline BF policy. No `illustrative_calibration` prior may enter a verified headline or visually blend with `published_catalog` / `reproduced_archive` outputs.
3. **Zero Data Leakage & Clean-SHA Pre-Fetch Freeze:** Every live archive experiment (unWISE coadds, large-N WISE controls, 288k Gaia background fields) requires a clean, committed `code_git_sha` before querying external servers and a pre-fetch freeze of all templates/thresholds on the calibration split before fetching or touching withheld data.
4. **No Surprise Quota:** Null separations, tighter negative results, or persisting unWISE deblending limits pass identically to positive findings. Never retune on a withheld split.
5. **100% Offline CI & Deterministic Replay:** `uv run pytest -v` (with non-loopback sockets blocked), `uv run fermi-desk validate`, FastAPI routes, and `web/` views remain completely offline and seed-deterministic.

---

## Sequencing Overview

| Sprint | Focus | Primary Scientific Question Resolved | Live Archive Fetch? |
|---|---|---|---|
| **v0.2.0** | Finite Galactic Disk (`R_gal = 15 kpc`) & `T` Marginalization | Does removing the infinite-disk breakdown (`T_valid = 52.2 kyr`) change the Frequent/Slow vs. Rare/Fast separation? | No (100% offline physics) |
| **v0.2.1** | `union_targets_by_pointing` & 288k Background Ingestion | Can Wlodarczyk-Sroka's 288,315 Gaia field stars sharing `bl_gbt_lband_2017` be merged without double-counting, and do real distances match the `0.0145 bits` EIG prediction? | Yes (opt-in script + locked artifact) |
| **v0.2.2** | Unsmoothed `unWISE` W3 Coadds + Correlated-Noise Screen | Does removing AllWISE Atlas matched-filter smoothing (`13.69"` FWHM) rescue image-level two-PSF deblending on a fresh `|b| > 30 deg` split? | Yes (pre-fetch freeze on fresh split) |
| **v0.2.3** | Large-N WISE Control Campaign & Public-Host Hardening | Tighten the `0/77` (`95% UL < 4.7%`) false-alarm bound (`wise_independent_control_fields`, #1 BOED job) and lock `--read-only` public hosting against cache-miss CPU load. | Yes (large-N `|b| > 30 deg` run) |

Execute **v0.2.0 first** (pure offline physics, zero external API risk), then **v0.2.1** or **v0.2.2** in either order depending on whether you want to push the radio or infrared frontier first, and finish with **v0.2.3**.

---

## Sprint 1 (`v0.2.0`) — Finite Galactic Disk Geometry (`R_gal = 15 kpc`) & Lookback `T` Marginalization

### Problem Statement
At `v = 0.5c`, `rare_fast_expanders` reaches `v * T = 153 kpc` at the default `T = 1 Myr`, violating the local infinite-disk validity boundary (`8 kpc`) for every `T > 52.185 kyr`. The desk currently flags `LOCAL APPROX VIOLATED` honestly, but cannot make valid macro inferences for fast expanders at Myr timescales.

### Deliverables
1. **New Explicit Geometry Mode (`finite_disk_3d`) in `src/fermi_desk/models/expansion.py`:**
   - Model an axisymmetric Galactic disk centered on the Galactic Center (`R_sun = 8.178 kpc`, `z_sun = 20.8 pc`, `R_gal = 15.0 kpc`) with explicit vertical profile choice: uniform slab (`|z| <= h_z`) or exponential disk (`sech^2` / `exp(-|z|/h_z)`, `exp(-R/R_d)` with `R_d = 2.6 kpc` documented as a parameter in `data/parameters.yaml`).
   - Keep `thin_disk_2d` and `thick_disk_3d` intact and byte-identical for backwards-compatible benchmark comparison (`results_sha256 = 4b70118a...` unchanged).
   - Compute `P(Earth reached)` and conditional target reach probabilities `P(star reached | Earth unreached)` via seeded, unit-checked Monte Carlo / importance-sampled past causal intersections bounded strictly inside `(R <= R_gal, |z| <= z_max)`.
2. **Lookback-Window `T` Marginalization (`inference/engine.py`, `cli.py`):**
   - Support an optional `--marginalize-lookback` mode integrating across a documented log-uniform prior on `T` (e.g., `50 kyr` to `10 Myr`) rather than fixing `T = 1 Myr`.
   - Preserve exact single-`T` defaults so existing `v0.1.x` CLI calls remain unchanged.
3. **API & UI Exposure (`api.py`, `web/src/views/MacroDesk.tsx`):**
   - Add `finite_disk_3d` as a third option in the Macro Desk geometry selector alongside `thin_disk_2d` and `thick_disk_3d`, showing the `LOCAL APPROX VIOLATED` badge clear automatically when finite boundaries are active.

### Acceptance Criteria (All Offline)
- **Physical limiting invariants (`tests/test_invariants.py`):**
  - As `v * T << R_gal - R_sun` (e.g., `T = 10 kyr`), `finite_disk_3d` with uniform density converges to `thick_disk_3d` within Monte Carlo tolerance.
  - Zero-emission invariance (`p_radio = 0` -> `LR = 1.0`), Earth-unreached conditioning monotonicity, and zero-speed limits hold identically for `finite_disk_3d`.
  - `rare_fast_expanders` at `T = 1 Myr` reports `local_approx = valid_finite_galaxy` under `finite_disk_3d` while continuing to report `VIOLATED` under `thin_disk_2d` and `thick_disk_3d`.
- All existing `194` Python tests, new finite-disk tests, `fermi-desk validate`, and `web/` checks pass 100% offline.

---

## Sprint 2 (`v0.2.1`) — `union_targets_by_pointing` Merge Policy & 288k Background Ingestion

### Problem Statement
Wlodarczyk-Sroka et al. (2020) identified 288,315 Gaia stars serendipitously inside the same `bl_gbt_lband_2017` GBT L-band beams as the 692 Enriquez et al. (2017) primary targets. The current duplicate-pointing guard (`keep_first` or hard rejection) prevents combining primary and background catalogs that legitimately share a telescope pointing group.

### Deliverables
1. **Per-Pointing Target Union Policy (`registry.py`, `inference/engine.py`, `adversary/checks.py`):**
   - Implement an explicit `pointing_merge_policy: union_targets_by_pointing` option for surveys sharing a `pointing_group_id`.
   - Deduplicate stars strictly by canonical `gaia_dr3_source_id` (falling back to angular match `< 3"` only when Gaia ID is missing), ensuring each physical star enters the joint likelihood **at most once** across primary + background catalogs.
   - Explicitly forbid and test against `keep_first` (which would silently drop either the primary or background survey).
   - Keep `reject_duplicate_pointing_group` as the default when surveys do not declare a verified parent-pointing relationship.
2. **Opt-In Archival Ingestion Script (`scripts/fetch_wlodarczyk_sroka_288k.py`):**
   - Pulls the published Wlodarczyk-Sroka et al. (2020) background-field catalog / Gaia cross-match (or reproducible cone queries around the 692 GBT `8.4'` FWHM beams with primary-beam Gaussian attenuation off-axis), writes a compressed/indexed local catalog artifact with exact SHA-256 and provenance receipt (`reproduced_archive`), and supports `--promote-from` strictly from a clean commit SHA.
   - Store summary shell distributions + deterministic subsampled/binned validation tables in Git (keeping Git artifact footprint compact, `< 5 MB`, while full raw catalogs stay in gitignored `data/cache/`).
3. **Updated BOED EIG & Radio Drill-Down:**
   - Compare actual achieved information gain (bits) against the Milestone 2 synthetic-proxy prediction (`0.01452 bits` physical vs. `0.827 bits` flat), marking `wlodarczyk_sroka_288k_background_extension` completed with a verified `output_sha256` if executed.

### Acceptance Criteria
- Double-counting invariant test: merging the 692 primary stars with overlapping background stars produces **0 duplicated Gaia IDs** and rejects any attempt to invoke `keep_first`.
- Off-axis primary beam attenuation increases `EIRP_min` monotonically with radial offset from beam center (`astropy.units` in Watts).
- Existing `bl_692_targets.csv` (`c299d61d...16ee2`) remains untouched and byte-identical.

---

## Sprint 3 (`v0.2.2`) — Unsmoothed `unWISE` W3 Coadds & Correlated-Noise Screen (Fresh Split)

### Problem Statement
In Gate #2 (`v0.1.2`), real AllWISE W3 Atlas cutouts revealed an empirical PSF FWHM of `13.69"` (vs. `6.5"` nominal) due to Atlas matched-filter coadd smoothing, causing the frozen scalar-variance two-PSF `delta_chi2` screen to fail out of sample (`5/77 = 6.5%` null false alarms, `0.934` blend survival within `3.25"`). Retuning on the locked 77 withheld stars is forbidden (`LeakageError`).

### Deliverables
1. **Fresh High-Latitude Control Sample (`|b| > 30 deg`):**
   - Sample a **strictly disjoint** set of isolated Gaia M dwarfs (`N >= 200`, `100` calibration / `100` withheld, zero Gaia `source_id` overlap with the 154 stars in `wise_sigma_sys_control.json`), using the upgraded `|b| > 30 deg` floor across `>= 12` patches to eliminate inner-Galaxy crowding (such as Patch P01).
2. **Unsmoothed `unWISE` W3 Cutout Fetcher (`scripts/run_unwise_gate2.py`):**
   - Fetch unsmoothed `unWISE` W3 intensity (`-img-m.fits`) and inverse-variance (`-invvar-m.fits`) cutouts into gitignored `data/cache/unwise_fits/` with per-file SHA-256 manifests.
   - Enforce strict pre-fetch freeze: stack the empirical `unWISE` W3 PSF and freeze all screen thresholds strictly on the calibration half *before* fetching a single withheld cutout.
3. **Upgraded Image-Level Noise Model (`src/fermi_desk/models/hephaistos_fits.py`):**
   - Implement two predeclared image-level candidates (both frozen on calibration before withheld fetch):
     1. **Correlated-noise generalized least squares (GLS) two-PSF `delta_chi2`** using an empirical pixel covariance matrix estimated strictly from calibration sky annuli.
     2. **Non-parametric empirical calibration-quantile threshold** (no assumed `chi2(3)` tail shape).
   - Run the identical **Arm A (Off-Center Blend)** vs. **Arm B (Matched Centered Stellar Excess)** adversarial protocol plus the centroid screen on the fresh withheld split.
4. **Side-by-Side UI Comparison (`InfraredDrilldown.tsx`):**
   - Add `unWISE W3 (Unsmoothed — Fresh Split)` as a third toggle alongside `AllWISE W3 Atlas (Gate #2)` and `Synthetic Gaussian PSF`, showing empirical PSF FWHM comparison (`unWISE` vs. `13.69"` Atlas vs. `6.5"` nominal), Arm A blend survival, and Arm B centered-excess rejection.

### Acceptance Criteria
- Disjoint-sample invariant test: `validate` asserts zero `gaia_dr3_source_id` intersection between the `v0.1.2` AllWISE split (`154` stars) and the `v0.2.2` `unWISE` split.
- `frozen_at_utc` strictly precedes the earliest withheld-cutout fetch timestamp; `code_git_sha` is a clean commit SHA.
- Both `v0.1.2` locked WISE artifacts (`2da2a234...` and `d4818c10...`) remain untouched and byte-identical.
- Whatever the `unWISE` image-level screen yields—rescue or continued close-blend blind spot—is promoted intact without post-hoc tuning.

---

## Sprint 4 (`v0.2.3`) — Large-N WISE False-Alarm Campaign & Public-Host Hardening

### Problem Statement
Two practical gaps remain before hosting the desk on a public URL or closing BOED Archival Job #1 (`wise_independent_control_fields`, `1.058 bits` EIG):
1. With `n = 77` withheld stars, `0/77` only bounds the centroid false-alarm rate below `4.68%` (95% Clopper-Pearson), and the calibration half showed `2/77` (`p = 0.0187`), hinting at a heavier-than-Rayleigh tail.
2. In `--read-only` mode, unauthenticated visitors passing arbitrary `seed` values to `POST /api/compare` or `GET /api/infrared/drilldown` can trigger repeated uncached Monte Carlo runs, and `GET /api/queue` performs a one-time SQLite seed write if the DB file is absent.

### Deliverables
1. **Large-N Control Execution (`wise_independent_control_fields`):**
   - Run a large-N isolated M-dwarf control query (`N >= 1,000` withheld controls across `>= 24` patches at `|b| > 30 deg`) using the frozen `sigma_sys = 0.4356"` floor to push the 95% false-alarm upper bound below `0.4%` (if `0` false alarms) or directly measure the true non-Rayleigh tail rate.
   - Promote the verified artifact (`data/calibrations/wise_large_n_false_alarm.json`, `reproduced_archive`), wire it into `fermi-desk validate`, and update BOED Queue Job #1 status/evidence accordingly.
2. **Public-Host Read-Only Hardening (`api.py`, `cli.py`, `queue/store.py`):**
   - When `read_only=True`:
     - Add `--allowed-seeds` (defaulting to `[42]` when `read_only=True`) so public `/api/compare`, `/api/infrared/drilldown`, and `/api/adversary` requests stay strictly cache-bound and return HTTP `400` on disallowed seeds with a note to run custom seeds locally.
     - Pre-warm core caches (`seed=42` default compare, IR drill-down, adversary) at startup (optional `--prewarm` flag) and refuse to write to `runs/queue.sqlite` on `GET /api/queue` if the file is pre-seeded read-only.
   - Add a `Dockerfile` / `compose.yaml` example running `fermi-desk serve --host 0.0.0.0 --read-only` against a pre-built `web/dist` and pre-seeded read-only SQLite queue.

### Acceptance Criteria
- `tests/test_api.py` verifies `read_only=True` rejects non-allowlisted seeds with HTTP `400`, serves allowlisted `seed=42` from cache with zero SQLite writes, and keeps `403` on all queue mutations.
- `fermi-desk validate` verifies all locked SHA-256 hashes (both `v0.1.x` and `v0.2.x` artifacts) 100% offline.
- Fresh-clone Gate #3 replay passes cleanly on `v0.2.3`.

---

## Explicitly Out of Scope for v0.2 (Defer to v0.3+)
- Autonomous LLM agent loops or external job worker daemons.
- Multi-user accounts, OAuth, or public job submission endpoints.
- Non-radio / non-mid-IR channels (optical laser SETI, technosignature atmospheric spectroscopy) until the finite-Galaxy radio + unWISE IR channels are fully locked.