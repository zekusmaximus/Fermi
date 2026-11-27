# Fermi Paradox Empirical Constraint Desk — Planning Ledger

Single source of truth for architectural rulings, locked data artifacts, canonical baseline numbers, release history, and deferred backlog. Both the planning/audit agent (Neon) and the local coding agent treat this ledger alongside `AGENTS.md` and `SPEC.md` as authoritative.

---

## 1. Current Release State

- **Latest Release:** `v0.2.0` (Sprint 1 complete: `finite_disk_3d` Galactic geometry + lookback-window `T` marginalization; all four Public-Launch Exit Gates cleared in `v0.1.2`)
- **Release & Milestone History:**
  - **Milestone 1:** Core factor graph, 2D/3D expansion & radio/IR kernels, verified BL 692 catalog (`76` tests, `28/28` validate)
  - **Milestone 2:** Hephaistos IR drill-down, Adversary runner, BOED SQLite queue, distance-preserving 2D projection fix (`106` tests, `29/29` validate)
  - **Milestone 2.5a (`b02ec7df820d`):** Audit rulings, default `verified_channels_only` Headline BF, Tier-0 diagnostic queue, multi-patch WISE control script
  - **Milestone 2.5b (`7f22a3691833`):** Live 154-star / 12-patch AllWISE `sigma_sys` calibration (`reproduced_archive`), side-by-side IR drill-down (`123` tests, `30/30` validate)
  - **Milestone 3 (`727daa7c` baseline):** M2.5 polish rulings, offline FastAPI backend (`src/fermi_desk/api.py`), React/TS/Vite desk (`web/`), receipt schema v4 (`159` Python + `13` Vitest, `30/30` validate)
  - **`v0.1.1` (`0667f973a2b4`):** Live `Overview` default landing tab, `serve --read-only` deploy safeguard (exit code `2` on non-loopback read-write), `240 s` EIRP sweep estimate, Gate #3 fresh-clone replay verified (`174` Python + `23` Vitest, `30/30` validate)
  - **`v0.1.2a` (`51a73ca4ff93`):** Frozen real-FITS matched centered-vs-off-center Gate #2 pipeline (`models/hephaistos_fits.py`, `scripts/run_wise_fits_gate2.py`, `LeakageError`, `docs/V0_1_2_GATE_2.md`)
  - **`v0.1.2b` / `v0.1.2`:** Promoted real-FITS Gate #2 withheld artifact (`reproduced_archive`), wired into IR drill-down, `/api/health`, `InfraredDrilldown.tsx`, and `Overview.tsx` (`194` Python + `26` Vitest, `31/31` validate, `45` ruff-formatted files)
  - **`v0.2.0` (Sprint 1):** Axisymmetric finite Galactic disk (`finite_disk_3d`, `R_gal = 15 kpc`, `uniform` and `exponential` profiles), likelihood-space `logsumexp` lookback-window `T` marginalization (`[50 kyr, 10 Myr]`, 16-node Gauss–Legendre in `ln T`), CLI/API/Macro Desk UI support (`275` Python + `38` Vitest, `36/36` validate, `48` ruff-formatted files)
- **Runtime Invariant:** `uv run pytest -v`, `uv run fermi-desk validate`, FastAPI routes, and `web/` views run **100% offline** (non-loopback sockets and DNS lookups blocked globally in `tests/conftest.py`).

---

## 2. Locked Empirical Artifacts (Do Not Modify Without New Provenance Run)

All locked artifacts below are protected against CRLF checkout drift via `.gitattributes` (`-text`).

| Artifact Path | Data Tier | Pinned SHA-256 (Full 64-Char) | Source Commit | Verified Scope |
|---|---|---|---|---|
| `data/catalogs/bl_692_targets.csv` | `published_catalog` (Gaia DR3 cols: `reproduced_archive`) | `c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2` | M1 | 692/692 Enriquez (2017) targets matched to Isaacson (2017) Table 1 (`vizier.cfa.harvard.edu`); 517 Gaia DR3 parallaxes (`SNR >= 5`, `gaia.ari.uni-heidelberg.de`), 175 legacy fallback |
| `data/calibrations/wise_sigma_sys_control.json` | `reproduced_archive` | `2da2a234ee0ccb07fb571431f5f178e362ff4d9d4f96d4ee878b921916127b7a` | `b02ec7df820d` | 154 isolated Gaia M dwarfs (`RUWE < 1.4`, `cc_flags=0000`, `ext_flg=0`, W3 SNR `[8, 20]`, no neighbor `< 15"`) across 12 patches (`\|b\| > 20°`); 77 cal / 77 withheld |
| `data/calibrations/wise_fits_gate2_withheld.json` | `reproduced_archive` | `d4818c100b02cea98d5beafcdea66e679c072e5d47e28fe8d3a960f78c64da99` | `51a73ca4ff93` | 154/154 real IRSA AllWISE W3 Atlas intensity + uncertainty cutout pairs (308 FITS, 0 failures); PSF & thresholds frozen at `2026-11-26T21:01:58Z` before any withheld fetch |

---

## 3. Canonical Baseline Numbers (`seed = 42`, Receipt Schema v4)

### Macro Desk (`frequent_slow_expanders` vs. `rare_fast_expanders`, `flat_probability`, `verified_channels_only`)

| Configuration | HEADLINE `log10 BF` | `earth_unreached` | `radio` | Canonical `results_sha256` |
|---|---:|---:|---:|---|
| `thin_disk_2d` (`T = 1 Myr`) | **`-8.04966e-05`** | `-8.67362e-19` | `-8.04966e-05` | `4b70118aaded7861cb14d77379843dfacec2d4c517bbce5efc9654e139fbc269` |
| `thick_disk_3d` (`T = 1 Myr`) | `+9.62322e-05` | `+1.49587e-04` | `-5.3355e-05` | `be2053912fc0a1d6970878a473dc3ee37d92d62f7904e73dfe369762634f5d41` |
| `finite_disk_3d` uniform (`T = 1 Myr`) | **`-0.00406404`** | `-0.00401261` | `-5.14293e-05` | `87a746589ac07447fdad129f335cd32f164dc11e170220f7a7a81ce0b13b9422` |
| `finite_disk_3d` exponential (`T = 1 Myr`) | **`-0.00364308`** | `-0.00359895` | `-4.41212e-05` | `741ca0d7a3be615790ea610b6a717bb6f2936328a297166b3289bf04c8e51c9e` |
| `finite_disk_3d` uniform, `--marginalize-lookback` | **`-0.0856571`** | `-0.0855352` | `-1.21949e-04` | `ffcb470ae44a900ef74eda1b88e9b124b07356210f4f7b871ca801a059d48ec6` |
| `finite_disk_3d` exponential, `--marginalize-lookback` | `-0.0937013` | `-0.0935913` | `-1.09938e-04` | `b6e3eaafb145752ea55e8c187817b22a001e91897b0c5f92dece7a1d4a34bcf1` |
| `thin_disk_2d`, `--marginalize-lookback` | `-1.72762e-04` | `0` | `-1.72762e-04` | `2cce0a4092f00490e16fdaa741693c8e3b5cfaf975050194abb8c0b102820940` |

- **Illustrative Sensitivity (NOT an observational bound):**
  - `thin_disk_2d` (`T = 1 Myr`, `--include-illustrative`): `+0.128762` (driven by illustrative IR `+0.128843`).
  - `finite_disk_3d` uniform (`--marginalize-lookback`, `--include-illustrative`): `-0.0398042` (IR `+0.045853`, Solar System `+0`).
- **Quadrature Precision (`--marginalize-lookback`, `T in [50 kyr, 10 Myr]`):** 16-node vs. 32-node Gauss–Legendre in `ln T` agrees to `|diff| = 6.11e-06 dex` on `frequent_slow_expanders` (`-0.08575627` vs. `-0.085750163`).
- **Per-Hypothesis Benchmark Values at `T = 1 Myr` (FS = `frequent_slow_expanders`, RF = `rare_fast_expanders`):**

| Geometry | `P(Earth reached)` FS / RF | `E[N \| Earth unreached]` FS / RF | `P0` exact (MC) FS / RF | RF Validity Status |
|---|---|---|---|---|
| `thin_disk_2d` | `0.00979582` / `0.00979582` | `0.10073` / `3.4884e-4` | `0.99981398 ± 8.7e-6` / `0.99999932 ± 3.9e-8` | `VIOLATED` (`T_valid = 52.185 kyr`) |
| `thick_disk_3d` | `0.00945466` / `0.00979579` | `0.072554` / `1.7926e-4` | `0.99987679 ± 6.9e-6` / `0.99999963 ± 2.6e-8` | `VIOLATED` (`T_valid = 52.185 kyr`) |
| `finite_disk_3d` uniform | `0.00945466` / **`0.000260229`** | `0.07232 ± 0.0003` / `1.2388e-6 ± 1.7e-8` | `0.99988158 ± 6.8e-6` / `0.9999999977 ± 2.8e-10` | `valid_finite_galaxy` |
| `finite_disk_3d` exponential | `0.00861234` / `0.000362676` | `0.065815 ± 0.00031` / `1.3192e-6 ± 3.1e-8` | `0.99989841 ± 1.9e-5` / `0.9999999983 ± 1.2e-9` | `valid_finite_galaxy` |

- **Finite Disk Normalization (`R_gal = 15 kpc`, `R_0 = 8.178 kpc`, `R_d = 2.6 kpc`, `h_z = 300 pc`, `z_sun = 20.8 pc`, `z_max = 5 h_z`):**
  - Sun at `(R_E, Z_E) = (sqrt(R_0^2 - z_sun^2), z_sun) = (8.17797, 0.0208) kpc`; both profiles integrate vertically to `Gamma_2d` at `R = R_E`:
    - Uniform: `rho = Gamma_2d / (2 h_z)` for `R <= R_gal`, `|Z| <= h_z`.
    - Exponential: `rho = Gamma_2d * exp(-(R - R_E) / R_d) * exp(-|Z| / h_z) / (2 h_z * (1 - exp(-z_max / h_z)))` for `R <= R_gal`, `|Z| <= z_max`.
- **Local Infinite-Disk Validity Cutoffs (`v_exp * T > 8 kpc`):**
  - `rare_fast_expanders` (`0.5c`): **`T_valid = 52.185 kyr`** (`5.218502e-5 Gyr`) — flagged `VIOLATED` under `thin_disk_2d` / `thick_disk_3d` at `T = 1 Myr`; cleared (`valid_finite_galaxy`) under `finite_disk_3d`.
  - `frequent_slow_expanders` (`0.005c`): **`T_valid = 5218.5 kyr`** (`5.218502e-3 Gyr`).
- **Scientific Verdict (Sprint 1):** Finite-disk edge truncation breaks the 2D scaling symmetry via the `earth_unreached` model term (`-0.00406` fixed-`T`, `-0.0857` marginalized), but separation remains **null** on the Jeffreys scale (`|log10 BF| < 0.5`) across all modes, with the BL 692 radio term remaining at `~1e-4 dex`.
- **Deterministic Queue Replay Benchmark:**
  - `lookback_window_t_sweep` (`seed=42`): `output_sha256 = 3dfab5f3b66aaa3741f2927c46ea10efbbea70d733002d5f3dd22dffab436f36`.

### Radio Drill-Down & Adversary Anchors
- **BL Radiometer Limit:** `minimum_flux_limit_jy = 17.0 Jy` (Enriquez et al. 2017 §5.1; radiometer equation `17.678 Jy`, within 5%); `sefd_jy = 10.0 Jy`; beam FWHM `8.4'`.
- **HIP 1692 Triple Distance (`F_min = 1.7678e-25 W/m^2`):**
  - Legacy (Isaacson 2017): `23.00 pc` -> `1.1189e12 W` (`1.0x`)
  - Historical anchor (Bailer-Jones DR2 / Wlodarczyk-Sroka 2020): `752.61 pc` -> `1.1981e15 W` (`1070.8x`)
  - Canonical (Gaia DR3 inverse parallax): `788.71 pc` -> `1.3157e15 W` (`1175.9x`)
- **Beam-Sharing Audit:** `16` target pairs within one `8.4'` GBT L-band beam (including visual binary `GJ 702A/B` at `0.00'` legacy catalog separation), verified `counted_once_each`.

### Infrared Drill-Down (Gate #1 + Gate #2)
- **Gate #1 Astrometric Floor (`reproduced_archive`, `n = 77` cal / `77` withheld):**
  - Measured `sigma_sys[robust_median] = 0.4356"` (68% CI `0.379"–0.502"`, 95% CI `0.356"–0.580"`; MLE `0.441"`, moment `0.457"`) vs. legacy Suazo (2024) `0.500"`.
  - Withheld 3-sigma false alarms: **`0/77`** (Clopper-Pearson 95% UL `0.0468`, Rayleigh KS `p = 0.448`; legacy `0.500"` KS `p = 0.014`; zero floor `11/77`).
  - Calibration in-sample tail hint: `2/77` (`p = 0.0187`).
- **Gate #2 Real AllWISE W3 Atlas FITS Withheld Result (`reproduced_archive`, `n = 77`):**
  - Empirical W3 Atlas PSF FWHM: **`13.690"`** (Gaussian fit; `11.71"` half-max area) vs. `6.5"` nominal Gaussian.
  - **Centroid screen (frozen `sigma_sys = 0.4356"`):** `0/77` null FA, **`0.671`** blend survival (`<= 3.25"`, vs. `0.699` synthetic), **`1/385` (`0.003`)** Arm B matched centered-excess false rejection.
  - **Image-level two-PSF `delta_chi2` screen (frozen scale `40.2`, threshold `569.3`):** **`5/77` (`0.065`)** null FA, **`0.934`** blend survival (`<= 3.25"`, vs. `0.760` synthetic), `24/385` (`0.062`) Arm B rejection — **documented negative result** caused by Atlas matched-filter smoothing and correlated pixel noise.
  - **Naive flux-excess reference rule:** `0/77` null FA, `0.200` survival, **`308/385` (`0.800`; `1.000` for every `f_bg >= 0.3`)** Arm B false rejection.
- **Hephaistos Candidate Anchors (measured `0.4356"` floor):** Candidate D (`0.75" +/- 0.22"`, `1.54σ`, `p = 0.307`, `no_evidence`); Candidate E (`1.50" +/- 0.24"`, `p = 0.011`, `marginal`, `>2σ` point-blend tension).
- **Contamination Convention Flip (`illustrative_calibration`):** Suazo (2024) subsample convention `LR = 0.303–0.493` vs. Zackrisson (2026) full-sample critique `LR = 1.36–1.38`.

### BOED Action Queue
- **Tier-1 Archival (ranked by EIG bits/hr):**
  1. `wise_independent_control_fields`: **`1.058 bits`** (`0.2117 bits/hr`)
  2. `wlodarczyk_sroka_288k_background_extension`: **`0.01452 bits`** physical (`distance_scaled_eirp`) vs. `0.827 bits` unphysical flat mode
- **Tier-0 Compute (ranked by `diagnostic_priority`, `expected_eig_bits = 0.0`):** `lookback_window_t_sweep` (`120 s`), `hephaistos_blend_recovery_grid` (`10 s`, runs both `0.500"` and `0.4356"` floors), `disk_scale_height_sweep` (`30 s`), `eirp_luminosity_function_sweep` (`240 s`).

---

## 4. Public-Launch Exit Gate Scorecard (`SPEC.md` §5)

| Gate | Requirement | Verified Evidence | Status |
|---|---|---|---|
| **Gate #1** | Previously unrun real-archive experiment with query receipt | 154-star / 12-patch AllWISE + Gaia DR3 `sigma_sys = 0.4356"` (`b02ec7df820d`) | **CLEARED** |
| **Gate #2** | Frozen-method withheld test with matched centered-excess vs. off-center blend on real AllWISE FITS | 154/154 real W3 FITS (`51a73ca4ff93`), pre-fetch freeze, centroid `0.671` / Arm B `0.003`, Atlas `delta_chi2` negative result preserved | **CLEARED** |
| **Gate #3** | Fresh-clone claim and deterministic replay of an open queue job | Fresh `%TEMP%` clone of `0667f973a2b4`, `30/30` validate, `MATCH=True` on `lookback_window_t_sweep` | **CLEARED** |
| **Gate #4** | No surprise quota (null & negative results reported transparently) | Flagship radio null (`-8.05e-5`), Atlas `delta_chi2` out-of-sample failure, and finite-disk null (`-0.0857` marginalized) surfaced intact | **CLEARED** |

---

## 5. Chronological Planning Rulings Log

1. **M1-R1 (`"dimensionless"` alias):** Approved `"dimensionless" -> astropy.units.dimensionless_unscaled` as the single canonical unit alias in `SPEC.md` §4.1.
2. **M1-R2 (`log_floor`):** Approved mandatory `log_floor` whenever a log-uniform prior has lower bound `0.0`.
3. **M1-R3 (BL flux limit):** Approved `minimum_flux_limit_jy: 17.0` (Enriquez 2017 §5.1 `S_min`) with optional `sefd_jy: 10.0`.
4. **M1-R4 (Decoupled IR coupling):** Keep `models/infrared.py` blind to expansion physics (`AGENTS.md` Law 6); `inference/engine.py` derives `f_ir_host = mean_P(star reached) * f_ir_loud` over a 256-point footprint lattice and passes it in.
5. **M1-R5 (HIP 1692 dual distance):** Gaia DR3 (`788.71 pc`, `1,176x`) is canonical in `bl_692_targets.csv`; Bailer-Jones DR2 (`752.61 pc`, `1,071x`) remains in `data/adversary_claims/hip_1692.yaml` as the historical anchor; Adversary verifies `> 1,000x` under both.
6. **M1-R6 (Likelihood normalization):** Separate `log10 LR vs null` from `raw log10 P(data | H)` cleanly; compute pairwise Bayes factors strictly over shared active channels and never sum raw joint scores across mismatched channel sets.
7. **M2-R1 (2D distance-preserving projection):** Approved the Adversary-discovered fix placing 2D targets at true radial distance (keeping azimuth) to eliminate the triangle-inequality bug that forced `q = 0` on high-latitude stars like HIP 1692.
8. **M2-R2 (Default verified Headline BF):** Policy locks default Headline BF to `verified_channels_only`. Folding `illustrative_calibration` channels into the headline requires explicit `--include-illustrative` (or loud UI warning toggle). In `distance_scaled_eirp` mode, radio is held out of the verified headline with explicit `headline_exclusion_notes` because the transmitter EIRP luminosity function is illustrative.
9. **M2-R3 (Contamination conventions):** Never pick a winner between Suazo (2024) and Zackrisson (2026); keep both conventions side-by-side.
10. **M2-R4 (Split Queue Views):** Keep `expected_eig_bits = 0.0` for Tier-0 compute jobs; never fabricate sky-photon bits for simulations. Rank Tier-0 separately by `diagnostic_priority` and `diagnostic_impact`.
11. **M2-R5 (Catalog quirks):** Keep `GJ 702A/B` (`0.00'` legacy separation) and all 16 beam-sharing pairs untouched in the catalog and surfaced in the Adversary feed.
12. **M2.5-R1 (Small-sample bounds & Atlas caveat):** Prominently display `0/77` (`95% UL < 4.7%`), `2/77` calibration tail, `11/77` zero-floor breakdown, and the Atlas smoothing caveat; raise future default latitude cut in `run_wise_control_experiment.py` to `|b| > 30°` without altering the locked 154-star artifact.
13. **M3-R1 (Code over prose):** Computed physics values always override stale brief prose (e.g., Candidate D `1.54σ`, flat-mode 288k EIG `0.827 bits`, Suazo measured centroid LR `0.303`).
14. **v0.1.1-R1 (UX & Deploy Safety):** Make live `Overview` the default tab; enforce fail-closed `serve --host <non-loopback>` refusal unless `--read-only` is passed; place `Depends(writable)` 403 guard ahead of job lookup; keep `queue seed` non-destructive on existing claims/runs; use Jeffreys-scale verdict wording with live terms (`earth_unreached + radio`).
15. **v0.1.2-R1 (Gate #2 negative result):** Lock the Atlas image-level `delta_chi2` out-of-sample failure (`6.5%` null FA, `0.934` survival) intact; forbid retuning on the withheld 77 stars; keep real-FITS survival side-by-side rather than overwriting `illustrative_calibration` LR rows; approve the 685 KB promoted JSON for offline recomputation.
16. **v0.2.0-R1 (`SPEC.md` ratification):** Ratified `finite_disk_3d` (`R_gal = 15 kpc`, `R_0 = 8.178 kpc`, `R_d = 2.6 kpc`, `z_max = 5 h_z`, uniform & exponential profiles normalized to local column rate `Gamma_2d` at `R_E = 8.17797 kpc`) and likelihood-space `logsumexp` `T`-marginalization (16-node Gauss–Legendre in `ln T`, `[50 kyr, 10 Myr]`) as canonical in `SPEC.md`.
17. **v0.2.0-R2 (Receipt schema stays v4):** Keep receipt schema version `4` for `v0.2.0` since fixed-`T` outputs and `results_sha256` hashes are byte-identical and new config keys are additive; defer `v5` until a breaking receipt schema lands (e.g., `union_targets_by_pointing` in `v0.2.1`).
18. **v0.2.0-R3 (Chain-rule marginalization decomposition):** Approved chain-rule ordering with verified terms first (`earth_unreached`, then `radio | earth_unreached`, then illustrative terms) so per-term BFs sum to the joint and illustrative channels never re-weight `T` or leak into `verified_channels_only`.
19. **v0.2.0-R4 (Model-term prominence caveat):** Keep `earth_unreached` labeled `verified (model term)`; schedule an inline UI caption in `v0.2.1`/`v0.2.3` clarifying that it depends on Galactic structure and lookback prior assumptions (`R_gal`, disk profile, `T` prior).
20. **v0.2.0-R5 (Adversary audit & public-host runtime):** Keep Adversary `local_approximation` auditing `thin_disk_2d` untouched; route marginalized `POST /api/compare` (1–2.5 min) into Sprint 4 (`v0.2.3`) `--read-only` cache-bound / allowlist hardening.
21. **v0.2.0-R6 (Version-string sync):** Synchronize `pyproject.toml` (`0.1.0`) and FastAPI app version (`0.3.0`) cleanly during `v0.2.1` alongside any pinned API version tests.

---

## 6. Active `v0.2` Backlog (See `docs/V0_2_ROADMAP.md`)

1. **Sprint 2 (`v0.2.1`) — `union_targets_by_pointing` Merge Policy & 288k Background Ingestion:** Implement per-pointing Gaia ID deduplication (forbidding `keep_first`), ingest Wlodarczyk-Sroka et al. (2020) 288,315 background stars with off-axis GBT beam attenuation, compare realized EIG against the `0.01452 bits` physical prediction, add the inline `earth_unreached` model-term UI caption, and sync package/API version strings.
2. **Sprint 3 (`v0.2.2`) — Unsmoothed `unWISE` W3 Coadds & Correlated-Noise Screen (Fresh `|b| > 30°` Split):** Sample `N >= 200` strictly disjoint isolated Gaia M dwarfs (zero overlap with the locked 154 AllWISE stars), pre-fetch freeze an empirical unsmoothed `unWISE` PSF plus GLS correlated-noise and non-parametric `delta_chi2` thresholds on the calibration half, and run the two-arm Arm A vs. Arm B withheld protocol.
3. **Sprint 4 (`v0.2.3`) — Large-N WISE False-Alarm Campaign & Public-Host Hardening:** Execute Tier-1 Archival Job #1 (`wise_independent_control_fields`, `N >= 1,000` withheld controls at `|b| > 30°`) under the frozen `0.4356"` floor, add `--allowed-seeds` and marginalized-request cache guards under `--read-only`, pre-seed `runs/queue.sqlite`, and add a production `Dockerfile` / `compose.yaml`.