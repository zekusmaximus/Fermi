# Fermi Paradox Empirical Constraint Desk — Planning Ledger

Single source of truth for architectural rulings, locked data artifacts, canonical baseline numbers, release history, and deferred backlog. Both the planning/audit agent (Neon) and the local coding agent treat this ledger alongside `AGENTS.md` and `SPEC.md` as authoritative.

---

## 1. Current Release State

- **Latest Release:** `v0.1.2` (Public-Launch Exit Gates #1–#4 all cleared)
- **Previous Tags / Checkpoints:**
  - Milestone 1: Core factor graph, 2D/3D expansion & radio/IR kernels, verified BL 692 catalog (`76` tests, `28/28` validate)
  - Milestone 2: Hephaistos IR drill-down, Adversary runner, BOED SQLite queue, distance-preserving 2D projection fix (`106` tests, `29/29` validate)
  - Milestone 2.5a (`b02ec7df820d`): Audit rulings, default `verified_channels_only` Headline BF, Tier-0 diagnostic queue, multi-patch WISE control script
  - Milestone 2.5b (`7f22a3691833`): Live 154-star / 12-patch AllWISE `sigma_sys` calibration (`reproduced_archive`), side-by-side IR drill-down (`123` tests, `30/30` validate)
  - Milestone 3 (`727daa7c` baseline): M2.5 polish rulings, offline FastAPI backend (`src/fermi_desk/api.py`), React/TS/Vite desk (`web/`), receipt schema v4 (`159` Python + `13` Vitest, `30/30` validate)
  - `v0.1.1` (`0667f973a2b4`): Live `Overview` default landing tab, `serve --read-only` deploy safeguard (exit code `2` on non-loopback read-write), `240 s` EIRP sweep estimate, Gate #3 fresh-clone replay verified (`174` Python + `23` Vitest, `30/30` validate)
  - `v0.1.2a` (`51a73ca4ff93`): Frozen real-FITS matched centered-vs-off-center Gate #2 pipeline (`models/hephaistos_fits.py`, `scripts/run_wise_fits_gate2.py`, `LeakageError`, `docs/V0_1_2_GATE_2.md`)
  - `v0.1.2b` / `v0.1.2`: Promoted real-FITS Gate #2 withheld artifact (`reproduced_archive`), wired into IR drill-down, `/api/health`, `InfraredDrilldown.tsx`, and `Overview.tsx` (`194` Python tests with non-loopback socket guard, `26` Vitest tests, `31/31` validate, `45` ruff-formatted files)
- **Runtime Invariant:** `uv run pytest -v`, `uv run fermi-desk validate`, FastAPI routes, and `web/` views run **100% offline** (non-loopback sockets blocked globally in `tests/conftest.py`).

---

## 2. Locked Empirical Artifacts (Do Not Modify Without New Provenance Run)

All paths below are protected against CRLF checkout drift via `.gitattributes` (`-text`).

| Artifact Path | Data Tier | Pinned SHA-256 | Source Commit | Verified Scope |
|---|---|---|---|---|
| `data/catalogs/bl_692_targets.csv` | `published_catalog` (Gaia DR3 cols: `reproduced_archive`) | `c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2` | M1 | 692/692 Enriquez (2017) targets matched to Isaacson (2017) Table 1 (`vizier.cfa.harvard.edu`); 517 Gaia DR3 parallaxes (`SNR >= 5`, `gaia.ari.uni-heidelberg.de`), 175 legacy fallback |
| `data/calibrations/wise_sigma_sys_control.json` | `reproduced_archive` | `2da2a234ee0ccb07...127b7a` *(copy full 64-char hash from `data/surveys/hephaistos_wise_mid_ir.yaml`)* | `b02ec7df820d` | 154 isolated Gaia M dwarfs (`RUWE < 1.4`, `cc_flags=0000`, `ext_flg=0`, W3 SNR `[8, 20]`, no neighbor `< 15"`) across 12 patches (`\|b\| > 20°`); 77 cal / 77 withheld |
| `data/calibrations/wise_fits_gate2_withheld.json` | `reproduced_archive` | `d4818c100b02cea9...` *(copy full 64-char hash from `data/surveys/hephaistos_wise_mid_ir.yaml`)* | `51a73ca4ff93` | 154/154 real IRSA AllWISE W3 Atlas intensity + uncertainty cutout pairs (308 FITS, 0 failures); PSF & thresholds frozen at `2026-11-26T21:01:58Z` before any withheld fetch |

---

## 3. Canonical Baseline Numbers (`seed = 42`, Receipt Schema v4)

### Macro Desk (`T = 1 Myr`, `frequent_slow_expanders` vs. `rare_fast_expanders`)
- **2D (`thin_disk_2d`, `flat_probability`, default `verified_channels_only`):**
  - `HEADLINE log10 BF` = **`-8.04966e-05`** (`earth_unreached = -8.67362e-19`, `radio = -8.04966e-05`) — verified null separation.
  - With `--include-illustrative`: **`+0.128762`** (not an observational bound; driven by illustrative IR `+0.128843`).
  - `P(Earth reached)` = `0.009796` for both hypotheses (exact 2D scaling degeneracy).
  - `E[N | Earth unreached]`: FS `0.1007`, RF `3.488e-4`.
  - `P0 exact (MC)`: FS `0.99981398 +/- 8.7e-6`, RF `0.99999932 +/- 3.9e-8`.
  - Canonical `results_sha256` (v4): `4b70118aaded7861cb14d77379843dfacec2d4c517bbce5efc9654e139fbc269`.
- **3D (`thick_disk_3d`):**
  - Slab thickness (`h_z = 300 pc`, `z_sun = 20.8 pc`) breaks 2D degeneracy weakly (`earth_unreached` relative split `3.482e-2`, `+1.496e-4 dex` toward FS).
  - Canonical `results_sha256` (v4): `be2053912fc0a1d6...`.
- **Local Infinite-Disk Validity Cutoffs (`v_exp * T > 8 kpc`):**
  - `rare_fast_expanders` (`0.5c`): **`T_valid = 52.185 kyr`** (`5.218502e-5 Gyr`) — flagged `VIOLATED` at `T = 1 Myr`.
  - `frequent_slow_expanders` (`0.005c`): **`T_valid = 5218.5 kyr`** (`5.218502e-3 Gyr`).
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
  - Measured `sigma_sys[robust_median] = 0.4356"` (68% CI `0.379"–0.502"`, 95% CI `0.356"–0.580"`; MLE