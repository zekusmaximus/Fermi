# Fermi Paradox Empirical Constraint Desk

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An offline-reproducible **empirical constraint engine and Bayesian Optimal Experimental Design (BOED) desk** for the Fermi paradox (`v0.2.2`).

Instead of cataloging verbal hypotheses or static Drake Equation sliders, this repository encodes Galactic-scale Fermi resolutions as parameterized probability models over a shared physical registry, evaluates them against SHA-256-locked astronomical catalogs and archive calibrations, audits catalog and geometric assumptions adversarially, and ranks the highest-leverage next archival queries and diagnostic compute sweeps by **Expected Information Gain (bits per hour)** and **diagnostic priority**.

---

## The Flagship Question

> **Can Breakthrough Listen's 692-star L-band radio survey distinguish civilizations that arise frequently and expand slowly from those that arise rarely and expand quickly—once we condition on Earth remaining unreached?**

From the interactive desk (starting on the live **Overview** landing view), you can toggle **radio**, **mid-infrared**, **Solar System**, **Earth-unreached** conditioning, Galactic geometry (`thin_disk_2d`, `thick_disk_3d`, `finite_disk_3d`), and lookback-window marginalization independently, then drill one click down into instrument reality:

- **Radio Drill-Down (`published_catalog` / `reproduced_archive`):** Toggle flat detection probabilities against per-target distance-scaled EIRP limits across all 692 Breakthrough Listen primary targets—inspecting **HIP 1692** (revised from `23.0 pc` legacy to `752.61 pc` Bailer-Jones DR2 / `788.71 pc` Gaia DR3 canonical, shifting minimum detectable EIRP by **`1,176x`**) and **16 GBT beam-sharing pairs** within the `8.4'` L-band FWHM beam.
- **Mid-IR Drill-Down (`reproduced_archive` + `illustrative_calibration`):** Inspect two live archive experiments on 154 isolated Gaia M dwarfs (77 calibration / 77 withheld) alongside synthetic Gaussian-PSF brackets, anchored on **Project Hephaistos Candidates D and E** and the Suazo (2024) vs. Zackrisson (2026) contamination-convention flip.
- **Adversary Feed & Tier-3 Quarantine Shelf:** Automated checks for catalog revisions, duplicate pointings, 2D-vs-3D symmetries, and infinite-disk cutoffs, with unfalsifiable resolutions (*Zoo Hypothesis*, *Dark Forest*) quarantined from empirical scoring and audited for **monoculture fragility**.
- **Split BOED Action Queue:** Separates **Tier-1 archival queries** ranked by EIG bits/hr (`wise_independent_control_fields`: `1.058 bits`; `wlodarczyk_sroka_288k_background_extension`: `0.0145 bits` physical vs. `0.827 bits` flat) from **Tier-0 diagnostic compute sweeps** (`EIG = 0.0 bits`) with deterministic `output_sha256` replay.

---

## Verified Empirical Results (`v0.2.2`)

Null verdicts and out-of-sample screen failures are first-class outputs here, never tuned away:

| Experiment / Benchmark | Data Tier | Key Verified Result (`seed=42`, `verified_channels_only`) |
|---|---|---|
| **Macro Radio — `thin_disk_2d` (`T = 1 Myr`)** | `published_catalog` | **`HEADLINE log10 BF = -8.05e-5`** (verified null; `rare_fast_expanders` flags `LOCAL APPROX VIOLATED`, `T_valid = 52.2 kyr`) |
| **Macro Radio — `finite_disk_3d` (`R_gal = 15 kpc`, `T = 1 Myr`)** | `published_catalog` | **`-0.00406`** (uniform) / **`-0.00364`** (exponential) — clears infinite-disk violation (`valid_finite_galaxy`); remains null |
| **Macro Radio — `finite_disk_3d`, `--marginalize-lookback` (`50 kyr–10 Myr`)** | `published_catalog` | **`-0.08566`** (uniform) / **`-0.09370`** (exponential) — null on Jeffreys scale (`\|log10 BF\| < 0.5`; BL 692 radio term `~1e-4 dex`) |
| **Gate #1 — AllWISE Astrometric Floor (`154` stars / `12` patches)** | `reproduced_archive` | **`sigma_sys = 0.4356"`** (vs. `0.500"` legacy); **`0/77`** withheld 3-sigma false alarms (`95% UL <= 0.047`; vs. `11/77` with no floor) |
| **Gate #2 — Real AllWISE W3 Atlas FITS Withheld Two-Arm Test (`154/154` cutouts)** | `reproduced_archive` | Empirical Atlas PSF FWHM **`13.69"`** (vs. `6.5"` nominal). Frozen centroid screen holds (**`0.671`** blend survival `<= 3.25"`, **`0.3%`** Arm B centered-excess rejection). Frozen image-level `delta_chi2` screen **fails out of sample** (`5/77 = 6.5%` null FA, `0.934` survival vs. `0.760` synthetic) — preserved as an honest negative result |

- **`v0.2.1` — Gaia DR3 background field in the 692 beams (`reproduced_archive`; EIG values additionally `illustrative_calibration`, never a verified bound):** **`119,947`** off-axis background stars in `46,888` shell cells (SHA-256 `eb43b118ab4d0258d9fe20474b73d187cb62a98018b20790b4112f875b0018a9`), merged star-by-star with the primaries via `union_targets_by_pointing` and attenuated by the GBT beam and distance. The realized information gain is a **null-strength result**: **`0.00235 bits`** in the physical mode (`distance_scaled_eirp`; `0.0145` predicted) vs. `0.651 bits` in the unphysical flat mode. The same audit exposed a **catalog erratum**: Isaacson et al. (2017) Table 1 prints **`14/692`** targets at `+10°` instead of `-10°`. The locked target list is left exactly as published; the affected background cones are corrected non-destructively through the SHA-256-pinned `data/adversary_claims/isaacson_2017_dec_sign_flip.yaml`. (`692stars.csv` separately prints 3 targets at `+00°` instead of `-00°`; the locked catalog does not inherit those.)
- **`v0.2.2` — Unsmoothed `unWISE` W3 blend screens on a fresh split (`reproduced_archive`, SHA-256 `b49e997054726bcfebd5b7ba5ca255a4071c3627af9ce7f871eee2799302f238`):** Gate #2's two-arm test repeated on **`252`** isolated M dwarfs in `16` new `|b| > 30°` patches (`126` calibration + `126` withheld, none shared with Gate #1), with every threshold frozen before the first withheld cutout was fetched. Without the Atlas smoothing the empirical W3 PSF narrows from `13.69"` to **`7.50"`** (Gaussian fit; `6.44"` half-max area), and null false-alarm control is restored: **`1/126 = 0.8%`** on all four image screens (scalar or GLS noise model × χ² or conformal cutoff), with `1.3%–1.4%` Arm B centered-excess rejection. The close-blend limit persists: **`71.8%–79.3%`** of blends within `3.25"` still survive every image screen, more than the `67.3%` that survive the centroid screen at the locked `0.4356"` floor. A partial improvement, promoted as measured.

---

## Architecture & Scientific Invariants

Every model, survey, and pull request is governed by six CI-enforced invariants (see [`SPEC.md`](SPEC.md)):

1. **Bipartite Factor Graph:** All hypotheses and surveys bind to a single canonical `data/parameters.yaml` registry with `astropy.units` enforcement.
2. **Three-Tier Stratification:** Predictive physical models (Tier 1), rate-constrained models (Tier 2), and unfalsifiable/monoculture claims (Tier 3) are never scored in the same pool.
3. **Zero-Emission Invariance:** Setting a channel's detectable emission probability to `0.0` must return a likelihood ratio of exactly `1.0` across all expansion models.
4. **Strict Non-Double-Counting:** Surveys declare a `pointing_group_id`; joint inference rejects duplicate conditioning on overlapping observations.
5. **Explicit Spatial Geometry:** Expansion kernels explicitly parameterize `thin_disk_2d`, `thick_disk_3d`, and `finite_disk_3d` (`R_gal = 15 kpc`, uniform or exponential density) so 2D scaling symmetries and infinite-disk boundaries are tested rather than assumed.
6. **Strict Provenance Tagging:** Every dataset declares its `data_tier` (`published_catalog`, `reproduced_archive`, `illustrative_calibration`, or `synthetic_proxy`). Default headlines enforce `verified_channels_only`; illustrative channels enter only via explicit `--include-illustrative`.

---

## Quickstart (100% Offline Out of the Box)

Requires Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and Node.js 20+ (only if building the web UI). All verified catalogs and calibration artifacts are already committed and SHA-256 locked under `data/`—no network access is needed to validate, test, or run the desk.

```bash
# 1. Install Python dependencies
uv sync

# 2. Validate all YAML schemas, Astropy units, locked SHA-256 hashes, and offline recomputations (42/42)
uv run fermi-desk validate

# 3. Run the offline Scientific CI test suite (460 tests; non-loopback sockets hard-blocked)
uv run pytest -v

# 4. Run flagship macro comparisons (2D baseline, 3D EIRP, and v0.2.0 finite disk + T marginalization)
uv run fermi-desk compare --geometry thin_disk_2d --radio-mode flat_probability --seed 42
uv run fermi-desk compare --geometry thick_disk_3d --radio-mode distance_scaled_eirp --seed 42
uv run fermi-desk compare --geometry finite_disk_3d --radio-mode flat_probability --seed 42
uv run fermi-desk compare --geometry finite_disk_3d --disk-profile exponential --radio-mode flat_probability --seed 42
uv run fermi-desk compare --geometry finite_disk_3d --radio-mode flat_probability --marginalize-lookback --seed 42

# 5. v0.2.1: merge the 119,947 Gaia DR3 background stars into the 692 beams (off-axis attenuation)
uv run fermi-desk compare --geometry thin_disk_2d --radio-mode distance_scaled_eirp --pointing-merge-policy union_targets_by_pointing --seed 42
```

`--marginalize-lookback` integrates the lookback window `T` over the registry's log-uniform prior (`50 kyr–10 Myr`) in likelihood space via 16-node Gauss–Legendre quadrature in `ln T`; expect roughly 40 s (`thin_disk_2d`), 95 s (`finite_disk_3d`), or 2.5 min (`thick_disk_3d`). Every `compare` run writes a deterministic JSON `RunReceipt` (git SHA, RNG seed, active surveys, data tiers, `results_sha256`) to `runs/`.

### Drill-Downs, Adversary & Deterministic BOED Queue (Offline)

```bash
uv run fermi-desk ir-drilldown --seed 42               # Real-FITS Gate #2 vs. synthetic Gaussian grid + D/E anchors
uv run fermi-desk adversary run --seed 42              # HIP 1692 triple-distance audit, 16 beam pairs, D/E, symmetry, validity
uv run fermi-desk background-field                     # v0.2.1 background field: counts, pointing audit, realized vs. predicted EIG
uv run fermi-desk queue seed                           # Seed/refresh SQLite BOED queue (non-destructive on existing runs)
uv run fermi-desk queue list
uv run fermi-desk queue claim lookback_window_t_sweep --by you
uv run fermi-desk queue run lookback_window_t_sweep --seed 42
uv run fermi-desk queue replay lookback_window_t_sweep # Reproduces output_sha256 byte-for-byte
```

### Interactive Research Desk (FastAPI + React/TypeScript/Vite, 100% Offline)

```bash
cd web && npm ci && npm test && npm run build && cd .. # 98 Vitest tests + production bundle -> web/dist
uv run fermi-desk serve                                # [http://127.0.0.1:8000](http://127.0.0.1:8000) (UI at /, API at /api/*, docs at /docs)
# UI hot-reload dev: run `uv run fermi-desk serve` and, in web/, `npm run dev` (proxies /api)
uv run fermi-desk serve --host 0.0.0.0 --read-only     # LAN/public bind: queue Run/Replay return 403 (CLI only)
```

`serve` fails closed (exit code `2`) on any non-loopback `--host` unless `--read-only` is passed, preventing unauthenticated synchronous compute jobs from being exposed over the network.

### Opt-In Live Archive Scripts (Never Run by `pytest` or `serve`)

```bash
uv run python scripts/fetch_bl_692.py                  # Re-fetch BL 692 primary catalog + Gaia DR3 parallaxes
uv run python scripts/run_wise_control_experiment.py   # Gate #1: multi-patch AllWISE sigma_sys controls (|b| > 30 deg default)
uv run python scripts/run_wise_fits_gate2.py --seed 42 # Gate #2: IRSA W3 Atlas FITS pre-fetch-frozen withheld experiment
```

---

## Repository Structure

```text
├── SPEC.md                       # Master architecture, data contracts, and core invariants
├── AGENTS.md                     # Universal scientific & engineering laws + §5 handoff protocol
├── DESIGN.md                     # UI legibility, data-tier colors, and offline visualization rules for web/
├── docs/
│   ├── PLANNING_LEDGER.md        # Locked SHA-256 hashes, canonical baselines, and chronological rulings
│   ├── V0_2_ROADMAP.md           # v0.2 sprint roadmap (finite disk, 288k union merge, unWISE, large-N)
│   ├── V0_1_2_GATE_2.md          # Predeclared Gate #2 real-FITS methodology and withheld results
│   └── MILESTONE_*.md            # Historical v0.1 milestone briefs (M1–M3)
├── data/
│   ├── parameters.yaml           # Canonical Astropy-validated physical Parameter Registry
│   ├── hypotheses/               # Tier 1–3 hypothesis definitions
│   ├── surveys/                  # Calibrated survey metadata and pinned artifact hashes
│   ├── catalogs/                 # Locked published_catalog targets (bl_692_targets.csv)
│   │   └── bl_gbt_lband_background_summary.json   # Locked reproduced_archive Gaia DR3 background field (v0.2.1)
│   ├── calibrations/             # Locked reproduced_archive WISE Gate #1 & Gate #2 artifacts
│   │   └── unwise_w3_gate_withheld.json           # Locked reproduced_archive unWISE W3 fresh-split gate (v0.2.2)
│   └── adversary_claims/         # Structured catalog/methodology anchor claims (HIP 1692, Hephaistos)
│       └── isaacson_2017_dec_sign_flip.yaml       # Isaacson (2017) -10° erratum + pinned pointing overlay (v0.2.1)
├── scripts/                      # Opt-in live archive fetchers with SHA-256 provenance receipts
│   └── run_unwise_gate.py                         # unWISE W3 two-stage gate: dry-run / fetch / promote (v0.2.2)
├── src/fermi_desk/               # Schemas, physics kernels, Bayesian engine, Adversary, Queue, CLI, API
├── web/                          # React + TypeScript + Vite interactive desk (6 views, zero chart bloat)
└── tests/                        # Offline Scientific CI invariants, API, and regression tests (460 tests)
```

---

## Contributing (Humans & AI Agents)

Null results, exposed assumptions, and failed screening rules are first-class contributions here. There is no quota for "surprising findings."

1. Read [`SPEC.md`](SPEC.md), [`AGENTS.md`](AGENTS.md), [`docs/PLANNING_LEDGER.md`](docs/PLANNING_LEDGER.md), and [`docs/V0_2_ROADMAP.md`](docs/V0_2_ROADMAP.md) before modifying models or data.
2. Never overwrite locked artifacts under `data/catalogs/` or `data/calibrations/`, and never tag synthetic or illustrative numbers as `published_catalog` or `reproduced_archive`.
3. Ensure `uv run pytest -v`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run fermi-desk validate`, and (if touching UI) `web/` `npm run typecheck && npm test && npm run build` pass cleanly offline before opening a pull request.
4. Include the deterministic `RunReceipt` (`results_sha256` and `rng_seed`) for any quantitative claim.

---

## Data Acknowledgments

Upstream astronomical data retrieved by scripts in this repository remains subject to the respective archive citation and usage policies:
- **Breakthrough Listen** (Enriquez et al. 2017, *ApJ* 849, 104; Isaacson et al. 2017, *PASP* 129, 054501; Wlodarczyk-Sroka et al. 2020, *MNRAS* 498, 5720).
- **CDS VizieR** (`J/PASP/129/E4501`, DOI: `10.26093/cds/vizier`).
- **ESA Gaia DR3** (Gaia Collaboration).
- **NASA/IPAC IRSA** (AllWISE / NEOWISE) and **Project Hephaistos** (Suazo et al. 2022, 2024; Ren et al. 2026; Zackrisson et al. 2026).

---

## License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.