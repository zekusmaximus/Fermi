# Fermi Paradox Empirical Constraint Desk

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An executable, versioned **empirical constraint engine and Bayesian Optimal Experimental Design (BOED) desk** for the Fermi paradox.

Instead of cataloging verbal hypotheses or static Drake Equation sliders, this repository encodes Galactic-scale Fermi resolutions as parameterized probability models over a shared physical registry, evaluates them against calibrated observational surveys, audits catalog and geometric assumptions adversarially, and ranks the highest-leverage next compute runs and archival queries by **Expected Information Gain (bits per unit cost and time)**.

---

## The Flagship Question (v0)

> **Can Breakthrough Listen's 692-star L-band radio survey distinguish civilizations that arise frequently and expand slowly from those that arise rarely and expand quickly—once we condition on Earth remaining unreached?**

From that macro desk, you can toggle **radio**, **mid-infrared**, **Solar System**, and **Earth-unreached** evidence independently, then drill one click down into the underlying instrument reality:
* **Radio Drill-Down:** Toggle flat detection probabilities against per-target distance-scaled EIRP limits across the Breakthrough Listen primary targets—and inspect how catalog distance revisions (such as **HIP 1692**, revised from 23 pc to >750 pc in Gaia) shift minimum detectable transmitter power by three orders of magnitude.
* **Mid-IR Drill-Down:** Inspect why 15 plausible AllWISE completeness/false-positive calibrations reverse the likelihood preference between quiet and conspicuous settlement—anchored on **Project Hephaistos Candidates D and E** and their JWST/MIRI follow-up as background-galaxy blends.
* **Tier-3 Quarantine Shelf:** Sociological and unfalsifiable resolutions (*Zoo Hypothesis*, *Dark Forest*) are explicitly quarantined from empirical exclusion races and audited for **monoculture fragility**.

---

## Architecture & Scientific Invariants

Every model, survey, and pull request in this repository is governed by six CI-enforced invariants (see [`SPEC.md`](SPEC.md)):

1. **Bipartite Factor Graph:** All hypotheses and surveys bind to a single canonical `data/parameters.yaml` registry with `astropy.units` enforcement.
2. **Three-Tier Stratification:** Predictive physical models (Tier 1), rate-constrained models (Tier 2), and unfalsifiable/monoculture claims (Tier 3) are never scored in the same pool.
3. **Zero-Emission Invariance:** Setting a channel's detectable emission probability to `0.0` must return a likelihood ratio of exactly `1.0` across all expansion models.
4. **Strict Non-Double-Counting:** Surveys declare a `pointing_group_id`; joint inference rejects duplicate conditioning on overlapping observations.
5. **Explicit 2D vs. 3D Geometry:** Expansion kernels parameterize disk thickness (`thin_disk_2d` vs. `thick_disk_3d`) so 2D scaling symmetries are tested rather than assumed.
6. **Provenance Tagging:** Every dataset declares its `data_tier` (`published_catalog`, `reproduced_archive`, `illustrative_calibration`, or `synthetic_proxy`).

---

## Quickstart

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Fetch and verify the published Breakthrough Listen 692-star catalog + Gaia DR3 distances
uv run python scripts/fetch_bl_692.py

# 3. Validate all YAML schemas, Astropy units, and catalog checksums
uv run fermi-desk validate

# 4. Run the offline Scientific CI test suite
uv run pytest -v

# 5. Run the flagship macro comparison (2D flat baseline vs. 3D distance-scaled EIRP)
uv run fermi-desk compare --geometry thin_disk_2d --radio-mode flat_probability --seed 42
uv run fermi-desk compare --geometry thick_disk_3d --radio-mode distance_scaled_eirp --seed 42
```

Every `compare` run writes a deterministic JSON `RunReceipt` (including git SHA, RNG seed, active surveys, and data tiers used) to `runs/`.

Milestone 2 drill-downs, adversary runner and BOED queue (all offline; see `docs/MILESTONE_2.md`):

```bash
uv run fermi-desk ir-drilldown --seed 42          # WISE centroid vs image-level blend recovery (Hephaistos D/E anchors)
uv run fermi-desk adversary run --seed 42         # HIP 1692 dual-distance audit, duplicate pointings, D/E, symmetry, validity
uv run fermi-desk queue seed                      # register jobs with exact EIG (bits) and cost
uv run fermi-desk queue list
uv run fermi-desk queue claim lookback_window_t_sweep --by you
uv run fermi-desk queue run lookback_window_t_sweep --seed 42
uv run fermi-desk queue replay lookback_window_t_sweep   # must reproduce output_sha256 exactly

# Opt-in LIVE archive experiment (never run by pytest): WISE W3 centroid floor on control stars
uv run python scripts/run_wise_control_experiment.py --n-controls 120 --seed 42
```

### Interactive desk (Milestone 3, fully offline)
```bash
cd web && npm ci && npm run build && cd ..        # React + TypeScript + Vite bundle -> web/dist
uv run fermi-desk serve                           # http://127.0.0.1:8000  (UI at /, API at /api/*, docs at /docs)
# UI development with hot reload: run `uv run fermi-desk serve` and, in web/, `npm run dev` (proxies /api)
uv run fermi-desk serve --host 0.0.0.0 --read-only # LAN/public bind: queue Run/Replay return 403 (CLI only)
```

`serve` refuses any non-loopback `--host` unless `--read-only` is passed, because queue Run/Replay
execute synchronously inside the HTTP request.

---

## Repository Structure

```text
├── SPEC.md                 # Master architecture, schemas, and 3-milestone roadmap
├── AGENTS.md               # Working laws for coding & research models
├── docs/MILESTONE_1.md     # Current sprint specification and acceptance criteria
├── data/                   # Parameter registry, Tier 1–3 hypotheses, surveys, verified catalogs
├── scripts/                # Reproducible archive fetchers with SHA-256 provenance receipts
├── src/fermi_desk/         # Schemas, physics kernels, Bayesian engine, Adversary, CLI
└── tests/                  # Offline Scientific CI invariants and regression tests
```

---

## Contributing (Humans & AI Agents)

Null results, exposed assumptions, and failed screening rules are first-class contributions here. There is no quota for "surprising findings."

1. Read [`SPEC.md`](SPEC.md) and [`AGENTS.md`](AGENTS.md) before modifying models or data.
2. Never submit synthetic or illustrative numbers tagged as `published_catalog`.
3. Ensure `uv run pytest -v`, `uv run ruff check .`, and `uv run fermi-desk validate` pass cleanly offline before opening a pull request.
4. Include the deterministic `RunReceipt` and exact RNG seed for any quantitative claim.

---

## Data Acknowledgments

Upstream astronomical data retrieved by scripts in this repository remains subject to the respective archive citation and usage policies:
* **Breakthrough Listen** (Enriquez et al. 2017, *ApJ* 849, 104; Isaacson et al. 2017, *PASP* 129, 054501; Wlodarczyk-Sroka et al. 2020, *MNRAS* 498, 5720).
* **CDS VizieR** (`J/PASP/129/E4501`, DOI: `10.26093/cds/vizier`).
* **ESA Gaia DR3** (Gaia Collaboration).
* **NASA/IPAC IRSA** (AllWISE / NEOWISE) and **Project Hephaistos** (Suazo et al. 2022, 2024; Ren et al. 2026; Zackrisson et al. 2026).

---

## License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.