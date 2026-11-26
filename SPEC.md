# Master Specification: Empirical Constraint & Experimental Design Desk for the Fermi Paradox

## 1. Mission & Core Thesis
This repository is a **versioned, executable empirical constraint engine and Bayesian Optimal Experimental Design (BOED) desk** for the Fermi paradox.

It is **not**:
- An educational explainer or static Drake Equation calculator.
- A formal Lean theorem-proving bench.
- A single-instrument photometry calibration package.

It **is** a living research substrate where:
1. Galactic-scale Fermi resolutions are expressed as parameterized, executable probability models over a canonical physical Parameter Registry.
2. Real observational surveys (radio, mid-IR, Solar System) enter through calibrated, non-double-counted forward models rather than ad-hoc prose claims.
3. An Adversary engine continuously tests assumptions, catalog errors, geometric approximations, and cross-paper contradictions.
4. An Action Queue ranks the highest-leverage next compute sweeps and archival data queries by cost- and time-adjusted Expected Information Gain (EIG).

---

## 2. Core Architectural Invariants

Every milestone and pull request must preserve six non-negotiable invariants:

1. **Bipartite Factor Graph (No Free-Form Hypothesis DAGs):**
   Hypotheses never link to each other with informal prose edges. All hypotheses and surveys anchor to a single canonical `data/parameters.yaml` registry with explicit physical units (`astropy.units`) and log-scale priors.
2. **Three-Tier Empirical Stratification:**
   - **Tier 1 (Quantitatively Predictive):** Predicts concrete physical observables (radio transmitters, waste heat, probes, expansion fronts). Scored by posterior parameter volume and likelihood ratios.
   - **Tier 2 (Rate / Boundary Constrained):** Constrains evolutionary or astrophysical rates via exoplanet demographics or Earth history.
   - **Tier 3 (Unfalsifiable / Monoculture Quarantine):** Sociological or metaphysical resolutions (Zoo Hypothesis, Dark Forest, Simulation). Explicitly quarantined from empirical exclusion races and required to declare a `monoculture_audit` (explaining why zero defectors occur over Gyr timescales).
3. **Zero-Emission Invariance:**
   Setting a channel's detectable emission/signature probability to `0.0` MUST force that survey's likelihood ratio to `1.0` across all expansion speeds and origin rates. Any violation indicates a leaked prior or bug in inference.
4. **Strict Non-Double-Counting:**
   Every survey record declares a `pointing_group_id` and `evidence_channel`. Macro combined runs and single-channel drill-downs must never silently condition twice on the same pointings or stars.
5. **Explicit Spatial Geometry (2D vs. 3D):**
   Galactic expansion models must explicitly parameterize disk geometry (`geometry_mode: "thin_disk_2d" | "thick_disk_3d"`, scale height `h_z`). Scaling symmetries that only hold in 2D (e.g., doubling speed while quartering origin rate) must be tested against 3D thick-disk volumes where bubble radii are smaller than `h_z`.
6. **Strict Data Provenance Tagging:**
   Every numeric table and calibration curve must carry a `data_tier` tag:
   - `published_catalog` (verbatim published catalog/paper data with citation + hash)
   - `reproduced_archive` (fetched from IRSA/Gaia/BL with query receipt + timestamp)
   - `illustrative_calibration` (synthetic sensitivity bracket used while empirical measurement is open)
   Illustrative calibrations must never be rendered as verified observational bounds.

---

## 3. Repository Layout

```text
.
├── AGENTS.md                       # Universal rules for coding & research models
├── SPEC.md                         # This master specification
├── pyproject.toml                  # Managed via uv; pinned scientific stack
├── docs/
│   ├── MILESTONE_1.md              # Active milestone brief
│   ├── MILESTONE_2.md              # WISE drill-down, Adversary runner, BOED queue
│   └── MILESTONE_3.md              # Public interactive desk UI
├── data/
│   ├── parameters.yaml             # Canonical physical Parameter Registry
│   ├── hypotheses/                 # Tier 1, Tier 2, Tier 3 hypothesis definitions
│   ├── surveys/                    # Calibrated survey metadata & sensitivity specs
│   ├── catalogs/                   # Small verified target lists (e.g., BL 692 targets) + checksums
│   └── adversary_claims/           # Reproducible audits (e.g., HIP 1692, Hephaistos D/E)
├── src/fermi_desk/
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic v2 models + Astropy unit validators
│   ├── registry.py                 # Factor Graph loader & double-counting validator
│   ├── models/
│   │   ├── expansion.py            # 2D thin-disk & 3D thick-disk spatial correlation kernel
│   │   ├── radio.py                # Breakthrough Listen 692-star forward model (flat & per-target EIRP)
│   │   ├── infrared.py             # Mid-IR waste-heat & calibration-reversal forward model
│   │   ├── hephaistos.py           # WISE centroid vs. image-level blend-recovery grid (M2)
│   │   └── solar_system.py         # Local Solar System / ISO null-bound forward model
│   ├── inference/
│   │   ├── engine.py               # Joint likelihood engine + separate Earth-unreached conditioning
│   │   └── eig.py                  # Expected Information Gain (bits) & cost/time utility scorer
│   ├── adversary/
│   │   └── checks.py               # Automated physical, catalog, and contradiction audits
│   ├── queue/
│   │   └── store.py                # SQLite job queue, run receipts, and deterministic replay
│   ├── cli.py                      # Reproducible CLI entrypoints
│   └── api.py                      # FastAPI read/query layer (Milestone 2/3)
├── web/                            # React + Vite interactive public desk (Milestone 3)
└── tests/
    ├── test_schemas.py
    ├── test_invariants.py          # Zero-emission, boundary limits, 2D vs 3D symmetry, double-counting
    ├── test_models.py              # Model regression & deterministic replay tests
    └── test_adversary.py           # HIP 1692 EIRP shift & IR calibration reversal tests
```

---

## 4. Core Data Contracts

### 4.1 Parameter Registry (`data/parameters.yaml`)
**Unit rules (planning ruling, Milestone 2).** Every `unit` string must parse with
`astropy.units.Unit(...)`, with one canonical alias: `"dimensionless"` maps to
`astropy.units.dimensionless_unscaled` (astropy itself does not parse that word).
No other aliases are permitted.

**Prior-floor rule (planning ruling, Milestone 2).** A `log_uniform` prior whose
lower bound is `0.0` is undefined on its own and MUST declare an explicit positive
`log_floor` (strictly below the upper bound). Grids and marginalisations use
`max(lower_bound, log_floor)` as the effective lower edge.

```yaml
parameters:
  - id: origin_rate_per_kpc2_gyr
    symbol: "Gamma_2d"
    unit: "1 / (kpc2 * Gyr)"
    prior_type: log_uniform
    bounds: [1.0e-4, 1.0e3]
    description: "Civilization origin rate per unit Galactic disk area per Gyr."

  - id: expansion_speed_c
    symbol: "v_exp"
    unit: "dimensionless"          # fraction of speed of light c
    prior_type: log_uniform
    bounds: [1.0e-5, 0.5]
    description: "Mean isotropic settlement front velocity as a fraction of c."

  - id: disk_scale_height_pc
    symbol: "h_z"
    unit: "pc"
    prior_type: fixed_or_bounded
    default: 300.0
    bounds: [100.0, 1000.0]
    description: "Galactic stellar thin-disk vertical scale height."

  - id: radio_detect_prob
    symbol: "p_radio"
    unit: "dimensionless"
    prior_type: log_uniform
    bounds: [0.0, 1.0]
    log_floor: 1.0e-8                 # required: log-uniform with zero lower bound
    description: "Probability a settled system emits a detectable narrowband radio signature during observation."

  - id: ir_covering_fraction
    symbol: "gamma_ir"
    unit: "dimensionless"
    prior_type: log_uniform
    bounds: [0.0, 1.0]
    log_floor: 1.0e-8
    description: "Fraction of host stellar bolometric luminosity reprocessed into mid-IR waste heat."

  - id: ir_conspicuous_fraction
    symbol: "f_ir_loud"
    unit: "dimensionless"
    prior_type: log_uniform
    bounds: [0.0, 1.0]
    log_floor: 1.0e-8
    description: "Fraction of settled systems maintaining a detectable mid-IR waste-heat signature."
```

### 4.2 Hypothesis Schema (`data/hypotheses/*.yaml`)
```yaml
id: frequent_slow_expanders
name: "Frequent Origins, Slower Expansion Fronts"
tier: 1                             # 1 = Predictive, 2 = Rate-Constrained, 3 = Quarantined
summary: "Civilizations arise relatively frequently but expand slowly across the Galactic disk."
parameter_constraints:
  origin_rate_per_kpc2_gyr: { mode: fixed, value: 4.0 }
  expansion_speed_c: { mode: fixed, value: 0.005 }
  radio_detect_prob: { mode: fixed, value: 0.0021 }
  ir_conspicuous_fraction: { mode: range, min: 0.01, max: 0.5 }
channels_activated: ["radio", "infrared", "solar_system"]
falsification_condition: "Joint multi-channel null detections + spatial clustering bounds push likelihood ratio below 1e-3."
citations:
  - "Enriquez et al. 2017 (arXiv:1709.03491)"
  - "Hanson et al. 2021 (arXiv:2102.01522)"
```

**Infrared macro-to-micro coupling (planning ruling, Milestone 2).** `models/infrared.py`
never imports expansion physics. When a hypothesis with expansion parameters
activates `infrared`, `inference/engine.py` derives the per-star signature fraction
`f_ir_host = P(star reached) * f_ir_loud`, where `P(star reached)` is evaluated over
the IR survey footprint under the run's Earth-unreached setting, and passes it to
`models/infrared.py`. Such hypotheses must not also declare `ir_host_fraction`;
hypotheses without expansion parameters declare `ir_host_fraction` explicitly.
`f_ir_loud = 0` therefore always yields an IR likelihood ratio of exactly 1.0.

For `tier: 3` hypotheses (e.g., `zoo_hypothesis.yaml`, `dark_forest.yaml`), the schema MUST require:
```yaml
tier: 3
quarantine_reason: "Predicts intentional concealment indistinguishable from non-existence under current photon channels."
monoculture_audit:
  required_compliance_fraction: 1.0
  timescale_gyr: 1.0
  defector_vulnerability: "A single non-compliant subgroup or breakaway probe breaks observational silence."
```

### 4.3 Survey Forward-Model Contract (`data/surveys/*.yaml`)
```yaml
id: bl_enriquez_2017_692
name: "Breakthrough Listen L-Band 692 Nearby Stars"
channel: radio                      # radio | infrared | solar_system
pointing_group_id: bl_gbt_lband_2017
data_tier: published_catalog
n_targets: 692
frequency_ghz: [1.1, 1.9]
minimum_flux_limit_jy: 17.0         # Enriquez et al. 2017 Sec. 5.1 S_min (1 Hz transmitter, S/N >= 25)
sefd_jy: 10.0                       # optional: GBT L-band SEFD (Enriquez et al. 2017 Sec. 5.1)
observed_detections: 0
catalog_path: "data/catalogs/bl_692_targets.csv"
catalog_sha256: "<sha256>"
citations:
  - "Enriquez et al. 2017, ApJ 849, 104 (arXiv:1709.03491)"
  - "Isaacson et al. 2017, PASP 129, 054501"
```

**Likelihood normalisation (planning ruling, Milestone 2).** Receipts and CLI tables
report two distinct quantities per component and never mix them:
- `log10 P(data | H)`: the raw probability of the observed data under H.
- `log10 LR vs null`: the same quantity divided by its hypothesis-independent null
  (no settlement / zero emission / silent sky), which equals 1 by construction for
  the zero-detection radio and Earth-unreached terms.
Joint values are always labelled with the channel set they cover. Pairwise Bayes
factors are computed only over the intersection of the two hypotheses' active
channels (plus Earth-unreached conditioning when both carry it), and list the
channels that entered.

**HIP 1692 canonical distance (planning ruling, Milestone 2).** Gaia DR3 inverse
parallax (788.71 pc, ~1,176x EIRP_min) is canonical in `bl_692_targets.csv`; the
Bailer-Jones DR2 distance (752.61 pc, ~1,071x) is the historical Wlodarczyk-Sroka
et al. (2020) anchor in `data/adversary_claims/hip_1692.yaml`.

### 4.4 Action Queue & Run Receipt Schema (SQLite)
Every queued or executed job records:
- `job_id`, `tier` (`0_compute` | `1_archival`), `title`, `target_module`, `parameters_json`
- `expected_eig_bits` (with explicit prior specification), `est_cpu_seconds`, `est_reviewer_minutes`
- `status` (`open` | `claimed` | `completed` | `rejected`), `code_git_sha`, `rng_seed`, `output_sha256`, `created_at`

---

## 5. Three-Vertical Roadmap & Launch Gate

### Milestone 1: Factor Graph Core, Macro Engine & Scientific CI (Complete)
- Canonical schemas (`parameters.yaml`, starter Tier 1/2/3 hypotheses, 3 benchmark surveys).
- Executable 2D/3D spatial expansion & Earth-unreached conditioning kernel.
- Radio (BL 692 primary targets: flat vs. per-target distance-scaled EIRP), Mid-IR (calibration reversal bracket), and Solar System channels with toggleable joint evaluation.
- Full Scientific CI test suite (`pytest`) passing all 6 architectural invariants.

### Milestone 2: Drill-Down Verticals, Adversary Runner & BOED Queue (Current; brief: `docs/MILESTONE_2.md`)
- **Infrared Drill-Down (`hephaistos`):** AllWISE centroid vs. image-level blend-recovery grid (angular separation vs. background flux fraction), anchored on Project Hephaistos Candidates D and E (Ren et al. 2026; Zackrisson et al. 2026).
- **Adversary Runner:** Automated reproduction of real catalog/methodology failure modes—starting with **HIP 1692** (distance revision from ~23 pc to ~752.6 pc raising minimum detectable EIRP by >1,000x; Wlodarczyk-Sroka et al. 2020) and duplicate-pointing detection.
- **SQLite Job Queue & EIG Scorer:** Computes exact expected bits gained per trial plus CPU/review cost; registers open jobs (including Wlodarczyk-Sroka 288,315-star background extension and independent WISE control fields).

### Milestone 3: Public Live Desk UI (FastAPI + React/Vite)
- **Homepage (Macro Desk):** Live comparison of *Frequent/Slow* vs. *Rare/Fast* expansion regimes against toggleable Radio, Mid-IR, Solar System, and Earth-unreached conditioning.
- **Drill-Down Views:** Radio 692-star spatial map & EIRP sensitivity toggle; Mid-IR calibration sensitivity & Hephaistos Candidate D failure card.
- **Adversary Feed, Tier-3 Quarantine Shelf & Claimable Action Queue:** Live run feed with git SHA + seed replay commands.

### Public-Launch Scientific Exit Gate (Mandatory Before Promotion)
1. One previously unrun experiment executed on real archival images/catalogs (with query receipt).
2. One frozen-method adversarial test on withheld sky fields containing matched centered stellar excesses and off-center background blends.
3. One open queue job cleanly claimed and deterministically replayed from a fresh clone.
4. **No surprise quota:** Null results and failed screening rules pass the gate on equal footing with detections.