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
   Galactic expansion models must explicitly parameterize disk geometry (`geometry_mode: "thin_disk_2d" | "thick_disk_3d" | "finite_disk_3d"`, scale height `h_z`). Scaling symmetries that only hold in 2D (e.g., doubling speed while quartering origin rate) must be tested against 3D thick-disk volumes where bubble radii are smaller than `h_z`. `finite_disk_3d` (v0.2.0) additionally requires an explicit `disk_profile: "uniform" | "exponential"` and must converge to `thick_disk_3d` (uniform profile) while `v_exp * T << R_gal - R_sun`.
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
│   ├── MILESTONE_2.md              # WISE drill-down, Adversary runner, BOED queue (final)
│   ├── MILESTONE_2_5.md            # Audit closeout & live WISE control experiment
│   └── MILESTONE_3.md              # Public interactive desk UI
├── data/
│   ├── parameters.yaml             # Canonical physical Parameter Registry
│   ├── hypotheses/                 # Tier 1, Tier 2, Tier 3 hypothesis definitions
│   ├── surveys/                    # Calibrated survey metadata & sensitivity specs
│   ├── catalogs/                   # Small verified target lists (e.g., BL 692 targets) + checksums
│   ├── calibrations/               # reproduced_archive calibrations (e.g., WISE sigma_sys controls) + checksums
│   └── adversary_claims/           # Reproducible audits (HIP 1692, Hephaistos D/E, Isaacson 2017 Dec sign flip + pinned pointing overlay)
├── src/fermi_desk/
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic v2 models + Astropy unit validators
│   ├── registry.py                 # Factor Graph loader & double-counting validator
│   ├── models/
│   │   ├── expansion.py            # 2D thin-disk, 3D thick-disk & finite-Galaxy (R_gal) spatial correlation kernel
│   │   ├── radio.py                # Breakthrough Listen 692-star forward model (flat & per-target EIRP), off-axis beam & background shells
│   │   ├── infrared.py             # Mid-IR waste-heat & calibration-reversal forward model
│   │   ├── hephaistos.py           # WISE centroid vs. image-level blend-recovery grid (M2)
│   │   └── solar_system.py         # Local Solar System / ISO null-bound forward model
│   ├── inference/
│   │   ├── engine.py               # Joint likelihood engine + separate Earth-unreached conditioning
│   │   ├── background_field.py     # Promoted background-field report: shells, attenuation, realized EIG (v0.2.1b)
│   │   └── eig.py                  # Expected Information Gain (bits) & cost/time utility scorer
│   ├── adversary/
│   │   └── checks.py               # Automated physical, catalog, and contradiction audits
│   ├── queue/
│   │   └── store.py                # SQLite job queue, run receipts, and deterministic replay
│   ├── cli.py                      # Reproducible CLI entrypoints
│   └── api.py                      # FastAPI offline read / replay layer (Milestone 3; `fermi-desk serve`)
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

**Geometry scope (v0.2.0).** A parameter may declare `geometry_modes: [...]`. It is then
part of the model only under those geometries and is omitted from the per-hypothesis
`parameters` block of every other run. The finite-disk structure parameters
(`galactic_disk_radius_kpc` = 15 kpc, `solar_galactocentric_radius_kpc` = 8.178 kpc,
`galactic_radial_scale_length_kpc` = 2.6 kpc) are scoped to `finite_disk_3d`, so
`thin_disk_2d` / `thick_disk_3d` receipts and their canonical `results_sha256` are
unchanged. `disk_scale_height_pc` and `solar_height_pc` are reused, unscoped.

**Finite-disk density normalisation (v0.2.0).** In Galactocentric cylindrical `(R, Z)`,
with the Sun at `(R_E, Z_E) = (sqrt(R_0^2 - z_sun^2), z_sun)`:

- `uniform`: `rho = Gamma_2d / (2 h_z)` for `R <= R_gal`, `|Z| <= h_z`.
- `exponential`: `rho = Gamma_2d * exp(-(R - R_E)/R_d) * exp(-|Z|/h_z) / (2 h_z (1 - exp(-z_max/h_z)))`
  for `R <= R_gal`, `|Z| <= z_max = 5 h_z`.

Both give a column-integrated origin rate of exactly `Gamma_2d` at the Sun's radius.

**Marginalisation prior (v0.2.0).** A `log_uniform` parameter may declare
`marginalization_prior_bounds`; the prior-floor rule applies to it unchanged
(`lookback_window_gyr`: `[5.0e-5, 1.0e-2]` Gyr, i.e. 50 kyr - 10 Myr).

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

**Headline Bayes factor policy (planning ruling, Milestone 2.5).** The default
headline Bayes factor in `fermi-desk compare` and in run receipts (`headline_policy`,
`headline_log10_bf`) covers **verified channels only**: a shared channel enters only
if no `illustrative_calibration` / `synthetic_proxy` tier appears in its component for
either hypothesis (with Earth-unreached conditioning: `earth_unreached + radio` in
`flat_probability` mode). Illustrative channels (currently `infrared`, `solar_system`,
and `radio` under `distance_scaled_eirp`, whose transmitter luminosity function is
illustrative) stay active in the engine and appear in the per-component tables tagged
with their tier. Channels are excluded *before* term selection, so a joint
`radio+solar_system` term falls back to the verified radio marginal. `--include-
illustrative` (default off) folds them into the headline and is labelled loudly.
Both IR contamination conventions (Suazo 2024 subsample, Zackrisson 2026 full
sample) stay side-by-side in `ir-drilldown`; no winner is picked.

**Lookback-window marginalisation (v0.2.0; ratified, rulings v0.2.0-R1 and v0.2.0-R3).** With
`marginalize_lookback` (CLI `--marginalize-lookback`, default off) the lookback window `T`
is integrated over the registry's log-uniform prior on a deterministic Gauss-Legendre grid
in `ln T`. Marginalisation is in **likelihood space**:
`P(data | H) = sum_k w_k * prod_c P_c(data_c | H, T_k)` via `logsumexp`; `log10 P` is never
averaged. Because the likelihood then no longer factorises over channels, each reported sum
is the joint marginal over its own term set, decomposed by the chain rule with **verified
terms first**. The verified headline is therefore the marginal Bayes factor of the verified
data alone: an `illustrative_calibration` term can never re-weight the lookback posterior
behind it. Receipts record the prior bounds, grid, weights and per-node diagnostics; fixed-`T`
receipts gain no keys and keep their `results_sha256`.

**Pointing merge policy (v0.2.1a; ratified, ruling v0.2.1-R1).** A run states one
`pointing_merge_policy` (CLI `--pointing-merge-policy`, API `pointing_merge_policy`):

- `reject_duplicate_pointing_group` (default): two surveys sharing a `pointing_group_id`
  without a declared parent-pointing relationship raise `DoubleCountingError` - under
  *every* policy. A survey that declares `parent_pointing_survey_id` (and therefore
  `pointing_merge_policy: union_targets_by_pointing`) is an *extension* of its parent,
  never an independent survey: under the default policy it stays out of the run and is
  listed in the receipt's `held_out_extension_surveys`, so the parent runs exactly as a
  single-survey run (same `results_sha256`).
- `union_targets_by_pointing`: the extension's stars are merged into the parent star by
  star. Keys: exact Gaia DR3 `source_id`; angular separation `< 3 arcsec` only when either
  record lacks an id (never joining two different ids, nor two targets of one published
  primary list). Of several sightings of one star the minimum beam offset `theta_min` is
  kept (on-axis primary first). The merge is recorded as `MergeProvenance`
  (`pointing_group_id`, `policy`, `input_surveys`, `input_sha256s`, `n_raw_targets`,
  `n_merged_targets`, `n_dedup_gaia_id`, `n_dedup_angular_fallback`,
  `n_multi_beam_resolved`).
- `keep_first` is **forbidden**: schema, registry, engine, CLI and API raise
  `MergePolicyError` (a `DoubleCountingError`); it is never accepted or aliased.

**Ruling v0.2.1-R1 (default policy holds extensions out).** Under the default
`reject_duplicate_pointing_group` a declared pointing extension is held out of the run and
recorded in `held_out_extension_surveys`; default single-survey CLI calls and every canonical
`v0.1.x` / `v0.2.0` `results_sha256` stay byte-identical. Unconfigured or unrelated duplicate
pointing groups still raise `DoubleCountingError` under every policy.

**Ruling v0.2.1-R2 (flat-mode merged-background headline guard).** When
`union_targets_by_pointing` merges a background field under `radio_mode: flat_probability`,
the merged radio term is held **out of the `verified_channels_only` headline**, whatever the
data tier of the star catalogue: one distance- and beam-invariant detection probability applied
to off-axis stars out to 10 kpc ignores the `d^2 / B(theta)` attenuation of `EIRP_min` and is
an illustrative sensitivity check, not an observational bound. The radio component carries the
illustrative input `flat_probability_on_merged_background_field` (tier
`illustrative_calibration`), the receipt carries one structured `headline_exclusion_note`
(`channel`, `illustrative_inputs`, `reasons`, `rulings: ["v0.2.1-R2"]`, `log10_bf_term`, and
a `merged_background` split into the primaries' exact Monte Carlo term and the shells'
Harris-FKG product), the term stays in every component table, and `--include-illustrative`
folds it into the headline under the loud `including_illustrative_channels` policy. Under
`distance_scaled_eirp` radio is already excluded (ruling M2-R2). Notes of single-survey runs
gain no keys, so their `results_sha256` is unchanged.

**Ruling v0.2.1-R3 (locked primaries, reported suspects).** `data/catalogs/bl_692_targets.csv`
stays immutable. A surviving background star within 120 arcsec (J2000) of a primary without a
Gaia id and within 30% of its distance is listed in the artifact as
`suspected_unmatched_primary` with the `pointing_index`, distance and beam offset that address
its shell cell. Suspects are never removed from the shells; the count is reported in the
artifact, `fermi-desk validate`, `fermi-desk compare`, `fermi-desk background-field`, the API
and the Radio drill-down, and every EIG is reported with and without them. Offline
re-verification (made exact in v0.2.1c): a validation-sample star inside the 3 arcsec fallback
radius of an ID-less primary is an unmerged collision (`DoubleCountingError`) **unless** that
primary already absorbed a nearer sighting carrying a different Gaia id - the one case in
which the merge rule above refuses the link (e.g. GJ 473AB, one published target resolved by
Gaia into two sources). Such stars may not outnumber the merge's own `n_angular_links_refused`.

**Ruling v0.2.1-R4 (Gaia ids are strings).** Gaia DR3 `source_id` values exceed 2^53. Every
`gaia_dr3_source_id` written into an artifact from v0.2.1 on, and every one in an API
response (including ids read from locked `v0.1.x` artifacts, which keep their integer form on
disk), is a decimal JSON string; in the background summary a JSON number is a schema error.
The API also stringifies any other integer beyond +-(2^53 - 1).

**Ruling v0.2.1-R5 (catalog erratum corrected by a pinned overlay, never in place).** Table 1
of Isaacson et al. (2017) prints every target whose true declination lies in (-11, -10] deg
without its minus sign (0 rows at -10 deg, 35 at +10 deg among 1,702 targets; the authors'
arXiv source `table1709_4paper.txt` has the same omission). `bl_692_targets.csv` takes its
coordinates from that table, so 14 of the 692 primaries sit 20.1 - 21.9 deg north of the star
that was observed. The catalog **stays immutable** (its SHA-256 and all eight canonical
single-survey `results_sha256` hashes are unchanged): it is the verbatim `published_catalog`
record, and a silent in-place edit would detach it from its citation and from every receipt
that pins it. The correction instead lives in
`data/adversary_claims/isaacson_2017_dec_sign_flip.yaml`, the single canonical table of the 14
J2000 positions (11 from Gaia DR3, 3 from Hipparcos-2, `reproduced_archive`), pinned by
`pointing_corrections_sha256` in the extension survey. It is applied as an overlay in exactly
three places: the cone centres of the background fetch, the primary positions inside the
de-duplication, and the offline re-verification of the summary (beam offsets and shell
directions). The summary records `pointing_corrections_path`, `pointing_corrections_sha256`,
`n_corrected_pointings`, `corrected_target_names` and, per corrected cone, the queried centre
and the hash, time and commit of its query; `fermi-desk validate` re-derives each query hash
at the corrected centre and refuses a promoted summary that ignores the pinned overlay. The
cone cache is keyed by the queried centre, so applying the overlay re-fetched exactly those
14 cones (the superseded replies are kept, not deleted). `pointing_centre_audit` reports
`corrected_via_overlay` (14) and `uncorrected_flipped_cones` (0), and the Adversary runs
`catalog_dec_sign_audit` offline on every run (a `finding`: 0 targets in [-11, -10) deg, 16 in
[+10, +11) deg, 14 sign-restored positions within 2 arcsec of truth, the two genuinely
northern controls HIP 59608 and HIP 92283 untouched). Scope limits, stated rather than hidden:
(a) the primaries' own positions in the primary likelihood term are still the catalog's
(no effect in `thin_disk_2d`, which uses distance only; in the 3D modes those 14 targets, at
13 - 52 pc, are still placed along the catalog's mirrored direction - size of the effect not
yet measured); (b) 2 of the 14 (HIP 34104, HIP 77349) have only two-parameter Gaia DR3 solutions, so
no sighting with a distance exists to absorb: they are in no shell and in no suspect list;
(c) `692stars.csv` (Enriquez et al. 2017) independently prints three (-1, 0) deg targets
(HIP 99711, HIP 12530, HIP 82588) with a plus sign. That is an upstream published-table
discrepancy which the catalog does not inherit; whether the telescope tracked the printed
coordinate is not established (filterbank headers unverified).

**Off-axis sensitivity and background shells (v0.2.1a; ratified in v0.2.1b).** A star at offset
`theta` from a beam centre has `EIRP_min(d, theta) = 4 pi d^2 F_min / B(theta)` with
`B(theta) = exp(-4 ln2 (theta / FWHM)^2)` (GBT L-band `FWHM = 8.4 arcmin`; `B(0) = 1`,
`B(FWHM/2) = 1/2`). Background-field stars enter as a compact, SHA-256-pinned summary
(`data/catalogs/bl_gbt_lband_background_summary.json`, `< 5 MB`): per-pointing counts on a
`(distance, EIRP_min)` grid, de-duplicated against the primaries before binning. Merged
runs keep the exact correlated Monte Carlo for the primaries and multiply on the
independent conditional-reach shell product `prod (1 - p_c q_c)^N_c`, which by Harris-FKG
is a **lower bound** on `P(no detection | H)` (it can only overstate the null's
constraining power); the Frechet upper bound is reported beside it. A `synthetic_proxy`
background is illustrative and leaves the verified headline.

**Promoted background field (v0.2.1b).** `data/surveys/bl_gbt_lband_background_gaia_dr3.yaml`
declares the extension (`parent_pointing_survey_id: bl_enriquez_2017_692`,
`pointing_merge_policy: union_targets_by_pointing`, `data_tier: reproduced_archive`) and pins
the summary by SHA-256. The summary reproduces the *method* of Wlodarczyk-Sroka et al. (2020)
on the 692 Enriquez et al. (2017) GBT L-band pointings with Gaia DR3 (119,947 stars since
v0.2.1c; 120,176 in v0.2.1b, of which 1,055 came from 14 cones at sign-flipped centres, ruling
v0.2.1-R5); it is not their catalogue (1,327 Price et al. 2020 multi-band pointings, Gaia DR2,
288,315 stars). A
`reproduced_archive` summary must carry a clean `code_git_sha`, all 692 pointings, the true
archive query window (`query_window_utc`, when cones were fetched) beside `assembled_at_utc`,
`cone_cache_manifest_sha256` (the raw cone bytes) and `cone_cache_code_git_shas` (every commit
that queried the archive, all clean). `fermi-desk validate` re-checks the pin, the schema, the
seal and recomputes the validation sample and every suspected star's shell cell offline.

**HIP 1692 canonical distance (planning ruling, Milestone 2).** Gaia DR3 inverse
parallax (788.71 pc, ~1,176x EIRP_min) is canonical in `bl_692_targets.csv`; the
Bailer-Jones DR2 distance (752.61 pc, ~1,071x) is the historical Wlodarczyk-Sroka
et al. (2020) anchor in `data/adversary_claims/hip_1692.yaml`.

### 4.4 Action Queue & Run Receipt Schema (SQLite)
Every queued or executed job records:
- `job_id`, `tier` (`0_compute` | `1_archival`), `title`, `target_module`, `parameters_json`
- `expected_eig_bits` (with explicit prior specification), `est_cpu_seconds`, `est_reviewer_minutes`
- `status` (`open` | `claimed` | `completed` | `rejected`), `code_git_sha`, `rng_seed`, `output_sha256`, `created_at`
- `diagnostic_impact`, `diagnostic_priority` (Milestone 2.5 ruling): Tier-0 compute jobs keep
  `expected_eig_bits = 0.0` exactly (simulations add no sky photons) and instead state the
  model uncertainty or validity boundary they resolve (text computed from the job parameters)
  plus a rank among compute jobs (1 = first). Tier-1 archival jobs leave both NULL and are ranked
  by EIG bits/hr. `fermi-desk queue list` renders the two tiers as separate tables.

**Run receipt schema v5 (v0.2.1a; ratified in v0.2.1b).** Comparison receipts
carry `pointing_merge_policy` (default `reject_duplicate_pointing_group`), `merge_provenance`
(`null` unless a union merge happened), `background_field` (`null` unless merged: pinned
SHA-256, tier, clean `code_git_sha`, exact `N_bg`, de-duplication counters including
`suspected_unmatched_primary`, and the retrieved-vs-published scope) and
`held_out_extension_surveys`; `config.merge_policies`
is replaced by `config.pointing_merge_policy`. `results_sha256` still hashes the numerical
`results` block only, so every single-survey `v0.1.x` / `v0.2.0` hash is unchanged.

**Tier-1 completion by promoted artifact (v0.2.1b).** A Tier-1 archival job is never run by
the queue. When its opt-in script has promoted a SHA-256-pinned artifact that passes the full
offline verification, `fermi-desk queue seed` records the job `completed` with
`output_sha256` = the artifact's file hash and `code_git_sha` = the commit that built it. The
transition is non-destructive: only `open` / `claimed` rows move (a claim keeps its
`claimed_by`), and `expected_eig_bits` stays the registered prediction; the realized value is
linked evidence (`fermi-desk background-field`, `GET /api/queue`).

---

## 5. Three-Vertical Roadmap & Launch Gate

### Milestone 1: Factor Graph Core, Macro Engine & Scientific CI (Complete)
- Canonical schemas (`parameters.yaml`, starter Tier 1/2/3 hypotheses, 3 benchmark surveys).
- Executable 2D/3D spatial expansion & Earth-unreached conditioning kernel.
- Radio (BL 692 primary targets: flat vs. per-target distance-scaled EIRP), Mid-IR (calibration reversal bracket), and Solar System channels with toggleable joint evaluation.
- Full Scientific CI test suite (`pytest`) passing all 6 architectural invariants.

### Milestone 2: Drill-Down Verticals, Adversary Runner & BOED Queue (Complete; brief: `docs/MILESTONE_2.md`)
- **Infrared Drill-Down (`hephaistos`):** AllWISE centroid vs. image-level blend-recovery grid (angular separation vs. background flux fraction), anchored on Project Hephaistos Candidates D and E (Ren et al. 2026; Zackrisson et al. 2026).
- **Adversary Runner:** Automated reproduction of real catalog/methodology failure modes—starting with **HIP 1692** (distance revision from ~23 pc to ~752.6 pc raising minimum detectable EIRP by >1,000x; Wlodarczyk-Sroka et al. 2020) and duplicate-pointing detection.
- **SQLite Job Queue & EIG Scorer:** Computes exact expected bits gained per trial plus CPU/review cost; registers open jobs (including Wlodarczyk-Sroka 288,315-star background extension and independent WISE control fields).

### Milestone 2.5: Audit Closeout & Live WISE Control Experiment (Complete; brief: `docs/MILESTONE_2_5.md`)
- Close the ten Milestone 2 audit rulings (verified-only headline BF, Tier-0 diagnostic queue,
  `slow` test marker, finalized Milestone 2 brief, structured beam-sharing records).
- Public-Launch Gate #1: live multi-patch AllWISE/Gaia control experiment measuring the W3
  centroid registration floor `sigma_sys` (calibration half) and its out-of-sample 3-sigma
  false-alarm rate (withheld half), promoted to `data/calibrations/wise_sigma_sys_control.json`
  (`reproduced_archive`) and reported side-by-side with the legacy 0.50 arcsec baseline.

### Milestone 3: Public Live Desk UI (FastAPI + React/Vite) (Current; brief: `docs/MILESTONE_3.md`)
- **Homepage (Macro Desk):** Live comparison of *Frequent/Slow* vs. *Rare/Fast* expansion regimes against toggleable Radio, Mid-IR, Solar System, and Earth-unreached conditioning.
- **Drill-Down Views:** Radio 692-star spatial map & EIRP sensitivity toggle; Mid-IR calibration sensitivity & Hephaistos Candidate D failure card.
- **Adversary Feed, Tier-3 Quarantine Shelf & Claimable Action Queue:** Live run feed with git SHA + seed replay commands.

### v0.2 / Post-M3 Roadmap (deferred, documented at Milestone 2.5)
1. **`union_targets_by_pointing` merge policy (required before ingesting the
   Wlodarczyk-Sroka et al. 2020 background stars).** The 288,315 background stars lie in
   the same GBT L-band pointings as the BL 692 primaries and therefore share
   `pointing_group_id: bl_gbt_lband_2017`. Today the only merge policy is `keep_first`,
   which resolves a pointing-group collision by keeping whichever survey loads first.
   `keep_first` must **never** be used for this ingestion: (a) the result depends on file
   load order; (b) it silently discards one catalog wholesale - either the background
   constraint or the primary targets; (c) the two catalogs are complementary (different
   stars in shared beams), not duplicates, so dropping one is data loss, not
   de-duplication. `union_targets_by_pointing` must merge the target lists per pointing,
   de-duplicate stars by Gaia DR3 `source_id` so each star enters the likelihood exactly
   once, attach per-star sensitivity (off-axis beam attenuation for background stars), and
   record the merge (input checksums, duplicate count) in the run receipt.
   **Implemented offline in v0.2.1a** (Sec. 4.3 "Pointing merge policy"): `keep_first` no
   longer exists and raises `MergePolicyError`. **Executed live in v0.2.1b**
   (`scripts/fetch_wlodarczyk_sroka_288k.py --fetch`, then `--promote-from`): 120,176 Gaia DR3
   background stars over the 692 Enriquez (2017) L-band pointings, promoted to
   `data/catalogs/bl_gbt_lband_background_summary.json` (`reproduced_archive`). Realized EIG:
   0.00235 bits in `distance_scaled_eirp` (predicted 0.01452) and 0.651 bits in the unphysical
   `flat_probability` mode (predicted 0.827): a null-strength result. **Corrected in v0.2.1c**
   (ruling v0.2.1-R5): 14 cones had been fetched at sign-flipped declinations; re-fetching
   only those 14 gives 119,947 stars and 0.002347 / 0.6507 bits. The null-strength verdict
   is unchanged.
2. **Finite-Galaxy disk geometry (`R_gal = 15 kpc`).** The expansion kernel is a local
   infinite-disk approximation valid only while `v_exp * T <= 8 kpc`; for
   `rare_fast_expanders` (`v_exp = 0.5 c`) this gives `T_valid = 52.2 kyr`, so receipts at
   the default lookback flag `local_approximation_valid = false`. A finite disk with
   `R_gal = 15 kpc` (edge truncation of origin sites and bubble intersection with the disk
   boundary) replaces the local limit; until then the `lookback_window_t_sweep` Tier-0 job
   records the cutoff and BF sensitivity across 50 kyr-10 Myr.
   **Implemented in v0.2.0** as `geometry_mode: finite_disk_3d` (receipts report
   `local_approx: valid_finite_galaxy`); the two local modes are kept, byte-identical, and
   still flag the violation.

3. **unWISE coadd re-measurement of `sigma_sys` (Milestone 3 Part A.3).** The
   Milestone 2.5 floor (`sigma_sys = 0.436 arcsec`, 154 controls / 12 patches) was
   measured on AllWISE Atlas W3 coadds, whose resampling kernel smooths the PSF; the
   floor therefore absorbs Atlas smoothing together with astrometric registration (the
   `atlas_smoothing_caveat` in `ir-drilldown` receipts). Re-measure the same controls on
   unWISE coadds (native, unblurred PSF) and report both floors side by side; never
   overwrite the locked Atlas artifact.

### v0.1.2: Public-Launch Gate #2 (brief: `docs/V0_1_2_GATE_2.md`)
- Frozen-method real AllWISE W3 FITS blend screens on the 154 locked controls: empirical
  PSF and thresholds frozen on the 77 calibration stars, withheld two-arm injection
  (off-center blends vs matched centered excess), promoted to
  `data/calibrations/wise_fits_gate2_withheld.json` (`reproduced_archive`, SHA-256 pinned)
  and shown beside the synthetic Gaussian grid. Result: centroid screen transfers; the
  image-level Delta chi^2 screen fails out of sample (reported as a negative result).

### Public-Launch Scientific Exit Gate (Mandatory Before Promotion)
1. One previously unrun experiment executed on real archival images/catalogs (with query receipt).
2. One frozen-method adversarial test on withheld sky fields containing matched centered stellar excesses and off-center background blends.
3. One open queue job cleanly claimed and deterministically replayed from a fresh clone.
4. **No surprise quota:** Null results and failed screening rules pass the gate on equal footing with detections.