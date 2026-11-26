# Milestone 2.5: Audit Closeout & Live WISE Control Experiment

> **Provenance of this brief.** Transcribed on 2026-11-26 by the implementing agent
> from the planning agent's Milestone 2.5 assignment (ten Milestone 2 audit rulings
> plus Public-Launch Gate #1). Implementation decisions that the assignment did not
> fix are listed under "Implementation decisions" and flagged in the handoff.

## Part A — Milestone 2 audit rulings
1. `docs/MILESTONE_2.md` is final: every `INFERRED` item approved, tags removed; job
   sources in `queue/store.py` cite the brief. The post-projection-fix 2D numbers are
   the canonical baseline (table at the end of `docs/MILESTONE_2.md`).
2. **Headline BF = verified channels only** (`SPEC.md` §4.3 ruling). `infrared` stays
   active and tier-tagged in per-component tables; both IR contamination conventions
   stay side-by-side in `ir-drilldown`. `fermi-desk compare --include-illustrative`
   (default off) folds illustrative channels in, labelled loudly. Invariant test:
   `tests/test_invariants.py::test_default_headline_bf_excludes_illustrative_channels`.
3. **Tier-0 diagnostics vs Tier-1 EIG** (`SPEC.md` §4.4). Tier-0 keeps
   `expected_eig_bits = 0.0`; new `diagnostic_impact` / `diagnostic_priority` columns;
   `queue list` prints a Tier-1 Archival Queue (EIG bits/hr) and a Tier-0 Compute Queue
   (diagnostic priority).
4. Deferred items documented in `SPEC.md` "v0.2 / Post-M3 Roadmap":
   `union_targets_by_pointing` (why `keep_first` must never be used) and finite-Galaxy
   geometry (`R_gal = 15 kpc`). The 16 GBT beam-sharing pairs are structured JSON
   records in the adversary receipt.
5. `slow` pytest marker; `uv run pytest -m "not slow"` for iteration, `uv run pytest -v`
   runs everything offline.

## Part B — Live WISE control experiment (Launch Gate #1)
`scripts/run_wise_control_experiment.py`:
- >= 120 control stars across >= 8 separated high-latitude patches (|b| > 20 deg);
- AllWISE `cc_flags == '0000'`, `ext_flg == 0`, W3 SNR in the drill-down regime;
  Gaia `ruwe < 1.4`; no Gaia neighbour within 15 arcsec; exact ADQL, patch centres and
  cut counts recorded;
- deterministic (`--seed 42`) 50/50 calibration / withheld split: `sigma_sys` from the
  calibration half, 3-sigma centroid false-alarm rate out-of-sample on the withheld half.

Sequence: offline verification -> clean commit (2.5a) -> live run (clean `git_sha`) ->
promotion to `data/calibrations/wise_sigma_sys_control.json` (`reproduced_archive`,
SHA-256 pinned) -> wiring into `validate`, `models/hephaistos.py` and `ir-drilldown`,
which report the legacy 0.50 arcsec baseline and the measured floor side-by-side.

## Part C — Queue reset, verification, commit (2.5b)

## Implementation decisions (not fixed by the assignment)
- **Primary `sigma_sys` estimator = robust median-matched Rayleigh estimator**
  (`models.hephaistos.sigma_sys_robust`), fixed *before* the seed-42 production run
  after a seed-7, two-patch smoke test showed a "clean" control with a 6.39 arcsec W3
  offset (an unseen background source). The heteroscedastic Rayleigh MLE and moment
  estimators are reported alongside. Withheld false-alarm counts include every control.
- Patches: 12 seeded isotropic centres, radius 2 deg, whole cone at |b| > 20 deg,
  pairwise separation >= 20 deg, avoiding the LMC, SMC and M31.
- W3 SNR band [8, 20] (Milestone 2 drill-down regime, simulated SNR 12.5).
- Split stratified by patch (seeded permutation, alternating labels).
- Tier-0 diagnostic priorities: 1 `lookback_window_t_sweep` (validity boundary of the
  benchmark headline), 2 `hephaistos_blend_recovery_grid` (screens feeding the IR
  likelihood; first to absorb the measured floor), 3 `disk_scale_height_sweep` (3D
  symmetry-breaking scale), 4 `eirp_luminosity_function_sweep` (illustrative channel
  only, outside the verified headline).
- The six core invariant tests are never marked `slow`.

## Results — Launch Gate #1 (live run, 2026-11-26)
Receipt `runs/wise_control/wise_control-seed42-20261126T125624Z.json` (sha256
`2f1167da…`, code `b02ec7d`, seed 42, queries 12:40:43–12:56:24 UTC, 442 HTTP/TAP
queries) promoted to `data/calibrations/wise_sigma_sys_control.json` (file sha256
`2da2a234…`, pinned in `data/surveys/hephaistos_wise_mid_ir.yaml`).
- 154 controls across 12 patches (11–13 per patch); 0 measurement failures; cuts:
  2476 Gaia M dwarfs -> 2018 RUWE < 1.4 -> 542 W3-band matches -> 495 `cc_flags=0000`
  -> 481 `ext_flg=0` -> 264 isolation-checked, 110 rejected (Gaia neighbour < 15 arcsec).
- Calibration half (n = 77): `sigma_sys` = 0.436 arcsec (robust; 68% bootstrap
  0.379–0.502, 95% 0.356–0.580); MLE 0.441, moment 0.457.
- Withheld half (n = 77), out-of-sample 3-sigma centroid false alarms: 0/77 at the
  measured floor (95% CP 0–0.047; KS vs Rayleigh p = 0.45); 0/77 at the legacy 0.5
  arcsec floor (KS p = 0.014: legacy floor over-wide); 11/77 with no floor.
- Drill-down shift (seed 42): centroid recovery +0.028 mean, blend survival within
  3.25 arcsec 0.743 -> 0.699; D/E verdicts unchanged (E p 0.026 -> 0.011, marginal).
