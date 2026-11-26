# v0.1.2 — Public-Launch Gate #2: frozen real-FITS blend screens

> **Provenance of this brief.** Transcribed on 2026-11-26 by the implementing agent
> from the planning agent's v0.1.2 assignment. The "Predeclared method" section was
> committed in `v0.1.2a` **before** any live IRSA cutout was evaluated, so the git
> history timestamps the freeze. Results are appended after the live run (`v0.1.2b`).

## Scope
SPEC.md exit-gate item 2: one frozen-method adversarial test on withheld sky fields
containing matched centered stellar excesses and off-center background blends.

- Inputs (locked, never modified): the 154 controls and their `calibration` (77) /
  `withheld` (77) labels in `data/calibrations/wise_sigma_sys_control.json`
  (`2da2a234…`), and its `sigma_sys = 0.4356"`.
- Data: AllWISE Atlas W3 intensity (`-int-3.fits`) and uncertainty (`-unc-3.fits.gz`)
  60" cutouts from IRSA IBE, cached in gitignored `data/cache/wise_fits/`.
- Code: `src/fermi_desk/models/hephaistos_fits.py` (pure kernels),
  `scripts/run_wise_fits_gate2.py` (opt-in live run + `--promote-from`),
  `schemas.WiseFitsGate2File`, `registry.load_fits_gate2`, `fermi-desk validate`.

## Predeclared method (fixed before the live run)
1. **Zero leakage.** `build_empirical_psf` and `calibrate_frozen_screens` raise
   `LeakageError` on any non-`calibration` cutout; `evaluate_withheld_matched_injection`
   raises on non-`withheld` cutouts or any overlap with the calibration / PSF stars. The
   live script fetches the calibration half, freezes (`frozen_at_utc`,
   `frozen_screens_sha256`), and only then fetches the withheld half.
2. **Empirical PSF.** Sky = median of pixels beyond 16 px; Gate #1 windowed centroid
   (6.5" window); cubic-spline recentring onto a 31x31 stamp; unit-sum normalisation;
   pixel-wise median over calibration stars; renormalised. FWHM reported by a circular
   Gaussian fit (primary) and by the half-maximum area (model-free).
3. **Centroid screen.** Gate #1 estimator; flag when the offset from the Gaia position
   exceeds `hypot(6.5"/(2.355 SNR_eff), 0.4356") * sqrt(-2 ln 0.0027)`.
4. **Image screen.** Profile Delta chi^2, one empirical PSF (position prior
   N(Gaia, sigma_sys)) vs two (second flux >= 0, within 8 px); sky projected out over a
   12 px fit region; grid search then Nelder-Mead refinement. Threshold =
   `s * chi2.isf(0.0027, 3)`, `s = median(Delta chi^2_cal) / median(chi2(3))`. Atlas
   pixels are correlated by the coadd kernel, so nominal chi^2(3) is not used.
5. **Injection.** `F_bg = F_star f_bg / (1 - f_bg)`, empirical-PSF shaped, no added
   noise. Arm A at separation theta from the star's measured image centroid with a seeded
   azimuth (`SeedSequence([42, 7])`); Arm B the same flux at theta = 0. Screens use
   `SNR_eff = w3snr / (1 - f_bg)`. Grid = synthetic drill-down axes
   (theta 0.5–6", f_bg 0.1–0.9).
6. **Reference rule.** `naive_flux_excess` (flag any 3/SNR aperture excess) is reported
   as the adversarial "cheating" rule the matched control must expose; it is not a screen.
7. **Outputs.** Per-cell Arm A recovery, Arm B false rejection, withheld null false alarm,
   all with exact Clopper-Pearson 95% intervals; area-weighted (p ~ theta) blend survival
   within 3.25". Null / weak results are reported as-is.
8. **Promotion.** Refused for a dirty `git_sha` or an incomplete split (every locked
   calibration and withheld star must be fetched and evaluated).

## Results — Launch Gate #2 (live run, 2026-11-26; appended in v0.1.2b)
Receipt `runs/wise_fits_gate2/wise_fits_gate2-seed42-20261126T210600Z.json` (sha256
`1420cbca…`, code `51a73ca` = v0.1.2a, seed 42, window 20:59:59–21:06:00 UTC, frozen at
21:01:58 UTC before the first withheld cutout) promoted to
`data/calibrations/wise_fits_gate2_withheld.json` (file sha256 `d4818c10…`, pinned in
`data/surveys/hephaistos_wise_mid_ir.yaml`). Locked inputs unchanged.

- **Fetch:** 154/154 int + unc cutout pairs (308 files, 0 failures, 0 from cache);
  Gate #1 centroid offsets reproduced to max |diff| = 0.00019".
- **Empirical W3 Atlas PSF** (77 calibration stamps): FWHM 13.69" (Gaussian fit),
  11.71" (half-max area) vs 6.5" nominal — Atlas coadd smoothing roughly doubles the width.
- **Frozen thresholds:** centroid 3.439 x hypot(sigma_stat, 0.4356") (1.71" at the
  calibration median SNR); Delta chi^2 scale 40.2 -> threshold 569.3. Calibration in-sample
  false alarms: centroid 2/77 (same 2 as Gate #1), image 3/77.

| screen (withheld, n = 77) | null false alarm | blend survival <= 3.25" | Arm B false rejection (pooled) |
|---|---|---|---|
| centroid (frozen) | 0/77 (95% CP <= 0.047) | **0.671** (synthetic Gaussian: 0.699) | 1/385 = 0.003 (CP 0.000–0.014) |
| image Delta chi^2 (frozen) | **5/77 = 0.065** (CP 0.021–0.145) | **0.934** (synthetic Gaussian: 0.760) | 24/385 = 0.062 (CP 0.040–0.091) |
| naive flux excess (reference) | 0/77 | 0.200 | 308/385 = 0.800 (1.000 for every f_bg >= 0.3) |

Readings (no winner tuned in):
1. **The centroid screen transfers to real Atlas data** at the locked floor: out-of-sample
   null 0/77, Arm B 1/385, survival 0.671 vs 0.699 synthetic.
2. **The image-level screen fails on real Atlas cutouts.** Its frozen threshold is not
   calibrated out of sample (5/77 vs the 0.27% target; withheld null median Delta chi^2 124
   vs 95 on calibration), and Arm A recovery stays at that null rate except at theta = 6"
   with f_bg >= 0.5. Its 24 Arm B "rejections" are exactly the 5 stars it already flags
   without injection — false alarms, not a response to the excess. The synthetic Gaussian
   grid (survival 0.760) overstates real image-level performance; the median-matched
   chi^2(3) tail assumption does not hold for correlated Atlas noise plus PSF mismatch.
   This is a first-class negative result; any replacement statistic needs a new frozen run.
3. **The matched negative control works:** the naive bright-excess rule recovers 100% of
   blends for f_bg >= 0.3 and also rejects 100% of matched centered excesses (net 0).
