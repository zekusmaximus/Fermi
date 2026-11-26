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
