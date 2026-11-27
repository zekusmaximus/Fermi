import { describe, expect, it } from "vitest";
import type { Gate2, Rate, UnwiseArchiveCell, UnwiseComparison, UnwiseComparisonRow, UnwiseGate, UnwiseScreenName } from "../api";
import { SOURCE_LABEL, SOURCE_TIER } from "./gate2";
import {
  HEAT_SOURCES,
  SOURCE_ENUM,
  UNWISE_HEAT_SCREENS,
  UNWISE_SCREEN_LABEL,
  archiveCell,
  comparisonRows,
  conformalLevel,
  fmtCp,
  fmtRate,
  freezeLeadSeconds,
  lag1Text,
  legacyCounterpart,
  legacyScreenCaption,
  likeForLikeStatement,
  overlapStatement,
  psfRows,
  resolveSource,
  screenHelp,
  sourceAvailable,
  sourceCaption,
  sourceHelp,
  syntheticCell,
  unwiseCounterpart,
  unwiseScreenCaption,
} from "./unwise";

// A small hand-made payload with the shape of GET /api/infrared/drilldown (fixture values,
// not the promoted measurement: the helpers must format whatever the payload carries).
const rate = (k: number, n: number, lo: number, hi: number): Rate => ({ k, n, rate: k / n, clopper_pearson_95: [lo, hi] });
const cell = (k: number, n: number, survival: number, bk: number): UnwiseArchiveCell => ({
  n_stars: n,
  null_false_alarm: rate(k, n, 0, 0.05),
  blend_survival_within_3p25_arcsec: survival,
  arm_b_false_rejection_pooled: rate(bk, 5 * n, 0.01, 0.03),
});
const imageRow = (
  screen: UnwiseScreenName,
  rule: "chi2" | "conformal",
  survival: number,
  atlas: UnwiseArchiveCell | null,
): UnwiseComparisonRow => ({
  screen,
  kind: "image",
  label: screen,
  role: "role",
  statistic: screen.includes("gls") ? "gls" : "scalar",
  threshold_rule: rule,
  threshold: 20.5,
  delta_chi2_scale: rule === "chi2" ? 1.5 : null,
  alpha_guaranteed: rule === "conformal" ? 1 / 101 : null,
  k_exceedances: rule === "conformal" ? 0 : null,
  calibration_false_alarm_in_sample: rate(0, 100, 0, 0.04),
  unwise_w3: cell(1, 100, survival, 5),
  allwise_atlas_gate2: atlas,
  synthetic_gaussian: atlas
    ? { n_null_trials: 240, null_false_alarm_rate: 0.004, blend_survival_within_3p25_arcsec: 0.75, arm_b_false_rejection_pooled: null, registration_floor: "measured_reproduced_archive" }
    : null,
  counterpart: atlas ? "image" : null,
});
const centroidRow = (screen: UnwiseScreenName, sys: number, source: string, atlas: UnwiseArchiveCell | null): UnwiseComparisonRow => ({
  screen,
  kind: "centroid",
  label: screen,
  role: "role",
  sigma_sys_arcsec: sys,
  sigma_sys_source: source,
  threshold_arcsec_at_calibration_median_snr: 1.6,
  calibration_false_alarm_in_sample: rate(0, 100, 0, 0.04),
  unwise_w3: cell(2, 100, 0.6, 9),
  allwise_atlas_gate2: atlas,
  synthetic_gaussian: null,
  counterpart: atlas ? "centroid" : null,
});

const comparison: UnwiseComparison = {
  max_separation_arcsec: 3.25,
  weighting: "p(theta) ~ theta",
  nominal_w3_fwhm_arcsec: 6.5,
  sources: {
    unwise_w3: {
      label: "unWISE W3 (Unsmoothed - Fresh Split)",
      data_tier: "reproduced_archive",
      psf_fwhm_arcsec: 7.8,
      psf_fwhm_halfmax_arcsec: 6.9,
      pixel_scale_arcsec: 2.75,
      n_psf_stars: 100,
      n_controls: 200,
      n_calibration: 100,
      n_withheld: 100,
      pixel_covariance_lag1_correlation: [0.21, 0.19],
      noise_model: "empirical covariance",
      code_git_sha: "a".repeat(40),
      file_sha256: "b".repeat(64),
    },
    allwise_atlas_gate2: {
      label: "AllWISE W3 Atlas (Gate #2)",
      data_tier: "reproduced_archive",
      psf_fwhm_arcsec: 13,
      psf_fwhm_halfmax_arcsec: 11,
      pixel_scale_arcsec: 1.375,
      n_psf_stars: 77,
      n_controls: 154,
      n_calibration: 77,
      n_withheld: 77,
      pixel_covariance_lag1_correlation: null,
      noise_model: "scalar variance",
      code_git_sha: "c".repeat(40),
      file_sha256: "d".repeat(64),
    },
    synthetic_gaussian: {
      label: "Synthetic Gaussian PSF",
      data_tier: "illustrative_calibration",
      psf_fwhm_arcsec: 6.5,
      psf_fwhm_halfmax_arcsec: null,
      pixel_scale_arcsec: null,
      n_psf_stars: null,
      n_controls: null,
      n_calibration: null,
      n_withheld: null,
      pixel_covariance_lag1_correlation: null,
      noise_model: "white noise",
      code_git_sha: null,
      file_sha256: null,
    },
  },
  image_screens: [
    imageRow("image_scalar_chi2", "chi2", 0.7, cell(5, 77, 0.9, 24)),
    imageRow("image_gls_chi2", "chi2", 0.65, null),
    imageRow("image_scalar_conformal", "conformal", 0.8, null),
    imageRow("image_gls_conformal", "conformal", 0.75, null),
  ],
  centroid_floors: [
    centroidRow("centroid_locked", 0.4356, "locked", cell(0, 77, 0.67, 1)),
    centroidRow("centroid_fresh", 0.31, "fresh_calibration_half", null),
  ],
  provenance: {
    pinned_by_survey: "hephaistos_wise_mid_ir",
    path: "data/calibrations/unwise_w3_gate_withheld.json",
    file_sha256: "b".repeat(64),
    content_sha256: "e".repeat(64),
    code_git_sha: "a".repeat(40),
    rule_version: "unwise-gate-v1",
    frozen_at_utc: "2026-01-01T00:00:10.000000+00:00",
    latest_calibration_fetch_utc: "2026-01-01T00:00:05.000000+00:00",
    earliest_withheld_fetch_utc: "2026-01-01T00:00:12.500000+00:00",
    frozen_before_withheld_fetch: true,
    disjoint_from_gate1_path: "data/calibrations/wise_sigma_sys_control.json",
    disjoint_from_gate1_sha256: "f".repeat(64),
    n_gate1_stars: 154,
    n_fresh_stars: 200,
    n_gate1_star_overlap: 0,
    min_abs_b_deg: 30.6,
    n_patches: 16,
  },
  note: "three inputs, never blended",
};

const screen = (k: number, survival: number) => ({
  role: "role",
  n_stars: 100,
  null_false_alarm: rate(k, 100, 0, 0.05),
  arm_a_blend_recovery: [[0.1, 0.2]],
  arm_a_k: [[10, 20]],
  arm_a_clopper_pearson_95: [[[0, 1] as [number, number], [0, 1] as [number, number]]],
  arm_b_false_rejection: [0.01, 0.02],
  arm_b_k: [1, 2],
  arm_b_clopper_pearson_95: [[0, 1] as [number, number], [0, 1] as [number, number]],
  arm_b_false_rejection_pooled: rate(3, 200, 0.003, 0.04),
  blend_survival_within_3p25_arcsec: survival,
  centered_excess_retained_pooled: 0.985,
});

const unwise = {
  data_tier: "reproduced_archive",
  n_patches: 16,
  n_controls: 200,
  n_calibration: 100,
  n_withheld: 100,
  frozen_at_utc: comparison.provenance.frozen_at_utc,
  earliest_withheld_fetch_utc: comparison.provenance.earliest_withheld_fetch_utc,
  target_false_alarm: 0.0027,
  conformal_alpha_requested: 0.01,
  conformal_alpha_nominal: 0.01,
  conformal_alpha_fallback: false,
  empirical_psf: { fwhm_gaussian_arcsec: 7.8, fwhm_halfmax_arcsec: 6.9, nominal_fwhm_arcsec: 6.5, n_stars: 100, pixel_scale_arcsec: 2.75 },
  frozen_screens: {
    image_scalar_conformal: { k_exceedances: 0, n_calibration: 100, alpha_guaranteed: 1 / 101, threshold: 30 },
  },
  screens: {
    centroid_locked: screen(1, 0.67),
    centroid_fresh: screen(2, 0.55),
    image_scalar_chi2: screen(1, 0.7),
    image_gls_chi2: screen(1, 0.65),
    image_scalar_conformal: screen(1, 0.8),
    image_gls_conformal: screen(1, 0.75),
    naive_flux_excess: screen(0, 0.2),
  },
  comparison,
} as unknown as UnwiseGate;

const gate2 = {
  n_withheld: 77,
  empirical_psf: { fwhm_gaussian_arcsec: 13, fwhm_halfmax_arcsec: 11, nominal_fwhm_arcsec: 6.5, n_stars: 77, pixel_scale_arcsec: 1.375 },
} as unknown as Gate2;

const ctx = { gate2, unwise, synthetic: { psfFwhmArcsec: 6.5, snr: 12.5, nTrials: 40 } };

describe("unWISE heatmap source control", () => {
  it("offers unWISE W3 as the third source, with plain labels, enums and separate tiers", () => {
    expect([...HEAT_SOURCES]).toEqual(["real", "synthetic", "unwise"]);
    expect(HEAT_SOURCES.map((s) => SOURCE_LABEL[s])).toEqual([
      "AllWISE W3 Atlas (Gate #2)",
      "Synthetic Gaussian PSF",
      "unWISE W3 (Unsmoothed — Fresh Split)",
    ]);
    expect(SOURCE_ENUM.unwise).toBe("unwise_w3_gate_withheld · reproduced_archive");
    expect(SOURCE_ENUM.synthetic).toContain("illustrative_calibration");
    expect(SOURCE_TIER.unwise).toBe("reproduced_archive");
    expect(SOURCE_TIER.synthetic).toBe("illustrative_calibration");
  });

  it("falls back to the synthetic grid, never to another archive, when an artifact is not pinned", () => {
    expect(sourceAvailable("unwise", gate2, null)).toBe(false);
    expect(sourceAvailable("real", null, unwise)).toBe(false);
    expect(sourceAvailable("synthetic", null, null)).toBe(true);
    expect(resolveSource("unwise", gate2, null)).toBe("synthetic");
    expect(resolveSource("real", null, unwise)).toBe("synthetic");
    expect(resolveSource("unwise", gate2, unwise)).toBe("unwise");
    expect(resolveSource("real", gate2, unwise)).toBe("real");
  });

  it("writes a one-line caption per source from the payload, each stating the headline effect", () => {
    const u = sourceCaption("unwise", ctx);
    expect(u).toContain("100 withheld unsmoothed unWISE cutouts");
    expect(u).toContain("measured PSF 7.80″ on 2.75″ pixels");
    expect(u).toContain("|b| > 30°");
    expect(u).toContain("shares no star with Gate #1");
    expect(u).toContain("the verified headline does not use it");
    const a = sourceCaption("real", ctx);
    expect(a).toContain("77 withheld AllWISE Atlas cutouts");
    expect(a).toContain("13.00″ (nominal 6.5″)");
    const s = sourceCaption("synthetic", ctx);
    expect(s).toContain("6.5″ Gaussian PSF at W3 SNR 12.5");
    expect(s).toContain("never the verified headline");
    expect(new Set([u, a, s]).size).toBe(3); // the caption changes on every switch
    for (const text of [u, a, s]) expect(text).not.toContain("\n");
    // An unpinned archive source gets the synthetic caption rather than invented numbers.
    expect(sourceCaption("unwise", { ...ctx, unwise: null })).toBe(s);
  });

  it("explains the disjoint |b| > 30 deg split and the zero-leakage freeze in the popover", () => {
    const h = sourceHelp(unwise, gate2);
    expect(h.tradeoff).toContain("13.00″ wide");
    expect(h.tradeoff).toContain("2.75″ pixels");
    expect(h.invariant).toContain("200 isolated Gaia M dwarfs in 16 patches whose whole cone lies at |b| > 30°");
    expect(h.invariant).toContain("0 of them among the 154 Gate #1 stars");
    expect(h.invariant).toContain("frozen on the 100 calibration stars at 2026-01-01T00:00:10.000000+00:00");
    expect(h.invariant).toContain("before the first withheld cutout was fetched (2026-01-01T00:00:12.500000+00:00)");
    expect(h.invariant).toContain("Nothing was retuned afterwards");
    expect(h.cli).toContain("uv run fermi-desk ir-drilldown --seed 42");
    const unpinned = sourceHelp(null, null);
    expect(unpinned.invariant).toContain("|b| > 30°");
    expect(unpinned.invariant).not.toContain("undefined");
  });
});

describe("unWISE blend-screen control", () => {
  it("lists both centroid floors and the 2 x 2 image screens with plain labels", () => {
    expect(UNWISE_HEAT_SCREENS.map((s) => UNWISE_SCREEN_LABEL[s])).toEqual([
      "Centroid · locked floor",
      "Centroid · fresh floor",
      "Scalar χ²",
      "GLS χ²",
      "Scalar Conformal",
      "GLS Conformal",
    ]);
    expect(UNWISE_HEAT_SCREENS).not.toContain("naive_flux_excess");
  });

  it("keeps the same rule selected across sources where one exists", () => {
    expect(unwiseCounterpart("centroid")).toBe("centroid_locked");
    expect(unwiseCounterpart("image")).toBe("image_scalar_chi2");
    expect(legacyCounterpart("centroid_fresh")).toBe("centroid");
    expect(legacyCounterpart("image_gls_conformal")).toBe("image");
  });

  it("captions the active screen with its withheld numbers", () => {
    const text = unwiseScreenCaption(unwise, "image_gls_chi2");
    expect(text).toContain("whitening the pixels with the measured noise covariance");
    expect(text).toContain("Withheld null false alarms 1/100 = 1.0%");
    expect(text).toContain("65.0% of blends within 3.25″ survive");
    expect(text).toContain("the verified headline does not use it");
    expect(unwiseScreenCaption(unwise, "centroid_fresh")).not.toBe(text);
    expect(legacyScreenCaption("image", "synthetic")).toContain("Illustrative");
    expect(legacyScreenCaption("centroid", "real")).toContain("the verified headline does not use it");
  });

  it("states the exact conformal level and flags the v0.2.2-R3 fallback", () => {
    expect(conformalLevel(unwise)).toEqual({ text: "1/101 = 0.99%", level: 1 / 101, fallback: false, requested: 0.01 });
    const small = {
      ...unwise,
      conformal_alpha_fallback: true,
      frozen_screens: { image_scalar_conformal: { k_exceedances: 0, n_calibration: 60 } },
    } as unknown as UnwiseGate;
    const lvl = conformalLevel(small);
    expect(lvl.text).toBe("1/61 = 1.64%");
    expect(lvl.fallback).toBe(true);
    const help = screenHelp(unwise);
    expect(help.tradeoff).toContain("exact level 1/101 = 0.99%");
    expect(help.tradeoff).toContain("χ²(3) tail at 0.3%");
    expect(help.invariant).toContain("v0.2.2-R2");
    expect(screenHelp(null).tradeoff).not.toContain("undefined");
  });
});

describe("AllWISE Atlas vs unWISE W3 comparison panel", () => {
  it("formats exact rates and Clopper-Pearson bounds", () => {
    expect(fmtRate(rate(5, 77, 0.021, 0.145))).toBe("5/77 = 6.5%");
    expect(fmtCp(rate(5, 77, 0.021, 0.145))).toBe("95% CP 2.1%–14.5%");
    expect(fmtRate(rate(0, 126, 0, 0.029))).toBe("0/126 = 0.0%");
  });

  it("compares the empirical PSF widths against the nominal value", () => {
    const rows = psfRows(comparison);
    expect(rows.map((r) => r.key)).toEqual(["unwise_w3", "allwise_atlas_gate2", "nominal"]);
    expect(rows.map((r) => r.fwhmArcsec)).toEqual([7.8, 13, 6.5]);
    expect(rows[0]!.ratioToNominal).toBeCloseTo(1.2);
    expect(rows[1]!.ratioToNominal).toBe(2);
    expect(rows.map((r) => r.tier)).toEqual(["reproduced_archive", "reproduced_archive", null]);
    expect(rows[0]!.detail).toBe("100 calibration stars stacked, 2.75″ pixels");
    // The Atlas pixel scale is stored as 1.37499998...: shown to 3 decimals, not raw.
    const atlas = { ...comparison.sources.allwise_atlas_gate2!, pixel_scale_arcsec: 1.37499998090796 };
    const raw = psfRows({ ...comparison, sources: { ...comparison.sources, allwise_atlas_gate2: atlas } });
    expect(raw[1]!.detail).toBe("77 calibration stars stacked, 1.375″ pixels");
    const noAtlas = psfRows({ ...comparison, sources: { ...comparison.sources, allwise_atlas_gate2: null } });
    expect(noAtlas.map((r) => r.key)).toEqual(["unwise_w3", "nominal"]);
  });

  it("lays out the 2 x 2 image screens then both centroid floors, leaving missing rules empty", () => {
    const rows = comparisonRows(comparison);
    expect(rows.map((r) => r.screen)).toEqual([
      "image_scalar_chi2",
      "image_gls_chi2",
      "image_scalar_conformal",
      "image_gls_conformal",
      "centroid_locked",
      "centroid_fresh",
    ]);
    expect(rows.map((r) => r.label)).toEqual([
      "Scalar χ²",
      "GLS χ²",
      "Scalar Conformal",
      "GLS Conformal",
      "Centroid · locked floor",
      "Centroid · fresh floor",
    ]);
    const scalar = rows[0]!;
    expect(scalar.frozen).toBe("Δχ² > 20.50 · χ²(3) tail × 1.50");
    expect(scalar.unwise).toMatchObject({ nullFa: "1/100 = 1.0%", survival: 0.7, armB: "5/500 = 1.0%" });
    expect(scalar.atlas).toMatchObject({ nullFa: "5/77 = 6.5%", survival: 0.9, armB: "24/385 = 6.2%" });
    expect(scalar.atlas).toMatchObject({ nullFaNote: "95% CP 0.0%–5.0%", nullFaBelow: "0.0%–5.0%", armBBelow: "1.0%–3.0%" });
    expect(scalar.synthetic).toMatchObject({ nullFaBelow: "240 simulated nulls", armBBelow: null });
    expect(scalar.synthetic).toMatchObject({ nullFa: "0.4%", nullFaNote: "240 simulated nulls", survival: 0.75, armB: null });
    // GLS and conformal screens never ran on the Atlas or the synthetic grid: no borrowed numbers.
    for (const r of rows.slice(1, 4)) {
      expect(r.atlas).toBeNull();
      expect(r.synthetic).toBeNull();
    }
    expect(rows[2]!.frozen).toBe("Δχ² > 20.50 · exact level 0.99%");
    expect(rows[4]!.frozen).toBe("σ_sys 0.4356″ · locked Gate #1");
    expect(rows[5]!.frozen).toBe("σ_sys 0.3100″ · fresh calibration half");
    expect(rows[4]!.atlas).not.toBeNull();
    expect(rows[5]!.atlas).toBeNull();
    expect(archiveCell(null)).toBeNull();
    expect(syntheticCell(undefined)).toBeNull();
  });

  it("reports provenance: star overlap, freeze lead and noise correlation", () => {
    expect(overlapStatement(comparison.provenance)).toBe("0 of 200 fresh stars among the 154 Gate #1 stars");
    expect(freezeLeadSeconds(comparison.provenance)).toBe(2.5);
    expect(lag1Text([0.21, 0.19])).toBe("0.210 / 0.190");
    expect(lag1Text(null)).toBe("—");
  });

  it("states the like-for-like scalar chi2 result as two numbers and their difference, ungraded", () => {
    const text = likeForLikeStatement(comparison)!;
    expect(text).toContain("70.0% of blends within 3.25″ survive on unsmoothed unWISE versus 90.0% on the AllWISE Atlas");
    expect(text).toContain("−20.0 points");
    expect(text).toContain("share no star");
    expect(text).not.toMatch(/rescue|fail|success/i);
    const worse = {
      ...comparison,
      image_screens: [imageRow("image_scalar_chi2", "chi2", 0.95, cell(5, 77, 0.9, 24))],
    };
    expect(likeForLikeStatement(worse)).toContain("+5.0 points");
    const noAtlas = { ...comparison, image_screens: [imageRow("image_scalar_chi2", "chi2", 0.7, null)] };
    expect(likeForLikeStatement(noAtlas)).toBeNull();
  });
});
