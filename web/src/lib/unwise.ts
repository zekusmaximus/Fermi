// Pure helpers for the unsmoothed unWISE W3 gate views (v0.2.2b; unit-tested in
// unwise.test.ts). Every number shown comes from the promoted reproduced_archive payload
// or the synthetic grid block; nothing here is a remembered value. No I/O, no DOM.
import type {
  Gate2,
  Rate,
  UnwiseArchiveCell,
  UnwiseComparison,
  UnwiseComparisonRow,
  UnwiseGate,
  UnwiseScreenName,
  UnwiseSyntheticCell,
} from "../api";
import { fmtPct } from "./format";
import type { HeatSource } from "./gate2";

/** Order of the heatmap source control: the two v0.1.2 sources, then unWISE third. */
export const HEAT_SOURCES: readonly HeatSource[] = ["real", "synthetic", "unwise"];

export const SOURCE_CONTROL_LABEL = "Image source";
export const SCREEN_CONTROL_LABEL = "Blend screen";

/** Exact artifact / tier enum shown in monospace under each plain-English source label. */
export const SOURCE_ENUM: Record<HeatSource, string> = {
  real: "wise_fits_gate2_withheld · reproduced_archive",
  synthetic: "synthetic_gaussian · illustrative_calibration",
  unwise: "unwise_w3_gate_withheld · reproduced_archive",
};

export function sourceAvailable(source: HeatSource, gate2: Gate2 | null | undefined, unwise: UnwiseGate | null | undefined): boolean {
  if (source === "real") return !!gate2;
  if (source === "unwise") return !!unwise;
  return true;
}

/** The requested source when its artifact is pinned, else the synthetic grid (always there). */
export function resolveSource(requested: HeatSource, gate2: Gate2 | null | undefined, unwise: UnwiseGate | null | undefined): HeatSource {
  return sourceAvailable(requested, gate2, unwise) ? requested : "synthetic";
}

const NOT_HEADLINE = "Diagnostic only: the verified headline does not use it.";

export interface SourceContext {
  gate2: Gate2 | null | undefined;
  unwise: UnwiseGate | null | undefined;
  synthetic: { psfFwhmArcsec: number; snr: number; nTrials: number };
}

/** One-line caption under the source control: what physically changes, and the headline. */
export function sourceCaption(source: HeatSource, ctx: SourceContext): string {
  const { gate2, unwise, synthetic } = ctx;
  if (source === "unwise" && unwise) {
    return (
      `Blends are injected into ${unwise.n_withheld} withheld unsmoothed unWISE cutouts (measured PSF ` +
      `${unwise.empirical_psf.fwhm_gaussian_arcsec.toFixed(2)}″ on ${unwise.empirical_psf.pixel_scale_arcsec}″ pixels) of a fresh ` +
      `|b| > 30° sample that shares no star with Gate #1. ${NOT_HEADLINE}`
    );
  }
  if (source === "real" && gate2) {
    return (
      `Blends are injected into ${gate2.n_withheld} withheld AllWISE Atlas cutouts, whose coadd smoothing widens the measured PSF to ` +
      `${gate2.empirical_psf.fwhm_gaussian_arcsec.toFixed(2)}″ (nominal ${gate2.empirical_psf.nominal_fwhm_arcsec}″). ${NOT_HEADLINE}`
    );
  }
  return (
    `Simulated stars with an ideal ${synthetic.psfFwhmArcsec.toFixed(1)}″ Gaussian PSF at W3 SNR ${synthetic.snr} ` +
    `(${synthetic.nTrials} injections per cell). Illustrative: it feeds only the illustrative LR rows, never the verified headline.`
  );
}

export interface HelpText {
  title: string;
  tradeoff: string;
  invariant: string;
  cli: readonly string[];
}

/** `?` popover of the source control: the disjoint |b| > 30 deg split and the freeze. */
export function sourceHelp(unwise: UnwiseGate | null | undefined, gate2: Gate2 | null | undefined): HelpText {
  const atlas = gate2 ? `${gate2.empirical_psf.fwhm_gaussian_arcsec.toFixed(2)}″` : "wider";
  const p = unwise?.comparison.provenance;
  const sample = unwise
    ? `${unwise.n_controls} isolated Gaia M dwarfs in ${unwise.n_patches} patches whose whole cone lies at |b| > 30° ` +
      `(${p!.n_gate1_star_overlap} of them among the ${p!.n_gate1_stars} Gate #1 stars)`
    : "a fresh sample at |b| > 30° that shares no star with Gate #1";
  const freeze = unwise
    ? `the PSF, the pixel-noise covariance and every threshold were frozen on the ${unwise.n_calibration} calibration stars at ` +
      `${unwise.frozen_at_utc}, before the first withheld cutout was fetched (${unwise.earliest_withheld_fetch_utc})`
    : "the PSF, the pixel-noise covariance and every threshold are frozen on the calibration half before any withheld cutout is fetched";
  return {
    title: "Three image sources, three data tiers",
    tradeoff:
      `AllWISE Atlas coadds are smoothed when they are built, so a star's W3 image is ${atlas} wide instead of the ` +
      `6.5″ single-frame PSF, and two close sources melt into one. unWISE coadds (Lang 2014) are not smoothed, at the price of coarser ` +
      `2.75″ pixels and noisier single stars. The synthetic grid is an ideal Gaussian at one SNR: cleanest, but not data. Switching ` +
      `source changes only this heatmap; the verified headline never uses any of the three.`,
    invariant:
      `Zero leakage: the unWISE test uses ${sample}; ${freeze}. Nothing was retuned afterwards — the withheld result is ` +
      `shown as measured. The two archive sources are cyan (reproduced_archive), the synthetic grid amber (illustrative_calibration); ` +
      `they are never averaged together.`,
    cli: ["uv run fermi-desk ir-drilldown --seed 42", "uv run python scripts/run_unwise_gate.py --dry-run"],
  };
}

// ------------------------------------------------------------------ screens
/** Screens offered for the unWISE source: both centroid floors, then the 2 x 2 image screens. */
export const UNWISE_HEAT_SCREENS: readonly UnwiseScreenName[] = [
  "centroid_locked",
  "centroid_fresh",
  "image_scalar_chi2",
  "image_gls_chi2",
  "image_scalar_conformal",
  "image_gls_conformal",
];

export const UNWISE_SCREEN_LABEL: Record<UnwiseScreenName, string> = {
  centroid_locked: "Centroid · locked floor",
  centroid_fresh: "Centroid · fresh floor",
  image_scalar_chi2: "Scalar χ²",
  image_gls_chi2: "GLS χ²",
  image_scalar_conformal: "Scalar Conformal",
  image_gls_conformal: "GLS Conformal",
  naive_flux_excess: "Naive bright excess",
};

export type LegacyScreen = "centroid" | "image";
export const LEGACY_SCREENS: readonly LegacyScreen[] = ["centroid", "image"];
export const LEGACY_SCREEN_LABEL: Record<LegacyScreen, string> = {
  centroid: "Centroid shift",
  image: "Image fit (Scalar χ²)",
};
export const LEGACY_SCREEN_ENUM: Record<LegacyScreen, string> = { centroid: "centroid", image: "image" };

/** The unWISE screen that applies the same rule as a legacy (Gate #2 / synthetic) screen. */
export function unwiseCounterpart(screen: LegacyScreen): UnwiseScreenName {
  return screen === "centroid" ? "centroid_locked" : "image_scalar_chi2";
}

/** The legacy screen to show when leaving the unWISE source (GLS / conformal have none). */
export function legacyCounterpart(screen: UnwiseScreenName): LegacyScreen {
  return screen.startsWith("centroid") ? "centroid" : "image";
}

const SCREEN_WHAT: Record<UnwiseScreenName, string> = {
  centroid_locked: "Flags a star whose W3 centroid sits too far from its Gaia position, using the registration floor locked in Gate #1",
  centroid_fresh: "Same centroid test, with the registration floor re-measured on this fresh calibration half",
  image_scalar_chi2: "Flags a star when a two-source fit beats the single-star fit, treating pixel noise as uncorrelated (the Gate #2 rule)",
  image_gls_chi2: "Same two-source fit after whitening the pixels with the measured noise covariance",
  image_scalar_conformal: "Uncorrelated-noise fit, with the cutoff set by the largest calibration null instead of a χ² tail",
  image_gls_conformal: "Noise-whitened fit, with the cutoff set by the largest calibration null instead of a χ² tail",
  naive_flux_excess: "Flags any bright excess (adversarial reference, not a proposed screen)",
};

/** "5/77 = 6.5%" */
export function fmtRate(r: Rate): string {
  return `${r.k}/${r.n} = ${fmtPct(r.rate)}`;
}

/** "95% CP 2.1%–14.5%" (exact Clopper-Pearson bounds). */
export function fmtCp(r: Rate): string {
  return `95% CP ${fmtPct(r.clopper_pearson_95[0])}–${fmtPct(r.clopper_pearson_95[1])}`;
}

/** One-line caption under the screen control for the unWISE source. */
export function unwiseScreenCaption(g: UnwiseGate, screen: UnwiseScreenName): string {
  const s = g.screens[screen];
  return (
    `${SCREEN_WHAT[screen]}. Withheld null false alarms ${fmtRate(s.null_false_alarm)}; ` +
    `${fmtPct(s.blend_survival_within_3p25_arcsec)} of blends within 3.25″ survive. ${NOT_HEADLINE}`
  );
}

/** One-line caption under the screen control for the Gate #2 and synthetic sources. */
export function legacyScreenCaption(screen: LegacyScreen, source: "real" | "synthetic"): string {
  const what =
    screen === "centroid"
      ? "Flags a star whose W3 centroid sits too far from its Gaia position"
      : "Flags a star when a two-source fit beats the single-star fit, treating pixel noise as uncorrelated";
  const tail =
    source === "real" ? NOT_HEADLINE : "Illustrative: it feeds only the illustrative LR rows, never the verified headline.";
  return `${what}. ${tail}`;
}

export function screenHelp(g: UnwiseGate | null | undefined): HelpText {
  const level = g ? conformalLevel(g) : null;
  const fa = g ? fmtPct(g.target_false_alarm) : "0.27%";
  return {
    title: "Two centroid floors and a 2 × 2 of image screens",
    tradeoff:
      `A centroid screen needs only a catalog position but misses a blend that barely moves the centroid. An image screen fits two ` +
      `sources to the pixels: more sensitive, but its false-alarm rate depends on the noise model. The 2 × 2 crosses the noise model ` +
      `(Scalar = uncorrelated pixels, GLS = measured pixel covariance) with the cutoff rule (χ² = an assumed χ²(3) tail at ${fa}, ` +
      `Conformal = the largest calibration null${level ? `, exact level ${level.text}` : ""}). GLS and Conformal exist only for unWISE.`,
    invariant:
      `All four image screens and both centroid floors were declared before any withheld cutout was fetched (rulings v0.2.2-R1 and ` +
      `v0.2.2-R2), so no corner of the 2 × 2 is chosen after the fact. The locked 0.4356″ Gate #1 floor is never overwritten; the fresh ` +
      `floor is reported beside it.`,
    cli: ["uv run fermi-desk ir-drilldown --seed 42", "uv run fermi-desk validate"],
  };
}

// ---------------------------------------------------------------- comparison
export interface ConformalLevel {
  /** "1/127 = 0.79%" */
  text: string;
  level: number;
  fallback: boolean;
  requested: number;
}

/** Exact finite-sample level of the frozen conformal screens, (k + 1) / (n_cal + 1). */
export function conformalLevel(g: UnwiseGate): ConformalLevel {
  const k = Number(g.frozen_screens.image_scalar_conformal.k_exceedances ?? 0);
  const n = Number(g.frozen_screens.image_scalar_conformal.n_calibration ?? g.n_calibration);
  const level = (k + 1) / (n + 1);
  return {
    text: `${k + 1}/${n + 1} = ${(100 * level).toFixed(2)}%`,
    level,
    fallback: g.conformal_alpha_fallback,
    requested: g.conformal_alpha_requested,
  };
}

export interface PsfRow {
  key: "unwise_w3" | "allwise_atlas_gate2" | "nominal";
  label: string;
  fwhmArcsec: number;
  /** data_tier of a measured width; null for the nominal single-frame value. */
  tier: string | null;
  /** Width relative to the nominal single-frame W3 PSF. */
  ratioToNominal: number;
  detail: string;
}

/** Empirical W3 PSF FWHM: unWISE vs the Atlas (Gate #2) vs the nominal single-frame value. */
export function psfRows(c: UnwiseComparison): PsfRow[] {
  const nominal = c.nominal_w3_fwhm_arcsec;
  const rows: PsfRow[] = [];
  const add = (key: "unwise_w3" | "allwise_atlas_gate2", label: string) => {
    const s = c.sources[key];
    if (!s) return;
    rows.push({
      key,
      label,
      fwhmArcsec: s.psf_fwhm_arcsec,
      tier: s.data_tier,
      ratioToNominal: s.psf_fwhm_arcsec / nominal,
      detail: `${s.n_psf_stars ?? "—"} calibration stars stacked, ${s.pixel_scale_arcsec === null ? "—" : +s.pixel_scale_arcsec.toFixed(3)}″ pixels`,
    });
  };
  add("unwise_w3", "unWISE W3 (unsmoothed)");
  add("allwise_atlas_gate2", "AllWISE W3 Atlas (Gate #2)");
  rows.push({
    key: "nominal",
    label: "Nominal single-frame W3",
    fwhmArcsec: nominal,
    tier: null,
    ratioToNominal: 1,
    detail: "WISE Explanatory Supplement value; also the synthetic Gaussian PSF",
  });
  return rows;
}

export interface ComparisonCell {
  /** Withheld null false alarm, e.g. "5/77 = 6.5%". */
  nullFa: string;
  /** Exact interval of the null false alarm ("95% CP …") or the trial count for synthetic. */
  nullFaNote: string;
  /** Compact line shown under the rate: the bare 95% interval, or the trial count. */
  nullFaBelow: string;
  /** Arm A blend survival within 3.25 arcsec, as a probability. */
  survival: number;
  /** Arm B matched centered-excess false rejection ("12/385 = 3.1%"); null when not measured. */
  armB: string | null;
  armBNote: string | null;
  armBBelow: string | null;
}

/** "2.1%–14.5%": the bare exact 95% Clopper-Pearson interval (the legend names it). */
export function fmtInterval(r: Rate): string {
  return `${fmtPct(r.clopper_pearson_95[0])}–${fmtPct(r.clopper_pearson_95[1])}`;
}

export function archiveCell(c: UnwiseArchiveCell | null | undefined): ComparisonCell | null {
  if (!c) return null;
  return {
    nullFa: fmtRate(c.null_false_alarm),
    nullFaNote: fmtCp(c.null_false_alarm),
    nullFaBelow: fmtInterval(c.null_false_alarm),
    survival: c.blend_survival_within_3p25_arcsec,
    armB: fmtRate(c.arm_b_false_rejection_pooled),
    armBNote: fmtCp(c.arm_b_false_rejection_pooled),
    armBBelow: fmtInterval(c.arm_b_false_rejection_pooled),
  };
}

export function syntheticCell(c: UnwiseSyntheticCell | null | undefined): ComparisonCell | null {
  if (!c) return null;
  return {
    nullFa: fmtPct(c.null_false_alarm_rate),
    nullFaNote: `${c.n_null_trials} simulated nulls`,
    nullFaBelow: `${c.n_null_trials} simulated nulls`,
    survival: c.blend_survival_within_3p25_arcsec,
    armB: null,
    armBNote: null,
    armBBelow: null,
  };
}

export interface ComparisonRow {
  screen: UnwiseScreenName;
  kind: "image" | "centroid";
  label: string;
  /** How the cutoff was frozen, from the payload (threshold, level or floor). */
  frozen: string;
  unwise: ComparisonCell;
  atlas: ComparisonCell | null;
  synthetic: ComparisonCell | null;
}

function frozenText(r: UnwiseComparisonRow): string {
  if (r.kind === "centroid") {
    return `σ_sys ${Number(r.sigma_sys_arcsec).toFixed(4)}″ · ${r.sigma_sys_source === "locked" ? "locked Gate #1" : "fresh calibration half"}`;
  }
  const thr = `Δχ² > ${Number(r.threshold).toFixed(2)}`;
  if (r.threshold_rule === "conformal") {
    return `${thr} · exact level ${(100 * Number(r.alpha_guaranteed)).toFixed(2)}%`;
  }
  return `${thr} · χ²(3) tail × ${Number(r.delta_chi2_scale).toFixed(2)}`;
}

/** The 2 x 2 image screens (payload order) followed by both centroid floors. */
export function comparisonRows(c: UnwiseComparison): ComparisonRow[] {
  return [...c.image_screens, ...c.centroid_floors].map((r) => ({
    screen: r.screen,
    kind: r.kind,
    label: UNWISE_SCREEN_LABEL[r.screen],
    frozen: frozenText(r),
    unwise: archiveCell(r.unwise_w3)!,
    atlas: archiveCell(r.allwise_atlas_gate2),
    synthetic: syntheticCell(r.synthetic_gaussian),
  }));
}

/** "0 of 252 fresh stars among the 154 Gate #1 stars" */
export function overlapStatement(p: UnwiseComparison["provenance"]): string {
  return `${p.n_gate1_star_overlap} of ${p.n_fresh_stars} fresh stars among the ${p.n_gate1_stars} Gate #1 stars`;
}

/** Seconds between the freeze and the first withheld fetch (positive = freeze came first). */
export function freezeLeadSeconds(p: UnwiseComparison["provenance"]): number {
  return (Date.parse(p.earliest_withheld_fetch_utc) - Date.parse(p.frozen_at_utc)) / 1000;
}

/** Lag-1 pixel-noise correlation "0.412 / 0.398" (x / y), or an em dash when not measured. */
export function lag1Text(v: readonly number[] | null | undefined): string {
  return v && v.length ? v.map((x) => x.toFixed(3)).join(" / ") : "—";
}

/**
 * Plain numeric reading of the like-for-like control (the scalar χ² rule on both archive
 * sources). It states the two measured survivals and their difference; it does not grade
 * the outcome.
 */
export function likeForLikeStatement(c: UnwiseComparison): string | null {
  const row = c.image_screens.find((r) => r.screen === "image_scalar_chi2");
  if (!row || !row.allwise_atlas_gate2) return null;
  const u = row.unwise_w3.blend_survival_within_3p25_arcsec;
  const a = row.allwise_atlas_gate2.blend_survival_within_3p25_arcsec;
  const d = u - a;
  return (
    `Same rule, different images: with the Gate #2 scalar χ² screen, ${fmtPct(u)} of blends within ${c.max_separation_arcsec}″ survive on ` +
    `unsmoothed unWISE versus ${fmtPct(a)} on the AllWISE Atlas (${d >= 0 ? "+" : "−"}${Math.abs(100 * d).toFixed(1)} points). ` +
    `The two samples share no star, so this is a comparison of two measurements, not a paired test.`
  );
}
