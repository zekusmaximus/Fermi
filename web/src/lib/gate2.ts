// Pure helpers for the Launch Gate #2 views (unit-tested in gate2.test.ts). Every number
// comes from the promoted reproduced_archive payload; nothing here is a remembered value.
import type { Gate2, Gate2ScreenName } from "../api";

export type HeatSource = "real" | "synthetic";

export const SOURCE_LABEL: Record<HeatSource, string> = {
  real: "Real AllWISE W3 FITS — Withheld (reproduced_archive)",
  synthetic: "Synthetic Gaussian PSF (illustrative_calibration)",
};

export const SOURCE_TIER: Record<HeatSource, string> = {
  real: "reproduced_archive",
  synthetic: "illustrative_calibration",
};

/** The real withheld grid is the default whenever it is pinned; otherwise the synthetic one. */
export function defaultSource(g: Gate2 | null | undefined): HeatSource {
  return g ? "real" : "synthetic";
}

/**
 * Arm A recovery per f_bg column, area-weighted over theta <= maxSep (p ~ theta) - the
 * same weighting as the server's blend survival, so 1 - mean(result) == survival.
 */
export function armARecoveryByFlux(rec: number[][], seps: number[], maxSep = 3.25): number[] {
  const idx = seps.map((t, i) => [t, i] as const).filter(([t]) => t <= maxSep);
  const wsum = idx.reduce((a, [t]) => a + t, 0);
  const ncol = rec[0]?.length ?? 0;
  if (wsum <= 0 || ncol === 0) return [];
  return Array.from({ length: ncol }, (_, j) => idx.reduce((a, [t, i]) => a + (t / wsum) * (rec[i]?.[j] ?? 0), 0));
}

export interface MatchedCell {
  fBg: number;
  armA: number;
  armB: number;
  armBk: number;
  n: number;
  armBcp: [number, number];
  /** Blend recall minus centered-excess false rejection (Youden-style net discrimination). */
  net: number;
}

export const SCREEN_ORDER: Gate2ScreenName[] = ["centroid", "image", "naive_flux_excess"];

export const SCREEN_LABEL: Record<Gate2ScreenName, string> = {
  centroid: "Centroid screen (frozen)",
  image: "Image-level Δχ² screen (frozen)",
  naive_flux_excess: "Naive 'flag any bright excess' (adversarial reference)",
};

export function matchedControl(g: Gate2, screen: Gate2ScreenName, maxSep = 3.25): MatchedCell[] {
  const s = g.screens[screen];
  const a = armARecoveryByFlux(s.arm_a_blend_recovery, g.separations_arcsec, maxSep);
  return g.flux_fractions.map((f, j) => {
    const armA = a[j] ?? Number.NaN;
    const armB = s.arm_b_false_rejection[j] ?? Number.NaN;
    return {
      fBg: f,
      armA,
      armB,
      armBk: s.arm_b_k[j] ?? 0,
      n: s.n_stars,
      armBcp: s.arm_b_clopper_pearson_95[j] ?? [Number.NaN, Number.NaN],
      net: armA - armB,
    };
  });
}

/** "k/n" for a cell of the real grid (for tooltips). */
export function kOverN(k: number | undefined, n: number): string {
  return k === undefined ? "—" : `${k}/${n}`;
}
