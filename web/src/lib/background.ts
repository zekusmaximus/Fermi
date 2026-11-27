// Pure formatting helpers for the Radio Drill-Down "Background Field Extension" section
// (unit-tested in background.test.ts). No I/O, no DOM. Every number shown comes from the
// /api/radio/drilldown payload through these functions; nothing is remembered client-side.
import type {
  BackgroundAttenuation,
  BackgroundDedup,
  BackgroundDistanceCuts,
  BackgroundScope,
  EigComparisonRow,
  MergeProvenance,
  PointingCentreAudit,
} from "../api";

/** Integer with thousands separators ("120,176"); "—" for a missing value. */
export function fmtCount(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

/** Bits with 4 significant figures; never rounds a non-zero value to "0". */
export function fmtBits(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  if (x === 0) return "0";
  return Math.abs(x) < 1e-3 ? x.toExponential(2) : x.toPrecision(4);
}

/**
 * A Gaia DR3 source id for display. Ids are 64-bit integers above 2^53, so the API sends
 * them as decimal strings (ruling v0.2.1-R4) and they must stay strings: converting through
 * Number would silently change the digits. Anything that is not a clean decimal string is
 * rejected rather than shown.
 */
export function gaiaIdText(id: unknown): string {
  return typeof id === "string" && /^[1-9][0-9]{0,18}$/.test(id) ? id : "invalid id";
}

/** "4.2′ (FWHM/2)" — the cone radius, stated with the beam it derives from. */
export function coneLabel(coneArcmin: number, fwhmArcmin: number): string {
  const half = Math.abs(coneArcmin * 2 - fwhmArcmin) < 1e-9;
  return `${coneArcmin}′${half ? " (FWHM/2)" : ""} of the ${fwhmArcmin}′ beam`;
}

/** "2026-11-27 16:18:22 → 16:39:02 UTC" (same day) or two full timestamps. */
export function utcWindow(window: readonly string[] | null | undefined): string {
  const [a, b] = window ?? [];
  if (!a || !b) return "—";
  const [da, ta] = [a.slice(0, 10), a.slice(11, 19)];
  const [db, tb] = [b.slice(0, 10), b.slice(11, 19)];
  return da === db ? `${da} ${ta} → ${tb} UTC` : `${da} ${ta} → ${db} ${tb} UTC`;
}

export interface CountRow {
  label: string;
  value: number;
  /** "drop" rows are subtracted on the way down; "flag" rows are reported in amber. */
  kind?: "total" | "drop" | "keep" | "flag";
}

/** Raw cone sightings → sightings with a usable distance, every cut counted. */
export function distanceCutRows(c: BackgroundDistanceCuts): CountRow[] {
  return [
    { label: "raw Gaia DR3 cone sightings", value: c.n_cone_sightings_raw, kind: "total" },
    { label: "dropped: parallax S/N ≤ 1 or no parallax", value: c.n_dropped_invalid_parallax, kind: "drop" },
    {
      label: `dropped: ${c.bailer_jones_min_snr_exclusive} < S/N < ${c.inverse_parallax_min_snr} without a Bailer-Jones distance`,
      value: c.n_dropped_no_bailer_jones,
      kind: "drop",
    },
    { label: `dropped: beyond ${fmtCount(c.d_max_pc)} pc`, value: c.n_dropped_beyond_d_max, kind: "drop" },
    { label: "dropped: outside the cone on recheck", value: c.n_dropped_outside_cone_recheck, kind: "drop" },
    { label: "sightings with a distance", value: c.n_sightings_with_distance, kind: "total" },
    { label: `inverse parallax (S/N ≥ ${c.inverse_parallax_min_snr})`, value: c.n_sightings_inverse_parallax, kind: "keep" },
    { label: "Bailer-Jones et al. (2021) geometric", value: c.n_sightings_bailer_jones_geo, kind: "keep" },
  ];
}

/** De-duplication counters, ending with the reported (never removed) suspected primaries. */
export function dedupRows(mp: MergeProvenance, d: BackgroundDedup): CountRow[] {
  return [
    { label: "raw rows merged (primaries + sightings)", value: mp.n_raw_targets, kind: "total" },
    { label: "removed by Gaia DR3 source_id (gaia_id)", value: mp.n_dedup_gaia_id, kind: "drop" },
    { label: `removed by < ${mp.angular_fallback_arcsec}″ fallback (no Gaia id)`, value: mp.n_dedup_angular_fallback, kind: "drop" },
    { label: "multi-beam stars kept at θ_min", value: mp.n_multi_beam_resolved, kind: "keep" },
    { label: "background sightings absorbed by primaries", value: mp.n_background_absorbed_by_primary, kind: "keep" },
    { label: "duplicate Gaia ids vs bl_692_targets.csv", value: d.n_duplicate_gaia_ids_vs_primary, kind: "keep" },
    { label: "unique background stars N_bg", value: mp.n_merged_by_role["background"] ?? 0, kind: "total" },
    { label: "suspected_unmatched_primary rows (kept, reported)", value: d.n_suspected_unmatched_primary, kind: "flag" },
    { label: "distinct suspected stars still in the shells", value: d.n_suspected_unmatched_primary_stars, kind: "flag" },
  ];
}

/** The dedup arithmetic must close: raw − removed = merged. False means "do not trust". */
export function dedupCloses(mp: MergeProvenance): boolean {
  return mp.n_raw_targets - mp.n_dedup_gaia_id - mp.n_dedup_angular_fallback === mp.n_merged_targets;
}

export interface ScopeRow {
  field: string;
  retrieved: string;
  published: string;
  /** Set when the published value was not verified against the paper. */
  caveat?: string;
}

/** Side-by-side scope: what was retrieved here vs the Wlodarczyk-Sroka et al. (2020) table. */
export function scopeRows(s: BackgroundScope): ScopeRow[] {
  const lband = s.published.n_stars_gbt_l_band;
  return [
    { field: "pointings", retrieved: fmtCount(s.retrieved.n_pointings), published: fmtCount(s.published.n_pointings) },
    { field: "pointing set", retrieved: s.retrieved.pointings, published: s.published.pointings },
    { field: "stars", retrieved: fmtCount(s.retrieved.n_background_stars), published: fmtCount(s.published.n_stars_total) },
    {
      field: "GBT L-band stars",
      retrieved: fmtCount(s.retrieved.n_background_stars),
      published: fmtCount(lband.value),
      ...(lband.verified_against_paper ? {} : { caveat: `unverified — ${lband.source}` }),
    },
    { field: "Gaia release", retrieved: s.retrieved.gaia_release, published: s.published.gaia_release },
    { field: "distances", retrieved: s.retrieved.distances, published: s.published.distances },
    { field: "selection", retrieved: s.retrieved.selection, published: s.published.selection },
  ];
}

export interface EigDisplayRow {
  mode: string;
  /** "physical" | "unphysical" — never shown without it. */
  label: string;
  physical: boolean;
  predicted: string;
  realized: string;
  /** e.g. "0.16× predicted". */
  ratio: string;
  range: string;
  withoutSuspected: string;
  observedNull: string;
  /** True when any tier is illustrative: the row is drawn amber/hatched, never as verified. */
  illustrative: boolean;
}

const ILLUSTRATIVE = new Set(["illustrative_calibration", "synthetic_proxy", "synthetic_proxy_692"]);

/** Realized-vs-predicted EIG rows, physical mode first. */
export function eigRows(rows: readonly EigComparisonRow[]): EigDisplayRow[] {
  return [...rows]
    .sort((a, b) => Number(b.physical) - Number(a.physical))
    .map((r) => {
      const [lo, hi] = r.realized_eig_bits_range_fkg_to_frechet;
      const ratio = r.realized_over_predicted;
      return {
        mode: r.radio_mode,
        label: r.label,
        physical: r.physical,
        predicted: fmtBits(r.predicted_eig_bits),
        realized: fmtBits(r.realized_eig_bits),
        ratio: ratio === null || !Number.isFinite(ratio) ? "—" : `${ratio.toPrecision(2)}× predicted`,
        range: `${fmtBits(lo)} – ${fmtBits(hi)}`,
        withoutSuspected: fmtBits(r.realized_eig_bits_excluding_suspected_unmatched_primary),
        observedNull: fmtBits(r.realized_information_bits_from_observed_null),
        illustrative: r.data_tiers.some((t) => ILLUSTRATIVE.has(t)),
      };
    });
}

/** Plain-English reading of the comparison; a smaller realized value is reported as such. */
export function eigVerdict(rows: readonly EigComparisonRow[]): string {
  const phys = rows.find((r) => r.physical);
  if (!phys) return "no physical-mode comparison available";
  const ratio = phys.realized_over_predicted;
  if (ratio === null || !Number.isFinite(ratio)) return "physical-mode prediction was zero; no ratio";
  const how = ratio < 1 ? "less informative than predicted" : ratio > 1 ? "more informative than predicted" : "as predicted";
  return `Physical mode: the real background field is ${how} (${fmtBits(phys.realized_eig_bits)} vs ${fmtBits(phys.predicted_eig_bits)} bits) — a null-strength result either way (≪ 1 bit).`;
}

/** "×1.6e+4 (distance ×1.1e+4, beam ×1.43)" — why off-axis background EIRP_min is so high. */
export function attenuationHeadline(a: BackgroundAttenuation): string {
  const d = a.decomposition_of_the_median_ratio;
  const big = (x: number) => (x >= 1000 ? x.toExponential(1) : x.toPrecision(3));
  return `median EIRP_min is ×${big(a.median_eirp_min_ratio_background_over_primary)} the on-axis primaries’ (distance term ×${big(d.distance_term)}, beam term ×${big(d.beam_term)})`;
}

/** Power in watts as "1.68e+16 W". */
export function fmtWatts(x: number | null | undefined): string {
  return x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x.toExponential(2)} W`;
}

/** One-line caveat for the pointing-centre audit, or null when nothing is flagged. */
export function auditCaveat(a: PointingCentreAudit | null | undefined): string | null {
  const d = a?.densest_band_of_unfound_primaries;
  if (!a || !d || d.n_no_counterpart === 0) return null;
  const [lo, hi] = d.declination_band_deg;
  const sign = (x: number) => (x >= 0 ? `+${x}` : `${x}`);
  return (
    `${d.n_no_counterpart} of the ${d.n_targets} primaries catalogued at Dec ${sign(lo)}° to ${sign(hi)}° have no Gaia counterpart in their own cone ` +
    `(${d.n_targets_in_mirror_band} targets in the mirror band): ${fmtCount(d.n_background_stars_in_those_pointings)} background stars ` +
    `(${(d.fraction_of_background_stars * 100).toFixed(2)}% of N_bg) may come from the wrong field.`
  );
}
