// Pure helpers for the Macro Desk geometry selector, the local-approximation validity
// badge and the lookback-marginalisation toggle (unit-tested in geometry.test.ts).
// No I/O, no DOM.
import type { CompareRequest, DiskProfile, Geometry } from "../api";

export const GEOMETRY_OPTIONS: readonly Geometry[] = ["thin_disk_2d", "thick_disk_3d", "finite_disk_3d"];

export const DISK_PROFILE_OPTIONS: readonly DiskProfile[] = ["uniform", "exponential"];

export const DISK_PROFILE_LABELS: Record<DiskProfile, string> = {
  uniform: "Uniform Slab",
  exponential: "Exponential Disk",
};

export const MARGINALIZE_LABEL = "Marginalize Lookback T (50 kyr–10 Myr)";

/** Status string the server reports for the finite Galactic disk (engine: local_approx). */
export const FINITE_STATUS = "valid_finite_galaxy";

export function isFiniteGeometry(geometry: string | null | undefined): boolean {
  return geometry === "finite_disk_3d";
}

/** The disk-profile toggle is only meaningful (and only shown) for finite_disk_3d. */
export function showDiskProfile(geometry: string | null | undefined): boolean {
  return isFiniteGeometry(geometry);
}

export type LocalApproxKind = "ok" | "violated" | "finite";

export interface LocalApproxBadge {
  kind: LocalApproxKind;
  label: string;
  className: string;
  title: string;
  /** Infinite-disk validity cutoff, shown only on a violated badge. */
  tValidKyr: number | null;
}

interface ExpansionLike {
  local_approximation_valid: boolean;
  local_approx?: string | null;
  t_valid_kyr?: number | null;
}

/**
 * Validity badge for one scored hypothesis. The green/cyan "VALID (FINITE GALAXY)" badge
 * is shown ONLY when the server's receipt says `local_approx: valid_finite_galaxy`; it is
 * never inferred from the geometry the user selected, so a stale infinite-disk response
 * can never be dressed up as valid. Anything not explicitly valid is treated as violated.
 */
export function localApproxBadge(expansion: ExpansionLike | null | undefined): LocalApproxBadge | null {
  if (!expansion) return null;
  if (expansion.local_approx === FINITE_STATUS) {
    return {
      kind: "finite",
      label: "VALID (FINITE GALAXY)",
      className: "badge-finite",
      title: "Finite Galactic disk: origins are confined to R ≤ R_gal, so there is no infinite-disk validity limit",
      tValidKyr: null,
    };
  }
  const violated = expansion.local_approx === "VIOLATED" || expansion.local_approximation_valid !== true;
  if (violated) {
    const t = expansion.t_valid_kyr;
    return {
      kind: "violated",
      label: "LOCAL APPROX VIOLATED",
      className: "badge-violated",
      title: "v_exp × T exceeds the 8 kpc infinite-disk validity radius",
      tValidKyr: typeof t === "number" && Number.isFinite(t) ? t : null,
    };
  }
  return { kind: "ok", label: "ok", className: "ok", title: "within the local infinite-disk validity radius", tValidKyr: null };
}

export interface CompareControls {
  geometry: Geometry;
  diskProfile: DiskProfile;
  marginalizeLookback: boolean;
}

/**
 * Geometry part of the /api/compare body. `disk_profile` is sent as the selected value
 * only for finite_disk_3d and as the default otherwise, so toggling the (hidden) profile
 * never changes an infinite-disk request or its server cache key.
 */
export function geometryRequest(c: CompareControls): Pick<CompareRequest, "geometry_mode" | "disk_profile" | "marginalize_lookback"> {
  return {
    geometry_mode: c.geometry,
    disk_profile: isFiniteGeometry(c.geometry) ? c.diskProfile : "uniform",
    marginalize_lookback: c.marginalizeLookback,
  };
}

export interface LookbackPrior {
  prior: { effective_bounds_gyr: [number, number] };
  quadrature: { n_points: number; rule: string };
}

const fmtSpan = (gyr: number): string => {
  const kyr = gyr * 1e6;
  return kyr < 1000 ? `${Number(kyr.toPrecision(4))} kyr` : `${Number((kyr / 1000).toPrecision(4))} Myr`;
};

/** One-line statement of how T was treated, read from the receipt (null => fixed T). */
export function lookbackSummary(marg: LookbackPrior | null | undefined): string {
  if (!marg) return "lookback T fixed at the registry default";
  const [lo, hi] = marg.prior.effective_bounds_gyr;
  return (
    `lookback T marginalised over a log-uniform prior ${fmtSpan(lo)}–${fmtSpan(hi)} ` +
    `(${marg.quadrature.n_points}-point ${marg.quadrature.rule}, likelihood space)`
  );
}
