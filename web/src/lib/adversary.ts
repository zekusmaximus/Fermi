// Pure helpers for the Adversary view (unit-tested in adversary.test.ts). No I/O, no DOM.
import type { DecSignAuditDetails, Finding } from "../api";
import { fmtCount } from "./background";

/**
 * The typed details of the `catalog_dec_sign_audit` finding (ruling v0.2.1-R5), or null for
 * any other finding or a payload that lacks the fields the view reads.
 */
export function decSignAudit(f: Finding): DecSignAuditDetails | null {
  if (f.check !== "catalog_dec_sign_audit") return null;
  const d = f.details as Partial<DecSignAuditDetails>;
  if (!Array.isArray(d.affected_targets) || !Array.isArray(d.northern_controls) || !d.band_census || !d.locked_catalog) {
    return null;
  }
  return d as DecSignAuditDetails;
}

const band = ([lo, hi]: readonly [number, number]) => {
  const sign = (x: number) => (x < 0 ? `−${Math.abs(x)}` : `+${x}`);
  return `${sign(lo)}° to ${sign(hi)}°`;
};

/** "14 of 692 targets are catalogued at Dec +10° to +11° but lie at −11° to −10° (20.13°–21.87° away); …" */
export function decSignHeadline(d: DecSignAuditDetails): string {
  const offsets = d.affected_targets.map((t) => t.offset_deg);
  const range = offsets.length ? `${Math.min(...offsets).toFixed(2)}°–${Math.max(...offsets).toFixed(2)}° away` : "no offsets";
  const c = d.band_census;
  return (
    `${fmtCount(d.n_affected_targets)} of ${fmtCount(d.n_catalog_targets)} targets are catalogued at Dec ${band(c.flipped_band_deg)} ` +
    `but lie at ${band(c.mirror_band_deg)} (${range}); the catalog has ${fmtCount(c.n_targets_in_mirror_band)} targets in the ` +
    `${band(c.mirror_band_deg)} band and ${fmtCount(c.n_targets_in_flipped_band)} in the ${band(c.flipped_band_deg)} band.`
  );
}

/** Second upstream table discrepancy, stated as a table discrepancy and nothing more. */
export function secondaryNote(d: DecSignAuditDetails): string | null {
  const s = d.secondary_discrepancies[0];
  if (!s) return null;
  const names = s.records.map((r) => r.target_name).join(", ");
  return (
    `Separate upstream discrepancy: 692stars.csv prints a plus sign for ${fmtCount(s.records.length)} targets just south of the ` +
    `equator (${names}). The locked catalog ${s.inherited_by_locked_catalog ? "INHERITS it" : "does not inherit it"}. ` +
    `Telescope pointing: ${s.telescope_pointing_claim}.`
  );
}
