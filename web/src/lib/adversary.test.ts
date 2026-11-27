import { describe, expect, it } from "vitest";
import type { DecSignAuditDetails, Finding } from "../api";
import { decSignAudit, decSignHeadline, secondaryNote } from "./adversary";

const DETAILS: DecSignAuditDetails = {
  claim_path: "data/adversary_claims/isaacson_2017_dec_sign_flip.yaml",
  claim_sha256: "7d976fe57e93ca1f38020529ae7e7f2fe106074576223c914547568b35027e74",
  ruling: "v0.2.1-R5",
  locked_catalog: {
    path: "data/catalogs/bl_692_targets.csv",
    sha256: "c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2",
    checksum_verified: true,
    modified_by_this_audit: false,
  },
  band_census: {
    flipped_band_deg: [10, 11],
    n_targets_in_flipped_band: 16,
    mirror_band_deg: [-11, -10],
    n_targets_in_mirror_band: 0,
  },
  match_tolerance_arcsec: 2,
  n_affected_targets: 14,
  n_catalog_targets: 692,
  affected_targets: [
    {
      target_name: "HIP 22288",
      gaia_dr3_source_id: "3181588219554237184",
      catalog_dec_deg: 10.933611,
      true_dec_deg: -10.933646,
      truth_source: "gaia_dr3",
      truth_data_tier: "reproduced_archive",
      offset_deg: 21.8673,
      sign_restored_separation_arcsec: 0.624,
      raw_isaacson_dec: "+10 56 01",
      raw_692stars_dec: "-10:56:02.0",
    },
    {
      target_name: "HIP 76219",
      gaia_dr3_source_id: null,
      catalog_dec_deg: 10.064444,
      true_dec_deg: -10.064533,
      truth_source: "hip2",
      truth_data_tier: "reproduced_archive",
      offset_deg: 20.129,
      sign_restored_separation_arcsec: 0.323,
      raw_isaacson_dec: "+10 03 52",
      raw_692stars_dec: "-10:03:55.8",
    },
  ],
  northern_controls: [
    { target_name: "HIP 59608", catalog_dec_deg: 10.262222, corrected: false },
    { target_name: "HIP 92283", catalog_dec_deg: 10.745278, corrected: false },
  ],
  secondary_discrepancies: [
    {
      id: "enriquez_2017_692stars_minus_zero_sign_drop",
      data_tier: "published_catalog",
      description: "three targets printed with a plus sign",
      inherited_by_locked_catalog: false,
      telescope_pointing_claim: "none (filterbank headers not verified)",
      records: [
        { target_name: "HIP 99711", catalog_dec_deg: -0.866667, raw_isaacson_dec: "-00 52 00", raw_692stars_dec: "+00:51:55.4" },
        { target_name: "HIP 12530", catalog_dec_deg: -0.695556, raw_isaacson_dec: "-00 41 44", raw_692stars_dec: "+00:41:46.3" },
        { target_name: "HIP 82588", catalog_dec_deg: -0.026389, raw_isaacson_dec: "-00 01 35", raw_692stars_dec: "+00:01:58.1" },
      ],
    },
  ],
};

const finding = (check: string, details: unknown): Finding => ({
  check,
  status: "finding",
  severity: "warning",
  summary: "",
  details: details as Record<string, unknown>,
  citations: [],
  repro_command: "uv run fermi-desk adversary run --seed 42",
});

describe("catalog_dec_sign_audit finding", () => {
  it("is recognised only by its check name and shape", () => {
    expect(decSignAudit(finding("catalog_dec_sign_audit", DETAILS))).toBe(DETAILS);
    expect(decSignAudit(finding("hip1692_distance_revision", DETAILS))).toBeNull();
    expect(decSignAudit(finding("catalog_dec_sign_audit", { error: "claim unreadable" }))).toBeNull();
  });
  it("states the flipped band, its mirror band and the offsets in plain words", () => {
    expect(decSignHeadline(DETAILS)).toBe(
      "14 of 692 targets are catalogued at Dec +10° to +11° but lie at −11° to −10° (20.13°–21.87° away); " +
        "the catalog has 0 targets in the −11° to −10° band and 16 in the +10° to +11° band.",
    );
  });
  it("describes the 692stars.csv rows as a table discrepancy with no pointing claim", () => {
    const note = secondaryNote(DETAILS)!;
    expect(note).toContain("692stars.csv prints a plus sign for 3 targets");
    expect(note).toContain("HIP 99711, HIP 12530, HIP 82588");
    expect(note).toContain("does not inherit it");
    expect(note).toContain("Telescope pointing: none (filterbank headers not verified).");
    expect(secondaryNote({ ...DETAILS, secondary_discrepancies: [] })).toBeNull();
  });
  it("keeps Gaia ids as strings or null (never numbers)", () => {
    for (const t of DETAILS.affected_targets) {
      expect(t.gaia_dr3_source_id === null || typeof t.gaia_dr3_source_id === "string").toBe(true);
    }
  });
});
