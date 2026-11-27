import { describe, expect, it } from "vitest";
import type {
  BackgroundAttenuation,
  BackgroundDedup,
  BackgroundDistanceCuts,
  BackgroundScope,
  EigComparisonRow,
  MergeProvenance,
  PointingCentreAudit,
} from "../api";
import {
  attenuationHeadline,
  auditCaveat,
  coneLabel,
  dedupCloses,
  dedupRows,
  distanceCutRows,
  eigRows,
  eigVerdict,
  fmtBits,
  fmtCount,
  fmtWatts,
  gaiaIdText,
  scopeRows,
  utcWindow,
} from "./background";

const CUTS: BackgroundDistanceCuts = {
  d_max_pc: 10000,
  inverse_parallax_min_snr: 5,
  bailer_jones_min_snr_exclusive: 1,
  n_cone_sightings_raw: 259285,
  n_dropped_invalid_parallax: 134601,
  n_dropped_no_bailer_jones: 0,
  n_dropped_beyond_d_max: 133,
  n_dropped_outside_cone_recheck: 0,
  n_sightings_with_distance: 124551,
  n_sightings_inverse_parallax: 37888,
  n_sightings_bailer_jones_geo: 86663,
  n_background_stars: 120176,
  n_background_inverse_parallax: 36069,
  n_background_bailer_jones_geo: 84107,
};

const MP: MergeProvenance = {
  pointing_group_id: "bl_gbt_lband_2017",
  policy: "union_targets_by_pointing",
  input_surveys: ["bl_enriquez_2017_692", "bl_gbt_lband_background_gaia_dr3"],
  input_sha256s: ["a".repeat(64), "b".repeat(64)],
  n_raw_targets: 125243,
  n_merged_targets: 120868,
  n_dedup_gaia_id: 4270,
  n_dedup_angular_fallback: 105,
  n_multi_beam_resolved: 3764,
  angular_fallback_arcsec: 3,
  n_raw_by_survey: { bl_enriquez_2017_692: 692, bl_gbt_lband_background_gaia_dr3: 124551 },
  n_merged_by_role: { primary: 692, background: 120176 },
  n_background_absorbed_by_primary: 635,
  n_angular_links_refused: 8,
  n_duplicate_gaia_ids_post_merge: 0,
};

const DEDUP: BackgroundDedup = {
  n_primary_targets: 692,
  n_primary_with_gaia_id: 522,
  n_primary_without_gaia_id: 170,
  n_background_sightings_matched_to_primary_gaia_id: 523,
  n_background_sightings_matched_to_primary_angular: 112,
  n_duplicate_gaia_ids_vs_primary: 0,
  n_duplicate_gaia_ids_within_background: 0,
  angular_fallback_arcsec: 3,
  angular_match_epoch: "J2000.0",
  n_suspected_unmatched_primary: 49,
  n_suspected_unmatched_primary_stars: 47,
  n_primaries_with_a_suspected_counterpart: 45,
  n_suspected_primaries_that_absorbed_a_sighting: 32,
  n_suspected_primaries_that_absorbed_nothing: 13,
  suspected_unmatched_primary_policy: "listed for review and kept in the shells, never removed (ruling v0.2.1-R3)",
};

const SCOPE: BackgroundScope = {
  retrieved: {
    survey_id: "bl_gbt_lband_background_gaia_dr3",
    data_tier: "reproduced_archive",
    pointings: "692 Enriquez et al. (2017) GBT L-band pointings",
    n_pointings: 692,
    n_background_stars: 120176,
    gaia_release: "Gaia DR3",
    distances: "inverse parallax / Bailer-Jones (2021)",
    selection: "inside the 8.4 arcmin FWHM beam",
    source_service: "ARI Heidelberg Gaia TAP",
  },
  published: {
    citation: "Wlodarczyk-Sroka, Garrett & Siemion 2020, MNRAS 498, 5720",
    data_tier: "published_catalog",
    pointings: "1,327 Price et al. (2020) primary targets",
    n_pointings: 1327,
    n_stars_total: 288315,
    n_stars_gbt_l_plus_s_band: { value: 249990, verified_against_paper: true, source: "Table 2" },
    n_stars_gbt_l_band: { value: 251983, verified_against_paper: false, source: "v0.2.1b planning brief" },
    gaia_release: "Gaia DR2",
    distances: "Bailer-Jones et al. (2018)",
    selection: "inside the primary-beam FWHM of each receiver",
  },
  comparable: false,
  statement: "not expected to agree",
};

const EIG: EigComparisonRow[] = [
  {
    radio_mode: "flat_probability",
    label: "unphysical",
    physical: false,
    predicted_eig_bits: 0.827341791920764,
    realized_eig_bits: 0.6514043242652702,
    realized_over_predicted: 0.7873460891573774,
    realized_eig_bits_range_fkg_to_frechet: [0, 0.700890842181218],
    realized_eig_bits_excluding_suspected_unmatched_primary: 0.6514023013095627,
    realized_information_bits_from_observed_null: 0.46890270968722636,
    data_tiers: ["illustrative_calibration", "reproduced_archive"],
  },
  {
    radio_mode: "distance_scaled_eirp",
    label: "physical",
    physical: true,
    predicted_eig_bits: 0.014522573794389263,
    realized_eig_bits: 0.0023511276893152517,
    realized_over_predicted: 0.16189469735892134,
    realized_eig_bits_range_fkg_to_frechet: [0, 0.002432265245932863],
    realized_eig_bits_excluding_suspected_unmatched_primary: 0.0023470309957474007,
    realized_information_bits_from_observed_null: 4.2395319305469845e-6,
    data_tiers: ["illustrative_calibration", "reproduced_archive"],
  },
];

describe("number and id formatting", () => {
  it("groups thousands and never shows NaN", () => {
    expect(fmtCount(120176)).toBe("120,176");
    expect(fmtCount(0)).toBe("0");
    expect(fmtCount(null)).toBe("—");
    expect(fmtCount(Number.NaN)).toBe("—");
  });
  it("keeps small EIG values visible instead of rounding them to zero", () => {
    expect(fmtBits(0.0023511276893152517)).toBe("0.002351");
    expect(fmtBits(0.6514043242652702)).toBe("0.6514");
    expect(fmtBits(4.2395319305469845e-6)).toBe("4.24e-6");
    expect(fmtBits(0)).toBe("0");
    expect(fmtBits(undefined)).toBe("—");
  });
  it("shows a Gaia DR3 id only as the exact decimal string the API sent (v0.2.1-R4)", () => {
    const id = "6898272528885931234"; // > 2^53: Number() would change the digits
    expect(gaiaIdText(id)).toBe(id);
    expect(String(Number(id))).not.toBe(id);
    expect(gaiaIdText(Number(id))).toBe("invalid id"); // a JSON number is refused, not shown
    expect(gaiaIdText("0123")).toBe("invalid id");
    expect(gaiaIdText("12345678901234567890")).toBe("invalid id");
    expect(gaiaIdText(null)).toBe("invalid id");
  });
  it("states the cone as FWHM/2 only when it is", () => {
    expect(coneLabel(4.2, 8.4)).toBe("4.2′ (FWHM/2) of the 8.4′ beam");
    expect(coneLabel(4.0, 8.4)).toBe("4′ of the 8.4′ beam");
  });
  it("formats the archive query window", () => {
    expect(utcWindow(["2026-11-27T16:18:22.081409+00:00", "2026-11-27T16:39:02.017439+00:00"])).toBe("2026-11-27 16:18:22 → 16:39:02 UTC");
    expect(utcWindow(["2026-11-27T23:59:00+00:00", "2026-11-28T00:10:00+00:00"])).toBe("2026-11-27 23:59:00 → 2026-11-28 00:10:00 UTC");
    expect(utcWindow(null)).toBe("—");
    expect(utcWindow(["2026-11-27T16:18:22+00:00"])).toBe("—");
  });
  it("formats watts", () => {
    expect(fmtWatts(1.6788040181225658e16)).toBe("1.68e+16 W");
    expect(fmtWatts(null)).toBe("—");
  });
});

describe("distance-cut and de-duplication tables", () => {
  it("accounts for every raw sighting", () => {
    const rows = distanceCutRows(CUTS);
    const raw = rows[0]!;
    const kept = rows.find((r) => r.label === "sightings with a distance")!;
    const dropped = rows.filter((r) => r.kind === "drop").reduce((s, r) => s + r.value, 0);
    expect(raw.value).toBe(259285);
    expect(raw.value - dropped).toBe(kept.value);
    expect(rows.filter((r) => r.kind === "keep").reduce((s, r) => s + r.value, 0)).toBe(kept.value);
    expect(rows.some((r) => r.label.includes("beyond 10,000 pc"))).toBe(true);
  });
  it("lists gaia_id, < 3″ fallback, multi-beam θ_min, absorbed primaries and suspected primaries", () => {
    const rows = dedupRows(MP, DEDUP);
    const value = (needle: string) => rows.find((r) => r.label.includes(needle))?.value;
    expect(value("gaia_id")).toBe(4270);
    expect(value("< 3″ fallback")).toBe(105);
    expect(value("θ_min")).toBe(3764);
    expect(value("absorbed by primaries")).toBe(635);
    expect(value("duplicate Gaia ids vs bl_692_targets.csv")).toBe(0);
    expect(value("N_bg")).toBe(120176);
    const flags = rows.filter((r) => r.kind === "flag");
    expect(flags.map((r) => r.value)).toEqual([49, 47]);
    expect(flags[0]?.label).toContain("suspected_unmatched_primary");
  });
  it("checks that the de-duplication arithmetic closes", () => {
    expect(dedupCloses(MP)).toBe(true);
    expect(dedupCloses({ ...MP, n_dedup_gaia_id: 4271 })).toBe(false);
  });
});

describe("scope contrast with Wlodarczyk-Sroka et al. (2020)", () => {
  it("puts 692 pointings / 120,176 stars beside 1,327 pointings / 288,315 stars", () => {
    const rows = Object.fromEntries(scopeRows(SCOPE).map((r) => [r.field, r]));
    expect(rows["pointings"]).toMatchObject({ retrieved: "692", published: "1,327" });
    expect(rows["stars"]).toMatchObject({ retrieved: "120,176", published: "288,315" });
    expect(rows["Gaia release"]).toMatchObject({ retrieved: "Gaia DR3", published: "Gaia DR2" });
  });
  it("marks a published count that was not verified against the paper", () => {
    const lband = scopeRows(SCOPE).find((r) => r.field === "GBT L-band stars")!;
    expect(lband.published).toBe("251,983");
    expect(lband.caveat).toBe("unverified — v0.2.1b planning brief");
    const verified: BackgroundScope = {
      ...SCOPE,
      published: { ...SCOPE.published, n_stars_gbt_l_band: { value: 1, verified_against_paper: true, source: "x" } },
    };
    expect(scopeRows(verified).find((r) => r.field === "GBT L-band stars")?.caveat).toBeUndefined();
  });
});

describe("realized vs predicted EIG table", () => {
  it("lists the physical mode first with both labels attached", () => {
    const rows = eigRows(EIG);
    expect(rows.map((r) => [r.mode, r.label])).toEqual([
      ["distance_scaled_eirp", "physical"],
      ["flat_probability", "unphysical"],
    ]);
  });
  it("formats predicted, realized, ratio, bounds range and the diagnostics", () => {
    const [phys, flat] = eigRows(EIG);
    expect(phys).toMatchObject({
      predicted: "0.01452",
      realized: "0.002351",
      ratio: "0.16× predicted",
      range: "0 – 0.002432",
      withoutSuspected: "0.002347",
      observedNull: "4.24e-6",
    });
    expect(flat).toMatchObject({ predicted: "0.8273", realized: "0.6514", ratio: "0.79× predicted", range: "0 – 0.7009" });
  });
  it("marks both rows illustrative, so neither is drawn as a verified bound", () => {
    expect(eigRows(EIG).every((r) => r.illustrative)).toBe(true);
    const verified = eigRows([{ ...EIG[1]!, data_tiers: ["reproduced_archive"] }]);
    expect(verified[0]?.illustrative).toBe(false);
  });
  it("reports a smaller-than-predicted value as a null-strength result", () => {
    expect(eigVerdict(EIG)).toBe(
      "Physical mode: the real background field is less informative than predicted (0.002351 vs 0.01452 bits) — a null-strength result either way (≪ 1 bit).",
    );
    expect(eigVerdict([{ ...EIG[1]!, realized_over_predicted: 2 }])).toContain("more informative than predicted");
    expect(eigVerdict([EIG[0]!])).toBe("no physical-mode comparison available");
    expect(eigRows([{ ...EIG[1]!, realized_over_predicted: null }])[0]?.ratio).toBe("—");
  });
});

describe("off-axis attenuation and pointing-centre caveat", () => {
  const att = {
    median_eirp_min_ratio_background_over_primary: 16038.4,
    decomposition_of_the_median_ratio: { distance_term: 11408.2, beam_term: 1.4345, note: "" },
  } as BackgroundAttenuation;
  it("summarises why background EIRP_min is orders of magnitude higher", () => {
    expect(attenuationHeadline(att)).toBe(
      "median EIRP_min is ×1.6e+4 the on-axis primaries’ (distance term ×1.1e+4, beam term ×1.43)",
    );
  });
  const audit: PointingCentreAudit = {
    n_primaries: 692,
    n_with_gaia_id: 522,
    n_with_gaia_id_sighted_in_own_cone: 506,
    with_gaia_id_not_sighted_in_own_cone: [],
    n_without_gaia_id: 170,
    n_no_counterpart_in_own_cone: 68,
    densest_band_of_unfound_primaries: {
      declination_band_deg: [10, 11],
      mirror_band_deg: [-11, -10],
      n_targets: 16,
      n_targets_in_mirror_band: 0,
      n_no_counterpart: 14,
      targets_no_counterpart: ["HIP 5364"],
      n_background_stars_in_those_pointings: 1055,
      fraction_of_background_stars: 1055 / 120176,
      reading: "consistent with a declination sign error",
    },
    policy: "reported, not corrected",
  };
  it("states the flagged declination band with its counts", () => {
    expect(auditCaveat(audit)).toBe(
      "14 of the 16 primaries catalogued at Dec +10° to +11° have no Gaia counterpart in their own cone (0 targets in the mirror band): 1,055 background stars (0.88% of N_bg) may come from the wrong field.",
    );
  });
  it("is silent when nothing is flagged", () => {
    expect(auditCaveat(null)).toBeNull();
    expect(auditCaveat({ ...audit, densest_band_of_unfound_primaries: null })).toBeNull();
    expect(
      auditCaveat({ ...audit, densest_band_of_unfound_primaries: { ...audit.densest_band_of_unfound_primaries!, n_no_counterpart: 0 } }),
    ).toBeNull();
  });
});
