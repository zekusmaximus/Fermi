import { describe, expect, it } from "vitest";
import type { MergeProvenance } from "../api";
import {
  CORRELATION_CAVEAT,
  DEFAULT_MERGE_POLICY,
  EARTH_UNREACHED_HELP,
  MERGE_POLICY_CONTROL_LABEL,
  MERGE_POLICY_HELP,
  MERGE_POLICY_LABELS,
  MERGE_POLICY_OPTIONS,
  earthUnreachedNote,
  flatBackgroundGuardActive,
  isEarthTerm,
  mergePolicyCaption,
  mergePolicyRequest,
  mergeProvenanceRows,
  noteRulingTag,
} from "./merge";

const MP: MergeProvenance = {
  pointing_group_id: "bl_gbt_lband_2017",
  policy: "union_targets_by_pointing",
  input_surveys: ["bl_enriquez_2017_692", "bl_gbt_lband_background_gaia_dr3"],
  input_sha256s: ["a".repeat(64), "b".repeat(64)],
  n_raw_targets: 125026,
  n_merged_targets: 120639,
  n_dedup_gaia_id: 4281,
  n_dedup_angular_fallback: 106,
  n_multi_beam_resolved: 3764,
  angular_fallback_arcsec: 3,
  n_raw_by_survey: { bl_enriquez_2017_692: 692, bl_gbt_lband_background_gaia_dr3: 124334 },
  n_merged_by_role: { primary: 692, background: 119947 },
  n_background_absorbed_by_primary: 647,
  n_angular_links_refused: 8,
  n_duplicate_gaia_ids_post_merge: 0,
};

describe("pointing merge policy control", () => {
  it("offers exactly the two policies, default first, and never keep_first", () => {
    expect(MERGE_POLICY_OPTIONS).toEqual(["reject_duplicate_pointing_group", "union_targets_by_pointing"]);
    expect(DEFAULT_MERGE_POLICY).toBe("reject_duplicate_pointing_group");
    expect(MERGE_POLICY_OPTIONS as readonly string[]).not.toContain("keep_first");
  });
  it("labels options in plain English, not as raw enums", () => {
    expect(MERGE_POLICY_LABELS.reject_duplicate_pointing_group).toBe("692 primaries only");
    expect(MERGE_POLICY_LABELS.union_targets_by_pointing).toBe("692 primaries + Gaia DR3 background stars");
    for (const label of Object.values(MERGE_POLICY_LABELS)) expect(label).not.toMatch(/_/);
    expect(MERGE_POLICY_CONTROL_LABEL).not.toMatch(/_/);
  });
  it("sends the selected policy in the compare body", () => {
    expect(mergePolicyRequest("union_targets_by_pointing")).toEqual({ pointing_merge_policy: "union_targets_by_pointing" });
    expect(mergePolicyRequest(DEFAULT_MERGE_POLICY)).toEqual({ pointing_merge_policy: "reject_duplicate_pointing_group" });
  });
  it("explains the trade-off, the invariant and the CLI flag in its ? popover", () => {
    expect(MERGE_POLICY_HELP.tradeoff).toContain("4πd²F_min / B(θ)");
    expect(MERGE_POLICY_HELP.invariant).toContain("Gaia DR3 source_id");
    expect(MERGE_POLICY_HELP.invariant).toContain("keep_first is forbidden");
    expect(MERGE_POLICY_HELP.invariant).toContain("Harris–FKG");
    expect(MERGE_POLICY_HELP.invariant).toContain("v0.2.1-R2");
    expect(MERGE_POLICY_HELP.cli).toEqual([
      "--pointing-merge-policy reject_duplicate_pointing_group",
      "--pointing-merge-policy union_targets_by_pointing",
    ]);
  });
});

describe("dynamic merge-policy caption", () => {
  const base = { radioOn: true, nBackground: 119947, heldOutSurveys: [] as string[] };
  it("says the default keeps the verified headline and names what is held out", () => {
    const c = mergePolicyCaption({
      ...base,
      policy: "reject_duplicate_pointing_group",
      radioMode: "flat_probability",
      heldOutSurveys: ["bl_gbt_lband_background_gaia_dr3"],
    });
    expect(c).toContain("Only the 692 on-axis targets");
    expect(c).toContain("bl_gbt_lband_background_gaia_dr3");
    expect(c).toContain("Verified headline: unchanged");
  });
  it("flags ruling v0.2.1-R2 when a flat probability meets the merged background", () => {
    const c = mergePolicyCaption({ ...base, policy: "union_targets_by_pointing", radioMode: "flat_probability" });
    expect(c).toContain("Adds 119,947 off-axis background stars");
    expect(c).toContain("d² / B(θ)");
    expect(c).toContain("HELD OUT of the verified headline (ruling v0.2.1-R2)");
  });
  it("states the physical mode leaves the verified headline unchanged", () => {
    const c = mergePolicyCaption({ ...base, policy: "union_targets_by_pointing", radioMode: "distance_scaled_eirp" });
    expect(c).toContain("EIRP_min = 4πd²F_min / B(θ)");
    expect(c).toContain("verified headline is unchanged");
    expect(c).not.toContain("v0.2.1-R2");
  });
  it("changes with every option and never invents a star count", () => {
    const captions = new Set(
      MERGE_POLICY_OPTIONS.flatMap((policy) =>
        (["flat_probability", "distance_scaled_eirp"] as const).map((radioMode) => mergePolicyCaption({ ...base, policy, radioMode })),
      ),
    );
    expect(captions.size).toBe(3); // the default caption does not depend on the radio mode
    const loading = mergePolicyCaption({ policy: "union_targets_by_pointing", radioMode: "flat_probability", radioOn: true, nBackground: null });
    expect(loading).toContain("Adds the off-axis background stars");
    expect(loading).not.toMatch(/\d{3},\d{3}/);
  });
  it("says the control is inert when the radio channel is off", () => {
    expect(mergePolicyCaption({ ...base, policy: "union_targets_by_pointing", radioMode: "flat_probability", radioOn: false })).toBe(
      "Radio channel is off: this choice changes nothing in the current run.",
    );
  });
  it("activates the R2 guard only for union + flat + radio", () => {
    expect(flatBackgroundGuardActive("union_targets_by_pointing", "flat_probability", true)).toBe(true);
    expect(flatBackgroundGuardActive("union_targets_by_pointing", "distance_scaled_eirp", true)).toBe(false);
    expect(flatBackgroundGuardActive("reject_duplicate_pointing_group", "flat_probability", true)).toBe(false);
    expect(flatBackgroundGuardActive("union_targets_by_pointing", "flat_probability", false)).toBe(false);
  });
});

describe("merge provenance read-out", () => {
  it("lists the receipt counters and flags the suspected primaries", () => {
    const rows = mergeProvenanceRows(MP, 47);
    const by = Object.fromEntries(rows.map((r) => [r.label, r]));
    expect(by["raw sightings (primaries + cone stars)"]?.value).toBe(125026);
    expect(by["stars after the merge"]?.value).toBe(120639);
    expect(by["background stars (N_bg)"]?.value).toBe(119947);
    expect(by["removed: same Gaia DR3 source_id"]?.value).toBe(4281);
    expect(by["removed: < 3″ fallback (no Gaia id)"]?.value).toBe(106);
    expect(by["multi-beam stars kept at minimum θ"]?.value).toBe(3764);
    expect(by["duplicate Gaia ids after the merge"]?.value).toBe(0);
    const flagged = rows.filter((r) => r.flag);
    expect(flagged).toHaveLength(1);
    expect(flagged[0]).toMatchObject({ value: 47 });
    expect(flagged[0]?.label).toContain("v0.2.1-R3");
  });
  it("is empty without a merge and omits the suspected row when unknown", () => {
    expect(mergeProvenanceRows(null)).toEqual([]);
    expect(mergeProvenanceRows(MP).some((r) => r.flag)).toBe(false);
  });
  it("states the Harris–FKG / Fréchet caveat without claiming a sign", () => {
    expect(CORRELATION_CAVEAT).toContain("Harris–FKG a LOWER bound");
    expect(CORRELATION_CAVEAT).toContain("Fréchet upper bound");
    expect(CORRELATION_CAVEAT).toContain("sign of the effect on a Bayes factor is not fixed");
  });
  it("tags an exclusion note with its ruling, if any", () => {
    expect(noteRulingTag({ rulings: ["v0.2.1-R2"] })).toBe("v0.2.1-R2");
    expect(noteRulingTag({ rulings: [] })).toBeNull();
    expect(noteRulingTag({})).toBeNull();
  });
});

describe("earth_unreached assumption note (v0.2.0-R4)", () => {
  it("applies only to the earth_unreached row", () => {
    expect(isEarthTerm("earth_unreached")).toBe(true);
    expect(isEarthTerm("radio")).toBe(false);
  });
  it("names R_gal, the density profile and the T treatment under the finite disk", () => {
    const fixed = earthUnreachedNote({ geometry: "finite_disk_3d", diskProfile: "exponential", marginalized: false });
    expect(fixed).toContain("Verified model term — no survey data");
    expect(fixed).toContain("R_gal = 15 kpc");
    expect(fixed).toContain("exponential disk density profile");
    expect(fixed).toContain("fixed lookback T");
    const marg = earthUnreachedNote({ geometry: "finite_disk_3d", diskProfile: "uniform", marginalized: true });
    expect(marg).toContain("uniform-slab disk density profile");
    expect(marg).toContain("lookback T prior (log-uniform, 50 kyr–10 Myr)");
  });
  it("says the local disks have no R_gal edge and points at the finite disk", () => {
    const thin = earthUnreachedNote({ geometry: "thin_disk_2d", diskProfile: "uniform", marginalized: false });
    expect(thin).toContain("infinite thin 2D disk (no R_gal edge)");
    expect(thin).toContain("finite_disk_3d");
    const thick = earthUnreachedNote({ geometry: "thick_disk_3d", diskProfile: "uniform", marginalized: true });
    expect(thick).toContain("infinite thick 3D slab (no R_gal edge)");
    expect(thick).toContain("lookback T prior");
  });
  it("updates whenever geometry, profile or lookback treatment changes", () => {
    const notes = new Set<string>();
    for (const geometry of ["thin_disk_2d", "thick_disk_3d", "finite_disk_3d"] as const)
      for (const marginalized of [false, true]) notes.add(earthUnreachedNote({ geometry, diskProfile: "uniform", marginalized }));
    notes.add(earthUnreachedNote({ geometry: "finite_disk_3d", diskProfile: "exponential", marginalized: false }));
    expect(notes.size).toBe(7);
    for (const n of notes) expect(n.split("\n")).toHaveLength(1); // one line
  });
  it("explains in its ? popover why a model term is verified, with CLI flags", () => {
    expect(EARTH_UNREACHED_HELP.tradeoff).toContain("no illustrative calibration can leak into it");
    expect(EARTH_UNREACHED_HELP.tradeoff).toContain("R_gal");
    expect(EARTH_UNREACHED_HELP.invariant).toContain("v0.2.0-R4");
    expect(EARTH_UNREACHED_HELP.cli).toContain("--marginalize-lookback");
    expect(EARTH_UNREACHED_HELP.cli.join(" ")).toContain("--geometry finite_disk_3d");
  });
});
