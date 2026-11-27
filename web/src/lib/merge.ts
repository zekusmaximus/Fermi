// Pure helpers for the Macro Desk pointing-merge-policy control and the earth_unreached
// assumption note (unit-tested in merge.test.ts). No I/O, no DOM.
//
// DESIGN.md Sec. 2: every control ships a plain-English label (the exact API / CLI enum stays
// visible in monospace), a one-line caption that changes with the active option and says
// what physically changes and whether the verified headline is affected, and a `?` popover
// with the trade-off, the invariant or citation behind it, and the CLI flag.
import type { DiskProfile, ExclusionNote, Geometry, MergeProvenance, PointingMergePolicy, RadioMode } from "../api";

export const MERGE_POLICY_OPTIONS: readonly PointingMergePolicy[] = [
  "reject_duplicate_pointing_group",
  "union_targets_by_pointing",
];

export const DEFAULT_MERGE_POLICY: PointingMergePolicy = "reject_duplicate_pointing_group";

/** Plain-English option labels; the enum itself is rendered beneath in monospace. */
export const MERGE_POLICY_LABELS: Record<PointingMergePolicy, string> = {
  reject_duplicate_pointing_group: "692 primaries only",
  union_targets_by_pointing: "692 primaries + Gaia DR3 background stars",
};

export const MERGE_POLICY_CONTROL_LABEL = "Radio targets in the BL beams";

export interface HelpContent {
  title: string;
  /** What physically changes between the options. */
  tradeoff: string;
  /** The invariant, ruling or citation that constrains the control. */
  invariant: string;
  /** Exact CLI equivalent(s). */
  cli: string[];
}

export const MERGE_POLICY_HELP: HelpContent = {
  title: "Which stars does the radio null constrain?",
  tradeoff:
    "Each GBT L-band pointing also covered every star inside its 8.4′ beam. “692 primaries only” conditions on the on-axis targets alone. " +
    "“+ Gaia DR3 background stars” also folds in the promoted background field: each extra star enters once, at its own distance and beam offset, " +
    "so its minimum detectable EIRP is 4πd²F_min / B(θ) — far higher than for the nearby on-axis primaries.",
  invariant:
    "Strict non-double-counting (SPEC.md Sec. 2, invariant 4): stars are merged by Gaia DR3 source_id (3″ fallback only when an id is missing) and the " +
    "minimum-offset sighting is kept; keep_first is forbidden. Background shells use the independent conditional-reach product, a Harris–FKG lower bound " +
    "on P(no detection); the Fréchet upper bound is shown beside it. Ruling v0.2.1-R2: in flat_probability mode the merged radio term never enters the verified headline.",
  cli: ["--pointing-merge-policy reject_duplicate_pointing_group", "--pointing-merge-policy union_targets_by_pointing"],
};

/** Only the merge policy part of the /api/compare body. */
export function mergePolicyRequest(policy: PointingMergePolicy): { pointing_merge_policy: PointingMergePolicy } {
  return { pointing_merge_policy: policy };
}

/** Ruling v0.2.1-R2 applies: merged background + sensitivity-blind detection probability. */
export function flatBackgroundGuardActive(policy: PointingMergePolicy, radioMode: RadioMode, radioOn: boolean): boolean {
  return radioOn && policy === "union_targets_by_pointing" && radioMode === "flat_probability";
}

const count = (n: number | null | undefined): string =>
  typeof n === "number" && Number.isFinite(n) ? n.toLocaleString("en-US") : "the";

export interface MergeCaptionInput {
  policy: PointingMergePolicy;
  radioMode: RadioMode;
  /** Whether the radio channel is selected at all. */
  radioOn: boolean;
  /** N_bg from the receipt of a merged run (null while loading or when held out). */
  nBackground?: number | null;
  /** Declared extensions the server reported as held out under the default policy. */
  heldOutSurveys?: readonly string[];
}

/**
 * One-line dynamic caption under the control: what physically changes and whether the
 * verified headline is affected. Star counts come from the receipt, never from memory.
 */
export function mergePolicyCaption(c: MergeCaptionInput): string {
  if (!c.radioOn) return "Radio channel is off: this choice changes nothing in the current run.";
  if (c.policy === "reject_duplicate_pointing_group") {
    const held = c.heldOutSurveys ?? [];
    return held.length > 0
      ? `Only the 692 on-axis targets enter the radio term; the declared background extension (${held.join(", ")}) is held out of this run. Verified headline: unchanged.`
      : "Only the 692 on-axis targets enter the radio term. Verified headline: unchanged.";
  }
  const n = count(c.nBackground);
  if (c.radioMode === "flat_probability") {
    return (
      `Adds ${n} off-axis background stars — but a flat detection probability ignores their d² / B(θ) attenuation, ` +
      "so the merged radio term is illustrative and HELD OUT of the verified headline (ruling v0.2.1-R2)."
    );
  }
  return (
    `Adds ${n} off-axis background stars at EIRP_min = 4πd²F_min / B(θ). Radio is already held out of the verified headline in this mode ` +
    "(illustrative transmitter luminosity function), so the verified headline is unchanged."
  );
}

/** Harris–FKG / Fréchet caveat shown wherever merged background numbers are displayed. */
export const CORRELATION_CAVEAT =
  "Primaries keep the exact correlated Monte Carlo. Background shells multiply on the independent product of conditional reach probabilities: " +
  "by Harris–FKG a LOWER bound on P(no detection | H), so it can only overstate how strongly the null disfavours a hypothesis. " +
  "The Fréchet upper bound is valid under any dependence; the exact value lies between the two, and the sign of the effect on a Bayes factor is not fixed by the bounds.";

export interface MergeRow {
  label: string;
  value: number | null;
  /** True for the ruling v0.2.1-R3 counter, which is reported in amber, never hidden. */
  flag?: boolean;
}

/** Counters of a union merge for the Macro Desk provenance panel (receipt values only). */
export function mergeProvenanceRows(
  mp: MergeProvenance | null | undefined,
  suspectedStars?: number | null,
): MergeRow[] {
  if (!mp) return [];
  const rows: MergeRow[] = [
    { label: "raw sightings (primaries + cone stars)", value: mp.n_raw_targets },
    { label: "stars after the merge", value: mp.n_merged_targets },
    { label: "primary targets kept on-axis", value: mp.n_merged_by_role["primary"] ?? null },
    { label: "background stars (N_bg)", value: mp.n_merged_by_role["background"] ?? null },
    { label: "removed: same Gaia DR3 source_id", value: mp.n_dedup_gaia_id },
    { label: `removed: < ${mp.angular_fallback_arcsec}″ fallback (no Gaia id)`, value: mp.n_dedup_angular_fallback },
    { label: "multi-beam stars kept at minimum θ", value: mp.n_multi_beam_resolved },
    { label: "background sightings absorbed by a primary", value: mp.n_background_absorbed_by_primary },
    { label: "duplicate Gaia ids after the merge", value: mp.n_duplicate_gaia_ids_post_merge },
  ];
  if (typeof suspectedStars === "number") {
    rows.push({ label: "suspected unmatched primaries (kept in shells, ruling v0.2.1-R3)", value: suspectedStars, flag: true });
  }
  return rows;
}

/** Short tag for an exclusion note: the planning ruling(s) behind it, if any. */
export function noteRulingTag(note: Pick<ExclusionNote, "rulings">): string | null {
  const r = note.rulings ?? [];
  return r.length > 0 ? r.join(", ") : null;
}

// ---------------------------------------------------- earth_unreached assumption note
export const EARTH_TERM = "earth_unreached";

export interface EarthNoteInput {
  geometry: Geometry;
  diskProfile: DiskProfile;
  /** Whether the receipt says the lookback window was marginalised. */
  marginalized: boolean;
}

const DISK_PROFILE_TEXT: Record<DiskProfile, string> = {
  uniform: "uniform-slab",
  exponential: "exponential",
};

/**
 * Carry-over v0.2.0-R4: `earth_unreached` is a verified MODEL term (no illustrative survey
 * data enters it), but it is only as good as the Galactic-structure and lookback assumptions
 * behind it. One line, specific to the active geometry and lookback treatment.
 */
export function earthUnreachedNote(c: EarthNoteInput): string {
  const lookback = c.marginalized
    ? "the lookback T prior (log-uniform, 50 kyr–10 Myr)"
    : "the fixed lookback T (registry default)";
  if (c.geometry === "finite_disk_3d") {
    return `Verified model term — no survey data, but it depends on Galactic structure: R_gal = 15 kpc, the ${DISK_PROFILE_TEXT[c.diskProfile]} disk density profile and ${lookback}.`;
  }
  const disk = c.geometry === "thin_disk_2d" ? "an infinite thin 2D disk (no R_gal edge)" : "an infinite thick 3D slab (no R_gal edge)";
  return `Verified model term — no survey data, but it assumes ${disk} and ${lookback}; switch to finite_disk_3d to see its dependence on R_gal and the density profile.`;
}

export const EARTH_UNREACHED_HELP: HelpContent = {
  title: "Why is a model term in the verified headline?",
  tradeoff:
    "earth_unreached is P(no settlement front has reached Earth | H). It uses no survey at all, so no illustrative calibration can leak into it — " +
    "that is why it counts as verified. But it is computed from the expansion model, so it moves with the Galactic structure you assume: the disk radius R_gal, " +
    "the density profile (uniform slab vs exponential disk) and how the lookback window T is treated (fixed, or marginalised over its prior).",
  invariant:
    "Ruling v0.2.0-R4: keep the label “verified (model term)” and state its assumptions at the point of use. In an infinite 2D disk the term cancels between the two benchmark hypotheses " +
    "(the Γ·v² scaling symmetry, SPEC.md Sec. 2 invariant 5); a finite disk or a T prior breaks that symmetry, which is where its Bayes-factor contribution comes from.",
  cli: ["--geometry finite_disk_3d --disk-profile exponential", "--marginalize-lookback", "--no-earth-condition"],
};

/** True for the Bayes-factor row that gets the assumption note. */
export function isEarthTerm(term: string): boolean {
  return term === EARTH_TERM;
}
