// Typed client for the offline FastAPI desk. Same-origin relative URLs only
// (/api/...): the production bundle is served by FastAPI; `npm run dev` proxies /api.

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  if (!path.startsWith("/api/")) throw new Error("only same-origin /api paths are allowed");
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail ?? body ?? res.statusText;
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

export const get = <T,>(path: string) => call<T>(path);
export const post = <T,>(path: string, body: unknown) =>
  call<T>(path, { method: "POST", body: JSON.stringify(body) });

// ---------------------------------------------------------------- payloads
export interface Health {
  status: string;
  git_sha: string;
  git_head: string;
  validation_passed: boolean;
  validation_checks_passed: number;
  validation_checks_total: number;
  validation_failures: string[];
  catalog_sha256: string | null;
  wise_calibration_sha256: string | null;
  /** v0.1.2: live control-sample provenance, read from the verified calibration files. */
  n_controls: number | null;
  n_calibration: number | null;
  n_withheld: number | null;
  calibration_data_tier: string | null;
  fits_gate2_sha256: string | null;
  fits_gate2_data_tier: string | null;
  /** v0.2.1b: promoted Gaia DR3 background-field extension (null until a survey pins one). */
  background_survey_id: string | null;
  background_summary_sha256: string | null;
  background_summary_data_tier: string | null;
  background_summary_code_git_sha: string | null;
  n_background_stars: number | null;
  n_suspected_unmatched_primary: number | null;
  receipt_schema_version: number;
  read_only: boolean;
  version: string;
}

export interface MonocultureAudit {
  required_compliance_fraction: number;
  timescale_gyr: number;
  defector_vulnerability: string;
}

export interface HypothesisDef {
  id: string;
  name: string;
  tier: number;
  summary: string;
  quarantine_reason: string | null;
  monoculture_audit: MonocultureAudit | null;
  is_quarantined: boolean;
  citations: string[];
}

export interface Registry {
  parameters: { id: string; symbol: string; unit: string; default: number | null }[];
  hypotheses: { tier_1: HypothesisDef[]; tier_2: HypothesisDef[]; tier_3: HypothesisDef[] };
  surveys: {
    id: string;
    name: string;
    channel: string;
    data_tier: string;
    n_targets: number;
    /** v0.2.1: set on a declared pointing extension (e.g. the Gaia DR3 background field). */
    parent_pointing_survey_id?: string | null;
  }[];
}

export type Geometry = "thin_disk_2d" | "thick_disk_3d" | "finite_disk_3d";
export type DiskProfile = "uniform" | "exponential";
export type RadioMode = "flat_probability" | "distance_scaled_eirp";
export type Channel = "radio" | "infrared" | "solar_system";
/** v0.2.1: how surveys sharing a pointing group are combined ("keep_first" is refused). */
export type PointingMergePolicy = "reject_duplicate_pointing_group" | "union_targets_by_pointing";

export interface CompareRequest {
  geometry_mode: Geometry;
  radio_mode: RadioMode;
  condition_on_earth_unreached: boolean;
  channels: Channel[];
  include_illustrative: boolean;
  seed: number;
  /** v0.2.0: applied to finite_disk_3d only (server default "uniform"). */
  disk_profile: DiskProfile;
  /** v0.2.0: integrate the lookback window T over its log-uniform prior (default false). */
  marginalize_lookback: boolean;
  /** v0.2.1: default holds declared background extensions out of the run. */
  pointing_merge_policy: PointingMergePolicy;
}

/** v0.2.1: star-level audit of one union_targets_by_pointing merge. */
export interface MergeProvenance {
  pointing_group_id: string;
  policy: string;
  input_surveys: string[];
  input_sha256s: string[];
  n_raw_targets: number;
  n_merged_targets: number;
  n_dedup_gaia_id: number;
  n_dedup_angular_fallback: number;
  n_multi_beam_resolved: number;
  angular_fallback_arcsec: number;
  n_raw_by_survey: Record<string, number>;
  n_merged_by_role: Record<string, number>;
  n_background_absorbed_by_primary: number;
  n_angular_links_refused: number;
  n_duplicate_gaia_ids_post_merge: number;
}

export interface BackgroundDistanceCuts {
  d_max_pc: number;
  inverse_parallax_min_snr: number;
  bailer_jones_min_snr_exclusive: number;
  n_cone_sightings_raw: number;
  n_dropped_invalid_parallax: number;
  n_dropped_no_bailer_jones: number;
  n_dropped_beyond_d_max: number;
  n_dropped_outside_cone_recheck: number;
  n_sightings_with_distance: number;
  n_sightings_inverse_parallax: number;
  n_sightings_bailer_jones_geo: number;
  n_background_stars: number;
  n_background_inverse_parallax: number;
  n_background_bailer_jones_geo: number;
}

/** De-duplication counters, including the reported (never removed) suspected primaries. */
export interface BackgroundDedup {
  n_primary_targets: number;
  n_primary_with_gaia_id: number;
  n_primary_without_gaia_id: number;
  n_background_sightings_matched_to_primary_gaia_id: number;
  n_background_sightings_matched_to_primary_angular: number;
  n_duplicate_gaia_ids_vs_primary: number;
  n_duplicate_gaia_ids_within_background: number;
  angular_fallback_arcsec: number;
  angular_match_epoch: string;
  n_suspected_unmatched_primary: number;
  n_suspected_unmatched_primary_stars: number;
  n_primaries_with_a_suspected_counterpart: number;
  n_suspected_primaries_that_absorbed_a_sighting: number;
  n_suspected_primaries_that_absorbed_nothing: number;
  suspected_unmatched_primary_policy: string;
}

export interface PublishedCount {
  value: number;
  verified_against_paper: boolean;
  source: string;
}

export interface BackgroundScope {
  retrieved: {
    survey_id: string;
    data_tier: string;
    pointings: string;
    n_pointings: number;
    n_background_stars: number;
    gaia_release: string;
    distances: string;
    selection: string;
    source_service: string;
  };
  published: {
    citation: string;
    data_tier: string;
    pointings: string;
    n_pointings: number;
    n_stars_total: number;
    n_stars_gbt_l_plus_s_band: PublishedCount;
    n_stars_gbt_l_band: PublishedCount;
    gaia_release: string;
    distances: string;
    selection: string;
  };
  comparable: boolean;
  statement: string;
}

/** Verified provenance of the promoted background summary (reproduced_archive). */
export interface BackgroundProvenance {
  id: string;
  survey_id: string;
  parent_survey_id: string;
  path: string;
  file_sha256: string;
  content_sha256: string;
  data_tier: string;
  code_git_sha: string;
  fetched_at_utc: string;
  query_window_utc: string[];
  assembled_at_utc: string | null;
  cone_cache_manifest_sha256: string | null;
  cone_cache_code_git_shas: string[];
  source_service: string;
  n_pointings: number;
  n_background_stars: number;
  n_cells: number;
  beam_fwhm_arcmin: number;
  cone_radius_arcmin: number;
  distance_cuts: BackgroundDistanceCuts;
  merge_provenance: MergeProvenance;
  dedup_audit: BackgroundDedup;
  scope: BackgroundScope;
  methodology_note: string;
}

export interface HeldOutSurvey {
  survey_id: string;
  pointing_group_id: string;
  parent_pointing_survey_id: string;
  reason: string;
}

/** v0.2.0: prior and grid of a lookback-marginalised run (absent on fixed-T receipts). */
export interface LookbackMarginalization {
  parameter: string;
  prior: { type: string; declared_bounds_gyr: [number, number]; effective_bounds_gyr: [number, number]; log_floor: number | null };
  quadrature: { rule: string; n_points: number; lookback_gyr: number[]; weights: number[] };
}

export interface Component {
  log10_p_data_given_h: number | null;
  log10_lr_vs_null: number | null;
  data_tiers: string[];
  illustrative_inputs: string[];
}

export interface ExclusionNote {
  channel: string;
  held_out_of_reported_headline: boolean;
  illustrative_inputs?: string[];
  reasons: string[];
  data_tiers: string[];
  log10_bf_term: number | null;
  /** v0.2.1b: planning ruling(s) behind the exclusion (e.g. ["v0.2.1-R2"]); merged runs only. */
  rulings?: string[];
  /** v0.2.1b: where a merged radio term comes from; merged runs only. */
  merged_background?: {
    survey_id: string;
    data_tier: string;
    summary_sha256: string;
    n_background_stars: number;
    n_primary_targets: number;
    log10_bf_primary_targets_exact_mc?: number | null;
    log10_bf_background_shells_fkg_product?: number | null;
  };
}

/** v0.2.1: zero-detection term of the merged background shells (merged runs only). */
export interface RadioBackgroundTerm {
  survey_id: string;
  data_tier: string;
  n_stars: number;
  n_cells: number;
  expected_reached: number | null;
  expected_detections: number | null;
  log10_p_zero_independent_fkg_lower: number | null;
  p_zero_independent_fkg_lower: number | null;
  p_zero_markov_lower: number | null;
  p_zero_frechet_upper: number | null;
}

export interface ScoredHypothesis {
  hypothesis_id: string;
  tier: number;
  channels_evaluated: string[];
  expansion: {
    p_earth_reached: number | null;
    local_approximation_valid: boolean;
    max_bubble_radius_kpc: number | null;
    t_valid_kyr?: number | null;
    /** v0.2.0: "valid_finite_galaxy" under finite_disk_3d; "ok" | "VIOLATED" when marginalised. */
    local_approx?: string | null;
    finite_disk?: { disk_profile: string; galactic_disk_radius_kpc: number; density_normalisation: string } | null;
  } | null;
  radio?: {
    expected_reached: number | null;
    expected_reached_unconditional: number | null;
    p_zero_exact_mc: number | null;
    p_zero_exact_mc_se: number | null;
    background?: RadioBackgroundTerm;
    n_targets_merged?: number;
  };
  infrared?: { p_star_reached?: number | null };
  components: Record<string, Component>;
  own_joint: { terms: string[]; log10_lr_vs_null: number | null };
}

export interface HeadlineBlock {
  numerator: string;
  denominator: string;
  terms: string[];
  log10_bf_by_term: Record<string, number | null>;
  term_data_tiers: Record<string, string[]>;
  log10_bf: number | null;
  verified: { log10_bf: number | null; terms: string[]; log10_bf_by_term: Record<string, number | null> };
  headline: {
    policy: string;
    log10_bf: number | null;
    terms: string[];
    channels: string[];
    data_tiers: string[];
    warning: string | null;
    headline_exclusion_notes: ExclusionNote[];
  };
}

export interface CompareResponse {
  cached: boolean;
  headline_policy: string;
  headline_log10_bf: number | null;
  headline_exclusion_notes: ExclusionNote[];
  receipt: {
    git_sha: string;
    rng_seed: number;
    data_tiers_used: string[];
    geometry_mode?: string;
    disk_profile?: string;
    lookback_marginalization?: LookbackMarginalization;
    /** Receipt schema v5 (v0.2.1). */
    pointing_merge_policy: string;
    merge_provenance: MergeProvenance | null;
    background_field: BackgroundProvenance | null;
    held_out_extension_surveys: HeldOutSurvey[];
    results: {
      hypotheses: Record<string, ScoredHypothesis>;
      tier3_quarantine: {
        hypothesis_id: string;
        quarantine_reason: string;
        monoculture_audit: MonocultureAudit;
        empirical_score: null;
      }[];
      headline: HeadlineBlock;
    };
  };
}

export interface Target {
  target_name: string;
  ra_deg: number;
  dec_deg: number;
  dist_pc: number;
  dist_source: string;
  dist_legacy_pc: number | null;
  eirp_min_w: number;
  eirp_revision_factor: number | null;
}

export interface BeamPair {
  pair_id: string;
  a: string;
  b: string;
  separation_arcmin: number;
  separation_over_beam_fwhm: number;
  dist_a_pc: number;
  dist_b_pc: number;
  counted_once_each: boolean;
}

export interface RadioDrill {
  survey_id: string;
  data_tier: string;
  catalog_sha256: string;
  f_min_w_m2: number;
  beam_fwhm_arcmin: number;
  n_targets: number;
  targets: Target[];
  hip1692_audit: {
    status: string;
    summary: string;
    target_name: string;
    canonical_label: string;
    historical_anchor_label: string;
    threshold: number;
    distances: Record<
      string,
      { distance_pc: number; data_tier: string; eirp_min_w: number; eirp_factor_vs_legacy: number; exceeds_threshold: boolean }
    >;
  };
  hip1692_citations: string[];
  beam_sharing_pairs: BeamPair[];
  beam_sharing_interpretation: string;
  /** v0.2.1b: promoted Gaia DR3 background-field extension. */
  background_field: BackgroundFieldBlock;
}

/** One suspected unmatched primary (ruling v0.2.1-R3: listed, never removed). */
export interface SuspectedPrimary {
  primary_target_name: string;
  /** Always a decimal STRING (ruling v0.2.1-R4): 64-bit ids exceed 2^53. Never Number() it. */
  gaia_dr3_source_id: string;
  separation_arcsec: number;
  distance_pc: number;
  primary_distance_pc: number;
  pointing_index: number;
  theta_arcmin: number;
}

export type Quantiles = Record<"p05" | "p25" | "p50" | "p75" | "p95", number>;

export interface EigComparisonRow {
  radio_mode: RadioMode;
  label: string;
  physical: boolean;
  predicted_eig_bits: number;
  realized_eig_bits: number;
  realized_over_predicted: number | null;
  realized_eig_bits_range_fkg_to_frechet: [number, number];
  realized_eig_bits_excluding_suspected_unmatched_primary: number;
  realized_information_bits_from_observed_null: number;
  data_tiers: string[];
}

export interface BackgroundEig {
  job_id: string;
  summary_sha256: string;
  n_background_stars: number;
  definition: { prior: { type: string; over: string[] }; geometry: string; outcome: string };
  predicted: { data_tier: string; population: string; n_stars: number };
  realized_excluding_suspected_unmatched_primary: { n_stars_removed: number; n_stars: number; policy: string };
  comparison: EigComparisonRow[];
  scope: BackgroundScope;
  statement: string;
  tier_note: string;
}

export interface BoundsRow {
  radio_mode: RadioMode;
  radio_mode_label: string;
  hypothesis_id: string;
  expected_reached: number;
  expected_detections: number;
  p_zero_markov_lower: number;
  p_zero_fkg_lower: number;
  p_zero_frechet_upper: number;
  data_tiers: string[];
}

export interface PointingCentreAudit {
  n_primaries: number;
  n_with_gaia_id: number;
  n_with_gaia_id_sighted_in_own_cone: number;
  with_gaia_id_not_sighted_in_own_cone: string[];
  n_without_gaia_id: number;
  n_no_counterpart_in_own_cone: number;
  densest_band_of_unfound_primaries: {
    declination_band_deg: [number, number];
    mirror_band_deg: [number, number];
    n_targets: number;
    n_targets_in_mirror_band: number;
    n_no_counterpart: number;
    targets_no_counterpart: string[];
    n_background_stars_in_those_pointings: number;
    fraction_of_background_stars: number;
    reading: string;
  } | null;
  policy: string;
}

export interface BackgroundAttenuation {
  formula: string;
  f_min_w_m2: number;
  beam_fwhm_arcmin: number;
  cone_radius_arcmin: number;
  beam_response_at_cone_edge: number;
  validation_sample: {
    n_stars: number;
    theta_arcmin: { median: number; max: number };
    beam_response: { min: number; median: number; mean: number };
    beam_attenuation_factor: { median: number; mean: number; max: number };
  };
  background_eirp_min_w: Quantiles;
  background_distance_pc: Quantiles;
  primary_eirp_min_w: Quantiles & { max: number; n_targets: number };
  primary_median_distance_pc: number;
  median_eirp_min_ratio_background_over_primary: number;
  decomposition_of_the_median_ratio: { distance_term: number; beam_term: number; note: string };
}

export interface BackgroundShellSummary {
  n_cells: number;
  n_stars: number;
  stars_per_pointing: { min: number; median: number; max: number; n_pointings: number; n_pointings_with_stars: number };
  distance_cumulative: { d_max_pc: number; n_stars: number }[];
  distance_quantiles_pc: Quantiles;
  eirp_min_cumulative_by_decade: { eirp_min_max_w: number; n_stars: number }[];
  eirp_min_quantiles_w: Quantiles;
}

export interface BackgroundFieldPromoted {
  promoted: true;
  provenance: BackgroundProvenance;
  validation_detail: string;
  shells: BackgroundShellSummary;
  attenuation: BackgroundAttenuation;
  pointing_centre_audit: PointingCentreAudit;
  suspected_unmatched_primary: SuspectedPrimary[];
  eig: BackgroundEig;
  bounds: BoundsRow[];
  correlation_assumption: string;
  commands: { report: string; merged_physical: string; validate: string };
}

export type BackgroundFieldBlock = BackgroundFieldPromoted | { promoted: false; reason: string };

export interface Grid {
  separations_arcsec: number[];
  flux_fractions: number[];
  centroid_recovery: number[][];
  image_recovery: number[][];
  centroid_false_alarm_empirical: number;
  image_false_alarm_empirical: number;
  n_null_trials: number;
  n_trials_per_cell: number;
  spec: { sigma_sys_arcsec: number; snr: number; false_alarm_nominal: number; psf_fwhm_arcsec: number };
}

export interface FalseAlarm {
  n_flagged: number;
  n_stars: number;
  rate: number;
  clopper_pearson_95: [number, number];
  clopper_pearson_95_upper: number;
  ks_rayleigh_p_value: number | null;
  binomial_p_at_least_k_given_nominal: number | null;
  sigma_sys_arcsec: number | null;
}

export interface SmallSampleBounds {
  linked_archival_job: string;
  withheld_at_measured_floor: FalseAlarm;
  withheld_at_legacy_floor: FalseAlarm;
  withheld_with_no_floor: FalseAlarm;
  calibration_in_sample_tail: FalseAlarm;
  statement: string;
}

export interface FloorBlock {
  registration_floor: RegistrationFloor;
  grid: Grid;
  screens_within_3p25_arcsec: Record<string, { survival: number; false_alarm: number }>;
}

export interface RegistrationFloor {
  label: string;
  sigma_sys_arcsec: number;
  data_tier: string;
  source: string;
  interval_68_arcsec: [number, number] | null;
}

export interface AnchorBenchmark {
  candidate: string;
  /** Decimal string (ruling v0.2.1-R4): Gaia DR3 ids exceed 2^53. */
  gaia_dr3_source_id: string;
  measured_w3_offset_arcsec: number;
  measured_err_arcsec: number;
  predicted_photocentre_shift_arcsec: [number, number];
  shift_tension_sigma: number;
  published_verdict: string;
  by_floor: Record<
    string,
    {
      sigma_sys_arcsec: number;
      combined_sigma_arcsec: number;
      offset_significance_sigma: number;
      centroid_screen_p_value: number;
      reproduced_verdict: string;
      verdict_matches_published: boolean;
    }
  >;
}

export interface Rate {
  k: number;
  n: number;
  rate: number;
  clopper_pearson_95: [number, number];
}

export type Gate2ScreenName = "centroid" | "image" | "naive_flux_excess";

export interface Gate2Screen {
  n_stars: number;
  null_false_alarm: Rate;
  arm_a_blend_recovery: number[][];
  arm_a_k: number[][];
  arm_a_clopper_pearson_95: [number, number][][];
  arm_b_false_rejection: number[];
  arm_b_k: number[];
  arm_b_clopper_pearson_95: [number, number][];
  arm_b_false_rejection_pooled: Rate;
  blend_survival_within_3p25_arcsec: number;
  centered_excess_retained_pooled: number;
}

/** Promoted Launch Gate #2 result (real AllWISE W3 FITS, reproduced_archive). */
export interface Gate2 {
  id: string;
  path: string;
  file_sha256: string;
  content_sha256: string;
  data_tier: string;
  code_git_sha: string;
  rng_seed: number;
  query_window_utc: string[];
  frozen_at_utc: string;
  n_requested: number;
  n_fetched_ok: number;
  n_calibration: number;
  n_withheld: number;
  locked_sigma_sys_arcsec: number;
  empirical_psf: {
    fwhm_gaussian_arcsec: number;
    fwhm_halfmax_arcsec: number;
    nominal_fwhm_arcsec: number;
    n_stars: number;
    pixel_scale_arcsec: number;
  };
  frozen_screens: {
    rule_version: string;
    target_false_alarm: number;
    centroid_multiplier: number;
    centroid_threshold_arcsec_at_calibration_median_snr: number;
    delta_chi2_scale: number;
    delta_chi2_threshold: number;
    calibration_false_alarm_in_sample: Record<string, Rate>;
    snr_eff_rule: string | null;
  };
  separations_arcsec: number[];
  flux_fractions: number[];
  screens: Record<Gate2ScreenName, Gate2Screen>;
  survival_weighting: string;
  caveats: string[];
}

export interface SurvivalSideBySide {
  max_separation_arcsec: number;
  weighting: string;
  real_fits: { label: string; data_tier: string; psf_fwhm_arcsec: number; sigma_sys_arcsec: number; n_stars: number };
  synthetic: {
    label: string;
    data_tier: string;
    psf_fwhm_arcsec: number;
    sigma_sys_arcsec: number;
    registration_floor: string;
    snr: number;
  };
  rows: {
    screen: string;
    real_fits_survival: number;
    synthetic_survival: number;
    real_fits_null_false_alarm: Rate;
    synthetic_null_false_alarm: number;
    real_fits_centered_excess_false_rejection: Rate;
  }[];
  note: string;
}

export interface IrDrill {
  git_sha: string;
  rng_seed: number;
  results_sha256: string;
  config: { n_trials: number; snr: number };
  results: {
    data_tier: string;
    real_fits_gate2: Gate2 | null;
    blend_survival_side_by_side: SurvivalSideBySide | null;
    registration_floors: RegistrationFloor[];
    small_sample_false_alarm_bounds: SmallSampleBounds | null;
    control_experiment: {
      n_controls: number;
      n_patches_with_controls: number;
      n_calibration: number;
      n_withheld: number;
      min_abs_b_deg: number;
      code_git_sha: string;
      file_sha256: string;
      query_window_utc: string[];
      patches: {
        patch_id: string;
        l_deg: number;
        b_deg: number;
        measured_controls: number;
        rejected_gaia_neighbour_within_15arcsec: number;
        n_calibration: number;
        n_withheld: number;
      }[];
    } | null;
    atlas_smoothing_caveat: string;
    anchor_benchmarks: AnchorBenchmark[];
    contamination_flip: {
      scenarios: Record<string, number>;
      rows: ({ registration_floor: string; screen: string } & Record<string, unknown>)[];
      policy: string;
      lr_reading: string;
    };
    by_floor: Record<string, FloorBlock>;
    floor_shift: {
      centroid_recovery_delta: number[][];
      image_recovery_delta: number[][];
      centroid_recovery_mean_delta: number;
      image_recovery_mean_delta: number;
      survival_delta: Record<string, number>;
    } | null;
    likelihood_ratio: { numerator: string; denominator: string; observed_candidates: number; n_targets: number };
  };
}

export interface Finding {
  check: string;
  status: "pass" | "finding" | "fail";
  severity: string;
  summary: string;
  details: Record<string, unknown>;
  citations: string[];
  repro_command: string;
}

export interface Adversary {
  git_sha: string;
  rng_seed: number;
  results_sha256: string;
  results: { findings: Finding[] };
}

export interface JobCore {
  job_id: string;
  tier: string;
  title: string;
  status: string;
  expected_eig_bits: number;
  est_cpu_seconds: number;
  est_reviewer_minutes: number;
  source: string;
  rng_seed: number | null;
  code_git_sha: string | null;
  output_sha256: string | null;
  parameters: Record<string, unknown>;
}

export interface Tier1Job extends JobCore {
  utility_score: number;
  eig_basis: string;
  eig_details: Record<string, unknown>;
  archival_script: string | null;
  commands: { claim: string; script: string | null; report?: string | null };
  linked_evidence: SmallSampleBounds | BackgroundLinkedEvidence | null;
}

/** v0.2.1b: realized-vs-predicted EIG linked to the completed background-field job. */
export interface BackgroundLinkedEvidence {
  kind: "background_field_eig";
  linked_archival_job: string;
  statement: string;
  tier_note: string;
  summary_path: string;
  summary_sha256: string;
  summary_data_tier: string;
  summary_code_git_sha: string;
  n_background_stars: number;
  comparison: EigComparisonRow[];
  scope: BackgroundScope;
}

export interface Tier0Job extends JobCore {
  diagnostic_priority: number;
  diagnostic_impact: string;
  eig_note: string;
  commands: { claim: string; run: string; replay: string };
}

export interface Queue {
  queue_path: string;
  tier1_archival: Tier1Job[];
  tier0_compute: Tier0Job[];
}

export interface RunResult {
  job: JobCore;
  output_json: { job_id: string; rng_seed: number; results: Record<string, unknown> };
  output_sha256: string;
  output_path: string;
}

export interface ReplayResult {
  job_id: string;
  rng_seed: number;
  recorded_sha256: string;
  replayed_sha256: string;
  match: boolean;
}

export interface RunSummary {
  path: string;
  kind: string;
  run_id: string;
  created_at: string | null;
  git_sha: string | null;
  rng_seed: number | null;
  sha256: string | null;
  sha256_kind: string;
  headline_policy: string | null;
  headline_log10_bf: number | null;
  data_tier: string | null;
}
