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
  receipt_schema_version: number;
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
  surveys: { id: string; name: string; channel: string; data_tier: string; n_targets: number }[];
}

export type Geometry = "thin_disk_2d" | "thick_disk_3d";
export type RadioMode = "flat_probability" | "distance_scaled_eirp";
export type Channel = "radio" | "infrared" | "solar_system";

export interface CompareRequest {
  geometry_mode: Geometry;
  radio_mode: RadioMode;
  condition_on_earth_unreached: boolean;
  channels: Channel[];
  include_illustrative: boolean;
  seed: number;
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
  reasons: string[];
  data_tiers: string[];
  log10_bf_term: number | null;
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
  } | null;
  radio?: {
    expected_reached: number | null;
    expected_reached_unconditional: number | null;
    p_zero_exact_mc: number | null;
    p_zero_exact_mc_se: number | null;
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
}

export interface Grid {
  separations_arcsec: number[];
  flux_fractions: number[];
  centroid_recovery: number[][];
  image_recovery: number[][];
  centroid_false_alarm_empirical: number;
  image_false_alarm_empirical: number;
  n_null_trials: number;
  n_trials_per_cell: number;
  spec: { sigma_sys_arcsec: number; snr: number; false_alarm_nominal: number };
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
  gaia_dr3_source_id: number;
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

export interface IrDrill {
  git_sha: string;
  rng_seed: number;
  results_sha256: string;
  config: { n_trials: number; snr: number };
  results: {
    data_tier: string;
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
  commands: { claim: string; script: string | null };
  linked_evidence: SmallSampleBounds | null;
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
