import { describe, expect, it } from "vitest";
import type { Adversary, Queue, Tier1Job } from "../api";
import { READ_ONLY_NOTICE, bfReading, fmtMultiplier, longRunLabel, tier0Gate } from "./format";
import { adversaryFacts, topArchivalJob, wiseControlCount } from "./overview";

describe("long-running Tier-0 job label (>= 120 s)", () => {
  it("labels jobs at or above the threshold with their estimated minutes", () => {
    expect(longRunLabel(240)).toBe("~4 min synchronous run");
    expect(longRunLabel(120)).toBe("~2 min synchronous run");
    expect(longRunLabel(600)).toBe("~10 min synchronous run");
  });
  it("stays silent for short or missing estimates", () => {
    expect(longRunLabel(119.9)).toBeNull();
    expect(longRunLabel(30)).toBeNull();
    expect(longRunLabel(null)).toBeNull();
    expect(longRunLabel(Number.NaN)).toBeNull();
  });
});

describe("read-only action gating", () => {
  it("disables Run and Replay with the CLI notice when the server is read-only", () => {
    for (const status of ["open", "claimed", "completed"]) {
      expect(tier0Gate(true, status, false)).toEqual({ runDisabled: true, replayDisabled: true, notice: READ_ONLY_NOTICE });
    }
    expect(READ_ONLY_NOTICE).toBe("Read-only mode — copy CLI command to run locally");
  });
  it("keeps both buttons disabled until the server mode is known", () => {
    const g = tier0Gate(null, "open", false);
    expect(g.runDisabled && g.replayDisabled).toBe(true);
    expect(g.notice).not.toBe(READ_ONLY_NOTICE);
    expect(tier0Gate(undefined, "completed", false).replayDisabled).toBe(true);
  });
  it("in read-write mode follows the job state machine", () => {
    expect(tier0Gate(false, "open", false)).toEqual({ runDisabled: false, replayDisabled: true, notice: null });
    expect(tier0Gate(false, "claimed", false)).toEqual({ runDisabled: false, replayDisabled: true, notice: null });
    expect(tier0Gate(false, "completed", false)).toEqual({ runDisabled: true, replayDisabled: false, notice: null });
    expect(tier0Gate(false, "completed", true)).toEqual({ runDisabled: true, replayDisabled: true, notice: null });
  });
});

describe("Overview readings", () => {
  it("reads |log10 BF| < 0.5 as a null separation and does not inflate it", () => {
    expect(bfReading(-8.04966e-5).kind).toBe("null");
    expect(bfReading(0.12876).kind).toBe("null");
    expect(bfReading(0.7).kind).toBe("weak");
    expect(bfReading(-1.2).kind).toBe("strong");
    expect(bfReading(null).kind).toBe("unknown");
  });
  it("formats multipliers", () => {
    expect(fmtMultiplier(1175.9135323721925)).toBe("1,176×");
    expect(fmtMultiplier(null)).toBe("—");
  });
});

const adv = {
  git_sha: "x",
  rng_seed: 42,
  results_sha256: "y",
  results: {
    findings: [
      {
        check: "hip1692_distance_revision",
        details: {
          canonical_label: "gaia_dr3_inverse_parallax",
          threshold: 1000,
          distances: {
            isaacson_2017_legacy: { eirp_factor_vs_legacy: 1 },
            gaia_dr2_bailer_jones_2018_r_est: { eirp_factor_vs_legacy: 1070.75 },
            gaia_dr3_inverse_parallax: { eirp_factor_vs_legacy: 1175.9135 },
          },
        },
      },
      { check: "duplicate_pointings", details: { n_beam_sharing_pairs: 16 } },
      {
        check: "local_approximation",
        details: {
          rows: [
            { hypothesis_id: "frequent_slow_expanders", t_valid_gyr: 0.0052185, valid: true },
            { hypothesis_id: "rare_fast_expanders", t_valid_gyr: 5.218502e-5, valid: false },
          ],
        },
      },
    ],
  },
} as unknown as Adversary;

describe("Overview extractors read live payloads, never constants", () => {
  it("pulls the canonical HIP 1692 multiplier, beam pairs and violated T_valid", () => {
    const f = adversaryFacts(adv);
    expect(f.hipLabel).toBe("gaia_dr3_inverse_parallax");
    expect(fmtMultiplier(f.hipMultiplier)).toBe("1,176×");
    expect(f.beamPairs).toBe(16);
    expect(f.tValidKyr?.toFixed(1)).toBe("52.2");
    expect(f.tValidHypothesis).toBe("rare_fast_expanders");
  });
  it("returns nulls (not remembered values) for a missing payload", () => {
    const f = adversaryFacts(null);
    expect([f.hipMultiplier, f.beamPairs, f.tValidKyr]).toEqual([null, null, null]);
  });
  it("picks the top Tier-1 job by utility and counts WISE controls from linked evidence", () => {
    const j = (id: string, u: number, bits: number, ev: unknown = null) =>
      ({ job_id: id, utility_score: u, expected_eig_bits: bits, linked_evidence: ev }) as unknown as Tier1Job;
    const q = {
      queue_path: "q",
      tier0_compute: [],
      tier1_archival: [
        j("wlodarczyk_sroka_288k_background_extension", 0.0046, 0.0145),
        j("wise_independent_control_fields", 0.2117, 1.0584, {
          calibration_in_sample_tail: { n_stars: 77 },
          withheld_at_measured_floor: { n_stars: 77 },
        }),
      ],
    } as unknown as Queue;
    expect(topArchivalJob(q)?.job_id).toBe("wise_independent_control_fields");
    expect(topArchivalJob(q)?.expected_eig_bits.toFixed(3)).toBe("1.058");
    expect(wiseControlCount(q)).toBe(154);
    expect(topArchivalJob(null)).toBeNull();
  });
});
