// Pure extractors for the Overview tab (unit-tested in overview.test.ts). Every number on
// the Overview comes from an existing endpoint payload through these functions; when a
// field is missing they return null and the UI shows "—" rather than a remembered value.
import type { Adversary, Queue, Tier1Job } from "../api";

export interface AdversaryFacts {
  hipLabel: string | null;
  hipMultiplier: number | null;
  hipThreshold: number | null;
  beamPairs: number | null;
  /** Shortest local-disk validity time among hypotheses whose approximation is violated. */
  tValidKyr: number | null;
  tValidHypothesis: string | null;
  nFindings: number;
}

type Details = Record<string, unknown>;
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function adversaryFacts(adv: Adversary | null | undefined): AdversaryFacts {
  const findings = adv?.results.findings ?? [];
  const by = (check: string): Details | undefined => findings.find((f) => f.check === check)?.details;

  const hip = by("hip1692_distance_revision");
  const label = typeof hip?.["canonical_label"] === "string" ? (hip["canonical_label"] as string) : null;
  const dists = (hip?.["distances"] ?? {}) as Record<string, Details>;
  const hipMultiplier = label ? num(dists[label]?.["eirp_factor_vs_legacy"]) : null;

  const dup = by("duplicate_pointings");
  const pairsList = dup?.["beam_sharing_pairs"];
  const beamPairs = num(dup?.["n_beam_sharing_pairs"]) ?? (Array.isArray(pairsList) ? pairsList.length : null);

  const rows = (by("local_approximation")?.["rows"] ?? []) as Details[];
  let tValidKyr: number | null = null;
  let tValidHypothesis: string | null = null;
  for (const r of rows) {
    const t = num(r["t_valid_gyr"]);
    if (r["valid"] === false && t !== null && (tValidKyr === null || t * 1e6 < tValidKyr)) {
      tValidKyr = t * 1e6;
      tValidHypothesis = typeof r["hypothesis_id"] === "string" ? (r["hypothesis_id"] as string) : null;
    }
  }
  return {
    hipLabel: label,
    hipMultiplier,
    hipThreshold: num(hip?.["threshold"]),
    beamPairs,
    tValidKyr,
    tValidHypothesis,
    nFindings: findings.length,
  };
}

/** Highest-utility Tier-1 archival job (bits per hour), regardless of server ordering. */
export function topArchivalJob(q: Queue | null | undefined): Tier1Job | null {
  const jobs = q?.tier1_archival ?? [];
  return jobs.reduce<Tier1Job | null>((best, j) => (best === null || j.utility_score > best.utility_score ? j : best), null);
}

// v0.1.2: the AllWISE control count is read from /api/health (n_controls), which the server
// takes from the verified calibration file, instead of summing the queue's linked evidence.
