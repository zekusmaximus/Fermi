// Pure presentation helpers (unit-tested in format.test.ts). No I/O, no DOM.

export type TierClass = "published" | "archive" | "illustrative" | "model" | "unknown";

const ILLUSTRATIVE = new Set(["illustrative_calibration", "synthetic_proxy", "synthetic_proxy_692"]);

/** Map a data_tier string to its visual class (emerald / cyan / amber / slate). */
export function tierClass(tier: string | null | undefined): TierClass {
  if (!tier) return "model";
  if (tier === "published_catalog") return "published";
  if (tier === "reproduced_archive") return "archive";
  if (ILLUSTRATIVE.has(tier)) return "illustrative";
  return "unknown";
}

/** True when ANY of the tiers is illustrative / synthetic (illustrative dominates). */
export function isIllustrative(tiers: readonly string[] | null | undefined): boolean {
  return (tiers ?? []).some((t) => ILLUSTRATIVE.has(t));
}

/** Least-verified class of a tier set: illustrative > archive > published > model. */
export function worstTier(tiers: readonly string[] | null | undefined): TierClass {
  const list = tiers ?? [];
  if (list.length === 0) return "model";
  if (isIllustrative(list)) return "illustrative";
  if (list.some((t) => tierClass(t) === "unknown")) return "unknown";
  if (list.includes("reproduced_archive")) return "archive";
  return "published";
}

export const ILLUSTRATIVE_BANNER =
  "ILLUSTRATIVE CHANNELS INCLUDED — NOT AN OBSERVATIONAL BOUND";

/**
 * The loud banner is shown whenever the reported headline policy folds illustrative
 * channels in, OR when the user asked for that, OR when any term in the reported
 * headline carries an illustrative tier. It is never hidden on a verified-only policy
 * that somehow contains illustrative tiers (defence in depth).
 */
export function showIllustrativeBanner(
  policy: string | null | undefined,
  headlineTiers: readonly string[] | null | undefined,
  requested = false,
): boolean {
  return requested || policy === "including_illustrative_channels" || isIllustrative(headlineTiers);
}

/** Short, display-safe SHA-256 / git hash. Non-hex markers (dirty, unknown) pass through. */
export function shortSha(sha: string | null | undefined, n = 8): string {
  if (!sha) return "—";
  return /^[0-9a-f]{7,64}$/i.test(sha) ? sha.slice(0, n) : sha;
}

/** True only for a clean 40-hex commit hash (receipts from a dirty tree are flagged). */
export function isCleanGitSha(sha: string | null | undefined): boolean {
  return !!sha && /^[0-9a-f]{40}$/i.test(sha);
}

/** Signed log10 Bayes factor, scientific below 1e-3. */
export function fmtLog10(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  if (x === 0) return "0";
  const a = Math.abs(x);
  const body = a < 1e-3 || a >= 1e4 ? a.toExponential(digits - 1) : a.toPrecision(digits);
  return (x < 0 ? "−" : "+") + body;
}

/** General number: fixed for moderate magnitudes, scientific otherwise. */
export function fmtNum(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  if (x === 0) return "0";
  const a = Math.abs(x);
  return a < 1e-3 || a >= 1e5 ? x.toExponential(digits - 1) : x.toPrecision(digits);
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

/** Bounded integer seed (0 .. 2^32-1) or null if the text is not one. */
export function parseSeed(text: string): number | null {
  if (!/^\d{1,10}$/.test(text.trim())) return null;
  const n = Number(text.trim());
  return n <= 4294967295 ? n : null;
}

/** Heatmap colour: sequential for probabilities, diverging for deltas. */
export function cellColor(v: number, mode: "prob" | "delta"): string {
  if (!Number.isFinite(v)) return "transparent";
  if (mode === "prob") {
    const t = Math.max(0, Math.min(1, v));
    return `rgba(34, 211, 238, ${0.06 + 0.84 * t})`;
  }
  const t = Math.max(-1, Math.min(1, v / 0.2));
  return t >= 0 ? `rgba(52, 211, 153, ${0.08 + 0.8 * t})` : `rgba(248, 113, 113, ${0.08 + 0.8 * -t})`;
}

/** log10 axis ticks (decades) covering [lo, hi]. */
export function decadeTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(10 ** e);
  return out;
}

// ------------------------------------------------------------- v0.1.1 helpers

/** Tier-0 jobs at or above this estimate get an explicit "synchronous run" warning. */
export const LONG_RUN_THRESHOLD_S = 120;

/**
 * Inline label for a long synchronous Tier-0 run, so a multi-minute browser request is not
 * mistaken for a frozen page. Null below the threshold (or for a missing estimate).
 */
export function longRunLabel(estCpuSeconds: number | null | undefined): string | null {
  if (estCpuSeconds === null || estCpuSeconds === undefined || !Number.isFinite(estCpuSeconds)) return null;
  if (estCpuSeconds < LONG_RUN_THRESHOLD_S) return null;
  return `~${Math.round(estCpuSeconds / 60)} min synchronous run`;
}

export const READ_ONLY_NOTICE = "Read-only mode — copy CLI command to run locally";

export interface Tier0Gate {
  runDisabled: boolean;
  replayDisabled: boolean;
  notice: string | null;
}

/**
 * Enable/disable the Tier-0 Run and Verify Replay buttons. `readOnly` comes from
 * /api/health: `null` means not yet known, and the buttons stay disabled until the server
 * has confirmed read-write mode (the server also enforces this with HTTP 403).
 */
export function tier0Gate(readOnly: boolean | null | undefined, status: string, busy: boolean): Tier0Gate {
  if (readOnly !== false) {
    return {
      runDisabled: true,
      replayDisabled: true,
      notice: readOnly === true ? READ_ONLY_NOTICE : "checking server mode…",
    };
  }
  const done = status === "completed";
  return { runDisabled: busy || done, replayDisabled: busy || !done, notice: null };
}

/**
 * Plain-English reading of a log10 Bayes factor on the Jeffreys scale. |log10 BF| < 0.5
 * ("barely worth mentioning") is reported as a null separation, never dressed up.
 */
export function bfReading(log10Bf: number | null | undefined): { kind: "null" | "weak" | "strong" | "unknown"; text: string } {
  if (log10Bf === null || log10Bf === undefined || !Number.isFinite(log10Bf)) {
    return { kind: "unknown", text: "headline not available" };
  }
  const a = Math.abs(log10Bf);
  if (a < 0.5) return { kind: "null", text: "null separation (|log10 BF| < 0.5, Jeffreys: barely worth mentioning)" };
  if (a < 1) return { kind: "weak", text: "substantial but not strong evidence (0.5 ≤ |log10 BF| < 1)" };
  return { kind: "strong", text: "strong evidence (|log10 BF| ≥ 1)" };
}

/** "1,176×" style multiplier. */
export function fmtMultiplier(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return `${Math.round(x).toLocaleString("en-US")}×`;
}
