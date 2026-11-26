import { useCallback, useEffect, useState, type ReactNode } from "react";
import { ApiError } from "../api";
import { shortSha, tierClass, worstTier, type TierClass } from "../lib/format";

const TIER_LABEL: Record<TierClass, string> = {
  published: "published_catalog",
  archive: "reproduced_archive",
  illustrative: "illustrative_calibration",
  model: "model term",
  unknown: "unclassified tier",
};

/** One data-tier badge. Illustrative badges are amber AND dashed, so they never read as verified. */
export function TierBadge({ tier, label }: { tier?: string | null; label?: string }) {
  const cls = tierClass(tier ?? null);
  return (
    <span className={`tier tier-${cls}`} title={tier ?? "model term (no data tier)"}>
      {label ?? (tier || TIER_LABEL[cls])}
    </span>
  );
}

/** Badge for a set of tiers, showing the least-verified class and listing all of them. */
export function TierSetBadge({ tiers }: { tiers: readonly string[] }) {
  const cls = worstTier(tiers);
  return (
    <span className={`tier tier-${cls}`} title={tiers.join(", ") || "model term"}>
      {cls === "illustrative" ? "ILLUSTRATIVE" : TIER_LABEL[cls]}
    </span>
  );
}

export function Hash({ sha, n = 8 }: { sha: string | null | undefined; n?: number }) {
  return (
    <code className="hash" title={sha ?? ""}>
      {shortSha(sha, n)}
    </code>
  );
}

export function CopyButton({ text, label = "copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="btn btn-ghost btn-xs"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          window.setTimeout(() => setDone(false), 1200);
        });
      }}
      title={text}
    >
      {done ? "copied" : label}
    </button>
  );
}

export function Command({ text }: { text: string }) {
  return (
    <div className="cmd">
      <code>{text}</code>
      <CopyButton text={text} />
    </div>
  );
}

export function Panel({
  title,
  aside,
  children,
  className = "",
}: {
  title: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-head">
        <h3>{title}</h3>
        {aside ? <div className="panel-aside">{aside}</div> : null}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function StatusPill({ status }: { status: string }) {
  return <span className={`pill pill-${status}`}>{status}</span>;
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg =
    error instanceof ApiError
      ? `HTTP ${error.status}: ${error.message}`
      : error instanceof Error
        ? error.message
        : String(error);
  return <div className="error-box">⚠ {msg}</div>;
}

export function Loading({ what }: { what: string }) {
  return <div className="loading">computing {what}…</div>;
}

/** Fetch-on-deps hook with stale-response protection. */
export function useApi<T>(fetcher: () => Promise<T>, deps: readonly unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    fetcher()
      .then((d) => live && setData(d))
      .catch((e: unknown) => live && setError(e))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { data, error, loading, reload };
}
