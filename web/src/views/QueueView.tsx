import { useState } from "react";
import { ApiError, get, post, type Queue, type ReplayResult, type RunResult, type RunSummary, type Tier0Job } from "../api";
import { Command, ErrorBox, Hash, Loading, Panel, StatusPill, useApi } from "../components/ui";
import { fmtLog10, fmtNum, isCleanGitSha } from "../lib/format";

export function QueueView() {
  const [tab, setTab] = useState<"t1" | "t0">("t1");
  const q = useApi(() => get<Queue>("/api/queue"), []);
  const runs = useApi(() => get<{ runs: RunSummary[]; limit: number }>("/api/runs"), []);
  return (
    <div className="view">
      <p className="eyebrow">Split BOED action queue {q.data ? <>· {q.data.queue_path}</> : null}</p>
      <div className="tabs tabs-lg">
        <button className={tab === "t1" ? "tab on" : "tab"} onClick={() => setTab("t1")}>
          Tier-1 Archival Queue · EIG bits/hr
        </button>
        <button className={tab === "t0" ? "tab on" : "tab"} onClick={() => setTab("t0")}>
          Tier-0 Compute Queue · diagnostic priority
        </button>
      </div>
      {q.error ? <ErrorBox error={q.error} /> : null}
      {/* Keep stale data on screen during reloads: unmounting the rows would discard
          the inline run viewer and replay verdict the user just produced. */}
      {!q.data ? (
        <Loading what="queue" />
      ) : tab === "t1" ? (
        <Tier1 q={q.data} />
      ) : (
        <Tier0 q={q.data} onChange={() => (q.reload(), runs.reload())} />
      )}
      <Panel title="Recent receipts" aside={<button className="btn btn-xs" onClick={runs.reload}>refresh</button>}>
        {runs.error ? <ErrorBox error={runs.error} /> : null}
        {runs.data ? (
          runs.data.runs.length === 0 ? (
            <p className="muted">No receipts in runs/ yet.</p>
          ) : (
            <ol className="feed">
              {runs.data.runs.map((r) => (
                <li key={r.path}>
                  <span className="feed-kind">{r.kind}</span>
                  <span className="num muted">{r.created_at?.replace("T", " ").slice(0, 19) ?? "—"}</span>
                  <span>
                    git{" "}
                    <span className={isCleanGitSha(r.git_sha) ? "" : "warn"}>
                      <Hash sha={r.git_sha} />
                    </span>
                  </span>
                  <span>
                    seed <span className="num">{r.rng_seed ?? "—"}</span>
                  </span>
                  <span>
                    {r.sha256_kind === "results_sha256" ? "results" : "output"} <Hash sha={r.sha256} />
                  </span>
                  {r.headline_policy ? (
                    <span className={r.headline_policy === "verified_channels_only" ? "policy policy-verified" : "policy policy-illustrative"}>
                      {r.headline_policy} {fmtLog10(r.headline_log10_bf)}
                    </span>
                  ) : null}
                  <span className="muted small">{r.path}</span>
                </li>
              ))}
            </ol>
          )
        ) : (
          <Loading what="receipts" />
        )}
      </Panel>
    </div>
  );
}

function Tier1({ q }: { q: Queue }) {
  return (
    <Panel title="Tier-1 archival jobs (live data; run from a terminal, never from the desk)">
      {q.tier1_archival.map((j, i) => {
        const modes = (j.eig_details["by_radio_mode"] ?? null) as Record<string, { eig_bits: number }> | null;
        return (
          <article key={j.job_id} className="job">
            <header>
              <span className="rank num">#{i + 1}</span> <b>{j.job_id}</b> <StatusPill status={j.status} />
            </header>
            <p>{j.title}</p>
            <div className="job-stats num">
              <span>EIG {fmtNum(j.expected_eig_bits, 4)} bits</span>
              <span>utility {fmtNum(j.utility_score, 4)} bits/hr</span>
              <span>cpu {j.est_cpu_seconds.toFixed(0)} s</span>
              <span>review {j.est_reviewer_minutes.toFixed(0)} min</span>
            </div>
            {modes ? (
              <p className="small num">
                by radio mode: distance_scaled_eirp (physical) {fmtNum(modes["distance_scaled_eirp"]?.eig_bits, 3)} bits · flat_probability{" "}
                {fmtNum(modes["flat_probability"]?.eig_bits, 3)} bits
              </p>
            ) : null}
            <p className="muted small">{j.eig_basis}</p>
            {j.linked_evidence ? (
              <div className="linked">
                <p className="eyebrow">linked small-sample evidence (Launch Gate #1)</p>
                <p className="small">{j.linked_evidence.statement}</p>
              </div>
            ) : null}
            <Command text={j.commands.claim} />
            {j.commands.script ? <Command text={j.commands.script} /> : <p className="muted small">script: {j.archival_script}</p>}
          </article>
        );
      })}
    </Panel>
  );
}

function Tier0({ q, onChange }: { q: Queue; onChange: () => void }) {
  return (
    <Panel title="Tier-0 compute jobs (offline, deterministic; EIG = 0.0 bits by construction)">
      {q.tier0_compute.map((j) => (
        <Tier0Row key={j.job_id} j={j} onChange={onChange} />
      ))}
    </Panel>
  );
}

function Tier0Row({ j, onChange }: { j: Tier0Job; onChange: () => void }) {
  const [busy, setBusy] = useState<"" | "run" | "replay">("");
  const [run, setRun] = useState<RunResult | null>(null);
  const [rep, setRep] = useState<ReplayResult | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const act = async (kind: "run" | "replay") => {
    setBusy(kind);
    setErr(null);
    try {
      if (kind === "run") setRun(await post<RunResult>(`/api/queue/${j.job_id}/run`, { seed: 42, contributor: "local-desk-ui" }));
      else setRep(await post<ReplayResult>(`/api/queue/${j.job_id}/replay`, {}));
      onChange();
    } catch (e) {
      setErr(e);
    } finally {
      setBusy("");
    }
  };
  const done = j.status === "completed";
  return (
    <article className="job">
      <header>
        <span className="rank num">P{j.diagnostic_priority}</span> <b>{j.job_id}</b> <StatusPill status={j.status} />{" "}
        <span className="muted small">EIG = {j.expected_eig_bits.toFixed(1)} bits</span>
      </header>
      <p>{j.diagnostic_impact}</p>
      <p className="muted small">{j.eig_note}</p>
      <div className="job-stats num">
        <span>cpu ≈ {j.est_cpu_seconds.toFixed(0)} s</span>
        <span>
          output <Hash sha={j.output_sha256} />
        </span>
        <span>seed {j.rng_seed ?? "—"}</span>
      </div>
      <div className="row-btns">
        <button className="btn" disabled={busy !== "" || done} onClick={() => void act("run")}>
          {busy === "run" ? "running…" : "Run (seed 42)"}
        </button>
        <button className="btn" disabled={busy !== "" || !done} onClick={() => void act("replay")}>
          {busy === "replay" ? "replaying…" : "Verify Replay"}
        </button>
        <Command text={j.commands.replay} />
      </div>
      {err ? <ErrorBox error={err instanceof ApiError ? err : err} /> : null}
      {rep ? (
        <p className={rep.match ? "ok" : "bad"}>
          replay {rep.match ? "MATCH" : "MISMATCH"} · recorded <Hash sha={rep.recorded_sha256} n={16} /> · replayed{" "}
          <Hash sha={rep.replayed_sha256} n={16} />
        </p>
      ) : null}
      {run ? <RunViewer r={run} /> : null}
    </article>
  );
}

interface LookbackRow {
  geometry: string;
  lookback_gyr: number;
  hypotheses: Record<string, { p_earth_reached: number; local_approx: string }>;
  log10_bf_headline_verified: number | null;
}

function RunViewer({ r }: { r: RunResult }) {
  const res = r.output_json.results;
  const thresholds = res["thresholds"] as Record<string, { t_valid_gyr: number; expansion_speed_c: number }> | undefined;
  const rows = res["rows"] as LookbackRow[] | undefined;
  return (
    <div className="run-viewer">
      <p className="small">
        completed · output_sha256 <Hash sha={r.output_sha256} n={16} /> · {r.output_path}
      </p>
      {thresholds && rows ? (
        <>
          <ul className="thresholds">
            {Object.entries(thresholds).map(([h, t]) => (
              <li key={h} className="num">
                T_valid({h}) = 8 kpc / ({t.expansion_speed_c} c) = <b>{(t.t_valid_gyr * 1e6).toFixed(1)} kyr</b>
              </li>
            ))}
          </ul>
          <table className="tbl tbl-compact">
            <thead>
              <tr>
                <th>geometry</th>
                <th className="r">T [kyr]</th>
                {Object.keys(thresholds).map((h) => (
                  <th key={h}>local approx · {h}</th>
                ))}
                <th className="r">verified headline log10 BF</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.geometry}-${row.lookback_gyr}`}>
                  <td>{row.geometry}</td>
                  <td className="r num">{(row.lookback_gyr * 1e6).toFixed(1)}</td>
                  {Object.keys(thresholds).map((h) => (
                    <td key={h}>
                      {row.hypotheses[h]?.local_approx === "ok" ? <span className="ok">ok</span> : <span className="badge-violated">VIOLATED</span>}
                    </td>
                  ))}
                  <td className="r num">{fmtLog10(row.log10_bf_headline_verified)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className="muted small">result keys: {Object.keys(res).join(", ")}</p>
      )}
    </div>
  );
}
