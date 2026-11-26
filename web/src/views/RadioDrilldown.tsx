import { useMemo, useState } from "react";
import { get, type BeamPair, type RadioDrill, type Target } from "../api";
import { ErrorBox, Hash, Loading, Panel, TierBadge, useApi } from "../components/ui";
import { decadeTicks, fmtNum } from "../lib/format";

const SOURCE_TIER: Record<string, string> = {
  gaia_dr3_parallax: "reproduced_archive",
  isaacson_2017_vizier: "published_catalog",
};
const LABELS: Record<string, string> = {
  isaacson_2017_legacy: "Legacy (Isaacson 2017)",
  gaia_dr2_bailer_jones_2018_r_est: "Gaia DR2 Bailer-Jones",
  gaia_dr3_inverse_parallax: "Gaia DR3 1/parallax",
};

export function RadioDrilldown() {
  const { data, error, loading } = useApi(() => get<RadioDrill>("/api/radio/drilldown"), []);
  if (error) return <ErrorBox error={error} />;
  if (loading || !data) return <Loading what="radio drill-down" />;
  return (
    <div className="view">
      <p className="eyebrow">
        Radio drill-down · {data.survey_id} · <TierBadge tier={data.data_tier} /> · catalog <Hash sha={data.catalog_sha256} />
      </p>
      <HipCard d={data} />
      <Panel
        title={`${data.n_targets} targets: distance vs minimum detectable EIRP (log-log)`}
        aside={
          <span className="legend">
            <i className="dot dot-archive" /> gaia_dr3_parallax <TierBadge tier="reproduced_archive" />
            <i className="dot dot-published" /> isaacson_2017_vizier <TierBadge tier="published_catalog" />
            <i className="dot dot-hip" /> HIP 1692
          </span>
        }
      >
        <Scatter targets={data.targets} fmin={data.f_min_w_m2} />
      </Panel>
      <Pairs pairs={data.beam_sharing_pairs} fwhm={data.beam_fwhm_arcmin} note={data.beam_sharing_interpretation} />
    </div>
  );
}

function HipCard({ d }: { d: RadioDrill }) {
  const a = d.hip1692_audit;
  const order = ["isaacson_2017_legacy", "gaia_dr2_bailer_jones_2018_r_est", "gaia_dr3_inverse_parallax"];
  return (
    <Panel title={`${a.target_name} spotlight — distance revision`} aside={<span className={`pill pill-${a.status}`}>{a.status}</span>}>
      <div className="cards3">
        {order
          .filter((k) => k in a.distances)
          .map((k) => {
            const v = a.distances[k]!;
            const role = k === a.canonical_label ? "canonical" : k === a.historical_anchor_label ? "historical anchor" : "legacy";
            return (
              <div key={k} className={`stat-card stat-${role.replace(" ", "-")}`}>
                <p className="eyebrow">
                  {LABELS[k] ?? k} · {role}
                </p>
                <p className="stat-big num">{v.distance_pc.toFixed(2)} pc</p>
                <p className="num">EIRP_min = {v.eirp_min_w.toExponential(3)} W</p>
                <p className="num">
                  ×{v.eirp_factor_vs_legacy.toLocaleString("en-US", { maximumFractionDigits: 0 })} vs legacy
                  {k !== "isaacson_2017_legacy" ? (v.exceeds_threshold ? ` (> ${a.threshold}×)` : " (BELOW threshold)") : ""}
                </p>
                <TierBadge tier={v.data_tier} />
              </div>
            );
          })}
      </div>
      <p className="muted small">
        EIRP_min = 4π d² F_min with F_min = {d.f_min_w_m2.toExponential(4)} W m⁻² (Enriquez et al. 2017 §5.1).{" "}
        {d.hip1692_citations.join(" · ")}
      </p>
    </Panel>
  );
}

const W = 860;
const H = 380;
const M = { l: 70, r: 16, t: 12, b: 44 };

function Scatter({ targets, fmin }: { targets: Target[]; fmin: number }) {
  const [hover, setHover] = useState<Target | null>(null);
  const dom = useMemo(() => {
    const xs = targets.map((t) => t.dist_pc);
    const ys = targets.map((t) => t.eirp_min_w);
    return { x0: Math.min(...xs) * 0.8, x1: Math.max(...xs) * 1.25, y0: Math.min(...ys) * 0.6, y1: Math.max(...ys) * 1.6 };
  }, [targets]);
  const sx = (x: number) => M.l + ((Math.log10(x) - Math.log10(dom.x0)) / (Math.log10(dom.x1) - Math.log10(dom.x0))) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - ((Math.log10(y) - Math.log10(dom.y0)) / (Math.log10(dom.y1) - Math.log10(dom.y0))) * (H - M.t - M.b);
  const xt = decadeTicks(dom.x0, dom.x1).filter((v) => v >= dom.x0 && v <= dom.x1);
  const yt = decadeTicks(dom.y0, dom.y1).filter((v) => v >= dom.y0 && v <= dom.y1);
  const hip = targets.find((t) => t.target_name === "HIP 1692");
  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="distance versus EIRP_min scatter">
        {xt.map((v) => (
          <g key={`x${v}`}>
            <line x1={sx(v)} x2={sx(v)} y1={M.t} y2={H - M.b} className="grid-line" />
            <text x={sx(v)} y={H - M.b + 16} className="axis-text" textAnchor="middle">
              {v >= 1 ? v : v.toExponential(0)}
            </text>
          </g>
        ))}
        {yt.map((v) => (
          <g key={`y${v}`}>
            <line x1={M.l} x2={W - M.r} y1={sy(v)} y2={sy(v)} className="grid-line" />
            <text x={M.l - 6} y={sy(v) + 4} className="axis-text" textAnchor="end">
              1e{Math.round(Math.log10(v))}
            </text>
          </g>
        ))}
        <text x={(W + M.l) / 2} y={H - 6} className="axis-label" textAnchor="middle">
          distance [pc]
        </text>
        <text x={14} y={(H - M.b) / 2} className="axis-label" textAnchor="middle" transform={`rotate(-90 14 ${(H - M.b) / 2})`}>
          EIRP_min [W]
        </text>
        {targets.map((t) => (
          <circle
            key={t.target_name}
            cx={sx(t.dist_pc)}
            cy={sy(t.eirp_min_w)}
            r={hover?.target_name === t.target_name ? 5 : 2.6}
            className={t.dist_source === "gaia_dr3_parallax" ? "pt pt-archive" : "pt pt-published"}
            onMouseEnter={() => setHover(t)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {hip ? (
          <g>
            <circle cx={sx(hip.dist_pc)} cy={sy(hip.eirp_min_w)} r={9} className="pt-hip" />
            <text x={sx(hip.dist_pc) - 12} y={sy(hip.eirp_min_w) - 12} className="axis-text hip-label" textAnchor="end">
              HIP 1692
            </text>
          </g>
        ) : null}
      </svg>
      <div className="tooltip" aria-live="polite">
        {hover ? (
          <>
            <b>{hover.target_name}</b> · <span className="num">{hover.dist_pc.toFixed(2)} pc</span> ·{" "}
            <span className="num">{hover.eirp_min_w.toExponential(3)} W</span> · {hover.dist_source}{" "}
            <TierBadge tier={SOURCE_TIER[hover.dist_source] ?? null} />
            {hover.eirp_revision_factor ? <span className="num"> · revision ×{fmtNum(hover.eirp_revision_factor)}</span> : null}
          </>
        ) : (
          <span className="muted">hover a point · F_min = {fmin.toExponential(3)} W m⁻² for all targets</span>
        )}
      </div>
    </div>
  );
}

type SortKey = "pair_id" | "a" | "separation_arcmin" | "separation_over_beam_fwhm";

function Pairs({ pairs, fwhm, note }: { pairs: BeamPair[]; fwhm: number; note: string }) {
  const [q, setQ] = useState("");
  const [key, setKey] = useState<SortKey>("separation_arcmin");
  const [asc, setAsc] = useState(true);
  const rows = useMemo(() => {
    const f = pairs.filter((p) => `${p.a} ${p.b}`.toLowerCase().includes(q.toLowerCase()));
    return [...f].sort((x, y) => {
      const a = x[key];
      const b = y[key];
      const c = typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
      return asc ? c : -c;
    });
  }, [pairs, q, key, asc]);
  const th = (k: SortKey, label: string) => (
    <th className="sortable" onClick={() => (k === key ? setAsc(!asc) : (setKey(k), setAsc(true)))}>
      {label} {k === key ? (asc ? "▲" : "▼") : ""}
    </th>
  );
  return (
    <Panel
      title={`${pairs.length} GBT beam-sharing pairs (< ${fwhm}′ FWHM)`}
      aside={<input className="search" placeholder="search pairs…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="search pairs" />}
    >
      <table className="tbl">
        <thead>
          <tr>
            {th("pair_id", "pair")}
            {th("a", "targets")}
            {th("separation_arcmin", "separation [′]")}
            {th("separation_over_beam_fwhm", "fraction of FWHM")}
            <th>distances [pc]</th>
            <th>counting</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.pair_id}>
              <td className="num">{p.pair_id}</td>
              <td>
                {p.a} / {p.b}
              </td>
              <td className="num">{p.separation_arcmin.toFixed(3)}</td>
              <td className="num">{(p.separation_over_beam_fwhm * 100).toFixed(1)}%</td>
              <td className="num">
                {p.dist_a_pc.toFixed(1)} / {p.dist_b_pc.toFixed(1)}
              </td>
              <td>{p.counted_once_each ? <span className="ok">✓ counted once each</span> : <span className="bad">DOUBLE</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">{note}</p>
    </Panel>
  );
}
