import { useMemo, useState } from "react";
import { get, type BackgroundFieldBlock, type BeamPair, type PointingCentreAudit, type RadioDrill, type Target } from "../api";
import { Command, ErrorBox, Hash, HelpPopover, Loading, Panel, TierBadge, useApi } from "../components/ui";
import {
  attenuationHeadline,
  auditCaveat,
  coneLabel,
  dedupCloses,
  dedupRows,
  distanceCutRows,
  eigRows,
  eigVerdict,
  fmtCount,
  fmtDec,
  fmtWatts,
  gaiaIdText,
  overlayStatus,
  scopeRows,
  utcWindow,
} from "../lib/background";
import { decadeTicks, fmtNum, isCleanGitSha } from "../lib/format";

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
      <BackgroundField block={data.background_field} />
    </div>
  );
}

/**
 * v0.2.1c (ruling v0.2.1-R5): the pointing-centre audit block. Isaacson et al. (2017)
 * Table 1 prints 14 targets at +10° instead of −10°; the locked catalog keeps the published
 * values (emerald, published_catalog) and the background cones are re-centred by a
 * SHA-256-pinned overlay whose positions come from Gaia DR3 / Hipparcos (cyan,
 * reproduced_archive). Cyan when nothing is left uncorrected, red while any cone is.
 */
function PointingAudit({ audit }: { audit: PointingCentreAudit }) {
  const status = overlayStatus(audit);
  if (!status) return null;
  const fixed = audit.corrected_via_overlay;
  const stale = audit.uncorrected_flipped_cones;
  return (
    <div className={status.clean ? "audit-block audit-ok" : "audit-block audit-bad"} aria-label="Pointing-centre audit">
      <p className="eyebrow">
        Pointing-centre audit — catalog erratum (Isaacson et al. 2017, declination −10° band){" "}
        <HelpPopover
          label="the catalog erratum and how it is corrected"
          title="Catalog erratum, corrected without editing the catalog"
          tradeoff={
            `Isaacson et al. (2017) Table 1 prints every target with a true declination between −11° and −10° without its minus ` +
            `sign, so ${fmtCount(fixed.n_pinned_corrections)} of the ${fmtCount(audit.n_primaries)} beams were catalogued about 21° ` +
            `north of the star that was observed. The target list is kept exactly as published; a pinned overlay file re-centres ` +
            `only those background cones on the stars’ J2000 positions (Gaia DR3 / Hipparcos). The verified headline is unchanged: ` +
            `the background field never enters it in either radio mode.`
          }
          invariant={
            `Ruling ${fixed.ruling ?? "v0.2.1-R5"}: the locked catalog is immutable (its SHA-256 is pinned by every receipt); ` +
            `corrections live only in the SHA-256-pinned overlay. Adversary finding: catalog_dec_sign_audit.`
          }
          cli={["uv run fermi-desk validate", "uv run fermi-desk adversary run --seed 42", "uv run fermi-desk background-field"]}
        />
      </p>
      <p className="audit-headline">
        <span className="stat-big num">{status.corrected}</span> catalog-erratum cones corrected non-destructively ·{" "}
        <span className={stale.n === 0 ? "ok num" : "bad num"}>{status.remaining}</span>
      </p>
      <p>
        Cones re-centred on J2000 positions <TierBadge tier="reproduced_archive" /> by the pinned overlay{" "}
        <code className="num">{fixed.overlay_path}</code> · overlay SHA-256 <Hash sha={fixed.overlay_sha256} n={16} /> ·{" "}
        <span className="num">{fixed.ruling}</span>
      </p>
      <p>
        Locked catalog <code className="num">{fixed.locked_catalog_path}</code> <TierBadge tier="published_catalog" />{" "}
        <Hash sha={fixed.locked_catalog_sha256} n={16} />{" "}
        <span className={fixed.locked_catalog_modified ? "bad" : "ok"}>{fixed.locked_catalog_modified ? "MODIFIED" : "unmodified (read-only)"}</span>
      </p>
      <p className="small">
        Corrected primaries: {status.absorbed}. {fixed.not_absorbed.length ? <span className="muted">Not matched means: {fixed.not_absorbed_reading}.</span> : null}{" "}
        <span className="num">{fmtCount(fixed.n_listed_as_suspected_unmatched_primary)}</span> of them listed as suspected unmatched primaries.
      </p>
      {stale.n > 0 ? (
        <p className="bad small">
          Still on a sign-flipped centre: {stale.targets.join(", ")}. {stale.reading}
        </p>
      ) : null}
      <details>
        <summary>list the {fmtCount(fixed.cones.length)} corrected cones (catalog value → centre used)</summary>
        <div className="scroll-x">
          <table className="tbl tbl-compact">
            <thead>
              <tr>
                <th>target</th>
                <th className="r">
                  catalog Dec <TierBadge tier="published_catalog" />
                </th>
                <th className="r">
                  cone centre Dec (J2000) <TierBadge tier="reproduced_archive" />
                </th>
                <th className="r">cone centre RA (J2000)</th>
                <th>position from</th>
                <th>fetched by commit</th>
              </tr>
            </thead>
            <tbody>
              {fixed.cones.map((c) => (
                <tr key={c.target_name}>
                  <td>{c.target_name}</td>
                  <td className="r num">{fmtDec(c.catalog_dec_deg)}</td>
                  <td className="r num">{fmtDec(c.cone_dec_deg)}</td>
                  <td className="r num">{c.cone_ra_deg.toFixed(4)}°</td>
                  <td>{c.truth_source === "gaia_dr3" ? "Gaia DR3" : "Hipparcos-2"}</td>
                  <td>
                    <Hash sha={c.code_git_sha} n={12} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

/**
 * v0.2.1b: the promoted Gaia DR3 background-field extension. Tier discipline (DESIGN.md
 * Sec. 1): provenance, counts and de-duplication are reproduced_archive (cyan); every EIG
 * and P(no detection) also depends on an illustrative input and is drawn amber / hatched.
 */
function BackgroundField({ block }: { block: BackgroundFieldBlock }) {
  if (!block.promoted) {
    return (
      <Panel title="Background Field Extension — not promoted">
        <p className="muted">{block.reason}</p>
      </Panel>
    );
  }
  const p = block.provenance;
  const d = p.dedup_audit;
  const mp = p.merge_provenance;
  const att = block.attenuation;
  const caveat = auditCaveat(block.pointing_centre_audit);
  const band = block.pointing_centre_audit.densest_band_of_unfound_primaries;
  const kindClass = (k: string | undefined) => (k === "flag" ? "row-flag" : k === "total" ? "row-total" : "");
  return (
    <>
      <Panel
        className="panel-archive"
        title={`Background Field Extension (${p.data_tier})`}
        aside={
          <span className="legend">
            <TierBadge tier={p.data_tier} /> {p.survey_id}
          </span>
        }
      >
        <div className="cards3">
          <div className="stat-card stat-canonical">
            <p className="eyebrow">background stars N_bg</p>
            <p className="stat-big num">{fmtCount(p.n_background_stars)}</p>
            <p className="num">
              {fmtCount(p.n_pointings)} pointings · cone {coneLabel(p.cone_radius_arcmin, p.beam_fwhm_arcmin)}
            </p>
            <p className="num">
              {fmtCount(p.n_cells)} shell cells · d ≤ {fmtCount(p.distance_cuts.d_max_pc)} pc
            </p>
          </div>
          <div className="stat-card">
            <p className="eyebrow">provenance</p>
            <p>
              pinned SHA-256 <Hash sha={p.file_sha256} n={16} />
            </p>
            <p>
              code_git_sha{" "}
              <span className={isCleanGitSha(p.code_git_sha) ? "ok" : "bad"}>
                <Hash sha={p.code_git_sha} n={12} /> {isCleanGitSha(p.code_git_sha) ? "clean" : "NOT CLEAN"}
              </span>
            </p>
            <p>
              cone fetch by{" "}
              {p.cone_cache_code_git_shas.map((s) => (
                <Hash key={s} sha={s} n={12} />
              ))}{" "}
              · cache <Hash sha={p.cone_cache_manifest_sha256} />
            </p>
            <p className="num small">{utcWindow(p.query_window_utc)}</p>
          </div>
          <div className="stat-card">
            <p className="eyebrow">source</p>
            <p className="small">{p.source_service}</p>
            <p className="small muted">{block.validation_detail}</p>
          </div>
        </div>

        <div className="cards2">
          <div>
            <p className="eyebrow">distance cuts (every sighting counted)</p>
            <table className="tbl tbl-compact">
              <tbody>
                {distanceCutRows(p.distance_cuts).map((r) => (
                  <tr key={r.label} className={kindClass(r.kind)}>
                    <td className={r.kind === "drop" || r.kind === "keep" ? "indent" : ""}>{r.label}</td>
                    <td className="r num">{fmtCount(r.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <p className="eyebrow">
              de-duplication against the 692 primaries {dedupCloses(mp) ? <span className="ok">✓ arithmetic closes</span> : <span className="bad">COUNTS DO NOT CLOSE</span>}
            </p>
            <table className="tbl tbl-compact">
              <tbody>
                {dedupRows(mp, d).map((r) => (
                  <tr key={r.label} className={kindClass(r.kind)}>
                    <td className={r.kind === "drop" || r.kind === "keep" ? "indent" : ""}>{r.label}</td>
                    <td className="r num">{fmtCount(r.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="caveat">
          <b>suspected_unmatched_primary = {d.n_suspected_unmatched_primary}</b> rows ({d.n_suspected_unmatched_primary_stars} distinct stars beside{" "}
          {d.n_primaries_with_a_suspected_counterpart} of the {d.n_primary_without_gaia_id} primaries without a Gaia id;{" "}
          {d.n_suspected_primaries_that_absorbed_nothing} of those primaries absorbed no sighting at all). {d.suspected_unmatched_primary_policy}.
          <details>
            <summary>list the {block.suspected_unmatched_primary.length} rows</summary>
            <div className="scroll-x">
              <table className="tbl tbl-compact">
                <thead>
                  <tr>
                    <th>primary (no Gaia id)</th>
                    <th>Gaia DR3 source_id</th>
                    <th className="r">separation [″]</th>
                    <th className="r">distance [pc]</th>
                    <th className="r">primary distance [pc]</th>
                    <th className="r">θ [′]</th>
                  </tr>
                </thead>
                <tbody>
                  {block.suspected_unmatched_primary.map((s) => (
                    <tr key={`${s.primary_target_name}-${s.gaia_dr3_source_id}`}>
                      <td>{s.primary_target_name}</td>
                      <td className="num">{gaiaIdText(s.gaia_dr3_source_id)}</td>
                      <td className="r num">{s.separation_arcsec.toFixed(2)}</td>
                      <td className="r num">{s.distance_pc.toFixed(2)}</td>
                      <td className="r num">{s.primary_distance_pc.toFixed(2)}</td>
                      <td className="r num">{s.theta_arcmin.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </div>
        <PointingAudit audit={block.pointing_centre_audit} />
        {caveat ? (
          <div className="caveat">
            <b>Pointing-centre audit:</b> {caveat} {band?.reading} {block.pointing_centre_audit.policy}.
            {band ? <span className="muted small"> Targets: {band.targets_no_counterpart.join(", ")}.</span> : null}
          </div>
        ) : null}

        <p className="eyebrow">scope — this is the method of Wlodarczyk-Sroka et al. (2020), not their catalogue</p>
        <div className="scroll-x">
          <table className="tbl tbl-compact">
            <thead>
              <tr>
                <th></th>
                <th>
                  retrieved here <TierBadge tier={p.scope.retrieved.data_tier} />
                </th>
                <th>
                  Wlodarczyk-Sroka et al. (2020) <TierBadge tier={p.scope.published.data_tier} />
                </th>
              </tr>
            </thead>
            <tbody>
              {scopeRows(p.scope).map((r) => (
                <tr key={r.field}>
                  <td className="muted">{r.field}</td>
                  <td className="num">{r.retrieved}</td>
                  <td className="num">
                    {r.published}
                    {r.caveat ? <span className="warn small"> ({r.caveat})</span> : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          {p.scope.statement} {p.scope.published.citation}.
        </p>
      </Panel>

      <Panel
        className="panel-illustrative"
        title="Realized vs predicted EIG of the background-field null"
        aside={<TierBadge tier="illustrative_calibration" label="ILLUSTRATIVE inputs in both modes" />}
      >
        <div className="scroll-x">
          <table className="tbl">
            <thead>
              <tr>
                <th>radio mode</th>
                <th className="r">
                  predicted [bits] <TierBadge tier={block.eig.predicted.data_tier} />
                </th>
                <th className="r">realized [bits]</th>
                <th className="r">realized / predicted</th>
                <th className="r">range FKG – Fréchet</th>
                <th className="r">without suspected primaries</th>
                <th className="r">gained from the observed null</th>
              </tr>
            </thead>
            <tbody>
              {eigRows(block.eig.comparison).map((r) => (
                <tr key={r.mode} className={r.illustrative ? "row-illustrative" : ""}>
                  <td>
                    {r.mode} <span className={r.physical ? "mode-tag mode-physical" : "mode-tag mode-unphysical"}>{r.label}</span>
                  </td>
                  <td className="r num">{r.predicted}</td>
                  <td className="r num">
                    <b>{r.realized}</b>
                  </td>
                  <td className="r num">{r.ratio}</td>
                  <td className="r num">{r.range}</td>
                  <td className="r num">{r.withoutSuspected}</td>
                  <td className="r num">{r.observedNull}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="statement">{eigVerdict(block.eig.comparison)}</p>
        <p className="muted small">
          Uniform prior over {block.eig.definition.prior.over.join(" / ")}; {block.eig.definition.geometry}; {block.eig.definition.outcome}. Predicted:{" "}
          {block.eig.predicted.population} ({fmtCount(block.eig.predicted.n_stars)} stars). Realized: {fmtCount(block.eig.n_background_stars)} Gaia DR3
          stars; “without suspected primaries” removes {block.eig.realized_excluding_suspected_unmatched_primary.n_stars_removed} stars as a diagnostic only.
        </p>
        <p className="caveat small">{block.eig.tier_note}</p>

        <p className="eyebrow">Harris–FKG / Fréchet bounds on P(no detection among the background stars | H)</p>
        <div className="scroll-x">
          <table className="tbl tbl-compact">
            <thead>
              <tr>
                <th>radio mode</th>
                <th>hypothesis</th>
                <th className="r">E[N reached]</th>
                <th className="r">E[detections]</th>
                <th className="r">Markov lower</th>
                <th className="r">Harris–FKG lower (used)</th>
                <th className="r">Fréchet upper</th>
              </tr>
            </thead>
            <tbody>
              {block.bounds.map((b) => (
                <tr key={`${b.radio_mode}-${b.hypothesis_id}`} className="row-illustrative">
                  <td>
                    {b.radio_mode} <span className={b.radio_mode_label === "physical" ? "mode-tag mode-physical" : "mode-tag mode-unphysical"}>{b.radio_mode_label}</span>
                  </td>
                  <td>{b.hypothesis_id}</td>
                  <td className="r num">{fmtNum(b.expected_reached, 5)}</td>
                  <td className="r num">{fmtNum(b.expected_detections, 5)}</td>
                  <td className="r num">{b.p_zero_markov_lower.toFixed(8)}</td>
                  <td className="r num">{b.p_zero_fkg_lower.toFixed(8)}</td>
                  <td className="r num">{b.p_zero_frechet_upper.toFixed(10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">{block.correlation_assumption}</p>
      </Panel>

      <Panel className="panel-archive" title="Off-axis EIRP_min attenuation" aside={<code className="small">{att.formula}</code>}>
        <p className="statement num">
          Background {attenuationHeadline(att)}. Beam response B(θ): median {att.validation_sample.beam_response.median.toFixed(3)}, minimum{" "}
          {att.validation_sample.beam_response.min.toFixed(3)} (cone edge = {att.beam_response_at_cone_edge}), from the exact validation sample of{" "}
          {fmtCount(att.validation_sample.n_stars)} stars.
        </p>
        <div className="cards2">
          <table className="tbl tbl-compact">
            <thead>
              <tr>
                <th>quantile</th>
                <th className="r">background distance [pc]</th>
                <th className="r">background EIRP_min</th>
                <th className="r">primary EIRP_min (on-axis)</th>
              </tr>
            </thead>
            <tbody>
              {(["p05", "p25", "p50", "p75", "p95"] as const).map((q) => (
                <tr key={q}>
                  <td className="num">{q}</td>
                  <td className="r num">{fmtCount(att.background_distance_pc[q])}</td>
                  <td className="r num">{fmtWatts(att.background_eirp_min_w[q])}</td>
                  <td className="r num">{fmtWatts(att.primary_eirp_min_w[q])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="tbl tbl-compact">
            <thead>
              <tr>
                <th>EIRP_min at or below</th>
                <th className="r">background stars (cumulative)</th>
              </tr>
            </thead>
            <tbody>
              {block.shells.eirp_min_cumulative_by_decade.map((s) => (
                <tr key={s.eirp_min_max_w}>
                  <td className="num">{s.eirp_min_max_w.toExponential(0)} W</td>
                  <td className="r num">{fmtCount(s.n_stars)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          {att.decomposition_of_the_median_ratio.note}. Shells exclude the 692 primaries (they are de-duplicated before binning), so the lowest-EIRP_min
          shells are nearly empty: the nearby, sensitive stars are the primaries themselves.
        </p>
        <Command text={block.commands.report} />
        <Command text={block.commands.merged_physical} />
      </Panel>
    </>
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
