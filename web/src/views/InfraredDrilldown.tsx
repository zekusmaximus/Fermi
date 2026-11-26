import { useState } from "react";
import { get, type FalseAlarm, type IrDrill } from "../api";
import { ErrorBox, Hash, Loading, Panel, TierBadge, useApi } from "../components/ui";
import { cellColor, fmtNum, fmtPct } from "../lib/format";

const LEGACY = "legacy_baseline";
const MEASURED = "measured_reproduced_archive";
const SCEN = {
  suazo: "suazo_2024_subsample_convention",
  zack: "zackrisson_2026_full_sample_critique",
} as const;

export function InfraredDrilldown({ seed }: { seed: number }) {
  const { data, error, loading } = useApi(() => get<IrDrill>(`/api/infrared/drilldown?seed=${seed}`), [seed]);
  if (error) return <ErrorBox error={error} />;
  if (loading || !data) return <Loading what={`IR blend-recovery grids (seed ${seed})`} />;
  return (
    <div className="view">
      <p className="eyebrow">
        Infrared drill-down · Project Hephaistos · seed <span className="num">{data.rng_seed}</span> · results{" "}
        <Hash sha={data.results_sha256} />
      </p>
      <CalibrationCard d={data} />
      <HeatmapPanel d={data} />
      <AnchorCard d={data} />
      <FlipPanel d={data} />
    </div>
  );
}

function Fa({ label, fa, note }: { label: string; fa: FalseAlarm; note?: string }) {
  return (
    <div className="fa">
      <p className="eyebrow">{label}</p>
      <p className="stat-big num">
        {fa.n_flagged}/{fa.n_stars}
      </p>
      <p className="num small">
        95% CP [{fmtPct(fa.clopper_pearson_95[0])}, {fmtPct(fa.clopper_pearson_95[1])}]
        {fa.ks_rayleigh_p_value !== null ? ` · KS p = ${fmtNum(fa.ks_rayleigh_p_value, 2)}` : ""}
        {fa.binomial_p_at_least_k_given_nominal !== null && fa.n_flagged > 0
          ? ` · P(≥k | nominal) = ${fmtNum(fa.binomial_p_at_least_k_given_nominal, 2)}`
          : ""}
      </p>
      {note ? <p className="muted small">{note}</p> : null}
    </div>
  );
}

function CalibrationCard({ d }: { d: IrDrill }) {
  const r = d.results;
  const b = r.small_sample_false_alarm_bounds;
  const ce = r.control_experiment;
  const floors = Object.fromEntries(r.registration_floors.map((f) => [f.label, f]));
  const meas = floors[MEASURED];
  const leg = floors[LEGACY];
  if (!b || !ce || !meas) {
    return (
      <Panel title="Empirical calibration">
        <p className="muted">No measured sigma_sys calibration is pinned; only the legacy floor is shown.</p>
      </Panel>
    );
  }
  return (
    <Panel
      title="Live empirical calibration — AllWISE W3 centroid registration floor"
      aside={<TierBadge tier="reproduced_archive" />}
      className="panel-archive"
    >
      <div className="cal-head">
        <div>
          <p className="eyebrow">measured floor (calibration half, n = {ce.n_calibration})</p>
          <p className="stat-big num">σ_sys = {meas.sigma_sys_arcsec.toFixed(3)}″</p>
          <p className="num small">
            68% bootstrap {meas.interval_68_arcsec ? `${meas.interval_68_arcsec[0].toFixed(3)}–${meas.interval_68_arcsec[1].toFixed(3)}″` : "—"}
          </p>
        </div>
        <div>
          <p className="eyebrow">legacy baseline</p>
          <p className="stat-big num">{leg ? leg.sigma_sys_arcsec.toFixed(3) : "—"}″</p>
          <p className="small">
            Suazo et al. 2024 §4 WISE astrometric precision <TierBadge tier={leg?.data_tier ?? null} />
          </p>
        </div>
        <div>
          <p className="eyebrow">control sample</p>
          <p className="stat-big num">
            {ce.n_controls} stars / {ce.n_patches_with_controls} patches
          </p>
          <p className="small num">
            |b| &gt; {ce.min_abs_b_deg}° · code <Hash sha={ce.code_git_sha} /> · file <Hash sha={ce.file_sha256} />
          </p>
        </div>
      </div>
      <div className="fa-row">
        <Fa label="withheld · measured floor (out-of-sample)" fa={b.withheld_at_measured_floor} />
        <Fa label="calibration half · in-sample tail" fa={b.calibration_in_sample_tail} note="tail hint, not a test" />
        <Fa label="withheld · legacy 0.5″ floor" fa={b.withheld_at_legacy_floor} />
        <Fa label="withheld · no floor" fa={b.withheld_with_no_floor} note="why a floor is needed" />
      </div>
      <p className="statement">{b.statement}</p>
      <div className="caveat">
        <b>Atlas smoothing caveat.</b> {r.atlas_smoothing_caveat}
      </div>
      <details>
        <summary>Patch breakdown ({ce.patches.length} patches)</summary>
        <table className="tbl tbl-compact">
          <thead>
            <tr>
              <th>patch</th>
              <th className="r">l [°]</th>
              <th className="r">b [°]</th>
              <th className="r">measured</th>
              <th className="r">cal / withheld</th>
              <th className="r">rejected (Gaia neighbour &lt; 15″)</th>
            </tr>
          </thead>
          <tbody>
            {ce.patches.map((p) => (
              <tr key={p.patch_id}>
                <td>{p.patch_id}</td>
                <td className="r num">{p.l_deg.toFixed(2)}</td>
                <td className="r num">{p.b_deg.toFixed(2)}</td>
                <td className="r num">{p.measured_controls}</td>
                <td className="r num">
                  {p.n_calibration} / {p.n_withheld}
                </td>
                <td className="r num">{p.rejected_gaia_neighbour_within_15arcsec}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </Panel>
  );
}

type Screen = "centroid" | "image";
type FloorView = "measured" | "legacy" | "delta";

function HeatmapPanel({ d }: { d: IrDrill }) {
  const r = d.results;
  const [screen, setScreen] = useState<Screen>("centroid");
  const [floor, setFloor] = useState<FloorView>(r.floor_shift ? "measured" : "legacy");
  const base = r.by_floor[LEGACY]!;
  const grid = (floor === "measured" ? r.by_floor[MEASURED] ?? base : base).grid;
  const key = screen === "centroid" ? "centroid_recovery" : "image_recovery";
  const values: number[][] =
    floor === "delta" && r.floor_shift
      ? screen === "centroid"
        ? r.floor_shift.centroid_recovery_delta
        : r.floor_shift.image_recovery_delta
      : grid[key];
  const mode = floor === "delta" ? "delta" : "prob";
  const fa = screen === "centroid" ? grid.centroid_false_alarm_empirical : grid.image_false_alarm_empirical;
  return (
    <Panel
      title="Blend-recovery heatmap — P(blend flagged)"
      aside={<TierBadge tier="illustrative_calibration" label="illustrative_calibration · synthetic injection-recovery" />}
      className="panel-illustrative"
    >
      <div className="controls controls-inline">
        <div className="seg">
          <button className={screen === "centroid" ? "on" : ""} onClick={() => setScreen("centroid")}>
            Centroid
          </button>
          <button className={screen === "image" ? "on" : ""} onClick={() => setScreen("image")}>
            Image-level (Δχ²)
          </button>
        </div>
        <div className="seg">
          <button disabled={!r.floor_shift} className={floor === "measured" ? "on" : ""} onClick={() => setFloor("measured")}>
            {fmtNum(r.by_floor[MEASURED]?.registration_floor.sigma_sys_arcsec, 3)}″ (Measured)
          </button>
          <button className={floor === "legacy" ? "on" : ""} onClick={() => setFloor("legacy")}>
            {base.registration_floor.sigma_sys_arcsec.toFixed(3)}″ (Legacy Suazo)
          </button>
          <button disabled={!r.floor_shift} className={floor === "delta" ? "on" : ""} onClick={() => setFloor("delta")}>
            Delta (+gain)
          </button>
        </div>
      </div>
      <div className="scroll-x">
        <table className="heat">
          <thead>
            <tr>
              <th>θ \ f_bg</th>
              {grid.flux_fractions.map((f) => (
                <th key={f} className="num">
                  {f.toFixed(1)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.separations_arcsec.map((th, i) => (
              <tr key={th}>
                <th className="num">{th.toFixed(1)}″</th>
                {(values[i] ?? []).map((v, j) => (
                  <td key={j} className="num" style={{ background: cellColor(v, mode) }} title={`θ=${th}″ f_bg=${grid.flux_fractions[j]}`}>
                    {mode === "delta" ? (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3) : v.toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small num">
        {grid.n_trials_per_cell} injections/cell (MC error ≈ ±{(Math.sqrt(0.25 / grid.n_trials_per_cell)).toFixed(2)} at p = 0.5) · W3 SNR{" "}
        {grid.spec.snr} · simulated blend-free false-alarm rate {fmtNum(fa, 3)} (nominal {grid.spec.false_alarm_nominal}) · both floors use
        the same seed, so deltas reflect σ_sys alone
        {r.floor_shift
          ? ` · mean shift ${screen}: ${fmtNum(screen === "centroid" ? r.floor_shift.centroid_recovery_mean_delta : r.floor_shift.image_recovery_mean_delta, 3)}`
          : ""}
      </p>
    </Panel>
  );
}

function AnchorCard({ d }: { d: IrDrill }) {
  const rows = d.results.anchor_benchmarks;
  const labels = d.results.registration_floors.map((f) => f.label);
  return (
    <Panel title="Candidates D & E — JWST/MIRI anchors vs the catalog centroid screen" aside={<TierBadge tier="published_catalog" />}>
      <div className="cards2">
        {rows.map((a) => (
          <div key={a.candidate} className="stat-card">
            <p className="eyebrow">
              Candidate {a.candidate} · Gaia DR3 <span className="num">{a.gaia_dr3_source_id}</span>
            </p>
            <p className="num">
              measured W3 offset {a.measured_w3_offset_arcsec.toFixed(2)} ± {a.measured_err_arcsec.toFixed(2)}″ · blend model predicts{" "}
              {a.predicted_photocentre_shift_arcsec[0].toFixed(2)}–{a.predicted_photocentre_shift_arcsec[1].toFixed(2)}″ (tension{" "}
              {a.shift_tension_sigma.toFixed(2)}σ)
            </p>
            <table className="tbl tbl-compact">
              <thead>
                <tr>
                  <th>floor</th>
                  <th className="r">offset / σ</th>
                  <th className="r">p (Rayleigh)</th>
                  <th>verdict</th>
                </tr>
              </thead>
              <tbody>
                {labels.map((lb) => {
                  const f = a.by_floor[lb];
                  if (!f) return null;
                  return (
                    <tr key={lb}>
                      <td>
                        {f.sigma_sys_arcsec.toFixed(3)}″ <span className="muted small">{lb === MEASURED ? "measured" : "legacy"}</span>
                      </td>
                      <td className="r num">{f.offset_significance_sigma.toFixed(2)}σ</td>
                      <td className="r num">{fmtNum(f.centroid_screen_p_value, 3)}</td>
                      <td>
                        {f.reproduced_verdict} {f.verdict_matches_published ? <span className="ok">= published</span> : <span className="bad">≠ published</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="small">
              {a.candidate === "D"
                ? `JWST/MIRI resolved a ~1″ background galaxy carrying ≥85% of the mid-IR flux. At WISE resolution (6.5″ FWHM) its photocentre pull is comparable to the measured offset, yet the offset is only ${(a.by_floor[MEASURED] ?? a.by_floor[LEGACY])?.offset_significance_sigma.toFixed(2) ?? "—"}σ against the screen's noise budget (p = ${fmtNum((a.by_floor[MEASURED] ?? a.by_floor[LEGACY])?.centroid_screen_p_value, 3)}), so the catalog centroid screen cannot flag it.`
                : "E shows a marginal centroid offset; its extended clumpy galaxy (~1″) is in >2σ tension with a point-blend photocentre model, as expected for an extended source."}{" "}
              Published verdict: <b>{a.published_verdict}</b>.
            </p>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function FlipPanel({ d }: { d: IrDrill }) {
  const f = d.results.contamination_flip;
  const lr = d.results.likelihood_ratio;
  const val = (row: Record<string, unknown>, k: string) => (row[k] as { lr: number } | undefined)?.lr;
  return (
    <Panel
      title={`Contamination convention flip — LR [${lr.numerator} : ${lr.denominator}]`}
      aside={<TierBadge tier="illustrative_calibration" label="illustrative completeness 0.5" />}
      className="panel-illustrative"
    >
      <p className="small">
        Both published conventions side by side, no preferred convention. {f.lr_reading}. k = {lr.observed_candidates} candidates among
        n = {lr.n_targets.toLocaleString("en-US")}.
      </p>
      <table className="tbl">
        <thead>
          <tr>
            <th>floor</th>
            <th>screen</th>
            <th className="r">
              Suazo (2024) subsample
              <br />
              <span className="muted small num">rate {fmtNum(f.scenarios[SCEN.suazo], 3)}</span>
            </th>
            <th className="r">
              Zackrisson (2026) full sample
              <br />
              <span className="muted small num">rate {fmtNum(f.scenarios[SCEN.zack], 3)}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {f.rows.map((row) => (
            <tr key={`${row.registration_floor}-${row.screen}`}>
              <td>{row.registration_floor === MEASURED ? "measured" : "legacy"}</td>
              <td>{row.screen}</td>
              <td className="r num">{fmtNum(val(row, SCEN.suazo), 3)}</td>
              <td className="r num">{fmtNum(val(row, SCEN.zack), 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
