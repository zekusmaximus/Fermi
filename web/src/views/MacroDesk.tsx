import { useState } from "react";
import {
  post,
  type Channel,
  type CompareRequest,
  type CompareResponse,
  type DiskProfile,
  type Geometry,
  type PointingMergePolicy,
  type RadioMode,
} from "../api";
import { ErrorBox, Hash, HelpPopover, Loading, Panel, TierBadge, TierSetBadge, useApi } from "../components/ui";
import { fmtCount } from "../lib/background";
import {
  CORRELATION_CAVEAT,
  DEFAULT_MERGE_POLICY,
  EARTH_UNREACHED_HELP,
  MERGE_POLICY_CONTROL_LABEL,
  MERGE_POLICY_HELP,
  MERGE_POLICY_LABELS,
  MERGE_POLICY_OPTIONS,
  earthUnreachedNote,
  flatBackgroundGuardActive,
  isEarthTerm,
  mergePolicyCaption,
  mergePolicyRequest,
  mergeProvenanceRows,
  noteRulingTag,
} from "../lib/merge";
import {
  ILLUSTRATIVE_BANNER,
  fmtLog10,
  fmtNum,
  isIllustrative,
  parseSeed,
  showIllustrativeBanner,
} from "../lib/format";
import {
  DISK_PROFILE_LABELS,
  DISK_PROFILE_OPTIONS,
  GEOMETRY_OPTIONS,
  MARGINALIZE_LABEL,
  geometryRequest,
  localApproxBadge,
  lookbackSummary,
  showDiskProfile,
} from "../lib/geometry";

export type View = "overview" | "macro" | "radio" | "infrared" | "adversary" | "queue";

const COMPONENT_ORDER = ["earth_unreached", "radio", "solar_system", "radio+solar_system", "infrared"];
const CHANNELS: { id: Channel; label: string }[] = [
  { id: "radio", label: "Radio" },
  { id: "infrared", label: "Infrared" },
  { id: "solar_system", label: "Solar System" },
];

export function MacroDesk({
  seed,
  setSeed,
  go,
}: {
  seed: number;
  setSeed: (s: number) => void;
  go: (v: View) => void;
}) {
  const [geometry, setGeometry] = useState<Geometry>("thin_disk_2d");
  const [diskProfile, setDiskProfile] = useState<DiskProfile>("uniform");
  const [marginalize, setMarginalize] = useState(false);
  const [radioMode, setRadioMode] = useState<RadioMode>("flat_probability");
  const [mergePolicy, setMergePolicy] = useState<PointingMergePolicy>(DEFAULT_MERGE_POLICY);
  const [earth, setEarth] = useState(true);
  const [channels, setChannels] = useState<Channel[]>(["radio", "infrared", "solar_system"]);
  const [illustrative, setIllustrative] = useState(false);
  const [seedText, setSeedText] = useState(String(seed));
  const [tab, setTab] = useState<"lr" | "raw">("lr");

  const geo = geometryRequest({ geometry, diskProfile, marginalizeLookback: marginalize });
  const req: CompareRequest = {
    ...geo,
    radio_mode: radioMode,
    condition_on_earth_unreached: earth,
    channels,
    include_illustrative: illustrative,
    seed,
    ...mergePolicyRequest(mergePolicy),
  };
  const { data, error, loading } = useApi(
    () => post<CompareResponse>("/api/compare", req),
    [geometry, geo.disk_profile, marginalize, radioMode, mergePolicy, earth, channels.join(","), illustrative, seed],
  );

  const toggleChannel = (c: Channel) =>
    setChannels((cur) =>
      cur.includes(c) ? (cur.length > 1 ? cur.filter((x) => x !== c) : cur) : [...cur, c].sort(),
    );
  const commitSeed = () => {
    const s = parseSeed(seedText);
    if (s === null) setSeedText(String(seed));
    else setSeed(s);
  };

  const res = data?.receipt.results;
  const marg = data?.receipt.lookback_marginalization ?? null;
  const hl = res?.headline;
  const head = hl?.headline;
  const banner = showIllustrativeBanner(head?.policy, head?.data_tiers, illustrative);
  const hyps = res ? Object.values(res.hypotheses) : [];
  const present = COMPONENT_ORDER.filter((c) => hyps.some((h) => c in h.components));
  const colTiers = (c: string) =>
    [...new Set(hyps.flatMap((h) => h.components[c]?.data_tiers ?? []))].sort();
  // Merge-policy read-outs come from the receipt (what the server actually did), never
  // from the selected option alone.
  const radioOn = channels.includes("radio");
  const mergeProv = data?.receipt.merge_provenance ?? null;
  const bgField = data?.receipt.background_field ?? null;
  const heldOut = data?.receipt.held_out_extension_surveys ?? [];
  const mergeCaption = mergePolicyCaption({
    policy: mergePolicy,
    radioMode,
    radioOn,
    nBackground: bgField?.n_background_stars ?? null,
    heldOutSurveys: heldOut.map((h) => h.survey_id),
  });
  const guardActive = flatBackgroundGuardActive(mergePolicy, radioMode, radioOn);
  const earthNote = earthUnreachedNote({ geometry, diskProfile, marginalized: marg !== null });
  const bgHyps = hyps.filter((h) => h.radio?.background);

  return (
    <div className="view">
      <div className="flagship">
        <p className="eyebrow">Macro Desk · flagship question</p>
        <h1>
          Can Breakthrough Listen's 692-star survey separate frequent/slow expanders from
          rare/fast expanders—once we condition on Earth remaining unreached?
        </h1>
        <aside className="why" aria-label="why this question">
          <p className="eyebrow">Why this question?</p>
          <p>
            In a flat Galactic disk, doubling expansion speed while quartering civilization origin rates leaves the odds
            of reaching Earth identical—but predicts different numbers of settled neighbors around an unreached Earth.
            Use the controls below to test how much real survey data separates those two regimes, and how quickly
            illustrative assumptions can distort the verdict if mixed in.
          </p>
        </aside>
      </div>

      <div className="controls" role="group" aria-label="comparison controls">
        <Seg
          label="Geometry"
          value={geometry}
          options={[...GEOMETRY_OPTIONS]}
          onChange={(v) => setGeometry(v as Geometry)}
        />
        {showDiskProfile(geometry) ? (
          <Seg
            label="Disk profile"
            value={diskProfile}
            options={[...DISK_PROFILE_OPTIONS]}
            labels={DISK_PROFILE_LABELS}
            onChange={(v) => setDiskProfile(v as DiskProfile)}
          />
        ) : null}
        <Seg
          label="Radio mode"
          value={radioMode}
          options={["flat_probability", "distance_scaled_eirp"]}
          onChange={(v) => setRadioMode(v as RadioMode)}
          warn={radioMode === "distance_scaled_eirp" ? "luminosity function is illustrative" : undefined}
        />
        <div className="ctl ctl-wide">
          <span className="ctl-label">
            {MERGE_POLICY_CONTROL_LABEL}{" "}
            <HelpPopover label={MERGE_POLICY_CONTROL_LABEL} {...MERGE_POLICY_HELP} />
          </span>
          <div className="seg seg-tall" role="radiogroup" aria-label={MERGE_POLICY_CONTROL_LABEL}>
            {MERGE_POLICY_OPTIONS.map((o) => (
              <button
                key={o}
                role="radio"
                aria-checked={o === mergePolicy}
                className={o === mergePolicy ? "on" : ""}
                onClick={() => setMergePolicy(o)}
              >
                <span className="seg-plain">{MERGE_POLICY_LABELS[o]}</span>
                <span className="seg-enum">{o}</span>
              </button>
            ))}
          </div>
          <span className={guardActive ? "ctl-caption ctl-caption-warn" : "ctl-caption"} aria-live="polite">
            {mergeCaption}
          </span>
        </div>
        <div className="ctl">
          <span className="ctl-label">Conditioning &amp; channels</span>
          <div className="checks">
            <label>
              <input type="checkbox" checked={earth} onChange={(e) => setEarth(e.target.checked)} /> Earth
              unreached
            </label>
            {CHANNELS.map((c) => (
              <label key={c.id}>
                <input
                  type="checkbox"
                  checked={channels.includes(c.id)}
                  disabled={channels.length === 1 && channels.includes(c.id)}
                  onChange={() => toggleChannel(c.id)}
                />{" "}
                {c.label}
              </label>
            ))}
          </div>
        </div>
        <div className="ctl">
          <span className="ctl-label">Seed</span>
          <input
            className="seed-input num"
            value={seedText}
            inputMode="numeric"
            aria-label="rng seed"
            onChange={(e) => setSeedText(e.target.value)}
            onBlur={commitSeed}
            onKeyDown={(e) => e.key === "Enter" && commitSeed()}
          />
        </div>
        <label className={`illus-toggle ${illustrative ? "on" : ""}`}>
          <input type="checkbox" checked={illustrative} onChange={(e) => setIllustrative(e.target.checked)} />
          <span>Include Illustrative Channels in Headline BF</span>
        </label>
        <label className={`marg-toggle ${marginalize ? "on" : ""}`}>
          <input type="checkbox" checked={marginalize} onChange={(e) => setMarginalize(e.target.checked)} />
          <span>{MARGINALIZE_LABEL}</span>
        </label>
      </div>

      {error ? <ErrorBox error={error} /> : null}

      <div className={`verdict ${banner ? "verdict-illustrative" : "verdict-verified"} ${loading ? "stale" : ""}`}>
        {banner ? <div className="banner-illustrative">{ILLUSTRATIVE_BANNER}</div> : null}
        <div className="verdict-main">
          <div>
            <p className="eyebrow">HEADLINE log10 BF</p>
            <p className="verdict-number num">{head ? fmtLog10(head.log10_bf, 5) : "…"}</p>
            <p className="muted">
              {hl ? (
                <>
                  <b>{hl.numerator}</b> : <b>{hl.denominator}</b>
                </>
              ) : null}
            </p>
          </div>
          <div className="verdict-meta">
            <div>
              policy{" "}
              <span className={`policy ${banner ? "policy-illustrative" : "policy-verified"}`}>
                {head?.policy ?? "verified_channels_only"}
              </span>
            </div>
            <div>
              terms entered:{" "}
              {(head?.terms ?? []).map((t) => (
                <span key={t} className="term">
                  {t} <TierSetBadge tiers={hl?.term_data_tiers[t] ?? []} />
                </span>
              ))}
            </div>
            <div className="muted small">
              cached: {data ? String(data.cached) : "—"} · receipt git <Hash sha={data?.receipt.git_sha} /> ·
              seed <span className="num">{data?.receipt.rng_seed ?? seed}</span>
            </div>
            <div className="muted small">
              {data ? lookbackSummary(marg) : "—"}
              {marg ? " · terms are chain-rule conditionals that sum to the joint marginal" : null}
            </div>
          </div>
        </div>
        {loading ? (
          <Loading what={marginalize ? "Bayes factors over the lookback grid (one full run per T node — may take minutes)" : "Bayes factors"} />
        ) : null}
        {(head?.headline_exclusion_notes ?? []).length > 0 ? (
          <ul className="notes">
            {head!.headline_exclusion_notes.map((n) => (
              <li key={n.channel}>
                <TierBadge tier="illustrative_calibration" label={n.channel} />{" "}
                <b>{n.held_out_of_reported_headline ? "held out of this headline" : "INCLUDED in this headline"}</b>
                {" — "}
                {n.reasons.join("; ")}
                {n.log10_bf_term !== null ? (
                  <span className="muted num"> (its term: {fmtLog10(n.log10_bf_term)})</span>
                ) : null}
                {noteRulingTag(n) ? <span className="ruling-tag"> ruling {noteRulingTag(n)}</span> : null}
                {n.merged_background ? (
                  <span className="muted small num">
                    {" "}
                    · {fmtCount(n.merged_background.n_primary_targets)} primaries (exact MC){" "}
                    {fmtLog10(n.merged_background.log10_bf_primary_targets_exact_mc)} +{" "}
                    {fmtCount(n.merged_background.n_background_stars)} background stars (FKG product){" "}
                    {fmtLog10(n.merged_background.log10_bf_background_shells_fkg_product)}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      {hl ? (
        <Panel title="Bayes-factor terms (shared channels)" aside={<span className="muted small">illustrative rows are hatched</span>}>
          <table className="tbl">
            <thead>
              <tr>
                <th>term</th>
                <th>data tier</th>
                <th className="r">log10 BF</th>
                <th>in headline</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(hl.log10_bf_by_term).map(([t, v]) => {
                const tiers = hl.term_data_tiers[t] ?? [];
                return (
                  <tr key={t} className={isIllustrative(tiers) ? "row-illustrative" : ""}>
                    <td>
                      {t}
                      {isEarthTerm(t) ? (
                        <>
                          {" "}
                          <HelpPopover label="earth_unreached model term" {...EARTH_UNREACHED_HELP} />
                          <span className="term-caption" aria-live="polite">
                            {earthNote}
                          </span>
                        </>
                      ) : null}
                    </td>
                    <td>
                      <TierSetBadge tiers={tiers} />
                    </td>
                    <td className="r num">{fmtLog10(v)}</td>
                    <td>{head?.terms.includes(t) ? "yes" : "no"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      ) : null}

      {mergeProv && bgField ? (
        <Panel
          className="panel-archive"
          title={`Merged radio targets — ${mergeProv.policy}`}
          aside={
            <span className="legend">
              <TierBadge tier={bgField.data_tier} /> background field <Hash sha={bgField.file_sha256} /> · git{" "}
              <Hash sha={bgField.code_git_sha} />
            </span>
          }
        >
          <div className="cards2">
            <table className="tbl tbl-compact">
              <tbody>
                {mergeProvenanceRows(mergeProv, bgField.dedup_audit.n_suspected_unmatched_primary_stars).map((r) => (
                  <tr key={r.label} className={r.flag ? "row-flag" : ""}>
                    <td>{r.label}</td>
                    <td className="r num">{fmtCount(r.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div>
              <table className="tbl tbl-compact">
                <thead>
                  <tr>
                    <th>background shells · hypothesis</th>
                    <th className="r">E[N reached]</th>
                    <th className="r">E[detections]</th>
                    <th className="r">P0 Harris–FKG lower</th>
                    <th className="r">P0 Fréchet upper</th>
                  </tr>
                </thead>
                <tbody>
                  {bgHyps.map((h) => (
                    <tr key={h.hypothesis_id} className={guardActive || radioMode === "distance_scaled_eirp" ? "row-illustrative" : ""}>
                      <td>{h.hypothesis_id}</td>
                      <td className="r num">{fmtNum(h.radio?.background?.expected_reached)}</td>
                      <td className="r num">{fmtNum(h.radio?.background?.expected_detections)}</td>
                      <td className="r num">{fmtNum(h.radio?.background?.p_zero_independent_fkg_lower, 8)}</td>
                      <td className="r num">{fmtNum(h.radio?.background?.p_zero_frechet_upper, 8)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small">{CORRELATION_CAVEAT}</p>
            </div>
          </div>
          <p className="muted small">{bgField.scope.statement}</p>
          <p className="caveat small">
            <b>suspected_unmatched_primary:</b> {bgField.dedup_audit.n_suspected_unmatched_primary} row(s),{" "}
            {bgField.dedup_audit.n_suspected_unmatched_primary_stars} distinct stars —{" "}
            {bgField.dedup_audit.suspected_unmatched_primary_policy}. Details in the Radio drill-down.
          </p>
        </Panel>
      ) : heldOut.length > 0 && radioOn ? (
        <p className="muted small held-out">
          Held out of this run:{" "}
          {heldOut.map((h) => (
            <span key={h.survey_id}>
              <code>{h.survey_id}</code> (extension of <code>{h.parent_pointing_survey_id}</code>){" "}
            </span>
          ))}
          — its stars are not in the radio term. Select “{MERGE_POLICY_LABELS.union_targets_by_pointing}” to merge them.
        </p>
      ) : null}

      <Panel
        title="Hypothesis comparison"
        aside={
          <div className="tabs">
            <button className={tab === "lr" ? "tab on" : "tab"} onClick={() => setTab("lr")}>
              log10 LR vs null
            </button>
            <button className={tab === "raw" ? "tab on" : "tab"} onClick={() => setTab("raw")}>
              raw log10 P(data | H)
            </button>
          </div>
        }
      >
        <div className="scroll-x">
          <table className="tbl">
            <thead>
              <tr>
                <th>hypothesis</th>
                <th>tier</th>
                <th className="r">P(Earth reached)</th>
                <th className="r">E[N | Earth unreached]</th>
                <th className="r">P0 (MC ± se)</th>
                <th>local approx</th>
                {present.map((c) => (
                  <th key={c} className="r">
                    {c}
                    <br />
                    <TierSetBadge tiers={colTiers(c)} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {hyps.map((h) => {
                const badge = localApproxBadge(h.expansion);
                return (
                <tr key={h.hypothesis_id}>
                  <td>{h.hypothesis_id}</td>
                  <td className="num">{h.tier}</td>
                  <td className="r num">{fmtNum(h.expansion?.p_earth_reached)}</td>
                  <td className="r num">{fmtNum(h.radio?.expected_reached)}</td>
                  <td className="r num">
                    {h.radio ? `${fmtNum(h.radio.p_zero_exact_mc, 8)} ± ${fmtNum(h.radio.p_zero_exact_mc_se, 2)}` : "—"}
                  </td>
                  <td>
                    {!badge ? (
                      "—"
                    ) : (
                      <span className={badge.className} title={badge.title}>
                        {badge.label}
                        {badge.tValidKyr !== null ? (
                          <span className="num"> · T_valid = {badge.tValidKyr.toFixed(1)} kyr</span>
                        ) : null}
                      </span>
                    )}
                  </td>
                  {present.map((c) => {
                    const comp = h.components[c];
                    const v = comp ? (tab === "lr" ? comp.log10_lr_vs_null : comp.log10_p_data_given_h) : null;
                    return (
                      <td key={c} className={`r num ${comp && isIllustrative(comp.data_tiers) ? "cell-illustrative" : ""}`}>
                        {comp ? fmtNum(v, 5) : "—"}
                      </td>
                    );
                  })}
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          LR = P(data|H) / P(data|component null). For zero-detection radio / Solar System / Earth-unreached the null
          probability is exactly 1 (raw = LR); for the mid-IR count it is the silent-sky model. Tier-3 hypotheses are
          never scored here — see the Quarantine Shelf.
          {marg
            ? " With the lookback marginalised, P(Earth reached) is prior-weighted over T, E[N] and P0 are weighted by the lookback posterior given Earth unreached, and each component is conditional on the components to its left."
            : null}
        </p>
      </Panel>

      <div className="quick">
        <button className="btn" onClick={() => go("radio")}>→ Radio drill-down</button>
        <button className="btn" onClick={() => go("infrared")}>→ Infrared drill-down</button>
        <button className="btn" onClick={() => go("adversary")}>→ Tier-3 Quarantine &amp; Adversary</button>
      </div>
    </div>
  );
}

function Seg({
  label,
  value,
  options,
  onChange,
  warn,
  labels,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  warn?: string | undefined;
  labels?: Record<string, string> | undefined;
}) {
  return (
    <div className="ctl">
      <span className="ctl-label">{label}</span>
      <div className="seg" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button key={o} role="radio" aria-checked={o === value} className={o === value ? "on" : ""} onClick={() => onChange(o)}>
            {labels?.[o] ?? o}
          </button>
        ))}
      </div>
      {warn ? <span className="ctl-warn">⚠ {warn}</span> : null}
    </div>
  );
}
