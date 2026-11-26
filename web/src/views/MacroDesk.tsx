import { useState } from "react";
import { post, type Channel, type CompareRequest, type CompareResponse, type Geometry, type RadioMode } from "../api";
import { ErrorBox, Hash, Loading, Panel, TierBadge, TierSetBadge, useApi } from "../components/ui";
import {
  ILLUSTRATIVE_BANNER,
  fmtLog10,
  fmtNum,
  isIllustrative,
  parseSeed,
  showIllustrativeBanner,
} from "../lib/format";

export type View = "macro" | "radio" | "infrared" | "adversary" | "queue";

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
  const [radioMode, setRadioMode] = useState<RadioMode>("flat_probability");
  const [earth, setEarth] = useState(true);
  const [channels, setChannels] = useState<Channel[]>(["radio", "infrared", "solar_system"]);
  const [illustrative, setIllustrative] = useState(false);
  const [seedText, setSeedText] = useState(String(seed));
  const [tab, setTab] = useState<"lr" | "raw">("lr");

  const req: CompareRequest = {
    geometry_mode: geometry,
    radio_mode: radioMode,
    condition_on_earth_unreached: earth,
    channels,
    include_illustrative: illustrative,
    seed,
  };
  const { data, error, loading } = useApi(
    () => post<CompareResponse>("/api/compare", req),
    [geometry, radioMode, earth, channels.join(","), illustrative, seed],
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
  const hl = res?.headline;
  const head = hl?.headline;
  const banner = showIllustrativeBanner(head?.policy, head?.data_tiers, illustrative);
  const hyps = res ? Object.values(res.hypotheses) : [];
  const present = COMPONENT_ORDER.filter((c) => hyps.some((h) => c in h.components));
  const colTiers = (c: string) =>
    [...new Set(hyps.flatMap((h) => h.components[c]?.data_tiers ?? []))].sort();

  return (
    <div className="view">
      <div className="flagship">
        <p className="eyebrow">Macro Desk · flagship question</p>
        <h1>
          Can Breakthrough Listen's 692-star survey separate frequent/slow expanders from
          rare/fast expanders—once we condition on Earth remaining unreached?
        </h1>
      </div>

      <div className="controls" role="group" aria-label="comparison controls">
        <Seg
          label="Geometry"
          value={geometry}
          options={["thin_disk_2d", "thick_disk_3d"]}
          onChange={(v) => setGeometry(v as Geometry)}
        />
        <Seg
          label="Radio mode"
          value={radioMode}
          options={["flat_probability", "distance_scaled_eirp"]}
          onChange={(v) => setRadioMode(v as RadioMode)}
          warn={radioMode === "distance_scaled_eirp" ? "luminosity function is illustrative" : undefined}
        />
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
          </div>
        </div>
        {loading ? <Loading what="Bayes factors" /> : null}
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
                    <td>{t}</td>
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
              {hyps.map((h) => (
                <tr key={h.hypothesis_id}>
                  <td>{h.hypothesis_id}</td>
                  <td className="num">{h.tier}</td>
                  <td className="r num">{fmtNum(h.expansion?.p_earth_reached)}</td>
                  <td className="r num">{fmtNum(h.radio?.expected_reached)}</td>
                  <td className="r num">
                    {h.radio ? `${fmtNum(h.radio.p_zero_exact_mc, 8)} ± ${fmtNum(h.radio.p_zero_exact_mc_se, 2)}` : "—"}
                  </td>
                  <td>
                    {!h.expansion ? (
                      "—"
                    ) : h.expansion.local_approximation_valid ? (
                      <span className="ok">ok</span>
                    ) : (
                      <span className="badge-violated" title="v_exp × T exceeds the 8 kpc infinite-disk validity radius">
                        LOCAL APPROX VIOLATED
                        {h.expansion.t_valid_kyr ? (
                          <span className="num"> · T_valid = {h.expansion.t_valid_kyr.toFixed(1)} kyr</span>
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
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          LR = P(data|H) / P(data|component null). For zero-detection radio / Solar System / Earth-unreached the null
          probability is exactly 1 (raw = LR); for the mid-IR count it is the silent-sky model. Tier-3 hypotheses are
          never scored here — see the Quarantine Shelf.
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
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  warn?: string | undefined;
}) {
  return (
    <div className="ctl">
      <span className="ctl-label">{label}</span>
      <div className="seg" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button key={o} role="radio" aria-checked={o === value} className={o === value ? "on" : ""} onClick={() => onChange(o)}>
            {o}
          </button>
        ))}
      </div>
      {warn ? <span className="ctl-warn">⚠ {warn}</span> : null}
    </div>
  );
}
