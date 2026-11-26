import { get, post, type Adversary, type CompareRequest, type CompareResponse, type Health, type Queue, type Registry } from "../api";
import { ErrorBox, TierBadge, TierSetBadge, useApi } from "../components/ui";
import { bfReading, fmtLog10, fmtMultiplier, fmtNum } from "../lib/format";
import { adversaryFacts, topArchivalJob } from "../lib/overview";
import type { View } from "./MacroDesk";

// The Overview's fixed reference run: the Macro Desk defaults at seed 42 (the same request
// body, so the server's compare cache is shared with the Macro Desk's first render).
const OVERVIEW_SEED = 42;
const DEFAULT_COMPARE: CompareRequest = {
  geometry_mode: "thin_disk_2d",
  radio_mode: "flat_probability",
  condition_on_earth_unreached: true,
  channels: ["radio", "infrared", "solar_system"],
  include_illustrative: false,
  seed: OVERVIEW_SEED,
};

const dash = "—";

export function Overview({ health, go }: { health: Health | null; go: (v: View) => void }) {
  const cmp = useApi(() => post<CompareResponse>("/api/compare", DEFAULT_COMPARE), []);
  const adv = useApi(() => get<Adversary>(`/api/adversary?seed=${OVERVIEW_SEED}`), []);
  const queue = useApi(() => get<Queue>("/api/queue"), []);
  const reg = useApi(() => get<Registry>("/api/registry"), []);

  const radioSurvey = reg.data?.surveys.find((s) => s.channel === "radio");
  // Live provenance straight from /api/health (read from the verified calibration file).
  const nControls = health?.n_controls ?? null;
  const facts = adversaryFacts(adv.data);
  const top = topArchivalJob(queue.data);
  const nOpen = queue.data ? [...queue.data.tier1_archival, ...queue.data.tier0_compute].filter((j) => j.status === "open").length : null;
  const head = cmp.data?.receipt.results.headline.headline;
  const reading = bfReading(head?.log10_bf);
  const nQuarantined = cmp.data?.receipt.results.tier3_quarantine.length ?? null;
  const nExcluded = cmp.data?.headline_exclusion_notes.filter((n) => n.held_out_of_reported_headline).length ?? 0;

  return (
    <div className="view overview">
      {/* 1 - hero strip */}
      <section className="hero">
        <p className="eyebrow">Overview · offline instrument · seed {OVERVIEW_SEED}</p>
        <h1>Fermi Paradox Empirical Constraint Desk</h1>
        <p className="hero-mission">
          This desk turns competing resolutions of the Fermi paradox into executable physical models and tests them
          against real survey catalogs, with every number tagged by where its data came from. Each verdict is audited
          adversarially by the desk itself and paired with a queue of the experiments that would shrink our ignorance
          the most per hour of effort.
        </p>
        <div className="hero-actions">
          <button className="btn btn-primary" onClick={() => go("macro")}>
            Open Macro Desk
          </button>
          <button className="btn" onClick={() => go("queue")}>
            View Open Experiments{nOpen !== null ? <span className="num muted"> · {nOpen} open</span> : null}
          </button>
        </div>
      </section>

      {/* 2 - live status telemetry */}
      <section className="telemetry" aria-label="live status">
        <article className="tele-card">
          <p className="eyebrow">Verified substrate</p>
          <p className="tele-big num">
            {radioSurvey ? radioSurvey.n_targets : dash} <span className="tele-unit">Breakthrough Listen targets</span>
          </p>
          <p>
            + <span className="num">{nControls ?? dash}</span> <span className="muted">AllWISE multi-patch controls</span>
          </p>
          <p className="tele-foot">
            {radioSurvey ? <TierBadge tier={radioSurvey.data_tier} /> : null}{" "}
            {health?.calibration_data_tier ? <TierBadge tier={health.calibration_data_tier} /> : null}{" "}
            <span className={health ? (health.validation_passed ? "valid-ok" : "valid-bad") : "muted"}>
              {health
                ? `${health.validation_checks_passed}/${health.validation_checks_total} ${health.validation_passed ? "OK" : "FAIL"}`
                : "validating…"}
            </span>
          </p>
        </article>

        <article className={`tele-card ${head?.policy === "verified_channels_only" ? "tele-verified" : ""}`}>
          <p className="eyebrow">Current flagship verdict</p>
          <p className="tele-big num">{head ? fmtLog10(head.log10_bf, 4) : cmp.loading ? "…" : dash}</p>
          <p className="small muted">
            HEADLINE log10 BF · {head ? <span className="policy policy-verified">{head.policy}</span> : null}{" "}
            {head ? <TierSetBadge tiers={head.data_tiers} /> : null}
          </p>
          {head ? (
            reading.kind === "null" ? (
              <p className="small">
                Verified data alone (<span className="num">{head.terms.join(" + ")}</span>) yields a{" "}
                <b>null separation</b> between frequent/slow and rare/fast expanders. Null results are first-class
                outputs here, not failures.
              </p>
            ) : (
              <p className="small">Verified headline reading: {reading.text}.</p>
            )
          ) : null}
          {nExcluded > 0 ? <p className="small muted">{nExcluded} illustrative channel(s) held out of this headline.</p> : null}
          {cmp.error ? <ErrorBox error={cmp.error} /> : null}
        </article>

        <article className="tele-card">
          <p className="eyebrow">Active adversary findings</p>
          <p className="tele-big num">
            {adv.data ? fmtMultiplier(facts.hipMultiplier) : adv.loading ? "…" : dash}{" "}
            <span className="tele-unit">HIP 1692 EIRP floor vs legacy</span>
          </p>
          <p>
            <span className="num">{facts.beamPairs ?? dash}</span> <span className="muted">beam-sharing pointing pairs (each counted once)</span>
          </p>
          <p>
            {facts.tValidKyr !== null ? <span className="badge-violated">{facts.tValidKyr.toFixed(1)} kyr</span> : dash}{" "}
            <span className="muted">local-disk validity cutoff{facts.tValidHypothesis ? ` (${facts.tValidHypothesis})` : ""}</span>
          </p>
          {adv.error ? <ErrorBox error={adv.error} /> : null}
        </article>

        <article className="tele-card">
          <p className="eyebrow">Top ignorance-shrinking target</p>
          <p className="tele-id">{top ? top.job_id : queue.loading ? "…" : dash}</p>
          <p className="tele-big num">
            {top ? fmtNum(top.expected_eig_bits, 4) : dash} <span className="tele-unit">expected bits</span>
          </p>
          <p className="small muted">
            Tier-1 archival · <span className="num">{top ? `${fmtNum(top.utility_score, 3)} bits/hr` : dash}</span>
          </p>
          {queue.error ? <ErrorBox error={queue.error} /> : null}
        </article>
      </section>

      {/* 3 - how the desk works */}
      <section className="how" aria-label="how the desk works">
        <article>
          <h3>Strict provenance</h3>
          <p className="how-badges">
            <TierBadge tier="published_catalog" /> <TierBadge tier="reproduced_archive" />{" "}
            <TierBadge tier="illustrative_calibration" />
          </p>
          <p className="small">
            Every dataset declares its tier. Illustrative inputs are drawn dashed amber and are held out of the verified
            headline unless you opt in. Tier-3 monoculture hypotheses (which need every civilization to behave alike
            forever) are quarantined with no empirical score
            {nQuarantined !== null ? <span className="num"> — {nQuarantined} quarantined in this run</span> : null}.
          </p>
        </article>
        <article>
          <h3>Adversarial self-auditing</h3>
          <p className="small">
            The desk attacks its own result: zero-emission invariance (a silent hypothesis must score a likelihood
            ratio of exactly 1), explicit 2D-vs-3D geometry checks, double-counting of beam-sharing pointings, and
            catalog-error audits such as the HIP 1692 distance revision
            {adv.data ? <span className="num"> — {facts.nFindings} checks in the latest receipt</span> : null}.
          </p>
        </article>
        <article>
          <h3>Active experiment queue (BOED)</h3>
          <p className="small">
            Tier-1 archival jobs are ranked by expected information gain per hour (bits/hr) and run only from a
            terminal against live archives. Tier-0 compute sweeps add no sky photons (0 bits by construction) and are
            ranked by diagnostic priority, with seeded, byte-for-byte replay.
          </p>
        </article>
      </section>

      {/* 4 - guided entry doors */}
      <section aria-label="start here">
        <p className="eyebrow">Start here</p>
        <div className="doors">
          <button className="door" onClick={() => go("macro")}>
            <span className="door-q">Can real survey data tell frequent, slow expanders from rare, fast ones?</span>
            <span className="door-to">Macro Desk →</span>
          </button>
          <div className="door door-split">
            <span className="door-q">Which stars were actually searched, and could a "Dyson sphere" just be a background galaxy?</span>
            <span className="door-links">
              <button className="btn btn-xs" onClick={() => go("radio")}>
                Radio drill-down →
              </button>
              <button className="btn btn-xs" onClick={() => go("infrared")}>
                Infrared drill-down →
              </button>
            </span>
          </div>
          <button className="door" onClick={() => go("adversary")}>
            <span className="door-q">Where does this analysis break, and which ideas can no data test at all?</span>
            <span className="door-to">Adversary &amp; Quarantine →</span>
          </button>
          <button className="door" onClick={() => go("queue")}>
            <span className="door-q">Which experiment would shrink our ignorance the most, per hour of work?</span>
            <span className="door-to">Action Queue →</span>
          </button>
        </div>
      </section>
    </div>
  );
}
