import { get, type Adversary, type Registry } from "../api";
import { Command, ErrorBox, Hash, Loading, Panel, StatusPill, useApi } from "../components/ui";

export function AdversaryShelf({ seed }: { seed: number }) {
  const adv = useApi(() => get<Adversary>(`/api/adversary?seed=${seed}`), [seed]);
  const reg = useApi(() => get<Registry>("/api/registry"), []);
  return (
    <div className="view">
      <p className="eyebrow">Adversary feed · seed <span className="num">{seed}</span></p>
      {adv.error ? <ErrorBox error={adv.error} /> : null}
      {adv.loading || !adv.data ? (
        <Loading what="adversary audits" />
      ) : (
        <Panel
          title={`Adversary checks (${adv.data.results.findings.length})`}
          aside={
            <span className="small">
              results <Hash sha={adv.data.results_sha256} /> · git <Hash sha={adv.data.git_sha} />
            </span>
          }
        >
          <div className="findings">
            {adv.data.results.findings.map((f) => (
              <article key={f.check} className={`finding finding-${f.status}`}>
                <header>
                  <StatusPill status={f.status} /> <span className={`sev sev-${f.severity}`}>{f.severity}</span> <b>{f.check}</b>
                </header>
                <p>{f.summary}</p>
                {f.citations.length ? <p className="muted small">{f.citations.join(" · ")}</p> : null}
                <Command text={f.repro_command} />
              </article>
            ))}
          </div>
        </Panel>
      )}

      <section className="quarantine" aria-label="Tier-3 quarantine shelf">
        <header>
          <h2>Tier-3 Quarantine Shelf</h2>
          <p className="muted">
            Unfalsifiable / monoculture resolutions. Physically separated from the empirical desk: they never receive likelihood
            ratios, Bayes factors, or exclusion scores.
          </p>
        </header>
        {reg.error ? <ErrorBox error={reg.error} /> : null}
        {reg.loading || !reg.data ? (
          <Loading what="registry" />
        ) : (
          <div className="cards2">
            {reg.data.hypotheses.tier_3.map((h) => (
              <article key={h.id} className="q-card">
                <span className="badge-quarantine">QUARANTINED — ZERO EMPIRICAL SCORE</span>
                <h3>{h.name}</h3>
                <p className="muted small">{h.id}</p>
                <p>{h.quarantine_reason}</p>
                {h.monoculture_audit ? (
                  <div className="mono-audit">
                    <p className="eyebrow">Monoculture fragility audit</p>
                    <dl>
                      <dt>required_compliance_fraction</dt>
                      <dd className="num">{h.monoculture_audit.required_compliance_fraction}</dd>
                      <dt>timescale_gyr</dt>
                      <dd className="num">{h.monoculture_audit.timescale_gyr}</dd>
                      <dt>defector_vulnerability</dt>
                      <dd>{h.monoculture_audit.defector_vulnerability}</dd>
                    </dl>
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
