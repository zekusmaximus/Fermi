import { get, type Adversary, type Finding, type Registry } from "../api";
import { Command, ErrorBox, Hash, Loading, Panel, StatusPill, TierBadge, useApi } from "../components/ui";
import { decSignAudit, decSignHeadline, secondaryNote } from "../lib/adversary";
import { fmtCount, fmtDec, gaiaIdOrNone } from "../lib/background";

/**
 * v0.2.1c (ruling v0.2.1-R5): readable body of the `catalog_dec_sign_audit` finding. Tier
 * colours (DESIGN.md Sec. 1): the catalog values are published_catalog (emerald), the J2000
 * truth positions are reproduced_archive (cyan). Renders nothing for any other finding.
 */
function DecSignDetail({ finding }: { finding: Finding }) {
  const d = decSignAudit(finding);
  if (!d) return null;
  const note = secondaryNote(d);
  return (
    <div className="finding-detail">
      <p className="small">{decSignHeadline(d)}</p>
      <p className="small">
        Locked catalog <TierBadge tier="published_catalog" /> <Hash sha={d.locked_catalog.sha256} n={16} />{" "}
        <span className={d.locked_catalog.modified_by_this_audit ? "bad" : "ok"}>
          {d.locked_catalog.modified_by_this_audit ? "MODIFIED" : "unmodified (read-only)"}
        </span>{" "}
        · claim file <Hash sha={d.claim_sha256} n={16} /> · <span className="num">{d.ruling}</span>
      </p>
      <details>
        <summary>list the {fmtCount(d.affected_targets.length)} affected targets</summary>
        <div className="scroll-x">
          <table className="tbl tbl-compact">
            <thead>
              <tr>
                <th>target</th>
                <th>Gaia DR3 source_id</th>
                <th className="r">
                  catalog Dec <TierBadge tier="published_catalog" />
                </th>
                <th className="r">
                  true Dec (J2000) <TierBadge tier="reproduced_archive" />
                </th>
                <th className="r">offset</th>
                <th className="r">sign restored − truth [″]</th>
                <th>position from</th>
              </tr>
            </thead>
            <tbody>
              {d.affected_targets.map((t) => (
                <tr key={t.target_name}>
                  <td>{t.target_name}</td>
                  <td className="num">{gaiaIdOrNone(t.gaia_dr3_source_id)}</td>
                  <td className="r num">{fmtDec(t.catalog_dec_deg)}</td>
                  <td className="r num">{fmtDec(t.true_dec_deg)}</td>
                  <td className="r num">{t.offset_deg.toFixed(2)}°</td>
                  <td className="r num">{t.sign_restored_separation_arcsec.toFixed(2)}</td>
                  <td>{t.truth_source === "gaia_dr3" ? "Gaia DR3" : "Hipparcos-2"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <p className="small">
        Genuinely northern, never flipped: {d.northern_controls.map((c) => `${c.target_name} (${fmtDec(c.catalog_dec_deg)})`).join(", ")}.
      </p>
      {note ? <p className="small muted">{note}</p> : null}
    </div>
  );
}

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
                <DecSignDetail finding={f} />
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
