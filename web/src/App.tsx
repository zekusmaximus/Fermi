import { useState } from "react";
import { get, type Health } from "./api";
import { Hash, useApi } from "./components/ui";
import { isCleanGitSha } from "./lib/format";
import { AdversaryShelf } from "./views/AdversaryShelf";
import { InfraredDrilldown } from "./views/InfraredDrilldown";
import { MacroDesk, type View } from "./views/MacroDesk";
import { QueueView } from "./views/QueueView";
import { RadioDrilldown } from "./views/RadioDrilldown";

const NAV: { id: View; label: string }[] = [
  { id: "macro", label: "Macro Desk" },
  { id: "radio", label: "Radio" },
  { id: "infrared", label: "Infrared" },
  { id: "adversary", label: "Adversary & Quarantine" },
  { id: "queue", label: "Action Queue" },
];

export function App() {
  const [view, setView] = useState<View>("macro");
  const [seed, setSeed] = useState(42);
  const health = useApi(() => get<Health>("/api/health"), []);
  const h = health.data;
  return (
    <div className="app">
      <header className="prov" aria-label="provenance">
        <div className="brand">
          <span className="brand-mark" aria-hidden>◎</span> Fermi Constraint Desk
        </div>
        <div className="prov-items">
          <span>
            git{" "}
            <span className={isCleanGitSha(h?.git_sha) ? "" : "warn"} title={h?.git_head}>
              <Hash sha={h?.git_sha} />
            </span>
          </span>
          <span className={h ? (h.validation_passed ? "valid-ok" : "valid-bad") : ""}>
            {h ? `${h.validation_checks_passed}/${h.validation_checks_total} ${h.validation_passed ? "OK" : "FAIL"}` : "validating…"}
          </span>
          <span>
            catalog <Hash sha={h?.catalog_sha256} />
          </span>
          <span>
            WISE σ_sys <Hash sha={h?.wise_calibration_sha256} />
          </span>
          <span>
            rng_seed <span className="num">{seed}</span>
          </span>
          <span className="muted">receipt v{h?.receipt_schema_version ?? "—"} · offline</span>
        </div>
      </header>
      {health.error ? <div className="error-box">⚠ API unreachable — start it with `uv run fermi-desk serve`</div> : null}
      <nav className="nav">
        {NAV.map((n) => (
          <button key={n.id} className={view === n.id ? "nav-btn on" : "nav-btn"} onClick={() => setView(n.id)}>
            {n.label}
          </button>
        ))}
      </nav>
      <main>
        {view === "macro" ? <MacroDesk seed={seed} setSeed={setSeed} go={setView} /> : null}
        {view === "radio" ? <RadioDrilldown /> : null}
        {view === "infrared" ? <InfraredDrilldown seed={seed} /> : null}
        {view === "adversary" ? <AdversaryShelf seed={seed} /> : null}
        {view === "queue" ? <QueueView /> : null}
      </main>
      <footer className="foot muted small">
        Every number is computed by the local fermi-desk engine (no network). Tier colours: <span className="tier tier-published">published_catalog</span>{" "}
        <span className="tier tier-archive">reproduced_archive</span> <span className="tier tier-illustrative">illustrative_calibration</span>
      </footer>
    </div>
  );
}
