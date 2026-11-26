"""Reproducible CLI entrypoints for the Fermi Paradox constraint desk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from fermi_desk.errors import FermiDeskError
from fermi_desk.inference.engine import CHANNELS, RunConfig, run_comparison, write_receipt
from fermi_desk.models.expansion import GEOMETRY_MODES
from fermi_desk.models.radio import RADIO_MODES
from fermi_desk.registry import load_factor_graph, run_validation

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
queue_app = typer.Typer(add_completion=False, no_args_is_help=True, help="BOED job queue.")
adversary_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Adversary runner.")
app.add_typer(queue_app, name="queue")
app.add_typer(adversary_app, name="adversary")
console = Console(width=150)

Root = Annotated[
    Path | None, typer.Option("--root", help="Project root holding data/ (default: auto-detect)")
]
DbOpt = Annotated[
    Path | None, typer.Option("--db", help="SQLite path (default runs/queue.sqlite).")
]
PAIR_DEFAULT = "frequent_slow_expanders,rare_fast_expanders"
IR_PAIR_DEFAULT = "quiet_expanders,partial_dyson_conspicuous"


def _pair(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise typer.BadParameter("expected two comma-separated hypothesis ids")
    return parts[0], parts[1]


def _fmt(x: float | None, spec: str = ".4g") -> str:
    return "-" if x is None else format(x, spec)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _table(title: str, cols: tuple[str, ...], first_no_wrap: bool = True) -> Table:
    t = Table(title=escape(title))
    for i, c in enumerate(cols):
        t.add_column(escape(c), no_wrap=first_no_wrap and i == 0, overflow="fold")
    return t


# ------------------------------------------------------------------- validate
@app.command()
def validate(root: Root = None) -> None:
    """Validate all YAML schemas, astropy units, catalog checksums, and provenance."""
    results = run_validation(root)
    table = Table(title="fermi-desk validate", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for r in results:
        table.add_row(r.name, "[green]OK[/]" if r.ok else "[red]FAIL[/]", escape(r.detail))
    console.print(table)
    n_fail = sum(not r.ok for r in results)
    console.print(f"{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        raise typer.Exit(code=1)


# -------------------------------------------------------------------- compare
def _print_bf(bf: dict, title: str) -> None:
    t = _table(title, ("term entering the Bayes factor", "log10 BF"))
    for k, v in bf["log10_bf_by_term"].items():
        mark = "  (illustrative tier)" if k in bf["illustrative_terms"] else ""
        t.add_row(escape(k + mark), f"{v:+.6g}")
    t.add_row("[bold]all shared terms[/]", f"[bold]{_fmt(bf['log10_bf'], '+.6g')}[/]")
    t.add_row(
        "shared terms excluding illustrative", _fmt(bf["log10_bf_excluding_illustrative"], "+.6g")
    )
    console.print(t)
    console.print(
        escape(
            f"  channels entered: {bf['channels_in_bf']} | Earth-unreached term: "
            f"{bf['earth_unreached_in_bf']} | excluded (not shared): {bf['excluded_channels']} | "
            f"data tiers: {bf['data_tiers_in_bf']}"
        )
    )


@app.command()
def compare(
    geometry: Annotated[str, typer.Option(help=f"One of {GEOMETRY_MODES} (required).")],
    radio_mode: Annotated[str, typer.Option(help=f"One of {RADIO_MODES} (required).")],
    seed: Annotated[int, typer.Option(help="RNG seed recorded in the receipt (required).")],
    channels: Annotated[str, typer.Option(help="Comma-separated channels.")] = ",".join(CHANNELS),
    condition_on_earth_unreached: Annotated[
        bool,
        typer.Option(
            "--earth-condition/--no-earth-condition",
            help="Condition on Earth being unreached (separately reported likelihood factor).",
        ),
    ] = True,
    n_mc: Annotated[int, typer.Option(help="Monte Carlo realisations per hypothesis.")] = 20000,
    pair: Annotated[str, typer.Option(help="Headline comparison 'A,B'.")] = PAIR_DEFAULT,
    ir_pair: Annotated[str, typer.Option(help="IR calibration sweep 'A,B'.")] = IR_PAIR_DEFAULT,
    out_dir: Annotated[Path | None, typer.Option(help="Receipt directory (default runs/).")] = None,
    root: Root = None,
) -> None:
    """Headline hypothesis comparison; writes a JSON RunReceipt to runs/."""
    cfg = RunConfig(
        geometry=geometry,
        radio_mode=radio_mode,
        seed=seed,
        channels=tuple(c.strip() for c in channels.split(",") if c.strip()),
        condition_on_earth_unreached=condition_on_earth_unreached,
        n_mc=n_mc,
        headline_pair=_pair(pair),
        ir_pair=_pair(ir_pair),
    )
    graph = load_factor_graph(root)
    receipt = run_comparison(graph, cfg)
    res = receipt["results"]
    rows = list(res["hypotheses"].items())

    console.rule(
        f"fermi-desk compare | geometry={geometry} radio_mode={radio_mode} seed={seed} "
        f"earth_unreached={'on' if condition_on_earth_unreached else 'off'}"
    )
    t1 = _table(
        "Expansion & radio forward model (per scored hypothesis)",
        (
            "hypothesis",
            "tier",
            "P(Earth reached)",
            "E[N] uncond.",
            "E[N] model",
            "P0 exact (MC) +/- se",
            "P0 FKG lower",
            "P0 Markov lower",
            "IR P(star reached)",
            "local approx",
        ),
    )
    for hid, h in rows:
        ex, rd, ir = h.get("expansion") or {}, h.get("radio") or {}, h.get("infrared") or {}
        ir_cell = "-"
        if ir:
            ir_cell = _fmt(ir["p_star_reached"]) if "p_star_reached" in ir else "declared host"
        t1.add_row(
            hid,
            str(h["tier"]),
            _fmt(ex.get("p_earth_reached")),
            _fmt(rd.get("expected_reached_unconditional")),
            _fmt(rd.get("expected_reached")),
            f"{rd['p_zero_exact_mc']:.8f} +/- {rd['p_zero_exact_mc_se']:.1e}" if rd else "-",
            _fmt(rd.get("p_zero_independent_fkg_lower"), ".8f"),
            _fmt(rd.get("p_zero_markov_lower"), ".8f"),
            ir_cell,
            "-" if not ex else ("ok" if ex["local_approximation_valid"] else "VIOLATED"),
        )
    console.print(t1)

    names = ("earth_unreached", "radio", "solar_system", "radio+solar_system", "infrared")
    present = [c for c in names if any(c in h["components"] for _, h in rows)]
    for key, title, joint in (
        (
            "log10_lr_vs_null",
            "log10 LR vs null  =  log10 [P(data|H) / P(data|component null)]",
            True,
        ),
        (
            "log10_p_data_given_h",
            "raw log10 P(data | H)  (no joint shown: channel sets differ)",
            False,
        ),
    ):
        cols = ("hypothesis", *present, *(("joint LR [own channels only]",) if joint else ()))
        t = _table(title, cols)
        for hid, h in rows:
            vals = [
                _fmt(h["components"][c][key], ".6g") if c in h["components"] else "-"
                for c in present
            ]
            if joint:
                j = h["own_joint"]
                vals.append(escape(f"{j['log10_lr_vs_null']:.6g} [{' + '.join(j['terms'])}]"))
            t.add_row(hid, *vals)
        console.print(t)
    console.print(
        "  nulls: earth_unreached / radio / solar_system -> P(data|null) = 1, so raw = LR; "
        "infrared -> silent sky (all k flags are false positives). Joint LRs of rows with "
        "different channel sets are not comparable; use the Bayes factors below."
    )

    if "headline" in res:
        hl = res["headline"]
        _print_bf(hl, f"Headline: log10 BF [{hl['numerator']} : {hl['denominator']}]")

    if radio_mode == "distance_scaled_eirp":
        for hid, h in rows:
            rd = h.get("radio")
            if rd:
                m = rd["max_eirp_min_target"]
                console.print(
                    escape(
                        f"Largest EIRP_min target: {m['target_name']} ({m['dist_pc']:.1f} pc, "
                        f"{m['dist_source']}) EIRP_min={m['eirp_min_w']:.3e} W; "
                        f"p_detect|reached={m['p_detect_given_reached']:.3e} ({hid})"
                    )
                )
                break

    console.print("[bold]Tier-3 quarantine shelf[/] (no empirical exclusion scores assigned):")
    for q in res["tier3_quarantine"]:
        ma = q["monoculture_audit"]
        console.print(
            escape(
                f"  - {q['hypothesis_id']}: {q['quarantine_reason']} | monoculture audit: "
                f"compliance={ma['required_compliance_fraction']} over {ma['timescale_gyr']} Gyr; "
                f"{ma['defector_vulnerability']}"
            )
        )

    if "ir_calibration_sweep" in res:
        sw = res["ir_calibration_sweep"]
        console.print(
            "[bold]IR calibration reversal summary[/] "
            + escape(
                f"({sw['calibration_data_tier']}; NOT an observational bound): "
                f"LR[{sw['numerator']} : {sw['denominator']}] across {sw['n_calibrations']} "
                f"calibrations: min={sw['min_lr']:.4g} ({sw['min_lr_calibration']}), "
                f"max={sw['max_lr']:.4g} ({sw['max_lr_calibration']}); "
                f"preference reverses: {sw['preference_reverses']}"
            )
        )

    out = write_receipt(receipt, out_dir or graph.root / "runs")
    console.print(
        escape(
            f"Receipt: {_rel(out, graph.root)} | rng_seed={receipt['rng_seed']} | "
            f"git_sha={receipt['git_sha']} | data_tiers_used={receipt['data_tiers_used']}"
        )
    )


# ---------------------------------------------------------------- ir-drilldown
@app.command("ir-drilldown")
def ir_drilldown(
    seed: Annotated[int, typer.Option(help="RNG seed for the injection-recovery grid (required).")],
    n_trials: Annotated[int, typer.Option(help="Injections per grid cell.")] = 40,
    snr: Annotated[float, typer.Option(help="W3 SNR of the simulated sources.")] = 12.5,
    sigma_sys: Annotated[float, typer.Option(help="Registration floor (arcsec).")] = 0.5,
    pair: Annotated[str, typer.Option(help="LR numerator,denominator.")] = IR_PAIR_DEFAULT,
    out_dir: Annotated[
        Path | None, typer.Option(help="Receipt dir (default runs/ir_drilldown).")
    ] = None,
    root: Root = None,
) -> None:
    """Hephaistos blend-recovery drill-down (centroid vs. image-level screens)."""
    from fermi_desk.inference.ir_drilldown import run_ir_drilldown

    graph = load_factor_graph(root)
    rec = run_ir_drilldown(
        graph, seed=seed, pair=_pair(pair), n_trials=n_trials, snr=snr, sigma_sys_arcsec=sigma_sys
    )
    r = rec["results"]
    g = r["grid"]
    console.rule(
        f"fermi-desk ir-drilldown | seed={seed} | W3 SNR={snr} | sigma_sys={sigma_sys} arcsec | "
        f"{n_trials} injections/cell | tier={r['data_tier']}"
    )
    for key, title in (
        ("centroid_recovery", "Centroid screen: P(blend flagged) [rows: separation, cols: f_bg]"),
        ("image_recovery", "Image-level screen: P(blend flagged) [rows: separation, cols: f_bg]"),
    ):
        t = _table(title, ("theta", *[f"f_bg={f:.1f}" for f in g["flux_fractions"]]))
        for th, vals in zip(g["separations_arcsec"], g[key], strict=True):
            t.add_row(f"{th:.1f} arcsec", *[f"{v:.3f}" for v in vals])
        console.print(t)
    console.print(
        f"  empirical false-alarm rate (blend-free, n={g['n_null_trials']}): centroid "
        f"{g['centroid_false_alarm_empirical']:.4f}, image {g['image_false_alarm_empirical']:.4f} "
        f"(nominal {g['spec']['false_alarm_nominal']})"
    )
    s = r["screens_within_3p25_arcsec"]
    console.print(
        f"  blend survival within 3.25 arcsec (area-uniform theta, uniform f_bg): "
        f"centroid {s['centroid']['survival']:.3f}, image {s['image']['survival']:.3f}"
    )

    t = _table(
        "Published anchors: Hephaistos candidates D and E (W3 centroid screen)",
        (
            "cand.",
            "predicted shift",
            "measured offset",
            "tension",
            "p (Rayleigh)",
            "reproduced",
            "published",
        ),
    )
    for c in r["anchors"]["candidates"]:
        lo, hi = c["predicted_photocentre_shift_arcsec"]
        t.add_row(
            c["candidate"],
            f"{lo:.2f}-{hi:.2f} arcsec",
            f"{c['measured_w3_offset_arcsec']:.2f} +/- {c['measured_err_arcsec']:.2f} arcsec",
            f"{c['shift_tension_sigma']:.2f} sigma",
            f"{c['centroid_screen_p_value']:.3g}",
            c["reproduced_verdict"],
            c["published_verdict"],
        )
    console.print(t)
    console.print(escape(f"  anchor audit: {r['anchors']['status']} | {r['anchors']['summary']}"))

    lr = r["likelihood_ratio"]
    t = _table(
        f"LR [{lr['numerator']} : {lr['denominator']}] (k={lr['observed_candidates']}, "
        f"n={lr['n_targets']}, base completeness {lr['base_completeness_illustrative']} illustrative)",
        (
            "contamination scenario",
            "screen",
            "raw blend rate",
            "survival",
            "E[false pos.]",
            "completeness",
            "LR",
            "log10 LR",
        ),
    )
    for row in lr["rows"]:
        t.add_row(
            row["contamination_scenario"],
            row["screen"],
            f"{row['raw_blend_rate']:.3g}",
            f"{row['blend_survival']:.3f}",
            f"{row['expected_false_positives']:.3g}",
            f"{row['completeness']:.3f}",
            f"{row['lr']:.4g}",
            f"{row['log10_lr']:+.4g}",
        )
    console.print(t)
    out_dir = out_dir or graph.root / "runs" / "ir_drilldown"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{rec['run_id']}.json"
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    console.print(
        escape(
            f"Receipt: {_rel(out, graph.root)} | rng_seed={seed} | git_sha={rec['git_sha']} | "
            f"results_sha256={rec['results_sha256']}"
        )
    )


# ------------------------------------------------------------------ adversary
@adversary_app.command("run")
def adversary_run(
    seed: Annotated[int, typer.Option(help="RNG seed (required).")],
    pair: Annotated[str, typer.Option(help="Hypotheses for likelihood deltas.")] = PAIR_DEFAULT,
    n_mc: Annotated[int, typer.Option(help="MC realisations for likelihood deltas.")] = 4000,
    out_dir: Annotated[
        Path | None, typer.Option(help="Receipt dir (default runs/adversary).")
    ] = None,
    root: Root = None,
) -> None:
    """Run every adversary audit and write a receipt (exit 1 if any audit fails)."""
    from fermi_desk.adversary.checks import run_adversary, write_adversary_receipt

    graph = load_factor_graph(root)
    rec = run_adversary(graph, seed=seed, pair=_pair(pair), n_mc=n_mc)
    findings = rec["results"]["findings"]
    console.rule(f"fermi-desk adversary run | seed={seed}")
    t = _table("Adversary findings", ("check", "status", "severity", "summary"))
    colour = {"pass": "green", "finding": "yellow", "fail": "red"}
    for f in findings:
        t.add_row(
            f["check"],
            f"[{colour[f['status']]}]{f['status']}[/]",
            f["severity"],
            escape(f["summary"]),
        )
    console.print(t)

    d = next(f for f in findings if f["check"] == "hip1692_distance_revision")["details"]
    t = _table(
        f"{d['target_name']} EIRP_min = 4 pi d^2 F_min under each recorded distance "
        f"(F_min = {d['f_min_w_m2']:.4e} W/m^2)",
        (
            "distance label",
            "tier",
            "d [pc]",
            "EIRP_min [W]",
            "x legacy",
            f"> {d['threshold']:.0f}x",
        ),
    )
    for lb, v in d["distances"].items():
        tag = (
            " (canonical)"
            if lb == d["canonical_label"]
            else " (historical anchor)"
            if lb == d["historical_anchor_label"]
            else ""
        )
        t.add_row(
            lb + tag,
            v["data_tier"],
            f"{v['distance_pc']:.2f}",
            f"{v['eirp_min_w']:.4e}",
            f"{v['eirp_factor_vs_legacy']:.1f}",
            str(v["exceeds_threshold"]),
        )
    console.print(t)
    t = _table(
        "distance_scaled_eirp radio log10 LR vs null by distance "
        "(deterministic FKG bound; MC +/- se for reference)",
        (
            "hypothesis",
            "distance label",
            "log10 LR (FKG)",
            "delta vs canonical",
            "log10 LR (MC)",
            "target p_detect|reached",
        ),
    )
    for hid, by in d["radio_likelihood_by_distance"].items():
        for lb, v in by.items():
            t.add_row(
                hid,
                lb,
                f"{v['log10_lr_radio_fkg_bound']:.8e}",
                f"{v['delta_log10_lr_fkg_vs_canonical']:+.3e}",
                f"{v['log10_lr_radio_mc']:.4e} +/- {v['mc_se_log10']:.1e}",
                f"{v['target_p_detect_given_reached']:.3e}",
            )
    console.print(t)
    pairs = next(f for f in findings if f["check"] == "duplicate_pointings")["details"][
        "beam_sharing_pairs"
    ]
    console.print(
        escape(
            f"Beam-sharing pairs ({len(pairs)}): "
            + ", ".join(f"{p['a']}/{p['b']} ({p['separation_arcmin']:.2f} arcmin)" for p in pairs)
        )
    )
    out = write_adversary_receipt(rec, out_dir or graph.root / "runs" / "adversary")
    console.print(
        escape(
            f"Receipt: {_rel(out, graph.root)} | rng_seed={seed} | git_sha={rec['git_sha']} | "
            f"results_sha256={rec['results_sha256']}"
        )
    )
    if any(f["status"] == "fail" for f in findings):
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------- queue
def _store(db: Path | None, root: Path | None):
    from fermi_desk.queue.store import JobStore

    graph = load_factor_graph(root)
    return graph, JobStore(db or graph.root / "runs" / "queue.sqlite")


def _fail(exc: Exception) -> None:
    console.print(f"[red]{escape(type(exc).__name__ + ': ' + str(exc))}[/]")
    raise typer.Exit(code=1) from exc


@queue_app.command("seed")
def queue_seed(db: DbOpt = None, root: Root = None) -> None:
    """Register all job definitions (idempotent) with their EIG and cost."""
    graph, store = _store(db, root)
    added = store.seed(graph)
    console.print(escape(f"Registered {len(added)} new job(s): {added}"))
    console.print(f"Queue: {_rel(store.path, graph.root)} ({len(store.list())} jobs total)")
    store.close()


@queue_app.command("list")
def queue_list(db: DbOpt = None, root: Root = None) -> None:
    """List jobs ranked by EIG per hour of compute + review."""
    _, store = _store(db, root)
    rows = store.list()
    t = _table(
        "BOED action queue (ranked by EIG bits / (cpu hours + reviewer hours))",
        (
            "job_id",
            "tier",
            "status",
            "EIG [bits]",
            "cpu [s]",
            "review [min]",
            "bits/hour",
            "output_sha256",
            "source",
        ),
    )
    for r in rows:
        t.add_row(
            r["job_id"],
            r["tier"],
            r["status"],
            f"{r['expected_eig_bits']:.4g}",
            f"{r['est_cpu_seconds']:.0f}",
            f"{r['est_reviewer_minutes']:.0f}",
            f"{r['utility_bits_per_hour']:.4g}",
            (r["output_sha256"] or "-")[:16],
            escape(r["source"]),
        )
    console.print(t)
    for r in rows:
        console.print(escape(f"  {r['job_id']}: prior={r['eig_prior_json']} | {r['eig_basis']}"))
    store.close()


@queue_app.command("claim")
def queue_claim(
    job_id: str,
    by: Annotated[str, typer.Option(help="Claimant name.")],
    db: DbOpt = None,
    root: Root = None,
) -> None:
    """Claim an open job."""
    _, store = _store(db, root)
    try:
        store.claim(job_id, by)
    except FermiDeskError as exc:
        _fail(exc)
    console.print(f"Claimed {job_id} by {by}")
    store.close()


@queue_app.command("run")
def queue_run(
    job_id: str,
    seed: Annotated[int, typer.Option(help="RNG seed (required).")],
    db: DbOpt = None,
    root: Root = None,
) -> None:
    """Execute a claimed Tier-0 job offline and record output_sha256."""
    graph, store = _store(db, root)
    try:
        out = store.run(graph, job_id, seed, graph.root / "runs" / "queue")
    except FermiDeskError as exc:
        _fail(exc)
    console.print(
        escape(
            f"Completed {job_id} | rng_seed={seed} | code_git_sha={out['code_git_sha']} | "
            f"output_sha256={out['output_sha256']} | output={_rel(Path(out['output_path']), graph.root)}"
        )
    )
    _summarise_job(job_id, out["results"])
    store.close()


@queue_app.command("replay")
def queue_replay(job_id: str, db: DbOpt = None, root: Root = None) -> None:
    """Re-execute a completed job with its recorded parameters and seed; hashes must match."""
    graph, store = _store(db, root)
    try:
        out = store.replay(graph, job_id, graph.root / "runs" / "queue")
    except FermiDeskError as exc:
        _fail(exc)
    console.print(
        escape(
            f"Replayed {job_id} | rng_seed={out['rng_seed']} | "
            f"recorded output_sha256={out['recorded_output_sha256']} | "
            f"replay output_sha256={out['output_sha256']} | MATCH={out['matches_recorded']}"
        )
    )
    store.close()


@queue_app.command("reject")
def queue_reject(
    job_id: str,
    reason: Annotated[str, typer.Option(help="Reason for rejection.")],
    db: DbOpt = None,
    root: Root = None,
) -> None:
    """Reject an open or claimed job."""
    _, store = _store(db, root)
    try:
        store.reject(job_id, reason)
    except FermiDeskError as exc:
        _fail(exc)
    console.print(f"Rejected {job_id}: {reason}")
    store.close()


def _summarise_job(job_id: str, res: dict) -> None:
    """Human-readable summary of a completed job's deterministic output."""
    if "thresholds" in res and "rows" in res and "t_grid_gyr" in res:
        a, b = res["pair"]
        for hid, th in res["thresholds"].items():
            console.print(
                f"  T_valid({hid}) = 8 kpc / v_exp = {th['t_valid_gyr']:.6e} Gyr "
                f"({th['t_valid_gyr'] * 1e6:.6g} kyr)"
            )
        t = _table(
            f"{job_id}: A = {a}, B = {b}",
            (
                "geometry",
                "T [Gyr]",
                "P(E reached) A",
                "P(E reached) B",
                "E[N|E unr] A",
                "E[N|E unr] B",
                "log10 BF radio",
                "log10 BF joint",
                "joint excl. illustr.",
                "local A",
                "local B",
            ),
        )
        for r in res["rows"]:
            ha, hb = r["hypotheses"][a], r["hypotheses"][b]
            t.add_row(
                r["geometry"],
                f"{r['lookback_gyr']:.3e}",
                f"{ha['p_earth_reached']:.4g}",
                f"{hb['p_earth_reached']:.4g}",
                f"{ha['expected_reached_given_earth_unreached']:.4g}",
                f"{hb['expected_reached_given_earth_unreached']:.4g}",
                _fmt(r["log10_bf_radio"], "+.4g"),
                _fmt(r["log10_bf_joint"], "+.4g"),
                _fmt(r["log10_bf_joint_excluding_illustrative"], "+.4g"),
                ha["local_approx"],
                hb["local_approx"],
            )
        console.print(t)
    else:
        console.print(escape(f"  results keys: {sorted(res)}"))


if __name__ == "__main__":
    app()
