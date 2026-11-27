"""Reproducible CLI entrypoints for the Fermi Paradox constraint desk."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from fermi_desk.errors import DoubleCountingError, FermiDeskError
from fermi_desk.inference.engine import (
    CHANNELS,
    HEADLINE_VERIFIED,
    ILLUSTRATIVE_TIERS,
    LOOKBACK_MARGINALIZATION,
    RunConfig,
    results_digest,
    run_comparison,
    write_receipt,
)
from fermi_desk.models.expansion import (
    DISK_PROFILES,
    FINITE_DISK,
    GEOMETRY_MODES,
    LOCAL_APPROX_OK,
    LOCAL_APPROX_VIOLATED,
)
from fermi_desk.models.radio import RADIO_MODES
from fermi_desk.registry import load_factor_graph, run_validation
from fermi_desk.schemas import POINTING_MERGE_POLICIES, REJECT_DUPLICATE_POINTING_GROUP

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
def _local_approx(ex: dict) -> str:
    """ok | VIOLATED (local infinite disks) | valid_finite_galaxy (finite_disk_3d)."""
    return ex.get("local_approx") or (
        LOCAL_APPROX_OK if ex["local_approximation_valid"] else LOCAL_APPROX_VIOLATED
    )


def _print_lookback_nodes(res: dict, pair: tuple[str, str]) -> None:
    """Per-T diagnostic table of a lookback-marginalised run (headline pair)."""
    a, b = (res["hypotheses"][h][LOOKBACK_MARGINALIZATION] for h in pair)
    t = _table(
        f"Lookback grid diagnostics: A = {pair[0]}, B = {pair[1]} (per-node, fixed T)",
        (
            "T [Gyr]",
            "prior w",
            "P(E reached) A",
            "P(E reached) B",
            "E[N] A",
            "E[N] B",
            "post. w A",
            "post. w B",
            "log10 BF verified @T",
            "local A",
            "local B",
        ),
    )
    for col in t.columns[-2:]:
        col.min_width = 12  # "valid_finite" / "_galaxy" folds on the underscore
    verified = res["headline"]["verified"]["terms"]
    for ra, rb in zip(a["per_node"], b["per_node"], strict=True):
        bf_t = sum(ra["log10_p_by_component"][c] - rb["log10_p_by_component"][c] for c in verified)
        t.add_row(
            f"{ra['lookback_gyr']:.3e}",
            f"{ra['prior_weight']:.4f}",
            _fmt(ra.get("p_earth_reached")),
            _fmt(rb.get("p_earth_reached")),
            _fmt(ra.get("expected_reached")),
            _fmt(rb.get("expected_reached")),
            f"{ra['posterior_weight']:.4f}",
            f"{rb['posterior_weight']:.4f}",
            f"{bf_t:+.4g}",
            ra.get("local_approx", "-"),
            rb.get("local_approx", "-"),
        )
    console.print(t)
    for hid in pair:
        chk = res["hypotheses"][hid][LOOKBACK_MARGINALIZATION].get("earth_term_quadrature_check")
        if chk:
            console.print(
                escape(
                    f"  {hid}: grid check log10 P(Earth unreached) = {chk['value']:.8g} "
                    f"({chk['n_points']} nodes) vs {chk['value_fine']:.8g} "
                    f"({chk['n_points_fine']} nodes); |diff| = {chk['abs_difference_dex']:.2e} dex"
                )
            )


def _print_background_field(bf: dict | None) -> None:
    """Provenance, exact N_bg and de-duplication counters of a merged background field."""
    if not bf:
        return
    da, cuts = bf["dedup_audit"], bf["distance_cuts"]
    console.print(
        escape(
            f"  background field {bf['survey_id']} [{bf['data_tier']}]: N_bg = "
            f"{bf['n_background_stars']} stars in {bf['n_cells']} shell cells over "
            f"{bf['n_pointings']} pointings (cone {bf['cone_radius_arcmin']:g} arcmin = FWHM/2, "
            f"d <= {cuts['d_max_pc']:g} pc); sha256 {bf['file_sha256']}; git "
            f"{bf['code_git_sha'][:12]}; fetched {bf['query_window_utc'][0][:19]}Z .. "
            f"{bf['query_window_utc'][1][:19]}Z"
        )
    )
    console.print(
        "[yellow]  suspected_unmatched_primary[/] "
        + escape(
            f"= {da['n_suspected_unmatched_primary']} row(s), "
            f"{da['n_suspected_unmatched_primary_stars']} distinct stars beside "
            f"{da['n_primaries_with_a_suspected_counterpart']} of the "
            f"{da['n_primary_without_gaia_id']} primaries without a Gaia id: "
            f"{da['suspected_unmatched_primary_policy']}"
        )
    )
    console.print(escape(f"  scope: {bf['scope']['statement']}"))


def _tier_tag(tiers: list[str]) -> str:
    bad = sorted(t for t in tiers if t in ILLUSTRATIVE_TIERS)
    return ", ".join(bad) if bad else "verified (" + (", ".join(tiers) or "model term") + ")"


def _print_bf(bf: dict, title: str) -> None:
    hl = bf.get("headline")
    t = _table(title, ("term", "data tier", "log10 BF", "in HEADLINE BF"))
    head_terms = set(hl["terms"]) if hl else set()
    for k, v in bf["log10_bf_by_term"].items():
        tiers = bf["term_data_tiers"].get(k, [])
        t.add_row(
            escape(k), escape(_tier_tag(tiers)), f"{v:+.6g}", "yes" if k in head_terms else "no"
        )
    if hl and hl["policy"] == HEADLINE_VERIFIED:
        for k, v in bf["verified"]["log10_bf_by_term"].items():
            if k not in bf["log10_bf_by_term"]:  # verified marginal replacing a joint term
                tiers = bf["term_data_tiers"].get(k, [])
                t.add_row(
                    escape(k + " (verified marginal)"), escape(_tier_tag(tiers)), f"{v:+.6g}", "yes"
                )
    t.add_row("all shared terms (incl. illustrative)", "-", _fmt(bf["log10_bf"], "+.6g"), "-")
    t.add_row("verified channels only", "-", _fmt(bf["verified"]["log10_bf"], "+.6g"), "-")
    console.print(t)
    if hl:
        if hl["policy"] == HEADLINE_VERIFIED:
            console.print(
                f"[bold green]HEADLINE log10 BF = {_fmt(hl['log10_bf'], '+.6g')}[/] "
                + escape(
                    f"(policy: {hl['policy']}; terms: {' + '.join(hl['terms']) or 'none'}; "
                    f"illustrative channels excluded: "
                    f"{hl['excluded_illustrative_channels'] or 'none'}; "
                    "use --include-illustrative to fold them in)"
                )
            )
        else:
            console.print(
                "[bold red]!!! --include-illustrative: ILLUSTRATIVE CHANNELS FOLDED INTO THE "
                "HEADLINE BF - NOT AN OBSERVATIONAL BOUND !!![/]"
            )
            console.print(
                escape(
                    f"HEADLINE log10 BF = {_fmt(hl['log10_bf'], '+.6g')} (policy: {hl['policy']}; "
                    f"terms: {' + '.join(hl['terms'])}; illustrative channels included: "
                    f"{hl['includes_illustrative_channels'] or 'none'})"
                )
            )
    for n in (hl or {}).get("headline_exclusion_notes", []):
        verb = "held out of" if n["held_out_of_reported_headline"] else "INCLUDED in"
        console.print(
            "[yellow]  headline_exclusion_note[/] "
            + escape(
                f"{n['channel']} ({verb} this headline; tiers {n['data_tiers']}): "
                + "; ".join(n["reasons"])
            )
        )
    console.print(
        escape(
            f"  channels shared: {bf['channels_in_bf']} | Earth-unreached term: "
            f"{bf['earth_unreached_in_bf']} | excluded (not shared): {bf['excluded_channels']} | "
            f"data tiers (all shared terms): {bf['data_tiers_in_bf']}"
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
    include_illustrative: Annotated[
        bool,
        typer.Option(
            "--include-illustrative",
            help=(
                "Fold illustrative_calibration channels into the headline Bayes factor "
                "(default: verified channels only). Output is labelled loudly."
            ),
        ),
    ] = False,
    disk_profile: Annotated[
        str,
        typer.Option(help=f"{FINITE_DISK} density profile, one of {DISK_PROFILES} (else ignored)."),
    ] = "uniform",
    marginalize_lookback: Annotated[
        bool,
        typer.Option(
            "--marginalize-lookback",
            help=(
                "Integrate the lookback window T over its registry log-uniform prior "
                "(50 kyr - 10 Myr) in likelihood space instead of fixing T (default: off)."
            ),
        ),
    ] = False,
    lookback_grid_points: Annotated[
        int, typer.Option(min=2, help="Gauss-Legendre nodes in ln T for --marginalize-lookback.")
    ] = 16,
    n_importance: Annotated[
        int, typer.Option(min=16, help=f"{FINITE_DISK} per-target importance samples.")
    ] = 16384,
    pointing_merge_policy: Annotated[
        str,
        typer.Option(
            help=(
                f"How surveys sharing a pointing group are combined, one of "
                f"{POINTING_MERGE_POLICIES}. The default refuses undeclared collisions and "
                "leaves declared background extensions out; keep_first is forbidden."
            )
        ),
    ] = REJECT_DUPLICATE_POINTING_GROUP,
    out_dir: Annotated[Path | None, typer.Option(help="Receipt directory (default runs/).")] = None,
    root: Root = None,
) -> None:
    """Headline hypothesis comparison; writes a JSON RunReceipt to runs/."""
    try:
        cfg = RunConfig(
            include_illustrative_in_headline=include_illustrative,
            geometry=geometry,
            radio_mode=radio_mode,
            seed=seed,
            channels=tuple(c.strip() for c in channels.split(",") if c.strip()),
            condition_on_earth_unreached=condition_on_earth_unreached,
            n_mc=n_mc,
            headline_pair=_pair(pair),
            ir_pair=_pair(ir_pair),
            disk_profile=disk_profile,
            marginalize_lookback=marginalize_lookback,
            lookback_grid_points=lookback_grid_points,
            n_importance=n_importance,
            pointing_merge_policy=pointing_merge_policy,
        )
        graph = load_factor_graph(root)
        receipt = run_comparison(graph, cfg)
    except DoubleCountingError as exc:  # includes MergePolicyError (keep_first)
        _fail(exc)
    res = receipt["results"]
    rows = list(res["hypotheses"].items())
    finite = geometry == FINITE_DISK
    marg = res.get(LOOKBACK_MARGINALIZATION)

    console.rule(
        f"fermi-desk compare | geometry={geometry} radio_mode={radio_mode} seed={seed} "
        f"earth_unreached={'on' if condition_on_earth_unreached else 'off'} | "
        f"headline={receipt['headline_policy']}"
        + (f" | disk_profile={disk_profile}" if finite else "")
        + (" | lookback=MARGINALIZED" if marg else "")
    )
    if marg:
        lo, hi = marg["prior"]["effective_bounds_gyr"]
        console.print(
            escape(
                f"Lookback T marginalised: log-uniform prior [{lo * 1e6:g} kyr, {hi * 1e3:g} Myr], "
                f"{marg['quadrature']['n_points']}-point {marg['quadrature']['rule']}, in "
                "likelihood space (logsumexp; log-probabilities are never averaged). Table "
                "values below are lookback-averaged; Bayes-factor terms are chain-rule "
                "conditionals that sum to the joint marginal."
            )
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
    t1.columns[-1].min_width = 12  # "valid_finite" / "_galaxy" folds on the underscore
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
            "-" if not ex else _local_approx(ex),
        )
    console.print(t1)
    if finite:
        for hid, h in rows:
            fd = (h.get("radio") or {}).get("finite_disk_reach")
            if not fd or marg:
                continue
            console.print(
                escape(
                    f"  {hid}: E[N] model = {h['radio']['expected_reached']:.5g} +/- "
                    f"{fd['expected_reached_se']:.2g} (importance-sampled, "
                    f"{fd['n_importance_samples']} samples); Earth cone Lambda = "
                    f"{h['expansion']['earth_lambda']:.6g} (quadrature) vs "
                    f"{fd['earth_lambda_importance_crosscheck']:.6g} +/- "
                    f"{fd['earth_lambda_importance_crosscheck_se']:.2g} (importance cross-check)"
                )
            )
        geo = next((h["expansion"]["finite_disk"] for _, h in rows if h.get("expansion")), None)
        if geo:
            console.print(
                escape(
                    f"  finite_disk_3d: R_gal = {geo['galactic_disk_radius_kpc']:g} kpc, Sun at "
                    f"(R, Z) = ({geo['solar_cylindrical_radius_kpc']:.5f}, "
                    f"{geo['solar_height_kpc']:.4f}) kpc, profile {geo['disk_profile']}: "
                    f"{geo['density_normalisation']}; local approx: valid_finite_galaxy "
                    "(no infinite-disk limit)"
                )
            )
    if marg and "headline" in res:
        pair_ids = (res["headline"]["numerator"], res["headline"]["denominator"])
        _print_lookback_nodes(res, pair_ids)
    for held in receipt["held_out_extension_surveys"]:
        console.print(
            "[yellow]  pointing extension held out[/] "
            + escape(f"{held['survey_id']} (parent {held['parent_pointing_survey_id']}): ")
            + escape(held["reason"])
        )
    mp = receipt["merge_provenance"]
    if mp:
        console.print(
            escape(
                f"  {mp['policy']} [{mp['pointing_group_id']}]: {' + '.join(mp['input_surveys'])} "
                f"-> {mp['n_merged_targets']} stars from {mp['n_raw_targets']} raw sightings "
                f"({mp['n_dedup_gaia_id']} Gaia-id + {mp['n_dedup_angular_fallback']} angular "
                f"de-duplications, {mp['n_multi_beam_resolved']} multi-beam stars kept at "
                "minimum offset); background shells use the independent conditional-reach "
                "product (FKG lower bound on P0), primaries the exact correlated MC"
            )
        )
        for hid, h in rows:
            bg = (h.get("radio") or {}).get("background")
            if bg:
                console.print(
                    escape(
                        f"  {hid}: background E[N reached] = {bg['expected_reached']:.5g}, "
                        f"E[detections] = {bg['expected_detections']:.5g}, log10 P0 (FKG lower) = "
                        f"{bg['log10_p_zero_independent_fkg_lower']:.6g}, Frechet upper P0 = "
                        f"{bg['p_zero_frechet_upper']:.8f} [{bg['data_tier']}]"
                    )
                )
        _print_background_field(receipt.get("background_field"))

    names = ("earth_unreached", "radio", "solar_system", "radio+solar_system", "infrared")
    present = [c for c in names if any(c in h["components"] for _, h in rows)]
    col_tiers = {
        c: sorted(
            {t for _, h in rows if c in h["components"] for t in h["components"][c]["data_tiers"]}
        )
        for c in present
    }
    headers = [
        f"{c} [{'ILLUSTRATIVE: ' if any(t in ILLUSTRATIVE_TIERS for t in col_tiers[c]) else ''}"
        f"{', '.join(col_tiers[c]) or 'model term'}]"
        for c in present
    ]
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
        cols = ("hypothesis", *headers, *(("joint LR [own channels only]",) if joint else ()))
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
            f"git_sha={receipt['git_sha']} | headline_policy={receipt['headline_policy']} | "
            f"headline_log10_bf={_fmt(receipt['headline_log10_bf'], '+.6g')} | "
            f"data_tiers_used={receipt['data_tiers_used']}"
        )
    )
    console.print(f"results_sha256={results_digest(receipt)}")


# ---------------------------------------------------------------- ir-drilldown
@app.command("ir-drilldown")
def ir_drilldown(
    seed: Annotated[int, typer.Option(help="RNG seed for the injection-recovery grid (required).")],
    n_trials: Annotated[int, typer.Option(help="Injections per grid cell.")] = 40,
    snr: Annotated[float, typer.Option(help="W3 SNR of the simulated sources.")] = 12.5,
    sigma_sys: Annotated[
        float, typer.Option(help="LEGACY baseline registration floor (arcsec).")
    ] = 0.5,
    pair: Annotated[str, typer.Option(help="LR numerator,denominator.")] = IR_PAIR_DEFAULT,
    out_dir: Annotated[
        Path | None, typer.Option(help="Receipt dir (default runs/ir_drilldown).")
    ] = None,
    root: Root = None,
) -> None:
    """Hephaistos blend-recovery drill-down under the legacy AND measured registration floors."""
    from fermi_desk.inference.ir_drilldown import LEGACY, MEASURED, run_ir_drilldown

    graph = load_factor_graph(root)
    rec = run_ir_drilldown(
        graph,
        seed=seed,
        pair=_pair(pair),
        n_trials=n_trials,
        snr=snr,
        legacy_sigma_sys_arcsec=sigma_sys,
    )
    r = rec["results"]
    labels = list(r["by_floor"])
    console.rule(
        f"fermi-desk ir-drilldown | seed={seed} | W3 SNR={snr} | {n_trials} injections/cell | "
        f"floors={labels} | grid tier={r['data_tier']}"
    )
    t = _table(
        "W3 centroid registration floors (side by side; same seed => common random numbers)",
        (
            "floor",
            "sigma_sys [arcsec]",
            "68% interval",
            "data tier",
            "withheld 3-sigma false alarms",
            "source",
        ),
    )
    meas = r["registration_floor_sources"]["measured_calibration"]
    for f in r["registration_floors"]:
        ci = f["interval_68_arcsec"]
        fa = "-"
        if meas:
            key = (
                "out_of_sample_false_alarm_at_calibrated_floor"
                if f["label"] == MEASURED
                else "false_alarm_at_legacy_baseline_floor"
            )
            w = meas["withheld_false_alarm"][key]
            if math.isclose(w["sigma_sys_arcsec"], f["sigma_sys_arcsec"], abs_tol=1e-12):
                lo, hi = w["clopper_pearson_95"]
                fa = f"{w['n_flagged']}/{w['n_stars']} = {w['rate']:.4f} (95% {lo:.4f}-{hi:.4f})"
        t.add_row(
            f["label"],
            f"{f['sigma_sys_arcsec']:.4f}",
            "-" if ci is None else f"{ci[0]:.3f}-{ci[1]:.3f}",
            f["data_tier"],
            fa,
            escape(f["source"]),
        )
    console.print(t)
    if meas is None:
        console.print(
            "  [yellow]No measured sigma_sys calibration pinned in the IR survey: legacy "
            "baseline only.[/]"
        )
    else:
        est = ", ".join(f"{k}={v:.4f}" for k, v in meas["sigma_sys_by_estimator_arcsec"].items())
        console.print(
            escape(
                f"  measured calibration: {meas['path']} sha256={meas['file_sha256'][:16]}... "
                f"git={meas['code_git_sha'][:12]} seed={meas['rng_seed']} "
                f"n_cal={meas['n_calibration']} n_withheld={meas['n_withheld']} | "
                f"calibration-half estimators (arcsec): {est}"
            )
        )

    g0 = r["by_floor"][labels[0]]["grid"]
    for key, name in (("centroid_recovery", "Centroid"), ("image_recovery", "Image-level")):
        for lb in labels:
            g = r["by_floor"][lb]["grid"]
            t = _table(
                f"{name} screen P(blend flagged) | {lb} (sigma_sys="
                f"{g['spec']['sigma_sys_arcsec']:.3f} arcsec) [rows: separation, cols: f_bg]",
                ("theta", *[f"f_bg={x:.1f}" for x in g0["flux_fractions"]]),
            )
            for th, vals in zip(g["separations_arcsec"], g[key], strict=True):
                t.add_row(f"{th:.1f} arcsec", *[f"{v:.3f}" for v in vals])
            console.print(t)
    sh = r["floor_shift"]
    if sh:
        t = _table(
            f"Shift of the centroid grid: {MEASURED} - {LEGACY} "
            f"(sigma_sys {sh['sigma_sys_arcsec'][0]:.3f} -> {sh['sigma_sys_arcsec'][1]:.3f} arcsec)",
            ("theta", *[f"f_bg={x:.1f}" for x in g0["flux_fractions"]]),
        )
        for th, vals in zip(g0["separations_arcsec"], sh["centroid_recovery_delta"], strict=True):
            t.add_row(f"{th:.1f} arcsec", *[f"{v:+.3f}" for v in vals])
        console.print(t)
        console.print(
            f"  mean recovery shift: centroid {sh['centroid_recovery_mean_delta']:+.4f}, image "
            f"{sh['image_recovery_mean_delta']:+.4f}; simulated false-alarm shift: centroid "
            f"{sh['centroid_false_alarm_delta']:+.4f}, image {sh['image_false_alarm_delta']:+.4f}; "
            f"blend-survival shift (<=3.25 arcsec): centroid "
            f"{sh['survival_delta']['centroid']:+.4f}, image {sh['survival_delta']['image']:+.4f}"
        )
    for lb in labels:
        b = r["by_floor"][lb]
        g = b["grid"]
        s = b["screens_within_3p25_arcsec"]
        console.print(
            f"  [{lb}] simulated blend-free false-alarm rate (n={g['n_null_trials']}): centroid "
            f"{g['centroid_false_alarm_empirical']:.4f}, image {g['image_false_alarm_empirical']:.4f} "
            f"(nominal {g['spec']['false_alarm_nominal']}); blend survival within 3.25 arcsec: "
            f"centroid {s['centroid']['survival']:.3f}, image {s['image']['survival']:.3f}"
        )

    gate = r.get("real_fits_gate2")
    if gate is None:
        console.print("  [yellow]No real-FITS Gate #2 result pinned in the IR survey.[/]")
    else:
        psf = gate["empirical_psf"]
        console.print(
            escape(
                f"  Gate #2 real AllWISE W3 FITS (reproduced_archive): {gate['path']} "
                f"sha256={gate['file_sha256'][:16]}... git={gate['code_git_sha'][:12]} "
                f"seed={gate['rng_seed']} | {gate['n_fetched_ok']}/{gate['n_requested']} cutouts, "
                f"{gate['n_calibration']} calibration / {gate['n_withheld']} withheld | empirical "
                f"PSF FWHM {psf['fwhm_gaussian_arcsec']:.3f} arcsec (nominal "
                f"{psf['nominal_fwhm_arcsec']}) | frozen Delta chi^2 threshold "
                f"{gate['frozen_screens']['delta_chi2_threshold']:.2f}"
            )
        )
        for m, sc in gate["screens"].items():
            t = _table(
                f"Real-FITS withheld {m} screen: Arm A P(blend flagged) [rows: separation, cols: "
                f"f_bg] | last row: Arm B matched centered-excess false rejection",
                ("theta", *[f"f_bg={x:.1f}" for x in gate["flux_fractions"]]),
            )
            for th, vals in zip(
                gate["separations_arcsec"], sc["arm_a_blend_recovery"], strict=True
            ):
                t.add_row(f"{th:.1f} arcsec", *[f"{v:.3f}" for v in vals])
            t.add_row("Arm B (theta=0)", *[f"{v:.3f}" for v in sc["arm_b_false_rejection"]])
            console.print(t)
            nf, pb = sc["null_false_alarm"], sc["arm_b_false_rejection_pooled"]
            console.print(
                f"  [{m}] withheld null false alarm {nf['k']}/{nf['n']} (95% CP "
                f"{nf['clopper_pearson_95'][1]:.3f} upper); blend survival within 3.25 arcsec "
                f"{sc['blend_survival_within_3p25_arcsec']:.3f}; centered excess rejected "
                f"{pb['k']}/{pb['n']} pooled = {pb['rate']:.3f}"
            )
        side = r["blend_survival_side_by_side"]
        t = _table(
            "Blend survival within 3.25 arcsec: real FITS (reproduced_archive) vs synthetic "
            f"Gaussian ({side['synthetic']['registration_floor']}, illustrative_calibration)",
            ("screen", "real FITS withheld", "synthetic Gaussian", "real Arm B rejected"),
        )
        for row in side["rows"]:
            b = row["real_fits_centered_excess_false_rejection"]
            t.add_row(
                row["screen"],
                f"{row['real_fits_survival']:.3f}",
                f"{row['synthetic_survival']:.3f}",
                f"{b['k']}/{b['n']} = {b['rate']:.3f}",
            )
        console.print(t)

    t = _table(
        "Published anchors: Hephaistos candidates D and E (W3 centroid screen) under each floor",
        (
            "cand.",
            "predicted shift",
            "measured offset",
            "tension",
            *[f"p / verdict [{lb}]" for lb in labels],
            "published",
        ),
    )
    per = [{c["candidate"]: c for c in r["by_floor"][lb]["anchors"]["candidates"]} for lb in labels]
    for cand, c in per[0].items():
        lo, hi = c["predicted_photocentre_shift_arcsec"]
        t.add_row(
            cand,
            f"{lo:.2f}-{hi:.2f} arcsec",
            f"{c['measured_w3_offset_arcsec']:.2f} +/- {c['measured_err_arcsec']:.2f} arcsec",
            f"{c['shift_tension_sigma']:.2f} sigma",
            *[
                f"{p[cand]['centroid_screen_p_value']:.3g} / {p[cand]['reproduced_verdict']}"
                for p in per
            ],
            c["published_verdict"],
        )
    console.print(t)
    for lb in labels:
        a = r["by_floor"][lb]["anchors"]
        console.print(escape(f"  anchor audit [{lb}]: {a['status']} | {a['summary']}"))

    lr = r["likelihood_ratio"]
    rows = {lb: r["by_floor"][lb]["likelihood_ratio_rows"] for lb in labels}
    t = _table(
        f"LR [{lr['numerator']} : {lr['denominator']}] (k={lr['observed_candidates']}, "
        f"n={lr['n_targets']}, base completeness {lr['base_completeness_illustrative']} "
        "illustrative); both contamination conventions side by side",
        (
            "contamination scenario",
            "screen",
            "raw blend rate",
            *[f"survival [{lb}]" for lb in labels],
            *[f"log10 LR [{lb}]" for lb in labels],
        ),
    )
    for i, row in enumerate(rows[labels[0]]):
        t.add_row(
            row["contamination_scenario"],
            row["screen"],
            f"{row['raw_blend_rate']:.3g}",
            *[f"{rows[lb][i]['blend_survival']:.3f}" for lb in labels],
            *[f"{rows[lb][i]['log10_lr']:+.4g}" for lb in labels],
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


# ------------------------------------------------------------ background-field
@app.command("background-field")
def background_field_cmd(
    pair: Annotated[str, typer.Option(help="EIG prior 'A,B' (uniform).")] = PAIR_DEFAULT,
    out_dir: Annotated[
        Path | None, typer.Option(help="Receipt dir (default runs/background_field).")
    ] = None,
    root: Root = None,
) -> None:
    """Promoted Gaia DR3 background field: provenance, de-duplication, realized vs predicted EIG."""
    from fermi_desk.inference.background_field import (
        attenuation_summary,
        bounds_table,
        load_background_field,
        pointing_centre_audit,
        shell_summary,
    )
    from fermi_desk.inference.engine import canonical_sha256, git_state
    from fermi_desk.queue import runners
    from fermi_desk.queue.store import JOB_DEFINITIONS

    graph = load_factor_graph(root)
    try:
        field = load_background_field(graph)
    except FermiDeskError as exc:
        _fail(exc)
    if field is None:
        console.print(
            "[yellow]No background field is promoted: no survey pins a background summary "
            "(scripts/fetch_wlodarczyk_sroka_288k.py --fetch, then --promote-from).[/]"
        )
        raise typer.Exit(code=1)
    job = next(j for j in JOB_DEFINITIONS if j.job_id == runners.BACKGROUND_JOB_ID)
    params = {**job.parameters, "pair": list(_pair(pair))}
    ev = runners.background_extension_evidence(graph, params)
    assert ev is not None
    prov, shells, att = field.summary(), shell_summary(field), attenuation_summary(graph, field)
    da, cuts, mp = prov["dedup_audit"], prov["distance_cuts"], prov["merge_provenance"]
    console.rule(
        f"fermi-desk background-field | {prov['survey_id']} | tier={prov['data_tier']} | "
        f"N_bg={prov['n_background_stars']}"
    )
    t = _table("Provenance (reproduced_archive; verified offline on load)", ("field", "value"))
    for k, v in (
        ("artifact", prov["path"]),
        ("file sha256 (pinned)", prov["file_sha256"]),
        ("content_sha256", prov["content_sha256"]),
        ("code_git_sha (assembled)", prov["code_git_sha"]),
        ("cone-fetch commits", ", ".join(prov["cone_cache_code_git_shas"])),
        ("cone cache manifest sha256", prov["cone_cache_manifest_sha256"]),
        ("archive query window (UTC)", " .. ".join(prov["query_window_utc"])),
        ("assembled (UTC)", prov["assembled_at_utc"]),
        ("source service", prov["source_service"]),
        (
            "pointings / beam",
            f"{prov['n_pointings']} pointings; FWHM {prov['beam_fwhm_arcmin']:g} arcmin, cone "
            f"radius {prov['cone_radius_arcmin']:g} arcmin (FWHM/2)",
        ),
    ):
        t.add_row(k, escape(str(v)))
    console.print(t)
    t = _table("Live fetch counts (exact)", ("stage", "count"))
    for k, v in (
        ("raw Gaia DR3 cone sightings", cuts["n_cone_sightings_raw"]),
        ("  dropped: parallax_over_error <= 1 or no parallax", cuts["n_dropped_invalid_parallax"]),
        (
            "  dropped: 1 < S/N < 5 without a Bailer-Jones distance",
            cuts["n_dropped_no_bailer_jones"],
        ),
        (f"  dropped: beyond {cuts['d_max_pc']:g} pc", cuts["n_dropped_beyond_d_max"]),
        ("  dropped: outside the cone on recheck", cuts["n_dropped_outside_cone_recheck"]),
        ("sightings with a distance", cuts["n_sightings_with_distance"]),
        ("    inverse parallax (S/N >= 5)", cuts["n_sightings_inverse_parallax"]),
        ("    Bailer-Jones 2021 geometric", cuts["n_sightings_bailer_jones_geo"]),
        ("raw rows merged (692 primaries + sightings)", mp["n_raw_targets"]),
        ("  de-duplicated by Gaia DR3 source_id", mp["n_dedup_gaia_id"]),
        (
            f"  de-duplicated by < {mp['angular_fallback_arcsec']:g} arcsec fallback",
            mp["n_dedup_angular_fallback"],
        ),
        ("  multi-beam stars kept at minimum theta", mp["n_multi_beam_resolved"]),
        ("  background sightings absorbed by primaries", mp["n_background_absorbed_by_primary"]),
        ("    matched by Gaia id", da["n_background_sightings_matched_to_primary_gaia_id"]),
        (
            "    matched by angular fallback",
            da["n_background_sightings_matched_to_primary_angular"],
        ),
        ("  angular links refused (two ids / two primaries)", mp["n_angular_links_refused"]),
        ("duplicate Gaia ids vs bl_692_targets.csv", da["n_duplicate_gaia_ids_vs_primary"]),
        ("duplicate Gaia ids within the background", da["n_duplicate_gaia_ids_within_background"]),
        ("unique background stars N_bg", prov["n_background_stars"]),
        ("shell cells", prov["n_cells"]),
        ("suspected_unmatched_primary rows", da["n_suspected_unmatched_primary"]),
        ("  distinct suspected stars (kept in shells)", da["n_suspected_unmatched_primary_stars"]),
        (
            "  primaries with a suspected counterpart",
            da["n_primaries_with_a_suspected_counterpart"],
        ),
        ("  primaries without a Gaia id", da["n_primary_without_gaia_id"]),
    ):
        t.add_row(k, str(v))
    console.print(t)
    console.print(
        "[yellow]  suspected_unmatched_primary[/] "
        + escape(da["suspected_unmatched_primary_policy"])
    )
    console.print(escape(f"  scope: {prov['scope']['statement']}"))
    pub = prov["scope"]["published"]
    lband = pub["n_stars_gbt_l_band"]
    console.print(
        escape(
            f"  published scope ({pub['citation']}): {pub['pointings']}; "
            f"{pub['n_stars_total']} stars; {pub['gaia_release']}; {pub['distances']}. GBT L+S "
            f"stars (Table 2): {pub['n_stars_gbt_l_plus_s_band']['value']}. GBT L-band only: "
            f"{lband['value']} ({'verified' if lband['verified_against_paper'] else 'UNVERIFIED'}: "
            f"{lband['source']})"
        )
    )
    audit = pointing_centre_audit(graph, field)
    dense = audit["densest_band_of_unfound_primaries"]
    line = (
        f"{audit['n_with_gaia_id_sighted_in_own_cone']} of {audit['n_with_gaia_id']} primaries "
        "with a Gaia id were sighted in their own cone; "
        f"{audit['n_no_counterpart_in_own_cone']} of {audit['n_primaries']} primaries have no "
        "Gaia counterpart there at all."
    )
    if dense:
        lo, hi = dense["declination_band_deg"]
        line += (
            f" {dense['n_no_counterpart']} of them lie in catalog Dec [{lo:+d}, {hi:+d}) deg "
            f"({dense['n_targets']} targets there, {dense['n_targets_in_mirror_band']} in the "
            f"mirror band; {dense['n_background_stars_in_those_pointings']} background stars = "
            f"{100 * dense['fraction_of_background_stars']:.2f}% of N_bg): {dense['reading']} "
            f"Targets: {', '.join(dense['targets_no_counterpart'])}. {audit['policy']}"
        )
    console.print("[yellow]  pointing-centre audit[/] " + escape(line))
    dq, eq = shells["distance_quantiles_pc"], shells["eirp_min_quantiles_w"]
    v = att["validation_sample"]
    console.print(
        escape(
            f"  shells: distance p05/p50/p95 = {dq['p05']:.0f} / {dq['p50']:.0f} / {dq['p95']:.0f} "
            f"pc; off-axis EIRP_min p05/p50/p95 = {eq['p05']:.2e} / {eq['p50']:.2e} / "
            f"{eq['p95']:.2e} W (primaries, on-axis: median "
            f"{att['primary_eirp_min_w']['p50']:.2e} W, max {att['primary_eirp_min_w']['max']:.2e} W)"
        )
    )
    dec = att["decomposition_of_the_median_ratio"]
    console.print(
        escape(
            f"  off-axis attenuation: median EIRP_min is "
            f"{att['median_eirp_min_ratio_background_over_primary']:.3g}x the primaries' "
            f"(distance term {dec['distance_term']:.3g}x, beam term {dec['beam_term']:.3g}x; "
            f"validation sample of {v['n_stars']}: median B(theta) = "
            f"{v['beam_response']['median']:.3f}, min {v['beam_response']['min']:.3f})"
        )
    )
    t = _table(
        "Realized vs predicted EIG [bits] (uniform prior over the pair; "
        f"{ev['definition']['geometry']}; ILLUSTRATIVE inputs in both modes - not a verified bound)",
        (
            "radio mode",
            "predicted (synthetic_proxy)",
            "realized (Gaia DR3 shells)",
            "realized / predicted",
            "range FKG..Frechet",
            "realized w/o suspected",
            "info from observed null",
        ),
    )
    for r in ev["comparison"]:
        lo, hi = r["realized_eig_bits_range_fkg_to_frechet"]
        t.add_row(
            f"{r['radio_mode']} ({r['label']})",
            f"{r['predicted_eig_bits']:.5g}",
            f"{r['realized_eig_bits']:.5g}",
            _fmt(r["realized_over_predicted"], ".3f"),
            f"{lo:.4g} .. {hi:.4g}",
            f"{r['realized_eig_bits_excluding_suspected_unmatched_primary']:.5g}",
            f"{r['realized_information_bits_from_observed_null']:.4g}",
        )
    console.print(t)
    console.print(escape(f"  {ev['tier_note']}"))
    t = _table(
        "Bounds on P(no detection among the background stars | H), given Earth unreached",
        (
            "radio mode",
            "hypothesis",
            "E[N reached]",
            "E[detections]",
            "Markov lower",
            "Harris-FKG lower (used)",
            "Frechet upper",
        ),
    )
    bounds = bounds_table(ev["realized"])
    for b in bounds:
        t.add_row(
            f"{b['radio_mode']} ({b['radio_mode_label']})",
            b["hypothesis_id"],
            f"{b['expected_reached']:.5g}",
            f"{b['expected_detections']:.5g}",
            f"{b['p_zero_markov_lower']:.8f}",
            f"{b['p_zero_fkg_lower']:.8f}",
            f"{b['p_zero_frechet_upper']:.10f}",
        )
    console.print(t)
    console.print(
        "  The exact P(no detection) lies between the Harris-FKG product and the Frechet upper "
        "bound; the sign of the effect on a Bayes factor is not fixed by the bounds."
    )
    git_sha, git_head = git_state(graph.root)
    results = {
        "provenance": prov,
        "shells": shells,
        "attenuation": att,
        "pointing_centre_audit": audit,
        "eig": ev,
        "bounds": bounds,
        "suspected_unmatched_primary": [
            r.model_dump() for r in field.record.dedup_audit.suspected_unmatched_primary
        ],
    }
    out_dir = out_dir or graph.root / "runs" / "background_field"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = canonical_sha256(results)
    out = out_dir / f"background_field-{sha[:10]}.json"
    out.write_text(
        json.dumps(
            {
                "kind": "background_field",
                "git_sha": git_sha,
                "git_head": git_head,
                "rng_seed": None,
                "deterministic": "no RNG: analytic reach probabilities over the pinned shells",
                "data_tier": prov["data_tier"],
                "results": results,
                "results_sha256": sha,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(
        escape(
            f"Receipt: {_rel(out, graph.root)} | rng_seed=none (deterministic) | git_sha={git_sha} "
            f"| results_sha256={sha}"
        )
    )


# ---------------------------------------------------------------------- serve
@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address (loopback by default).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="TCP port.")] = 8000,
    static_dir: Annotated[
        Path | None,
        typer.Option(help="Built Vite bundle to serve at / (default: web/dist if present)."),
    ] = None,
    read_only: Annotated[
        bool,
        typer.Option(
            "--read-only/--no-read-only",
            help="Refuse queue run/replay (HTTP 403). Required for any non-loopback --host.",
        ),
    ] = False,
    root: Root = None,
) -> None:
    """Serve the offline desk: /api/* plus the compiled web UI (Milestone 3)."""
    from fermi_desk.api import create_app, is_loopback_host

    if not is_loopback_host(host) and not read_only:
        # Run/Replay execute engine code synchronously in the request; exposed on a LAN or
        # public interface they are a CPU-exhaustion vector, so they must be switched off.
        typer.echo(
            f"Refusing to bind non-loopback host {host!r} in read-write mode: queue "
            "run/replay execute synchronously and would be exposed to the network. "
            "Pass --read-only for public/LAN binds (or use --host 127.0.0.1).",
            err=True,
        )
        raise typer.Exit(code=2)

    import uvicorn

    graph = load_factor_graph(root)
    static = static_dir or graph.root / "web" / "dist"
    if not static.is_absolute():
        static = (Path.cwd() / static).resolve()
    has_ui = (static / "index.html").is_file()
    console.print(
        escape(
            f"fermi-desk serve | http://{host}:{port} | api=/api/health | "
            f"mode={'read-only' if read_only else 'read-write (loopback)'} | "
            f"ui={'served from ' + _rel(static, graph.root) if has_ui else 'not built (npm run build)'}"
        )
    )
    app_ = create_app(graph.root, static_dir=static if has_ui else None, read_only=read_only)
    uvicorn.run(app_, host=host, port=port)


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
    before = {r["job_id"]: r["status"] for r in store.list()}
    added = store.seed(graph)
    console.print(escape(f"Registered {len(added)} new job(s): {added}"))
    for r in store.list_archival():
        if r["status"] == "completed" and before.get(r["job_id"]) != "completed":
            console.print(
                escape(
                    f"Completed by promoted artifact: {r['job_id']} -> {r['output_path']} "
                    f"(output_sha256 {r['output_sha256']}, code_git_sha {r['code_git_sha']})"
                )
            )
    console.print(f"Queue: {_rel(store.path, graph.root)} ({len(store.list())} jobs total)")
    store.close()


@queue_app.command("list")
def queue_list(db: DbOpt = None, root: Root = None) -> None:
    """List the Tier-1 archival queue (by EIG bits/hr) and Tier-0 compute queue (by priority)."""
    graph, store = _store(db, root)
    archival, compute = store.list_archival(), store.list_compute()
    t = _table(
        "Tier-1 Archival Queue (ranked by EIG bits / (cpu hours + reviewer hours))",
        (
            "rank",
            "job_id",
            "status",
            "EIG [bits]",
            "cpu [s]",
            "review [min]",
            "bits/hour",
            "archival script",
            "source",
        ),
    )
    for i, r in enumerate(archival, 1):
        t.add_row(
            str(i),
            r["job_id"],
            r["status"],
            f"{r['expected_eig_bits']:.4g}",
            f"{r['est_cpu_seconds']:.0f}",
            f"{r['est_reviewer_minutes']:.0f}",
            f"{r['utility_bits_per_hour']:.4g}",
            escape(r["archival_script"] or "-"),
            escape(r["source"]),
        )
    console.print(t)
    for r in archival:
        console.print(escape(f"  {r['job_id']}: prior={r['eig_prior_json']} | {r['eig_basis']}"))
        if r["output_sha256"]:
            console.print(
                escape(
                    f"  {r['job_id']}: completed by {r['output_path']} "
                    f"(output_sha256 {r['output_sha256']}, code_git_sha {r['code_git_sha']})"
                )
            )
    from fermi_desk.queue import runners

    job = next((r for r in archival if r["job_id"] == runners.BACKGROUND_JOB_ID), None)
    if job is not None and runners.ARCHIVAL_COMPLETIONS[job["job_id"]](graph) is not None:
        console.print(
            "[cyan]  linked evidence[/] "
            + escape(
                f"({job['job_id']}): the background field is promoted; EIG [bits] above stays "
                "the registered synthetic-proxy prediction. Realized vs predicted EIG, exact "
                "N_bg and the published-scope contrast: `uv run fermi-desk background-field`"
            )
        )
    t = _table(
        "Tier-0 Compute Queue (ranked by diagnostic priority; EIG = 0 bits by construction)",
        (
            "priority",
            "job_id",
            "status",
            "EIG [bits]",
            "cpu [s]",
            "review [min]",
            "diagnostic impact",
            "output_sha256",
        ),
        first_no_wrap=True,
    )
    for r in compute:
        t.add_row(
            "-" if r["diagnostic_priority"] is None else str(r["diagnostic_priority"]),
            r["job_id"],
            r["status"],
            f"{r['expected_eig_bits']:.4g}",
            f"{r['est_cpu_seconds']:.0f}",
            f"{r['est_reviewer_minutes']:.0f}",
            escape(r["diagnostic_impact"] or "-"),
            (r["output_sha256"] or "-")[:16],
        )
    console.print(t)
    for r in compute:
        console.print(escape(f"  {r['job_id']}: source={r['source']}"))
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
                "log10 BF joint (incl. illustr.)",
                "HEADLINE (verified)",
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
                _fmt(r["log10_bf_headline_verified"], "+.4g"),
                ha["local_approx"],
                hb["local_approx"],
            )
        console.print(t)
    else:
        console.print(escape(f"  results keys: {sorted(res)}"))


if __name__ == "__main__":
    app()
