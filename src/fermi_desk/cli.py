"""Reproducible CLI entrypoints: ``fermi-desk validate`` and ``fermi-desk compare``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from fermi_desk.inference.engine import CHANNELS, RunConfig, run_comparison, write_receipt
from fermi_desk.models.expansion import GEOMETRY_MODES
from fermi_desk.models.radio import RADIO_MODES
from fermi_desk.registry import find_project_root, load_factor_graph, run_validation

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console(width=150)

DataDir = Annotated[
    Path | None, typer.Option("--root", help="Project root holding data/ (default: auto-detect)")
]


def _pair(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise typer.BadParameter("expected two comma-separated hypothesis ids")
    return parts[0], parts[1]


@app.command()
def validate(root: DataDir = None) -> None:
    """Validate all YAML schemas, astropy units, catalog checksums, and provenance."""
    results = run_validation(root)
    table = Table(title="fermi-desk validate", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for r in results:
        table.add_row(r.name, "[green]OK[/]" if r.ok else "[red]FAIL[/]", r.detail)
    console.print(table)
    n_fail = sum(not r.ok for r in results)
    console.print(f"{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        raise typer.Exit(code=1)


def _fmt(x: float | None, spec: str = ".4g") -> str:
    return "-" if x is None else format(x, spec)


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
    pair: Annotated[
        str, typer.Option(help="Headline comparison 'A,B' (log10 BF of A over B).")
    ] = "frequent_slow_expanders,rare_fast_expanders",
    ir_pair: Annotated[
        str, typer.Option(help="IR calibration sweep 'A,B'.")
    ] = "quiet_expanders,partial_dyson_conspicuous",
    out_dir: Annotated[Path | None, typer.Option(help="Receipt directory (default runs/).")] = None,
    root: DataDir = None,
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

    console.rule(
        f"fermi-desk compare | geometry={geometry} radio_mode={radio_mode} seed={seed} "
        f"earth_unreached={'on' if condition_on_earth_unreached else 'off'}"
    )
    rows = list(res["hypotheses"].items())
    t1 = Table(title="Expansion & radio forward model (per scored hypothesis)")
    for col in (
        "hypothesis",
        "tier",
        "P(Earth reached)",
        "E[N] uncond.",
        "E[N] model",
        "P0 exact (MC) +/- se",
        "P0 FKG lower",
        "P0 Markov lower",
        "local approx",
    ):
        t1.add_column(col, no_wrap=col == "hypothesis")
    for hid, h in rows:
        ex, rd = h.get("expansion") or {}, h.get("radio") or {}
        t1.add_row(
            hid,
            str(h["tier"]),
            _fmt(ex.get("p_earth_reached")),
            _fmt(rd.get("expected_reached_unconditional")),
            _fmt(rd.get("expected_reached")),
            f"{rd['p_zero_exact_mc']:.8f} +/- {rd['p_zero_exact_mc_se']:.1e}" if rd else "-",
            _fmt(rd.get("p_zero_independent_fkg_lower"), ".8f"),
            _fmt(rd.get("p_zero_markov_lower"), ".8f"),
            "-" if not ex else ("ok" if ex["local_approximation_valid"] else "VIOLATED"),
        )
    console.print(t1)

    t2 = Table(title="log10 likelihood components (non-Earth terms are ratios vs. silent sky)")
    for col in ("hypothesis", "Earth unreached", "radio", "solar_system", "infrared", "joint"):
        t2.add_column(col, no_wrap=col == "hypothesis")
    for hid, h in rows:
        c = h["log10_likelihood_components"]
        t2.add_row(
            hid,
            _fmt(c.get("earth_unreached"), ".6g"),
            _fmt(c.get("radio", c.get("radio+solar_system")), ".6g"),
            _fmt(c.get("solar_system"), ".6g") if "radio+solar_system" not in c else "(joint)",
            _fmt(c.get("infrared"), ".6g"),
            _fmt(h["joint_log10_likelihood"], ".6g"),
        )
    console.print(t2)

    if "headline" in res:
        hl = res["headline"]
        ht = Table(title=escape(f"Headline: log10 BF [{hl['numerator']} : {hl['denominator']}]"))
        ht.add_column("component")
        ht.add_column("log10 BF")
        for k, v in hl["log10_bayes_factor_by_component"].items():
            ht.add_row(k, f"{v:+.6g}")
        ht.add_row("[bold]joint[/]", f"[bold]{hl['log10_bayes_factor_joint']:+.6g}[/]")
        console.print(ht)
        if hl["unshared_components"]:
            console.print(f"  unshared components (excluded): {hl['unshared_components']}")

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
    try:
        shown = out.relative_to(find_project_root())
    except (ValueError, FileNotFoundError):
        shown = out
    console.print(
        f"Receipt: {shown.as_posix()} | rng_seed={receipt['rng_seed']} | git_sha={receipt['git_sha']}"
        f" | data_tiers_used={receipt['data_tiers_used']}"
    )


if __name__ == "__main__":
    app()
