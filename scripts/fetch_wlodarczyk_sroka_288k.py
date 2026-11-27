#!/usr/bin/env python3
"""Opt-in LIVE ingestion: Gaia DR3 background-field stars inside the 692 BL L-band beams.

Reproduces the *method* of Wlodarczyk-Sroka, Garrett & Siemion (2020, MNRAS 498, 5720) -
every Gaia star inside a Breakthrough Listen primary beam is an additional, off-axis
technosignature target - on the **692 locked Enriquez et al. (2017) GBT L-band pointings**
of ``data/catalogs/bl_692_targets.csv``, using **Gaia DR3**. It does NOT reproduce their
catalogue: Wlodarczyk-Sroka et al. used the 1,327 multi-band pointings of Price et al.
(2020) with Gaia DR2 and Bailer-Jones (2018) distances and found 288,315 stars. Star
counts from this script are therefore not expected to equal 288,315.

Exactly one mode must be chosen:

``--dry-run``
    100% offline rehearsal. Verifies the locked catalog SHA-256, generates the ADQL of all
    692 cones, reports the cache state without touching it, and runs the full
    de-duplication / shell-binning / schema pipeline on a small seeded SYNTHETIC cone set
    (``synthetic_proxy``, in memory only). No outbound network traffic, no files written.
``--fetch``
    Live Gaia DR3 TAP cone queries (radius FWHM / 2 = 4.2 arcmin) around every pointing,
    cached per pointing under the gitignored ``data/cache/bl_288k_background/`` with a
    SHA-256 manifest, so an interrupted run resumes without re-querying finished
    pointings. Refuses a dirty git tree unless ``--allow-dirty`` (then the receipt can
    never be promoted). Writes a receipt under ``runs/bl_288k_background/``.
``--promote-from <receipt.json>``
    Offline. Refuses a dirty ``git_sha``, incomplete pointing coverage (< 692) or an
    oversized (>= 5 MB) output; otherwise writes the compact ``reproduced_archive``
    artifact ``data/catalogs/bl_gbt_lband_background_summary.json`` and prints the
    SHA-256 pin for ``data/surveys/``.

Distances: inverse parallax for ``parallax_over_error >= 5``; Bailer-Jones et al. (2021)
geometric distances (ARI: ``gaiaedr3_complements.distances``; identical source ids in EDR3
and DR3) for ``1 < parallax_over_error < 5``; everything else, and every star beyond 10 kpc,
is dropped and counted. A live run stops querying after ``MAX_CONSECUTIVE_FAILURES`` failed
cones in a row (a schema or outage error is never repeated across all 692 pointings), and
``--max-new-pointings`` caps attempted queries, failed ones included. De-duplication across overlapping cones and against the 692 primaries uses
the shared ``union_targets_by_pointing`` kernel (minimum beam offset kept).

Never imported or run by the offline test suite except through its offline modes
(``--dry-run``, ``--promote-from``); TLS verification is never disabled. Usage:
    uv run python scripts/fetch_wlodarczyk_sroka_288k.py --dry-run
    uv run python scripts/fetch_wlodarczyk_sroka_288k.py --fetch
    uv run python scripts/fetch_wlodarczyk_sroka_288k.py --promote-from runs/bl_288k_background/<r>.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from fermi_desk.errors import FermiDeskError
from fermi_desk.inference.engine import canonical_sha256, git_state
from fermi_desk.models import radio
from fermi_desk.registry import (
    MISSING_GAIA_ID,
    MergedTargets,
    SightingTable,
    background_summary_bytes,
    build_background_summary,
    load_factor_graph,
    primary_sightings,
    seal_background_summary,
    sightings_sha256,
    survey_f_min,
    union_targets_by_pointing,
)
from fermi_desk.schemas import (
    BACKGROUND_SUMMARY_MAX_BYTES,
    BACKGROUND_SUMMARY_PATH,
    BL_LBAND_POINTING_GROUP,
    BackgroundSummaryFile,
)

ROOT = Path(__file__).resolve().parents[1]
UA = "fermi-desk-background-fetch/0.2.1 (MIT)"
PARENT_SURVEY_ID = "bl_enriquez_2017_692"
BACKGROUND_SURVEY_ID = "bl_gbt_lband_background_gaia_dr3"
LOCKED_CATALOG_SHA256 = "c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2"
N_POINTINGS = 692
KIND = "bl_288k_background"
CACHE_DIR = ROOT / "data/cache/bl_288k_background"
OUT_DIR = ROOT / "runs/bl_288k_background"
OUT_FILE = ROOT / BACKGROUND_SUMMARY_PATH
D_MAX_PC = 10000.0
SNR_INVERSE_PARALLAX = 5.0  # parallax_over_error >= 5: d = 1 / parallax
SNR_BAILER_JONES = 1.0  # 1 < parallax_over_error < 5: Bailer-Jones geometric distance
GAIA_DR3_EPOCH = 2016.0
MATCH_EPOCH = 2000.0
MAXREC = 2_000_000
MAX_CONSECUTIVE_FAILURES = 3  # stop querying: a schema / outage error will not fix itself
SYNTHETIC_ID_BASE = 9_000_000_000_000_000_000  # above every real Gaia DR3 source_id
SUSPECT_RADIUS = 120.0 * u.arcsec
SUSPECT_DISTANCE_TOLERANCE = 0.30
ENDPOINTS = {
    "ari": {
        "url": "https://gaia.ari.uni-heidelberg.de/tap/sync",
        "label": "ARI Heidelberg Gaia TAP (gaia.ari.uni-heidelberg.de)",
        "gaia_source": "gaiadr3.gaia_source",
        # Bailer-Jones et al. (2021) EDR3 distances as served by ARI (verified live
        # 2026-11-27 via TAP_SCHEMA). `gedr3dist.main` is the GAVO data-centre name of the
        # same catalogue and does NOT exist on this service (HTTP 400, unknown table).
        "distance_table": "gaiaedr3_complements.distances",
    },
    "esa": {
        "url": "https://gea.esac.esa.int/tap-server/tap/sync",
        "label": "ESA Gaia Archive TAP (gea.esac.esa.int)",
        "gaia_source": "gaiadr3.gaia_source",
        "distance_table": "external.gaiaedr3_distance",
    },
}
COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "phot_g_mean_mag",
    "ruwe",
    "r_med_geo",
    "r_lo_geo",
    "r_hi_geo",
)
ADQL_TEMPLATE = (
    "SELECT g.source_id, g.ra, g.dec, g.ref_epoch, g.pmra, g.pmdec, g.parallax, "
    "g.parallax_error, g.parallax_over_error, g.phot_g_mean_mag, g.ruwe, "
    "d.r_med_geo, d.r_lo_geo, d.r_hi_geo "
    "FROM {gaia_source} AS g LEFT OUTER JOIN {distance_table} AS d "
    "ON d.source_id = g.source_id "
    "WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
    "CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))"
)
COUNT_TEMPLATE = (
    "SELECT COUNT(*) AS n FROM {gaia_source} AS g "
    "WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
    "CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))"
)
METHODOLOGY_NOTE = (
    "Reproduces the methodology of Wlodarczyk-Sroka, Garrett & Siemion (2020, MNRAS 498, "
    "5720) - Gaia stars inside Breakthrough Listen primary beams as off-axis targets with "
    "Gaussian beam attenuation - on the 692 locked Enriquez et al. (2017) GBT L-band "
    "pointings (data/catalogs/bl_692_targets.csv) using Gaia DR3 cones of radius FWHM/2 = "
    "4.2 arcmin, inverse-parallax distances for parallax_over_error >= 5 and Bailer-Jones "
    "et al. (2021) geometric distances for 1 < parallax_over_error < 5, d <= 10 kpc. It is "
    "NOT the published catalogue: Wlodarczyk-Sroka et al. used the 1,327 multi-band "
    "pointings of Price et al. (2020), Gaia DR2 and Bailer-Jones (2018) distances, and "
    "report 288,315 stars; counts here are not expected to match that number. Pointing "
    "centres are the published J2000 target coordinates (the telescope tracked each "
    "target's then-current position; for the fastest proper-motion primaries the centre is "
    "uncertain by up to ~0.5 arcmin of the 8.4 arcmin beam)."
)
ANGULAR_MATCH_EPOCH = (
    "J2000.0: Gaia DR3 (epoch 2016.0) positions are propagated back with pmra / pmdec "
    "before the 3 arcsec angular fallback against the published primary coordinates"
)


class LiveFetchError(RuntimeError):
    pass


_NETWORK_ALLOWED = False  # set True only inside --fetch


# -------------------------------------------------------------------- pointings
def load_pointings(root: Path = ROOT):
    """(graph, parent survey, primary sightings); verifies the locked catalog SHA-256 first."""
    graph = load_factor_graph(root)
    parent = graph.surveys[PARENT_SURVEY_ID]
    if parent.catalog_sha256 != LOCKED_CATALOG_SHA256:
        raise SystemExit(f"{PARENT_SURVEY_ID} no longer pins the locked catalog SHA-256")
    primary = primary_sightings(parent, root)  # raises ProvenanceError on a checksum mismatch
    if len(primary) != N_POINTINGS or parent.pointing_group_id != BL_LBAND_POINTING_GROUP:
        raise SystemExit(f"expected {N_POINTINGS} pointings in {BL_LBAND_POINTING_GROUP}")
    return graph, parent, primary


def beam(parent) -> tuple[u.Quantity, u.Quantity]:
    """(beam FWHM, cone radius = FWHM / 2) from the parent survey's radio sensitivity."""
    fwhm = (parent.radio_sensitivity.beam_fwhm_arcmin or 8.4) * u.arcmin
    return fwhm, fwhm / 2.0


def _fmt_query(template: str, endpoint: str, ra: float, dec: float, radius: u.Quantity) -> str:
    ep = ENDPOINTS[endpoint]
    return template.format(
        gaia_source=ep["gaia_source"],
        distance_table=ep["distance_table"],
        ra_deg=f"{ra:.6f}",
        dec_deg=f"{dec:.6f}",
        radius_deg=f"{radius.to_value(u.deg):.8f}",
    )


def cone_adql(endpoint: str, ra_deg: float, dec_deg: float, radius: u.Quantity) -> str:
    return _fmt_query(ADQL_TEMPLATE, endpoint, ra_deg, dec_deg, radius)


def count_adql(endpoint: str, ra_deg: float, dec_deg: float, radius: u.Quantity) -> str:
    return _fmt_query(COUNT_TEMPLATE, endpoint, ra_deg, dec_deg, radius)


def adql_template(endpoint: str) -> str:
    ep = ENDPOINTS[endpoint]
    return ADQL_TEMPLATE.replace("{gaia_source}", ep["gaia_source"]).replace(
        "{distance_table}", ep["distance_table"]
    )


# ------------------------------------------------------------------------ HTTP
def _post(url: str, adql: str, timeout: int = 180, tries: int = 3) -> str:
    if not _NETWORK_ALLOWED:
        raise LiveFetchError("network access attempted outside --fetch (dry-run guard)")
    if url not in {ep["url"] for ep in ENDPOINTS.values()}:
        raise LiveFetchError(f"refusing non-allowlisted TAP endpoint {url!r}")
    body = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "RESPONSEFORMAT": "csv",
            "MAXREC": str(MAXREC),
            "QUERY": adql,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if text.lstrip().startswith("<"):  # VOTable error document or an HTML page
                raise LiveFetchError("non-CSV TAP reply: " + " ".join(text[:300].split()))
            return text
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", errors="replace")
            if exc.code < 500 or attempt == tries:
                raise LiveFetchError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == tries:
                raise LiveFetchError(str(exc)) from exc
        time.sleep(5.0 * attempt)
    raise AssertionError("unreachable")


def parse_cone_csv(text: str) -> pd.DataFrame:
    """Parse a cone reply. ``source_id`` stays text (float64 would corrupt 64-bit ids)."""
    df = pd.read_csv(io.StringIO(text), dtype={"source_id": str})
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise LiveFetchError(f"cone reply lacks columns {missing} (got {list(df.columns)})")
    return df[list(COLUMNS)]


def fetch_cone(endpoint: str, ra: float, dec: float, radius: u.Quantity) -> tuple[str, int]:
    """One cone, with truncation detection: the row count must equal a COUNT(*) query."""
    url = ENDPOINTS[endpoint]["url"]
    text = _post(url, cone_adql(endpoint, ra, dec, radius))
    n_rows = len(parse_cone_csv(text))
    count = pd.read_csv(io.StringIO(_post(url, count_adql(endpoint, ra, dec, radius))))
    n_expected = int(count.iloc[0, 0])
    if n_rows != n_expected or n_rows >= MAXREC:
        raise LiveFetchError(f"truncated cone: {n_rows} rows returned, COUNT(*) = {n_expected}")
    return text, n_rows


# ----------------------------------------------------------------------- cache
class Cache:
    """Per-pointing cone replies with a SHA-256 manifest (``manifest.json``).

    A cached file is reused only if its bytes still hash to the manifest entry AND the
    entry was made with the same ADQL (so a changed query template invalidates it).
    ``writable=False`` (dry run) never creates the directory or any file.
    """

    def __init__(self, root: Path, writable: bool) -> None:
        self.root, self.writable = root, writable
        self.path = root / "manifest.json"
        if writable:
            root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict = (
            json.loads(self.path.read_text("utf-8")) if self.path.is_file() else {}
        )

    @staticmethod
    def name(index: int) -> str:
        return f"pointing_{index:03d}.csv"

    def get(self, index: int, adql_sha: str) -> tuple[str, dict] | None:
        entry = self.manifest.get(self.name(index))
        f = self.root / self.name(index)
        if not entry or entry.get("adql_sha256") != adql_sha or not f.is_file():
            return None
        raw = f.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            return None
        return raw.decode("utf-8"), entry

    def put(self, index: int, text: str, entry: dict) -> dict:
        if not self.writable:
            raise LiveFetchError("cache is read-only in this mode")
        raw = text.encode("utf-8")
        entry = {**entry, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        (self.root / self.name(index)).write_bytes(raw)
        self.manifest[self.name(index)] = entry
        self.path.write_text(json.dumps(self.manifest, indent=1, sort_keys=True) + "\n", "utf-8")
        return entry


# ------------------------------------------------------------ distances & epochs
def attach_distances(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach ``distance_pc`` / ``distance_method`` and count every cut (pure)."""

    def num(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    poe, plx, bj = num("parallax_over_error"), num("parallax"), num("r_med_geo")
    with np.errstate(invalid="ignore"):
        inverse = (poe >= SNR_INVERSE_PARALLAX) & (plx > 0)
        zone = (poe > SNR_BAILER_JONES) & (poe < SNR_INVERSE_PARALLAX)
        bj_ok = zone & np.isfinite(bj) & (bj > 0)
    dist = np.full(len(df), np.nan)
    if inverse.any():
        dist[inverse] = (plx[inverse] * u.mas).to_value(u.pc, equivalencies=u.parallax())
    dist[bj_ok] = bj[bj_ok]
    method = np.where(
        inverse, "gaia_dr3_inverse_parallax", np.where(bj_ok, "bailer_jones_2021_geo", "")
    )
    has = inverse | bj_ok
    with np.errstate(invalid="ignore"):
        far = has & (dist > D_MAX_PC)
    keep = has & ~far
    out = df.loc[keep].copy()
    out["distance_pc"] = dist[keep]
    out["distance_method"] = method[keep]
    cuts = {
        "n_cone_sightings_raw": len(df),
        "n_dropped_invalid_parallax": int(np.sum(~(inverse | zone))),
        "n_dropped_no_bailer_jones": int(np.sum(zone & ~bj_ok)),
        "n_dropped_beyond_d_max": int(far.sum()),
    }
    return out, cuts


def to_match_epoch(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Positions propagated linearly from the catalog epoch to J2000.0 (pm missing -> 0)."""

    def num(col: str, fill: float) -> np.ndarray:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(v), v, fill)

    ra, dec = num("ra", np.nan), num("dec", np.nan)
    dt = (MATCH_EPOCH - num("ref_epoch", GAIA_DR3_EPOCH)) * u.yr
    d_dec = (num("pmdec", 0.0) * u.mas / u.yr * dt).to_value(u.deg)
    d_ra = (num("pmra", 0.0) * u.mas / u.yr * dt).to_value(u.deg) / np.cos(np.radians(dec))
    return ra + d_ra, dec + d_dec


def build_sightings(
    frames: dict[int, pd.DataFrame], primary: SightingTable, cone_radius: u.Quantity, sha256: str
) -> tuple[SightingTable, dict, dict]:
    """Per-pointing cone tables -> one background SightingTable (+ per-row extras, cut counts).

    Rows are ordered by pointing index, then source_id, so the result does not depend on
    the order in which the archive returned them.
    """
    keys = (
        "n_cone_sightings_raw",
        "n_dropped_invalid_parallax",
        "n_dropped_no_bailer_jones",
        "n_dropped_beyond_d_max",
    )
    cuts = dict.fromkeys(keys, 0)
    n_outside = 0
    parts = []
    for i in sorted(frames):
        kept, c = attach_distances(frames[i])
        for k in keys:
            cuts[k] += c[k]
        if not len(kept):
            continue
        kept = kept.assign(_id=[int(x) for x in kept["source_id"]]).sort_values("_id")
        ra = pd.to_numeric(kept["ra"]).to_numpy(float)
        dec = pd.to_numeric(kept["dec"]).to_numpy(float)
        centre = SkyCoord(primary.ra_deg[i] * u.deg, primary.dec_deg[i] * u.deg)
        theta = centre.separation(SkyCoord(ra * u.deg, dec * u.deg)).to(u.arcmin)
        inside = theta <= cone_radius * (1.0 + 1e-9)
        n_outside += int(np.sum(~inside))
        ra_m, dec_m = to_match_epoch(kept)
        parts.append(
            pd.DataFrame(
                {
                    "gaia": kept["_id"].to_numpy()[inside],
                    "ra": ra[inside],
                    "dec": dec[inside],
                    "ra_match": ra_m[inside],
                    "dec_match": dec_m[inside],
                    "distance_pc": kept["distance_pc"].to_numpy(float)[inside],
                    "method": kept["distance_method"].to_numpy(object)[inside],
                    "pointing": i,
                    "theta_arcmin": theta.to_value(u.arcmin)[inside],
                }
            )
        )
    cols = ["gaia", "ra", "dec", "ra_match", "dec_match", "distance_pc", "method", "pointing"]
    df = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=[*cols, "theta_arcmin"])
    )
    n = len(df)
    cuts["n_dropped_outside_cone_recheck"] = n_outside
    cuts["n_sightings_with_distance"] = n
    cuts["n_sightings_inverse_parallax"] = int(np.sum(df["method"] == "gaia_dr3_inverse_parallax"))
    cuts["n_sightings_bailer_jones_geo"] = int(np.sum(df["method"] == "bailer_jones_2021_geo"))
    gaia = df["gaia"].to_numpy(dtype=np.int64) if n else np.zeros(0, np.int64)
    pointing = df["pointing"].to_numpy(dtype=np.int64) if n else np.zeros(0, np.int64)
    theta_arcmin = df["theta_arcmin"].to_numpy(dtype=float)
    names = np.array([f"Gaia DR3 {g}" for g in gaia], dtype=object)
    table = SightingTable(
        survey_id=BACKGROUND_SURVEY_ID,
        role="background",
        sha256=sha256 or sightings_sha256(names, gaia, pointing, theta_arcmin),
        names=names,
        gaia_dr3_source_id=gaia,
        ra_deg=df["ra_match"].to_numpy(dtype=float),
        dec_deg=df["dec_match"].to_numpy(dtype=float),
        distance=df["distance_pc"].to_numpy(dtype=float) * u.pc,
        pointing_index=pointing,
        theta=theta_arcmin * u.arcmin,
    )
    extras = {
        "sky_ra_deg": df["ra"].to_numpy(dtype=float),
        "sky_dec_deg": df["dec"].to_numpy(dtype=float),
        "distance_method": df["method"].to_numpy(dtype=object),
    }
    return table, extras, cuts


def suspected_unmatched_primaries(merged: MergedTargets, primary: SightingTable) -> list[dict]:
    """Surviving background stars that may be a primary without a Gaia id (audit only).

    A background star within 120 arcsec (J2000 positions) of an ID-less primary whose
    distance agrees within 30% is listed for human review. Nothing is removed here: the
    de-duplication rule is Gaia id, then the 3 arcsec fallback, and nothing else.
    """
    win = merged.winner
    bg = win[merged.winner_role == "background"]
    no_id = np.flatnonzero(primary.gaia_dr3_source_id == MISSING_GAIA_ID)
    if not (bg.size and no_id.size):
        return []
    stars = SkyCoord(merged.ra_deg[bg] * u.deg, merged.dec_deg[bg] * u.deg)
    prim = SkyCoord(primary.ra_deg[no_id] * u.deg, primary.dec_deg[no_id] * u.deg)
    i_p, i_s, sep, _ = stars.search_around_sky(prim, SUSPECT_RADIUS)
    d_p = primary.distance.to_value(u.pc)[no_id][i_p]
    d_s = merged.distance.to_value(u.pc)[bg][i_s]
    ok = np.abs(d_s - d_p) <= SUSPECT_DISTANCE_TOLERANCE * d_p
    rows = [
        {
            "primary_target_name": str(primary.names[no_id[p]]),
            "gaia_dr3_source_id": int(merged.gaia_dr3_source_id[bg[s]]),
            "separation_arcsec": float(a),
            "distance_pc": float(ds),
            "primary_distance_pc": float(dp),
            "theta_arcmin": float(merged.theta.to_value(u.arcmin)[bg[s]]),
        }
        for p, s, a, ds, dp in zip(
            i_p[ok], i_s[ok], sep.to_value(u.arcsec)[ok], d_s[ok], d_p[ok], strict=True
        )
    ]
    return sorted(rows, key=lambda r: (r["primary_target_name"], r["gaia_dr3_source_id"]))


def assemble(
    frames: dict[int, pd.DataFrame],
    primary: SightingTable,
    parent,
    *,
    data_tier: str,
    code_git_sha: str,
    window: tuple[str, str],
    source_service: str,
    adql: str,
    n_completed: int,
    rng_seed: int,
    n_validation: int,
    background_sha256: str = "",
    binning: radio.ShellBinning = radio.DEFAULT_SHELL_BINNING,
) -> tuple[dict, MergedTargets]:
    """Cones -> distances -> union_targets_by_pointing -> compact summary document."""
    fwhm, cone = beam(parent)
    table, extras, cuts = build_sightings(frames, primary, cone, background_sha256)
    merged = union_targets_by_pointing(BL_LBAND_POINTING_GROUP, [primary, table])
    n_p = len(primary)
    pad = np.full(n_p, np.nan)
    doc = build_background_summary(
        merged,
        primary=primary,
        sky_ra_deg=np.concatenate([pad, extras["sky_ra_deg"]]),
        sky_dec_deg=np.concatenate([pad, extras["sky_dec_deg"]]),
        distance_method=np.concatenate([np.full(n_p, "", dtype=object), extras["distance_method"]]),
        f_min=survey_f_min(parent),
        beam_fwhm=fwhm,
        cuts={
            "d_max_pc": D_MAX_PC,
            "inverse_parallax_min_snr": SNR_INVERSE_PARALLAX,
            "bailer_jones_min_snr_exclusive": SNR_BAILER_JONES,
            **cuts,
        },
        meta={
            "id": "bl_gbt_lband_background_summary",
            "data_tier": data_tier,
            "description": (
                "Gaia DR3 stars inside the 692 GBT L-band primary beams of the BL Enriquez "
                "et al. (2017) targets (cone radius FWHM/2), de-duplicated across beams and "
                "against the primaries, binned per pointing in (distance, off-axis EIRP_min)."
            ),
            "methodology_note": METHODOLOGY_NOTE,
            "code_git_sha": code_git_sha,
            "fetched_at_utc": window[1],
            "query_window_utc": list(window),
            "source_service": source_service,
            "adql_template": adql,
            "primary_catalog_path": parent.catalog_path,
            "n_pointings_completed": n_completed,
            "angular_match_epoch": ANGULAR_MATCH_EPOCH,
        },
        rng_seed=rng_seed,
        n_validation=n_validation,
        binning=binning,
        suspected_unmatched_primary=suspected_unmatched_primaries(merged, primary),
    )
    return doc, merged


def shell_binning(bin_dex: float) -> radio.ShellBinning:
    """Shell grids for one bin width in dex (distance 1 pc .. > 10 kpc, EIRP_min 1e8 .. 1e20 W).

    The default 0.05 dex is ``radio.DEFAULT_SHELL_BINNING``. A coarser width is the lever
    when a promoted summary would reach the 5 MB budget: re-run ``--fetch`` (served from the
    cache, no new queries) with e.g. ``--bin-dex 0.1``. The grid is recorded in the artifact.
    """
    if not 0.01 <= bin_dex <= 0.25:
        raise SystemExit("--bin-dex must lie in [0.01, 0.25] dex")
    if bin_dex == radio.DEFAULT_SHELL_BINNING.distance_bin_dex:
        return radio.DEFAULT_SHELL_BINNING
    return radio.ShellBinning(
        distance_bin_dex=bin_dex,
        distance_log10_pc_min=0.0,
        n_distance_bins=math.floor(4.0 / bin_dex) + 2,
        eirp_bin_dex=bin_dex,
        eirp_log10_w_min=8.0,
        n_eirp_bins=math.ceil(12.0 / bin_dex),
    )


# -------------------------------------------------------- synthetic rehearsal
def synthetic_cone_frames(
    primary: SightingTable,
    pointing_indices: Sequence[int],
    cone_radius: u.Quantity,
    rng: np.random.Generator,
    stars_per_cone: int = 40,
) -> dict[int, pd.DataFrame]:
    """Seeded SYNTHETIC cone replies (``synthetic_proxy``; dry run and offline tests only).

    Builds one synthetic star field and answers every requested cone from it, exactly as a
    TAP cone search would, so stars in overlapping beams are returned by both cones. The
    field contains, per pointing: ``stars_per_cone`` random stars cycling through every
    distance-cut category, plus the primary's own "Gaia sighting" - with the primary's real
    source_id when the catalog has one, otherwise a synthetic-id star whose 2016.0 position
    is displaced by its proper motion (so only the J2000 back-propagation lets the 3 arcsec
    fallback recover it). Synthetic ids are >= 9e18, above every real Gaia DR3 id.
    """
    rows: list[dict] = []
    serial = 0
    d_prim = primary.distance.to_value(u.pc)
    for i in pointing_indices:
        centre = SkyCoord(primary.ra_deg[i] * u.deg, primary.dec_deg[i] * u.deg)
        r = cone_radius * np.sqrt(rng.uniform(0.0004, 1.0, stars_per_cone))
        pos = centre.directional_offset_by(rng.uniform(0, 360, stars_per_cone) * u.deg, r)
        for k in range(stars_per_cone):
            cat = k % 6
            dist = float(10 ** rng.uniform(1.7, 3.9))
            row = {"plx": 1000.0 / dist, "poe": 20.0, "bj": np.nan}
            if cat == 1:  # low-SNR parallax with a Bailer-Jones distance
                row.update(poe=3.0, bj=dist)
            elif cat == 2:  # low-SNR parallax, no Bailer-Jones entry -> dropped
                row.update(poe=2.0)
            elif cat == 3:  # invalid parallax -> dropped
                row.update(plx=-0.1, poe=-0.4)
            elif cat == 4:  # beyond 10 kpc (inverse parallax) -> dropped
                row.update(plx=0.05, poe=6.0)
            elif cat == 5:  # beyond 10 kpc (Bailer-Jones) -> dropped
                row.update(poe=1.5, bj=15000.0)
            serial += 1
            rows.append(
                {
                    "source_id": SYNTHETIC_ID_BASE + serial,
                    "ra": float(pos.ra.deg[k]),
                    "dec": float(pos.dec.deg[k]),
                    "pmra": 0.0,
                    "pmdec": 0.0,
                    **row,
                }
            )
        gid = int(primary.gaia_dr3_source_id[i])
        own = {"plx": 1000.0 / d_prim[i], "poe": 100.0, "bj": np.nan}
        if gid != MISSING_GAIA_ID:
            rows.append(
                {
                    "source_id": gid,
                    "ra": centre.ra.deg,
                    "dec": centre.dec.deg,
                    "pmra": 0.0,
                    "pmdec": 0.0,
                    **own,
                }
            )
        else:
            serial += 1
            pmra, pmdec = 300.0, -200.0  # mas / yr -> ~5.8 arcsec over 16 yr
            dt = GAIA_DR3_EPOCH - MATCH_EPOCH
            rows.append(
                {
                    "source_id": SYNTHETIC_ID_BASE + serial,
                    "ra": centre.ra.deg + pmra * dt / 3.6e6 / np.cos(centre.dec.rad),
                    "dec": centre.dec.deg + pmdec * dt / 3.6e6,
                    "pmra": pmra,
                    "pmdec": pmdec,
                    **own,
                }
            )
    field = pd.DataFrame(rows).drop_duplicates("source_id").reset_index(drop=True)
    sky = SkyCoord(field["ra"].to_numpy(float) * u.deg, field["dec"].to_numpy(float) * u.deg)
    frames = {}
    for i in pointing_indices:
        centre = SkyCoord(primary.ra_deg[i] * u.deg, primary.dec_deg[i] * u.deg)
        sub = field.loc[centre.separation(sky) <= cone_radius]
        frames[int(i)] = pd.DataFrame(
            {
                "source_id": sub["source_id"].map(str),
                "ra": sub["ra"],
                "dec": sub["dec"],
                "ref_epoch": GAIA_DR3_EPOCH,
                "pmra": sub["pmra"],
                "pmdec": sub["pmdec"],
                "parallax": sub["plx"],
                "parallax_error": (sub["plx"] / sub["poe"]).abs(),
                "parallax_over_error": sub["poe"],
                "phot_g_mean_mag": 15.0,
                "ruwe": 1.0,
                "r_med_geo": sub["bj"],
                "r_lo_geo": sub["bj"] * 0.8,
                "r_hi_geo": sub["bj"] * 1.2,
            }
        ).reset_index(drop=True)
    return frames


def rehearsal_pointings(primary: SightingTable, cone_radius: u.Quantity) -> list[int]:
    """A beam-sharing pair, one primary with a Gaia id and one without (deterministic)."""
    sky = SkyCoord(primary.ra_deg * u.deg, primary.dec_deg * u.deg)
    i1, i2, sep, _ = sky.search_around_sky(sky, cone_radius)
    pairs = sorted(
        (int(a), int(b)) for a, b, s in zip(i1, i2, sep, strict=True) if a < b and s > 1 * u.arcsec
    )
    chosen = list(pairs[0]) if pairs else []
    has_id = primary.gaia_dr3_source_id != MISSING_GAIA_ID
    for want in (True, False):
        chosen.append(next(int(i) for i in np.flatnonzero(has_id == want) if i not in chosen))
    return sorted(set(chosen))


# --------------------------------------------------------------------- dry run
def dry_run(args: argparse.Namespace) -> int:
    started = datetime.now(UTC).isoformat()
    _graph, parent, primary = load_pointings()
    fwhm, cone = beam(parent)
    print(
        f"[1/5] Locked pointings: {len(primary)} targets from {parent.catalog_path} "
        f"(sha256 {LOCKED_CATALOG_SHA256[:16]}... verified); beam FWHM "
        f"{fwhm.to_value(u.arcmin):g} arcmin, cone radius {cone.to_value(u.arcmin):g} arcmin"
    )
    queries = [
        cone_adql(args.endpoint, float(primary.ra_deg[i]), float(primary.dec_deg[i]), cone)
        for i in range(len(primary))
    ]
    digest = hashlib.sha256("\n".join(queries).encode()).hexdigest()
    ep = ENDPOINTS[args.endpoint]
    print(
        f"[2/5] ADQL generated for {len(queries)} cones against {ep['label']} "
        f"(sha256 of all queries {digest[:16]}...). First query ({primary.names[0]}):"
    )
    print(f"      {queries[0]}")
    cache = Cache(args.cache_dir, writable=False)
    done = sum(
        cache.get(i, hashlib.sha256(q.encode()).hexdigest()) is not None
        for i, q in enumerate(queries)
    )
    print(
        f"[3/5] Cache {args.cache_dir.as_posix()} "
        f"({'exists' if args.cache_dir.is_dir() else 'not created yet'}; gitignored): "
        f"{done}/{len(queries)} pointings complete for this ADQL; files pointing_000.csv .. "
        f"pointing_{len(queries) - 1:03d}.csv + manifest.json; a live --fetch would query "
        f"{len(queries) - done} pointing(s)"
    )
    picks = rehearsal_pointings(primary, cone)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x53594E]))
    frames = synthetic_cone_frames(primary, picks, cone, rng)
    doc, merged = assemble(
        frames,
        primary,
        parent,
        data_tier="synthetic_proxy",
        code_git_sha=git_state(ROOT)[0],
        window=(started, datetime.now(UTC).isoformat()),
        source_service="SYNTHETIC in-memory rehearsal (no archive queried)",
        adql=adql_template(args.endpoint),
        n_completed=len(picks),
        rng_seed=args.seed,
        n_validation=args.n_validation,
        binning=shell_binning(args.bin_dex),
    )
    p = merged.provenance
    dups = merged.duplicate_gaia_ids()
    print(
        f"[4/5] De-duplication rehearsal on SYNTHETIC cones (synthetic_proxy, in memory) for "
        f"pointings {[str(primary.names[i]) for i in picks]}: {p.n_raw_targets} raw rows -> "
        f"{p.n_merged_targets} stars; {p.n_dedup_gaia_id} Gaia-id + "
        f"{p.n_dedup_angular_fallback} angular (<{p.angular_fallback_arcsec:g} arcsec) "
        f"removals, {p.n_multi_beam_resolved} multi-beam stars kept at minimum offset, "
        f"{p.n_background_absorbed_by_primary} background sightings absorbed by primaries, "
        f"{len(dups)} duplicated Gaia ids"
    )
    BackgroundSummaryFile.model_validate(doc)
    size = len(background_summary_bytes(doc))
    cuts = doc["distance_cuts"]
    ok = (
        not len(dups)
        and p.n_merged_by_role["primary"] == len(primary)
        and p.n_multi_beam_resolved > 0
        and p.n_dedup_angular_fallback > 0
        and size < BACKGROUND_SUMMARY_MAX_BYTES
    )
    print(
        f"[5/5] Summary schema valid: {doc['shells']['n_stars']} background stars in "
        f"{doc['shells']['n_cells']} shell cells, {size} bytes (< 5 MB); cuts: "
        f"{cuts['n_cone_sightings_raw']} raw, {cuts['n_dropped_invalid_parallax']} invalid "
        f"parallax, {cuts['n_dropped_no_bailer_jones']} no Bailer-Jones, "
        f"{cuts['n_dropped_beyond_d_max']} beyond {D_MAX_PC:g} pc"
    )
    if not ok:
        print("DRY RUN FAILED: de-duplication rehearsal invariants violated", file=sys.stderr)
        return 1
    print("DRY RUN OK: zero outbound network requests, no files written.")
    return 0


# ------------------------------------------------------------------ live fetch
def live_fetch(args: argparse.Namespace) -> int:
    global _NETWORK_ALLOWED
    git_sha, git_head = git_state(ROOT)
    if git_sha in ("dirty", "unknown") and not args.allow_dirty:
        print(
            f"Refusing a live fetch from a {git_sha} git tree: every archive experiment needs "
            "a clean committed code_git_sha before querying external servers (V0_2_ROADMAP "
            "guardrail 3). Commit first, or pass --allow-dirty for a smoke test whose receipt "
            "can never be promoted.",
            file=sys.stderr,
        )
        return 2
    started = datetime.now(UTC).isoformat()
    _graph, parent, primary = load_pointings()
    _fwhm, cone = beam(parent)
    ep = ENDPOINTS[args.endpoint]
    cache = Cache(args.cache_dir, writable=True)
    print(
        f"[1/3] {len(primary)} locked pointings (git_sha={git_sha}); cones of "
        f"{cone.to_value(u.arcmin):g} arcmin against {ep['label']}; cache {args.cache_dir}"
    )
    _NETWORK_ALLOWED = True
    frames: dict[int, pd.DataFrame] = {}
    per_pointing, failures, n_new = [], [], 0
    n_attempted = consecutive_failures = 0
    aborted = False
    try:
        for i in range(len(primary)):
            ra, dec = float(primary.ra_deg[i]), float(primary.dec_deg[i])
            adql_sha = hashlib.sha256(cone_adql(args.endpoint, ra, dec, cone).encode()).hexdigest()
            hit = cache.get(i, adql_sha)
            if hit is None:
                capped = (
                    args.max_new_pointings is not None and n_attempted >= args.max_new_pointings
                )
                if aborted or capped:
                    continue
                n_attempted += 1
                try:
                    text, n_rows = fetch_cone(args.endpoint, ra, dec, cone)
                except LiveFetchError as exc:
                    reason = " ".join(str(exc).split())[:600]
                    failures.append({"pointing_index": i, "reason": reason})
                    print(f"      [{i:3d}] {primary.names[i]}: FAILED {reason}")
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        aborted = True
                        print(
                            f"      {consecutive_failures} consecutive failures: no further "
                            "pointings are queried in this run (fix the cause, then re-run; "
                            "finished pointings stay cached)",
                            file=sys.stderr,
                        )
                    continue
                consecutive_failures = 0
                entry = cache.put(
                    i,
                    text,
                    {
                        "pointing_index": i,
                        "target_name": str(primary.names[i]),
                        "endpoint": ep["url"],
                        "adql_sha256": adql_sha,
                        "n_rows": n_rows,
                        "code_git_sha": git_sha,
                        "fetched_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                n_new += 1
                time.sleep(args.sleep)
            else:
                text, entry = hit
            frames[i] = parse_cone_csv(text)
            per_pointing.append({k: entry[k] for k in sorted(entry)})
            if (i + 1) % 50 == 0 or hit is None:
                print(f"      [{i:3d}] {primary.names[i]}: {entry['n_rows']} rows")
    finally:
        _NETWORK_ALLOWED = False
    finished = datetime.now(UTC).isoformat()
    n_done = len(frames)
    complete = n_done == len(primary) == N_POINTINGS
    print(f"[2/3] {n_done}/{len(primary)} pointings available ({n_new} fetched this run)")
    manifest_digest = canonical_sha256(
        {str(r["pointing_index"]): r["sha256"] for r in per_pointing}
    )
    summary = None
    if n_done:
        summary, _ = assemble(
            frames,
            primary,
            parent,
            data_tier="reproduced_archive",
            code_git_sha=git_sha,
            window=(started, finished),
            source_service=f"{ep['label']}: {ep['url']} ({ep['gaia_source']}, {ep['distance_table']})",
            adql=adql_template(args.endpoint),
            n_completed=n_done,
            rng_seed=args.seed,
            n_validation=args.n_validation,
            binning=shell_binning(args.bin_dex),
            background_sha256=manifest_digest,
        )
    receipt = {
        "kind": KIND,
        "data_tier": "reproduced_archive",
        "status": "ok" if complete and not failures else "incomplete",
        "code_git_sha": git_sha,
        "git_head": git_head,
        "rng_seed": args.seed,
        "query_window_utc": [started, finished],
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "endpoint": ep,
        "methodology_note": METHODOLOGY_NOTE,
        "locked_inputs": {
            "primary_catalog_path": parent.catalog_path,
            "primary_catalog_sha256": primary.sha256,
        },
        "pointing_coverage": {
            "n_pointings": len(primary),
            "n_pointings_completed": n_done,
            "missing_pointing_indices": [i for i in range(len(primary)) if i not in frames],
        },
        "cache": {
            "dir": args.cache_dir.as_posix(),
            "manifest_digest_sha256": manifest_digest,
            "n_fetched_this_run": n_new,
            "n_attempted_this_run": n_attempted,
            "aborted_after_consecutive_failures": aborted,
            "code_git_shas": sorted({str(r.get("code_git_sha")) for r in per_pointing}),
        },
        "failures": failures,
        "per_pointing": per_pointing,
        "summary": summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{KIND}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    n_stars = summary["shells"]["n_stars"] if summary else 0
    print(f"[3/3] Receipt ({receipt['status']}): {out.as_posix()} [{n_stars} background stars]")
    return 0 if receipt["status"] == "ok" else 1


# --------------------------------------------------------------------- promote
def promote(receipt_path: Path, out_file: Path = OUT_FILE, root: Path = ROOT) -> int:
    raw = receipt_path.read_bytes()
    rec = json.loads(raw)
    problems = []
    if rec.get("kind") != KIND or rec.get("data_tier") != "reproduced_archive":
        problems.append(f"not a reproduced_archive {KIND} receipt")
    if rec.get("status") != "ok":
        problems.append(f"receipt status {rec.get('status')!r}")
    clean = re.compile(r"^[0-9a-f]{40}$")
    if not clean.match(str(rec.get("code_git_sha"))):
        problems.append(f"code_git_sha {rec.get('code_git_sha')!r} is not a clean commit")
    cache_shas = (rec.get("cache") or {}).get("code_git_shas") or ["missing"]
    if any(not clean.match(str(s)) for s in cache_shas):
        problems.append(f"cone cache holds pointings fetched from a non-clean tree: {cache_shas}")
    cov = rec.get("pointing_coverage") or {}
    n_done = cov.get("n_pointings_completed")
    if cov.get("n_pointings") != N_POINTINGS or n_done != N_POINTINGS:
        problems.append(f"incomplete pointing coverage: {n_done} of {N_POINTINGS} pointings")
    locked = (rec.get("locked_inputs") or {}).get("primary_catalog_sha256")
    if locked != LOCKED_CATALOG_SHA256:
        problems.append("receipt was not built against the locked bl_692_targets.csv")
    summary = rec.get("summary")
    doc = None
    if not isinstance(summary, dict):
        problems.append("receipt carries no summary")
    else:
        if summary.get("data_tier") != "reproduced_archive":
            problems.append(
                f"summary data_tier {summary.get('data_tier')!r} is not reproduced_archive "
                "(synthetic rehearsal output can never be promoted)"
            )
        try:
            rel = receipt_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = receipt_path.name
        doc = seal_background_summary(
            {**summary, "source_receipt": {"path": rel, "sha256": hashlib.sha256(raw).hexdigest()}}
        )
        size = len(background_summary_bytes(doc))
        if size >= BACKGROUND_SUMMARY_MAX_BYTES:
            problems.append(
                f"oversized output: {size} bytes is not < {BACKGROUND_SUMMARY_MAX_BYTES} (5 MB)"
            )
        if not problems:
            try:
                BackgroundSummaryFile.model_validate(doc)
            except (FermiDeskError, ValueError) as exc:
                problems.append(f"summary fails its schema: {type(exc).__name__}: {exc}")
    if problems or doc is None:
        print("Refusing to promote: " + "; ".join(problems), file=sys.stderr)
        return 1
    out_bytes = background_summary_bytes(doc)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(out_bytes)
    sha = hashlib.sha256(out_bytes).hexdigest()
    print(f"Promoted -> {out_file.as_posix()} ({len(out_bytes)} bytes, < 5 MB)")
    print(f"  file sha256    {sha}")
    print(f"  content_sha256 {doc['content_sha256']}")
    print(f"SHA-256 pin for data/surveys/{BACKGROUND_SURVEY_ID}.yaml:")
    print(f"  id: {BACKGROUND_SURVEY_ID}")
    print(f"  pointing_group_id: {BL_LBAND_POINTING_GROUP}")
    print("  pointing_merge_policy: union_targets_by_pointing")
    print(f"  parent_pointing_survey_id: {PARENT_SURVEY_ID}")
    print(f'  background_summary_path: "{BACKGROUND_SUMMARY_PATH}"')
    print(f'  background_summary_sha256: "{sha}"')
    return 0


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="offline rehearsal, no network")
    mode.add_argument("--fetch", action="store_true", help="LIVE Gaia TAP cone queries (opt-in)")
    mode.add_argument("--promote-from", type=Path, default=None, metavar="RECEIPT")
    ap.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="ari")
    ap.add_argument("--seed", type=int, default=42, help="validation-sample seed (recorded)")
    ap.add_argument("--n-validation", type=int, default=1000)
    ap.add_argument("--bin-dex", type=float, default=0.05, help="shell bin width [dex], recorded")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="receipt directory")
    ap.add_argument("--out-file", type=Path, default=OUT_FILE, help="promoted artifact path")
    ap.add_argument(
        "--max-new-pointings",
        type=int,
        default=None,
        help="cap pointings queried live per run (failed attempts count)",
    )
    ap.add_argument("--sleep", type=float, default=0.25, help="pause between live queries [s]")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke test; never promotable")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.promote_from is not None:
        return promote(args.promote_from, args.out_file)
    if args.dry_run:
        return dry_run(args)
    return live_fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
