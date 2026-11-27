#!/usr/bin/env python3
"""Opt-in LIVE experiment (v0.2.2): frozen blend screens on unsmoothed unWISE W3 coadds.

Sprint 3 asks, out of sample and on a FRESH control sample that shares no patch and no star
with Gate #1 / Gate #2, whether (1) removing the AllWISE Atlas matched-filter smoothing
tightens the W3 PSF back towards 6.5 arcsec and improves image-level blend recovery, and
(2) a correlated-noise (GLS) statistic or a distribution-free conformal threshold restores
null false-alarm control. Kernels: ``fermi_desk.models.unwise_fits``.

Exactly one mode must be chosen:

``--dry-run``
    100% offline. Verifies the four locked artifact SHA-256s, draws the fresh |b| > 30 deg
    patches and re-checks them against the 12 locked Gate #1 patches, prints the exact ADQL
    and the unWISE cutout URL scheme, reports the cache state without touching it, and runs
    the whole two-stage pipeline in memory on SYNTHETIC cutouts (``synthetic_proxy``: 6.5
    arcsec PSF, correlated pixel noise). No outbound network traffic, no files written.
``--fetch``
    LIVE (v0.2.2b). Refuses a dirty git tree unless ``--allow-dirty`` (then the receipt can
    never be promoted). Strict two-stage physical ordering:
      Stage 1  catalog controls + CALIBRATION cutouts only -> PSF, pixel covariance and every
               threshold frozen -> immutable ``frozen_calibration_state.json`` (UTC + SHA-256).
      Stage 2  only after that file is written and re-read: WITHHELD cutouts (every fetch
               timestamp must be later than the freeze) -> two-arm injection -> receipt.
    Per-star cutouts are cached under the gitignored ``data/cache/unwise_w3/`` with a
    SHA-256 manifest; an interrupted run resumes from the cache and never re-freezes.
    Three consecutive failed requests stop the run.
``--promote-from <receipt.json>``
    Offline. Refuses a dirty ``git_sha``, a synthetic payload, any leakage violation, an
    undersized split (< 60 calibration or < 60 withheld stars) or an output >= 5 MB;
    otherwise writes ``data/calibrations/unwise_w3_gate_withheld.json``
    (``reproduced_archive``) and prints the YAML pin for ``data/surveys/``.

Never imported or run by the offline test suite except through its offline modes; TLS
verification is never disabled. Usage:
    uv run python scripts/run_unwise_gate.py --dry-run
    uv run python scripts/run_unwise_gate.py --fetch
    uv run python scripts/run_unwise_gate.py --promote-from runs/unwise_w3_gate/<r>.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from fermi_desk.errors import FermiDeskError, LeakageError, ProvenanceError
from fermi_desk.inference.engine import canonical_sha256, git_state
from fermi_desk.models import unwise_fits as uf
from fermi_desk.registry import (
    load_factor_graph,
    load_fits_gate2,
    load_sigma_sys_calibration,
    locked_gate1_centres,
    seal_unwise_gate,
    sha256_file,
    unwise_gate_bytes,
    verify_unwise_gate,
)
from fermi_desk.schemas import (
    UNWISE_GATE_MAX_BYTES,
    UNWISE_GATE_MIN_SPLIT,
    UNWISE_GATE_PATH,
    UnwiseGateWithheldFile,
)

ROOT = Path(__file__).resolve().parents[1]
UA = "fermi-desk-unwise-gate/0.2.2 (MIT)"
IR_SURVEY = "hephaistos_wise_mid_ir"
KIND = "unwise_w3_gate"
ARTIFACT_ID = "unwise_w3_gate_withheld"
CACHE_DIR = ROOT / "data/cache/unwise_w3"
OUT_DIR = ROOT / "runs/unwise_w3_gate"
OUT_FILE = ROOT / UNWISE_GATE_PATH
CLAIM = "data/adversary_claims/hephaistos_candidates_d_e.yaml"
SAMPLE_NAME = "sample.json"
FROZEN_NAME = "frozen_calibration_state.json"
MANIFEST_NAME = "manifest.json"
# Locked empirical artifacts: verified before anything else, never written by this script.
LOCKED_ARTIFACTS = {
    "data/catalogs/bl_692_targets.csv": (
        "c299d61df3cd5f7eb9f2a28111fbcb1ae0c375a1dc7f219b2e86fdb005716ee2"
    ),
    "data/calibrations/wise_sigma_sys_control.json": (
        "2da2a234ee0ccb07fb571431f5f178e362ff4d9d4f96d4ee878b921916127b7a"
    ),
    "data/calibrations/wise_fits_gate2_withheld.json": (
        "d4818c100b02cea98d5beafcdea66e679c072e5d47e28fe8d3a960f78c64da99"
    ),
    "data/catalogs/bl_gbt_lband_background_summary.json": (
        "eb43b118ab4d0258d9fe20474b73d187cb62a98018b20790b4112f875b0018a9"
    ),
}
LOCKED_SIGMA_SYS = "data/calibrations/wise_sigma_sys_control.json"
MAX_CONSECUTIVE_FAILURES = 3  # stop querying: a schema / outage error will not fix itself
WISE_MEAN_EPOCH_YR = 2010.5  # the unWISE "allwise" coadds stack the AllWISE frame set
SYNTHETIC_ID_BASE = 9_000_000_000_000_000_000  # above every real Gaia DR3 source_id
REHEARSAL_NOISE_CORRELATION = 4.5 * u.arcsec  # lag-1 pixel correlation ~0.58 at 2.75"/px
ATLAS_LIKE_FWHM = 13.69 * u.arcsec  # the Gate #2 Atlas width, for the PSF discrimination check

ARI_TAP = "https://gaia.ari.uni-heidelberg.de/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
UNWISE_CUTOUT = "https://unwise.me/cutout_fits"
ALLOWED_URL_PREFIXES = (ARI_TAP, IRSA_TAP, UNWISE_CUTOUT)
# The Gate #1 control-population query, unchanged (scripts/run_wise_control_experiment.py).
GAIA_ADQL = (
    "SELECT gs.source_id, gs.ra, gs.dec, gs.l, gs.b, gs.pmra, gs.pmdec, gs.ref_epoch, "
    "gs.parallax, gs.parallax_over_error, gs.bp_rp, gs.phot_g_mean_mag, gs.ruwe, "
    "xm.original_ext_source_id "
    "FROM gaiadr3.gaia_source AS gs "
    "JOIN gaiadr3.allwise_best_neighbour AS xm ON gs.source_id = xm.source_id "
    "WHERE 1 = CONTAINS(POINT('ICRS', gs.ra, gs.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r:.4f})) "
    "AND gs.parallax > 3.3333 AND gs.parallax_over_error > 10 AND gs.bp_rp >= 1.84 "
    "AND gs.phot_g_mean_mag BETWEEN 10 AND 16"
)
# Gate #1's AllWISE query plus coadd_id (the Atlas tile; its first 8 characters name the
# unWISE tile on the same grid).
ALLWISE_ADQL = (
    "SELECT designation, ra, dec, w3snr, w3mpro, w3sigmpro, cc_flags, ext_flg, nb, na, ph_qual, "
    "coadd_id FROM allwise_p3as_psd WHERE CONTAINS(POINT('ICRS', ra, dec), "
    "CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r:.4f})) = 1 "
    "AND w3snr >= {snr_lo:g} AND w3snr <= {snr_hi:g}"
)
ISOLATION_ADQL = (
    "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS("
    "POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.7f}))"
)
CUTOUT_URL_TEMPLATE = (
    UNWISE_CUTOUT + "?version=allwise&ra={ra:.7f}&dec={dec:.7f}&size={size:d}&bands=3"
    "&file_img_m=on&file_invvar_m=on"
)
MEMBER_RE = re.compile(r"unwise-(\d{4}[pm]\d{3})-w3-(img|invvar)-m\.fits(\.gz)?$")
# Every archive assumption in one place. Nothing here was queried in v0.2.2a: the entries
# marked ``verified_live: False`` are to be ratified before v0.2.2b touches a server.
ENDPOINTS = {
    "gaia_tap": {
        "url": ARI_TAP,
        "label": "ARI Heidelberg Gaia TAP (gaiadr3.gaia_source, gaiadr3.allwise_best_neighbour)",
        "verified_live": "used unchanged by Gate #1 (b02ec7df820d) and v0.2.1 (9f3740ee71b0)",
    },
    "allwise_tap": {
        "url": IRSA_TAP,
        "label": "NASA/IPAC IRSA TAP (allwise_p3as_psd)",
        "verified_live": "Gate #1 query; the added coadd_id column is NOT yet verified live",
    },
    "unwise_cutout": {
        "service": "unwise.me cutout service, Lang (2014) unWISE coadds, release 'allwise' "
        "(the only unWISE release with W3; the NEOWISE-era releases are W1/W2 only)",
        "url_template": CUTOUT_URL_TEMPLATE,
        "parameters": {
            "version": "allwise",
            "ra, dec": "ICRS degrees: the Gaia DR3 position propagated to epoch 2010.5",
            "size": "cutout side in NATIVE 2.75 arcsec pixels (45 px = 123.75 arcsec)",
            "bands": "3 (W3)",
            "file_img_m": "on: masked (outlier-rejected) intensity, unwise-<tile>-w3-img-m.fits",
            "file_invvar_m": "on: its inverse variance, unwise-<tile>-w3-invvar-m.fits",
        },
        "expected_reply": "gzipped tar archive holding one img-m / invvar-m FITS pair per "
        "overlapping tile, cut from the coadd on its native pixel grid (no resampling), "
        "each with a TAN WCS",
        "tile_choice": "the pair whose tile is the AllWISE coadd_id[:8] of the star when it "
        "covers the star with the required margin, otherwise the pair with the largest "
        "margin; a reply with no such pair is recorded as unusable, never resampled",
        "native_pixel_check": "the WCS pixel scale must be 2.75 arcsec within 1e-3, else the "
        "cutout is refused",
        "verified_live": False,
        "assumptions_to_ratify": [
            "host and path https://unwise.me/cutout_fits answer over TLS",
            "query keys version / ra / dec / size / bands / file_img_m / file_invvar_m",
            "'size' is in pixels and 45 is within the service limit",
            "the reply is a (gzipped) tar whose members are named "
            "unwise-<RRRRsDDD>-w3-{img,invvar}-m.fits[.gz]",
            "invvar-m is the per-pixel inverse variance of the img-m coadd in the same units",
        ],
    },
    "fallbacks": {
        "full_tiles": {
            "url_template": "https://unwise.me/data/allwise/unwise-coadds/fulldepth/"
            "{tile[:3]}/{tile}/unwise-{tile}-w3-{img-m.fits|invvar-m.fits.gz}",
            "status": "documented, NOT implemented: a 1.56 deg tile holds calibration and "
            "withheld stars together, so downloading one before the freeze would weaken the "
            "physical two-stage ordering from 'bytes not fetched' to 'bytes not read'. Needs "
            "a planning ruling before use",
            "verified_live": False,
        },
        "legacy_survey_viewer": {
            "url_template": "https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}"
            "&layer=<unWISE W3/W4 layer>&pixscale=2.75&size={size}",
            "status": "rejected for promotion: the viewer resamples onto the requested grid "
            "(re-correlating the pixel noise that Screen B measures) and serves no "
            "inverse-variance map. Diagnostic use only; layer name unverified",
            "verified_live": False,
        },
    },
}
CAVEATS = [
    "Arm B pins the excess to the star's measured (noisy) windowed centroid, so at high f_bg "
    "the centered excess inherits the uninjected centroid noise while the screen threshold "
    "assumes SNR_eff; centroid-screen Arm B false rejection is conservative (biased high)",
    "injection adds the empirical PSF without extra noise (W3 is background-limited; source "
    "shot noise neglected); real blends are extended or coloured, the injected ones are "
    "PSF-shaped point sources",
    "the empirical PSF is aligned on noise-locked windowed centroids (shift-and-add), which "
    "sharpens a stacked PSF by a few percent at W3 SNR 8-20; its Gaussian-fit FWHM is a "
    "slight underestimate, and at 2.75 arcsec pixels the W3 core is close to critically "
    "sampled, so the cubic-spline PSF model carries interpolation error",
    "the pixel covariance is assumed stationary and identical (in units of each cutout's "
    "scalar variance) for every star; it is estimated from sky annuli only, so source-"
    "dependent noise under the star is not modelled",
    "the covariance is regularised (taper, spectral floor, 5% shrinkage towards white noise): "
    "the GLS statistic is therefore not exactly chi^2-distributed, and its threshold comes "
    "from calibration nulls, not from theory",
    "the conformal level (k+1)/(n_cal+1) is exact for exchangeable scores. Calibration scores "
    "are leave-one-out (PSF and covariance rebuilt without the star), withheld scores use "
    "all n_cal stars: the two differ by one star in n_cal, an O(1/n_cal) departure from "
    "exact exchangeability that is not corrected",
    "the conformal level is MARGINAL over calibration draws: given this one calibration half, "
    "the realised false-alarm probability of the frozen threshold is a Beta(k+1, n_cal-k) "
    "variate with mean (k+1)/(n_cal+1), so a withheld rate above the level is not by itself "
    "a miscalibration",
    "PSF stamps are normalised by their least-squares amplitude against the nominal 6.5 "
    "arcsec Gaussian, NOT by their own summed flux as in Gate #2 (that sum is noise-"
    "dominated on 2.75 arcsec pixels and its positive-sum selection leaves a pedestal); "
    "the stacked PSF wings still carry noise (about 10% of the total flux in the strongly "
    "correlated synthetic rehearsal), so the flux inside the 16.5 arcsec aperture differs "
    "from the unit sum by that much; both arms inject the same PSF, so they stay matched",
    "Screens A and B keep Gate #2's in-sample calibration scores (like-for-like control); "
    "the leave-one-out scores are stored beside them",
    "calibration 'nulls' are isolated Gaia-clean M dwarfs; unseen W3 background sources "
    "among them are part of the realistic null, not removed",
    "SNR_eff = w3snr / (1 - f_bg) assumes background-limited photometry of the blend; w3snr "
    "is the AllWISE catalog value, not re-measured on unWISE",
    "epoch propagation uses proper motion only; the coadd mean epoch is taken as 2010.5",
]


class LiveFetchError(RuntimeError):
    pass


_NETWORK_ALLOWED = False  # set True only inside --fetch


class Clock:
    """UTC timestamps that strictly increase (two stamps of one run are never equal)."""

    def __init__(self) -> None:
        self._last: datetime | None = None

    def __call__(self) -> str:
        now = datetime.now(UTC)
        if self._last is not None and now <= self._last:
            now = self._last + timedelta(microseconds=1)
        self._last = now
        return now.isoformat()


# ---------------------------------------------------------------- locked inputs
class Locked(NamedTuple):
    """Everything read (never written) from the locked Gate #1 / Gate #2 artifacts."""

    calibration: object  # registry.SigmaSysCalibration
    patch_ids: list[str]
    centres: SkyCoord
    star_ids: list[int]
    sigma_sys: u.Quantity
    atlas_fwhm_arcsec: float | None
    artifacts: dict


def load_locked(root: Path = ROOT) -> Locked:
    """Verify the four locked SHA-256s and load the Gate #1 sample (ProvenanceError on drift)."""
    for rel, want in LOCKED_ARTIFACTS.items():
        got = sha256_file(root / rel)
        if got != want:
            raise ProvenanceError(f"locked artifact {rel} changed: sha256 {got} != {want}")
    graph = load_factor_graph(root)
    survey = graph.surveys[IR_SURVEY]
    cal = load_sigma_sys_calibration(survey, root)
    if cal is None or cal.path != LOCKED_SIGMA_SYS:
        raise ProvenanceError(f"{IR_SURVEY} no longer pins {LOCKED_SIGMA_SYS}")
    gate2 = load_fits_gate2(survey, root)
    patch_ids, centres = locked_gate1_centres(cal)
    return Locked(
        cal,
        patch_ids,
        centres,
        [int(c.gaia_dr3_source_id) for c in cal.record.controls],
        cal.sigma_sys,
        gate2.record.empirical_psf.fwhm_gaussian_arcsec if gate2 is not None else None,
        dict(LOCKED_ARTIFACTS),
    )


def protocol(args: argparse.Namespace) -> dict:
    """Every setting that defines the sample and the frozen method (compared on resume)."""
    return {
        "rule_version": uf.RULE_VERSION,
        "seed": args.seed,
        "n_controls": args.n_controls,
        "n_patches": args.n_patches,
        "patch_radius_deg": args.patch_radius_deg,
        "min_abs_b_deg": args.min_abs_b,
        "min_locked_separation_deg": args.min_locked_separation_deg,
        "min_patch_separation_deg": args.min_patch_separation_deg,
        "w3_snr_min": args.w3_snr_min,
        "w3_snr_max": args.w3_snr_max,
        "ruwe_max": args.ruwe_max,
        "isolation_arcsec": args.isolation_arcsec,
        "target_false_alarm": args.target_false_alarm,
        "conformal_alpha": args.conformal_alpha,
        "cutout_size_pix": args.cutout_size_pix,
    }


def control_cuts(args: argparse.Namespace) -> uf.ControlCuts:
    return uf.ControlCuts(
        min_abs_b=args.min_abs_b * u.deg,
        ruwe_max=args.ruwe_max,
        w3snr_min=args.w3_snr_min,
        w3snr_max=args.w3_snr_max,
        isolation=args.isolation_arcsec * u.arcsec,
    )


def draw_patches(args: argparse.Namespace, locked: Locked) -> tuple[list[dict], int, dict]:
    """Fresh patches for ``args.seed`` plus the independent re-check against Gate #1."""
    kw = {
        "min_abs_b": args.min_abs_b * u.deg,
        "min_locked_separation": args.min_locked_separation_deg * u.deg,
    }
    patches, draws = uf.choose_fresh_patches(
        args.seed,
        locked.centres,
        n_patches=args.n_patches,
        radius=args.patch_radius_deg * u.deg,
        min_patch_separation=args.min_patch_separation_deg * u.deg,
        **kw,
    )
    for p in patches:
        p["nearest_locked_patch_id"] = locked.patch_ids[p.pop("nearest_locked_patch_index")]
    return patches, draws, uf.patch_separation_audit(patches, locked.centres, **kw)


def patch_adql(patch: dict, args: argparse.Namespace) -> tuple[str, str]:
    ra, dec, r = patch["ra_deg"], patch["dec_deg"], patch["radius_deg"]
    return (
        GAIA_ADQL.format(ra=ra, dec=dec, r=r),
        ALLWISE_ADQL.format(ra=ra, dec=dec, r=r, snr_lo=args.w3_snr_min, snr_hi=args.w3_snr_max),
    )


def cutout_url(ra_deg: float, dec_deg: float, size_pix: int) -> str:
    return CUTOUT_URL_TEMPLATE.format(ra=ra_deg, dec=dec_deg, size=int(size_pix))


# ------------------------------------------------------------------------ HTTP
def _http(url: str, data: bytes | None = None, timeout: int = 300, tries: int = 3) -> bytes:
    if not _NETWORK_ALLOWED:
        raise LiveFetchError("network access attempted outside --fetch (dry-run guard)")
    if not url.startswith(ALLOWED_URL_PREFIXES):
        raise LiveFetchError(f"refusing non-allowlisted URL {url!r}")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    host = urllib.parse.urlparse(url).netloc
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", errors="replace")
            if exc.code < 500 or attempt == tries:
                raise LiveFetchError(f"{host}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == tries:
                raise LiveFetchError(f"{host}: {exc}") from exc
        time.sleep(5.0 * attempt)
    raise AssertionError("unreachable")


def _tap_csv(endpoint: str, adql: str) -> pd.DataFrame:
    body = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
    ).encode()
    text = _http(endpoint, body).decode("utf-8", errors="replace")
    if text.lstrip().startswith("<"):  # VOTable error document or an HTML page
        raise LiveFetchError(f"non-CSV TAP reply from {endpoint}: " + " ".join(text[:300].split()))
    text_columns = ("source_id", "cc_flags", "ph_qual", "coadd_id", "designation")
    return pd.read_csv(io.StringIO(text), dtype=dict.fromkeys(text_columns, str))


def _num(v: object) -> float:
    return 0.0 if pd.isna(v) else float(v)


def propagate(row: dict) -> tuple[float, float]:
    """Gaia position at the WISE mean epoch (proper motion only), as in Gate #1."""
    dt = WISE_MEAN_EPOCH_YR - float(row["ref_epoch"])
    dec = float(row["dec_deg"]) + _num(row["pmdec"]) * dt / 3.6e6
    ra = float(row["ra_deg"]) + _num(row["pmra"]) * dt / 3.6e6 / math.cos(
        math.radians(float(row["dec_deg"]))
    )
    return ra, dec


# -------------------------------------------------------------------- archives
class LiveArchive:
    """ARI Gaia TAP, IRSA AllWISE TAP and the unwise.me cutout service (``--fetch`` only)."""

    data_tier = "reproduced_archive"
    label = "live archives (ARI Gaia TAP, IRSA AllWISE TAP, unwise.me cutouts)"

    def __init__(self, sleep: float, clock: Clock) -> None:
        self.sleep, self.clock, self.log = sleep, clock, []

    def _query(self, endpoint: str, adql: str, **meta: object) -> pd.DataFrame:
        df = _tap_csv(endpoint, adql)
        self.log.append(
            {
                "endpoint": endpoint,
                "queried_at_utc": self.clock(),
                "adql_sha256": hashlib.sha256(adql.encode()).hexdigest(),
                "n_rows": len(df),
                **meta,
            }
        )
        time.sleep(self.sleep)
        return df

    def candidates(self, patch: dict, args: argparse.Namespace) -> tuple[list[dict], dict]:
        g_adql, a_adql = patch_adql(patch, args)
        pid = patch["patch_id"]
        gaia = self._query(ARI_TAP, g_adql, patch_id=pid, role="gaia_m_dwarfs")
        wise = self._query(IRSA_TAP, a_adql, patch_id=pid, role="allwise_w3_band")
        wise = wise.rename(columns={"ra": "ra_aw", "dec": "dec_aw"})
        m = gaia.merge(wise, left_on="original_ext_source_id", right_on="designation", how="inner")
        rows = [
            {
                "gaia_dr3_source_id": str(r.source_id),
                "allwise": str(r.designation),
                "coadd_id": None if pd.isna(r.coadd_id) else str(r.coadd_id),
                "ra_deg": float(r.ra),
                "dec_deg": float(r.dec),
                "l_deg": float(r.l),
                "b_deg": float(r.b),
                "pmra": _num(r.pmra),
                "pmdec": _num(r.pmdec),
                "ref_epoch": float(r.ref_epoch),
                "parallax_mas": float(r.parallax),
                "bp_rp": float(r.bp_rp),
                "phot_g_mean_mag": float(r.phot_g_mean_mag),
                "ruwe": float(r.ruwe),
                "w3snr": float(r.w3snr),
                "w3mpro": float(r.w3mpro),
                "cc_flags": str(r.cc_flags),
                "ext_flg": int(r.ext_flg),
            }
            for r in m.itertuples(index=False)
        ]
        meta = {"gaia_m_dwarfs_in_cone": len(gaia), "allwise_w3_band_rows_in_cone": len(wise)}
        return rows, meta

    def n_neighbours(self, row: dict, radius: u.Quantity) -> int:
        adql = ISOLATION_ADQL.format(ra=row["ra_deg"], dec=row["dec_deg"], r=radius.to_value(u.deg))
        df = self._query(
            ARI_TAP, adql, role="isolation", gaia_dr3_source_id=row["gaia_dr3_source_id"]
        )
        return int((df["source_id"].astype(str) != row["gaia_dr3_source_id"]).sum())

    def cutout(self, control: dict, size_pix: int) -> tuple[bytes, str]:
        url = cutout_url(
            control["gaia_ra_epoch_wise_deg"], control["gaia_dec_epoch_wise_deg"], size_pix
        )
        data = _http(url)
        time.sleep(self.sleep)
        return data, url


def pack_cutout(
    flux: np.ndarray,
    invvar: np.ndarray,
    gaia_xy: tuple[float, float],
    ra_deg: float,
    dec_deg: float,
    tile: str,
    pixel_scale: u.Quantity = uf.UNWISE_PIXEL_SCALE,
) -> bytes:
    """A reply in the documented cutout-service format: gzipped tar of two FITS files.

    Used by the synthetic rehearsal and the offline tests so that they run the same byte
    parser as a live reply (deterministic bytes: fixed tar and gzip timestamps).
    """
    step = pixel_scale.to_value(u.deg)
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = float(ra_deg), float(dec_deg)
    header["CRPIX1"], header["CRPIX2"] = float(gaia_xy[0]) + 1.0, float(gaia_xy[1]) + 1.0
    header["CD1_1"], header["CD1_2"], header["CD2_1"], header["CD2_2"] = -step, 0.0, 0.0, step
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for kind, arr in (("img", flux), ("invvar", invvar)):
            buf = io.BytesIO()
            fits.PrimaryHDU(np.asarray(arr, np.float32), header).writeto(buf)
            info = tarfile.TarInfo(f"unwise-{tile}-w3-{kind}-m.fits")
            info.size, info.mtime = buf.tell(), 0
            buf.seek(0)
            tar.addfile(info, buf)
    return gzip.compress(raw.getvalue(), mtime=0)


class SyntheticArchive:
    """Seeded in-memory stand-in for all three archives (``synthetic_proxy``, dry run only).

    Stars are synthetic M-dwarf-like rows with ids above every real Gaia DR3 id, plus a few
    decoys that each fail one cut; cutouts are Gaussian-PSF stamps with correlated noise.
    """

    data_tier = "synthetic_proxy"
    label = "SYNTHETIC in-memory rehearsal (no archive queried)"

    def __init__(
        self,
        seed: int,
        n_stars: int,
        n_patches: int,
        *,
        psf_fwhm: u.Quantity = uf.NOMINAL_W3_FWHM_ARCSEC * u.arcsec,
        noise_correlation_fwhm: u.Quantity = REHEARSAL_NOISE_CORRELATION,
    ) -> None:
        self.seed, self.n_stars, self.n_patches = int(seed), int(n_stars), int(n_patches)
        self.psf_fwhm, self.noise_correlation_fwhm = psf_fwhm, noise_correlation_fwhm
        self.log: list[dict] = []
        self._neighbours: dict[str, int] = {}

    def _rng(self, *stream: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, 0x53594E, *stream]))

    def candidates(self, patch: dict, args: argparse.Namespace) -> tuple[list[dict], dict]:
        index = int(patch["patch_id"][1:]) - 1
        n_good = self.n_stars // self.n_patches + (index < self.n_stars % self.n_patches)
        rng = self._rng(1, index)
        decoys = (
            {"ruwe": 1.9},
            {"cc_flags": "h000"},
            {"ext_flg": 1},
            {"_neighbours": 1},
        )
        rows = []
        for i in range(n_good + len(decoys)):
            b = patch["b_deg"] + float(rng.uniform(-1.0, 1.0))
            lon = patch["l_deg"] + float(rng.uniform(-1.0, 1.0)) / math.cos(math.radians(b))
            icrs = SkyCoord(l=lon * u.deg, b=b * u.deg, frame="galactic").icrs
            sid = str(SYNTHETIC_ID_BASE + 1000 * (index + 1) + i)
            row = {
                "gaia_dr3_source_id": sid,
                "allwise": f"SYNTHETIC-{sid}",
                "coadd_id": f"{index:04d}p000_ac51",
                "ra_deg": float(icrs.ra.deg),
                "dec_deg": float(icrs.dec.deg),
                "l_deg": lon % 360.0,
                "b_deg": b,
                "pmra": float(rng.normal(0.0, 50.0)),
                "pmdec": float(rng.normal(0.0, 50.0)),
                "ref_epoch": 2016.0,
                "parallax_mas": float(rng.uniform(3.5, 20.0)),
                "bp_rp": float(rng.uniform(1.9, 3.0)),
                "phot_g_mean_mag": float(rng.uniform(11.0, 15.5)),
                "ruwe": float(rng.uniform(0.9, 1.2)),
                "w3snr": float(rng.uniform(args.w3_snr_min, args.w3_snr_max)),
                "w3mpro": float(rng.uniform(10.0, 11.5)),
                "cc_flags": "0000",
                "ext_flg": 0,
            }
            fail = decoys[i - n_good] if i >= n_good else {}
            self._neighbours[sid] = int(fail.get("_neighbours", 0))
            row.update({k: v for k, v in fail.items() if not k.startswith("_")})
            rows.append(row)
        g_adql, a_adql = patch_adql(patch, args)
        for adql, role in ((g_adql, "gaia_m_dwarfs"), (a_adql, "allwise_w3_band")):
            self.log.append(
                {
                    "endpoint": "SYNTHETIC",
                    "adql_sha256": hashlib.sha256(adql.encode()).hexdigest(),
                    "n_rows": len(rows),
                    "patch_id": patch["patch_id"],
                    "role": role,
                }
            )
        return rows, {
            "gaia_m_dwarfs_in_cone": len(rows),
            "allwise_w3_band_rows_in_cone": len(rows),
        }

    def n_neighbours(self, row: dict, radius: u.Quantity) -> int:
        return self._neighbours[row["gaia_dr3_source_id"]]

    def cutout(self, control: dict, size_pix: int) -> tuple[bytes, str]:
        sid = control["gaia_dr3_source_id"]
        cut = uf.synthetic_cutout(
            self._rng(2, int(sid) % 2**31),
            sid,
            control["split"],
            psf_fwhm=self.psf_fwhm,
            noise_correlation_fwhm=self.noise_correlation_fwhm,
            snr=float(control["w3snr"]),
            shape=(size_pix, size_pix),
        )
        ra, dec = control["gaia_ra_epoch_wise_deg"], control["gaia_dec_epoch_wise_deg"]
        data = pack_cutout(
            cut.flux, 1.0 / cut.unc**2, cut.gaia_xy, ra, dec, control["coadd_id"][:8]
        )
        return data, "synthetic://" + cutout_url(ra, dec, size_pix).split("://", 1)[1]


# ---------------------------------------------------------------- cutout parser
class ParsedCutout(NamedTuple):
    flux: np.ndarray
    invvar: np.ndarray
    gaia_xy: tuple[float, float]
    pixel_scale_measured_arcsec: float
    tile: str
    shape: tuple[int, int]


def _fits_image(data: bytes) -> tuple[np.ndarray, fits.Header]:
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if not data.startswith(b"SIMPLE"):
        raise LiveFetchError(f"archive member is not FITS: {data[:60]!r}")
    with fits.open(io.BytesIO(data)) as hdul:
        return np.asarray(hdul[0].data, float), hdul[0].header.copy()


def parse_cutout(data: bytes, control: dict, need_pix: float) -> ParsedCutout:
    """Native-pixel W3 intensity + inverse variance from one cutout-service reply.

    Raises ``LiveFetchError`` when the reply is not a tar of FITS pairs, when no tile covers
    the star with ``need_pix`` clear pixels on every side, or when the pixels are not the
    native 2.75 arcsec grid (a resampled image is never accepted).
    """
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise LiveFetchError(f"reply is not a tar archive: {exc}; head {data[:60]!r}") from exc
    pairs: dict[str, dict[str, bytes]] = {}
    for name, blob in members.items():
        hit = MEMBER_RE.search(name.rsplit("/", 1)[-1])
        if hit:
            pairs.setdefault(hit.group(1), {})[hit.group(2)] = blob
    ra, dec = control["gaia_ra_epoch_wise_deg"], control["gaia_dec_epoch_wise_deg"]
    preferred = (control.get("coadd_id") or "")[:8]
    best: tuple[float, ParsedCutout] | None = None
    for tile in sorted(pairs):
        if set(pairs[tile]) != {"img", "invvar"}:
            continue
        flux, header = _fits_image(pairs[tile]["img"])
        invvar, _ = _fits_image(pairs[tile]["invvar"])
        if flux.ndim != 2 or flux.shape != invvar.shape:
            continue
        wcs = WCS(header)
        gx, gy = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        pix = float(np.mean(proj_plane_pixel_scales(wcs.celestial)) * 3600.0)
        margin = min(gx, gy, flux.shape[1] - 1 - gx, flux.shape[0] - 1 - gy)
        if margin < need_pix or abs(pix - uf.UNWISE_PIXEL_SCALE.to_value(u.arcsec)) > 1e-3:
            continue
        rank = margin + (1e6 if tile == preferred else 0.0)
        if best is None or rank > best[0]:
            best = (rank, ParsedCutout(flux, invvar, (gx, gy), pix, tile, flux.shape))
    if best is None:
        raise LiveFetchError(
            f"no native-pixel W3 img-m / invvar-m pair covers the star with {need_pix:.0f} px "
            f"of margin (members: {sorted(members)[:6]})"
        )
    return best[1]


# ----------------------------------------------------------------------- cache
class Cache:
    """Per-star cutout replies, the sample and the frozen state, with a SHA-256 manifest.

    ``root=None`` keeps everything in memory (dry run: nothing touches the disk). A cached
    cutout is reused only if its bytes still hash to the manifest entry. ``write_once``
    never overwrites: the sample and the frozen calibration state are immutable.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root
        self._mem: dict[str, bytes] = {}
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict = self.read_json(MANIFEST_NAME) or {}

    def _read(self, name: str) -> bytes | None:
        if self.root is None:
            return self._mem.get(name)
        f = self.root / name
        return f.read_bytes() if f.is_file() else None

    def _write(self, name: str, data: bytes) -> None:
        if self.root is None:
            self._mem[name] = data
        else:
            (self.root / name).write_bytes(data)

    def read_json(self, name: str) -> dict | None:
        raw = self._read(name)
        return None if raw is None else json.loads(raw)

    def write_once(self, name: str, obj: dict) -> None:
        if self._read(name) is not None:
            raise LeakageError(f"{name} already exists and is immutable; it is never rewritten")
        self._write(name, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8"))

    @staticmethod
    def cutout_name(star_id: str) -> str:
        return f"{star_id}_unwise_w3_cutout.tar.gz"

    def get_cutout(self, star_id: str) -> tuple[bytes, dict] | None:
        entry = self.manifest.get(self.cutout_name(star_id))
        raw = self._read(self.cutout_name(star_id))
        if not entry or raw is None or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            return None
        return raw, entry

    def put_cutout(self, star_id: str, data: bytes, entry: dict) -> dict:
        entry = {**entry, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        self._write(self.cutout_name(star_id), data)
        self.manifest[self.cutout_name(star_id)] = entry
        self._write(
            MANIFEST_NAME, (json.dumps(self.manifest, indent=1, sort_keys=True) + "\n").encode()
        )
        return entry

    def entries(self, split: str) -> list[dict]:
        return [e for e in self.manifest.values() if e.get("split") == split]


def cache_report(root: Path) -> str:
    """State of the on-disk cache, read without creating or changing anything."""
    if not root.is_dir():
        return "not created yet"
    manifest = root / MANIFEST_NAME
    entries = json.loads(manifest.read_text("utf-8")).values() if manifest.is_file() else []
    by = {s: sum(e.get("split") == s for e in entries) for s in uf.SPLITS}
    return (
        f"exists: {by['calibration']} calibration + {by['withheld']} withheld cutout(s), "
        f"{SAMPLE_NAME} {'present' if (root / SAMPLE_NAME).is_file() else 'absent'}, "
        f"{FROZEN_NAME} {'present' if (root / FROZEN_NAME).is_file() else 'absent'}"
    )


# ------------------------------------------------------------- stage 0: sample
def build_sample(archive, locked: Locked, args: argparse.Namespace, clock: Clock) -> dict:
    """Catalog stage: fresh patches, cuts, isolation, disjointness, 50/50 split.

    No image is touched here, so the sample cannot depend on any cutout.
    """
    cuts = control_cuts(args)
    steps = cuts.steps()
    patches, draws, audit = draw_patches(args, locked)
    exclude = {
        str(int(c["gaia_dr3_source_id"]))
        for c in yaml.safe_load((ROOT / CLAIM).read_text("utf-8"))["candidates"]
    }
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    cap = math.ceil(args.n_controls / args.n_patches) + 2
    started = clock()
    chosen: list[dict] = []
    for p in patches:
        rows, meta = archive.candidates(p, args)
        counts = {**meta, "joined_candidates": len(rows)}
        rows = [r for r in rows if r["gaia_dr3_source_id"] not in exclude]
        counts["after_hephaistos_exclusion"] = len(rows)
        _, c = uf.apply_control_cuts(rows, cuts)  # neighbours not queried yet: counts only
        counts.update({k: c[k] for k, _ in steps[:-1]})
        pre = sorted(
            (r for r in rows if all(cuts.passes(r)[:-1])),
            key=lambda r: int(r["gaia_dr3_source_id"]),
        )
        pre = [pre[i] for i in rng.permutation(len(pre))]  # seeded order, not server order
        checked = rejected = selected = 0
        for r in pre:
            if selected >= cap:
                break
            checked += 1
            if archive.n_neighbours(r, cuts.isolation) > 0:
                rejected += 1
                continue
            selected += 1
            chosen.append({**r, "n_gaia_neighbours": 0, "patch_id": p["patch_id"]})
        counts.update(
            isolation_checked=checked,
            rejected_gaia_neighbour_within_isolation=rejected,
            selected_controls=selected,
            unexamined_candidates=len(pre) - checked,
        )
        p["gaia_adql"], p["allwise_adql"] = patch_adql(p, args)
        p["cut_counts"] = counts
        print(f"      {p['patch_id']}: {counts}")
    # Hard checks: every selected star passes every cut, and none is a locked Gate #1 star.
    kept, _ = uf.apply_control_cuts(chosen, cuts)
    if len(kept) != len(chosen):
        raise ProvenanceError("a selected control fails the declared cuts")
    uf.assert_disjoint_from_locked([c["gaia_dr3_source_id"] for c in chosen], locked.star_ids)
    split = uf.split_calibration_withheld(chosen, [p["patch_id"] for p in patches], args.seed)
    controls = []
    for r in sorted(chosen, key=lambda r: int(r["gaia_dr3_source_id"])):
        ra, dec = propagate(r)
        controls.append(
            {
                "gaia_dr3_source_id": r["gaia_dr3_source_id"],
                "allwise": r["allwise"],
                "coadd_id": r["coadd_id"],
                "patch_id": r["patch_id"],
                "split": split[r["gaia_dr3_source_id"]],
                "l_deg": r["l_deg"],
                "b_deg": r["b_deg"],
                "gaia_ra_epoch_wise_deg": ra,
                "gaia_dec_epoch_wise_deg": dec,
                "phot_g_mean_mag": r["phot_g_mean_mag"],
                "bp_rp": r["bp_rp"],
                "parallax_mas": r["parallax_mas"],
                "ruwe": r["ruwe"],
                "w3snr": r["w3snr"],
                "w3mpro": r["w3mpro"],
                "cc_flags": r["cc_flags"],
                "ext_flg": r["ext_flg"],
                "n_gaia_neighbours": 0,
            }
        )
    keys = list(patches[0]["cut_counts"])
    sample = {
        "patches": patches,
        "controls": controls,
        "selection": {
            "gaia": "parallax > 3.3333 mas (d < 300 pc), parallax_over_error > 10, BP-RP >= "
            "1.84, 10 <= G <= 16, AllWISE best neighbour (gaiadr3.allwise_best_neighbour)",
            "cuts_in_order": [
                "exclude published Hephaistos candidates (adversary claim file)",
                *[text for _, text in steps],
            ],
            "isolation_adql_template": ISOLATION_ADQL,
            "avoided_fields": [
                {"name": n, "l": a, "b": b, "radius_deg": r} for n, a, b, r in uf.AVOID_FIELDS
            ],
            "patch_draw": f"isotropic, SeedSequence([{args.seed}, {uf.PATCH_STREAM}]), "
            f"{draws} draws",
            "candidate_order": f"seeded permutation per patch, SeedSequence([{args.seed}, 1])",
            "per_patch_cap": cap,
            "split": f"stratified by patch, alternating labels, SeedSequence([{args.seed}, 2])",
            "wise_mean_epoch_yr": WISE_MEAN_EPOCH_YR,
            "gate1_patch_audit": audit,
            "disjointness": f"0 of {len(controls)} fresh Gaia DR3 ids among the "
            f"{len(locked.star_ids)} locked Gate #1 / Gate #2 controls",
            "protocol": protocol(args),
            "catalog_query_window_utc": [started, clock()],
            "n_catalog_queries": len(archive.log),
            "catalog_queries_sha256": canonical_sha256(
                [{k: v for k, v in q.items() if k != "queried_at_utc"} for q in archive.log]
            ),
        },
        "cut_counts_total": {k: sum(p["cut_counts"][k] for p in patches) for k in keys},
    }
    return sample


def load_or_build_sample(archive, cache: Cache, locked: Locked, args, clock: Clock, git_sha: str):
    """The sample is built once and then only ever read back (so a resumed run cannot drift)."""
    stored = cache.read_json(SAMPLE_NAME)
    if stored is None:
        if cache.manifest:
            raise LeakageError("the cache holds cutouts but no sample: refusing to draw one now")
        sample = build_sample(archive, locked, args, clock)
        stored = {
            "sample": sample,
            "sample_sha256": canonical_sha256(sample),
            "code_git_sha": git_sha,
            "catalog_query_log": archive.log,
        }
        cache.write_once(SAMPLE_NAME, stored)
        stored = cache.read_json(SAMPLE_NAME)
    sample = stored["sample"]
    if canonical_sha256(sample) != stored["sample_sha256"]:
        raise ProvenanceError(f"{SAMPLE_NAME} does not match its own sha256")
    if sample["selection"]["protocol"] != protocol(args):
        raise LeakageError(
            f"{SAMPLE_NAME} was drawn under a different protocol; use the original settings "
            "or a new --cache-dir (a sample is never redrawn over an existing cache)"
        )
    # Re-checked on every run: the locked file is read again, never trusted from the cache.
    uf.assert_disjoint_from_locked(
        [c["gaia_dr3_source_id"] for c in sample["controls"]], locked.star_ids
    )
    uf.patch_separation_audit(
        sample["patches"],
        locked.centres,
        min_abs_b=args.min_abs_b * u.deg,
        min_locked_separation=args.min_locked_separation_deg * u.deg,
    )
    return sample, stored


# ----------------------------------------------------------- stages 1 and 2: cutouts
class SplitFetch(NamedTuple):
    cutouts: list
    files: list[dict]
    failures: list[dict]
    aborted: bool
    n_new: int


def fetch_split(
    archive,
    cache: Cache,
    controls: list[dict],
    split: str,
    args: argparse.Namespace,
    clock: Clock,
    git_sha: str,
    *,
    frozen_at: str | None = None,
) -> SplitFetch:
    """Fetch (or reuse) the cutouts of ONE split. Withheld needs the freeze timestamp."""
    if split == "withheld" and frozen_at is None:
        raise LeakageError("withheld cutouts may only be fetched after the calibration freeze")
    need = uf.Geometry.for_pixel_scale(uf.UNWISE_PIXEL_SCALE).need_pix
    cutouts, files, failures = [], [], []
    consecutive = n_new = 0
    aborted = False
    mine = sorted(
        (c for c in controls if c["split"] == split), key=lambda c: int(c["gaia_dr3_source_id"])
    )
    for c in mine:
        sid = c["gaia_dr3_source_id"]
        hit = cache.get_cutout(sid)
        if hit is None:
            capped = args.max_new_cutouts is not None and n_new >= args.max_new_cutouts
            if aborted or capped:
                reason = "circuit breaker open" if aborted else "--max-new-cutouts reached"
                failures.append(_failure(sid, split, "not_attempted", reason))
                continue
            try:
                data, url = archive.cutout(c, args.cutout_size_pix)
            except LiveFetchError as exc:
                failures.append(_failure(sid, split, "transport", str(exc)))
                consecutive += 1
                print(f"      {split[:3]} {sid}: FAILED {' '.join(str(exc).split())[:160]}")
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    aborted = True
                    print(
                        f"      {consecutive} consecutive failures: no further cutouts are "
                        "requested in this run (fix the cause, then re-run; finished cutouts "
                        "stay cached)",
                        file=sys.stderr,
                    )
                continue
            consecutive = 0
            n_new += 1
            entry = cache.put_cutout(
                sid,
                data,
                {
                    "gaia_dr3_source_id": sid,
                    "split": split,
                    "url": url,
                    "fetched_at_utc": clock(),
                    "code_git_sha": git_sha,
                },
            )
        else:
            data, entry = hit
        if entry["split"] != split:
            raise LeakageError(f"star {sid} is cached as {entry['split']}, requested as {split}")
        if split == "withheld":
            uf.assert_withheld_after_freeze(frozen_at, [entry["fetched_at_utc"]])
        try:
            parsed = parse_cutout(data, c, need)
        except LiveFetchError as exc:
            failures.append(_failure(sid, split, "unusable_content", str(exc)))
            continue
        cutouts.append(
            uf.UnwiseCutout.from_invvar(
                sid,
                split,
                parsed.flux,
                parsed.invvar,
                parsed.gaia_xy,
                uf.UNWISE_PIXEL_SCALE,
                float(c["w3snr"]),
                c["patch_id"],
                entry["fetched_at_utc"],
            )
        )
        files.append(
            {
                "gaia_dr3_source_id": sid,
                "split": split,
                "url": entry["url"],
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "fetched_at_utc": entry["fetched_at_utc"],
                "cached": hit is not None,
                "tile": parsed.tile,
                "shape": list(parsed.shape),
                "gaia_xy": list(parsed.gaia_xy),
                "pixel_scale_measured_arcsec": parsed.pixel_scale_measured_arcsec,
                "code_git_sha": entry["code_git_sha"],
            }
        )
    return SplitFetch(cutouts, files, failures, aborted, n_new)


def _failure(star_id: str, split: str, kind: str, reason: str) -> dict:
    return {
        "gaia_dr3_source_id": star_id,
        "split": split,
        "kind": kind,
        "reason": " ".join(reason.split())[:300],
    }


def seal_state(body: dict) -> dict:
    return {**body, "state_file_sha256": canonical_sha256(body)}


def load_frozen_state(state: dict, sample_sha: str, args: argparse.Namespace):
    """Rebuild the frozen method from its stored file; any mismatch is a hard refusal."""
    body = {k: v for k, v in state.items() if k != "state_file_sha256"}
    if canonical_sha256(body) != state.get("state_file_sha256"):
        raise LeakageError(f"{FROZEN_NAME} was modified after it was written")
    if state["sample_sha256"] != sample_sha or state["protocol"] != protocol(args):
        raise LeakageError(
            f"{FROZEN_NAME} belongs to a different sample or protocol; a freeze is never "
            "repeated once it exists (use the original settings or a new --cache-dir)"
        )
    if uf.frozen_state_sha256(state["frozen"]) != state["frozen_state_sha256"]:
        raise LeakageError(f"{FROZEN_NAME}: frozen blocks do not match frozen_state_sha256")
    frozen = uf.FrozenCalibration.from_payload(state["frozen"], state["frozen_at_utc"])
    if frozen.state_sha256 != state["frozen_state_sha256"]:
        raise LeakageError(f"{FROZEN_NAME}: the stored state does not rebuild to itself")
    return frozen


class GateRun(NamedTuple):
    receipt: dict
    frozen: object
    withheld: object
    calibration_cutouts: list
    withheld_cutouts: list


def run_gate(
    archive,
    cache: Cache,
    locked: Locked,
    args: argparse.Namespace,
    git: tuple[str, str],
    *,
    separations: u.Quantity | None = None,
    flux_fractions: Sequence[float] = uf.FLUX_FRACTIONS,
) -> GateRun:
    """The two-stage experiment, identical for the live archives and the synthetic rehearsal."""
    clock = Clock()
    git_sha, git_head = git
    started = clock()
    print(f"  [stage 0] Sample from {archive.label} (git_sha={git_sha}) ...")
    sample, stored = load_or_build_sample(archive, cache, locked, args, clock, git_sha)
    controls = sample["controls"]
    n_split = {s: sum(c["split"] == s for c in controls) for s in uf.SPLITS}
    print(
        f"      {len(controls)} controls in {len(sample['patches'])} patches "
        f"({n_split['calibration']} calibration / {n_split['withheld']} withheld); "
        f"{sample['selection']['disjointness']}"
    )

    print("  [stage 1] Calibration cutouts ONLY, then the freeze ...")
    state = cache.read_json(FROZEN_NAME)
    if state is None and cache.entries("withheld"):
        raise LeakageError(
            "the cache holds withheld cutouts but no frozen calibration state: the freeze "
            "can no longer precede them"
        )
    cal = fetch_split(archive, cache, controls, "calibration", args, clock, git_sha)
    transport = [f for f in cal.failures if f["kind"] != "unusable_content"]
    if transport or (state is None and len(cal.cutouts) < uf.MIN_PSF_STAMPS):
        # An incomplete calibration half is never frozen (and a frozen one is never used
        # with a cutout missing): the next run resumes from the cache.
        print(
            f"      calibration half incomplete ({len(cal.cutouts)}/{n_split['calibration']} "
            f"cutouts, {len(transport)} not fetched): nothing "
            f"{'frozen' if state is None else 'evaluated'}, no withheld cutout requested. "
            "Re-run to resume."
        )
        return GateRun(
            _partial_receipt(archive, args, git, started, clock(), sample, cal), None, None, [], []
        )
    if state is None:
        frozen = uf.freeze_calibration(
            cal.cutouts,
            locked.sigma_sys,
            target_false_alarm=args.target_false_alarm,
            conformal_alpha=args.conformal_alpha,
        ).seal(clock())
        cache.write_once(
            FROZEN_NAME,
            seal_state(
                {
                    "kind": "unwise_w3_frozen_calibration_state",
                    "data_tier": archive.data_tier,
                    "frozen_at_utc": frozen.frozen_at_utc,
                    "frozen_state_sha256": frozen.state_sha256,
                    "code_git_sha": git_sha,
                    "sample_sha256": stored["sample_sha256"],
                    "protocol": protocol(args),
                    "calibration_cutout_sha256": {
                        f["gaia_dr3_source_id"]: f["sha256"] for f in cal.files
                    },
                    "frozen": frozen.payload(),
                }
            ),
        )
        state = cache.read_json(FROZEN_NAME)
        resumed = False
    else:
        resumed = True
    # Whatever happened above, the method used from here on is the one read back from the file.
    frozen = load_frozen_state(state, stored["sample_sha256"], args)
    now = {f["gaia_dr3_source_id"]: f["sha256"] for f in cal.files}
    if any(now.get(k) != v for k, v in state["calibration_cutout_sha256"].items()):
        raise LeakageError(f"calibration cutouts changed since {FROZEN_NAME} was written")
    psf = frozen.psf
    rules = frozen.rules()["screens"]
    widths = uf.psf_widths_arcsec(psf.stamp, psf.pix)
    print(
        f"      {'RESUMED from' if resumed else 'frozen at'} {frozen.frozen_at_utc} (state "
        f"sha256 {frozen.state_sha256[:16]}..., git {state['code_git_sha'][:12]}): empirical "
        f'PSF FWHM {widths["fwhm_gaussian_arcsec"]:.3f}" (Gaussian fit), '
        f'{widths["fwhm_halfmax_arcsec"]:.3f}" (half-max area) from {len(psf.star_ids)} stamps; '
        f"covariance V(0) = {frozen.noise.variance_inflation:.3f}, lag-1 correlation "
        f"{frozen.noise.torus_covariance[0, 1] / frozen.noise.variance_inflation:.3f}"
    )
    for name in (*uf.CHI2_SCREENS, *uf.CONFORMAL_SCREENS):
        sc = rules[name]
        extra = (
            f"scale {sc['delta_chi2_scale']:.3f}"
            if name in uf.CHI2_SCREENS
            else f"k = {sc['k_exceedances']}, level (k+1)/(n_cal+1) = "
            f"{sc['order_statistic_rank_from_top']}/{sc['n_calibration'] + 1} = "
            f"{sc['alpha_guaranteed']:.4f}"
        )
        print(f"      {name}: threshold {sc['threshold']:.2f} ({extra})")

    print("  [stage 2] Withheld cutouts (after the freeze), then the two-arm injection ...")
    wh = fetch_split(
        archive, cache, controls, "withheld", args, clock, git_sha, frozen_at=frozen.frozen_at_utc
    )
    result = None
    if wh.cutouts:
        result = uf.evaluate_withheld(
            wh.cutouts, frozen, seed=args.seed, separations=separations,
            flux_fractions=flux_fractions,
        )  # fmt: skip
    finished = clock()
    files = [*cal.files, *wh.files]
    failures = [*cal.failures, *wh.failures]
    blocked = [f for f in failures if f["kind"] != "unusable_content"]
    status = "ok" if result is not None and not blocked else "incomplete"
    receipt = {
        "kind": KIND,
        "data_tier": archive.data_tier,
        "status": status,
        "rule_version": uf.RULE_VERSION,
        "code_git_sha": git_sha,
        "git_head": git_head,
        "rng_seed": args.seed,
        "query_window_utc": [started, finished],
        "frozen_before_withheld_fetch": bool(wh.files)
        and all(
            uf.parse_utc(f["fetched_at_utc"], "fetch") > uf.parse_utc(frozen.frozen_at_utc, "f")
            for f in wh.files
        ),
        "disjoint_from_gate1_sha256": locked.calibration.file_sha256,
        "locked_inputs": {
            "sigma_sys_calibration_path": locked.calibration.path,
            "sigma_sys_calibration_sha256": locked.calibration.file_sha256,
            "sigma_sys_arcsec": float(locked.sigma_sys.to_value(u.arcsec)),
            "n_locked_stars": len(locked.star_ids),
            "n_locked_patches": len(locked.patch_ids),
        },
        "locked_artifacts_verified": locked.artifacts,
        "sample": sample,
        "endpoints": ENDPOINTS,
        "fetch": {
            "service": archive.label,
            "url_template": CUTOUT_URL_TEMPLATE,
            "cutout_size_pix": args.cutout_size_pix,
            "n_requested": len(controls),
            "n_ok": len(files),
            "n_failed": len(failures),
            "files": files,
            "failures": failures,
            "cache_manifest_sha256": canonical_sha256(
                {f["gaia_dr3_source_id"]: f["sha256"] for f in files}
            ),
        },
        "frozen_state_file": {
            "name": FROZEN_NAME,
            "state_file_sha256": state["state_file_sha256"],
            "code_git_sha": state["code_git_sha"],
            "resumed": resumed,
        },
        "split_counts": {
            "n_calibration_controls": n_split["calibration"],
            "n_withheld_controls": n_split["withheld"],
            "n_calibration_evaluated": len(frozen.star_ids),
            "n_withheld_evaluated": len(result.star_ids) if result is not None else 0,
            "n_psf_stamps": len(psf.star_ids),
            "n_cutouts_fetched_this_run": cal.n_new + wh.n_new,
            "aborted_after_consecutive_failures": cal.aborted or wh.aborted,
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "caveats": CAVEATS,
        "catalog_query_log": stored["catalog_query_log"],
        **frozen.payload(),
        "frozen_at_utc": frozen.frozen_at_utc,
        "frozen_state_sha256": frozen.state_sha256,
        "withheld": result.as_dict() if result is not None else None,
    }
    return GateRun(receipt, frozen, result, cal.cutouts, wh.cutouts)


def _partial_receipt(archive, args, git, started, finished, sample, cal: SplitFetch) -> dict:
    return {
        "kind": KIND,
        "data_tier": archive.data_tier,
        "status": "incomplete",
        "code_git_sha": git[0],
        "git_head": git[1],
        "rng_seed": args.seed,
        "query_window_utc": [started, finished],
        "frozen_before_withheld_fetch": False,
        "sample": sample,
        "fetch": {"n_ok": len(cal.files), "n_failed": len(cal.failures), "failures": cal.failures},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "withheld": None,
    }


# ----------------------------------------------------------- artifact & promote
ARTIFACT_KEYS = (
    "data_tier",
    "rule_version",
    "code_git_sha",
    "rng_seed",
    "query_window_utc",
    "frozen_at_utc",
    "frozen_state_sha256",
    "frozen_before_withheld_fetch",
    "disjoint_from_gate1_sha256",
    "locked_inputs",
    "sample",
    "endpoints",
    "fetch",
    "geometry",
    "empirical_psf",
    "noise_model",
    "frozen_screens",
    "withheld",
    "config",
    "caveats",
)
DESCRIPTION = (
    "Sprint 3 (v0.2.2): frozen-method blend screens on unsmoothed unWISE W3 coadds (Lang "
    "2014, 2.75 arcsec pixels) of a fresh |b| > 30 deg sample of isolated Gaia DR3 M dwarfs "
    "sharing no patch and no star with Gate #1 / Gate #2. Empirical PSF, pixel-noise "
    "covariance and every threshold (scalar-variance chi^2, GLS chi^2, conformal order "
    "statistic) frozen on the calibration half before any withheld cutout was fetched; "
    "withheld two-arm injection (off-center blends vs matched centered excess)."
)


def artifact_doc(receipt: dict, source_receipt: dict | None) -> dict:
    """The promoted-file layout of a receipt (sealed; not yet validated)."""
    doc = {
        "id": ARTIFACT_ID,
        "schema_version": 1,
        "description": DESCRIPTION,
        **{k: receipt[k] for k in ARTIFACT_KEYS},
    }
    if source_receipt is not None:
        doc["source_receipt"] = source_receipt
    return seal_unwise_gate(doc)


def promotion_problems(rec: dict) -> list[str]:
    """Every reason a receipt may not be promoted (cheap, receipt-level checks)."""
    problems = []
    if rec.get("kind") != KIND:
        problems.append(f"not a {KIND} receipt")
    if rec.get("data_tier") != "reproduced_archive":
        problems.append(
            f"data_tier {rec.get('data_tier')!r} is not reproduced_archive (a synthetic "
            "rehearsal payload can never be promoted)"
        )
    if rec.get("status") != "ok":
        problems.append(f"receipt status {rec.get('status')!r}")
    clean = re.compile(r"^[0-9a-f]{40}$")
    shas = {
        "code_git_sha": rec.get("code_git_sha"),
        "frozen state": (rec.get("frozen_state_file") or {}).get("code_git_sha"),
        **{
            f"cutout {f.get('gaia_dr3_source_id')}": f.get("code_git_sha")
            for f in (rec.get("fetch") or {}).get("files", [])
        },
    }
    dirty = sorted({str(v) for v in shas.values() if not clean.match(str(v))})
    if dirty:
        problems.append(f"not built from a clean commit (git sha {dirty})")
    if rec.get("frozen_before_withheld_fetch") is not True:
        problems.append("leakage: the freeze did not precede every withheld fetch")
    counts = rec.get("split_counts") or {}
    n_cal, n_wh = counts.get("n_calibration_evaluated", 0), counts.get("n_withheld_evaluated", 0)
    if n_cal < UNWISE_GATE_MIN_SPLIT or n_wh < UNWISE_GATE_MIN_SPLIT:
        problems.append(
            f"undersized split: {n_cal} calibration / {n_wh} withheld stars evaluated, at "
            f"least {UNWISE_GATE_MIN_SPLIT} required in each half"
        )
    if (rec.get("locked_artifacts_verified") or {}) != LOCKED_ARTIFACTS:
        problems.append("receipt was not built against the four locked artifacts")
    return problems


def promote(receipt_path: Path, out_file: Path = OUT_FILE, root: Path = ROOT) -> int:
    raw = receipt_path.read_bytes()
    rec = json.loads(raw)
    problems = promotion_problems(rec)
    doc = out_bytes = None
    if not problems:
        try:
            rel = receipt_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = receipt_path.name
        try:
            locked = load_locked(root)
            doc = artifact_doc(rec, {"path": rel, "sha256": hashlib.sha256(raw).hexdigest()})
            out_bytes = unwise_gate_bytes(doc)
            if len(out_bytes) >= UNWISE_GATE_MAX_BYTES:
                problems.append(
                    f"oversized output: {len(out_bytes)} bytes is not < {UNWISE_GATE_MAX_BYTES} "
                    "(5 MB)"
                )
            else:
                # Validate what will actually be on disk: the serialised bytes, re-read.
                on_disk = json.loads(out_bytes)
                model = UnwiseGateWithheldFile.model_validate(on_disk)
                verify_unwise_gate(on_disk, model, locked.calibration)
        except (FermiDeskError, KeyError, ValueError) as exc:
            kind = "leakage violation: " if isinstance(exc, LeakageError) else ""
            problems.append(f"{kind}{type(exc).__name__}: {exc}")
    if problems or doc is None or out_bytes is None:
        print("Refusing to promote: " + "; ".join(problems), file=sys.stderr)
        return 1
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(out_bytes)
    sha = hashlib.sha256(out_bytes).hexdigest()
    print(f"Promoted -> {out_file.as_posix()} ({len(out_bytes)} bytes, < 5 MB)")
    print(f"  file sha256    {sha}")
    print(f"  content_sha256 {doc['content_sha256']}")
    print(f"SHA-256 pin for data/surveys/{IR_SURVEY}.yaml (inside the `infrared:` block):")
    print(f'  unwise_gate_path: "{UNWISE_GATE_PATH}"')
    print(f'  unwise_gate_sha256: "{sha}"')
    return 0


# ---------------------------------------------------------------------- report
def _fmt_rate(r: dict) -> str:
    lo, hi = r["clopper_pearson_95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.3f} (95% CP {lo:.3f}-{hi:.3f})"


def report(summary: dict, indent: str = "      ") -> None:
    for name, v in summary["screens"].items():
        print(f"{indent}[{name}] {v['role']}")
        print(
            f"{indent}  withheld null false alarm {_fmt_rate(v['null_false_alarm'])}; Arm A "
            f'blend survival within 3.25": {v["blend_survival_within_3p25_arcsec"]:.3f}; Arm B '
            f"centered-excess false rejection {_fmt_rate(v['arm_b_false_rejection_pooled'])}"
        )


def write_receipt(receipt: dict, out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{KIND}-seed{seed}-{stamp}.json"
    out.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------- dry run
def rehearse(
    args: argparse.Namespace,
    locked: Locked | None = None,
    cache: Cache | None = None,
    *,
    n_stars: int | None = None,
    separations: u.Quantity | None = None,
    flux_fractions: Sequence[float] = uf.FLUX_FRACTIONS,
) -> GateRun:
    """The full two-stage pipeline on SYNTHETIC cutouts, in memory (``synthetic_proxy``)."""
    n = int(args.rehearsal_stars if n_stars is None else n_stars)
    sized = argparse.Namespace(**{**vars(args), "n_controls": n})
    archive = SyntheticArchive(args.seed, n, args.n_patches)
    return run_gate(
        archive,
        cache if cache is not None else Cache(None),
        locked if locked is not None else load_locked(),
        sized,
        git_state(ROOT),
        separations=separations,
        flux_fractions=flux_fractions,
    )


def leakage_guards(run: GateRun, locked: Locked) -> list[tuple[str, bool]]:
    """Every zero-leakage guard, provoked on purpose: each must raise ``LeakageError``."""
    cal, wh, frozen = run.calibration_cutouts, run.withheld_cutouts, run.frozen
    fresh = [c["gaia_dr3_source_id"] for c in run.receipt["sample"]["controls"]]
    unsealed = replace(frozen, frozen_at_utc=None)
    early = replace(wh[0], fetched_at_utc=cal[0].fetched_at_utc)
    relabelled = replace(cal[0], split="withheld", fetched_at_utc=wh[0].fetched_at_utc)
    on_locked = [
        {
            "patch_id": "X01",
            "l_deg": float(locked.centres[0].l.deg),
            "b_deg": float(locked.centres[0].b.deg),
            "radius_deg": 2.0,
        }
    ]
    doc = artifact_doc(run.receipt, None)
    late = json.loads(json.dumps(doc))
    late["withheld"]["fetched_at_utc"][0] = late["frozen_screens"]["calibration"]["fetched_at_utc"][
        0
    ]
    cases = {
        "a locked Gate #1 star in the fresh sample": lambda: uf.assert_disjoint_from_locked(
            [*fresh, str(locked.star_ids[0])], locked.star_ids
        ),
        "a fresh patch on a locked Gate #1 patch": lambda: uf.patch_separation_audit(
            on_locked, locked.centres, min_abs_b=0.0 * u.deg
        ),
        "PSF stack given a withheld cutout": lambda: uf.stack_empirical_psf([*cal, wh[0]]),
        "pixel covariance given a withheld cutout": lambda: uf.collect_noise_samples([*cal, wh[0]]),
        "thresholds frozen on withheld cutouts": lambda: uf.freeze_calibration(
            wh, locked.sigma_sys
        ),
        "withheld evaluation before the freeze is sealed": lambda: uf.evaluate_withheld(
            wh[:1], unsealed
        ),
        "a withheld cutout fetched before the freeze": lambda: uf.evaluate_withheld(
            [early], frozen
        ),
        "a calibration star relabelled withheld": lambda: uf.evaluate_withheld(
            [relabelled], frozen
        ),
        "a second freeze of a sealed state": lambda: frozen.seal(frozen.frozen_at_utc),
        "an artifact whose withheld fetch precedes the freeze": lambda: (
            UnwiseGateWithheldFile.model_validate(seal_unwise_gate(late))
        ),
    }
    out = []
    for name, call in cases.items():
        try:
            call()
        except LeakageError:
            out.append((name, True))
        else:
            out.append((name, False))
    return out


def psf_discrimination(seed: int, n: int = 16) -> dict:
    """Stacked FWHM of synthetic unsmoothed (6.5") vs Atlas-like (13.69") stars."""
    out = {}
    for label, fwhm in (
        ("unsmoothed", uf.NOMINAL_W3_FWHM_ARCSEC * u.arcsec),
        ("atlas_like", ATLAS_LIKE_FWHM),
    ):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x505346, len(label)]))
        cuts = [
            uf.synthetic_cutout(rng, str(SYNTHETIC_ID_BASE + i), "calibration", psf_fwhm=fwhm)
            for i in range(n)
        ]
        psf = uf.stack_empirical_psf(cuts).psf()
        out[label] = {
            "true_fwhm_arcsec": fwhm.to_value(u.arcsec),
            **uf.psf_widths_arcsec(psf.stamp, psf.pix),
        }
    return out


def dry_run(args: argparse.Namespace) -> int:
    locked = load_locked()
    print("[1/6] Locked artifacts (SHA-256 verified, read-only):")
    for rel, sha in locked.artifacts.items():
        print(f"      {rel}  {sha}")
    atlas = locked.atlas_fwhm_arcsec
    print(
        f"      Gate #1 sample: {len(locked.star_ids)} stars in {len(locked.patch_ids)} patches, "
        f'locked sigma_sys = {locked.sigma_sys.to_value(u.arcsec):.4f}"; Gate #2 Atlas PSF FWHM '
        f'{atlas:.3f}"'
    )

    patches, draws, audit = draw_patches(args, locked)
    print(
        f"[2/6] {len(patches)} fresh patches (seed {args.seed}, {draws} draws; whole cone at "
        f"|b| > {args.min_abs_b:g} deg, LMC/SMC/M31 avoided, >= "
        f"{args.min_patch_separation_deg:g} deg apart):"
    )
    for p in patches:
        print(
            f"      {p['patch_id']}: l={p['l_deg']:7.2f} b={p['b_deg']:+7.2f} deg  nearest "
            f"Gate #1 patch {p['nearest_locked_patch_id']} at "
            f"{p['nearest_locked_patch_separation_deg']:6.2f} deg"
        )
    print(
        f"      disjoint from all {audit['n_locked_patches']} Gate #1 patches: minimum "
        f"separation {audit['min_locked_separation_deg']:.2f} deg (required >= "
        f"{audit['min_locked_separation_required_deg']:g} deg)"
    )

    g_adql, a_adql = patch_adql(patches[0], args)
    cut = ENDPOINTS["unwise_cutout"]
    print(
        f"[3/6] Queries a live --fetch would send (NOT sent now). Patch {patches[0]['patch_id']}:"
    )
    print(f"      Gaia    [{ARI_TAP}]\n        {g_adql}")
    print(f"      AllWISE [{IRSA_TAP}]\n        {a_adql}")
    print(f"      Isolation, per candidate [{ARI_TAP}]\n        {ISOLATION_ADQL}")
    print(f"      unWISE W3 cutout, per star (verified live: {cut['verified_live']}):")
    print(f"        {cut['url_template']}")
    print(
        f"        e.g. {cutout_url(patches[0]['ra_deg'], patches[0]['dec_deg'], args.cutout_size_pix)}"
    )
    for k, v in cut["parameters"].items():
        print(f"        {k:14s} {v}")
    print(f"        reply          {cut['expected_reply']}")
    print(f"        tile choice    {cut['tile_choice']}")
    for name, fb in ENDPOINTS["fallbacks"].items():
        print(f"      fallback {name}: {fb['url_template']}\n        {fb['status']}")

    print(f"[4/6] Cache {args.cache_dir.as_posix()} (gitignored): {cache_report(args.cache_dir)}")

    print(
        f"[5/6] SYNTHETIC rehearsal (synthetic_proxy, in memory): {args.rehearsal_stars} stars, "
        f'{uf.NOMINAL_W3_FWHM_ARCSEC}" PSF, pixel noise correlated over '
        f'{REHEARSAL_NOISE_CORRELATION.to_value(u.arcsec):g}" FWHM'
    )
    run = rehearse(args, locked)
    receipt = run.receipt
    if run.withheld is None or receipt["status"] != "ok":
        print("DRY RUN FAILED: the synthetic rehearsal did not complete", file=sys.stderr)
        return 1
    report(receipt["withheld"]["summary"])
    widths = psf_discrimination(args.seed)
    print(
        "      PSF discrimination (stacked Gaussian-fit FWHM): unsmoothed "
        f'{widths["unsmoothed"]["fwhm_gaussian_arcsec"]:.2f}" vs Atlas-like '
        f'{widths["atlas_like"]["fwhm_gaussian_arcsec"]:.2f}" (true 6.50" / 13.69")'
    )
    guards = leakage_guards(run, locked)
    for name, ok in guards:
        print(f"      LeakageError guard - {name}: {'raised' if ok else 'NOT RAISED'}")

    doc = artifact_doc(receipt, None)
    model = UnwiseGateWithheldFile.model_validate(doc)
    size = len(unwise_gate_bytes(doc))
    detail = verify_unwise_gate(doc, model, locked.calibration)
    refused = promotion_problems(receipt)
    print(
        f"[6/6] Schema UnwiseGateWithheldFile valid ({model.data_tier}), {size} bytes (< 5 MB); "
        f"offline recomputation: {detail}"
    )
    print(f"      --promote-from would refuse this payload: {refused[0]}")
    psf_fwhm = receipt["empirical_psf"]["fwhm_gaussian_arcsec"]
    ok = (
        all(flag for _, flag in guards)
        and abs(psf_fwhm - uf.NOMINAL_W3_FWHM_ARCSEC) < 0.6
        and widths["atlas_like"]["fwhm_gaussian_arcsec"] > 2 * psf_fwhm - 1.0
        and size < UNWISE_GATE_MAX_BYTES
        and any("synthetic" in p for p in refused)
        and receipt["frozen_before_withheld_fetch"] is True
    )
    if not ok:
        print("DRY RUN FAILED: rehearsal invariants violated", file=sys.stderr)
        return 1
    print("DRY RUN OK: zero outbound network requests, no files written.")
    return 0


# ------------------------------------------------------------------ live fetch
def live_fetch(args: argparse.Namespace) -> int:
    global _NETWORK_ALLOWED
    git = git_state(ROOT)
    if git[0] in ("dirty", "unknown") and not args.allow_dirty:
        print(
            f"Refusing a live fetch from a {git[0]} git tree: every archive experiment needs a "
            "clean committed code_git_sha before querying external servers (V0_2_ROADMAP "
            "guardrail 3). Commit first, or pass --allow-dirty for a smoke test whose receipt "
            "can never be promoted.",
            file=sys.stderr,
        )
        return 2
    locked = load_locked()
    print(
        f"[1/2] Locked artifacts verified; cache {args.cache_dir.as_posix()}: "
        f"{cache_report(args.cache_dir)}"
    )
    cache = Cache(args.cache_dir)
    _NETWORK_ALLOWED = True
    try:
        run = run_gate(LiveArchive(args.sleep, Clock()), cache, locked, args, git)
    finally:
        _NETWORK_ALLOWED = False
    if run.withheld is not None:
        report(run.receipt["withheld"]["summary"])
    out = write_receipt(run.receipt, args.out, args.seed)
    print(f"[2/2] Receipt ({run.receipt['status']}): {out.as_posix()}")
    return 0 if run.receipt["status"] == "ok" else 1


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="offline rehearsal, no network")
    mode.add_argument("--fetch", action="store_true", help="LIVE archive queries (opt-in)")
    mode.add_argument("--promote-from", type=Path, default=None, metavar="RECEIPT")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-controls", type=int, default=160, help="target number of controls")
    ap.add_argument("--n-patches", type=int, default=16)
    ap.add_argument("--patch-radius-deg", type=float, default=2.0)
    ap.add_argument("--min-abs-b", type=float, default=30.0)
    ap.add_argument("--min-locked-separation-deg", type=float, default=15.0)
    ap.add_argument("--min-patch-separation-deg", type=float, default=20.0)
    ap.add_argument("--w3-snr-min", type=float, default=8.0)
    ap.add_argument("--w3-snr-max", type=float, default=20.0)
    ap.add_argument("--ruwe-max", type=float, default=1.4)
    ap.add_argument("--isolation-arcsec", type=float, default=15.0)
    ap.add_argument("--target-false-alarm", type=float, default=uf.RAYLEIGH_3SIGMA_FALSE_ALARM)
    ap.add_argument(
        "--conformal-alpha",
        type=float,
        default=uf.DEFAULT_CONFORMAL_ALPHA,
        help="nominal level of the conformal screens; the frozen level is (k+1)/(n_cal+1)",
    )
    ap.add_argument("--cutout-size-pix", type=int, default=45)
    ap.add_argument("--rehearsal-stars", type=int, default=100, help="dry-run sample size")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="receipt directory")
    ap.add_argument("--out-file", type=Path, default=OUT_FILE, help="promoted artifact path")
    ap.add_argument(
        "--max-new-cutouts", type=int, default=None, help="cap cutouts fetched per split per run"
    )
    ap.add_argument("--sleep", type=float, default=0.5, help="pause between live requests [s]")
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
