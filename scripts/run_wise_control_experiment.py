#!/usr/bin/env python3
"""Opt-in LIVE experiment: empirical W3 centroid registration floor on WISE control stars.

Queue job ``wise_independent_control_fields`` (Tier-1 archival); Public-Launch Gate #1.
Never imported or run by pytest. Milestone 2.5 design:

1. **Multi-patch sampling.** ``--n-patches`` (default 12) seeded, mutually separated
   sky patches whose entire cone lies at |b| > ``--min-abs-b`` (default 20 deg), away
   from the Magellanic Clouds and M31, so no single tile or crowded field dominates.
2. **Control population.** Gaia DR3 M dwarfs within 300 pc (parallax > 3.333 mas,
   parallax_over_error > 10, BP-RP >= 1.84, 10 <= G <= 16) with an AllWISE best
   neighbour (ARI Gaia TAP mirror), excluding the published Hephaistos candidates.
3. **Quality / isolation / binary cuts** (each count recorded per patch):
   star-level |b| > min_abs_b; Gaia RUWE < 1.4 (unresolved binaries); AllWISE W3 SNR in
   the drill-down regime (default [8, 20], IRSA TAP); ``cc_flags == '0000'``;
   ``ext_flg == 0``; no other Gaia DR3 source within 15 arcsec.
4. **Measurement.** 60 arcsec AllWISE Atlas W3 cutout (IRSA SIA + IBE); Gaussian-
   windowed W3 centroid; offset from the Gaia position propagated to the AllWISE mean
   epoch.
5. **Deterministic 50/50 split** (stratified by patch, seeded): ``sigma_sys`` is
   estimated on the *calibration* half only (heteroscedastic Rayleigh MLE, the
   estimator in ``fermi_desk.models.hephaistos``); the 3-sigma centroid false-alarm
   rate is evaluated strictly out-of-sample on the *withheld* half.

Every ADQL text, patch centre, cut count and HTTP query is recorded in a
``reproduced_archive`` receipt under ``runs/wise_control/``. ``--promote-from`` turns a
verified receipt into ``data/calibrations/wise_sigma_sys_control.json``.

TLS verification is never disabled. Usage:
    uv run python scripts/run_wise_control_experiment.py --n-controls 120 --seed 42
    uv run python scripts/run_wise_control_experiment.py --promote-from runs/wise_control/<r>.json
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
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from fermi_desk.inference.engine import canonical_sha256, git_state
from fermi_desk.models import hephaistos

ROOT = Path(__file__).resolve().parents[1]
ARI_TAP = "https://gaia.ari.uni-heidelberg.de/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_SIA = "https://irsa.ipac.caltech.edu/SIA"
UA = "fermi-desk-wise-control/0.2 (MIT)"
W3_FWHM_ARCSEC = 6.5  # Ren et al. 2026 Table 2
W3_PIXEL_ARCSEC = 1.375  # AllWISE Atlas pixel scale
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
ALLWISE_MEAN_EPOCH_YR = 2010.5  # AllWISE combines WISE cryo + NEOWISE-post-cryo (2010)
LEGACY_SIGMA_SYS_ARCSEC = 0.5  # Milestone 2 drill-down default (legacy baseline)
PRIMARY = "robust_median"
PRIMARY_RATIONALE = (
    "Fixed BEFORE the seed-42 production run, after a seed-7 two-patch smoke test showed a "
    "'clean' control with a 6.39 arcsec W3 offset (an unseen background source). The Rayleigh "
    "MLE / moment estimators are not outlier-robust: such stars would inflate sigma_sys and "
    "blind the centroid screen to the blends it exists to catch. The median-matched estimator "
    "(50% breakdown) is primary; MLE and moment values are reported alongside. Withheld "
    "false-alarm counts include every control, so contamination is reported, not removed."
)
CLAIM = ROOT / "data/adversary_claims/hephaistos_candidates_d_e.yaml"
CALIBRATION_OUT = ROOT / "data/calibrations/wise_sigma_sys_control.json"
W3_URL_RE = re.compile(
    r"https://irsa\.ipac\.caltech\.edu/ibe/data/wise/allwise/[^,\s\"]+-w3-int-3\.fits"
)
# Crowded extragalactic fields excluded from patch centres: (name, l, b, radius deg).
AVOID = (("LMC", 280.47, -32.89, 12.0), ("SMC", 302.81, -44.33, 8.0), ("M31", 121.17, -21.57, 5.0))

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
IRSA_ADQL = (
    "SELECT designation, ra, dec, w3snr, w3mpro, w3sigmpro, cc_flags, ext_flg, nb, na, ph_qual "
    "FROM allwise_p3as_psd WHERE CONTAINS(POINT('ICRS', ra, dec), "
    "CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {r:.4f})) = 1 "
    "AND w3snr >= {snr_lo:g} AND w3snr <= {snr_hi:g}"
)
ISOLATION_ADQL = (
    "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS("
    "POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.7f}))"
)


class LiveFetchError(RuntimeError):
    pass


# ------------------------------------------------------------------- HTTP / TAP
def _get(url: str, data: bytes | None = None, timeout: int = 300, tries: int = 3):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:600].decode("utf-8", errors="replace")
            if exc.code < 500 or attempt == tries:
                raise LiveFetchError(
                    f"{urllib.parse.urlparse(url).netloc}: HTTP {exc.code}: {detail}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == tries:
                raise LiveFetchError(f"{urllib.parse.urlparse(url).netloc}: {exc}") from exc
        time.sleep(5.0 * attempt)  # transient server error: back off and retry
    raise AssertionError("unreachable")


def tap_csv(endpoint: str, adql: str, log: list, record_text: bool = True, **meta) -> pd.DataFrame:
    body = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
    ).encode()
    raw, ctype = _get(endpoint, body)
    if "csv" not in ctype.lower():
        raise LiveFetchError(f"{endpoint}: expected CSV, got {ctype!r}: {raw[:200]!r}")
    df = pd.read_csv(io.BytesIO(raw), dtype={"cc_flags": str, "ph_qual": str})
    entry = {
        "endpoint": endpoint,
        "queried_at_utc": datetime.now(UTC).isoformat(),
        "adql_sha256": hashlib.sha256(adql.encode()).hexdigest(),
        "n_rows": len(df),
        **meta,
    }
    if record_text:
        entry["adql"] = adql
    log.append(entry)
    return df


# --------------------------------------------------------------------- patches
def choose_patches(
    seed: int, n: int, radius: float, min_abs_b: float, min_sep: float
) -> tuple[list[dict], int]:
    """Seeded isotropic draws with |b_centre| >= min_abs_b + radius and pairwise separation."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    b_min = min_abs_b + radius
    centres: list[SkyCoord] = []
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < 100_000:
        attempts += 1
        l_deg = float(rng.uniform(0.0, 360.0))
        b_deg = float(np.degrees(np.arcsin(rng.uniform(-1.0, 1.0))))
        if abs(b_deg) < b_min:
            continue
        c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
        if any(
            c.separation(SkyCoord(l=al * u.deg, b=ab * u.deg, frame="galactic")).deg < ar + radius
            for _, al, ab, ar in AVOID
        ):
            continue
        if any(c.separation(o).deg < min_sep for o in centres):
            continue
        centres.append(c)
        icrs = c.icrs
        out.append(
            {
                "patch_id": f"P{len(out) + 1:02d}",
                "l_deg": l_deg,
                "b_deg": b_deg,
                "ra_deg": float(icrs.ra.deg),
                "dec_deg": float(icrs.dec.deg),
                "radius_deg": radius,
            }
        )
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} patches satisfy the separation constraints")
    return out, attempts


# ----------------------------------------------------------------- measurement
def propagate(row) -> tuple[float, float]:
    """Gaia position at the AllWISE mean epoch (proper motion only)."""
    dt = ALLWISE_MEAN_EPOCH_YR - float(row.ref_epoch)
    pmra = 0.0 if pd.isna(row.pmra) else float(row.pmra)
    pmdec = 0.0 if pd.isna(row.pmdec) else float(row.pmdec)
    dec = float(row.dec) + pmdec * dt / 3.6e6
    ra = float(row.ra) + pmra * dt / 3.6e6 / math.cos(math.radians(float(row.dec)))
    return ra, dec


def w3_cutout(ra: float, dec: float, log: list) -> tuple[np.ndarray, WCS, str]:
    q = urllib.parse.urlencode(
        {"COLLECTION": "wise_allwise", "POS": f"circle {ra} {dec} 0.002", "RESPONSEFORMAT": "CSV"}
    )
    raw, _ = _get(f"{IRSA_SIA}?{q}")
    # The SIA CSV carries a JSON `cloud_access` column with embedded commas, so the
    # W3 atlas URL is extracted with a strict pattern instead of a CSV parse.
    urls = sorted(set(W3_URL_RE.findall(raw.decode("utf-8", errors="replace"))))
    if not urls:
        raise LiveFetchError(f"no AllWISE W3 atlas image at ({ra}, {dec})")
    cut = f"{urls[0]}?center={ra},{dec}&size=60arcsec&gzip=false"
    data, ctype = _get(cut)
    if not data.startswith(b"SIMPLE"):
        raise LiveFetchError(f"cutout is not FITS ({ctype}): {data[:120]!r}")
    log.append(
        {
            "endpoint": cut,
            "queried_at_utc": datetime.now(UTC).isoformat(),
            "bytes": len(data),
            "fits_sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    with fits.open(io.BytesIO(data)) as hdul:
        img = np.asarray(hdul[0].data, float)
        wcs = WCS(hdul[0].header)
    return img, wcs, urls[0]


def windowed_centroid(img: np.ndarray, x0: float, y0: float, s_pix: float) -> tuple[float, float]:
    """Gaussian-windowed (SExtractor-style) centroid, same estimator as models/hephaistos."""
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(float)
    data = np.nan_to_num(img - np.nanmedian(img))
    x, y = x0, y0
    for _ in range(20):
        w = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s_pix**2)) * data
        norm = w.sum()
        if abs(norm) < 1e-12:
            break
        dx = 2 * (w * (xx - x)).sum() / norm
        dy = 2 * (w * (yy - y)).sum() / norm
        x, y = x + float(np.clip(dx, -1, 1)), y + float(np.clip(dy, -1, 1))
    return x, y


def measure(row, ra: float, dec: float, log: list) -> dict:
    s_pix = W3_FWHM_ARCSEC / FWHM_PER_SIGMA / W3_PIXEL_ARCSEC
    img, wcs, url = w3_cutout(ra, dec, log)
    gx, gy = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    core = (xx - gx) ** 2 + (yy - gy) ** 2 <= (3 * W3_FWHM_ARCSEC / W3_PIXEL_ARCSEC) ** 2
    if not np.all(np.isfinite(img[core])):
        raise LiveFetchError("non-finite pixels within 3 FWHM of the star")
    cx, cy = windowed_centroid(img, gx, gy, s_pix)
    if not (0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]):
        raise LiveFetchError("centroid left the cutout")
    cra, cdec = (float(v) for v in wcs.pixel_to_world_values(cx, cy))
    off = SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(cra * u.deg, cdec * u.deg))
    aw = SkyCoord(float(row.ra_aw) * u.deg, float(row.dec_aw) * u.deg)
    return {
        "w3_centroid_ra_deg": cra,
        "w3_centroid_dec_deg": cdec,
        "offset_arcsec": float(off.to_value(u.arcsec)),
        "sigma_stat_arcsec": W3_FWHM_ARCSEC / (FWHM_PER_SIGMA * float(row.w3snr)),
        "allwise_catalog_offset_arcsec": float(
            SkyCoord(ra * u.deg, dec * u.deg).separation(aw).to_value(u.arcsec)
        ),
        "image": url,
    }


# ------------------------------------------------------------------- one patch
def run_patch(patch: dict, args, exclude: set[int], log: list, rng: np.random.Generator) -> dict:
    ra, dec, r = patch["ra_deg"], patch["dec_deg"], patch["radius_deg"]
    g_adql = GAIA_ADQL.format(ra=ra, dec=dec, r=r)
    i_adql = IRSA_ADQL.format(ra=ra, dec=dec, r=r, snr_lo=args.w3_snr_min, snr_hi=args.w3_snr_max)
    patch["gaia_adql"], patch["irsa_adql"] = g_adql, i_adql
    gaia = tap_csv(ARI_TAP, g_adql, log, patch_id=patch["patch_id"], role="gaia_m_dwarfs")
    wise = tap_csv(IRSA_TAP, i_adql, log, patch_id=patch["patch_id"], role="allwise_w3_band")
    counts = {"gaia_m_dwarfs_in_cone": len(gaia), "allwise_w3_band_rows_in_cone": len(wise)}
    gaia = gaia[gaia["b"].abs() > args.min_abs_b]
    counts["after_star_abs_b_cut"] = len(gaia)
    gaia = gaia[~gaia["source_id"].isin(exclude)]
    counts["after_hephaistos_exclusion"] = len(gaia)
    gaia = gaia[gaia["ruwe"] < args.ruwe_max]
    counts["after_ruwe_cut"] = len(gaia)
    wise = wise.rename(columns={"ra": "ra_aw", "dec": "dec_aw"})
    m = gaia.merge(wise, left_on="original_ext_source_id", right_on="designation", how="inner")
    counts["after_w3_snr_band_match"] = len(m)
    m = m[m["cc_flags"].astype(str) == "0000"]
    counts["after_cc_flags_0000"] = len(m)
    m = m[pd.to_numeric(m["ext_flg"], errors="coerce") == 0]
    counts["after_ext_flg_0"] = len(m)
    # Seeded processing order (independent of server row order).
    m = m.sort_values("source_id", kind="mergesort").reset_index(drop=True)
    m = m.iloc[rng.permutation(len(m))].reset_index(drop=True)
    patch["_candidates"] = m
    patch["cut_counts"] = counts
    patch["isolation"] = {"checked": 0, "rejected_neighbour_within_15arcsec": 0}
    patch["measurement_failures"] = []
    patch["_cursor"] = 0
    patch["controls"] = []
    return patch


def fill_patch(patch: dict, cap: int, args, log: list) -> None:
    """Advance through the seeded candidate list until ``cap`` controls are measured."""
    m = patch["_candidates"]
    while len(patch["controls"]) < cap and patch["_cursor"] < len(m):
        row = m.iloc[patch["_cursor"]]
        patch["_cursor"] += 1
        iso = tap_csv(
            ARI_TAP,
            ISOLATION_ADQL.format(ra=row.ra, dec=row.dec, r=args.isolation_arcsec / 3600.0),
            log,
            record_text=False,
            patch_id=patch["patch_id"],
            role="isolation",
            gaia_dr3_source_id=int(row.source_id),
        )
        patch["isolation"]["checked"] += 1
        if int((iso["source_id"] != row.source_id).sum()) > 0:
            patch["isolation"]["rejected_neighbour_within_15arcsec"] += 1
            continue
        ra, dec = propagate(row)
        try:
            meas = measure(row, ra, dec, log)
        except LiveFetchError as exc:
            patch["measurement_failures"].append(
                {"gaia_dr3_source_id": int(row.source_id), "reason": str(exc)[:200]}
            )
            continue
        patch["controls"].append(
            {
                "gaia_dr3_source_id": int(row.source_id),
                "allwise": str(row.designation),
                "patch_id": patch["patch_id"],
                "l_deg": float(row.l),
                "b_deg": float(row.b),
                "gaia_ra_epoch_allwise_deg": ra,
                "gaia_dec_epoch_allwise_deg": dec,
                "phot_g_mean_mag": float(row.phot_g_mean_mag),
                "bp_rp": float(row.bp_rp),
                "parallax_mas": float(row.parallax),
                "ruwe": float(row.ruwe),
                "w3snr": float(row.w3snr),
                "w3mpro": float(row.w3mpro),
                "cc_flags": str(row.cc_flags),
                "ext_flg": int(row.ext_flg),
                "nb": int(row.nb),
                **meas,
            }
        )
        print(
            f"      {patch['patch_id']} [{len(patch['controls']):3d}] {row.designation} "
            f"W3 SNR {row.w3snr:5.1f}: offset {meas['offset_arcsec']:.3f} arcsec"
        )


# ------------------------------------------------------------------- statistics
def split_half(controls: list[dict], patches: list[dict], seed: int) -> None:
    """Stratified-by-patch, seeded 50/50 split; labels alternate across the global order."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    toggle = 0
    for p in patches:
        mine = sorted(
            (c for c in controls if c["patch_id"] == p["patch_id"]),
            key=lambda c: c["gaia_dr3_source_id"],
        )
        for i in rng.permutation(len(mine)):
            mine[i]["split"] = "calibration" if toggle % 2 == 0 else "withheld"
            toggle += 1


def statistics(controls: list[dict], seed: int, n_boot: int) -> dict:
    def arr(rows, key):
        return np.array([c[key] for c in rows]) * u.arcsec

    cal = [c for c in controls if c["split"] == "calibration"]
    wh = [c for c in controls if c["split"] == "withheld"]
    r_c, s_c = arr(cal, "offset_arcsec"), arr(cal, "sigma_stat_arcsec")
    r_w, s_w = arr(wh, "offset_arcsec"), arr(wh, "sigma_stat_arcsec")
    est = {
        k: float(fn(r_c, s_c).to_value(u.arcsec))
        for k, fn in hephaistos.SIGMA_SYS_ESTIMATORS.items()
    }
    boot = hephaistos.sigma_sys_bootstrap(
        r_c, s_c, np.random.default_rng(np.random.SeedSequence([seed, 3])), n_boot, PRIMARY
    )
    fa = hephaistos.RAYLEIGH_3SIGMA_FALSE_ALARM

    def screen(r, s, sys_arcsec):
        return hephaistos.centroid_screen_false_alarms(r, s, sys_arcsec * u.arcsec, fa)

    def above_half_fwhm(rows):
        return sorted(c["offset_arcsec"] for c in rows if c["offset_arcsec"] > W3_FWHM_ARCSEC / 2)

    sys_c = est[PRIMARY]
    return {
        "primary_estimator": PRIMARY,
        "primary_estimator_rationale": PRIMARY_RATIONALE,
        "estimator_definitions": {
            "robust_median": "x >= 0 solving median_i[r_i / sqrt(sigma_stat_i^2 + x^2)] = "
            "sqrt(2 ln 2) (fermi_desk.models.hephaistos.sigma_sys_robust)",
            "rayleigh_mle": "heteroscedastic Rayleigh MLE, s_i^2 = sigma_stat_i^2 + x^2 "
            "(fermi_desk.models.hephaistos.sigma_sys_mle); not outlier-robust",
            "moment": "sqrt(max(0, <r^2>/2 - <sigma_stat^2>)); not outlier-robust",
        },
        "calibration": {
            "n_stars": len(cal),
            "sigma_sys_arcsec": sys_c,
            "sigma_sys_by_estimator_arcsec": est,
            "bootstrap": {
                "estimator": PRIMARY,
                "n_boot": n_boot,
                "rng": f"SeedSequence([{seed}, 3])",
                "p16_p84_arcsec": [float(np.percentile(boot, 16)), float(np.percentile(boot, 84))],
                "p2p5_p97p5_arcsec": [
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)),
                ],
            },
            "rayleigh_scale_arcsec": float(np.sqrt(np.mean(r_c.value**2) / 2.0)),
            "median_offset_arcsec": float(np.median(r_c.value)),
            "median_sigma_stat_arcsec": float(np.median(s_c.value)),
            "offsets_above_half_fwhm_arcsec": above_half_fwhm(cal),
            "in_sample_false_alarm_at_calibrated_floor": screen(r_c, s_c, sys_c),
        },
        "withheld": {
            "n_stars": len(wh),
            "sigma_sys_by_estimator_arcsec_for_reference_only": {
                k: float(fn(r_w, s_w).to_value(u.arcsec))
                for k, fn in hephaistos.SIGMA_SYS_ESTIMATORS.items()
            },
            "offsets_above_half_fwhm_arcsec": above_half_fwhm(wh),
            "out_of_sample_false_alarm_at_calibrated_floor": screen(r_w, s_w, sys_c),
            "false_alarm_at_legacy_baseline_floor": screen(r_w, s_w, LEGACY_SIGMA_SYS_ARCSEC),
            "false_alarm_with_no_floor": screen(r_w, s_w, 0.0),
        },
    }


# --------------------------------------------------------------------- promote
def promote(receipt_path: Path) -> int:
    raw = receipt_path.read_bytes()
    rec = json.loads(raw)
    problems = []
    if rec.get("kind") != "wise_control_experiment" or rec.get("data_tier") != "reproduced_archive":
        problems.append("not a reproduced_archive wise_control_experiment receipt")
    if rec.get("status") != "ok":
        problems.append(f"receipt status {rec.get('status')!r}")
    if rec.get("code_git_sha") in (None, "dirty", "unknown"):
        problems.append(f"code_git_sha {rec.get('code_git_sha')!r} is not a clean commit")
    if rec["n_controls"] < 100 or rec["n_patches_with_controls"] < 8:
        problems.append("needs >= 100 controls across >= 8 patches")
    if problems:
        print("Refusing to promote: " + "; ".join(problems), file=sys.stderr)
        return 1
    doc = {
        "id": "wise_sigma_sys_control",
        "schema_version": 1,
        "data_tier": "reproduced_archive",
        "description": (
            "Empirical AllWISE W3 centroid registration floor sigma_sys measured on isolated, "
            "RUWE-clean Gaia DR3 M-dwarf control stars across separated high-latitude patches; "
            "sigma_sys fitted on the calibration half, 3-sigma centroid false-alarm rate "
            "evaluated out-of-sample on the withheld half."
        ),
        "legacy_baseline_sigma_sys_arcsec": LEGACY_SIGMA_SYS_ARCSEC,
        "source_receipt": {
            "path": receipt_path.resolve().relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        **{k: v for k, v in rec.items() if k not in ("kind", "data_tier", "status")},
    }
    doc["content_sha256"] = canonical_sha256(doc)
    CALIBRATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    CALIBRATION_OUT.write_bytes(text.encode("utf-8"))
    print(f"Promoted -> {CALIBRATION_OUT.relative_to(ROOT).as_posix()}")
    print(f"  file sha256    {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    print(f"  content_sha256 {doc['content_sha256']}")
    return 0


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-controls", type=int, default=120, help="minimum measured controls")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-patches", type=int, default=12)
    ap.add_argument("--min-patches", type=int, default=8)
    ap.add_argument("--patch-radius-deg", type=float, default=2.0)
    ap.add_argument("--min-patch-separation-deg", type=float, default=20.0)
    ap.add_argument("--min-abs-b", type=float, default=20.0)
    ap.add_argument("--w3-snr-min", type=float, default=8.0)
    ap.add_argument("--w3-snr-max", type=float, default=20.0)
    ap.add_argument("--ruwe-max", type=float, default=1.4)
    ap.add_argument("--isolation-arcsec", type=float, default=15.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wise_control")
    ap.add_argument("--promote-from", type=Path, default=None)
    args = ap.parse_args()
    if args.promote_from is not None:
        return promote(args.promote_from)
    if args.seed is None:
        ap.error("--seed is required for a live run")

    git_sha, git_head = git_state(ROOT)
    started = datetime.now(UTC).isoformat()
    log: list = []
    exclude = {
        int(c["gaia_dr3_source_id"]) for c in yaml.safe_load(CLAIM.read_text())["candidates"]
    }
    patches, attempts = choose_patches(
        args.seed, args.n_patches, args.patch_radius_deg, args.min_abs_b,
        args.min_patch_separation_deg,
    )  # fmt: skip
    print(f"[1/5] {len(patches)} patches (seed={args.seed}, git_sha={git_sha}):")
    for p in patches:
        print(f"      {p['patch_id']}: l={p['l_deg']:7.2f} b={p['b_deg']:+7.2f} deg")

    print("[2/5] Patch queries (ARI Gaia TAP + IRSA AllWISE TAP) and cuts ...")
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    for p in patches:
        run_patch(p, args, exclude, log, rng)
        print(f"      {p['patch_id']}: {p['cut_counts']}")

    print("[3/5] Isolation checks, W3 cutouts and centroids ...")
    cap = math.ceil(args.n_controls / len(patches)) + 3
    for p in patches:
        fill_patch(p, cap, args, log)
    total = sum(len(p["controls"]) for p in patches)
    while total < args.n_controls and any(p["_cursor"] < len(p["_candidates"]) for p in patches):
        cap += 2  # top up from patches that still have unexamined candidates
        for p in patches:
            fill_patch(p, cap, args, log)
        total = sum(len(p["controls"]) for p in patches)

    controls = [c for p in patches for c in p["controls"]]
    n_with = sum(1 for p in patches if p["controls"])
    status = "ok" if total >= args.n_controls and n_with >= args.min_patches else "insufficient"
    for p in patches:
        p["cut_counts"]["isolation_checked"] = p["isolation"]["checked"]
        p["cut_counts"]["rejected_gaia_neighbour_within_15arcsec"] = p["isolation"][
            "rejected_neighbour_within_15arcsec"
        ]
        p["cut_counts"]["measurement_failures"] = len(p["measurement_failures"])
        p["cut_counts"]["measured_controls"] = len(p["controls"])
        p["cut_counts"]["unexamined_candidates"] = len(p["_candidates"]) - p["_cursor"]
        for k in ("_candidates", "_cursor", "isolation", "controls"):
            p.pop(k)
    totals = {k: sum(p["cut_counts"][k] for p in patches) for k in patches[0]["cut_counts"]}

    print(f"[4/5] {total} controls across {n_with} patches; deterministic 50/50 split ...")
    stats = None
    if controls:
        split_half(controls, patches, args.seed)
        stats = statistics(controls, args.seed, args.n_boot)
        c, w = stats["calibration"], stats["withheld"]
        fa = w["out_of_sample_false_alarm_at_calibrated_floor"]
        print(
            f"      calibration (n={c['n_stars']}): sigma_sys[{PRIMARY}] = "
            f"{c['sigma_sys_arcsec']:.3f} arcsec (68% boot "
            f"{c['bootstrap']['p16_p84_arcsec'][0]:.3f}-{c['bootstrap']['p16_p84_arcsec'][1]:.3f}"
            f"); by estimator {c['sigma_sys_by_estimator_arcsec']}"
        )
        print(
            f"      withheld (n={w['n_stars']}): 3-sigma centroid false alarms {fa['n_flagged']}/"
            f"{fa['n_stars']} = {fa['empirical_false_alarm']:.4f} (95% CP "
            f"{fa['clopper_pearson_95'][0]:.4f}-{fa['clopper_pearson_95'][1]:.4f}; nominal "
            f"{fa['nominal_false_alarm']}); legacy 0.5 arcsec floor: "
            f"{w['false_alarm_at_legacy_baseline_floor']['n_flagged']} flags; no floor: "
            f"{w['false_alarm_with_no_floor']['n_flagged']} flags"
        )

    finished = datetime.now(UTC).isoformat()
    receipt = {
        "kind": "wise_control_experiment",
        "data_tier": "reproduced_archive",
        "status": status,
        "code_git_sha": git_sha,
        "git_head": git_head,
        "rng_seed": args.seed,
        "query_window_utc": [started, finished],
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "selection": {
            "gaia": "parallax > 3.3333 mas (d < 300 pc), parallax_over_error > 10, BP-RP >= 1.84, "
            "10 <= G <= 16, AllWISE best neighbour (gaiadr3.allwise_best_neighbour)",
            "cuts_in_order": [
                f"star-level |b| > {args.min_abs_b} deg",
                "exclude published Hephaistos candidates (adversary claim file)",
                f"Gaia RUWE < {args.ruwe_max} (unresolved-binary rejection)",
                f"AllWISE {args.w3_snr_min} <= w3snr <= {args.w3_snr_max} (drill-down regime)",
                "AllWISE cc_flags == '0000'",
                "AllWISE ext_flg == 0",
                f"no other Gaia DR3 source within {args.isolation_arcsec} arcsec",
                "W3 cutout finite within 3 FWHM; centroid converged inside cutout",
            ],
            "isolation_adql_template": ISOLATION_ADQL,
            "avoided_fields": [
                {"name": n, "l": a, "b": b, "radius_deg": r} for n, a, b, r in AVOID
            ],
            "patch_draw": f"isotropic, SeedSequence([{args.seed}, 0]), {attempts} draws",
            "candidate_order": f"seeded permutation per patch, SeedSequence([{args.seed}, 1])",
            "split": f"stratified by patch, alternating labels, SeedSequence([{args.seed}, 2])",
            "allwise_mean_epoch_yr": ALLWISE_MEAN_EPOCH_YR,
            "w3_fwhm_arcsec": W3_FWHM_ARCSEC,
            "w3_pixel_arcsec": W3_PIXEL_ARCSEC,
        },
        "patches": patches,
        "n_patches_with_controls": n_with,
        "cut_counts_total": totals,
        "n_controls": total,
        "statistics": stats,
        "controls": controls,
        "query_receipts": log,
        "caveats": [
            "epoch propagation uses proper motion only (no parallax/RV terms)",
            "AllWISE mean epoch approximated as 2010.5 for all sources",
            "sigma_stat uses the Ren et al. 2026 eq. 1 form FWHM / (2.355 SNR); any departure "
            "of the real windowed-centroid noise from it is absorbed into sigma_sys",
            "controls are W3-detected M dwarfs: sigma_sys applies to the drill-down W3 SNR regime",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"wise_control-seed{args.seed}-{stamp}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"[5/5] Receipt ({status}): {out.as_posix()}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
