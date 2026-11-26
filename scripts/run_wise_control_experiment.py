#!/usr/bin/env python3
"""Opt-in LIVE experiment: empirical W3 centroid-offset floor on WISE control stars.

Queue job ``wise_independent_control_fields`` (Tier-1 archival). Never imported or
run by pytest. It:

1. draws a deterministic (seeded) random sample of Gaia DR3 M dwarfs within 300 pc
   with an AllWISE best-neighbour match (ARI Gaia TAP mirror), excluding the
   published Hephaistos candidates;
2. keeps those whose AllWISE W3 SNR lies in a band matched to the candidates
   (IRSA TAP cone search on ``allwise_p3as_psd``);
3. fetches a 60 arcsec AllWISE Atlas W3 cutout for each (IRSA SIA + IBE cutout);
4. measures the Gaussian-windowed W3 centroid and its offset from the Gaia
   position propagated to the AllWISE mean epoch;
5. writes a ``reproduced_archive`` receipt with query receipts and the empirical
   Rayleigh offset scale, from which the registration floor ``sigma_sys`` used by
   the blend-recovery drill-down (default 0.5 arcsec) can be checked.

TLS verification is never disabled. Usage:
    uv run python scripts/run_wise_control_experiment.py --n-controls 30 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.wcs import WCS

ARI_TAP = "https://gaia.ari.uni-heidelberg.de/tap/sync"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_SIA = "https://irsa.ipac.caltech.edu/SIA"
UA = "fermi-desk-wise-control/0.1 (MIT)"
W3_FWHM_ARCSEC = 6.5  # Ren et al. 2026 Table 2
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
ALLWISE_MEAN_EPOCH_YR = 2010.5  # AllWISE combines WISE cryo + NEOWISE-post-cryo (2010)
CLAIM = Path("data/adversary_claims/hephaistos_candidates_d_e.yaml")
W3_URL_RE = re.compile(
    r"https://irsa\.ipac\.caltech\.edu/ibe/data/wise/allwise/[^,\s\"]+-w3-int-3\.fits"
)


class LiveFetchError(RuntimeError):
    pass


def _get(url: str, data: bytes | None = None, timeout: int = 300) -> tuple[bytes, str]:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:600].decode("utf-8", errors="replace")
        raise LiveFetchError(
            f"{urllib.parse.urlparse(url).netloc}: HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LiveFetchError(f"{urllib.parse.urlparse(url).netloc}: {exc}") from exc


def tap_csv(endpoint: str, adql: str, receipts: list) -> pd.DataFrame:
    body = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}
    ).encode()
    raw, ctype = _get(endpoint, body)
    if "csv" not in ctype.lower():
        raise LiveFetchError(f"{endpoint}: expected CSV, got {ctype!r}: {raw[:200]!r}")
    df = pd.read_csv(io.BytesIO(raw))
    receipts.append(
        {
            "endpoint": endpoint,
            "queried_at_utc": datetime.now(UTC).isoformat(),
            "adql_sha256": hashlib.sha256(adql.encode()).hexdigest(),
            "n_rows": len(df),
        }
    )
    return df


def gaia_controls(seed: int, n_draw: int, receipts: list, n_patches: int = 24) -> pd.DataFrame:
    """Seeded random HEALPix level-5 sky patches -> primary-key source_id range scans.

    Gaia DR3 source_id // 2**35 is the HEALPix level-12 (nested) index, so a level-5
    pixel hp spans source_id in [hp * 2**49, (hp + 1) * 2**49).
    """
    rng = np.random.default_rng(seed)
    pix = rng.choice(12 * 4**5, size=n_patches, replace=False)  # seeded patch order
    span = 2**49
    frames = []
    for p in pix:
        # One patch per query, no TOP / ORDER BY: every matching row is returned, so the
        # sample is deterministic after the client-side sort below (and fast: PK range scan).
        adql = (
            "SELECT gs.source_id, gs.ra, gs.dec, gs.pmra, gs.pmdec, gs.ref_epoch, gs.parallax, "
            "gs.bp_rp, gs.phot_g_mean_mag, gs.ruwe, xm.original_ext_source_id "
            "FROM gaiadr3.gaia_source AS gs "
            "JOIN gaiadr3.allwise_best_neighbour AS xm ON gs.source_id = xm.source_id "
            f"WHERE gs.source_id BETWEEN {int(p) * span} AND {(int(p) + 1) * span - 1} "
            "AND gs.parallax > 3.3333 AND gs.parallax_over_error > 10 AND gs.bp_rp >= 1.84 "
            "AND gs.ruwe < 1.4 AND gs.phot_g_mean_mag BETWEEN 10 AND 16"
        )
        df = tap_csv(ARI_TAP, adql, receipts)
        frames.append(df.sort_values("source_id", kind="mergesort"))
        if sum(len(f) for f in frames) >= n_draw:
            break
    receipts.append({"healpix_level5_patches_queried": [int(p) for p in pix[: len(frames)]]})
    return pd.concat(frames, ignore_index=True).head(n_draw)


def allwise_w3(ra: float, dec: float, receipts: list) -> dict | None:
    adql = (
        "SELECT designation, ra, dec, w3snr, w3mpro, w3sigmpro, ext_flg, cc_flags "
        "FROM allwise_p3as_psd WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {3.0 / 3600:.7f})) = 1"
    )
    df = tap_csv(IRSA_TAP, adql, receipts)
    return None if df.empty else df.iloc[0].to_dict()


def w3_cutout(ra: float, dec: float, receipts: list) -> tuple[np.ndarray, WCS, str]:
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
    receipts.append(
        {"endpoint": cut, "queried_at_utc": datetime.now(UTC).isoformat(), "bytes": len(data)}
    )
    with fits.open(io.BytesIO(data)) as hdul:
        img = np.asarray(hdul[0].data, float)
        wcs = WCS(hdul[0].header)
    return img, wcs, urls[0]


def windowed_centroid(img: np.ndarray, x0: float, y0: float, s_pix: float) -> tuple[float, float]:
    """Gaussian-windowed (SExtractor-style) centroid, same estimator as models/hephaistos."""
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(float)
    data = img - np.nanmedian(img)
    data = np.nan_to_num(data)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-controls", type=int, default=30)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--w3-snr-min", type=float, default=8.0)
    ap.add_argument("--w3-snr-max", type=float, default=20.0)
    ap.add_argument("--max-draw", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("runs/wise_control"))
    args = ap.parse_args()

    receipts: list = []
    exclude = {
        int(c["gaia_dr3_source_id"]) for c in yaml.safe_load(CLAIM.read_text())["candidates"]
    }
    print(f"[1/4] Drawing Gaia DR3 M-dwarf controls (seed={args.seed}) from ARI TAP ...")
    cands = gaia_controls(args.seed, args.max_draw, receipts)
    cands = cands[~cands["source_id"].isin(exclude)].reset_index(drop=True)
    print(f"      {len(cands)} candidates after excluding Hephaistos D/E")

    rows = []
    s_pix = W3_FWHM_ARCSEC / FWHM_PER_SIGMA / 1.375
    print(f"[2/4] Selecting W3 SNR in [{args.w3_snr_min}, {args.w3_snr_max}] via IRSA TAP ...")
    for c in cands.itertuples(index=False):
        if len(rows) >= args.n_controls:
            break
        dt = ALLWISE_MEAN_EPOCH_YR - float(c.ref_epoch)
        dec = float(c.dec) + float(c.pmdec) * dt / 3.6e6
        ra = float(c.ra) + float(c.pmra) * dt / 3.6e6 / math.cos(math.radians(float(c.dec)))
        aw = allwise_w3(ra, dec, receipts)
        if aw is None or not args.w3_snr_min <= float(aw["w3snr"]) <= args.w3_snr_max:
            continue
        if str(aw.get("ext_flg", 0)) not in ("0", "0.0"):
            continue
        img, wcs, url = w3_cutout(ra, dec, receipts)
        gx, gy = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        cx, cy = windowed_centroid(img, gx, gy, s_pix)
        cra, cdec = (float(v) for v in wcs.pixel_to_world_values(cx, cy))
        off = math.hypot((cra - ra) * math.cos(math.radians(dec)), cdec - dec) * 3600.0
        sig_stat = W3_FWHM_ARCSEC / (FWHM_PER_SIGMA * float(aw["w3snr"]))
        rows.append(
            {
                "gaia_dr3_source_id": int(c.source_id),
                "allwise": aw["designation"],
                "w3snr": float(aw["w3snr"]),
                "offset_arcsec": off,
                "sigma_stat_arcsec": sig_stat,
                "image": url,
            }
        )
        print(
            f"      [{len(rows):3d}] {aw['designation']} W3 SNR {aw['w3snr']:.1f}: offset {off:.3f} arcsec"
        )

    if not rows:
        print("No controls selected; nothing to report.", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    r2 = df["offset_arcsec"].to_numpy() ** 2
    sigma_r = float(np.sqrt(r2.mean() / 2.0))  # Rayleigh MLE
    sigma_sys = float(np.sqrt(max(0.0, sigma_r**2 - np.mean(df["sigma_stat_arcsec"] ** 2))))
    print("[3/4] Empirical offsets:")
    print(f"      Rayleigh scale sigma_R = {sigma_r:.3f} arcsec over {len(df)} controls")
    print(
        f"      implied registration floor sigma_sys = {sigma_sys:.3f} arcsec (drill-down default 0.5)"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"wise_control-seed{args.seed}-{stamp}.json"
    receipt = {
        "kind": "wise_control_experiment",
        "data_tier": "reproduced_archive",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rng_seed": args.seed,
        "selection": {
            "gaia": "parallax > 3.333 mas (d < 300 pc), parallax_over_error > 10, bp_rp >= 1.84, "
            "ruwe < 1.4, 10 <= G <= 16, AllWISE best neighbour; Hephaistos D/E excluded",
            "w3_snr_band": [args.w3_snr_min, args.w3_snr_max],
            "allwise_mean_epoch_yr": ALLWISE_MEAN_EPOCH_YR,
            "w3_fwhm_arcsec": W3_FWHM_ARCSEC,
        },
        "n_controls": len(df),
        "rayleigh_sigma_arcsec": sigma_r,
        "implied_sigma_sys_arcsec": sigma_sys,
        "controls": rows,
        "query_receipts": receipts,
        "caveats": [
            "epoch propagation uses proper motion only (no parallax/RV terms)",
            "AllWISE mean epoch approximated as 2010.5 for all sources",
            "control sample is small and not SNR-weighted; treat sigma_sys as indicative",
        ],
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"[4/4] Receipt: {out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
