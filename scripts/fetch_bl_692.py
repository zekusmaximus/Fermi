#!/usr/bin/env python3
"""Fetch, cross-match, and checksum the Breakthrough Listen 692-star target catalog.

Provenance pipeline:
1. Enriquez et al. (2017) L-band 692-star target list (Berkeley SETI, or --bl-csv).
2. Isaacson et al. (2017) Table 1 (VizieR ``J/PASP/129/E4501/table1``): J2000
   coordinates, spectral types, and published legacy distances.
3. Gaia DR3 parallaxes via ``gaiadr3.hipparcos2_best_neighbour`` joined to
   ``gaiadr3.gaia_source`` over TAP (VOTable responses parsed with
   ``astropy.io.votable``; HTML / error responses are rejected explicitly).
4. Writes ``data/catalogs/bl_692_targets.csv`` and
   ``data/catalogs/bl_692_targets.provenance.json`` (SHA-256 + query receipts).

This script performs live network access and is never invoked by the test suite.

Usage:
    uv run python scripts/fetch_bl_692.py
    uv run python scripts/fetch_bl_692.py --bl-csv path/to/local_692.csv
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse as parse_votable

BL_PUB = "https://seti.berkeley.edu/lband2017/publications.html"
VIZ_SRC = "J/PASP/129/E4501/table1"
VIZ_ASU = [
    f"https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source={VIZ_SRC}&-out.all&-out.max=unlimited",
    f"https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source={VIZ_SRC}&-out.all&-out.max=unlimited",
]
VIZ_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
GAIA_TAPS = [
    "https://gea.esac.esa.int/tap-server/tap/sync",
    "https://gaia.ari.uni-heidelberg.de/tap/sync",
]
GROUP_ID = "bl_gbt_lband_2017"
UA = "fermi-desk-fetcher/0.3 (MIT)"
STAR_RE = re.compile(
    r"^(HIP|GJ|HD|LHS|GL|WD|TYC|HR|BD|CD|CPD|LP|ROSS|WOLF|PROXIMA|BARNARD)\s*[\d+-]",
    re.I,
)


def get_text(url: str, timeout: int = 45) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


class TapResponseError(RuntimeError):
    """A TAP endpoint answered, but not with a usable VOTable result."""


def _snippet(raw: bytes, n: int = 160) -> str:
    txt = raw[:2000].decode("utf-8", errors="replace")
    title = re.search(r"<title>(.*?)</title>", txt, flags=re.I | re.S)
    if title:
        return f"<title>{title.group(1).strip()}</title>"
    return re.sub(r"\s+", " ", txt)[:n]


def tap_query(url: str, adql: str, timeout: int = 90) -> pd.DataFrame:
    """Run a synchronous ADQL query and return the result table.

    Requests a VOTable (the TAP default serialization) and validates the reply:
    - Non-XML replies (e.g. an archive web page served with HTTP 200 instead of
      the TAP service, as observed on gea.esac.esa.int) raise ``TapResponseError``.
    - VOTables carrying ``INFO name="QUERY_STATUS" value="ERROR"`` raise with the
      server's ADQL error message instead of an opaque parser failure.
    TLS certificate verification is never disabled.
    """
    body = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "RESPONSEFORMAT": "votable",
            "QUERY": adql,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # TAP services often return a VOTable error document with HTTP 400/500.
        ctype, raw = exc.headers.get("Content-Type", ""), exc.read()
        if b"QUERY_STATUS" not in raw:
            raise TapResponseError(f"HTTP {exc.code}: {_snippet(raw)}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise TapResponseError(
                f"TLS certificate verification failed ({exc.reason.verify_message}); "
                "verification is intentionally not disabled. Point SSL_CERT_FILE at a "
                "CA bundle containing the server's chain if your OS store is incomplete."
            ) from exc
        raise

    head = raw.lstrip()[:512].lower()
    if b"<votable" not in head and not head.startswith(b"<?xml"):
        raise TapResponseError(
            f"Expected VOTable, got Content-Type={ctype!r}, {len(raw)} bytes: {_snippet(raw)}"
        )
    vot = parse_votable(io.BytesIO(raw), verify="warn")
    for res in vot.resources:
        for info in res.infos:
            if info.name == "QUERY_STATUS" and str(info.value).upper() == "ERROR":
                msg = (info.content or "").strip() or "unspecified error"
                raise TapResponseError(f"ADQL QUERY_STATUS=ERROR: {msg}")
    tables = list(vot.iter_tables())
    if not tables:
        raise TapResponseError("VOTable contained no TABLE element")
    return tables[0].to_table(use_names_over_ids=True).to_pandas()


def norm_id(val: Any) -> str:
    if pd.isna(val):
        return ""
    s = re.sub(r"\s+", " ", str(val).strip().upper().replace("_", " "))
    return re.sub(r"^(HIP|GJ|HD|LHS|GL|WD|TYC)\s*0*(\d+)", r"\1 \2", s)


def hip_num(name: str) -> int | None:
    m = re.match(r"^HIP\s+(\d+)$", name)
    return int(m.group(1)) if m else None


def load_bl_692(bl_csv: str | None) -> tuple[pd.DataFrame, str]:
    if bl_csv:
        src = bl_csv
        txt = (
            get_text(bl_csv)
            if bl_csv.startswith(("http://", "https://"))
            else Path(bl_csv).read_text(encoding="utf-8", errors="replace")
        )
    else:
        html = get_text(BL_PUB)
        links = re.findall(r'href=["\']([^"\']+\.csv)["\']', html, flags=re.I)
        if not links:
            raise RuntimeError(f"No CSV link found on {BL_PUB}")
        pick = next(
            (h for h in links if any(k in h.lower() for k in ("692", "star", "target"))), links[0]
        )
        src = urllib.parse.urljoin(BL_PUB, pick)
        txt = get_text(src)

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    first = lines[0].split(",")[0].strip().strip("\"'")
    if STAR_RE.match(first):
        df = pd.read_csv(io.StringIO(txt), header=None)
        col = 0
        print(f"      Headerless CSV ({df.shape[1]} cols, {len(df)} raw rows)")
    else:
        df = pd.read_csv(io.StringIO(txt))
        cmap = {str(c).lower().strip(): c for c in df.columns}
        col = next(
            (cmap[k] for k in ("target_name", "target", "star", "source", "name") if k in cmap),
            df.columns[0],
        )
        print(f"      Headered CSV; cols={list(df.columns)} (using '{col}')")

    out = pd.DataFrame({"target_name": df[col].map(norm_id)})
    out = out[out["target_name"] != ""].reset_index(drop=True)
    dupes = out[out.duplicated("target_name", keep=False)]
    if not dupes.empty:
        print(f"      Collapsed duplicates: {sorted(dupes['target_name'].unique())}")
    return out.drop_duplicates("target_name").reset_index(drop=True), src


def parse_asu_tsv(raw: str) -> pd.DataFrame:
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]
    header = [c.strip() for c in lines[0].split("\t")]
    start = 1
    for i in range(1, min(6, len(lines))):
        s = lines[i].replace("\t", "").strip()
        if s and set(s) <= {"-", " "}:
            start = i + 1
            break
    return pd.read_csv(io.StringIO("\n".join(lines[start:])), sep="\t", names=header, dtype=str)


def fetch_vizier() -> tuple[pd.DataFrame, str]:
    df, used, errs = None, "", []
    for url in VIZ_ASU:
        host = urllib.parse.urlparse(url).netloc
        try:
            print(f"      Trying VizieR mirror: {host}")
            df, used = parse_asu_tsv(get_text(url)), url
            break
        except Exception as exc:
            errs.append(f"{host}: {exc}")
    if df is None:
        try:
            print("      Falling back to CDS VizieR TAP")
            df, used = tap_query(VIZ_TAP, f'SELECT * FROM "{VIZ_SRC}"').astype(str), VIZ_TAP
        except Exception as exc:
            errs.append(f"TAP: {exc}")
    if df is None:
        raise RuntimeError("VizieR mirrors failed: " + "; ".join(errs))

    cm = {str(c).strip().lower(): c for c in df.columns}
    sc = cm.get("star") or cm.get("name")
    rc = cm.get("raj2000") or cm.get("_raj2000")
    dc = cm.get("dej2000") or cm.get("_dej2000")
    lc = cm.get("dist")
    if not all([sc, rc, dc, lc]):
        raise RuntimeError(f"Unexpected VizieR columns: {list(df.columns)}")

    ra_s = df[rc].astype(str).str.strip()
    de_s = df[dc].astype(str).str.strip()
    if ra_s.str.contains(r"[:\s]").any():
        coords = SkyCoord(ra_s.tolist(), de_s.tolist(), unit=(u.hourangle, u.deg))
    else:
        coords = SkyCoord(
            ra_s.astype(float).to_numpy() * u.deg, de_s.astype(float).to_numpy() * u.deg
        )

    sp, vm = cm.get("sptype"), cm.get("vmag")
    res = pd.DataFrame(
        {
            "target_name": df[sc].map(norm_id),
            "ra_deg": np.round(coords.ra.to(u.deg).value, 6),
            "dec_deg": np.round(coords.dec.to(u.deg).value, 6),
            "dist_legacy_pc": pd.to_numeric(df[lc], errors="coerce"),
            "spectral_type": df[sp].astype(str).str.strip() if sp else "",
            "vmag": pd.to_numeric(df[vm], errors="coerce") if vm else np.nan,
        }
    )
    return res.drop_duplicates("target_name").reset_index(drop=True), used


def fetch_gaia(
    hips: list[int], batch: int = 250
) -> tuple[pd.DataFrame, str, list[dict], list[str]]:
    ids = sorted(set(hips))
    errs = []
    for url in GAIA_TAPS:
        host = urllib.parse.urlparse(url).netloc
        try:
            print(f"      Trying Gaia TAP mirror: {host}")
            frames, receipts = [], []
            for i in range(0, len(ids), batch):
                in_list = ",".join(map(str, ids[i : i + batch]))
                q = (
                    "SELECT xm.original_ext_source_id AS hip_id, gs.source_id AS gaia_dr3_source_id, "
                    "gs.parallax AS gaia_parallax_mas, gs.parallax_error AS gaia_parallax_err_mas, "
                    "gs.parallax_over_error AS plx_snr, gs.ruwe AS gaia_ruwe, xm.angular_distance AS sep "
                    "FROM gaiadr3.hipparcos2_best_neighbour AS xm "
                    "JOIN gaiadr3.gaia_source AS gs ON xm.source_id = gs.source_id "
                    f"WHERE xm.original_ext_source_id IN ({in_list})"
                )
                part = tap_query(url, q)
                frames.append(part)
                receipts.append(
                    {
                        "endpoint": url,
                        "queried_at_utc": datetime.now(UTC).isoformat(),
                        "adql_sha256": hashlib.sha256(q.encode()).hexdigest(),
                        "n_hip_requested": len(ids[i : i + batch]),
                        "n_rows_returned": len(part),
                    }
                )
                print(f"        batch {i // batch + 1}: {len(part)} rows")
            g = pd.concat(frames, ignore_index=True)
            g["hip_id"] = pd.to_numeric(g["hip_id"], errors="coerce").astype("Int64")
            # 64-bit Gaia source IDs must stay integer (float64 silently corrupts them).
            g["gaia_dr3_source_id"] = g["gaia_dr3_source_id"].astype("Int64")
            g = g.sort_values(["hip_id", "sep"]).drop_duplicates("hip_id").reset_index(drop=True)
            ok = (g["gaia_parallax_mas"] > 0) & (g["plx_snr"] >= 5.0)
            dist = np.full(len(g), np.nan)
            if ok.any():
                dist[ok.to_numpy()] = (
                    (g.loc[ok, "gaia_parallax_mas"].to_numpy() * u.mas)
                    .to(u.pc, equivalencies=u.parallax())
                    .value
                )
            g["dist_gaia_pc"] = np.round(dist, 4)
            return g, url, receipts, errs
        except Exception as exc:
            print(f"      {host} failed: {exc}")
            errs.append(f"{host}: {exc}")
    raise RuntimeError("Gaia mirrors failed: " + "; ".join(errs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/catalogs/bl_692_targets.csv"))
    ap.add_argument("--bl-csv", type=str, default=None)
    ap.add_argument("--skip-gaia", action="store_true")
    args = ap.parse_args()

    print("[1/4] Fetching Enriquez et al. (2017) target list...")
    bl, bl_src = load_bl_692(args.bl_csv)
    print(f"      Loaded {len(bl)} unique targets from {bl_src}")

    print("[2/4] Fetching Isaacson et al. (2017) Table 1 from VizieR...")
    viz, viz_url = fetch_vizier()
    print(f"      Retrieved {len(viz)} rows")

    m = bl.merge(viz, on="target_name", how="left")
    m["hip_id"] = m["target_name"].map(hip_num).astype("Int64")
    unmatched = m.loc[m["ra_deg"].isna(), "target_name"].tolist()
    if unmatched:
        print(f"      Unmatched in VizieR: {len(unmatched)} (e.g., {unmatched[:8]})")

    gaia_url, gaia_receipts, gaia_errors = None, [], []
    if args.skip_gaia:
        for c in (
            "gaia_dr3_source_id",
            "gaia_parallax_mas",
            "gaia_parallax_err_mas",
            "gaia_ruwe",
            "dist_gaia_pc",
        ):
            m[c] = np.nan
    else:
        hips = m["hip_id"].dropna().astype(int).tolist()
        print(f"[3/4] Cross-matching {len(hips)} HIP targets against Gaia DR3...")
        g, gaia_url, gaia_receipts, gaia_errors = fetch_gaia(hips)
        cols = [
            "hip_id",
            "gaia_dr3_source_id",
            "gaia_parallax_mas",
            "gaia_parallax_err_mas",
            "gaia_ruwe",
            "dist_gaia_pc",
        ]
        m = m.merge(g[cols], on="hip_id", how="left")
        print(
            f"      Matched high-SNR Gaia DR3 distances: {int(m['dist_gaia_pc'].notna().sum())}/{len(m)}"
        )

    has_g = m["dist_gaia_pc"].notna()
    m["dist_pc"] = np.where(has_g, m["dist_gaia_pc"], m["dist_legacy_pc"])
    m["dist_source"] = np.where(
        has_g,
        "gaia_dr3_parallax",
        np.where(m["dist_legacy_pc"].notna(), "isaacson_2017_vizier", "missing"),
    )
    both = has_g & (m["dist_legacy_pc"] > 0)
    m["eirp_revision_factor"] = np.nan
    m.loc[both, "eirp_revision_factor"] = np.round(
        (m.loc[both, "dist_gaia_pc"] / m.loc[both, "dist_legacy_pc"]) ** 2, 4
    )
    m["pointing_group_id"] = GROUP_ID
    m["data_tier"] = "published_catalog"

    keep = [
        "target_name",
        "ra_deg",
        "dec_deg",
        "dist_pc",
        "dist_source",
        "dist_legacy_pc",
        "dist_gaia_pc",
        "eirp_revision_factor",
        "spectral_type",
        "vmag",
        "gaia_dr3_source_id",
        "gaia_parallax_mas",
        "gaia_parallax_err_mas",
        "gaia_ruwe",
        "pointing_group_id",
        "data_tier",
    ]
    m = m[keep].sort_values("target_name", kind="mergesort").reset_index(drop=True)

    print(f"[4/4] Writing {args.output} ...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed LF terminator so the pinned SHA-256 is platform-independent.
    m.to_csv(args.output, index=False, lineterminator="\n")
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()

    sev = m[
        m["eirp_revision_factor"].notna()
        & ((m["eirp_revision_factor"] >= 2.0) | (m["eirp_revision_factor"] <= 0.5))
    ]
    sev = sev.sort_values("eirp_revision_factor", ascending=False)
    top_rev = [
        {
            "target_name": r.target_name,
            "dist_legacy_pc": float(r.dist_legacy_pc),
            "dist_gaia_pc": float(r.dist_gaia_pc),
            "eirp_revision_factor": float(r.eirp_revision_factor),
        }
        for r in sev.head(15).itertuples(index=False)
    ]

    prov = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_tier": "published_catalog",
        "column_data_tiers": {
            "published_catalog": [
                "target_name",
                "ra_deg",
                "dec_deg",
                "dist_legacy_pc",
                "spectral_type",
                "vmag",
            ],
            "reproduced_archive": [
                "dist_gaia_pc",
                "gaia_dr3_source_id",
                "gaia_parallax_mas",
                "gaia_parallax_err_mas",
                "gaia_ruwe",
                "eirp_revision_factor",
            ],
            "derived": ["dist_pc", "dist_source"],
        },
        "gaia_distance_rule": "dist_gaia_pc = 1/parallax when parallax > 0 and parallax_over_error >= 5",
        "gaia_query_receipts": gaia_receipts,
        "gaia_mirror_failures": gaia_errors,
        "pointing_group_id": GROUP_ID,
        "output_csv": args.output.as_posix(),
        "output_sha256": sha,
        "row_count": len(m),
        "isaacson_matched_count": int(m["ra_deg"].notna().sum()),
        "unmatched_vizier_targets": unmatched,
        "gaia_dr3_matched_count": int(m["dist_gaia_pc"].notna().sum()),
        "missing_distance_count": int(m["dist_pc"].isna().sum()),
        "sources": {"bl": bl_src, "vizier": viz_url, "gaia": gaia_url},
        "top_eirp_distance_revisions": top_rev,
    }
    prov_path = args.output.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")

    print(f"      SHA-256: {sha}")
    if top_rev:
        t = top_rev[0]
        print(
            f"      Largest EIRP shift: {t['target_name']} ({t['dist_legacy_pc']} -> {t['dist_gaia_pc']} pc, {t['eirp_revision_factor']}x)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
