#!/usr/bin/env python3
"""Fetch, cross-match, and checksum the Breakthrough Listen 692-star target catalog.

Provenance pipeline:
1. Downloads the Enriquez et al. (2017) L-band 692-star target CSV from the
   Berkeley SETI publications page (or a user-supplied --bl-csv local file).
2. Queries CDS VizieR TAP for Isaacson et al. (2017) Table 1
   ("J/PASP/129/E4501/table1") to attach J2000 coordinates, spectral types,
   and published legacy distances (pc).
3. Queries ESA Gaia DR3 TAP (gaiadr3.hipparcos2_best_neighbour joined to
   gaiadr3.gaia_source) in deterministic batches to attach Gaia DR3 parallaxes
   and flag major distance / minimum-EIRP revisions (e.g., HIP 1692).
4. Writes:
   - data/catalogs/bl_692_targets.csv
   - data/catalogs/bl_692_targets.provenance.json (with SHA-256 & query receipts)

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
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

BL_PUBLICATIONS_URL = "https://seti.berkeley.edu/lband2017/publications.html"
VIZIER_TAP_SYNC_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
VIZIER_ISAACSON_TABLE = '"J/PASP/129/E4501/table1"'
POINTING_GROUP_ID = "bl_gbt_lband_2017"
USER_AGENT = "fermi-desk-catalog-fetcher/0.1 (MIT; https://github.com)"


def http_get_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def tap_sync_csv(endpoint: str, adql: str, timeout: int = 90) -> pd.DataFrame:
    payload = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": adql,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(raw))


def normalize_star_id(raw: Any) -> str:
    """Canonicalize star names (e.g., 'HIP  1692', 'HIP1692', 'hip_1692' -> 'HIP 1692')."""
    if pd.isna(raw):
        return ""
    text = str(raw).strip().upper().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    # Normalize common catalog prefixes missing a space: HIP1692 -> HIP 1692
    text = re.sub(r"^(HIP|GJ|HD|LHS|GL|WD|TYC)\s*0*(\d+)", r"\1 \2", text)
    return text


def extract_hip_number(canonical_name: str) -> int | None:
    match = re.match(r"^HIP\s+(\d+)$", canonical_name)
    return int(match.group(1)) if match else None


def discover_bl_csv_url(publications_url: str) -> str:
    """Parse the Berkeley SETI L-band 2017 publications page for its target CSV link."""
    html = http_get_text(publications_url)
    matches = re.findall(r'href=["\']([^"\']+\.csv)["\']', html, flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError(
            f"Could not find a .csv link on {publications_url}. "
            "Pass --bl-csv with a local copy or direct URL."
        )
    # Prefer a link referencing target/star/692 if multiple CSVs exist
    for href in matches:
        if any(k in href.lower() for k in ("target", "star", "692", "enriquez", "list")):
            return urllib.parse.urljoin(publications_url, href)
    return urllib.parse.urljoin(publications_url, matches[0])


def load_enriquez_692(bl_csv: str | None) -> tuple[pd.DataFrame, str]:
    """Load the observed 692-star target list from Berkeley SETI or a local CSV."""
    if bl_csv:
        source_ref = bl_csv
        if bl_csv.startswith(("http://", "https://")):
            raw = http_get_text(bl_csv)
            df = pd.read_csv(io.StringIO(raw))
        else:
            df = pd.read_csv(Path(bl_csv))
    else:
        source_ref = discover_bl_csv_url(BL_PUBLICATIONS_URL)
        raw = http_get_text(source_ref)
        df = pd.read_csv(io.StringIO(raw))

    # Defensive column discovery (Berkeley CSVs sometimes omit headers or name it Star/Target/Source)
    cols_lower = {c.lower().strip(): c for c in df.columns}
    name_col = None
    for candidate in ("target_name", "target", "star", "source", "source_name", "name"):
        if candidate in cols_lower:
            name_col = cols_lower[candidate]
            break

    if name_col is None:
        # Handle single-column headerless CSVs cleanly
        if df.shape[1] == 1:
            first_val = str(df.columns[0])
            series = pd.concat([pd.Series([first_val]), df.iloc[:, 0]], ignore_index=True)
            df = pd.DataFrame({"raw_target_name": series})
            name_col = "raw_target_name"
        else:
            name_col = df.columns[0]

    out = pd.DataFrame()
    out["raw_target_name"] = df[name_col].astype(str).str.strip()
    out["target_name"] = out["raw_target_name"].map(normalize_star_id)
    out = out[out["target_name"] != ""].drop_duplicates(subset=["target_name"]).reset_index(drop=True)
    return out, source_ref


def fetch_isaacson_vizier() -> tuple[pd.DataFrame, str]:
    """Fetch Isaacson et al. (2017) Table 1 from CDS VizieR TAP."""
    adql = f"SELECT * FROM {VIZIER_ISAACSON_TABLE}"
    df = tap_sync_csv(VIZIER_TAP_SYNC_URL, adql)

    # Identify key columns robustly across VizieR formatting
    col_map = {c.lower(): c for c in df.columns}
    star_col = col_map.get("star") or col_map.get("name")
    ra_col = col_map.get("raj2000") or col_map.get("_raj2000")
    de_col = col_map.get("dej2000") or col_map.get("_dej2000")
    dist_col = col_map.get("dist")
    sptype_col = col_map.get("sptype")
    vmag_col = col_map.get("vmag")
    simbad_col = col_map.get("simbadname")

    if not all([star_col, ra_col, de_col, dist_col]):
        raise RuntimeError(
            f"Unexpected VizieR columns in {VIZIER_ISAACSON_TABLE}: {list(df.columns)}"
        )

    # Convert coordinates via Astropy (supports either decimal deg or h:m:s / d:m:s)
    ra_vals = df[ra_col].astype(str).str.strip()
    de_vals = df[de_col].astype(str).str.strip()
    if ra_vals.str.contains(r"[:\s]").any():
        coords = SkyCoord(ra_vals.tolist(), de_vals.tolist(), unit=(u.hourangle, u.deg), frame="icrs")
    else:
        coords = SkyCoord(
            ra_vals.astype(float).to_numpy() * u.deg,
            de_vals.astype(float).to_numpy() * u.deg,
            frame="icrs",
        )

    parsed = pd.DataFrame(
        {
            "target_name": df[star_col].map(normalize_star_id),
            "simbad_name": df[simbad_col].astype(str).str.strip() if simbad_col else "",
            "ra_deg": np.round(coords.ra.to(u.deg).value, 6),
            "dec_deg": np.round(coords.dec.to(u.deg).value, 6),
            "dist_legacy_pc": pd.to_numeric(df[dist_col], errors="coerce"),
            "spectral_type": df[sptype_col].astype(str).str.strip() if sptype_col else "",
            "vmag": pd.to_numeric(df[vmag_col], errors="coerce") if vmag_col else np.nan,
        }
    )
    parsed = parsed.drop_duplicates(subset=["target_name"]).reset_index(drop=True)
    return parsed, adql


def fetch_gaia_dr3_for_hips(hip_ids: list[int], batch_size: int = 250) -> tuple[pd.DataFrame, list[str]]:
    """Query ESA Gaia DR3 TAP for Hipparcos-matched parallaxes and quality metrics."""
    frames: list[pd.DataFrame] = []
    queries: list[str] = []
    sorted_hips = sorted(set(hip_ids))

    for i in range(0, len(sorted_hips), batch_size):
        batch = sorted_hips[i : i + batch_size]
        in_clause = ", ".join(str(h) for h in batch)
        adql = (
            "SELECT xm.original_ext_source_id AS hip_id, "
            "gs.source_id AS gaia_dr3_source_id, "
            "gs.ra AS gaia_ra_deg, gs.dec AS gaia_dec_deg, "
            "gs.parallax AS gaia_parallax_mas, "
            "gs.parallax_error AS gaia_parallax_err_mas, "
            "gs.parallax_over_error AS gaia_parallax_snr, "
            "gs.ruwe AS gaia_ruwe, "
            "xm.angular_distance AS xmatch_sep_arcsec "
            "FROM gaiadr3.hipparcos2_best_neighbour AS xm "
            "JOIN gaiadr3.gaia_source AS gs ON xm.source_id = gs.source_id "
            f"WHERE xm.original_ext_source_id IN ({in_clause})"
        )
        queries.append(adql)
        frames.append(tap_sync_csv(GAIA_TAP_SYNC_URL, adql))

    if not frames:
        return pd.DataFrame(columns=["hip_id"]), queries

    gaia_df = pd.concat(frames, ignore_index=True)
    gaia_df["hip_id"] = pd.to_numeric(gaia_df["hip_id"], errors="coerce").astype("Int64")
    # Keep closest match if multiple best_neighbour rows exist
    gaia_df = (
        gaia_df.sort_values(["hip_id", "xmatch_sep_arcsec"], ascending=[True, True])
        .drop_duplicates(subset=["hip_id"], keep="first")
        .reset_index(drop=True)
    )

    # Convert parallax (mas) -> distance (pc) using Astropy equivalencies where SNR > 5 and parallax > 0
    valid = (gaia_df["gaia_parallax_mas"] > 0) & (gaia_df["gaia_parallax_snr"] >= 5.0)
    dist_pc = np.full(len(gaia_df), np.nan, dtype=float)
    if valid.any():
        plx_qty = gaia_df.loc[valid, "gaia_parallax_mas"].to_numpy() * u.mas
        dist_pc[valid.to_numpy()] = plx_qty.to(u.pc, equivalencies=u.parallax()).value
    gaia_df["dist_gaia_pc"] = np.round(dist_pc, 4)
    return gaia_df, queries


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_catalog(
    output_csv: Path,
    bl_csv: str | None = None,
    skip_gaia: bool = False,
) -> dict[str, Any]:
    print("[1/4] Fetching Enriquez et al. (2017) 692-star target list...")
    bl_df, bl_source = load_enriquez_692(bl_csv)
    print(f"      Loaded {len(bl_df)} unique targets from {bl_source}")

    print("[2/4] Querying CDS VizieR for Isaacson et al. (2017) Table 1...")
    isaacson_df, vizier_adql = fetch_isaacson_vizier()
    print(f"      Retrieved {len(isaacson_df)} rows from {VIZIER_ISAACSON_TABLE}")

    merged = bl_df.merge(isaacson_df, on="target_name", how="left")
    merged["hip_id"] = merged["target_name"].map(extract_hip_number).astype("Int64")

    gaia_queries: list[str] = []
    if skip_gaia:
        print("[3/4] Skipping Gaia DR3 cross-match (--skip-gaia set).")
        merged["gaia_dr3_source_id"] = pd.NA
        merged["gaia_parallax_mas"] = np.nan
        merged["gaia_parallax_err_mas"] = np.nan
        merged["gaia_ruwe"] = np.nan
        merged["dist_gaia_pc"] = np.nan
    else:
        hip_list = merged["hip_id"].dropna().astype(int).tolist()
        print(f"[3/4] Cross-matching {len(hip_list)} HIP targets against ESA Gaia DR3 TAP...")
        gaia_df, gaia_queries = fetch_gaia_dr3_for_hips(hip_list)
        merged = merged.merge(
            gaia_df[
                [
                    "hip_id",
                    "gaia_dr3_source_id",
                    "gaia_parallax_mas",
                    "gaia_parallax_err_mas",
                    "gaia_ruwe",
                    "dist_gaia_pc",
                ]
            ],
            on="hip_id",
            how="left",
        )
        matched_gaia = int(merged["dist_gaia_pc"].notna().sum())
        print(f"      Matched high-SNR Gaia DR3 distances for {matched_gaia}/{len(merged)} targets")

    # Select canonical distance (prefer high-SNR Gaia DR3, fall back to Isaacson 2017)
    has_gaia = merged["dist_gaia_pc"].notna()
    merged["dist_pc"] = np.where(has_gaia, merged["dist_gaia_pc"], merged["dist_legacy_pc"])
    merged["dist_source"] = np.where(
        has_gaia,
        "gaia_dr3_parallax",
        np.where(merged["dist_legacy_pc"].notna(), "isaacson_2017_vizier", "missing"),
    )

    # Compute minimum-EIRP revision multiplier = (d_gaia / d_legacy)^2
    both_valid = merged["dist_gaia_pc"].notna() & (merged["dist_legacy_pc"] > 0)
    merged["eirp_revision_factor"] = np.nan
    merged.loc[both_valid, "eirp_revision_factor"] = np.round(
        (merged.loc[both_valid, "dist_gaia_pc"] / merged.loc[both_valid, "dist_legacy_pc"]) ** 2,
        4,
    )

    merged["pointing_group_id"] = POINTING_GROUP_ID
    merged["data_tier"] = "published_catalog"

    # Deterministic column order and row sort
    ordered_cols = [
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
    merged = merged[ordered_cols].sort_values("target_name", kind="mergesort").reset_index(drop=True)

    print(f"[4/4] Writing verified catalog to {output_csv}...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    digest = sha256_file(output_csv)

    # Audit summary: flag severe distance/EIRP revisions (>2x EIRP shift)
    severe = merged[
        merged["eirp_revision_factor"].notna()
        & ((merged["eirp_revision_factor"] >= 2.0) | (merged["eirp_revision_factor"] <= 0.5))
    ].sort_values("eirp_revision_factor", ascending=False)

    flagged_revisions = [
        {
            "target_name": row.target_name,
            "dist_legacy_pc": float(row.dist_legacy_pc),
            "dist_gaia_pc": float(row.dist_gaia_pc),
            "eirp_revision_factor": float(row.eirp_revision_factor),
        }
        for row in severe.head(15).itertuples(index=False)
    ]

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_tier": "published_catalog",
        "pointing_group_id": POINTING_GROUP_ID,
        "output_csv": str(output_csv),
        "output_sha256": digest,
        "row_count": int(len(merged)),
        "expected_enriquez_2017_count": 692,
        "isaacson_matched_count": int(merged["ra_deg"].notna().sum()),
        "gaia_dr3_matched_count": int(merged["dist_gaia_pc"].notna().sum()),
        "missing_distance_count": int(merged["dist_pc"].isna().sum()),
        "sources": {
            "enriquez_2017_target_list": bl_source,
            "vizier_endpoint": VIZIER_TAP_SYNC_URL,
            "vizier_adql": vizier_adql,
            "gaia_endpoint": GAIA_TAP_SYNC_URL if not skip_gaia else None,
            "gaia_query_batch_count": len(gaia_queries),
        },
        "top_eirp_distance_revisions": flagged_revisions,
    }

    prov_path = output_csv.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"      SHA-256: {digest}")
    print(f"      Provenance receipt: {prov_path}")
    if flagged_revisions:
        top = flagged_revisions[0]
        print(
            f"      Largest EIRP shift: {top['target_name']} "
            f"({top['dist_legacy_pc']} pc -> {top['dist_gaia_pc']} pc, "
            f"{top['eirp_revision_factor']}x EIRP_min)"
        )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the Breakthrough Listen 692-star catalog with Gaia DR3 distances."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/catalogs/bl_692_targets.csv"),
        help="Destination CSV path (default: data/catalogs/bl_692_targets.csv)",
    )
    parser.add_argument(
        "--bl-csv",
        type=str,
        default=None,
        help="Optional local path or direct URL to the Enriquez et al. (2017) 692-star CSV.",
    )
    parser.add_argument(
        "--skip-gaia",
        action="store_true",
        help="Skip ESA Gaia DR3 cross-match (keeps Isaacson 2017 VizieR distances only).",
    )
    args = parser.parse_args()

    try:
        build_catalog(output_csv=args.output, bl_csv=args.bl_csv, skip_gaia=args.skip_gaia)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())