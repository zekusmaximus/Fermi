#!/usr/bin/env python3
"""Opt-in LIVE experiment: Public-Launch Gate #2 on real AllWISE W3 Atlas FITS cutouts.

Frozen-method adversarial blend-recovery test with a matched centered-excess negative
control (SPEC.md "Public-Launch Scientific Exit Gate" item 2). Never imported or run by
pytest. Sequence (each step's timestamp is recorded in the receipt):

1. Read the 154 locked controls and their calibration / withheld labels from
   ``data/calibrations/wise_sigma_sys_control.json`` (verified against its pinned
   SHA-256; never modified).
2. Fetch the **calibration** cutouts: AllWISE Atlas W3 intensity (``-int-3.fits``) and
   uncertainty (``-unc-3.fits.gz``) from NASA/IPAC IRSA IBE, 60 arcsec, centred on the
   same Gaia-at-AllWISE-epoch position as Gate #1, using the coadd tile that Gate #1's
   IRSA SIA query selected (recorded in the locked file). Files are cached in the
   gitignored ``data/cache/wise_fits/`` with deterministic names and a SHA-256
   manifest; a cached file is reused only if its bytes still match the manifest.
3. Build the empirical PSF and **freeze** both screen thresholds from calibration
   cutouts only (``frozen_at_utc``, ``frozen_screens_sha256``).
4. Only then fetch the **withheld** cutouts and run the two-arm injection (``--seed``).
5. Write ``runs/wise_fits_gate2/wise_fits_gate2-seed<seed>-<UTC>.json``.

``--promote-from <receipt>`` refuses dirty ``git_sha`` and incomplete splits, then writes
``data/calibrations/wise_fits_gate2_withheld.json`` (``reproduced_archive``).

TLS verification is never disabled. Usage:
    uv run python scripts/run_wise_fits_gate2.py --seed 42
    uv run python scripts/run_wise_fits_gate2.py --promote-from runs/wise_fits_gate2/<r>.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from fermi_desk.inference.engine import canonical_sha256, git_state
from fermi_desk.models import hephaistos_fits as hf
from fermi_desk.registry import load_factor_graph, load_sigma_sys_calibration

ROOT = Path(__file__).resolve().parents[1]
UA = "fermi-desk-wise-gate2/0.1 (MIT)"
IR_SURVEY = "hephaistos_wise_mid_ir"
OUT_FILE = ROOT / "data/calibrations/wise_fits_gate2_withheld.json"
INT_SUFFIX = "-int-3.fits"
UNC_SUFFIX = "-unc-3.fits.gz"  # AllWISE Atlas uncertainty maps are gzipped on IBE
IBE_PREFIX = "https://irsa.ipac.caltech.edu/ibe/data/wise/allwise/"
CAVEATS = [
    "Arm B pins the excess to the star's measured (noisy) windowed centroid, so at high "
    "f_bg the centered excess inherits the uninjected centroid noise while the screen "
    "threshold assumes SNR_eff; centroid-screen Arm B false rejection is conservative "
    "(biased high) there",
    "injection adds the empirical PSF without extra noise (W3 is background-limited; source "
    "shot noise neglected); real blends are extended or coloured, the injected ones are "
    "PSF-shaped point sources",
    "the empirical PSF is aligned on noise-locked windowed centroids (shift-and-add), which "
    "sharpens a stacked PSF by ~1-3% at W3 SNR 8-20 (synthetic check); its FWHM is a "
    "slight underestimate",
    "the image screen uses one scalar noise variance per cutout (median unc^2 in the fit "
    "region); Atlas pixel-noise correlation is absorbed by the median-matched Delta chi^2 "
    "scale fitted on calibration nulls, and the chi^2(3) tail shape is an assumption whose "
    "out-of-sample check is the withheld null false-alarm rate",
    "calibration 'nulls' are isolated Gaia-clean M dwarfs; unseen W3 background sources "
    "among them are part of the realistic null, not removed",
    "SNR_eff = w3snr / (1 - f_bg) assumes background-limited photometry of the blend",
]


class LiveFetchError(RuntimeError):
    pass


# ------------------------------------------------------------------------ HTTP
def _get(url: str, timeout: int = 120, tries: int = 4) -> tuple[bytes, str]:
    if not url.startswith(IBE_PREFIX):
        raise LiveFetchError(f"refusing non-IRSA-IBE URL {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == tries:
                raise LiveFetchError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == tries:
                raise LiveFetchError(f"{exc} for {url}") from exc
        time.sleep(5.0 * attempt)
    raise AssertionError("unreachable")


def cutout_url(image_url: str, suffix: str, ra: float, dec: float, size_arcsec: float) -> str:
    if not image_url.endswith(INT_SUFFIX):
        raise LiveFetchError(f"unexpected locked image URL {image_url!r}")
    base = image_url[: -len(INT_SUFFIX)] + suffix
    q = urllib.parse.urlencode(
        {"center": f"{ra},{dec}", "size": f"{size_arcsec:g}arcsec", "gzip": "false"},
        safe=",",
    )
    return f"{base}?{q}"


class Cache:
    """Deterministic file cache with a SHA-256 manifest (``manifest.json``)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / "manifest.json"
        self.manifest = json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", "utf-8")

    def fetch(self, name: str, url: str) -> tuple[bytes, dict]:
        f = self.root / name
        entry = self.manifest.get(name)
        if f.exists() and entry and entry.get("url") == url:
            data = f.read_bytes()
            if hashlib.sha256(data).hexdigest() == entry["sha256"] and data.startswith(b"SIMPLE"):
                return data, {**entry, "cached": True}
        data, ctype = _get(url)
        if not data.startswith(b"SIMPLE"):
            raise LiveFetchError(f"not FITS ({ctype}): {data[:100]!r} from {url}")
        entry = {
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "fetched_at_utc": datetime.now(UTC).isoformat(),
        }
        f.write_bytes(data)
        self.manifest[name] = entry
        self._save()
        return data, {**entry, "cached": False}


def load_cutout(control: dict, cache: Cache, size: float) -> tuple[hf.W3Cutout, dict]:
    sid = int(control["gaia_dr3_source_id"])
    ra, dec = control["gaia_ra_epoch_allwise_deg"], control["gaia_dec_epoch_allwise_deg"]
    tile = control["image"].rsplit("/", 1)[-1][: -len(INT_SUFFIX)]
    rec: dict = {"gaia_dr3_source_id": sid, "split": control["split"], "coadd": tile}
    arrays, wcs = {}, None
    for kind, suffix in (("int", INT_SUFFIX), ("unc", UNC_SUFFIX)):
        url = cutout_url(control["image"], suffix, ra, dec, size)
        name = f"{sid}_{tile}-{kind}-3_c{size:g}as.fits"
        data, entry = cache.fetch(name, url)
        rec[f"{kind}_file"] = name
        rec[f"{kind}_url"] = url
        rec[f"{kind}_sha256"] = entry["sha256"]
        rec[f"{kind}_cached"] = entry["cached"]
        with fits.open(io.BytesIO(data)) as hdul:
            h = hdul[0]
            arrays[kind] = np.asarray(h.data, float)
            if kind == "int":
                wcs = WCS(h.header)
                rec["bunit"] = h.header.get("BUNIT")
    if arrays["int"].shape != arrays["unc"].shape:
        raise LiveFetchError("int / unc cutout shapes differ")
    gx, gy = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
    pix = float(np.mean(proj_plane_pixel_scales(wcs.celestial)) * 3600.0)
    rec.update({"shape": list(arrays["int"].shape), "gaia_xy": [gx, gy], "pixel_arcsec": pix})
    cut = hf.W3Cutout(
        sid,
        control["split"],
        arrays["int"],
        arrays["unc"],
        (gx, gy),
        pix * u.arcsec,
        float(control["w3snr"]),
        control["patch_id"],
    )
    return cut, rec


def fetch_split(controls: list[dict], split: str, cache: Cache, size: float, log: list) -> list:
    out = []
    for i, c in enumerate(x for x in controls if x["split"] == split):
        try:
            cut, rec = load_cutout(c, cache, size)
            why = hf.cutout_problem(cut)
            if why:
                raise LiveFetchError(why)
            s_pix = hf.nominal_window_sigma_pix(cut.pixel_scale)
            cx, cy = hf.windowed_centroid(cut.flux, *cut.gaia_xy, s_pix)
            rec["centroid_offset_arcsec_recomputed"] = float(
                np.hypot(cx - cut.gaia_xy[0], cy - cut.gaia_xy[1]) * cut.pix
            )
            rec["centroid_offset_arcsec_locked"] = float(c["offset_arcsec"])
            rec["status"] = "ok"
            out.append(cut)
        except (LiveFetchError, OSError, ValueError) as exc:
            rec = {"gaia_dr3_source_id": int(c["gaia_dr3_source_id"]), "split": split}
            rec.update({"status": "failed", "reason": str(exc)[:240]})
        log.append(rec)
        cached = rec.get("int_cached") and rec.get("unc_cached")
        print(
            f"      {split[:3]} [{i + 1:2d}] {rec['gaia_dr3_source_id']}: {rec['status']}"
            f"{' (cache)' if cached else ''}"
            + (
                f' offset {rec["centroid_offset_arcsec_recomputed"]:.3f}"'
                if rec["status"] == "ok"
                else f" {rec.get('reason')}"
            )
        )
    return out


# --------------------------------------------------------------------- promote
def promote(receipt_path: Path) -> int:
    raw = receipt_path.read_bytes()
    rec = json.loads(raw)
    problems = []
    if rec.get("kind") != "wise_fits_gate2" or rec.get("data_tier") != "reproduced_archive":
        problems.append("not a reproduced_archive wise_fits_gate2 receipt")
    if rec.get("status") != "ok":
        problems.append(f"receipt status {rec.get('status')!r}")
    if rec.get("code_git_sha") in (None, "dirty", "unknown"):
        problems.append(f"code_git_sha {rec.get('code_git_sha')!r} is not a clean commit")
    sp = rec.get("split_counts", {})
    for split in ("calibration", "withheld"):
        got, want = sp.get(f"n_{split}_evaluated"), sp.get(f"n_{split}_locked")
        if not want or got != want:
            problems.append(f"incomplete {split} split: {got} of {want} locked stars evaluated")
    if rec.get("fetch", {}).get("n_failed", 1) != 0:
        problems.append("cutout fetch had failures")
    if problems:
        print("Refusing to promote: " + "; ".join(problems), file=sys.stderr)
        return 1
    doc = {
        "id": "wise_fits_gate2_withheld",
        "schema_version": 1,
        "data_tier": "reproduced_archive",
        "description": (
            "Public-Launch Gate #2: frozen-method blend screens on real AllWISE W3 Atlas "
            "cutouts of the 154 locked control stars. Empirical PSF and both thresholds "
            "frozen on the 77 calibration stars; withheld two-arm injection (off-center "
            "blends vs matched centered excess)."
        ),
        "source_receipt": {
            "path": receipt_path.resolve().relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        **{k: v for k, v in rec.items() if k not in ("kind", "data_tier", "status")},
    }
    doc["content_sha256"] = canonical_sha256(doc)
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_bytes(text.encode("utf-8"))
    print(f"Promoted -> {OUT_FILE.relative_to(ROOT).as_posix()}")
    print(f"  file sha256    {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    print(f"  content_sha256 {doc['content_sha256']}")
    return 0


# ------------------------------------------------------------------------ main
def _fmt_rate(r: dict) -> str:
    lo, hi = r["clopper_pearson_95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.3f} (95% CP {lo:.3f}-{hi:.3f})"


def report(summary: dict, psf: hf.EmpiricalPSF, frozen: hf.FrozenScreens) -> None:
    print(
        f'      empirical PSF: FWHM {psf.fwhm_gaussian.to_value(u.arcsec):.3f}" (Gaussian fit), '
        f'{psf.fwhm_halfmax.to_value(u.arcsec):.3f}" (half-max area) vs nominal '
        f'{hf.NOMINAL_W3_FWHM_ARCSEC}"; {len(psf.star_ids)} calibration stamps'
    )
    for s, v in summary["screens"].items():
        rec = np.array(v["arm_a_blend_recovery"])
        print(f"      [{s}] withheld null false alarm {_fmt_rate(v['null_false_alarm'])}")
        print(
            f'        Arm A blend survival within 3.25": {v["blend_survival_within_3p25_arcsec"]:.3f}'
        )
        print("        Arm A recovery [rows theta, cols f_bg]:")
        for row in rec:
            print("          " + " ".join(f"{x:5.3f}" for x in row))
        print(
            "        Arm B centered-excess false rejection by f_bg: "
            + " ".join(f"{x:5.3f}" for x in v["arm_b_false_rejection"])
            + f" | pooled {_fmt_rate(v['arm_b_false_rejection_pooled'])}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--size-arcsec", type=float, default=60.0)
    ap.add_argument("--target-false-alarm", type=float, default=hf.RAYLEIGH_3SIGMA_FALSE_ALARM)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/wise_fits")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/wise_fits_gate2")
    ap.add_argument("--promote-from", type=Path, default=None)
    args = ap.parse_args()
    if args.promote_from is not None:
        return promote(args.promote_from)
    if args.seed is None:
        ap.error("--seed is required for a live run")

    git_sha, git_head = git_state(ROOT)
    started = datetime.now(UTC).isoformat()
    graph = load_factor_graph(ROOT)
    locked = load_sigma_sys_calibration(graph.surveys[IR_SURVEY], ROOT)
    if locked is None:
        raise SystemExit("IR survey pins no sigma_sys calibration")
    controls = sorted(
        (c.model_dump() for c in locked.record.controls), key=lambda c: c["gaia_dr3_source_id"]
    )
    n_locked = {s: sum(c["split"] == s for c in controls) for s in ("calibration", "withheld")}
    sigma_sys = locked.sigma_sys
    print(
        f"[1/5] Locked inputs (git_sha={git_sha}): {len(controls)} controls "
        f"({n_locked['calibration']} calibration / {n_locked['withheld']} withheld), "
        f"sigma_sys = {sigma_sys.to_value(u.arcsec):.4f} arcsec from {locked.path} "
        f"({locked.file_sha256[:12]}...)"
    )
    cache = Cache(args.cache_dir)
    log: list = []

    print("[2/5] Calibration cutouts (IRSA IBE int + unc) ...")
    cal = fetch_split(controls, "calibration", cache, args.size_arcsec, log)
    print(f"[3/5] Empirical PSF + frozen screens from {len(cal)} calibration cutouts ...")
    psf = hf.build_empirical_psf(cal)
    frozen = hf.calibrate_frozen_screens(cal, psf, sigma_sys, args.target_false_alarm)
    frozen_at = datetime.now(UTC).isoformat()
    frozen_sha = canonical_sha256({"psf": psf.as_dict(), "frozen": frozen.as_dict()})
    cfa = frozen.calibration_false_alarms()
    print(
        f"      frozen at {frozen_at} (sha256 {frozen_sha[:16]}...): centroid multiplier "
        f'{frozen.centroid_multiplier:.3f} x hypot(sigma_stat, {frozen.sigma_sys_arcsec:.4f}"); '
        f"Delta chi^2 scale {frozen.dchi2_scale:.3f} -> threshold {frozen.dchi2_threshold:.2f}; "
        f"in-sample FA centroid {_fmt_rate(cfa['centroid'])}, image {_fmt_rate(cfa['image'])}"
    )

    print("[4/5] Withheld cutouts, then the frozen two-arm injection ...")
    wh = fetch_split(controls, "withheld", cache, args.size_arcsec, log)
    res = hf.evaluate_withheld_matched_injection(wh, psf, frozen, seed=args.seed)
    payload = hf.gate2_payload(psf, frozen, res)
    report(payload["withheld"]["summary"], psf, frozen)

    ok = [r for r in log if r["status"] == "ok"]
    diffs = [
        abs(r["centroid_offset_arcsec_recomputed"] - r["centroid_offset_arcsec_locked"]) for r in ok
    ]
    split_counts = {
        "n_calibration_locked": n_locked["calibration"],
        "n_withheld_locked": n_locked["withheld"],
        "n_calibration_fetched": len(cal),
        "n_withheld_fetched": len(wh),
        "n_calibration_evaluated": len(frozen.calibration_star_ids),
        "n_withheld_evaluated": len(res.star_ids),
        "n_psf_stamps": len(psf.star_ids),
    }
    complete = all(
        split_counts[f"n_{s}_evaluated"] == n_locked[s] for s in ("calibration", "withheld")
    )
    status = "ok" if complete and len(ok) == len(controls) else "incomplete"
    finished = datetime.now(UTC).isoformat()
    receipt = {
        "kind": "wise_fits_gate2",
        "data_tier": "reproduced_archive",
        "status": status,
        "code_git_sha": git_sha,
        "git_head": git_head,
        "rng_seed": args.seed,
        "query_window_utc": [started, finished],
        "frozen_at_utc": frozen_at,
        "frozen_screens_sha256": frozen_sha,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "locked_inputs": {
            "sigma_sys_calibration_path": locked.path,
            "sigma_sys_calibration_sha256": locked.file_sha256,
            "sigma_sys_arcsec": float(sigma_sys.to_value(u.arcsec)),
        },
        "split_counts": split_counts,
        "fetch": {
            "service": "NASA/IPAC IRSA IBE cutouts of the AllWISE Atlas coadd selected by the "
            "Gate #1 IRSA SIA query (locked control 'image' URL)",
            "cutout_size_arcsec": args.size_arcsec,
            "n_requested": len(controls),
            "n_ok": len(ok),
            "n_failed": len(controls) - len(ok),
            "n_files_from_cache": sum(
                bool(r.get("int_cached")) + bool(r.get("unc_cached")) for r in ok
            ),
            "gate1_centroid_offset_reproduction_max_abs_diff_arcsec": max(diffs) if diffs else None,
            "files": ok,
            "failures": [r for r in log if r["status"] != "ok"],
        },
        **payload,
        "caveats": CAVEATS,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"wise_fits_gate2-seed{args.seed}-{stamp}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        f"[5/5] Receipt ({status}): {out.as_posix()}  [{len(ok)}/{len(controls)} cutout pairs; "
        f"Gate #1 offset reproduction max |diff| = {receipt['fetch']['gate1_centroid_offset_reproduction_max_abs_diff_arcsec']}]"
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
