from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

# geopandas/shapely are used only if vector outputs exist
try:
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover
    gpd = None


@dataclass
class Paths:
    repo_root: Path
    tests_dir: Path
    configs_dir: Path
    data_dir: Path
    outputs_dir: Path
    golden_dir: Path
    workspace_yml: Path


PIPELINES = [
    "raster_diff",
    "polygon_mosaic",
    "sample_points_from_raster_value_range",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run regression tests using golden-output fingerprints.")
    p.add_argument(
        "--mode",
        choices=["compare", "update"],
        required=True,
        help="update: regenerate golden fingerprints; compare: validate outputs match golden",
    )
    p.add_argument(
        "--pipelines",
        nargs="*",
        default=PIPELINES,
        help=f"Pipelines to run (default: {', '.join(PIPELINES)})",
    )
    return p.parse_args()


def get_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"
    return Paths(
        repo_root=repo_root,
        tests_dir=tests_dir,
        configs_dir=tests_dir / "configs",
        data_dir=tests_dir / "data",
        outputs_dir=tests_dir / "_outputs",
        golden_dir=tests_dir / "golden",
        workspace_yml=tests_dir / "configs" / "workspace_regression.yml",
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_float_array(arr: np.ndarray, ndigits: int = 6) -> np.ndarray:
    # Use float64 for stability; round to reduce tiny platform diffs
    arr64 = arr.astype(np.float64, copy=False)
    return np.round(arr64, ndigits)


def raster_fingerprint(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        band = src.read(1, masked=False)
        nodata = src.nodata

        arr = band.astype(np.float64)

        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        arrn = normalize_float_array(arr, ndigits=6)

        # stats (nan-aware)
        stats = {
            "min": float(np.nanmin(arrn)) if np.isfinite(np.nanmin(arrn)) else None,
            "max": float(np.nanmax(arrn)) if np.isfinite(np.nanmax(arrn)) else None,
            "mean": float(np.nanmean(arrn)) if np.isfinite(np.nanmean(arrn)) else None,
            "std": float(np.nanstd(arrn)) if np.isfinite(np.nanstd(arrn)) else None,
        }

        # hash of normalized values (replace nan with sentinel)
        nan_sentinel = -1.23456789e20
        arr_hashable = np.nan_to_num(arrn, nan=nan_sentinel)
        digest = sha256_bytes(arr_hashable.tobytes())

        # transform rounded for stability
        t = src.transform
        transform = [round(float(v), 9) for v in (t.a, t.b, t.c, t.d, t.e, t.f)]

        return {
            "type": "raster",
            "path": str(path.as_posix()),
            "width": src.width,
            "height": src.height,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": transform,  # [a,b,c,d,e,f]
            "nodata": nodata,
            "dtype": src.dtypes[0],
            "stats": stats,
            "sha256_norm": digest,
        }


def vector_fingerprint(path: Path) -> dict[str, Any]:
    if gpd is None:
        raise RuntimeError(
            f"geopandas is required to fingerprint vector outputs but isn't available. Missing: {path}"
        )

    gdf = gpd.read_file(path)
    crs = gdf.crs.to_string() if gdf.crs else None
    n = int(len(gdf))

    if n == 0:
        bounds = None
    else:
        b = gdf.total_bounds  # xmin, ymin, xmax, ymax
        bounds = [round(float(x), 6) for x in b.tolist()]

    # attribute summary (numeric only)
    numeric_stats: dict[str, dict[str, float]] = {}
    for col in gdf.columns:
        if col == "geometry":
            continue
        if np.issubdtype(gdf[col].dtype, np.number):
            s = gdf[col].to_numpy(dtype=float)
            if s.size:
                numeric_stats[col] = {
                    "min": float(np.nanmin(s)),
                    "max": float(np.nanmax(s)),
                    "mean": float(np.nanmean(s)),
                }

    # geometry hash: normalize by hashing WKB after sorting by bounds
    if n > 0:
        tmp = gdf.copy()
        tmp["_bx"] = tmp.geometry.bounds["minx"]
        tmp["_by"] = tmp.geometry.bounds["miny"]
        tmp = tmp.sort_values(["_by", "_bx"], kind="mergesort")
        wkbs = b"".join([geom.wkb for geom in tmp.geometry if geom is not None])
        geom_hash = sha256_bytes(wkbs)
    else:
        geom_hash = sha256_bytes(b"")

    return {
        "type": "vector",
        "path": str(path.as_posix()),
        "driver": path.suffix.lower(),
        "crs": crs,
        "feature_count": n,
        "bounds": bounds,
        "columns": [c for c in gdf.columns],
        "numeric_stats": numeric_stats,
        "geom_sha256": geom_hash,
    }


def csv_fingerprint(path: Path) -> dict[str, Any]:
    # Normalize CSV by:
    # - reading lines
    # - splitting by comma (simple; our generated CSV is simple)
    # - sorting rows (excluding header)
    # - rounding float tokens
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]

    if not lines:
        digest = sha256_bytes(b"")
        return {"type": "csv", "path": str(path.as_posix()), "rows": 0, "sha256_norm": digest}

    header = lines[0]
    rows = lines[1:]

    def norm_token(tok: str) -> str:
        tok = tok.strip()
        # try float rounding
        try:
            v = float(tok)
            return f"{v:.6f}"
        except Exception:
            return tok

    norm_rows = []
    for r in rows:
        toks = r.split(",")
        toks = [norm_token(t) for t in toks]
        norm_rows.append(",".join(toks))

    norm_rows.sort()
    normalized = header + "\n" + "\n".join(norm_rows) + "\n"
    digest = sha256_bytes(normalized.encode("utf-8"))

    return {
        "type": "csv",
        "path": str(path.as_posix()),
        "rows": len(norm_rows),
        "sha256_norm": digest,
    }


def generic_fingerprint(path: Path) -> dict[str, Any]:
    # For small text/json/other: hash bytes + size
    return {
        "type": "file",
        "path": str(path.as_posix()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fingerprint_tree(root_dir: Path) -> dict[str, Any]:
    """
    Create a stable fingerprint for an output directory.
    """
    if not root_dir.exists():
        return {"root": str(root_dir.as_posix()), "files": []}

    files = []
    for p in sorted(root_dir.rglob("*")):
        if p.is_dir():
            continue

        # ignore temp/lock files if any
        if p.name.lower().endswith(".lock"):
            continue

        suf = p.suffix.lower()
        try:
            if suf in [".tif", ".tiff"]:
                files.append(raster_fingerprint(p))
            elif suf in [".gpkg", ".geojson", ".shp"]:
                files.append(vector_fingerprint(p))
            elif suf == ".csv":
                files.append(csv_fingerprint(p))
            else:
                files.append(generic_fingerprint(p))
        except Exception as e:
            files.append(
                {
                    "type": "error",
                    "path": str(p.as_posix()),
                    "error": str(e),
                }
            )

    # overall digest (so a quick check is possible)
    canonical = json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
    overall = sha256_bytes(canonical)

    return {"root": str(root_dir.as_posix()), "sha256": overall, "files": files}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_test_data(paths: Paths) -> None:
    # Create test data if missing
    raster_a = paths.data_dir / "raster_a.tif"
    raster_b = paths.data_dir / "raster_b.tif"
    polygon = paths.data_dir / "polygon.shp"

    if raster_a.exists() and raster_b.exists() and polygon.exists():
        return

    print("Test data missing; generating...")
    subprocess.check_call([sys.executable, str(paths.tests_dir / "make_test_data.py")])


def clean_outputs(paths: Paths) -> None:
    if paths.outputs_dir.exists():
        shutil.rmtree(paths.outputs_dir)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)


def run_pipeline(paths: Paths, pipeline: str) -> None:
    # Create a temp config that sets top-level pipeline selector
    raw = load_json_from_yaml(paths.workspace_yml)
    raw["pipeline"] = pipeline

    tmp_cfg = paths.outputs_dir / f"_tmp_{pipeline}.yml"
    tmp_cfg.write_text(yaml_dump(raw), encoding="utf-8")

    # Run your normal entrypoint
    cmd = [sys.executable, "-m", "scripts.run_from_config", "--config", str(tmp_cfg)]
    print("Running:", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=str(paths.repo_root))


def load_json_from_yaml(yaml_path: Path) -> dict[str, Any]:
    import yaml  # local import to avoid adding dependency in environments that already have it

    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def yaml_dump(obj: dict[str, Any]) -> str:
    import yaml  # local import

    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def compare_dicts(golden: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """
    Return list of human-readable diffs. We compare the full fingerprint JSON
    (already normalized/rounded).
    """
    diffs = []
    if golden.get("sha256") != current.get("sha256"):
        diffs.append(f"Root sha256 differs: golden={golden.get('sha256')} current={current.get('sha256')}")

    # Compare per-file entries by "path"
    g_files = {f.get("path"): f for f in golden.get("files", [])}
    c_files = {f.get("path"): f for f in current.get("files", [])}

    missing = sorted(set(g_files) - set(c_files))
    extra = sorted(set(c_files) - set(g_files))
    if missing:
        diffs.append(f"Missing files: {missing}")
    if extra:
        diffs.append(f"Extra files: {extra}")

    for p in sorted(set(g_files) & set(c_files)):
        gf = g_files[p]
        cf = c_files[p]
        # strict compare dicts (already stabilized)
        if gf != cf:
            diffs.append(f"Fingerprint differs for: {p}")

    return diffs


def main() -> None:
    args = parse_args()
    paths = get_paths()

    if not paths.workspace_yml.exists():
        raise FileNotFoundError(f"Missing workspace regression config: {paths.workspace_yml}")

    ensure_test_data(paths)
    clean_outputs(paths)

    # Run all requested pipelines
    for pipe in args.pipelines:
        if pipe not in PIPELINES:
            raise ValueError(f"Unknown pipeline: {pipe}")
        run_pipeline(paths, pipe)

    # Fingerprint each pipeline output folder (as defined in workspace yml)
    workspace = load_json_from_yaml(paths.workspace_yml)
    results: dict[str, Any] = {}

    for pipe in args.pipelines:
        section = workspace.get(pipe) or {}
        outdir = Path(section.get("outdir", paths.outputs_dir / pipe))
        # outdir in YAML is relative; interpret relative to repo root
        outdir_abs = (paths.repo_root / outdir).resolve()
        fp = fingerprint_tree(outdir_abs)
        results[pipe] = fp

        current_fp_path = paths.outputs_dir / f"{pipe}_fingerprint.json"
        write_json(current_fp_path, fp)
        print(f"Wrote current fingerprint: {current_fp_path}")

    if args.mode == "update":
        for pipe, fp in results.items():
            golden_fp_path = paths.golden_dir / pipe / "fingerprint.json"
            write_json(golden_fp_path, fp)
            print(f"Updated golden fingerprint: {golden_fp_path}")
        print("Golden fingerprints updated.")
        return

    # compare mode
    failures = []
    for pipe, current_fp in results.items():
        golden_fp_path = paths.golden_dir / pipe / "fingerprint.json"
        if not golden_fp_path.exists():
            failures.append(f"[{pipe}] missing golden fingerprint: {golden_fp_path}")
            continue

        golden_fp = load_json(golden_fp_path)
        diffs = compare_dicts(golden_fp, current_fp)
        if diffs:
            failures.append(f"[{pipe}] " + "; ".join(diffs))

    if failures:
        print("\nREGRESSION FAILURES:")
        for f in failures:
            print("-", f)
        sys.exit(2)

    print("\nAll regression tests passed.")


if __name__ == "__main__":
    main()
