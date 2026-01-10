from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# These are common in geo envs; if missing, add them to requirements/environment
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample points from a raster within a value range and write outputs (GPKG/CSV)."
    )
    parser.add_argument("--raster", required=True, help="Path to input raster.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--name", required=True, help="Base name for outputs.")

    parser.add_argument("--value-min", type=float, required=True, help="Minimum value (inclusive).")
    parser.add_argument("--value-max", type=float, required=True, help="Maximum value (inclusive).")

    parser.add_argument("--method", choices=["random", "regular"], default="random")
    parser.add_argument("--n-points", type=int, default=1000, help="Used for random sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Used for random sampling.")
    parser.add_argument("--spacing", type=float, default=5.0, help="Used for regular sampling (map units).")

    parser.add_argument("--mask-polygon", default=None, help="Optional polygon file to constrain sampling area.")
    parser.add_argument("--nodata-is-invalid", action="store_true", help="Exclude NoData pixels.")

    parser.add_argument("--save-geopackage", action="store_true", help="Write {name}.gpkg (layer=sample_points).")
    parser.add_argument("--save-csv", action="store_true", help="Write {name}.csv (x,y,value).")

    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_polygon(mask_polygon: str | None) -> gpd.GeoDataFrame | None:
    if not mask_polygon:
        return None
    gdf = gpd.read_file(mask_polygon)
    if gdf.empty:
        raise ValueError(f"mask_polygon is empty: {mask_polygon}")
    if gdf.crs is None:
        raise ValueError("mask_polygon has no CRS. Assign CRS before using.")
    return gdf


def _build_valid_mask(
    arr: np.ndarray,
    transform,
    value_min: float,
    value_max: float,
    nodata: float | None,
    nodata_is_invalid: bool,
    polygon_gdf: gpd.GeoDataFrame | None,
    raster_crs,
) -> np.ndarray:
    # range mask
    mask = (arr >= value_min) & (arr <= value_max)

    # nodata
    if nodata_is_invalid and nodata is not None:
        mask &= (arr != nodata)

    # polygon constraint
    if polygon_gdf is not None:
        poly = polygon_gdf.to_crs(raster_crs)
        geom_mask = geometry_mask(
            geometries=[geom for geom in poly.geometry if geom is not None],
            out_shape=arr.shape,
            transform=transform,
            invert=True,  # True inside polygon
            all_touched=False,
        )
        mask &= geom_mask

    return mask


def _pixel_centers_from_mask(mask: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(rows), np.asarray(cols), np.asarray(xs), np.asarray(ys)


def _sample_random(xs: np.ndarray, ys: np.ndarray, vals: np.ndarray, n: int, seed: int):
    if len(xs) == 0:
        return xs, ys, vals
    n = min(n, len(xs))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=n, replace=False)
    return xs[idx], ys[idx], vals[idx]


def _sample_regular_thin(xs: np.ndarray, ys: np.ndarray, vals: np.ndarray, spacing: float):
    """
    Fast "regular-ish" sampling by thinning to a coarse grid using spacing.
    (Keeps at most 1 point per spacing x spacing cell.)
    """
    if len(xs) == 0:
        return xs, ys, vals
    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    gx = np.floor(xs / spacing).astype(np.int64)
    gy = np.floor(ys / spacing).astype(np.int64)

    # combine into a single key
    key = gx * 10_000_000_000 + gy
    _, keep_idx = np.unique(key, return_index=True)
    keep_idx = np.sort(keep_idx)
    return xs[keep_idx], ys[keep_idx], vals[keep_idx]


def main() -> None:
    args = parse_args()

    raster = Path(args.raster).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    name = str(args.name)

    _ensure_dir(outdir)

    poly_gdf = _load_polygon(args.mask_polygon)

    with rasterio.open(raster) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        valid_mask = _build_valid_mask(
            arr=arr,
            transform=transform,
            value_min=args.value_min,
            value_max=args.value_max,
            nodata=nodata,
            nodata_is_invalid=args.nodata_is_invalid,
            polygon_gdf=poly_gdf,
            raster_crs=crs,
        )

        _, _, xs, ys = _pixel_centers_from_mask(valid_mask, transform)
        vals = arr[valid_mask]

    # apply sampling strategy
    if args.method == "random":
        xs, ys, vals = _sample_random(xs, ys, vals, n=args.n_points, seed=args.seed)
    else:
        xs, ys, vals = _sample_regular_thin(xs, ys, vals, spacing=args.spacing)

    gdf = gpd.GeoDataFrame(
        {"value": vals.astype(float)},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )

    gpkg_path = outdir / f"{name}.gpkg"
    csv_path = outdir / f"{name}.csv"
    summary_path = outdir / "summary.txt"

    if args.save_geopackage:
        gdf.to_file(gpkg_path, layer="sample_points", driver="GPKG")

    if args.save_csv:
        df = gdf.copy()
        df["x"] = df.geometry.x
        df["y"] = df.geometry.y
        df = df.drop(columns=["geometry"])

        # Deterministic order for regression tests
        if "y" in df.columns and "x" in df.columns:
            df = df.sort_values(["y", "x"], kind="mergesort")

        # Optional: stabilize float formatting a bit
        for c in df.columns:
            if df[c].dtype.kind in {"f"}:
                df[c] = df[c].round(6)

        df.to_csv(csv_path, index=False)

    summary_path.write_text(
        "\n".join(
            [
                f"raster={raster}",
                f"value_min={args.value_min}",
                f"value_max={args.value_max}",
                f"method={args.method}",
                f"n_requested={args.n_points if args.method == 'random' else 'n/a'}",
                f"seed={args.seed if args.method == 'random' else 'n/a'}",
                f"spacing={args.spacing if args.method == 'regular' else 'n/a'}",
                f"mask_polygon={args.mask_polygon or 'none'}",
                f"points_written={len(gdf)}",
                f"gpkg={gpkg_path if args.save_geopackage else 'disabled'}",
                f"csv={csv_path if args.save_csv else 'disabled'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"- {summary_path}")
    if args.save_geopackage:
        print(f"- {gpkg_path}")
    if args.save_csv:
        print(f"- {csv_path}")


if __name__ == "__main__":
    main()
