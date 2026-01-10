from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None


def resolve_sample_points_config(raw_config: dict) -> dict[str, Any]:
    """
    Read the YAML root config, extract section 'sample_points_from_raster_value_range',
    apply defaults, and return a normalized dict config for the pipeline.
    """
    section = raw_config.get("sample_points_from_raster_value_range") or {}
    if not isinstance(section, dict) or not section:
        raise ValueError("Missing or invalid section: sample_points_from_raster_value_range")

    name = str(section.get("name", "sample_points"))
    outdir = Path(section.get("outdir", f"outputs/{name}")).expanduser()

    raster = section.get("raster")
    if not raster:
        raise ValueError("sample_points_from_raster_value_range.raster is required")
    raster_path = Path(str(raster)).expanduser()

    value_min = float(section.get("value_min"))
    value_max = float(section.get("value_max"))
    if value_max < value_min:
        raise ValueError("value_max must be >= value_min")

    sampling = section.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ValueError("sample_points_from_raster_value_range.sampling must be a mapping")

    method = str(sampling.get("method", "random")).lower()
    if method not in {"random", "regular"}:
        raise ValueError("sampling.method must be 'random' or 'regular'")

    n_points = int(sampling.get("n_points", 1000))
    seed = sampling.get("seed", None)
    seed = int(seed) if seed is not None else None

    spacing = sampling.get("spacing", None)
    spacing = float(spacing) if spacing is not None else None

    mask_polygon = section.get("mask_polygon", None)
    mask_polygon_path = Path(str(mask_polygon)).expanduser() if mask_polygon else None

    nodata_is_invalid = bool(section.get("nodata_is_invalid", True))
    save_geopackage = bool(section.get("save_geopackage", True))
    save_csv = bool(section.get("save_csv", True))
    qgis_assets = bool(section.get("qgis_assets", True))

    return {
        "pipeline": "sample_points_from_raster_value_range",
        "name": name,
        "outdir": str(outdir),
        "raster": str(raster_path),
        "value_min": value_min,
        "value_max": value_max,
        "sampling": {
            "method": method,
            "n_points": n_points,
            "seed": seed,
            "spacing": spacing,
        },
        "mask_polygon": str(mask_polygon_path) if mask_polygon_path else None,
        "nodata_is_invalid": nodata_is_invalid,
        "save_geopackage": save_geopackage,
        "save_csv": save_csv,
        "qgis_assets": qgis_assets,
    }


def run_sample_points_from_raster_value_range(config: dict[str, Any]) -> dict[str, str]:
    """
    Sample points from raster cells whose values fall within [value_min, value_max].
    Outputs a CSV and optionally a GeoPackage.
    """
    outdir = Path(config["outdir"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raster_path = Path(config["raster"]).expanduser().resolve()
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    name = str(config["name"])
    vmin = float(config["value_min"])
    vmax = float(config["value_max"])

    sampling = config.get("sampling") or {}
    method = str(sampling.get("method", "random")).lower()
    n_points = int(sampling.get("n_points", 1000))
    seed = sampling.get("seed", None)
    seed = int(seed) if seed is not None else None
    spacing = sampling.get("spacing", None)
    spacing = float(spacing) if spacing is not None else None

    nodata_is_invalid = bool(config.get("nodata_is_invalid", True))
    save_geopackage = bool(config.get("save_geopackage", True))
    save_csv = bool(config.get("save_csv", True))

    rng = np.random.default_rng(seed)

    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=False).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        raster_crs = src.crs

        valid = np.isfinite(band)
        if nodata_is_invalid and nodata is not None:
            valid &= band != nodata
        valid &= (band >= vmin) & (band <= vmax)

        rows, cols = np.where(valid)
        if rows.size == 0:
            raise ValueError(f"No pixels found in range [{vmin}, {vmax}] after nodata filtering.")

        if method == "random":
            take = min(n_points, rows.size)
            idx = rng.choice(rows.size, size=take, replace=False)
            rows_s = rows[idx]
            cols_s = cols[idx]
        else:
            # regular spacing: convert spacing (map units) to approx pixel step
            if spacing is None or spacing <= 0:
                raise ValueError("sampling.spacing must be provided and > 0 when method='regular'.")
            px = max(1, int(round(spacing / abs(transform.a))))
            mask = (rows % px == 0) & (cols % px == 0)
            rows_s = rows[mask]
            cols_s = cols[mask]
            if rows_s.size == 0:
                # fallback: if too strict, do a random take
                take = min(n_points, rows.size)
                idx = rng.choice(rows.size, size=take, replace=False)
                rows_s = rows[idx]
                cols_s = cols[idx]

        xs, ys = rasterio.transform.xy(transform, rows_s, cols_s, offset="center")
        vals = band[rows_s, cols_s]

    df = pd.DataFrame(
        {
            "x": np.array(xs, dtype=float),
            "y": np.array(ys, dtype=float),
            "value": np.array(vals, dtype=float),
            "row": rows_s.astype(int),
            "col": cols_s.astype(int),
        }
    )

    # stable output ordering
    df = df.sort_values(["y", "x"], kind="mergesort").reset_index(drop=True)
    df["value"] = df["value"].round(6)

    outputs: dict[str, str] = {"outdir": str(outdir)}

    if save_csv:
        csv_path = outdir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)

    if save_geopackage:
        if gpd is None or Point is None:
            raise RuntimeError("save_geopackage=true but geopandas is not installed.")
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df["x"], df["y"])],
            crs=raster_crs,
        )
        gpkg_path = outdir / f"{name}.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG")
        outputs["gpkg"] = str(gpkg_path)

    return outputs
