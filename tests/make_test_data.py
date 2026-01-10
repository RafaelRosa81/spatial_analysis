from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import geopandas as gpd
from shapely.geometry import box


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Small raster specs
    width, height = 120, 100
    pixel_size = 1.0
    # simple local CRS; for testing only
    crs = "EPSG:3857"
    transform = from_origin(0.0, 1000.0, pixel_size, pixel_size)
    nodata = -9999.0

    # Deterministic fields
    # raster_a values around 34.5..35.5 with smooth gradients (so the [34.8,35] band exists)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    a = 34.5 + 0.008 * xx + 0.004 * yy + 0.05 * np.sin(xx / 8.0) * np.cos(yy / 9.0)

    # raster_b is a small perturbation, plus a localized bump inside a region
    bump = np.exp(-(((xx - 60) ** 2) / (2 * 12**2) + ((yy - 45) ** 2) / (2 * 10**2))).astype(np.float32)
    b = a + 0.12 * bump - 0.03  # shift slightly so dz has positive and negative areas

    # Add NoData frame around edges (to test nodata handling)
    a[:2, :] = nodata
    a[-2:, :] = nodata
    a[:, :2] = nodata
    a[:, -2:] = nodata

    b[:2, :] = nodata
    b[-2:, :] = nodata
    b[:, :2] = nodata
    b[:, -2:] = nodata

    raster_a_path = data_dir / "raster_a.tif"
    raster_b_path = data_dir / "raster_b.tif"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "LZW",
    }

    with rasterio.open(raster_a_path, "w", **profile) as dst:
        dst.write(a.astype(np.float32), 1)

    with rasterio.open(raster_b_path, "w", **profile) as dst:
        dst.write(b.astype(np.float32), 1)

    # Polygon that overlaps central area
    # Coordinates in same CRS as raster (EPSG:3857)
    poly = box(25, 915, 95, 975)  # xmin, ymin, xmax, ymax
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs=crs)

    polygon_path = data_dir / "polygon.shp"
    # overwrite shapefile set if exists
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        p = polygon_path.with_suffix(ext)
        if p.exists():
            p.unlink()

    gdf.to_file(polygon_path)

    print("Created test data:")
    print(f"- {raster_a_path}")
    print(f"- {raster_b_path}")
    print(f"- {polygon_path}")


if __name__ == "__main__":
    main()
