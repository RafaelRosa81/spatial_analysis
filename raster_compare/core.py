"""Core raster alignment and comparison utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

DEFAULT_NODATA = -9999.0


def _ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _validate_input_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")


def _validate_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists and overwrite is False: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_resampling(resampling: str) -> Resampling:
    try:
        return Resampling[resampling]
    except KeyError as exc:
        valid = ", ".join([r.name for r in Resampling])
        raise ValueError(
            f"Unsupported resampling '{resampling}'. Valid options: {valid}"
        ) from exc


def align_to_reference(
    src_path: str | Path,
    ref_path: str | Path,
    out_path: str | Path,
    resampling: str = "bilinear",
    overwrite: bool = False,
) -> Path:
    """
    Align a source raster to match a reference raster grid.

    Parameters
    ----------
    src_path : str | Path
        Path to the source raster to be aligned.
    ref_path : str | Path
        Path to the reference raster providing CRS, transform, and shape.
    out_path : str | Path
        Output path for the aligned raster GeoTIFF.
    resampling : str, default "bilinear"
        Resampling method name from rasterio.enums.Resampling.
    overwrite : bool, default False
        Whether to overwrite an existing output file.

    Returns
    -------
    Path
        Path to the aligned raster.
    """
    src_path = _ensure_path(src_path)
    ref_path = _ensure_path(ref_path)
    out_path = _ensure_path(out_path)

    _validate_input_path(src_path)
    _validate_input_path(ref_path)
    _validate_output_path(out_path, overwrite)

    resampling_method = _parse_resampling(resampling)

    with rasterio.open(ref_path) as ref_ds:
        dst_height = ref_ds.height
        dst_width = ref_ds.width
        dst_crs = ref_ds.crs
        dst_transform = ref_ds.transform

    with rasterio.open(src_path) as src_ds:
        src_nodata = src_ds.nodata
        dst_nodata = src_nodata if src_nodata is not None else DEFAULT_NODATA
        dst_data = np.empty((dst_height, dst_width), dtype=np.float32)

        reproject(
            source=rasterio.band(src_ds, 1),
            destination=dst_data,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling_method,
        )

        profile = src_ds.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": dst_height,
                "width": dst_width,
                "crs": dst_crs,
                "transform": dst_transform,
                "count": 1,
                "dtype": "float32",
                "nodata": dst_nodata,
            }
        )

    with rasterio.open(out_path, "w", **profile) as dst_ds:
        dst_ds.write(dst_data, 1)

    return out_path


def compute_dz(
    raster1_aligned: str | Path,
    raster2_aligned: str | Path,
    out_dz: str | Path,
    out_abs_dz: str | Path,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """
    Compute signed and absolute elevation differences between aligned rasters.

    Parameters
    ----------
    raster1_aligned : str | Path
        Path to the first aligned raster (reference for nodata/metadata).
    raster2_aligned : str | Path
        Path to the second aligned raster.
    out_dz : str | Path
        Output path for the signed difference raster (raster2 - raster1).
    out_abs_dz : str | Path
        Output path for the absolute difference raster.
    overwrite : bool, default False
        Whether to overwrite existing output files.

    Returns
    -------
    (Path, Path)
        Paths to the signed and absolute difference rasters.
    """
    raster1_aligned = _ensure_path(raster1_aligned)
    raster2_aligned = _ensure_path(raster2_aligned)
    out_dz = _ensure_path(out_dz)
    out_abs_dz = _ensure_path(out_abs_dz)

    _validate_input_path(raster1_aligned)
    _validate_input_path(raster2_aligned)
    _validate_output_path(out_dz, overwrite)
    _validate_output_path(out_abs_dz, overwrite)

    with rasterio.open(raster1_aligned) as r1_ds, rasterio.open(
        raster2_aligned
    ) as r2_ds:
        if (
            r1_ds.width != r2_ds.width
            or r1_ds.height != r2_ds.height
            or r1_ds.transform != r2_ds.transform
            or r1_ds.crs != r2_ds.crs
        ):
            raise ValueError("Aligned rasters do not share the same grid.")

        r1 = r1_ds.read(1, masked=True).astype(np.float32)
        r2 = r2_ds.read(1, masked=True).astype(np.float32)
        combined_mask = np.ma.getmaskarray(r1) | np.ma.getmaskarray(r2)

        dz = np.ma.array(r2 - r1, mask=combined_mask)
        abs_dz = np.ma.array(np.abs(dz), mask=combined_mask)

        nodata = r1_ds.nodata
        if nodata is None:
            nodata = r2_ds.nodata
        if nodata is None:
            nodata = DEFAULT_NODATA

        dz_filled = dz.filled(nodata).astype(np.float32)
        abs_dz_filled = abs_dz.filled(nodata).astype(np.float32)

        profile = r1_ds.profile.copy()
        profile.update({"dtype": "float32", "count": 1, "nodata": nodata})

    with rasterio.open(out_dz, "w", **profile) as dz_ds:
        dz_ds.write(dz_filled, 1)

    with rasterio.open(out_abs_dz, "w", **profile) as abs_ds:
        abs_ds.write(abs_dz_filled, 1)

    return out_dz, out_abs_dz


def read_raster_info(path: str | Path) -> Dict[str, object]:
    """
    Read basic raster metadata for inspection.

    Parameters
    ----------
    path : str | Path
        Path to the raster file.

    Returns
    -------
    dict
        Dictionary containing CRS, pixel size, extent, nodata, dtype,
        width, and height.
    """
    path = _ensure_path(path)
    _validate_input_path(path)

    with rasterio.open(path) as ds:
        bounds = ds.bounds
        transform = ds.transform
        crs = ds.crs.to_string() if ds.crs else None
        return {
            "crs": crs,
            "pixel_size": (transform.a, transform.e),
            "extent": (bounds.left, bounds.bottom, bounds.right, bounds.top),
            "nodata": ds.nodata,
            "dtype": ds.dtypes[0],
            "width": ds.width,
            "height": ds.height,
        }
