from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import importlib
import importlib.util
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_geom

from raster_compare.core import DEFAULT_NODATA


ALLOWED_RESAMPLING = {r.name for r in Resampling}
_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
_HAS_FIONA = importlib.util.find_spec("fiona") is not None


DEFAULT_CONFIG: Dict[str, Any] = {
    "outputs": {
        "new_raster": "new_raster.tif",
        "excel_report": "polygon_mosaic_report.xlsx",
        "save_intermediates": True,
    },
    "alignment": {
        "resampling": "bilinear",
    },
    "vertical_adjustment": {
        "enabled": True,
        "method": "constant_offset",
        "robust_stat": "median",
        "mad_threshold": 0.10,
        "min_overlap_pixels": 50000,
        "exclude_polygon_buffer_px": 5,
    },
    "border_blending": {
        "enabled": True,
        "blend_width_px": 5,
        "weight_curve": "linear",
    },
    "nodata": {
        "use_raster1_nodata": True,
    },
}


@dataclass
class MosaicOutputs:
    new_raster: Path
    report_path: Path | None
    raster2_aligned: Path | None
    raster2_adjusted: Path | None
    dz_overlap: Path | None
    blend_weights: Path | None


@dataclass
class OverlapStats:
    pixel_count: int
    median: float
    mad: float
    mean: float
    std: float
    p05: float
    p25: float
    p75: float
    p95: float


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            base[key] = _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _resolve_polygon_mosaic_config(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        **DEFAULT_CONFIG,
    }
    config = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in config.items()
    }

    mosaic_section = raw_config.get("polygon_mosaic") or {}
    if not isinstance(mosaic_section, Mapping):
        raise ValueError("polygon_mosaic section must be a mapping.")

    _deep_update(config, mosaic_section)

    for key in ("name", "outdir", "excel", "resampling", "raster1", "raster2", "polygon"):
        if key in raw_config and raw_config[key] is not None:
            config[key] = raw_config[key]

    if "alignment" not in mosaic_section or "resampling" not in mosaic_section.get("alignment", {}):
        if raw_config.get("resampling"):
            config["alignment"]["resampling"] = raw_config["resampling"]

    if "excel" not in config:
        config["excel"] = True

    config["pipeline"] = "polygon_mosaic"
    return config


def _validate_polygon_mosaic_config(config: Mapping[str, Any]) -> None:
    missing = [key for key in ("raster1", "raster2", "outdir", "name") if not config.get(key)]
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")

    if not config.get("polygon"):
        raise ValueError("polygon_mosaic pipeline requires a 'polygon' path in the config.")

    resampling = str(config.get("alignment", {}).get("resampling", "")).lower()
    if resampling not in {r.lower() for r in ALLOWED_RESAMPLING}:
        allowed = ", ".join(sorted(ALLOWED_RESAMPLING))
        raise ValueError(f"alignment.resampling must be one of: {allowed}")


def resolve_polygon_mosaic_config(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    config = _resolve_polygon_mosaic_config(raw_config)
    _validate_polygon_mosaic_config(config)
    return config


def _load_polygon_geometries(path: Path, target_crs: rasterio.crs.CRS) -> List[Dict[str, Any]]:
    if not _HAS_FIONA:
        raise ImportError(
            "fiona is required to read polygon inputs. Install it via conda or pip."
        )

    fiona = importlib.import_module("fiona")
    geometries: List[Dict[str, Any]] = []
    with fiona.open(path) as src:
        src_crs = rasterio.crs.CRS.from_user_input(src.crs) if src.crs else None
        for feature in src:
            geom = feature.get("geometry")
            if geom is None:
                continue
            if src_crs and src_crs != target_crs:
                geom = transform_geom(src_crs, target_crs, geom)
            geometries.append(geom)

    if not geometries:
        raise ValueError(f"Polygon file contains no geometries: {path}")

    return geometries


def _rasterize_polygon(path: Path, ref_ds: rasterio.io.DatasetReader) -> np.ndarray:
    geometries = _load_polygon_geometries(path, ref_ds.crs)
    mask = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _align_raster_to_reference(
    src_path: Path,
    ref_ds: rasterio.io.DatasetReader,
    resampling: str,
) -> Tuple[np.ndarray, rasterio.profiles.Profile, float]:
    resampling_method = Resampling[resampling]

    with rasterio.open(src_path) as src_ds:
        src_nodata = src_ds.nodata
        dst_nodata = src_nodata if src_nodata is not None else DEFAULT_NODATA
        dst_data = np.empty((ref_ds.height, ref_ds.width), dtype=np.float32)

        reproject(
            source=rasterio.band(src_ds, 1),
            destination=dst_data,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            src_nodata=src_nodata,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            dst_nodata=dst_nodata,
            resampling=resampling_method,
        )

        profile = src_ds.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": ref_ds.height,
                "width": ref_ds.width,
                "crs": ref_ds.crs,
                "transform": ref_ds.transform,
                "count": 1,
                "dtype": "float32",
                "nodata": dst_nodata,
            }
        )

    return dst_data, profile, float(dst_nodata)


def _mask_from_nodata(data: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = ~np.isfinite(data)
    if nodata is not None:
        mask |= data == nodata
    return mask


def _compute_overlap_stats(values: np.ndarray) -> OverlapStats:
    if values.size == 0:
        nan = float("nan")
        return OverlapStats(0, nan, nan, nan, nan, nan, nan, nan, nan)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return OverlapStats(
        pixel_count=int(values.size),
        median=median,
        mad=mad,
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=0)),
        p05=float(np.quantile(values, 0.05)),
        p25=float(np.quantile(values, 0.25)),
        p75=float(np.quantile(values, 0.75)),
        p95=float(np.quantile(values, 0.95)),
    )


def _validate_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _boundary_buffer_mask(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros_like(mask, dtype=bool)
    dist = _distance_to_boundary(mask)
    return (dist <= float(width)) & mask


def _distance_to_boundary(mask: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        # scipy distance_transform_edt counts the boundary as 1; subtract 1 for consistency with fallback.
        distance_transform_edt = importlib.import_module("scipy.ndimage").distance_transform_edt
        dist = distance_transform_edt(mask)
        return np.maximum(dist - 1.0, 0.0)

    height, width = mask.shape
    dist = np.full(mask.shape, np.inf, dtype=np.float32)
    inside = mask.astype(bool)
    if not inside.any():
        return dist

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    import heapq

    heap: List[Tuple[float, int, int]] = []
    for row in range(height):
        for col in range(width):
            if not inside[row, col]:
                continue
            for dr, dc, _ in neighbors:
                rr = row + dr
                cc = col + dc
                if rr < 0 or cc < 0 or rr >= height or cc >= width or not inside[rr, cc]:
                    dist[row, col] = 0.0
                    heapq.heappush(heap, (0.0, row, col))
                    break

    while heap:
        d, row, col = heapq.heappop(heap)
        if d > dist[row, col]:
            continue
        for dr, dc, cost in neighbors:
            rr = row + dr
            cc = col + dc
            if rr < 0 or cc < 0 or rr >= height or cc >= width:
                continue
            if not inside[rr, cc]:
                continue
            nd = d + cost
            if nd < dist[rr, cc]:
                dist[rr, cc] = nd
                heapq.heappush(heap, (nd, rr, cc))

    return dist


def _blend_weights(mask: np.ndarray, blend_width: int) -> np.ndarray:
    if blend_width <= 0:
        return mask.astype(np.float32)
    dist = _distance_to_boundary(mask)
    weights = np.clip(dist / float(blend_width), 0.0, 1.0)
    weights[~mask] = 0.0
    return weights.astype(np.float32)


def _write_raster(path: Path, data: np.ndarray, profile: rasterio.profiles.Profile) -> None:
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def _overlap_values(
    r1: np.ndarray,
    r2: np.ndarray,
    mask: np.ndarray,
    exclude_mask: np.ndarray | None,
) -> np.ndarray:
    overlap = mask.copy()
    if exclude_mask is not None:
        overlap &= ~exclude_mask
    values = (r2 - r1)[overlap]
    return values.astype(np.float32, copy=False)


def _flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten_mapping(value, full_key)
        else:
            yield full_key, value


def run_polygon_mosaic(config: Mapping[str, Any]) -> Dict[str, Any]:
    raster1 = Path(config["raster1"]).expanduser().resolve()
    raster2 = Path(config["raster2"]).expanduser().resolve()
    polygon_path = Path(config["polygon"]).expanduser().resolve()
    outdir = Path(config["outdir"]).expanduser().resolve()
    name = str(config["name"])
    excel = bool(config.get("excel", True))

    _validate_path(raster1, "raster1")
    _validate_path(raster2, "raster2")
    _validate_path(polygon_path, "polygon")

    outdir.mkdir(parents=True, exist_ok=True)

    outputs_cfg = config["outputs"]
    save_intermediates = bool(outputs_cfg.get("save_intermediates", True))

    aligned_dir = outdir / "aligned"
    rasters_dir = outdir / "rasters"
    report_dir = outdir / "report"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)

    new_raster_path = rasters_dir / str(outputs_cfg.get("new_raster", "new_raster.tif"))
    excel_path = report_dir / str(outputs_cfg.get("excel_report", "polygon_mosaic_report.xlsx"))

    raster2_aligned_path = aligned_dir / f"{name}_raster2_aligned.tif"
    raster2_adjusted_path = aligned_dir / f"{name}_raster2_adjusted.tif"
    dz_overlap_path = rasters_dir / f"{name}_dz_overlap.tif"
    blend_weights_path = rasters_dir / f"{name}_blend_weights.tif"

    resampling = str(config["alignment"]["resampling"]).lower()

    with rasterio.open(raster1) as r1_ds:
        r1_data = r1_ds.read(1).astype(np.float32)
        r1_nodata = r1_ds.nodata
        r1_mask = _mask_from_nodata(r1_data, r1_nodata)
        r1_profile = r1_ds.profile.copy()

        r2_aligned, r2_profile, r2_nodata = _align_raster_to_reference(
            src_path=raster2,
            ref_ds=r1_ds,
            resampling=resampling,
        )

        polygon_mask = _rasterize_polygon(polygon_path, r1_ds)

    r2_mask = _mask_from_nodata(r2_aligned, r2_nodata)
    valid_overlap_mask = ~(r1_mask | r2_mask)

    exclude_mask = None
    exclude_buffer = int(config["vertical_adjustment"].get("exclude_polygon_buffer_px", 0))
    if exclude_buffer > 0:
        exclude_mask = _boundary_buffer_mask(polygon_mask, exclude_buffer)

    overlap_values = _overlap_values(r1_data, r2_aligned, valid_overlap_mask, exclude_mask)
    overlap_stats = _compute_overlap_stats(overlap_values)

    va_config = config["vertical_adjustment"]
    apply_offset = False
    offset_value = 0.0
    reason = "vertical adjustment disabled"
    if bool(va_config.get("enabled", True)):
        if overlap_stats.pixel_count < int(va_config["min_overlap_pixels"]):
            reason = "overlap pixel count below threshold"
        elif overlap_stats.mad > float(va_config["mad_threshold"]):
            reason = "overlap MAD above threshold"
        else:
            apply_offset = True
            offset_value = overlap_stats.median
            reason = "applied constant offset"

    r2_adjusted = r2_aligned - offset_value if apply_offset else r2_aligned

    if save_intermediates:
        _write_raster(raster2_aligned_path, r2_aligned, r2_profile)
        _write_raster(raster2_adjusted_path, r2_adjusted, r2_profile)

        dz_overlap = np.full(r1_data.shape, float(r1_nodata or DEFAULT_NODATA), dtype=np.float32)
        dz_overlap[valid_overlap_mask] = (r2_aligned - r1_data)[valid_overlap_mask]
        dz_profile = r1_profile.copy()
        dz_profile.update({"dtype": "float32", "nodata": r1_nodata or DEFAULT_NODATA, "count": 1})
        _write_raster(dz_overlap_path, dz_overlap, dz_profile)

    blend_cfg = config["border_blending"]
    blend_enabled = bool(blend_cfg.get("enabled", True))
    blend_width = int(blend_cfg.get("blend_width_px", 0))

    if blend_enabled:
        weights = _blend_weights(polygon_mask, blend_width)
    else:
        weights = polygon_mask.astype(np.float32)

    weights = weights * (~r2_mask)

    if save_intermediates:
        weights_profile = r1_profile.copy()
        weights_profile.update({"dtype": "float32", "nodata": 0.0, "count": 1})
        _write_raster(blend_weights_path, weights, weights_profile)

    use_r1_nodata = bool(config.get("nodata", {}).get("use_raster1_nodata", True))
    if use_r1_nodata:
        output_nodata = r1_nodata if r1_nodata is not None else (r2_nodata or DEFAULT_NODATA)
    else:
        output_nodata = r2_nodata or DEFAULT_NODATA

    output = (r1_data * (1.0 - weights)) + (r2_adjusted * weights)
    output[r1_mask] = float(output_nodata)

    out_profile = r1_profile.copy()
    out_profile.update({"dtype": "float32", "nodata": output_nodata, "count": 1})
    _write_raster(new_raster_path, output, out_profile)

    blend_band_pixel_count = int(np.sum((weights > 0.0) & (weights < 1.0)))

    file_inventory = [
        {"label": "new_raster", "path": str(new_raster_path)},
    ]

    if save_intermediates:
        file_inventory.extend(
            [
                {"label": "raster2_aligned", "path": str(raster2_aligned_path)},
                {"label": "raster2_adjusted", "path": str(raster2_adjusted_path)},
                {"label": "dz_overlap", "path": str(dz_overlap_path)},
                {"label": "blend_weights", "path": str(blend_weights_path)},
            ]
        )

    report_path = None
    if excel:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = excel_path
        file_inventory.append({"label": "excel_report", "path": str(report_path)})

    metrics = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "overlap_stats": {
            "pixel_count": overlap_stats.pixel_count,
            "median": overlap_stats.median,
            "mad": overlap_stats.mad,
            "mean": overlap_stats.mean,
            "std": overlap_stats.std,
            "p05": overlap_stats.p05,
            "p25": overlap_stats.p25,
            "p75": overlap_stats.p75,
            "p95": overlap_stats.p95,
            "mad_threshold": float(va_config["mad_threshold"]),
            "min_overlap_pixels": int(va_config["min_overlap_pixels"]),
        },
        "vertical_adjustment": {
            "enabled": bool(va_config.get("enabled", True)),
            "applied": apply_offset,
            "offset": float(offset_value),
            "reason": reason,
            "mad_threshold": float(va_config["mad_threshold"]),
            "min_overlap_pixels": int(va_config["min_overlap_pixels"]),
        },
        "border_blend": {
            "enabled": blend_enabled,
            "blend_width_px": blend_width,
            "weight_curve": str(blend_cfg.get("weight_curve", "linear")),
            "blend_band_pixel_count": blend_band_pixel_count,
        },
        "file_inventory": file_inventory,
    }

    config_flat = {key: value for key, value in _flatten_mapping(config)}
    metrics["config_flat"] = config_flat

    outputs = MosaicOutputs(
        new_raster=new_raster_path,
        report_path=report_path,
        raster2_aligned=raster2_aligned_path if save_intermediates else None,
        raster2_adjusted=raster2_adjusted_path if save_intermediates else None,
        dz_overlap=dz_overlap_path if save_intermediates else None,
        blend_weights=blend_weights_path if save_intermediates else None,
    )

    metrics["outputs"] = {
        "new_raster": str(outputs.new_raster),
        "report_path": str(outputs.report_path) if outputs.report_path else None,
    }

    return metrics
