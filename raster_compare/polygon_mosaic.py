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
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import reproject, transform_geom
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

from raster_compare.core import DEFAULT_NODATA


ALLOWED_RESAMPLING = {r.name for r in Resampling}
ALLOWED_RASTER_NAMES = {"raster1", "raster2"}
_HAS_SCIPY = importlib.util.find_spec("scipy") is not None
_HAS_FIONA = importlib.util.find_spec("fiona") is not None


DEFAULT_CONFIG: Dict[str, Any] = {
    "outputs": {
        "new_raster": "new_raster.tif",
        "excel_report": "polygon_mosaic_report.xlsx",
        "save_intermediates": True,
    },
    "selection": {
        "inside_polygon": "raster2",
        "outside_polygon": "raster1",
    },
    "output_grid": {
        "mode": "reference",
        "reference": "raster1",
        "extent_polygon": None,
        "crop_to_extent_polygon": False,
        "mask_to_extent_polygon": False,
    },
    "alignment": {
        "resampling": "bilinear",
    },
    "vertical_adjustment": {
        "enabled": True,
        "method": "constant_offset",
        "robust_stat": "median",
        "target": "raster2",
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
    raster1_adjusted: Path | None
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


def _validate_raster_choice(value: Any, label: str) -> str:
    choice = str(value).lower().strip()
    if choice not in ALLOWED_RASTER_NAMES:
        raise ValueError(f"{label} must be one of: raster1, raster2")
    return choice


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

    selection = config.get("selection", {})
    if not isinstance(selection, MutableMapping):
        raise ValueError("selection must be a mapping.")
    selection["inside_polygon"] = _validate_raster_choice(selection.get("inside_polygon", "raster2"), "selection.inside_polygon")
    selection["outside_polygon"] = _validate_raster_choice(selection.get("outside_polygon", "raster1"), "selection.outside_polygon")
    if selection["inside_polygon"] == selection["outside_polygon"]:
        raise ValueError("selection.inside_polygon and selection.outside_polygon must be different rasters.")

    output_grid = config.get("output_grid", {})
    if not isinstance(output_grid, MutableMapping):
        raise ValueError("output_grid must be a mapping.")
    output_grid["reference"] = _validate_raster_choice(output_grid.get("reference", "raster1"), "output_grid.reference")
    mode = str(output_grid.get("mode", "reference")).lower().strip()
    if mode not in {"reference", "extent_polygon"}:
        raise ValueError("output_grid.mode must be either 'reference' or 'extent_polygon'.")
    output_grid["mode"] = mode
    output_grid["crop_to_extent_polygon"] = bool(output_grid.get("crop_to_extent_polygon", False))
    output_grid["mask_to_extent_polygon"] = bool(output_grid.get("mask_to_extent_polygon", False))
    if mode == "extent_polygon" and not output_grid.get("extent_polygon"):
        raise ValueError("output_grid.extent_polygon is required when output_grid.mode is 'extent_polygon'.")

    va_config = config.get("vertical_adjustment", {})
    if not isinstance(va_config, MutableMapping):
        raise ValueError("vertical_adjustment must be a mapping.")
    va_config["target"] = _validate_raster_choice(va_config.get("target", "raster2"), "vertical_adjustment.target")


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


def _geometry_bounds(geometries: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    def walk_coords(coords: Any) -> Iterable[Tuple[float, float]]:
        if isinstance(coords, (list, tuple)) and len(coords) >= 2 and isinstance(coords[0], (int, float)):
            yield float(coords[0]), float(coords[1])
        elif isinstance(coords, (list, tuple)):
            for item in coords:
                yield from walk_coords(item)

    xs: List[float] = []
    ys: List[float] = []
    for geom in geometries:
        for x, y in walk_coords(geom.get("coordinates", [])):
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        raise ValueError("Could not calculate bounds for output extent polygon.")
    return min(xs), min(ys), max(xs), max(ys)


def _rasterize_geometries(geometries: List[Dict[str, Any]], shape: Tuple[int, int], transform: rasterio.Affine) -> np.ndarray:
    mask = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _rasterize_polygon(path: Path, ref_ds: rasterio.io.DatasetReader) -> np.ndarray:
    geometries = _load_polygon_geometries(path, ref_ds.crs)
    return _rasterize_geometries(geometries, (ref_ds.height, ref_ds.width), ref_ds.transform)


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


def _align_raster_to_grid(
    src_path: Path,
    dst_crs: rasterio.crs.CRS,
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
    resampling: str,
) -> Tuple[np.ndarray, rasterio.profiles.Profile, float]:
    resampling_method = Resampling[resampling]

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

    return dst_data, profile, float(dst_nodata)


def _read_reference_grid(
    raster1_path: Path,
    raster2_path: Path,
    output_grid: Mapping[str, Any],
    resampling: str,
) -> Tuple[np.ndarray, np.ndarray, rasterio.profiles.Profile, rasterio.profiles.Profile, float | None, float, np.ndarray | None, Dict[str, Any]]:
    reference_choice = output_grid.get("reference", "raster1")
    reference_path = raster1_path if reference_choice == "raster1" else raster2_path
    other_path = raster2_path if reference_choice == "raster1" else raster1_path
    other_name = "raster2" if reference_choice == "raster1" else "raster1"

    with rasterio.open(reference_path) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_width = ref_ds.width
        ref_height = ref_ds.height
        ref_bounds = ref_ds.bounds
        ref_nodata = ref_ds.nodata
        ref_profile = ref_ds.profile.copy()

        grid_meta: Dict[str, Any] = {
            "mode": output_grid.get("mode", "reference"),
            "reference": reference_choice,
            "base_width": ref_width,
            "base_height": ref_height,
            "base_bounds": tuple(ref_bounds),
        }

        output_mask = None
        dst_transform = ref_transform
        dst_width = ref_width
        dst_height = ref_height

        if output_grid.get("mode") == "extent_polygon":
            extent_polygon = Path(str(output_grid["extent_polygon"])).expanduser().resolve()
            _validate_path(extent_polygon, "output_grid.extent_polygon")
            extent_geoms = _load_polygon_geometries(extent_polygon, ref_crs)
            xmin, ymin, xmax, ymax = _geometry_bounds(extent_geoms)
            win = window_from_bounds(xmin, ymin, xmax, ymax, transform=ref_transform)
            win = win.round_offsets().round_lengths()
            row_off = max(0, int(win.row_off))
            col_off = max(0, int(win.col_off))
            row_stop = min(ref_height, int(win.row_off + win.height))
            col_stop = min(ref_width, int(win.col_off + win.width))
            dst_height = row_stop - row_off
            dst_width = col_stop - col_off
            if dst_height <= 0 or dst_width <= 0:
                raise ValueError("output_grid.extent_polygon does not overlap the selected reference raster grid.")
            window = rasterio.windows.Window(col_off, row_off, dst_width, dst_height)
            dst_transform = window_transform(window, ref_transform)
            output_mask = _rasterize_geometries(extent_geoms, (dst_height, dst_width), dst_transform)
            grid_meta.update(
                {
                    "extent_polygon": str(extent_polygon),
                    "crop_to_extent_polygon": True,
                    "mask_to_extent_polygon": bool(output_grid.get("mask_to_extent_polygon", False)),
                    "width": dst_width,
                    "height": dst_height,
                    "window_col_off": col_off,
                    "window_row_off": row_off,
                }
            )
        else:
            grid_meta.update(
                {
                    "crop_to_extent_polygon": False,
                    "mask_to_extent_polygon": bool(output_grid.get("mask_to_extent_polygon", False)),
                    "width": dst_width,
                    "height": dst_height,
                }
            )

        if reference_choice == "raster1":
            r1_data = ref_ds.read(1, window=rasterio.windows.Window(0, 0, ref_width, ref_height)).astype(np.float32)
            if output_grid.get("mode") == "extent_polygon":
                r1_data = ref_ds.read(1, window=rasterio.windows.Window(grid_meta["window_col_off"], grid_meta["window_row_off"], dst_width, dst_height)).astype(np.float32)
            r2_data, r2_profile, r2_nodata = _align_raster_to_grid(other_path, ref_crs, dst_transform, dst_width, dst_height, resampling)
            r1_profile = ref_profile.copy()
            r1_profile.update({"height": dst_height, "width": dst_width, "transform": dst_transform})
            return r1_data, r2_data, r1_profile, r2_profile, ref_nodata, r2_nodata, output_mask, grid_meta

    with rasterio.open(raster2_path) as ref_ds:
        r2_data = ref_ds.read(1)
        if output_grid.get("mode") == "extent_polygon":
            r2_data = ref_ds.read(1, window=rasterio.windows.Window(grid_meta["window_col_off"], grid_meta["window_row_off"], dst_width, dst_height)).astype(np.float32)
        else:
            r2_data = r2_data.astype(np.float32)
        r1_data, r1_profile, r1_nodata = _align_raster_to_grid(other_path, ref_crs, dst_transform, dst_width, dst_height, resampling)
        r2_profile = ref_profile.copy()
        r2_profile.update({"height": dst_height, "width": dst_width, "transform": dst_transform})
        return r1_data, r2_data, r1_profile, r2_profile, r1_nodata, ref_nodata if ref_nodata is not None else DEFAULT_NODATA, output_mask, grid_meta


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
    adjustment_target: str,
) -> np.ndarray:
    overlap = mask.copy()
    if exclude_mask is not None:
        overlap &= ~exclude_mask
    if adjustment_target == "raster2":
        values = (r2 - r1)[overlap]
    else:
        values = (r1 - r2)[overlap]
    return values.astype(np.float32, copy=False)


def _flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten_mapping(value, full_key)
        else:
            yield full_key, value


def _raster_data_by_name(name: str, r1_data: np.ndarray, r2_data: np.ndarray) -> np.ndarray:
    return r1_data if name == "raster1" else r2_data


def _raster_mask_by_name(name: str, r1_mask: np.ndarray, r2_mask: np.ndarray) -> np.ndarray:
    return r1_mask if name == "raster1" else r2_mask


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

    raster1_adjusted_path = aligned_dir / f"{name}_raster1_adjusted.tif"
    raster2_aligned_path = aligned_dir / f"{name}_raster2_aligned.tif"
    raster2_adjusted_path = aligned_dir / f"{name}_raster2_adjusted.tif"
    dz_overlap_path = rasters_dir / f"{name}_dz_overlap.tif"
    blend_weights_path = rasters_dir / f"{name}_blend_weights.tif"

    resampling = str(config["alignment"]["resampling"]).lower()
    selection = config["selection"]
    output_grid = config["output_grid"]
    inside_choice = selection["inside_polygon"]
    outside_choice = selection["outside_polygon"]
    va_config = config["vertical_adjustment"]
    adjustment_target = va_config.get("target", "raster2")

    r1_data, r2_aligned, r1_profile, r2_profile, r1_nodata, r2_nodata, output_extent_mask, output_grid_meta = _read_reference_grid(
        raster1_path=raster1,
        raster2_path=raster2,
        output_grid=output_grid,
        resampling=resampling,
    )
    r1_mask = _mask_from_nodata(r1_data, r1_nodata)
    r2_mask = _mask_from_nodata(r2_aligned, r2_nodata)
    polygon_geoms = _load_polygon_geometries(polygon_path, r1_profile["crs"])
    polygon_mask = _rasterize_geometries(polygon_geoms, (r1_profile["height"], r1_profile["width"]), r1_profile["transform"])
    valid_overlap_mask = ~(r1_mask | r2_mask)

    exclude_mask = None
    exclude_buffer = int(va_config.get("exclude_polygon_buffer_px", 0))
    if exclude_buffer > 0:
        exclude_mask = _boundary_buffer_mask(polygon_mask, exclude_buffer)

    overlap_values = _overlap_values(r1_data, r2_aligned, valid_overlap_mask, exclude_mask, adjustment_target)
    overlap_stats = _compute_overlap_stats(overlap_values)

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
            reason = f"applied constant offset to {adjustment_target}"

    r1_adjusted = r1_data - offset_value if (apply_offset and adjustment_target == "raster1") else r1_data
    r2_adjusted = r2_aligned - offset_value if (apply_offset and adjustment_target == "raster2") else r2_aligned

    if save_intermediates:
        if adjustment_target == "raster1":
            r1_adj_profile = r1_profile.copy()
            r1_adj_profile.update({"dtype": "float32", "count": 1})
            _write_raster(raster1_adjusted_path, r1_adjusted, r1_adj_profile)
        _write_raster(raster2_aligned_path, r2_aligned, r2_profile)
        _write_raster(raster2_adjusted_path, r2_adjusted, r2_profile)

        dz_overlap = np.full(r1_data.shape, float(r1_nodata or DEFAULT_NODATA), dtype=np.float32)
        if adjustment_target == "raster2":
            dz_overlap[valid_overlap_mask] = (r2_aligned - r1_data)[valid_overlap_mask]
        else:
            dz_overlap[valid_overlap_mask] = (r1_data - r2_aligned)[valid_overlap_mask]
        dz_profile = r1_profile.copy()
        dz_profile.update({"dtype": "float32", "nodata": r1_nodata or DEFAULT_NODATA, "count": 1})
        _write_raster(dz_overlap_path, dz_overlap, dz_profile)

    blend_cfg = config["border_blending"]
    blend_enabled = bool(blend_cfg.get("enabled", True))
    blend_width = int(blend_cfg.get("blend_width_px", 0))

    if blend_enabled:
        inside_weights = _blend_weights(polygon_mask, blend_width)
    else:
        inside_weights = polygon_mask.astype(np.float32)

    inside_data = _raster_data_by_name(inside_choice, r1_adjusted, r2_adjusted)
    outside_data = _raster_data_by_name(outside_choice, r1_adjusted, r2_adjusted)
    inside_mask = _raster_mask_by_name(inside_choice, r1_mask, r2_mask)
    outside_mask = _raster_mask_by_name(outside_choice, r1_mask, r2_mask)

    weights = inside_weights.copy()
    weights = weights * (~inside_mask)

    if output_extent_mask is not None and bool(output_grid.get("mask_to_extent_polygon", False)):
        weights = weights * output_extent_mask

    if save_intermediates:
        weights_profile = r1_profile.copy()
        weights_profile.update({"dtype": "float32", "nodata": 0.0, "count": 1})
        _write_raster(blend_weights_path, weights, weights_profile)

    use_r1_nodata = bool(config.get("nodata", {}).get("use_raster1_nodata", True))
    if use_r1_nodata:
        output_nodata = r1_nodata if r1_nodata is not None else (r2_nodata or DEFAULT_NODATA)
    else:
        output_nodata = r2_nodata or DEFAULT_NODATA

    output = (outside_data * (1.0 - weights)) + (inside_data * weights)
    output[outside_mask & (weights <= 0.0)] = float(output_nodata)
    output[inside_mask & (weights > 0.0)] = float(output_nodata)
    if output_extent_mask is not None and bool(output_grid.get("mask_to_extent_polygon", False)):
        output[~output_extent_mask] = float(output_nodata)

    out_profile = r1_profile.copy()
    out_profile.update({"dtype": "float32", "nodata": output_nodata, "count": 1})
    _write_raster(new_raster_path, output, out_profile)

    blend_band_pixel_count = int(np.sum((weights > 0.0) & (weights < 1.0)))

    file_inventory = [
        {"label": "new_raster", "path": str(new_raster_path)},
    ]

    if save_intermediates:
        if adjustment_target == "raster1":
            file_inventory.append({"label": "raster1_adjusted", "path": str(raster1_adjusted_path)})
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
            "adjustment_target": adjustment_target,
        },
        "selection": {
            "inside_polygon": inside_choice,
            "outside_polygon": outside_choice,
        },
        "output_grid": output_grid_meta,
        "vertical_adjustment": {
            "enabled": bool(va_config.get("enabled", True)),
            "target": adjustment_target,
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
        raster1_adjusted=raster1_adjusted_path if (save_intermediates and adjustment_target == "raster1") else None,
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
