from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import mapping, shape


def copy_qgis_assets(outdir: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "qgis"
    dest_dir = outdir / "qgis"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for qml_path in source_dir.glob("*.qml"):
        shutil.copy2(qml_path, dest_dir / qml_path.name)
    return dest_dir


def _crs_info_from_raster(raster_path: Path) -> Tuple[Optional[int], Optional[str], Optional[str], bool]:
    """
    Returns (epsg, proj4, wkt_short, is_projected).
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs

    if crs is None:
        return None, None, None, False

    epsg = crs.to_epsg()
    try:
        proj4 = crs.to_proj4()
    except Exception:
        proj4 = None

    # WKT can be huge; keep it shorter for attributes/logging
    try:
        wkt = crs.to_wkt()
        wkt_short = wkt[:2000]  # avoid gigantic attribute strings
    except Exception:
        wkt_short = None

    return epsg, proj4, wkt_short, bool(crs.is_projected)


def _pixel_area_m2_if_projected_meters(raster_path: Path) -> Tuple[Optional[float], str]:
    """
    If CRS is projected and units are meters, compute pixel area in m².
    Otherwise return None and a short explanation.
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
        transform = src.transform

    if crs is None:
        return None, "no_crs"
    if not crs.is_projected:
        return None, "not_projected"

    # Rasterio CRS doesn't always expose axis_info; keep it simple and robust:
    # Assume projected CRS uses meters for common EPSG (like UTM).
    # If you later want strict unit checking, we can use pyproj.CRS safely.
    area = abs(transform.a * transform.e)
    return float(area), "projected_assumed_meters"


def _geojson_with_optional_crs(
    features_list: list[dict],
    epsg: Optional[int],
) -> dict:
    """
    Build a GeoJSON FeatureCollection.
    Includes a "crs" member when EPSG is known (legacy but widely supported).
    """
    geojson: dict = {
        "type": "FeatureCollection",
        "features": features_list,
    }

    # GeoJSON RFC7946 removed "crs", but QGIS still accepts it and it helps
    # to avoid manual "Set Layer CRS" in many cases.
    if epsg is not None:
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": f"EPSG:{epsg}"},
        }

    return geojson


def polygonize_exceedance(
    abs_dz_path: Path,
    out_vector_path: Path,
    threshold: float,
    overwrite: bool = False,
) -> Path:
    """
    Polygonize abs(dz) exceedance: abs(dz) > threshold.

    Output GeoJSON includes CRS info (EPSG when available) so QGIS loads it correctly.
    """
    out_vector_path = Path(out_vector_path)
    if out_vector_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_vector_path}")
    out_vector_path.parent.mkdir(parents=True, exist_ok=True)

    epsg, proj4, wkt_short, is_projected = _crs_info_from_raster(abs_dz_path)
    pixel_area_m2, pixel_area_note = _pixel_area_m2_if_projected_meters(abs_dz_path)

    with rasterio.open(abs_dz_path) as src:
        data = src.read(1, masked=True)
        mask = (~data.mask) & np.isfinite(data) & (data > threshold)

        shapes_iter = features.shapes(
            mask.astype("uint8"),
            mask=mask,
            transform=src.transform,
        )

        features_list: list[dict] = []
        for geom, value in shapes_iter:
            if value != 1:
                continue

            geom_shape = shape(geom)
            area_map = float(geom_shape.area)
            if area_map == 0:
                continue

            # If projected meters, treat area_map as m²; otherwise leave as map units
            area_m2 = area_map if (is_projected and pixel_area_m2 is not None) else None

            features_list.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom_shape),
                    "properties": {
                        "kind": "abs_dz_exceedance",
                        "threshold": float(threshold),
                        "sign": "abs",
                        "crs_epsg": epsg,
                        "pixel_area_m2": pixel_area_m2,
                        "pixel_area_note": pixel_area_note,
                        "area_map": area_map,
                        "area_m2": area_m2,
                        "area_map_units": "m^2" if (is_projected and pixel_area_m2 is not None) else "map_units^2",
                        "crs_proj4": proj4,
                        "crs_wkt": wkt_short,
                    },
                }
            )

    geojson = _geojson_with_optional_crs(features_list, epsg)

    out_vector_path.write_text(
        json.dumps(geojson, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_vector_path


def polygonize_signed_exceedance(
    dz_path: Path,
    out_positive_path: Path,
    out_negative_path: Path,
    threshold: float,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Polygonize signed dz exceedance:
      - positive: dz > threshold
      - negative: dz < -threshold

    Output GeoJSON includes CRS info (EPSG when available) so QGIS loads it correctly.
    """
    out_positive_path = Path(out_positive_path)
    out_negative_path = Path(out_negative_path)

    if out_positive_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_positive_path}")
    if out_negative_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_negative_path}")

    out_positive_path.parent.mkdir(parents=True, exist_ok=True)
    out_negative_path.parent.mkdir(parents=True, exist_ok=True)

    epsg, proj4, wkt_short, is_projected = _crs_info_from_raster(dz_path)
    pixel_area_m2, pixel_area_note = _pixel_area_m2_if_projected_meters(dz_path)

    with rasterio.open(dz_path) as src:
        data = src.read(1, masked=True)

        def build_features(mask: np.ndarray, sign: str) -> list[dict]:
            shapes_iter = features.shapes(
                mask.astype("uint8"),
                mask=mask,
                transform=src.transform,
            )

            features_list: list[dict] = []
            for geom, value in shapes_iter:
                if value != 1:
                    continue

                geom_shape = shape(geom)
                area_map = float(geom_shape.area)
                if area_map == 0:
                    continue

                area_m2 = area_map if (is_projected and pixel_area_m2 is not None) else None

                features_list.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(geom_shape),
                        "properties": {
                            "kind": "dz_exceedance",
                            "sign": sign,  # "positive" or "negative"
                            "threshold": float(threshold),
                            "crs_epsg": epsg,
                            "pixel_area_m2": pixel_area_m2,
                            "pixel_area_note": pixel_area_note,
                            "area_map": area_map,
                            "area_m2": area_m2,
                            "area_map_units": "m^2" if (is_projected and pixel_area_m2 is not None) else "map_units^2",
                            "crs_proj4": proj4,
                            "crs_wkt": wkt_short,
                        },
                    }
                )
            return features_list

        positive_mask = (~data.mask) & np.isfinite(data) & (data > threshold)
        negative_mask = (~data.mask) & np.isfinite(data) & (data < -threshold)

        positive_features = build_features(positive_mask, "positive")
        negative_features = build_features(negative_mask, "negative")

    # Write positive
    pos_geojson = _geojson_with_optional_crs(positive_features, epsg)
    out_positive_path.write_text(
        json.dumps(pos_geojson, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write negative
    neg_geojson = _geojson_with_optional_crs(negative_features, epsg)
    out_negative_path.write_text(
        json.dumps(neg_geojson, ensure_ascii=False),
        encoding="utf-8",
    )

    return out_positive_path, out_negative_path
