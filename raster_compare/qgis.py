from __future__ import annotations

import json
import shutil
from pathlib import Path

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


def polygonize_exceedance(
    abs_dz_path: Path,
    out_vector_path: Path,
    threshold: float,
    overwrite: bool = False,
) -> Path:
    out_vector_path = Path(out_vector_path)
    if out_vector_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_vector_path}")
    out_vector_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(abs_dz_path) as src:
        data = src.read(1, masked=True)
        mask = (~data.mask) & np.isfinite(data) & (data > threshold)
        shapes_iter = features.shapes(
            mask.astype("uint8"),
            mask=mask,
            transform=src.transform,
        )
        features_list = []
        for geom, value in shapes_iter:
            if value != 1:
                continue
            geom_shape = shape(geom)
            area_map = float(geom_shape.area)
            if area_map == 0:
                continue
            features_list.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom_shape),
                    "properties": {
                        "threshold": float(threshold),
                        "area_map": area_map,
                    },
                }
            )

    geojson = {
        "type": "FeatureCollection",
        "features": features_list,
    }
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
    out_positive_path = Path(out_positive_path)
    out_negative_path = Path(out_negative_path)
    if out_positive_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_positive_path}")
    if out_negative_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_negative_path}")
    out_positive_path.parent.mkdir(parents=True, exist_ok=True)
    out_negative_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dz_path) as src:
        data = src.read(1, masked=True)
        is_projected = bool(src.crs and src.crs.is_projected)

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
                features_list.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(geom_shape),
                        "properties": {
                            "sign": sign,
                            "threshold": float(threshold),
                            "area_m2": area_map if is_projected else None,
                        },
                    }
                )
            return features_list

        positive_mask = (~data.mask) & np.isfinite(data) & (data > threshold)
        negative_mask = (~data.mask) & np.isfinite(data) & (data < -threshold)

        positive_features = build_features(positive_mask, "positive")
        negative_features = build_features(negative_mask, "negative")

    out_positive_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": positive_features}, ensure_ascii=False),
        encoding="utf-8",
    )
    out_negative_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": negative_features}, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_positive_path, out_negative_path
