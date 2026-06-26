from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


POINT_GEOMETRIES = {"Point", "MultiPoint"}
DEFAULT_POINT_PREDICATE = "within"
DEFAULT_POLYGON_PREDICATE = "intersects"


def _as_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _normalise_layers(layers: Sequence[str | Path | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise layer specifications from a YAML list.

    A layer can be written either as a path string or as a mapping with:
    - path: path to the vector layer
    - name: optional output-column name (defaults to the file stem)
    - predicate: optional spatial predicate overriding the geometry default
    - layer: optional layer name for a GeoPackage or other multi-layer datasource
    """
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)) or not layers:
        raise ValueError("layers must be a non-empty YAML list.")

    normalised: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for item in layers:
        if isinstance(item, (str, Path)):
            spec: dict[str, Any] = {"path": str(item)}
        elif isinstance(item, Mapping):
            spec = dict(item)
        else:
            raise ValueError("Every layer must be either a path string or a mapping.")

        if not spec.get("path"):
            raise ValueError("Each layer mapping must include a non-empty 'path'.")

        path = _as_path(spec["path"], "Input layer")
        name = str(spec.get("name") or path.stem).strip()
        if not name:
            raise ValueError(f"Could not determine an output name for layer: {path}")
        if name in used_names:
            raise ValueError(
                f"Duplicate output layer name '{name}'. Use the optional 'name' key to distinguish layers."
            )
        used_names.add(name)

        predicate = spec.get("predicate")
        if predicate is not None:
            predicate = str(predicate).lower()

        normalised.append(
            {
                "path": path,
                "name": name,
                "predicate": predicate,
                "layer": spec.get("layer"),
            }
        )

    return normalised


def resolve_sector_counts_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and resolve a ``count_features_by_sector`` YAML configuration."""
    section = raw_config.get("count_features_by_sector", raw_config)
    if not isinstance(section, Mapping):
        raise ValueError("count_features_by_sector must be a YAML mapping.")

    required = {"sectors", "sector_id_field", "layers", "outdir"}
    missing = sorted(key for key in required if not section.get(key))
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")

    sectors_path = _as_path(section["sectors"], "Sectors layer")
    outdir = Path(section["outdir"]).expanduser().resolve()
    name = str(section.get("name") or "feature_counts_by_sector").strip()
    if not name:
        raise ValueError("name must not be empty.")

    point_predicate = str(section.get("point_predicate", DEFAULT_POINT_PREDICATE)).lower()
    polygon_predicate = str(section.get("polygon_predicate", DEFAULT_POLYGON_PREDICATE)).lower()

    return {
        "pipeline": "count_features_by_sector",
        "sectors": sectors_path,
        "sectors_layer": section.get("sectors_layer"),
        "sector_id_field": str(section["sector_id_field"]),
        "layers": _normalise_layers(section["layers"]),
        "outdir": outdir,
        "name": name,
        "point_predicate": point_predicate,
        "polygon_predicate": polygon_predicate,
        "write_csv": bool(section.get("write_csv", True)),
        "write_excel": bool(section.get("write_excel", True)),
        "write_geopackage": bool(section.get("write_geopackage", False)),
    }


def _read_vector(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    kwargs: dict[str, Any] = {}
    if layer:
        kwargs["layer"] = layer
    gdf = gpd.read_file(path, **kwargs)
    if gdf.crs is None:
        raise ValueError(f"Layer has no CRS defined: {path}")
    return gdf


def _count_layer_by_sector(
    sectors: gpd.GeoDataFrame,
    sector_id_field: str,
    layer_spec: Mapping[str, Any],
    point_predicate: str,
    polygon_predicate: str,
) -> pd.Series:
    """Return one count per sector for one input layer.

    Point-like features use ``point_predicate`` and all remaining geometries use
    ``polygon_predicate``. This makes a homogeneous point or polygon layer work
    naturally, while also supporting a mixed-geometry vector layer.
    """
    features = _read_vector(layer_spec["path"], layer_spec.get("layer"))
    column_name = str(layer_spec["name"])

    if features.empty:
        return pd.Series(0, index=sectors[sector_id_field], name=column_name, dtype="int64")

    if features.crs != sectors.crs:
        features = features.to_crs(sectors.crs)

    features = features.loc[features.geometry.notna() & ~features.geometry.is_empty].copy()
    if features.empty:
        return pd.Series(0, index=sectors[sector_id_field], name=column_name, dtype="int64")

    feature_types = features.geometry.geom_type
    point_features = features.loc[feature_types.isin(POINT_GEOMETRIES)]
    non_point_features = features.loc[~feature_types.isin(POINT_GEOMETRIES)]

    join_parts: list[gpd.GeoDataFrame] = []
    override_predicate = layer_spec.get("predicate")

    if not point_features.empty:
        join_parts.append(
            gpd.sjoin(
                point_features[["geometry"]],
                sectors[[sector_id_field, "geometry"]],
                how="inner",
                predicate=override_predicate or point_predicate,
            )
        )

    if not non_point_features.empty:
        join_parts.append(
            gpd.sjoin(
                non_point_features[["geometry"]],
                sectors[[sector_id_field, "geometry"]],
                how="inner",
                predicate=override_predicate or polygon_predicate,
            )
        )

    if not join_parts:
        return pd.Series(0, index=sectors[sector_id_field], name=column_name, dtype="int64")

    joined = pd.concat(join_parts, ignore_index=True)
    return joined.groupby(sector_id_field).size().rename(column_name)


def run_count_features_by_sector(config: Mapping[str, Any]) -> dict[str, Path | pd.DataFrame]:
    """Count point and polygon features from named layers inside each sector.

    Notes
    -----
    - Points use ``within`` by default, so a point precisely on a sector boundary
      is not counted. Set ``point_predicate: covered_by`` when boundary points
      should count.
    - Polygons use ``intersects`` by default. A polygon crossing two sectors is
      counted once in each intersected sector. Set ``polygon_predicate: within``
      to count only polygons completely contained by a sector.
    """
    sectors = _read_vector(config["sectors"], config.get("sectors_layer"))
    sector_id_field = str(config["sector_id_field"])

    if sector_id_field not in sectors.columns:
        raise ValueError(
            f"Field '{sector_id_field}' does not exist in the sectors layer. "
            f"Available fields: {', '.join(map(str, sectors.columns))}"
        )
    if sectors.empty:
        raise ValueError("The sectors layer contains no features.")
    if sectors[sector_id_field].isna().any():
        raise ValueError(f"The sector identifier field '{sector_id_field}' contains null values.")
    if sectors[sector_id_field].duplicated().any():
        raise ValueError(
            f"The sector identifier field '{sector_id_field}' must be unique; duplicate values were found."
        )

    sectors = sectors.loc[sectors.geometry.notna() & ~sectors.geometry.is_empty].copy()
    if sectors.empty:
        raise ValueError("The sectors layer has no valid geometries.")

    result = sectors[[sector_id_field]].copy()
    for layer_spec in config["layers"]:
        counts = _count_layer_by_sector(
            sectors=sectors,
            sector_id_field=sector_id_field,
            layer_spec=layer_spec,
            point_predicate=str(config["point_predicate"]),
            polygon_predicate=str(config["polygon_predicate"]),
        )
        result = result.merge(counts, left_on=sector_id_field, right_index=True, how="left")
        result[layer_spec["name"]] = result[layer_spec["name"]].fillna(0).astype("int64")

    outdir = Path(config["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    name = str(config["name"])
    outputs: dict[str, Path | pd.DataFrame] = {"table": result}

    if config.get("write_csv", True):
        csv_path = outdir / f"{name}.csv"
        result.to_csv(csv_path, index=False, encoding="utf-8-sig")
        outputs["csv"] = csv_path

    if config.get("write_excel", True):
        excel_path = outdir / f"{name}.xlsx"
        config_rows = []
        for key, value in config.items():
            if key == "layers":
                continue
            config_rows.append({"parameter": key, "value": str(value)})
        layers_rows = [
            {
                "name": layer["name"],
                "path": str(layer["path"]),
                "layer": layer.get("layer") or "",
                "predicate": layer.get("predicate") or "geometry default",
            }
            for layer in config["layers"]
        ]
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name="Counts_by_sector", index=False)
            pd.DataFrame(layers_rows).to_excel(writer, sheet_name="Input_layers", index=False)
            pd.DataFrame(config_rows).to_excel(writer, sheet_name="Configuration", index=False)
        outputs["excel"] = excel_path

    if config.get("write_geopackage", False):
        gpkg_path = outdir / f"{name}.gpkg"
        sectors_with_counts = sectors.merge(result, on=sector_id_field, how="left")
        sectors_with_counts.to_file(gpkg_path, layer="sector_counts", driver="GPKG")
        outputs["geopackage"] = gpkg_path

    return outputs
