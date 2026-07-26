from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


ALLOWED_PREDICATES = {
    "intersects",
    "within",
    "contains",
    "overlaps",
    "crosses",
    "touches",
    "covers",
    "covered_by",
}
ALLOWED_JOIN_TYPES = {"left", "inner"}


def _resolve_path(value: str | Path, label: str, must_exist: bool = True) -> Path:
    path = Path(value).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _read_vector(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    kwargs: dict[str, Any] = {}
    if layer:
        kwargs["layer"] = layer
    gdf = gpd.read_file(path, **kwargs)
    if gdf.crs is None:
        raise ValueError(f"Layer has no CRS defined: {path}")
    if gdf.empty:
        raise ValueError(f"Layer contains no features: {path}")
    return gdf


def resolve_spatial_join_attributes_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    section = raw_config.get("spatial_join_attributes", raw_config)
    if not isinstance(section, Mapping):
        raise ValueError("spatial_join_attributes must be a YAML mapping.")

    zones = section.get("zones")
    input_spec = section.get("input")
    if not isinstance(zones, Mapping):
        raise ValueError("zones must be a YAML mapping with at least 'path'.")
    if not isinstance(input_spec, Mapping):
        raise ValueError("input must be a YAML mapping with at least 'path'.")

    missing: list[str] = []
    if not zones.get("path"):
        missing.append("zones.path")
    if not input_spec.get("path"):
        missing.append("input.path")
    if not zones.get("id_field"):
        missing.append("zones.id_field")
    if not section.get("outdir"):
        missing.append("outdir")
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")

    predicate = str(section.get("predicate", "within")).lower()
    if predicate not in ALLOWED_PREDICATES:
        raise ValueError(f"predicate must be one of: {', '.join(sorted(ALLOWED_PREDICATES))}")

    join_type = str(section.get("join_type", "left")).lower()
    if join_type not in ALLOWED_JOIN_TYPES:
        raise ValueError("join_type must be 'left' or 'inner'.")

    zone_fields = zones.get("fields")
    if zone_fields is not None:
        if not isinstance(zone_fields, list) or not all(isinstance(v, str) for v in zone_fields):
            raise ValueError("zones.fields must be a YAML list of field names.")

    return {
        "pipeline": "spatial_join_attributes",
        "name": str(section.get("name") or "spatial_join_attributes"),
        "zones": {
            "path": _resolve_path(zones["path"], "Zones layer"),
            "layer": zones.get("layer"),
            "id_field": str(zones["id_field"]),
            "fields": zone_fields,
        },
        "input": {
            "path": _resolve_path(input_spec["path"], "Input layer"),
            "layer": input_spec.get("layer"),
        },
        "outdir": _resolve_path(section["outdir"], "Output directory", must_exist=False),
        "predicate": predicate,
        "join_type": join_type,
        "zone_field_prefix": str(section.get("zone_field_prefix", "zone_")),
        "write_csv": bool(section.get("write_csv", True)),
        "write_excel": bool(section.get("write_excel", True)),
        "write_geopackage": bool(section.get("write_geopackage", True)),
        "write_unmatched": bool(section.get("write_unmatched", True)),
        "include_geometry_wkt_in_tables": bool(section.get("include_geometry_wkt_in_tables", False)),
    }


def _prepare_zones(
    zones: gpd.GeoDataFrame,
    id_field: str,
    selected_fields: list[str] | None,
    prefix: str,
) -> tuple[gpd.GeoDataFrame, str, dict[str, str]]:
    if id_field not in zones.columns:
        raise ValueError(
            f"Zone identifier field '{id_field}' does not exist. "
            f"Available fields: {', '.join(map(str, zones.columns))}"
        )

    if selected_fields is None:
        fields = [column for column in zones.columns if column != zones.geometry.name]
    else:
        missing = [field for field in selected_fields if field not in zones.columns]
        if missing:
            raise ValueError(f"Zone fields not found: {', '.join(missing)}")
        fields = list(selected_fields)
        if id_field not in fields:
            fields.insert(0, id_field)

    rename_map = {field: f"{prefix}{field}" for field in fields}
    prepared = zones[fields + [zones.geometry.name]].rename(columns=rename_map).copy()
    return prepared, rename_map[id_field], rename_map


def _table_without_geometry(
    joined: gpd.GeoDataFrame,
    include_geometry_wkt: bool,
) -> pd.DataFrame:
    table = pd.DataFrame(joined.drop(columns=joined.geometry.name))
    if include_geometry_wkt:
        table["geometry_wkt"] = joined.geometry.to_wkt()
    return table


def run_spatial_join_attributes(config: Mapping[str, Any]) -> dict[str, Path | int]:
    zones_spec = config["zones"]
    input_spec = config["input"]

    zones = _read_vector(zones_spec["path"], zones_spec.get("layer"))
    features = _read_vector(input_spec["path"], input_spec.get("layer"))

    zones = zones.loc[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    features = features.loc[features.geometry.notna() & ~features.geometry.is_empty].copy()
    if zones.empty:
        raise ValueError("Zones layer has no valid geometries.")
    if features.empty:
        raise ValueError("Input layer has no valid geometries.")

    zone_types = set(zones.geometry.geom_type.unique())
    if not zone_types.issubset({"Polygon", "MultiPolygon"}):
        raise ValueError("Zones layer must contain only Polygon or MultiPolygon geometries.")

    if features.crs != zones.crs:
        features = features.to_crs(zones.crs)

    input_geometry_name = features.geometry.name
    features = features.copy()
    features["source_feature_id"] = features.index.astype(str)

    prepared_zones, prefixed_zone_id, zone_rename_map = _prepare_zones(
        zones=zones,
        id_field=str(zones_spec["id_field"]),
        selected_fields=zones_spec.get("fields"),
        prefix=str(config["zone_field_prefix"]),
    )

    joined = gpd.sjoin(
        features,
        prepared_zones,
        how=str(config["join_type"]),
        predicate=str(config["predicate"]),
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.set_geometry(input_geometry_name)

    matched_mask = joined[prefixed_zone_id].notna()
    unmatched = joined.loc[~matched_mask].copy()
    matched = joined.loc[matched_mask].copy()

    summary = (
        matched.groupby(prefixed_zone_id, dropna=False)
        .size()
        .rename("matched_features")
        .reset_index()
        .sort_values(prefixed_zone_id)
    )

    outdir = Path(config["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    name = str(config["name"])
    outputs: dict[str, Path | int] = {
        "input_features": int(len(features)),
        "joined_rows": int(len(joined)),
        "matched_rows": int(len(matched)),
        "unmatched_rows": int(len(unmatched)),
    }

    table = _table_without_geometry(joined, bool(config["include_geometry_wkt_in_tables"]))
    unmatched_table = _table_without_geometry(unmatched, bool(config["include_geometry_wkt_in_tables"]))

    if config.get("write_csv", True):
        csv_path = outdir / f"{name}.csv"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        outputs["csv"] = csv_path
        if config.get("write_unmatched", True) and not unmatched.empty:
            unmatched_csv = outdir / f"{name}_unmatched.csv"
            unmatched_table.to_csv(unmatched_csv, index=False, encoding="utf-8-sig")
            outputs["unmatched_csv"] = unmatched_csv

    if config.get("write_excel", True):
        excel_path = outdir / f"{name}.xlsx"
        config_rows = [{"parameter": key, "value": str(value)} for key, value in config.items()]
        field_rows = [
            {"original_zone_field": original, "output_zone_field": renamed}
            for original, renamed in zone_rename_map.items()
        ]
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name="Spatial_join", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            unmatched_table.to_excel(writer, sheet_name="Unmatched", index=False)
            pd.DataFrame(field_rows).to_excel(writer, sheet_name="Zone_fields", index=False)
            pd.DataFrame(config_rows).to_excel(writer, sheet_name="Configuration", index=False)
        outputs["excel"] = excel_path

    if config.get("write_geopackage", True):
        gpkg_path = outdir / f"{name}.gpkg"
        joined.to_file(gpkg_path, layer="spatial_join", driver="GPKG")
        outputs["geopackage"] = gpkg_path
        if config.get("write_unmatched", True) and not unmatched.empty:
            unmatched.to_file(gpkg_path, layer="unmatched", driver="GPKG")

    return outputs
