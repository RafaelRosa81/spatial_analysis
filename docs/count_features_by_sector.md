# Count features by sector

This workflow creates a table that counts each supplied point or polygon layer in every polygon of a sectors layer.

## Input structure

- **Sectors**: a polygon layer with one unique identifier field, for example `sector` with values 1 through 8.
- **Layers**: a YAML list of point and/or polygon datasets. Each entry becomes one output column.

Each input layer may be a Shapefile, GeoPackage, GeoJSON, or another vector format supported by GeoPandas/Fiona.

## Configuration

Copy and adapt `config/count_features_by_sector_example.yml`.

```yaml
pipeline: count_features_by_sector

count_features_by_sector:
  sectors: path/to/sectores.shp
  sector_id_field: sector
  outdir: outputs/sector_feature_counts

  layers:
    - path: path/to/arboles.shp
      name: arboles
    - path: path/to/canteros.shp
      name: canteros
```

The `name` value is optional. When omitted, the output column uses the input filename without its extension.

## Spatial rules

By default:

- Point layers use `within`. A point must lie inside a sector; a point exactly on the boundary is not counted.
- Polygon layers use `intersects`. A polygon touching or crossing a sector is counted for that sector. A polygon that spans two sectors is counted once in each one.

To count only objects completely contained within a sector, set:

```yaml
polygon_predicate: within
```

To include points located exactly on a sector boundary, set:

```yaml
point_predicate: covered_by
```

An individual layer may override the default behavior using `predicate`:

```yaml
layers:
  - path: path/to/canteros.shp
    name: canteros
    predicate: within
```

## Outputs

The workflow writes the following files to `outdir`:

- `<name>.csv`: table suitable for import into QGIS or Excel.
- `<name>.xlsx`: Excel workbook with `Counts_by_sector`, `Input_layers`, and `Configuration` sheets.
- `<name>.gpkg`: optional copy of the sectors layer with all count fields, enabled with `write_geopackage: true`.
