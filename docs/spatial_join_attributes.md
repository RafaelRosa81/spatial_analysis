# Spatial join attributes

## Objective

`spatial_join_attributes` determines which input features belong to each polygon zone and preserves all attributes from the input layer.

The input layer may contain points, multipoints, polygons, or multipolygons. The zones layer must contain polygons or multipolygons.

This workflow is different from `count_features_by_sector`:

- `count_features_by_sector` produces one count column per input layer.
- `spatial_join_attributes` produces one row for every spatial relationship and retains the complete input attribute table.

## Typical uses

- Assign trees, emitters, valves, inspection points, or other surveyed points to sectors.
- Assign planting polygons, irrigation areas, or infrastructure footprints to management zones.
- Identify features that do not belong to any zone.
- Detect features that match more than one polygon.

## Configuration

Start from `config/spatial_join_attributes_example.yml`.

```yaml
pipeline: spatial_join_attributes

spatial_join_attributes:
  name: features_by_zone
  outdir: outputs/features_by_zone

  zones:
    path: path/to/sectors.shp
    id_field: sector_id
    fields:
      - sector_id
      - sector_name

  input:
    path: path/to/points_or_polygons.shp

  predicate: covered_by
  join_type: left
  zone_field_prefix: zone_

  write_csv: true
  write_excel: true
  write_geopackage: true
  write_unmatched: true
```

For GeoPackage inputs, add the optional `layer` key:

```yaml
zones:
  path: path/to/project.gpkg
  layer: sectors
  id_field: sector_id

input:
  path: path/to/project.gpkg
  layer: trees
```

## Spatial predicates

### Point inputs

- `within`: points must be strictly inside a polygon. Points exactly on the boundary are unmatched.
- `covered_by`: includes points inside polygons and points on polygon boundaries.
- `intersects`: also includes interior and boundary contacts.

For most point-to-zone assignments, `covered_by` is recommended.

### Polygon inputs

- `within`: the input polygon must be completely inside the zone.
- `covered_by`: the polygon may share its boundary with the zone but must otherwise be contained.
- `intersects`: any overlap or boundary contact creates a match.

With `intersects`, one input polygon may generate multiple output rows when it intersects multiple zones.

## Join type

### `left`

Keeps every input feature. Features without a spatial match remain in the output with empty zone fields.

### `inner`

Keeps only matched relationships.

Use `left` when auditing coverage or when unmatched features must be reviewed.

## Attributes and field names

All original input attributes are preserved.

Zone attributes are copied with the configured prefix. For example:

```yaml
zone_field_prefix: zone_
```

converts zone fields such as `sector_id` and `name` into:

```text
zone_sector_id
zone_name
```

This prevents collisions when both datasets contain fields with the same name.

The workflow also creates `source_feature_id`, based on the original GeoDataFrame index, to preserve traceability between input features and output rows.

## CRS handling

Both layers must have a defined CRS. When their CRS values differ, the input layer is automatically reprojected to the zones CRS before the spatial join.

## Outputs

Depending on the enabled options, the output directory contains:

```text
features_by_zone.csv
features_by_zone.xlsx
features_by_zone.gpkg
features_by_zone_unmatched.csv
```

### CSV

Contains all input attributes and copied zone attributes. Geometry is omitted by default. Set:

```yaml
include_geometry_wkt_in_tables: true
```

to add a `geometry_wkt` text column.

### Excel

The workbook includes:

- `Spatial_join`: complete joined table.
- `Summary`: number of matched rows per zone.
- `Unmatched`: features without a zone.
- `Zone_fields`: original and prefixed zone field names.
- `Configuration`: resolved execution settings.

### GeoPackage

The `spatial_join` layer preserves the original input geometry and all joined attributes. When unmatched features exist and `write_unmatched` is enabled, an additional `unmatched` layer is written to the same GeoPackage.

## Execution

```bash
python -m scripts.run_spatial_join_attributes \
  --config config/spatial_join_attributes_example.yml
```

On Windows CMD, the same command can be written on one line:

```cmd
python -m scripts.run_spatial_join_attributes --config config/spatial_join_attributes_example.yml
```

## Interpreting duplicated rows

Duplicated `source_feature_id` values are not automatically errors. They indicate that one input feature matched more than one zone. This is expected when:

- zones overlap;
- a polygon input intersects several zones;
- `intersects` is used for boundary contacts.

Review these cases using the GeoPackage output in QGIS or filter duplicated `source_feature_id` values in Excel.

## Limitations

- Invalid geometries are not automatically repaired.
- Shapefiles must include all required sidecar files, especially `.shx` and `.dbf`.
- A polygon matched with `intersects` is assigned to every intersected zone; the workflow does not currently select the zone with the greatest overlap area.
- Spatial relationships are evaluated in two dimensions.
