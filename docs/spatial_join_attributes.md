# Spatial join attributes

## Objective

`spatial_join_attributes` determines which input features belong to each polygon zone while preserving all attributes from the input layer.

The input layer may contain points, multipoints, polygons, or multipolygons. The zones layer must contain polygons or multipolygons.

This workflow is different from `count_features_by_sector`:

- `count_features_by_sector` aggregates and produces counts per zone.
- `spatial_join_attributes` produces one row for every spatial relationship and retains the complete input attribute table.

## Typical uses

- Assign junctions, emitters, valves, trees, inspection points, or other surveyed points to sectors.
- Assign planting polygons, irrigation areas, or infrastructure footprints to management zones.
- Identify features that do not belong to any zone.
- Detect features that match more than one polygon.
- Prepare auditable attribute tables for QGIS, EPANET, or engineering reports.

## Configuration

Start from `config/spatial_join_attributes_example.yml`.

```yaml
pipeline: spatial_join_attributes

spatial_join_attributes:
  name: features_by_zone
  outdir: 'D:\repos\spatial_analysis\outputs\features_by_zone'

  zones:
    path: 'D:\data\sectors.shp'
    layer: null
    id_field: num_sector
    fields:
      - id
      - num_sector
      - caudal_l_s
      - rieg_h_dia

  input:
    path: 'D:\data\network.gpkg'
    layer: junctions

  predicate: covered_by
  join_type: left
  zone_field_prefix: zone_

  write_csv: true
  write_excel: true
  write_geopackage: true
  write_unmatched: true
  include_geometry_wkt_in_tables: false
```

### Windows path convention

All Windows paths in this repository use backslashes and single quotes:

```yaml
path: 'D:\data\project\layer.shp'
```

Do not duplicate the backslashes. Avoid double quotes because YAML interprets backslashes as escape sequences. See `docs/configuration_conventions.md`.

### GeoPackage inputs

A GeoPackage can contain several layers. Always set `layer` when more than one layer exists:

```yaml
zones:
  path: 'D:\data\project.gpkg'
  layer: sectors
  id_field: sector_id

input:
  path: 'D:\data\project.gpkg'
  layer: junctions
```

If a multi-layer GeoPackage is read without `layer`, GeoPandas may select the first layer and issue a warning. Explicit layer names make the run reproducible.

### Zone identifier and copied fields

`id_field` must exactly match a field in the zones layer.

```yaml
id_field: num_sector
```

The optional `fields` list controls which zone attributes are copied. When omitted, every non-geometry zone field is copied.

Field names are validated before the join. If a field is missing, the error lists the available fields.

## Spatial predicates

### Point inputs

- `within`: points must be strictly inside a polygon. Points exactly on the boundary are unmatched.
- `covered_by`: includes points inside polygons and points on polygon boundaries.
- `intersects`: includes interior and boundary contacts.

For most point-to-zone assignments, `covered_by` is recommended.

### Polygon inputs

- `within`: the input polygon must be completely inside the zone.
- `covered_by`: the polygon may share its boundary with the zone but must otherwise be contained.
- `intersects`: any overlap or boundary contact creates a match.

With `intersects`, one input polygon may generate multiple output rows when it intersects multiple zones.

## Join type

### `left`

Keeps every input feature. Features without a spatial match remain in the output with empty zone fields.

Use `left` for auditing and when unmatched features must be reviewed.

### `inner`

Keeps only matched relationships.

Use `inner` only when unmatched features are intentionally excluded.

## Attributes and field names

All original input attributes are preserved.

Zone attributes are copied with the configured prefix. For example:

```yaml
zone_field_prefix: zone_
```

converts zone fields such as `num_sector` and `caudal_l_s` into:

```text
zone_num_sector
zone_caudal_l_s
```

This prevents collisions when both datasets contain fields with the same name.

The workflow also creates `source_feature_id`, based on the original GeoDataFrame index, to preserve traceability between input features and output rows.

## CRS handling

Both layers must have a defined CRS. When their CRS values differ, the input layer is automatically reprojected to the zones CRS before the spatial join.

The exported GeoPackage uses the zones CRS.

## Outputs

Depending on the enabled options, the output directory contains:

```text
features_by_zone.csv
features_by_zone.xlsx
features_by_zone.gpkg
features_by_zone_unmatched.csv
```

### CSV

Contains all input attributes and copied zone attributes. Geometry is omitted by default.

Set:

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

The `spatial_join` layer preserves the original input geometry and all joined attributes.

When unmatched features exist and `write_unmatched` is enabled, an additional `unmatched` layer is written to the same GeoPackage.

This is the recommended output for visual validation in QGIS.

## Execution

From the repository root:

```cmd
python -m scripts.run_spatial_join_attributes --config config\spatial_join_attributes_example.yml
```

The command prints the resolved configuration and the generated output paths.

## Validation checklist

After each run, verify:

1. The expected input layer was read from the source file.
2. The number of output rows is compatible with the input feature count.
3. `source_feature_id` is unique unless multiple spatial matches are expected.
4. The `Summary` sheet contains the expected sectors.
5. The `Unmatched` layer contains only features that should remain outside zones.
6. Zone attributes are populated in the original zones dataset.
7. The GeoPackage result aligns correctly with the zones layer in QGIS.

## Interpreting unmatched features

Unmatched features are not automatically errors. They may represent:

- main lines or common infrastructure outside irrigation sectors;
- pump rooms, reservoirs, tanks, or supply nodes;
- connections between sectors;
- points located in gaps between zone polygons;
- geometry or CRS problems.

Review the `unmatched` GeoPackage layer together with the original zones in QGIS.

## Interpreting duplicated rows

Duplicated `source_feature_id` values are not automatically errors. They indicate that one input feature matched more than one zone. This is expected when:

- zones overlap;
- a polygon input intersects several zones;
- `intersects` is used for boundary contacts.

Review these cases in QGIS or filter duplicated `source_feature_id` values in Excel.

## Limitations

- Invalid geometries are not automatically repaired.
- Shapefiles must include all required sidecar files, especially `.shx` and `.dbf`.
- A polygon matched with `intersects` is assigned to every intersected zone; the workflow does not currently select the zone with the greatest overlap area.
- Spatial relationships are evaluated in two dimensions.
- Empty source attributes remain empty in the output even when the spatial relationship is valid.
