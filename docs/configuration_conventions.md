# Configuration conventions

This repository uses YAML configuration files to make geospatial workflows reproducible.

## Windows paths

Use one consistent format throughout the repository:

- keep the Windows backslash separator (`\`);
- wrap the complete path in single quotes;
- do not duplicate backslashes.

Correct:

```yaml
path: 'D:\data\project\sectors.shp'
outdir: 'D:\repos\spatial_analysis\outputs\run_01'
```

Avoid double-quoted Windows paths:

```yaml
path: "D:\data\project\sectors.shp"
```

YAML interprets backslashes inside double quotes as escape sequences. A copied Windows path may therefore fail with an error such as `found unknown escape character`.

Single quotes preserve the path exactly as copied from Windows Explorer, QGIS, or another application.

## Relative paths

Relative paths are allowed when they improve portability:

```yaml
outdir: 'outputs\features_by_zone'
```

They are resolved from the directory where the command is executed. Run commands from the repository root unless a pipeline documents another behavior.

## GeoPackage layers

A GeoPackage can contain several layers. Always set `layer` when the file contains more than one layer:

```yaml
input:
  path: 'D:\data\network.gpkg'
  layer: junctions
```

For a Shapefile, `layer: null` may be used or the key may be omitted.

## Field names

Field names in the YAML must exactly match the source dataset. Before running a pipeline, check the QGIS attribute table or inspect the layer schema.

Example:

```yaml
zones:
  id_field: num_sector
  fields:
    - id
    - num_sector
    - caudal_l_s
```

## Booleans and null values

Use YAML-native values without quotes:

```yaml
write_excel: true
write_unmatched: false
layer: null
```

## Recommended workflow

1. Copy the example configuration for the selected pipeline.
2. Rename it for the project or run.
3. Paste Windows paths using backslashes and single quotes.
4. Confirm GeoPackage layer names and source field names.
5. Run the command from the repository root.
6. Keep the configuration with the generated outputs for traceability.
