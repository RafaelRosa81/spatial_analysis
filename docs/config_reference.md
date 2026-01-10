# Configuration Reference

The entrypoint `python -m scripts.run_from_config --config <path>` accepts YAML
configs with a single top-level `pipeline` selector and one or more pipeline
blocks.

This file is the canonical reference for the **nested YAML style** shown below.
Legacy (flat) keys are still supported for backwards compatibility, but new
configs should follow the nested structure.

```yaml
# Choose which workflow to run:
#   - "raster_diff"
#   - "polygon_mosaic"
#   - "sample_points_from_raster_value_range"
pipeline: "raster_diff"

raster_diff:
  name: "raster_diff"
  outdir: "outputs/raster_diff"
  resampling: "bilinear"
  excel: true
  raster1: "D:/path/to/raster1.tif"
  raster2: "D:/path/to/raster2.tif"
  thresholds: [0.10, 0.25, 0.50, 1.00]
  bins: 60
  qgis_assets: true
  vector_threshold: 0.5
  signed_vector_threshold: 0.2

polygon_mosaic:
  name: "mosaic"
  outdir: "outputs/mosaic"
  excel: true
  raster1: "D:/path/to/raster1.tif"
  raster2: "D:/path/to/raster2.tif"
  polygon: "H:/path/to/polygon.shp"
  outputs:
    new_raster: "mosaic_dem.tif"
    excel_report: "mosaic_report.xlsx"
    save_intermediates: true
  vertical_adjustment:
    enabled: true
    mad_threshold: 0.10
    min_overlap_pixels: 50000
    exclude_polygon_buffer_px: 5
  alignment:
    resampling: "bilinear"
  border_blending:
    enabled: true
    blend_width_px: 5

sample_points_from_raster_value_range:
  name: "sample_points_range"
  outdir: "outputs/sample_points_range"
  raster: "D:/path/to/your_raster.tif"
  value_min: 34.8
  value_max: 35.0
  sampling:
    method: "random"        # random | regular
    n_points: 2000          # used if random
    seed: 42                # used if random
    spacing: 5              # used if regular (map units)
  mask_polygon: null        # optional polygon path to constrain allowed area
  nodata_is_invalid: true   # ignore nodata pixels when filtering
  save_geopackage: true
  save_csv: true
  qgis_assets: true
```

## Config conventions

- Example configs (tracked): `config/*_example.yml` with placeholder paths.
- Local/project configs (ignored): `config/*_local.yml` for real datasets.

## Root keys (all pipelines)

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `pipeline` | Yes | — | Required in canonical configs. Supported values: `raster_diff`, `polygon_mosaic`, `sample_points_from_raster_value_range` (alias: `sample_points`). |

> Note: If `pipeline` is omitted, the runner defaults to `raster_diff` for
> backwards compatibility.

## Raster diff pipeline (`pipeline: raster_diff`)

### Required keys (under `raster_diff`)

- `name` (string): output name prefix.
- `outdir` (path): output directory.
- `raster1` (path): reference raster (alignment grid).
- `raster2` (path): raster to compare.

### Optional keys and defaults

| Key | Type | Default | Validation |
| --- | --- | --- | --- |
| `resampling` | string | `bilinear` | one of `nearest`, `bilinear`, `cubic` |
| `excel` | boolean | `true` | boolean |
| `thresholds` | list[number] | `[0.1, 0.25, 0.5, 1.0]` | non-empty list of numbers |
| `bins` | integer | `60` | positive integer |
| `qgis_assets` | boolean | `true` | boolean |
| `vector_threshold` | number/null | `0.5` | number > 0 or `null` to disable |
| `signed_vector_threshold` | number/null | `null` | number > 0 or `null` to disable |

### Raster diff validation rules

- `raster1` and `raster2` must exist on disk.
- `bins` must be a positive integer.
- `thresholds` must be a non-empty list of numeric values.
- `vector_threshold` and `signed_vector_threshold` must be > 0 when provided.

### Raster diff config example (minimal)

```yaml
pipeline: "raster_diff"

raster_diff:
  name: "demo_minimal"
  outdir: "outputs"
  raster1: "path/to/raster1.tif"
  raster2: "path/to/raster2.tif"
```

### Raster diff config example (full)

```yaml
pipeline: "raster_diff"

raster_diff:
  name: "demo_full"
  outdir: "outputs"
  excel: true
  resampling: "bilinear"
  raster1: "path/to/raster1.tif"
  raster2: "path/to/raster2.tif"
  thresholds: [0.1, 0.25, 0.5, 1.0]
  bins: 80
  qgis_assets: true
  vector_threshold: 0.5
  signed_vector_threshold: 0.5
```

## Polygon mosaic pipeline (`pipeline: polygon_mosaic`)

### Required keys (under `polygon_mosaic`)

- `name` (string): output name prefix.
- `outdir` (path): output directory.
- `raster1` (path)
- `raster2` (path)
- `polygon` (path): polygon file used to define the mosaic region.

### Optional keys and defaults

Defaults below mirror `raster_compare/polygon_mosaic.py`.

```yaml
excel: true
outputs:
  new_raster: "new_raster.tif"
  excel_report: "polygon_mosaic_report.xlsx"
  save_intermediates: true
alignment:
  resampling: "bilinear"
vertical_adjustment:
  enabled: true
  method: "constant_offset"
  robust_stat: "median"
  mad_threshold: 0.10
  min_overlap_pixels: 50000
  exclude_polygon_buffer_px: 5
border_blending:
  enabled: true
  blend_width_px: 5
  weight_curve: "linear"
nodata:
  use_raster1_nodata: true
```

> Note: A legacy top-level `resampling` key is supported and is forwarded to
> `alignment.resampling` when present.

### Polygon mosaic validation rules

- `polygon_mosaic.polygon` is required and must point to a polygon file.
- `alignment.resampling` must be a valid rasterio resampling name.
- `raster1`, `raster2`, and `polygon` must exist on disk.

### Polygon mosaic config example

See `config/polygon_mosaic_example.yml`.

## Sample points from raster value range (`pipeline: sample_points_from_raster_value_range`)

### Required keys (under `sample_points_from_raster_value_range`)

- `raster` (path): raster used for value filtering and point sampling.
- `value_min` (number): minimum value (inclusive).
- `value_max` (number): maximum value (inclusive).

### Optional keys and defaults

| Key | Type | Default | Validation/Notes |
| --- | --- | --- | --- |
| `name` | string | `sample_points` | Used as output filename prefix. |
| `outdir` | path | `outputs/<name>` | Created if missing. |
| `sampling.method` | string | `random` | Must be `random` or `regular`. |
| `sampling.n_points` | integer | `1000` | Used only when `method: random`. |
| `sampling.seed` | integer/null | `null` | Used only when `method: random`. |
| `sampling.spacing` | number/null | `null` | Required when `method: regular`; spacing in **map units**. |
| `mask_polygon` | path/null | `null` | Accepted in config but not applied by the pipeline yet (see note below). |
| `nodata_is_invalid` | boolean | `true` | When `true`, nodata pixels are excluded from the value range filter. |
| `save_geopackage` | boolean | `true` | Writes `{name}.gpkg` when `true`. |
| `save_csv` | boolean | `true` | Writes `{name}.csv` when `true`. |
| `qgis_assets` | boolean | `true` | Currently unused for this pipeline. |

### Sample points semantics

- **Value range:** pixels are kept when `value_min <= value <= value_max`
  (inclusive on both ends).
- **Random sampling:** draws up to `n_points` from qualifying pixels using
  `sampling.seed` for reproducibility.
- **Regular sampling:** converts `sampling.spacing` (map units) to an approximate
  pixel step. If your raster CRS is geographic (degrees), spacing is interpreted
  in degrees; use a projected CRS for meter-based spacing.
- **Mask polygon:** the YAML key is accepted but not currently applied in the
  pipeline implementation. If polygon masking is required, use the standalone
  CLI in `scripts/sample_points_from_raster.py`.
- **NoData handling:** when `nodata_is_invalid` is `true`, pixels with the
  dataset nodata value are excluded before filtering by value range.

### Sample points outputs

- CSV: `<outdir>/<name>.csv`
- GeoPackage: `<outdir>/<name>.gpkg` (layer name: `sample_points`)

Both outputs are optional based on `save_csv` and `save_geopackage`.

### Sample points validation rules

- `value_max` must be greater than or equal to `value_min`.
- `sampling.method` must be `random` or `regular`.
- `sampling.spacing` must be provided and > 0 when `method: regular`.
- `raster` must exist on disk.

### Sample points config example (minimal)

```yaml
pipeline: "sample_points_from_raster_value_range"

sample_points_from_raster_value_range:
  raster: "path/to/raster.tif"
  value_min: 34.8
  value_max: 35.0
```

### Sample points config example (full)

```yaml
pipeline: "sample_points_from_raster_value_range"

sample_points_from_raster_value_range:
  name: "sample_points_range"
  outdir: "outputs/sample_points_range"
  raster: "path/to/raster.tif"
  value_min: 34.8
  value_max: 35.0
  sampling:
    method: "regular"
    spacing: 5
  mask_polygon: null
  nodata_is_invalid: true
  save_geopackage: true
  save_csv: true
  qgis_assets: true
```
