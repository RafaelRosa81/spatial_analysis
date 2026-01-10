# Configuration Reference

The entrypoint `python -m scripts.run_from_config --config <path>` accepts YAML
configs. The root config may contain either a `pipeline` key or default to the
`raster_diff` pipeline when omitted.

## Config conventions

- Example configs (tracked): `config/*_example.yml` with placeholder paths.
- Local/project configs (ignored): `config/*_local.yml` for real datasets.

## Root keys (all pipelines)

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `pipeline` | No | `raster_diff` | Supported values: `raster_diff`, `polygon_mosaic` |
| `name` | Yes | — | Output name prefix |
| `outdir` | Yes | — | Output directory |
| `excel` | No | `true` | Whether to generate Excel reports |
| `resampling` | No | `bilinear` | Used by `raster_diff` or forwarded into `polygon_mosaic.alignment.resampling` |

## Raster diff pipeline (`pipeline: raster_diff`)

### Required keys

- `raster1` (path): reference raster (alignment grid)
- `raster2` (path): raster to compare
- `outdir` (path)
- `name` (string)

### Optional keys and defaults

| Key | Default | Validation |
| --- | --- | --- |
| `resampling` | `bilinear` | one of `nearest`, `bilinear`, `cubic` |
| `excel` | `true` | boolean |
| `thresholds` | `[0.1, 0.25, 0.5, 1.0]` | non-empty list of numbers |
| `bins` | `60` | positive integer |
| `qgis_assets` | `true` | boolean |
| `vector_threshold` | `0.5` | number > 0 or `null` to disable |
| `signed_vector_threshold` | `null` | number > 0 or `null` to disable |

### Raster diff validation rules

- `raster1` and `raster2` must exist on disk.
- `bins` must be a positive integer.
- Threshold lists must contain numeric values.
- `vector_threshold` and `signed_vector_threshold` must be > 0 when provided.

### Raster diff config example (minimal)

```yaml
raster1: "path/to/raster1.tif"
raster2: "path/to/raster2.tif"
outdir: "outputs"
name: "demo_minimal"
```

### Raster diff config example (full)

```yaml
pipeline: "raster_diff"
name: "demo_full"
outdir: "outputs"
excel: true
resampling: "bilinear"
raster_diff:
  raster1: "path/to/raster1.tif"
  raster2: "path/to/raster2.tif"
  thresholds: [0.1, 0.25, 0.5, 1.0]
  bins: 80
  qgis_assets: true
  vector_threshold: 0.5
  signed_vector_threshold: 0.5
```

## Polygon mosaic pipeline (`pipeline: polygon_mosaic`)

### Required keys

- `raster1` (path)
- `raster2` (path)
- `polygon` (path)
- `outdir` (path)
- `name` (string)

### Optional keys and defaults

The pipeline uses a nested configuration. Defaults below mirror
`raster_compare/polygon_mosaic.py`.

```yaml
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

### Polygon mosaic validation rules

- `polygon_mosaic.polygon` is required and must point to a polygon file.
- `alignment.resampling` must be a valid rasterio resampling name.
- `raster1`, `raster2`, and `polygon` must exist on disk.

### Polygon mosaic config example

See `config/polygon_mosaic_example.yml`.
