# Examples

## Minimal raster diff

Config: `config/minimal_raster_diff_example.yml`

```bash
python -m scripts.run_from_config --config config/minimal_raster_diff_example.yml
```

## Full-feature raster diff

Config: `config/full_raster_diff_example.yml`

```bash
python -m scripts.run_from_config --config config/full_raster_diff_example.yml
```

## Polygon mosaic

Config: `config/polygon_mosaic_example.yml`

```bash
python -m scripts.run_from_config --config config/polygon_mosaic_example.yml
```

## Sample points from raster value range (minimal)

```yaml
pipeline: "sample_points_from_raster_value_range"

sample_points_from_raster_value_range:
  raster: "path/to/raster.tif"
  value_min: 34.8
  value_max: 35.0
```

## Sample points from raster value range (full)

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
  mask_polygon: "path/to/mask_polygon.shp"
  nodata_is_invalid: true
  save_geopackage: true
  save_csv: true
  qgis_assets: true
```

> Note: All example configs use placeholder paths. Replace them with real data
> before running.
