# Quick Start Guide

This is the recommended entry point for a new user of `spatial_analysis`.

The repository contains reproducible geospatial workflows driven by YAML configuration files. Most pipelines are launched through a common runner; two vector-zone workflows currently use dedicated runners.

## 1. Install and prepare the repository

### Clone for the first time

```cmd
git clone https://github.com/RafaelRosa81/spatial_analysis.git
cd spatial_analysis
```

### Create the Conda environment

```cmd
conda env create -f environment.yml
conda activate spatial_analysis
```

When the environment already exists and `environment.yml` changed:

```cmd
conda activate spatial_analysis
conda env update -f environment.yml --prune
```

### Update an existing local clone

From the repository root:

```cmd
git status
git checkout main
git pull --ff-only origin main
```

Do not run `git pull` with uncommitted changes that you have not reviewed.

## 2. Windows paths in YAML

Use Windows paths exactly as copied, with backslashes and single quotes:

```yaml
path: 'D:\datos\archivo.shp'
```

Do not wrap Windows paths containing backslashes in double quotes. YAML may interpret sequences such as `\t`, `\n`, or `\U` as escapes.

Relative repository paths can also be used:

```yaml
outdir: 'outputs\my_run'
```

## 3. Available pipelines

| Pipeline | Purpose | Runner |
| --- | --- | --- |
| `raster_diff` | Compare two rasters and calculate signed and absolute differences. | `scripts.run_from_config` |
| `polygon_mosaic` | Combine rasters inside/outside polygons, with optional adjustment and blending. | `scripts.run_from_config` |
| `sample_points_from_raster_value_range` | Generate points from raster cells within a value interval. | `scripts.run_from_config` |
| `landxml_tin_to_mesh` | Convert a LandXML TIN into mesh products and an optional DEM. | `scripts.run_from_config` |
| `raster_adapt_polygon` | Reconstruct terrain inside polygons from surrounding terrain. | `scripts.run_from_config` |
| `count_features_by_sector` | Count one or more point/polygon layers inside sector polygons. | `scripts.run_count_features_by_sector` |
| `spatial_join_attributes` | Assign input features to polygon zones while preserving attributes. | `scripts.run_spatial_join_attributes` |

## 4. Common runner

The five pipelines integrated into the common runner use:

```cmd
python -m scripts.run_from_config --config config\your_config.yml
```

Show help:

```cmd
python -m scripts.run_from_config --help
```

The YAML must define one of these values:

```yaml
pipeline: 'raster_diff'
```

```yaml
pipeline: 'polygon_mosaic'
```

```yaml
pipeline: 'sample_points_from_raster_value_range'
```

```yaml
pipeline: 'landxml_tin_to_mesh'
```

```yaml
pipeline: 'raster_adapt_polygon'
```

## 5. Quick examples

### Raster difference

```yaml
pipeline: 'raster_diff'

raster_diff:
  name: 'demo_diff'
  outdir: 'outputs\demo_diff'
  raster1: 'D:\datos\dem_base.tif'
  raster2: 'D:\datos\dem_nuevo.tif'
  resampling: 'bilinear'
  excel: true
  thresholds: [0.10, 0.25, 0.50, 1.00]
```

Run:

```cmd
python -m scripts.run_from_config --config config\minimal_raster_diff_example.yml
```

### Polygon mosaic

```yaml
pipeline: 'polygon_mosaic'

polygon_mosaic:
  name: 'demo_mosaic'
  outdir: 'outputs\demo_mosaic'
  raster1: 'D:\datos\dem_interior.tif'
  raster2: 'D:\datos\dem_exterior.tif'
  polygon: 'D:\datos\area_seleccion.shp'
```

Run:

```cmd
python -m scripts.run_from_config --config config\polygon_mosaic_example.yml
```

### Sample points from a raster value range

```yaml
pipeline: 'sample_points_from_raster_value_range'

sample_points_from_raster_value_range:
  name: 'sample_points'
  outdir: 'outputs\sample_points'
  raster: 'D:\datos\dem.tif'
  value_min: 34.8
  value_max: 35.0
  sampling:
    method: 'random'
    n_points: 2000
    seed: 42
```

Run:

```cmd
python -m scripts.run_from_config --config config\sample_points_example.yml
```

### LandXML TIN to mesh

```yaml
pipeline: 'landxml_tin_to_mesh'

landxml_tin_to_mesh:
  name: 'landxml_demo'
  outdir: 'outputs\landxml_demo'
  input_xml: 'D:\datos\surface.xml'
  surface_name: 'TN'
  coordinate_order: 'yxz'
  crs: 'EPSG:32721'
  rasterize:
    enabled: true
    pixel_size: 0.50
    output_dem: 'tin_dem.tif'
```

Run:

```cmd
python -m scripts.run_from_config --config config\landxml_tin_to_mesh_example.yml
```

### Adapt raster inside a polygon

```yaml
pipeline: 'raster_adapt_polygon'

raster_adapt_polygon:
  name: 'adapted_dem'
  outdir: 'outputs\adapted_dem'
  raster: 'D:\datos\dem.tif'
  polygon: 'D:\datos\area_modificar.shp'
```

Run:

```cmd
python -m scripts.run_from_config --config config\raster_adapt_polygon_example.yml
```

## 6. Dedicated vector-zone runners

### Count features by sector

Start from the example configuration documented in `docs/count_features_by_sector.md`, then run:

```cmd
python -m scripts.run_count_features_by_sector --config config\count_features_by_sector_example.yml
```

This pipeline summarizes how many features from each configured input layer fall in each sector polygon.

### Spatial join attributes

Start from `config/spatial_join_attributes_example.yml`, then run:

```cmd
python -m scripts.run_spatial_join_attributes --config config\spatial_join_attributes_example.yml
```

This pipeline preserves all input attributes and appends selected zone attributes. It can generate CSV, Excel, GeoPackage, unmatched-feature outputs, and a summary by zone.

For a Shapefile, omit `layer` or use:

```yaml
layer: null
```

For a multi-layer GeoPackage, specify the internal layer explicitly:

```yaml
path: 'D:\datos\network.gpkg'
layer: 'junctions'
```

## 7. Recommended validation commands

Basic Python import check:

```cmd
python -c "import geopandas, rasterio, shapely, pandas, openpyxl, yaml; print('Imports OK')"
```

Pipeline-specific sanity checks:

```cmd
python -m scripts.landxml_tin_sanity
python scripts\polygon_mosaic_sanity.py
python -m scripts.raster_adapt_polygon_sanity
```

Run regression checks when changing code:

```cmd
python tests\run_regression.py
```

## 8. Git commands used most often

Review local state:

```cmd
git status
git branch --show-current
git log -1 --oneline
```

Update `main` safely:

```cmd
git checkout main
git pull --ff-only origin main
```

Start a new change:

```cmd
git checkout main
git pull --ff-only origin main
git checkout -b feature\short-description
```

Use forward slashes in Git branch names when typing the real command:

```cmd
git checkout -b feature/short-description
```

Review changes before committing:

```cmd
git status
git diff
git add <files>
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Clean obsolete remote-tracking references after branches are merged or deleted on GitHub:

```cmd
git fetch --prune
```

## 9. Common problems

### YAML error with a Windows path

Use:

```yaml
path: 'D:\datos\archivo.shp'
```

not double quotes around a path containing unescaped backslashes.

### `Layer ... could not be opened`

A `.shp` has one layer and normally should not define `layer`. A `.gpkg` may contain several layers and should specify the correct internal layer name.

### Missing CRS

Most spatial operations require a defined CRS. Assign the correct CRS to the source data rather than guessing it during processing.

### Output already exists

Some pipelines intentionally avoid silently overwriting results. Use a new output directory or remove/rename an old run after confirming it is no longer needed.

### Unmatched spatial features

Unmatched features are not necessarily processing errors. Review them in QGIS against the zone polygons and decide whether they are legitimately outside the zones or indicate gaps/boundary problems.

## 10. Documentation map

- `README.md`: repository overview and installation.
- `docs/documentation_index.md`: complete documentation index.
- `docs/config_reference.md`: detailed configuration reference for common-runner pipelines.
- `docs/pipeline_overview.md`: architecture of the configuration-driven workflows.
- `docs/troubleshooting.md`: common errors and diagnostics.
- `docs/polygon_mosaic_advanced.md`: advanced mosaic behavior.
- `docs/raster_adapt_polygon.md`: terrain reconstruction workflow.
- `docs/count_features_by_sector.md`: counts by sector.
- `docs/spatial_join_attributes.md`: spatial join with preserved attributes.
- `docs/configuration_conventions.md`: repository-wide YAML and path conventions.

## 11. Recommended first run

For a new installation:

```cmd
conda activate spatial_analysis
python -m scripts.run_from_config --help
python -c "import geopandas, rasterio, shapely, pandas, openpyxl, yaml; print('Imports OK')"
```

Then copy the closest example YAML, save it under a project-specific name, edit only the paths and required parameters, and run the corresponding command from the repository root.
