# Spatial Analysis Toolkit

Python tools for reproducible geospatial analysis with QGIS-oriented outputs and YAML-driven workflows.

## Start here

New users should read:

1. `docs/quick_start_guide.md` — installation, basic Git commands, Windows path rules, examples, and execution commands.
2. `docs/documentation_index.md` — complete documentation map.
3. The documentation for the pipeline being used.

## Installation

```cmd
git clone https://github.com/RafaelRosa81/spatial_analysis.git
cd spatial_analysis
conda env create -f environment.yml
conda activate spatial_analysis
```

To update an existing local clone:

```cmd
git status
git checkout main
git pull --ff-only origin main
```

## Windows paths in YAML

Repository convention:

```yaml
path: 'D:\datos\archivo.shp'
```

Use backslashes and single quotes. Avoid double quotes around Windows paths containing unescaped backslashes.

## Pipelines

### Common YAML runner

These pipelines are executed with:

```cmd
python -m scripts.run_from_config --config config\your_config.yml
```

| Pipeline | Purpose |
| --- | --- |
| `raster_diff` | Align and compare two rasters; produce signed/absolute differences, reports, and optional vectors. |
| `polygon_mosaic` | Combine two rasters using polygon-controlled selection, optional vertical adjustment, and border blending. |
| `sample_points_from_raster_value_range` | Create point samples from raster cells within a configured value range. |
| `landxml_tin_to_mesh` | Convert LandXML TIN surfaces into vertices/faces, OBJ/PLY meshes, reports, and an optional DEM. |
| `raster_adapt_polygon` | Reconstruct or adapt terrain inside polygons from surrounding terrain. |

Show common-runner help:

```cmd
python -m scripts.run_from_config --help
```

### Dedicated vector-zone runners

Count configured feature layers by sector:

```cmd
python -m scripts.run_count_features_by_sector --config config\count_features_by_sector_example.yml
```

Assign features to polygon zones while preserving attributes:

```cmd
python -m scripts.run_spatial_join_attributes --config config\spatial_join_attributes_example.yml
```

## Minimal raster-difference example

```yaml
pipeline: 'raster_diff'

raster_diff:
  name: 'demo_run'
  outdir: 'outputs\demo_run'
  raster1: 'D:\datos\dem_base.tif'
  raster2: 'D:\datos\dem_nuevo.tif'
  resampling: 'bilinear'
  excel: true
  thresholds: [0.10, 0.25, 0.50, 1.00]
```

The signed difference is:

```text
dz = raster2 - raster1
```

- `dz > 0`: raster 2 is higher than raster 1.
- `dz < 0`: raster 2 is lower than raster 1.
- `abs_dz = |dz|`: magnitude of change without sign.

## Main outputs

Depending on the pipeline, outputs may include:

- GeoTIFF rasters;
- GeoPackage or GeoJSON vector layers;
- CSV tables;
- Excel reports;
- OBJ/PLY meshes;
- JSON/CSV diagnostics;
- QGIS styles and review layers.

## Validation commands

```cmd
python -c "import geopandas, rasterio, shapely, pandas, openpyxl, yaml; print('Imports OK')"
python tests\run_regression.py
```

Pipeline-specific checks:

```cmd
python scripts\polygon_mosaic_sanity.py
python -m scripts.landxml_tin_sanity
python -m scripts.raster_adapt_polygon_sanity
```

## QGIS workflow

Typical workflow:

1. Prepare source layers with valid CRS information.
2. Copy and edit the closest YAML example.
3. Run the corresponding command from the repository root.
4. Load generated rasters, GeoPackages, GeoJSON files, or CSV outputs into QGIS.
5. Review unmatched features, topology issues, NoData areas, and CRS/alignment diagnostics before using results operationally.

## Documentation

- `docs/quick_start_guide.md` — first-use guide and essential commands.
- `docs/documentation_index.md` — central documentation index.
- `docs/config_reference.md` — detailed parameters for common-runner pipelines.
- `docs/pipeline_overview.md` — pipeline architecture and execution model.
- `docs/architecture.md` — repository organization.
- `docs/troubleshooting.md` — common runtime and data problems.
- `docs/polygon_mosaic_advanced.md` — advanced mosaic behavior.
- `docs/raster_adapt_polygon.md` — terrain adaptation workflow.
- `docs/count_features_by_sector.md` — counts by polygon sector.
- `docs/spatial_join_attributes.md` — attribute-preserving spatial join.
- `docs/configuration_conventions.md` — YAML and Windows path conventions.

## Development workflow

Create work on a dedicated branch:

```cmd
git checkout main
git pull --ff-only origin main
git checkout -b feature/short-description
```

Before committing:

```cmd
git status
git diff
python tests\run_regression.py
```

After a branch is merged and deleted remotely:

```cmd
git checkout main
git pull --ff-only origin main
git fetch --prune
```
