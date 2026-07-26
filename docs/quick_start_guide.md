# Quick Start Guide

This guide is intended for someone opening the repository for the first time. It explains how to obtain the code, activate the environment, choose a workflow, copy an example configuration, run it, and inspect the result.

For the complete documentation map, see:

```text
docs/documentation_index.md
```

---

## 1. Download and update the repository

Clone for the first time:

```bash
git clone https://github.com/RafaelRosa81/spatial_analysis.git
cd spatial_analysis
```

Update an existing local copy:

```bash
git checkout main
git pull origin main
```

Useful checks:

```bash
git status
git branch --show-current
git log --oneline --decorate --max-count=10
```

---

## 2. Activate or create the environment

Create it once:

```bash
conda env create -f environment.yml
```

Activate it for each working session:

```bash
conda activate spatial_analysis
```

Check the integrated runner:

```bash
python -m scripts.run_from_config --help
```

---

## 3. How configuration works

Most workflows use:

```bash
python -m scripts.run_from_config --config config/your_config.yml
```

The YAML selector is:

```yaml
pipeline: "pipeline_name"
```

Recommended workflow:

1. Find the closest example in `config/`.
2. Copy it to a project-specific filename.
3. Edit input paths, layer names, parameters, and output directory.
4. Run from the repository root.
5. Review generated rasters/vectors and reports in QGIS and Excel.

Windows paths may use forward slashes:

```yaml
path: "D:/data/project/layer.shp"
```

or single-quoted backslashes:

```yaml
path: 'D:\data\project\layer.shp'
```

See `docs/configuration_conventions.md`.

---

## 4. Workflow chooser

| Need | Workflow |
| --- | --- |
| Compare two DEMs or rasters | `raster_diff` |
| Insert/combine one raster inside another | `polygon_mosaic` |
| Reconstruct terrain inside a polygon | `raster_adapt_polygon` |
| Create points from a raster value interval | `sample_points_from_raster_value_range` |
| Convert a LandXML TIN to mesh/DEM | `landxml_tin_to_mesh` |
| Count points/polygons per sector | `count_features_by_sector` |
| Assign sector/zone attributes to features | `spatial_join_attributes` |

---

## 5. Raster Diff

Purpose: align two rasters and calculate:

```text
dz = raster2 - raster1
abs_dz = |dz|
```

Run:

```bash
python -m scripts.run_from_config --config config/minimal_raster_diff_example.yml
```

Minimal YAML:

```yaml
pipeline: "raster_diff"

raster_diff:
  name: "demo_diff"
  outdir: "outputs/demo_diff"
  raster1: "D:/path/to/base_dem.tif"
  raster2: "D:/path/to/new_dem.tif"
```

Typical outputs:

```text
aligned rasters
dz.tif
abs_dz.tif
Excel report
optional GeoJSON exceedance polygons
```

Read: `docs/config_reference.md`.

---

## 6. Polygon Mosaic

Purpose: combine two rasters using a polygon, with optional vertical adjustment, inside/outside selection, border blending, and custom output extent.

Run:

```bash
python -m scripts.run_from_config --config config/polygon_mosaic_example.yml
```

Minimal YAML:

```yaml
pipeline: "polygon_mosaic"

polygon_mosaic:
  name: "demo_mosaic"
  outdir: "outputs/demo_mosaic"
  raster1: "D:/path/to/raster1.tif"
  raster2: "D:/path/to/raster2.tif"
  polygon: "D:/path/to/selection_polygon.shp"
```

Important concepts:

```text
selection.inside_polygon
selection.outside_polygon
vertical_adjustment.target
border_blending
output_grid.extent_polygon
```

Read:

```text
docs/polygon_mosaic_advanced.md
docs/config_reference.md
```

---

## 7. Raster Adapt Polygon

Purpose: reconstruct the raster inside a polygon from surrounding terrain while preserving the original raster outside it.

Run:

```bash
python -m scripts.run_from_config --config config/raster_adapt_polygon_example.yml
```

Minimal YAML:

```yaml
pipeline: "raster_adapt_polygon"

raster_adapt_polygon:
  name: "adapt_demo"
  outdir: "outputs/adapt_demo"
  raster: "D:/path/to/input_dem.tif"
  modify_polygon: "D:/path/to/modify_polygon.shp"
```

Supported methods:

```text
boundary_idw
nearest_boundary
plane_fit
polynomial_fit
```

Typical outputs:

```text
adapted raster
adapted surface
modify mask
reference ring
blend weights
Excel report
JSON summary
```

Read: `docs/raster_adapt_polygon.md`.

---

## 8. Sample Points From Raster Value Range

Purpose: create point samples from raster cells between inclusive minimum and maximum values.

Run:

```bash
python -m scripts.run_from_config --config config/sample_points_example.yml
```

Minimal YAML:

```yaml
pipeline: "sample_points_from_raster_value_range"

sample_points_from_raster_value_range:
  raster: "D:/path/to/dem.tif"
  value_min: 34.8
  value_max: 35.0
```

Typical outputs:

```text
CSV point table
GeoPackage point layer
```

Read: `docs/config_reference.md`.

---

## 9. LandXML TIN To Mesh

Purpose: preserve the points and triangle faces from a LandXML TIN and optionally generate a raster DEM directly from the triangulation.

Run:

```bash
python -m scripts.run_from_config --config config/landxml_tin_to_mesh_example.yml
```

Minimal YAML:

```yaml
pipeline: "landxml_tin_to_mesh"

landxml_tin_to_mesh:
  name: "landxml_demo"
  outdir: "outputs/landxml_demo"
  input_xml: "D:/path/to/topographic_surface.xml"
  surface_name: "TN"
```

Typical outputs:

```text
vertices.csv
faces.csv
OBJ/PLY mesh
optional DEM GeoTIFF
Excel report
JSON summary
```

Read: `docs/config_reference.md` and the LandXML section in this repository documentation.

---

## 10. Count Features By Sector

This workflow currently uses a dedicated runner.

Purpose: count point and polygon features in every sector polygon.

Run:

```bash
python -m scripts.run_sector_counts --config config/count_features_by_sector_example.yml
```

Minimal YAML:

```yaml
pipeline: count_features_by_sector

count_features_by_sector:
  name: sector_feature_counts
  sectors: "D:/path/to/sectors.shp"
  sector_id_field: sector
  outdir: "outputs/sector_feature_counts"
  layers:
    - path: "D:/path/to/trees.shp"
      name: trees
```

Typical outputs:

```text
CSV counts table
Excel workbook
optional GeoPackage sectors with count fields
```

Read: `docs/count_features_by_sector.md`.

---

## 11. Spatial Join Attributes

This workflow currently uses a dedicated runner.

Purpose: assign polygon-zone attributes to point or polygon input features while preserving the source attributes and reporting unmatched features.

Run:

```bash
python -m scripts.run_spatial_join_attributes --config config/spatial_join_attributes_example.yml
```

Minimal YAML:

```yaml
pipeline: spatial_join_attributes

spatial_join_attributes:
  name: features_by_zone
  outdir: "outputs/features_by_zone"

  zones:
    path: "D:/path/to/sectors.shp"
    id_field: sector_id

  input:
    path: "D:/path/to/features.gpkg"
    layer: points

  predicate: covered_by
  join_type: left
```

Typical outputs:

```text
CSV joined table
Excel workbook
GeoPackage joined layer
optional unmatched outputs
```

Read:

```text
docs/spatial_join_attributes.md
docs/configuration_conventions.md
```

---

## 12. Sanity checks

Run after installation or code updates:

```bash
python -m scripts.run_from_config --help
python scripts/polygon_mosaic_sanity.py
python -m scripts.landxml_tin_sanity
python -m scripts.raster_adapt_polygon_sanity
```

---

## 13. Validation in QGIS

For raster workflows, load:

- final output raster;
- original input raster(s);
- masks/reference rings/blend weights;
- difference or overlap diagnostics.

For vector workflows, load:

- source zones;
- source features;
- output GeoPackage;
- unmatched layer when available.

Always check:

- CRS;
- expected feature/pixel counts;
- NoData;
- boundaries and overlaps;
- Excel/JSON report values;
- whether real project paths should remain local rather than committed.
