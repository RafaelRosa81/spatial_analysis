# Quick Start Guide

This guide gives a fast overview of the repository capabilities and shows minimal
YAML examples for each pipeline. It is intended as the first document to read after
opening the repo.

For deeper details, follow the documentation links at the end of each section.

---

## 1. What this repository does

`spatial_analysis` is a configuration-driven geospatial toolkit focused on raster,
polygon, mesh, and QGIS-oriented workflows.

The main idea is:

```text
YAML config -> scripts.run_from_config -> selected pipeline -> reproducible outputs
```

Most workflows are executed with:

```bash
python -m scripts.run_from_config --config config/your_config.yml
```

The YAML key that decides what runs is:

```yaml
pipeline: "pipeline_name"
```

---

## 2. Available pipelines

| Pipeline | Main purpose | Typical output |
| --- | --- | --- |
| `raster_diff` | Compare two rasters and generate difference rasters. | `dz.tif`, `abs_dz.tif`, Excel report, optional vectors. |
| `polygon_mosaic` | Combine two rasters using polygons, optional vertical adjustment and border blending. | Mosaic GeoTIFF, aligned/adjusted rasters, Excel report. |
| `sample_points_from_raster_value_range` | Generate sample points from raster cells within a value range. | CSV and GeoPackage point layers. |
| `landxml_tin_to_mesh` | Convert LandXML TIN surfaces to mesh files and optional DEM raster. | `vertices.csv`, `faces.csv`, `PLY/OBJ`, optional `tin_dem.tif`. |
| `raster_adapt_polygon` | Reconstruct terrain inside polygons using surrounding terrain. | Adapted DEM, adapted surface, diagnostic rasters, Excel report. |

---

## 3. Basic workflow

From the repository root:

```bash
conda activate spatial_analysis
python -m scripts.run_from_config --config config/your_config.yml
```

Recommended checks before running real projects:

```bash
python -m scripts.run_from_config --help
python -m scripts.landxml_tin_sanity
python scripts/polygon_mosaic_sanity.py
python -m scripts.raster_adapt_polygon_sanity
```

Some sanity scripts are pipeline-specific and only relevant when you are working on
that pipeline.

---

## 4. Pipeline: `raster_diff`

Use this when you need to compare two rasters, for example two DEM/MDT surfaces.

The core calculation is:

```text
dz = raster2 - raster1
abs_dz = |dz|
```

Minimal config:

```yaml
pipeline: "raster_diff"

raster_diff:
  name: "demo_diff"
  outdir: "outputs/demo_diff"
  raster1: "D:/path/to/base_dem.tif"
  raster2: "D:/path/to/new_dem.tif"
```

Common full config:

```yaml
pipeline: "raster_diff"

raster_diff:
  name: "demo_diff"
  outdir: "outputs/demo_diff"
  excel: true
  resampling: "bilinear"
  raster1: "D:/path/to/base_dem.tif"
  raster2: "D:/path/to/new_dem.tif"
  thresholds: [0.10, 0.25, 0.50, 1.00]
  bins: 60
  qgis_assets: true
  vector_threshold: 0.50
  signed_vector_threshold: 0.50
```

Typical outputs:

```text
outputs/demo_diff/
├─ aligned/
├─ rasters/
│  ├─ demo_diff_dz.tif
│  └─ demo_diff_abs_dz.tif
├─ report/
├─ vectors/
└─ qgis/
```

Read more:

- `docs/config_reference.md` -> Raster diff pipeline
- `docs/pipeline_overview.md`
- `README.md` -> Conceptual overview and QGIS integration

---

## 5. Pipeline: `polygon_mosaic`

Use this when you need to create a new raster by combining two rasters with one or
more polygons.

Typical use cases:

- insert a new surveyed DEM into a larger base DEM;
- keep one raster inside a polygon and another raster outside;
- vertically align one raster to the other before mosaicking;
- smooth the transition at polygon borders.

Minimal legacy-style config:

```yaml
pipeline: "polygon_mosaic"

polygon_mosaic:
  name: "demo_mosaic"
  outdir: "outputs/demo_mosaic"
  raster1: "D:/path/to/raster1.tif"
  raster2: "D:/path/to/raster2.tif"
  polygon: "D:/path/to/selection_polygon.shp"
```

Advanced config with explicit inside/outside selection:

```yaml
pipeline: "polygon_mosaic"

polygon_mosaic:
  name: "demo_mosaic"
  outdir: "outputs/demo_mosaic"
  excel: true

  raster1: "D:/path/to/raster_inside_polygon.tif"
  raster2: "D:/path/to/raster_outside_polygon.tif"

  polygon: "D:/path/to/selection_polygon.shp"

  selection:
    inside_polygon: "raster1"
    outside_polygon: "raster2"

  vertical_adjustment:
    enabled: true
    target: "raster2"
    mad_threshold: 0.25
    min_overlap_pixels: 30000
    exclude_polygon_buffer_px: 5

  border_blending:
    enabled: true
    blend_width_px: 10
```

Advanced config with a second polygon controlling final raster extent:

```yaml
pipeline: "polygon_mosaic"

polygon_mosaic:
  name: "demo_mosaic_extent"
  outdir: "outputs/demo_mosaic_extent"
  excel: true

  raster1: "D:/path/to/raster_inside_polygon.tif"
  raster2: "D:/path/to/raster_outside_polygon.tif"
  polygon: "D:/path/to/selection_polygon.shp"

  selection:
    inside_polygon: "raster1"
    outside_polygon: "raster2"

  output_grid:
    mode: "extent_polygon"
    reference: "raster2"
    extent_polygon: "D:/path/to/output_extent_polygon.shp"
    crop_to_extent_polygon: true
    mask_to_extent_polygon: true

  vertical_adjustment:
    enabled: true
    target: "raster2"
    mad_threshold: 0.25
    min_overlap_pixels: 30000
    exclude_polygon_buffer_px: 5

  outputs:
    new_raster: "mosaic_dem.tif"
    excel_report: "mosaic_report.xlsx"
    save_intermediates: true
```

Important ideas:

- `polygon` controls inside/outside raster selection.
- `output_grid.extent_polygon` controls output extent and can be a different polygon.
- `selection.inside_polygon` and `selection.outside_polygon` define which raster is used where.
- `vertical_adjustment.target` defines which raster is lifted/lowered.
- `output_grid.reference` defines CRS, pixel size, and grid alignment.
- GeoTIFF outputs are always rectangular; use `mask_to_extent_polygon: true` to set pixels outside the extent polygon to NoData.

Typical outputs:

```text
outputs/demo_mosaic/
├─ aligned/
│  ├─ demo_mosaic_raster2_aligned.tif
│  └─ demo_mosaic_raster2_adjusted.tif
├─ rasters/
│  ├─ mosaic_dem.tif
│  ├─ demo_mosaic_dz_overlap.tif
│  └─ demo_mosaic_blend_weights.tif
└─ report/
   └─ mosaic_report.xlsx
```

Read more:

- `docs/polygon_mosaic_advanced.md`
- `docs/config_reference.md`
- `docs/troubleshooting.md`

---

## 6. Pipeline: `sample_points_from_raster_value_range`

Use this when you need to create point samples from raster cells whose values are
inside a specific range.

Example use cases:

- sample all areas between two elevations;
- create inspection points from an inundation depth raster;
- generate CSV/GPKG points for QGIS.

Minimal config:

```yaml
pipeline: "sample_points_from_raster_value_range"

sample_points_from_raster_value_range:
  raster: "D:/path/to/dem.tif"
  value_min: 34.8
  value_max: 35.0
```

Full config:

```yaml
pipeline: "sample_points_from_raster_value_range"

sample_points_from_raster_value_range:
  name: "sample_points_cota"
  outdir: "outputs/sample_points_cota"
  raster: "D:/path/to/dem.tif"
  value_min: 34.8
  value_max: 35.0
  nodata_is_invalid: true
  mask_polygon: null

  sampling:
    method: "random"
    n_points: 2000
    seed: 42
    spacing: 5.0

  save_csv: true
  save_geopackage: true
  qgis_assets: true
```

Typical outputs:

```text
outputs/sample_points_cota/
├─ sample_points_cota.csv
└─ sample_points_cota.gpkg
```

Read more:

- `docs/config_reference.md`
- `docs/pipeline_overview.md`

---

## 7. Pipeline: `landxml_tin_to_mesh`

Use this when you have a LandXML/XML topographic file containing a TIN surface with
points and faces, and you want to preserve the original triangulation.

The pipeline reads:

```text
Pnts  -> vertices
Faces -> triangle connectivity
```

and exports mesh files and, optionally, a DEM GeoTIFF generated directly from the
TIN triangles.

Minimal config:

```yaml
pipeline: "landxml_tin_to_mesh"

landxml_tin_to_mesh:
  name: "landxml_demo"
  outdir: "outputs/landxml_demo"
  input_xml: "D:/path/to/topographic_surface.xml"
  surface_name: "TN May26"
```

Full config with coordinate order and rasterization:

```yaml
pipeline: "landxml_tin_to_mesh"

landxml_tin_to_mesh:
  name: "safa_1304_landxml"
  outdir: "outputs/safa_1304_landxml"
  excel: true

  input_xml: "D:/path/to/Relevamiento SAFA 1304_29May26.xml"
  surface_name: "TN May26"

  coordinate_order: "yxz"
  crs: "EPSG:32721"

  outputs:
    vertices_csv: "vertices.csv"
    faces_csv: "faces.csv"
    obj: "tin_mesh.obj"
    ply: "tin_mesh.ply"
    excel_report: "landxml_tin_report.xlsx"
    summary_json: "landxml_tin_summary.json"

  options:
    write_ply: true
    strict_faces: true
    validate_counts: true
    preserve_original_point_ids: true

  rasterize:
    enabled: true
    pixel_size: 0.50
    output_dem: "tin_dem.tif"
    nodata: -9999

  expected:
    surface_type: "TIN"
    n_points: 386
    n_faces: 752
```

Typical outputs:

```text
outputs/safa_1304_landxml/
├─ mesh/
│  ├─ vertices.csv
│  ├─ faces.csv
│  ├─ tin_mesh.obj
│  └─ tin_mesh.ply
├─ rasters/
│  └─ tin_dem.tif
├─ report/
│  └─ landxml_tin_report.xlsx
└─ metadata/
   └─ landxml_tin_summary.json
```

Read more:

- `docs/config_reference.md`
- `docs/pipeline_overview.md`
- `README.md`

---

## 8. Pipeline: `raster_adapt_polygon`

Use this when you need to reconstruct or adapt terrain inside a polygon using the
surrounding terrain as reference.

Typical use cases:

- remove artificial fills;
- reconstruct terrain continuity;
- smooth disturbed DEM zones;
- prepare alternative terrain scenarios for hydraulic modelling.

Conceptual workflow:

```text
input raster
    ↓
modify polygon
    ↓
reference ring outside polygon
    ↓
interpolation / fitted surface
    ↓
adapted surface
    ↓
optional border blending
    ↓
final adapted DEM
```

Minimal config:

```yaml
pipeline: "raster_adapt_polygon"

raster_adapt_polygon:
  name: "adapt_demo"
  outdir: "outputs/adapt_demo"

  raster: "D:/path/to/input_dem.tif"
  modify_polygon: "D:/path/to/modify_polygon.shp"
```

Advanced config:

```yaml
pipeline: "raster_adapt_polygon"

raster_adapt_polygon:
  name: "adapt_demo"
  outdir: "outputs/adapt_demo"
  excel: true

  raster: "D:/path/to/input_dem.tif"
  modify_polygon: "D:/path/to/modify_polygon.shp"

  outputs:
    adapted_raster: "adapted_dem.tif"
    excel_report: "adapt_report.xlsx"
    summary_json: "adapt_summary.json"
    save_intermediates: true

  reference_ring:
    outer_buffer_px: 50
    inner_buffer_px: 10
    min_reference_pixels: 1000

  adaptation:
    method: "boundary_idw"
    idw_power: 3.0
    k_nearest: 12
    max_search_distance_px: 20
    max_reference_points: 20000
    random_seed: 42
    polynomial_order: 2

  border_blending:
    enabled: true
    blend_width_px: 2

  nodata:
    preserve_nodata: true
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
outputs/adapt_demo/
├─ rasters/
│  ├─ adapted_dem.tif
│  ├─ adapt_demo_adapted_surface.tif
│  ├─ adapt_demo_modify_mask.tif
│  ├─ adapt_demo_reference_ring.tif
│  └─ adapt_demo_blend_weights.tif
├─ report/
│  └─ adapt_report.xlsx
└─ metadata/
   └─ adapt_summary.json
```

Read more:

- `docs/raster_adapt_polygon.md`
- `docs/config_reference.md`
- `docs/troubleshooting.md`

---

## 9. Recommended documentation map

Start here:

1. `README.md`
2. `docs/documentation_index.md`
3. `docs/quick_start_guide.md`
4. `docs/config_reference.md`
5. pipeline-specific documentation

---

## 10. Practical tips

### Use forward slashes in YAML paths

Recommended:

```yaml
raster1: "D:/path/to/raster.tif"
```

Also valid:

```yaml
raster1: 'D:\path\to\raster.tif'
```

Avoid double-quoted Windows paths with backslashes such as:

```yaml
raster1: "D:\path\to\raster.tif"
```

unless backslashes are properly escaped.

### Keep project configs local when they contain real data paths

Tracked examples should use placeholder paths:

```text
config/*_example.yml
```

Local project configs with real paths should normally be ignored:

```text
config/*_local.yml
```

### Always inspect intermediate rasters

For terrain workflows, inspect:

```text
aligned rasters
adjusted rasters
reference rings
blend weights
adapted surfaces
```

in QGIS before accepting the final DEM.
