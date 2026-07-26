# Configuration Reference

This document is the canonical configuration reference for the workflows currently
implemented in `spatial_analysis`.

It has been audited against the configuration resolvers and defaults in:

```text
scripts/run_from_config.py
raster_compare/polygon_mosaic.py
raster_compare/sample_points.py
raster_compare/landxml_tin.py
raster_compare/raster_adapt_polygon.py
raster_compare/sector_counts.py
raster_compare/spatial_join_attributes.py
```

For conceptual explanations and worked examples, also consult:

```text
docs/documentation_index.md
docs/quick_start_guide.md
docs/polygon_mosaic_advanced.md
docs/raster_adapt_polygon.md
docs/count_features_by_sector.md
docs/spatial_join_attributes.md
```

---

# 1. Execution model

The repository currently has two execution patterns.

## 1.1 Integrated YAML runner

These pipelines run through:

```bash
python -m scripts.run_from_config --config config/your_config.yml
```

Supported `pipeline` values:

```text
raster_diff
polygon_mosaic
sample_points_from_raster_value_range
sample_points                         # alias
landxml_tin_to_mesh
raster_adapt_polygon
```

If `pipeline` is omitted, the integrated runner defaults to `raster_diff` for
backwards compatibility.

## 1.2 Dedicated runners

These workflows have their own entrypoints:

```bash
python -m scripts.run_sector_counts --config config/count_features_by_sector_example.yml
python -m scripts.run_spatial_join_attributes --config config/spatial_join_attributes_example.yml
```

Their YAML files still contain a `pipeline` key for traceability, but they are not
currently dispatched by `scripts.run_from_config`.

---

# 2. General YAML conventions

## 2.1 Recommended nested structure

Use one top-level selector and one workflow-specific mapping:

```yaml
pipeline: "raster_diff"

raster_diff:
  name: "demo"
  outdir: "outputs/demo"
  raster1: "D:/data/base.tif"
  raster2: "D:/data/new.tif"
```

Some older raster workflows accept selected root-level keys for backwards
compatibility. New configurations should use the nested form.

## 2.2 Windows paths

Two safe conventions are used in the repository.

Forward slashes:

```yaml
raster: "D:/data/project/dem.tif"
```

Or copied Windows paths inside single quotes:

```yaml
path: 'D:\data\project\sectors.shp'
```

Avoid unescaped backslashes inside double quotes:

```yaml
path: "D:\data\project\sectors.shp"
```

because YAML may interpret sequences such as `\t` or `\n` as escapes.

See `docs/configuration_conventions.md`.

## 2.3 Example and local configs

Recommended convention:

```text
config/*_example.yml  -> tracked examples with placeholder paths
config/*_local.yml    -> project-specific paths, normally not committed
```

## 2.4 Boolean and null values

Use native YAML values:

```yaml
enabled: true
write_excel: false
layer: null
```

Do not quote them unless a string is intentionally required.

---

# 3. Pipeline summary

| Workflow | Runner | Required primary inputs |
| --- | --- | --- |
| `raster_diff` | `scripts.run_from_config` | `raster1`, `raster2`, `name`, `outdir` |
| `polygon_mosaic` | `scripts.run_from_config` | `raster1`, `raster2`, `polygon`, `name`, `outdir` |
| `sample_points_from_raster_value_range` | `scripts.run_from_config` | `raster`, `value_min`, `value_max` |
| `landxml_tin_to_mesh` | `scripts.run_from_config` | `input_xml` |
| `raster_adapt_polygon` | `scripts.run_from_config` | `raster`, `modify_polygon` |
| `count_features_by_sector` | `scripts.run_sector_counts` | `sectors`, `sector_id_field`, `layers`, `outdir` |
| `spatial_join_attributes` | `scripts.run_spatial_join_attributes` | `zones.path`, `zones.id_field`, `input.path`, `outdir` |

---

# 4. Raster difference

```yaml
pipeline: "raster_diff"
```

Runner:

```bash
python -m scripts.run_from_config --config config/minimal_raster_diff_example.yml
```

## 4.1 Required parameters

| Key | Type | Meaning |
| --- | --- | --- |
| `raster_diff.name` | string | Output prefix/run name. |
| `raster_diff.outdir` | path | Root output directory. |
| `raster_diff.raster1` | path | Reference raster and alignment grid. |
| `raster_diff.raster2` | path | Raster compared against `raster1`. |

The resolver also accepts these keys at YAML root level for backwards compatibility.

## 4.2 Optional parameters and defaults

| Key | Type | Default | Validation/meaning |
| --- | --- | --- | --- |
| `resampling` | string | `bilinear` | One of `nearest`, `bilinear`, `cubic`. Used to align `raster2` to `raster1`. |
| `excel` | boolean | `true` | Generate Excel comparison report. |
| `thresholds` | list[number] | `[0.1, 0.25, 0.5, 1.0]` | Non-empty numeric list used in statistics/reporting. |
| `bins` | integer | `60` | Positive histogram bin count. |
| `qgis_assets` | boolean | `true` | Copy QGIS style assets. |
| `vector_threshold` | number/null | `0.5` | Polygonize areas where `abs_dz` exceeds the threshold. Must be > 0 or `null`. |
| `signed_vector_threshold` | number/null | `null` | Polygonize positive and negative signed changes. Must be > 0 or `null`. |

## 4.3 Calculation

```text
dz     = raster2_aligned - raster1_aligned
abs_dz = absolute value of dz
```

Interpretation:

```text
dz > 0 -> raster2 is higher
dz < 0 -> raster2 is lower
```

## 4.4 Full example

```yaml
pipeline: "raster_diff"

raster_diff:
  name: "demo_diff"
  outdir: "outputs/demo_diff"
  raster1: "D:/data/base_dem.tif"
  raster2: "D:/data/new_dem.tif"
  resampling: "bilinear"
  excel: true
  thresholds: [0.10, 0.25, 0.50, 1.00]
  bins: 60
  qgis_assets: true
  vector_threshold: 0.50
  signed_vector_threshold: 0.50
```

## 4.5 Outputs

```text
<outdir>/aligned/<name>_raster1_aligned.tif
<outdir>/aligned/<name>_raster2_aligned.tif
<outdir>/rasters/<name>_dz.tif
<outdir>/rasters/<name>_abs_dz.tif
<outdir>/report/<name>_Comparison_Report.xlsx
<outdir>/report/<name>_alignment_report.json
<outdir>/report/<name>_alignment_report.csv
<outdir>/vectors/*.geojson                # when thresholds are enabled
<outdir>/qgis/                             # when qgis_assets is true
```

---

# 5. Polygon mosaic

```yaml
pipeline: "polygon_mosaic"
```

Runner:

```bash
python -m scripts.run_from_config --config config/polygon_mosaic_example.yml
```

Detailed guide: `docs/polygon_mosaic_advanced.md`.

## 5.1 Required parameters

| Key | Type | Meaning |
| --- | --- | --- |
| `polygon_mosaic.name` | string | Run/output prefix. |
| `polygon_mosaic.outdir` | path | Root output directory. |
| `polygon_mosaic.raster1` | path | First raster. |
| `polygon_mosaic.raster2` | path | Second raster. |
| `polygon_mosaic.polygon` | path | Polygon controlling inside/outside selection. |

The paths are validated during execution. The polygon must contain usable
geometries.

## 5.2 Canonical defaults

```yaml
polygon_mosaic:
  excel: true

  outputs:
    new_raster: "new_raster.tif"
    excel_report: "polygon_mosaic_report.xlsx"
    save_intermediates: true

  selection:
    inside_polygon: "raster2"
    outside_polygon: "raster1"

  output_grid:
    mode: "reference"
    reference: "raster1"
    extent_polygon: null
    crop_to_extent_polygon: false
    mask_to_extent_polygon: false

  alignment:
    resampling: "bilinear"

  vertical_adjustment:
    enabled: true
    method: "constant_offset"
    robust_stat: "median"
    target: "raster2"
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

## 5.3 Selection

| Key | Type | Default | Validation/meaning |
| --- | --- | --- | --- |
| `selection.inside_polygon` | string | `raster2` | Must be `raster1` or `raster2`. |
| `selection.outside_polygon` | string | `raster1` | Must be `raster1` or `raster2`. |

The two choices must be different.

Example:

```yaml
selection:
  inside_polygon: "raster1"
  outside_polygon: "raster2"
```

## 5.4 Output grid

| Key | Type | Default | Validation/meaning |
| --- | --- | --- | --- |
| `output_grid.mode` | string | `reference` | `reference` or `extent_polygon`. |
| `output_grid.reference` | string | `raster1` | Raster defining CRS, resolution and alignment; must be `raster1` or `raster2`. |
| `output_grid.extent_polygon` | path/null | `null` | Required when mode is `extent_polygon`. |
| `output_grid.crop_to_extent_polygon` | boolean | `false` | Crop rectangular output grid to polygon bounds. |
| `output_grid.mask_to_extent_polygon` | boolean | `false` | Set cells outside extent polygon to NoData. |

The selection polygon and extent polygon may be different files.

## 5.5 Alignment

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `alignment.resampling` | string | `bilinear` | Any rasterio `Resampling` name accepted by the installed version. |

A legacy root-level `resampling` key is forwarded when nested
`alignment.resampling` is absent.

For continuous DEMs, `bilinear` is normally appropriate. For categorical rasters,
use `nearest`.

## 5.6 Vertical adjustment

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vertical_adjustment.enabled` | boolean | `true` | Enable/disable vertical offset estimation. |
| `vertical_adjustment.method` | string | `constant_offset` | Current configured method name. |
| `vertical_adjustment.robust_stat` | string | `median` | Robust statistic configured for offset estimation. |
| `vertical_adjustment.target` | string | `raster2` | Raster to raise/lower; must be `raster1` or `raster2`. |
| `vertical_adjustment.mad_threshold` | number | `0.10` | Stability threshold based on median absolute deviation. |
| `vertical_adjustment.min_overlap_pixels` | integer | `50000` | Minimum usable overlap pixels required before applying adjustment. |
| `vertical_adjustment.exclude_polygon_buffer_px` | integer | `5` | Exclude a buffer around the selection polygon from offset estimation. |

`mad_threshold` does not define the vertical offset. It determines whether the
overlap differences are sufficiently stable for the configured adjustment.

## 5.7 Border blending

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `border_blending.enabled` | boolean | `true` | Enable transition at polygon boundary. |
| `border_blending.blend_width_px` | integer | `5` | Transition width in pixels. |
| `border_blending.weight_curve` | string | `linear` | Configured blending curve. |

## 5.8 NoData

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `nodata.use_raster1_nodata` | boolean | `true` | Use `raster1` NoData convention for output. |

## 5.9 Example with explicit selection and extent

```yaml
pipeline: "polygon_mosaic"

polygon_mosaic:
  name: "demo_mosaic"
  outdir: "outputs/demo_mosaic"
  excel: true

  raster1: "D:/data/local_surface.tif"
  raster2: "D:/data/base_surface.tif"
  polygon: "D:/data/selection_polygon.shp"

  selection:
    inside_polygon: "raster1"
    outside_polygon: "raster2"

  output_grid:
    mode: "extent_polygon"
    reference: "raster2"
    extent_polygon: "D:/data/output_extent.shp"
    crop_to_extent_polygon: true
    mask_to_extent_polygon: true

  alignment:
    resampling: "bilinear"

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

---

# 6. Sample points from raster value range

```yaml
pipeline: "sample_points_from_raster_value_range"
```

Alias accepted by the integrated runner:

```yaml
pipeline: "sample_points"
```

Runner:

```bash
python -m scripts.run_from_config --config config/sample_points_example.yml
```

## 6.1 Required parameters

| Key | Type | Meaning |
| --- | --- | --- |
| `raster` | path | Raster used for filtering and sampling. |
| `value_min` | number | Inclusive minimum value. |
| `value_max` | number | Inclusive maximum value. Must be >= `value_min`. |

These keys belong under `sample_points_from_raster_value_range`.

## 6.2 Optional parameters and defaults

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | `sample_points` | Output filename prefix. |
| `outdir` | path | `outputs/<name>` | Output directory. |
| `sampling.method` | string | `random` | `random` or `regular`. |
| `sampling.n_points` | integer | `1000` | Maximum number for random sampling and fallback random take. |
| `sampling.seed` | integer/null | `null` | Reproducible random seed. |
| `sampling.spacing` | number/null | `null` | Required and > 0 for regular sampling; interpreted in map units. |
| `mask_polygon` | path/null | `null` | Accepted and normalized by the resolver, but not applied by the current pipeline implementation. |
| `nodata_is_invalid` | boolean | `true` | Exclude the raster NoData value before range filtering. |
| `save_geopackage` | boolean | `true` | Write GeoPackage output. Requires GeoPandas/Shapely. |
| `save_csv` | boolean | `true` | Write CSV output. |
| `qgis_assets` | boolean | `true` | Accepted by the resolver; currently not used by the execution function. |

## 6.3 Sampling semantics

Random:

```yaml
sampling:
  method: "random"
  n_points: 2000
  seed: 42
```

The number returned is the smaller of `n_points` and the number of qualifying
pixels. Sampling is without replacement.

Regular:

```yaml
sampling:
  method: "regular"
  spacing: 5.0
```

`spacing` is converted to an approximate pixel step using the raster x resolution.
If no points survive the regular grid, the implementation falls back to a random
sample controlled by `n_points` and `seed`.

Use a projected CRS when spacing should represent metres.

## 6.4 Full example

```yaml
pipeline: "sample_points_from_raster_value_range"

sample_points_from_raster_value_range:
  name: "sample_points_cota"
  outdir: "outputs/sample_points_cota"
  raster: "D:/data/dem.tif"
  value_min: 34.8
  value_max: 35.0

  sampling:
    method: "random"
    n_points: 2000
    seed: 42
    spacing: null

  mask_polygon: null
  nodata_is_invalid: true
  save_geopackage: true
  save_csv: true
  qgis_assets: true
```

## 6.5 Outputs

```text
<outdir>/<name>.csv
<outdir>/<name>.gpkg
```

The GeoPackage geometry is created at raster cell centres and uses the raster CRS.

---

# 7. LandXML TIN to mesh

```yaml
pipeline: "landxml_tin_to_mesh"
```

Runner:

```bash
python -m scripts.run_from_config --config config/landxml_tin_to_mesh_example.yml
```

## 7.1 Required parameters

| Key | Type | Meaning |
| --- | --- | --- |
| `input_xml` | path | LandXML/XML input containing one or more `Surface` elements. |

## 7.2 General parameters and defaults

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | `landxml_tin` | Run name; used to derive default outdir. |
| `outdir` | path | `outputs/<name>` | Root output directory. |
| `excel` | boolean | `true` | Generate Excel report. |
| `surface_name` | string/null | `null` | Exact surface name to select. If null, first TIN surface is selected; a warning is recorded if several exist. |
| `coordinate_order` | string | `xyz` | Three-character permutation of `xyz`, e.g. `xyz` or `yxz`. |
| `crs` | string/null | `null` | Output CRS. Required when rasterization is enabled. |

`coordinate_order` describes how each point triplet is stored in the XML. For
LandXML files storing northing/easting/elevation, `yxz` may be required.

## 7.3 Output filenames

```yaml
outputs:
  vertices_csv: "vertices.csv"
  faces_csv: "faces.csv"
  obj: "tin_mesh.obj"
  ply: "tin_mesh.ply"
  excel_report: "landxml_tin_report.xlsx"
  summary_json: "landxml_tin_summary.json"
```

## 7.4 Options

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `options.write_ply` | boolean | `true` | Write PLY mesh in addition to OBJ. |
| `options.strict_faces` | boolean | `true` | Fail when faces reference missing point IDs. |
| `options.validate_counts` | boolean | `true` | Enforce configured expected counts/type when provided. |
| `options.preserve_original_point_ids` | boolean | `true` | Configuration flag retained for point-ID traceability. Original IDs are included in current CSV exports. |
| `options.degenerate_area_tolerance` | number | `1e-12` | 3D triangle-area threshold used to classify degenerate faces. |

## 7.5 Expected values

```yaml
expected:
  surface_type: "TIN"
  n_points: null
  n_faces: null
```

| Key | Default | Meaning |
| --- | --- | --- |
| `expected.surface_type` | `TIN` | Expected selected surface type. |
| `expected.n_points` | `null` | Optional exact point-count check. |
| `expected.n_faces` | `null` | Optional exact triangular-face count check. |

When `options.validate_counts: true`, configured expected checks must pass.

## 7.6 Rasterization

```yaml
rasterize:
  enabled: false
  pixel_size: 0.50
  output_dem: "tin_dem.tif"
  nodata: -9999.0
  all_touched: false
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `rasterize.enabled` | boolean | `false` | Create a DEM directly from TIN triangles. |
| `rasterize.pixel_size` | number | `0.50` | Positive output pixel size in CRS units. |
| `rasterize.output_dem` | string | `tin_dem.tif` | DEM filename. |
| `rasterize.nodata` | number | `-9999.0` | Output NoData value. |
| `rasterize.all_touched` | boolean | `false` | Accepted rasterization option. Current V1 still uses cell-centre triangle interpolation. |

`crs` is required when `rasterize.enabled` is true.

## 7.7 Full example

```yaml
pipeline: "landxml_tin_to_mesh"

landxml_tin_to_mesh:
  name: "landxml_demo"
  outdir: "outputs/landxml_demo"
  excel: true

  input_xml: "D:/data/surface.xml"
  surface_name: "Existing Ground"
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
    degenerate_area_tolerance: 1.0e-12

  rasterize:
    enabled: true
    pixel_size: 0.50
    output_dem: "tin_dem.tif"
    nodata: -9999
    all_touched: false

  expected:
    surface_type: "TIN"
    n_points: null
    n_faces: null
```

## 7.8 Outputs

```text
<outdir>/mesh/vertices.csv
<outdir>/mesh/faces.csv
<outdir>/mesh/tin_mesh.obj
<outdir>/mesh/tin_mesh.ply               # when write_ply is true
<outdir>/rasters/tin_dem.tif             # when rasterization is enabled
<outdir>/report/landxml_tin_report.xlsx  # when excel is true
<outdir>/metadata/landxml_tin_summary.json
```

---

# 8. Raster adapt polygon

```yaml
pipeline: "raster_adapt_polygon"
```

Runner:

```bash
python -m scripts.run_from_config --config config/raster_adapt_polygon_example.yml
```

Detailed guide: `docs/raster_adapt_polygon.md`.

## 8.1 Required parameters

| Key | Type | Meaning |
| --- | --- | --- |
| `raster` | path | Input raster/DEM to modify. Must exist and have a CRS. |
| `modify_polygon` | path | Polygon defining cells that may be replaced. Must exist and overlap the raster. |

## 8.2 General parameters and defaults

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | `raster_adapt_polygon` | Run/output prefix. |
| `outdir` | path | `outputs/<name>` | Root output directory. |
| `excel` | boolean | `true` | Write Excel diagnostic report. |

## 8.3 Outputs

```yaml
outputs:
  adapted_raster: "adapted_raster.tif"
  excel_report: "raster_adapt_polygon_report.xlsx"
  summary_json: "raster_adapt_polygon_summary.json"
  save_intermediates: true
```

| Key | Default | Meaning |
| --- | --- | --- |
| `outputs.adapted_raster` | `adapted_raster.tif` | Final DEM filename. |
| `outputs.excel_report` | `raster_adapt_polygon_report.xlsx` | Excel report filename. |
| `outputs.summary_json` | `raster_adapt_polygon_summary.json` | JSON report filename. |
| `outputs.save_intermediates` | `true` | Save masks, reference ring, fitted surface and blend weights. |

## 8.4 Reference ring

```yaml
reference_ring:
  outer_buffer_px: 20
  inner_buffer_px: 0
  min_reference_pixels: 500
```

| Key | Type | Default | Validation/meaning |
| --- | --- | --- | --- |
| `outer_buffer_px` | integer | `20` | Outer radius of reference zone in pixels. Must be greater than `inner_buffer_px`. |
| `inner_buffer_px` | integer | `0` | Excluded exterior band next to polygon. Must be >= 0. |
| `min_reference_pixels` | integer | `500` | Minimum valid reference samples; must be > 0. |

Reference cells are outside the modification polygon. With an inner value of 10 and
outer value of 50, cells 0-10 pixels from the polygon are excluded and cells 10-50
pixels away are candidates.

## 8.5 Adaptation methods

Supported values:

```text
boundary_idw
nearest_boundary
plane_fit
polynomial_fit
```

Canonical defaults:

```yaml
adaptation:
  method: "boundary_idw"
  idw_power: 2.0
  k_nearest: 32
  max_search_distance_px: null
  max_reference_points: 10000
  random_seed: 42
  polynomial_order: 2
```

### Shared parameters

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `adaptation.method` | string | `boundary_idw` | One of the four supported methods. |
| `adaptation.max_reference_points` | integer | `10000` | Maximum reference samples. Values <= 0 effectively disable subsampling. |
| `adaptation.random_seed` | integer/null | `42` | Seed used for reference subsampling. |

### `boundary_idw`

| Key | Type | Default | Validation/meaning |
| --- | --- | --- | --- |
| `idw_power` | number | `2.0` | Must be > 0. Higher values increase influence of nearby samples. |
| `k_nearest` | integer | `32` | Must be > 0. Number of neighbours requested per target cell. |
| `max_search_distance_px` | number/null | `null` | Optional positive radius in pixels. Converted to map units using mean pixel size. |

If no reference point exists inside the configured maximum distance for a target
cell, the current implementation falls back to unrestricted nearest-neighbour IDW
for that cell.

### `nearest_boundary`

Uses the closest reference-ring sample for each target cell. IDW-specific parameters
remain accepted in the resolved mapping but are not used by this method.

### `plane_fit`

Fits a first-order least-squares surface:

```text
z = a + b*x + c*y
```

`polynomial_order` is not used to change `plane_fit`; this method always fits order 1.

### `polynomial_fit`

Uses `polynomial_order`:

| Key | Type | Default | Validation |
| --- | --- | --- | --- |
| `polynomial_order` | integer | `2` | Must be `1` or `2`. |

Order 1 is planar. Order 2 includes squared and cross terms. Coordinates are centred
and scaled internally before least-squares fitting.

## 8.6 Border blending

```yaml
border_blending:
  enabled: true
  blend_width_px: 5
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Blend original and adapted surfaces near the inside edge. |
| `blend_width_px` | integer | `5` | Non-negative width in pixels. |

When disabled, the adapted surface replaces the original throughout the polygon.
A large blend width can preserve part of the original anomaly near the border.

## 8.7 NoData

```yaml
nodata:
  preserve_nodata: true
```

When true, input NoData cells remain NoData.

## 8.8 Full example

```yaml
pipeline: "raster_adapt_polygon"

raster_adapt_polygon:
  name: "adapt_demo"
  outdir: "outputs/adapt_demo"
  excel: true

  raster: "D:/data/input_dem.tif"
  modify_polygon: "D:/data/modify_polygon.shp"

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

## 8.9 Outputs

```text
<outdir>/rasters/<adapted_raster>
<outdir>/rasters/<name>_adapted_surface.tif
<outdir>/rasters/<name>_modify_mask.tif
<outdir>/rasters/<name>_reference_ring.tif
<outdir>/rasters/<name>_blend_weights.tif
<outdir>/report/<excel_report>
<outdir>/metadata/<summary_json>
```

Intermediate rasters are written only when `save_intermediates` is true.

---

# 9. Count features by sector

```yaml
pipeline: "count_features_by_sector"
```

Dedicated runner:

```bash
python -m scripts.run_sector_counts --config config/count_features_by_sector_example.yml
```

Detailed guide: `docs/count_features_by_sector.md`.

## 9.1 Required parameters

| Key | Type | Meaning |
| --- | --- | --- |
| `sectors` | path | Polygon zones/sectors dataset. |
| `sector_id_field` | string | Existing unique, non-null identifier field in sectors. |
| `layers` | list | Non-empty list of input point/polygon layer specifications. |
| `outdir` | path | Output directory. |

## 9.2 Optional parameters and defaults

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | `feature_counts_by_sector` | Output filename prefix. Must not be empty. |
| `sectors_layer` | string/null | `null` | Layer name for multi-layer sources such as GeoPackage. |
| `point_predicate` | string | `within` | Default spatial predicate for Point/MultiPoint inputs. |
| `polygon_predicate` | string | `intersects` | Default predicate for non-point geometries. |
| `write_csv` | boolean | `true` | Write count table CSV. |
| `write_excel` | boolean | `true` | Write Excel workbook. |
| `write_geopackage` | boolean | `false` | Write sectors layer with joined count columns. |

The current resolver normalizes predicates to lowercase but does not maintain a
repository-specific allow-list; the value must be supported by the installed
GeoPandas spatial join implementation.

## 9.3 Layer specifications

A layer may be written as a path:

```yaml
layers:
  - "D:/data/trees.shp"
```

or as a mapping:

```yaml
layers:
  - path: "D:/data/trees.gpkg"
    layer: trees
    name: trees
    predicate: covered_by
```

| Layer key | Required | Default/meaning |
| --- | --- | --- |
| `path` | yes | Input vector dataset. Must exist. |
| `name` | no | Defaults to file stem; must be unique across configured layers. |
| `layer` | no | Layer name for GeoPackage/multi-layer source. |
| `predicate` | no | Overrides geometry-specific default for this input. |

Point-like geometries use `point_predicate`; all other geometries use
`polygon_predicate`, unless overridden per layer.

## 9.4 Full example

```yaml
pipeline: "count_features_by_sector"

count_features_by_sector:
  name: "sector_feature_counts"
  sectors: "D:/data/sectors.shp"
  sectors_layer: null
  sector_id_field: "sector"
  outdir: "outputs/sector_feature_counts"

  point_predicate: "within"
  polygon_predicate: "intersects"

  layers:
    - path: "D:/data/trees.shp"
      name: "trees"

    - path: "D:/data/planting_areas.gpkg"
      layer: "planting_areas"
      name: "planting_areas"
      predicate: "within"

  write_csv: true
  write_excel: true
  write_geopackage: true
```

## 9.5 Validation and semantics

- Sectors and input layers must have defined CRS values.
- Inputs are reprojected to the sectors CRS when necessary.
- `sector_id_field` must exist, contain no nulls and be unique.
- Empty/invalid geometries are excluded.
- A polygon using `intersects` can count in more than one sector.
- A point using `within` exactly on a boundary is not counted; use `covered_by` to include boundary points.

## 9.6 Outputs

```text
<outdir>/<name>.csv
<outdir>/<name>.xlsx
<outdir>/<name>.gpkg   # when write_geopackage is true
```

The GeoPackage layer is named `sector_counts`.

---

# 10. Spatial join attributes

```yaml
pipeline: "spatial_join_attributes"
```

Dedicated runner:

```bash
python -m scripts.run_spatial_join_attributes --config config/spatial_join_attributes_example.yml
```

Detailed guide: `docs/spatial_join_attributes.md`.

## 10.1 Required parameters

| Key | Type | Meaning |
| --- | --- | --- |
| `zones.path` | path | Polygon/MultiPolygon zones dataset. |
| `zones.id_field` | string | Zone identifier field. |
| `input.path` | path | Input feature dataset to assign to zones. |
| `outdir` | path | Output directory; created if missing. |

## 10.2 Zone and input mappings

```yaml
zones:
  path: "D:/data/sectors.gpkg"
  layer: "sectors"
  id_field: "sector_id"
  fields:
    - sector_id
    - sector_name

input:
  path: "D:/data/network.gpkg"
  layer: "junctions"
```

| Key | Required | Default/meaning |
| --- | --- | --- |
| `zones.layer` | no | `null`; explicit layer for multi-layer source. |
| `zones.fields` | no | `null`; when omitted, all non-geometry zone fields are copied. The ID field is automatically included if absent from the list. |
| `input.layer` | no | `null`; explicit layer for multi-layer source. |

## 10.3 Optional parameters and defaults

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | `spatial_join_attributes` | Output filename prefix. |
| `predicate` | string | `within` | Spatial relation used by GeoPandas join. |
| `join_type` | string | `left` | `left` or `inner`. |
| `zone_field_prefix` | string | `zone_` | Prefix added to copied zone fields. |
| `write_csv` | boolean | `true` | Write joined CSV. |
| `write_excel` | boolean | `true` | Write workbook with joined data and diagnostics. |
| `write_geopackage` | boolean | `true` | Write spatial result. |
| `write_unmatched` | boolean | `true` | Write unmatched features when present. |
| `include_geometry_wkt_in_tables` | boolean | `false` | Add WKT geometry to CSV/Excel tables. |

Allowed predicates:

```text
contains
covered_by
covers
crosses
intersects
overlaps
touches
within
```

## 10.4 Join semantics

### `join_type: left`

Keeps every input feature. Unmatched rows have null zone fields.

### `join_type: inner`

Keeps only matched spatial relationships.

A single input may produce multiple rows if it matches more than one zone.

## 10.5 CRS and fields

- Both datasets must have a CRS.
- Input features are reprojected to the zones CRS when different.
- Zones must contain only Polygon/MultiPolygon geometries.
- All original input attributes are preserved.
- The workflow adds `source_feature_id` from the original input index.
- Copied zone fields are renamed with `zone_field_prefix` to avoid collisions.

## 10.6 Full example

```yaml
pipeline: "spatial_join_attributes"

spatial_join_attributes:
  name: "features_by_zone"
  outdir: "outputs/features_by_zone"

  zones:
    path: "D:/data/sectors.shp"
    layer: null
    id_field: "sector_id"
    fields:
      - sector_id
      - sector_name

  input:
    path: "D:/data/network.gpkg"
    layer: "junctions"

  predicate: "covered_by"
  join_type: "left"
  zone_field_prefix: "zone_"

  write_csv: true
  write_excel: true
  write_geopackage: true
  write_unmatched: true
  include_geometry_wkt_in_tables: false
```

## 10.7 Outputs

Depending on enabled options:

```text
<outdir>/<name>.csv
<outdir>/<name>_unmatched.csv
<outdir>/<name>.xlsx
<outdir>/<name>.gpkg
```

The GeoPackage contains the joined spatial layer and, when enabled and non-empty,
an unmatched layer. See the workflow-specific guide for workbook sheet names and
validation recommendations.

---

# 11. Parameters accepted but not fully active

The following keys are accepted by current resolvers but have limited or no effect
in the corresponding execution path:

| Workflow | Key | Current status |
| --- | --- | --- |
| sample points | `mask_polygon` | Normalized but not applied by `run_sample_points_from_raster_value_range`. |
| sample points | `qgis_assets` | Normalized but not used by the current sample execution function. |
| LandXML | `rasterize.all_touched` | Accepted; current V1 still uses centre-based interpolation and does not expand partially touched cells. |
| LandXML | `options.preserve_original_point_ids` | Present in config; current CSV output includes original IDs regardless. |

These notes are intentionally explicit so the reference does not imply behaviour
that the current code does not implement.

---

# 12. Validation checklist before running

1. Run commands from the repository root.
2. Activate the intended environment:

   ```bash
   conda activate spatial_analysis
   ```

3. Check YAML syntax and Windows quoting.
4. Confirm input paths exist.
5. For GeoPackages, specify `layer` when more than one layer exists.
6. Confirm CRS definitions in all spatial inputs.
7. Confirm field names exactly match source schemas.
8. Use unique `name`/`outdir` values when comparing parameter variants.
9. Review generated Excel/JSON reports and intermediate rasters/layers.
10. Validate final outputs in QGIS before engineering use.

---

# 13. Related documentation

```text
docs/documentation_index.md          central documentation map
docs/quick_start_guide.md            beginner workflow and commands
docs/configuration_conventions.md    YAML and Windows path conventions
docs/polygon_mosaic_advanced.md      detailed mosaic behaviour
docs/raster_adapt_polygon.md         detailed terrain adaptation guide
docs/count_features_by_sector.md     sector counting workflow
docs/spatial_join_attributes.md      feature-to-zone attribute workflow
docs/troubleshooting.md              common errors and fixes
```
