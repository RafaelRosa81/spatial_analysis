# Documentation Index

This is the central navigation page for the `spatial_analysis` repository.

Use it to identify:

- which workflow solves your problem;
- whether it uses the integrated YAML runner or a dedicated command;
- where its parameters are documented;
- which example configuration to copy;
- which outputs and checks to expect.

---

## 1. Recommended reading order

### New users

```text
README.md
↓
docs/quick_start_guide.md
↓
docs/configuration_conventions.md
↓
docs/config_reference.md or workflow-specific documentation
```

### Users preparing real projects

```text
docs/quick_start_guide.md
↓
example YAML in config/
↓
workflow-specific documentation
↓
docs/troubleshooting.md
```

### Developers

```text
docs/architecture.md
↓
docs/pipeline_overview.md
↓
processing modules and runner scripts
```

---

## 2. Essential documents

### `README.md`

Repository overview, installation, basic Git commands, execution model, available workflows, and first-run commands.

### `docs/quick_start_guide.md`

Beginner-oriented guide with commands and concise examples for every current workflow.

### `docs/configuration_conventions.md`

Repository-wide conventions for YAML, Windows paths, relative paths, GeoPackage layers, field names, booleans, and null values.

### `docs/config_reference.md`

Canonical reference for workflows integrated into:

```bash
python -m scripts.run_from_config --config <config.yml>
```

### `docs/troubleshooting.md`

Common failures involving YAML, CRS, NoData, vector layers, raster alignment, and output interpretation.

---

## 3. Workflow catalog

| Workflow | Main purpose | Runner | Main documentation |
| --- | --- | --- | --- |
| `raster_diff` | Compare two rasters. | `scripts.run_from_config` | `docs/config_reference.md` |
| `polygon_mosaic` | Merge rasters using polygon selection, adjustment, and blending. | `scripts.run_from_config` | `docs/polygon_mosaic_advanced.md` |
| `raster_adapt_polygon` | Reconstruct terrain inside a polygon. | `scripts.run_from_config` | `docs/raster_adapt_polygon.md` |
| `sample_points_from_raster_value_range` | Sample cells inside a raster value interval. | `scripts.run_from_config` | `docs/config_reference.md` |
| `landxml_tin_to_mesh` | Convert LandXML TIN to mesh and optional DEM. | `scripts.run_from_config` | `docs/quick_start_guide.md`, `docs/config_reference.md` |
| `count_features_by_sector` | Count vector features by polygon sector. | `scripts.run_sector_counts` | `docs/count_features_by_sector.md` |
| `spatial_join_attributes` | Assign zone attributes to input features. | `scripts.run_spatial_join_attributes` | `docs/spatial_join_attributes.md` |

---

## 4. Integrated YAML-runner workflows

Run with:

```bash
python -m scripts.run_from_config --config config/your_config.yml
```

Current integrated selector values:

```text
raster_diff
polygon_mosaic
sample_points_from_raster_value_range
landxml_tin_to_mesh
raster_adapt_polygon
```

### Raster Diff

```yaml
pipeline: "raster_diff"
```

Purpose:

- align two rasters;
- calculate signed and absolute elevation differences;
- report statistics and thresholds;
- optionally generate QGIS styles and exceedance polygons.

Primary documents:

- `README.md`
- `docs/quick_start_guide.md`
- `docs/config_reference.md`

Example configs:

- `config/minimal_raster_diff_example.yml`
- `config/full_raster_diff_example.yml`

Typical outputs:

```text
dz.tif
abs_dz.tif
alignment reports
Excel report
optional GeoJSON polygons
```

---

### Polygon Mosaic

```yaml
pipeline: "polygon_mosaic"
```

Purpose:

- combine two rasters;
- define which raster is used inside/outside a polygon;
- vertically adjust one raster to another;
- blend borders;
- control output extent and masking.

Primary documents:

- `docs/polygon_mosaic_advanced.md`
- `docs/quick_start_guide.md`
- `docs/config_reference.md`

Example config:

- `config/polygon_mosaic_example.yml`

Typical outputs:

```text
mosaic raster
aligned raster
adjusted raster
blend weights
overlap diagnostics
Excel report
```

---

### Raster Adapt Polygon

```yaml
pipeline: "raster_adapt_polygon"
```

Purpose:

- replace or reconstruct topography inside a polygon;
- use surrounding terrain as reference;
- compare alternative interpolation/fitting criteria;
- preserve the original raster outside the polygon.

Supported methods:

```text
boundary_idw
nearest_boundary
plane_fit
polynomial_fit
```

Primary documents:

- `docs/raster_adapt_polygon.md`
- `docs/quick_start_guide.md`
- `docs/config_reference.md`

Example config:

- `config/raster_adapt_polygon_example.yml`

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

---

### Sample Points From Raster Value Range

```yaml
pipeline: "sample_points_from_raster_value_range"
```

Alias:

```text
sample_points
```

Purpose:

- filter raster cells by an inclusive value range;
- sample qualifying cells randomly or regularly;
- export CSV and GeoPackage points.

Primary documents:

- `docs/quick_start_guide.md`
- `docs/config_reference.md`

Example config:

- `config/sample_points_example.yml`

---

### LandXML TIN To Mesh

```yaml
pipeline: "landxml_tin_to_mesh"
```

Purpose:

- read LandXML TIN points and faces;
- preserve original triangulation;
- export CSV, OBJ, and PLY products;
- optionally rasterize the TIN directly to a DEM.

Primary documents:

- `docs/quick_start_guide.md`
- `docs/config_reference.md`
- `README.md`

Example config:

- `config/landxml_tin_to_mesh_example.yml`

---

## 5. Dedicated vector workflows

These workflows are present in the repository but currently use dedicated runners rather than `scripts.run_from_config`.

### Count Features By Sector

```yaml
pipeline: count_features_by_sector
```

Run with:

```bash
python -m scripts.run_sector_counts --config config/count_features_by_sector_example.yml
```

Purpose:

- count point and polygon features per sector polygon;
- use configurable spatial predicates;
- export CSV, Excel, and optional GeoPackage results.

Primary document:

- `docs/count_features_by_sector.md`

Example config:

- `config/count_features_by_sector_example.yml`

---

### Spatial Join Attributes

```yaml
pipeline: spatial_join_attributes
```

Run with:

```bash
python -m scripts.run_spatial_join_attributes --config config/spatial_join_attributes_example.yml
```

Purpose:

- assign polygon-zone attributes to input features;
- preserve all source attributes;
- identify unmatched or multiply matched features;
- export CSV, Excel, and GeoPackage results.

Primary documents:

- `docs/spatial_join_attributes.md`
- `docs/configuration_conventions.md`

Example config:

- `config/spatial_join_attributes_example.yml`

---

## 6. Architecture and maintenance

### `docs/pipeline_overview.md`

Explains the configuration-driven execution model and expected pipeline lifecycle.

### `docs/architecture.md`

Explains package organization and implementation structure.

### `docs/epanet_gis_architecture.md`

Proposed architecture and roadmap for GIS-to-EPANET tooling. This document is currently maintained in an open draft pull request and is not yet part of `main`.

---

## 7. Sanity checks

From the repository root:

```bash
conda activate spatial_analysis
python -m scripts.run_from_config --help
python scripts/polygon_mosaic_sanity.py
python -m scripts.landxml_tin_sanity
python -m scripts.raster_adapt_polygon_sanity
```

The dedicated vector workflows do not currently have equivalent sanity commands listed here; validate them with their example configurations and small test datasets.

---

## 8. Repository consolidation notes

At the time of this index update:

- all implemented workflows listed above are present on `main`;
- `raster_adapt_polygon` and its documentation are included on `main`;
- `count_features_by_sector` and `spatial_join_attributes` are implemented and documented;
- one draft pull request remains open for the future GIS-to-EPANET architecture;
- some historical remote branches may remain after their pull requests were merged, but their changes are already represented in `main`.
