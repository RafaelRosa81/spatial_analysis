# Spatial Analysis Toolkit

Python-based tools for geospatial, raster, mesh, and vector analysis, designed to integrate with QGIS and support **reproducible, configuration-driven workflows**.

The repository currently includes workflows for:

- raster comparison;
- polygon-based raster mosaicking;
- terrain reconstruction inside polygons;
- sampling points from raster value ranges;
- LandXML TIN conversion to mesh and DEM;
- counting vector features by sector;
- assigning input attributes to polygon zones.

---

## Start here

For a complete documentation map, open:

```text
docs/documentation_index.md
```

For a practical first run, use:

```text
docs/quick_start_guide.md
```

For the canonical YAML reference, use:

```text
docs/config_reference.md
```

---

## Installation

### Conda (recommended)

From the repository root:

```bash
conda env create -f environment.yml
conda activate spatial_analysis
```

If the environment already exists:

```bash
conda activate spatial_analysis
```

### Pip-only verification

```bash
pip install -r requirements.txt
```

---

## Basic Git commands

Clone the repository:

```bash
git clone https://github.com/RafaelRosa81/spatial_analysis.git
cd spatial_analysis
```

Update the local default branch:

```bash
git checkout main
git pull origin main
```

Inspect repository state:

```bash
git status
git branch --show-current
git log --oneline --decorate --max-count=10
```

---

## Main execution model

Most reproducible workflows use a YAML configuration file:

```bash
python -m scripts.run_from_config --config config/your_config.yml
```

The selected workflow is defined by:

```yaml
pipeline: "pipeline_name"
```

The integrated runner currently supports:

```text
raster_diff
polygon_mosaic
sample_points_from_raster_value_range
landxml_tin_to_mesh
raster_adapt_polygon
```

Two additional vector workflows currently use dedicated runners:

```text
count_features_by_sector
spatial_join_attributes
```

Run help:

```bash
python -m scripts.run_from_config --help
```

---

## Available workflows

| Workflow | Purpose | Command style |
| --- | --- | --- |
| `raster_diff` | Compare two rasters and generate signed/absolute differences. | Integrated YAML runner |
| `polygon_mosaic` | Combine rasters inside/outside polygons with optional vertical adjustment and blending. | Integrated YAML runner |
| `raster_adapt_polygon` | Reconstruct terrain inside a polygon from surrounding terrain. | Integrated YAML runner |
| `sample_points_from_raster_value_range` | Generate sample points from cells inside a raster value range. | Integrated YAML runner |
| `landxml_tin_to_mesh` | Preserve LandXML TIN triangulation and export mesh/optional DEM. | Integrated YAML runner |
| `count_features_by_sector` | Count point/polygon features within polygon sectors. | Dedicated runner |
| `spatial_join_attributes` | Assign polygon-zone attributes to input point/polygon features. | Dedicated runner |

---

## Quick command examples

### Raster comparison

```bash
python -m scripts.run_from_config --config config/minimal_raster_diff_example.yml
```

### Polygon mosaic

```bash
python -m scripts.run_from_config --config config/polygon_mosaic_example.yml
```

### Raster adaptation inside polygon

```bash
python -m scripts.run_from_config --config config/raster_adapt_polygon_example.yml
```

### LandXML TIN to mesh

```bash
python -m scripts.run_from_config --config config/landxml_tin_to_mesh_example.yml
```

### Sample points from raster range

```bash
python -m scripts.run_from_config --config config/sample_points_example.yml
```

### Count features by sector

```bash
python -m scripts.run_sector_counts --config config/count_features_by_sector_example.yml
```

### Spatial join attributes

```bash
python -m scripts.run_spatial_join_attributes --config config/spatial_join_attributes_example.yml
```

---

## Sanity checks

Run these after installation or after major changes:

```bash
python -m scripts.run_from_config --help
python scripts/polygon_mosaic_sanity.py
python -m scripts.landxml_tin_sanity
python -m scripts.raster_adapt_polygon_sanity
```

---

## QGIS integration

Typical validation workflow:

1. Run the selected pipeline.
2. Load final and intermediate GeoTIFF/GeoPackage outputs in QGIS.
3. Verify CRS, NoData, polygon masks, reference rings, blend weights, and difference rasters.
4. Review Excel/JSON reports before accepting outputs.

For raster comparison, the key convention is:

```text
dz = raster2 - raster1
```

Therefore:

- `dz > 0`: raster2 is higher;
- `dz < 0`: raster2 is lower;
- `abs_dz = |dz|`: magnitude of change.

---

## Configuration conventions

For Windows paths, use either forward slashes:

```yaml
path: "D:/data/project/layer.shp"
```

or single-quoted backslash paths:

```yaml
path: 'D:\data\project\layer.shp'
```

Avoid double-quoted Windows paths with unescaped backslashes.

See:

```text
docs/configuration_conventions.md
```

---

## Documentation

- `docs/documentation_index.md`: central navigation index.
- `docs/quick_start_guide.md`: beginner-oriented workflow guide.
- `docs/config_reference.md`: canonical integrated-runner YAML reference.
- `docs/polygon_mosaic_advanced.md`: advanced mosaic configuration.
- `docs/raster_adapt_polygon.md`: full terrain-adaptation guide.
- `docs/count_features_by_sector.md`: sector-count workflow.
- `docs/spatial_join_attributes.md`: spatial attribute assignment workflow.
- `docs/configuration_conventions.md`: YAML and Windows path conventions.
- `docs/pipeline_overview.md`: workflow architecture.
- `docs/architecture.md`: code organization.
- `docs/troubleshooting.md`: common problems and fixes.

---

## Notes and pitfalls

- Ensure compatible CRS, horizontal units, vertical units, and vertical datum.
- Use projected CRS for calculations requiring distances or areas in metres.
- Choose raster resampling deliberately.
- Transparency often indicates NoData, not zero.
- Inspect diagnostic outputs and reports before replacing source data.
- Keep real project paths in local configuration files where appropriate.
