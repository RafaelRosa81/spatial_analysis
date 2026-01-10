# Spatial Analysis Toolkit

Python-based tools for geospatial and raster analysis, designed to integrate with
QGIS and support **reproducible, configuration-driven workflows**.

This toolkit focuses on **raster-to-raster comparison** (e.g. DEM differencing)
and related post-processing such as threshold analysis, vectorization, reporting,
and polygon-based mosaicking.

---

## Conceptual overview

The core quantity computed by this toolkit is:

**dz = raster2 − raster1**

- `dz > 0` → raster2 is higher than raster1 (fill / increase)
- `dz < 0` → raster2 is lower than raster1 (cut / decrease)
- `abs_dz = |dz|` → magnitude of change, ignoring sign

Both signed (`dz`) and absolute (`abs_dz`) products are generated and used for
interpretation, reporting, and visualization.

Typical use cases include:
- terrain change detection,
- cut/fill analysis,
- validation between repeated surveys,
- controlled replacement or blending of rasters within polygons.

---

## Installation

### Conda (recommended)

Create the conda environment and install dependencies:

Use environment.yml; requirements.txt is minimal runtime only.
```bash
conda env create -f environment.yml
conda activate spatial_analysis
```

### Pip-only (optional / verification)

```bash
pip install -r requirements.txt
```

### Quick import test

```bash
python - <<'PY'
import rasterio
import numpy
import pandas
import openpyxl
import shapely
import yaml
import tqdm

print("Imports OK")
PY
```

---

## Entrypoints and usage

### 1) Legacy CLI (direct raster comparison)

Run the raster comparison script to align inputs to `raster1` and compute dz products:

```bash
python scripts/compare_rasters.py \
  --raster1 data/dem_2020.tif \
  --raster2 data/dem_2022.tif \
  --outdir outputs \
  --name demo_run \
  --resampling bilinear
```

This interface is still available for ad hoc runs, but **new users are encouraged
to use the YAML-driven pipeline runner** for reproducibility.

---

### 2) YAML-driven pipelines (recommended)

All reproducible workflows should use:

```bash
python -m scripts.run_from_config --config <config.yml>
```

This runner executes a pipeline defined entirely in a YAML file, capturing:
- inputs,
- processing choices,
- thresholds,
- and outputs.

---

## Configuration explained (raster diff)

Minimal example (legacy style, backward compatible):

```yaml
raster1: "path/to/raster1.tif"        # Baseline raster (reference DEM)
raster2: "path/to/raster2.tif"        # Raster compared against raster1
outdir: "outputs"                     # Root output directory
name: "demo_run"                      # Run name / prefix
resampling: "bilinear"                # nearest | bilinear | cubic
excel: true                           # Generate Excel summary report
thresholds: [0.10, 0.25, 0.50, 1.00]  # Thresholds for reporting and interpretation
bins: 60                              # Histogram bins for dz
qgis_assets: true                     # Copy QGIS styles (.qml)
vector_threshold: 0.5                 # |dz| exceedance polygons (GeoJSON)
signed_vector_threshold: 0.5          # Signed dz vectorization (optional)
```

Notes:
- Use `nearest` resampling for categorical rasters.
- Use `bilinear` (recommended) for continuous surfaces (DEMs).
- Set vector thresholds to `null` to disable vector outputs.

---

## Pipeline selector (single YAML per project)

A single YAML file can define **multiple workflows**, selected via `pipeline`.

### Explicit raster diff pipeline (new style)

```yaml
pipeline: "raster_diff"
name: "demo_run"
outdir: "outputs"
excel: true
resampling: "bilinear"

raster_diff:
  raster1: "path/to/raster1.tif"
  raster2: "path/to/raster2.tif"
  thresholds: [0.10, 0.25, 0.50, 1.00]
  bins: 60
  qgis_assets: true
  vector_threshold: 0.5
  signed_vector_threshold: null
```

### Polygon mosaic pipeline

This workflow replaces or blends raster values **inside a polygon** using a
secondary raster, with optional vertical adjustment and border blending.

```yaml
pipeline: "polygon_mosaic"
name: "mosaic_run"
outdir: "outputs"
excel: true
resampling: "bilinear"

polygon_mosaic:
  raster1: "path/to/raster1.tif"
  raster2: "path/to/raster2.tif"
  polygon: "path/to/footprint.geojson"

  outputs:
    new_raster: "new_raster.tif"
    excel_report: "polygon_mosaic_report.xlsx"
    save_intermediates: true

  vertical_adjustment:
    enabled: true
    mad_threshold: 0.10
    min_overlap_pixels: 50000

  border_blending:
    enabled: true
    blend_width_px: 5
```

---

## Outputs

For a run with:

```text
outdir = outputs/run1
name   = run1
```

the workflow generates:

- `aligned/`
  - `run1_raster1_aligned.tif`
  - `run1_raster2_aligned.tif`
- `rasters/`
  - `run1_dz.tif`
  - `run1_abs_dz.tif`
- `report/`
  - `run1_Comparison_Report.xlsx`
  - `run1_alignment_report.json`
  - `run1_alignment_report.csv`
- `vectors/`
  - `run1_abs_dz_gt_<threshold>.geojson`
- `qgis/`
  - QML styles for dz and abs_dz visualization

---

## QGIS integration

1. Load `dz` and `abs_dz` rasters.
2. Apply styles from `outdir/qgis/` via:
   *Layer Properties → Symbology → Style → Load Style…*
3. Optional layer order:
   1. Hillshade (bottom)
   2. Raster1
   3. dz
   4. abs_dz
   5. Exceedance polygons (top)

---

## Notes & pitfalls

- Ensure rasters share **compatible vertical datum and units**.
- Resampling choice affects results; choose deliberately.
- If CRS is geographic (degrees), polygon areas are in degrees².
- Transparency usually indicates *masked / nodata* areas, not zero values.

---

## Further documentation

Extended explanations and internal details are available under `docs/`:

- `docs/pipeline_overview.md`
- `docs/config_reference.md`
- `docs/architecture.md`
- `docs/troubleshooting.md`
