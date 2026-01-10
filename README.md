# Spatial Analysis Toolkit

Python-based tools for geospatial and raster analysis,
designed to integrate with QGIS and reproducible workflows.

## Installation

Create the conda environment and install dependencies:

```bash
conda env create -f environment.yml
conda activate spatial_analysis
```

Optionally install the pip requirements (for pip-only installs or verification):

```bash
pip install -r requirements.txt
```

Quick import test:

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

Usage

Run the raster comparison script to align inputs to raster1 and compute dz products:

```bash
python scripts/compare_rasters.py \
  --raster1 data/dem_2020.tif \
  --raster2 data/dem_2022.tif \
  --outdir outputs \
  --name demo_run \
  --resampling bilinear
```

The signed dz raster is computed as raster2 - raster1 (positive means raster2 is higher), and
abs_dz is the magnitude of the difference.

QGIS Integration


```bash
python -m scripts.compare_rasters --raster1 "path/to/dem1.tif" --raster2 "path/to/dem2.tif" --outdir outputs/test --name test --excel

```

Load the dz and abs_dz rasters in QGIS, then apply styles via Layer Properties → Symbology → Style → Load Style… and select the QML files copied into the output folder (outdir/qgis).

If you enable the exceedance vector output, add the GeoJSON layer from outdir/vectors for a polygon overlay of |dz| > threshold.



---

# 4) Verificación inmediata (antes de commit)
```bat
python -m scripts.compare_rasters --help
```

## Quickstart

CLI example:

```bash
python scripts/compare_rasters.py --raster1 data/dem_2020.tif --raster2 data/dem_2022.tif --outdir outputs --name demo_run --resampling bilinear --excel --qgis-assets --vector-threshold 0.5
```

Config-based example (YAML-driven pipelines):

✅ Caso que SÍ funciona (el correcto)
```bash
python -m scripts.run_from_config
```
Python dice:
"Mi mundo empieza en spatial_analysis/"

```bash
python scripts/run_from_config.py --config config/example_config.yml
```

Explicación de la configuración: 

```bash
raster1: "path/to/raster1.tif"
➡ Baseline raster (reference DEM)
raster2: "path/to/raster2.tif"
➡ Raster to compare against raster1
outdir: "outputs"
➡ Folder where all results will be written
name: "demo_run"
➡ Prefix for output files (keeps runs organized)
resampling: "bilinear"
➡ Resampling method when aligning rasters
nearest → categorical rasters
bilinear → DEMs (recommended)
cubic → smoother, slower
excel: true
➡ Generate Excel summary report (*_Comparison_Report.xlsx)
thresholds: [0.10, 0.25, 0.50, 1.00]
➡ Thresholds (in raster units) used for:
Excel “area by change magnitude” table
Interpreting |dz|
bins: 60
➡ Number of bins for dz histogram in Excel
qgis_assets: true
➡ Copy .qml styles into output folder for 1-click QGIS styling
vector_threshold: 0.5
➡ Create polygons where |dz| > 0.5
➡ Output: GeoJSON for QGIS overlay
➡ Set to null to disable

```

### Pipeline selector (single YAML per project)

Use `pipeline` to choose which workflow to run:

Legacy raster diff (no pipeline key, backward compatible):

```yaml
raster1: "path/to/raster1.tif"
raster2: "path/to/raster2.tif"
outdir: "outputs"
name: "demo_run"
resampling: "bilinear"
excel: true
thresholds: [0.10, 0.25, 0.50, 1.00]
bins: 60
qgis_assets: true
vector_threshold: 0.5
signed_vector_threshold: 0.5
```

Explicit raster diff pipeline (new style):

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

Polygon mosaic pipeline:

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
## Outputs

The signed dz raster is computed as raster2 - raster1 (positive means raster2 is higher), and
abs_dz is the magnitude of the difference.

dz > 0 → raster2 is higher than raster1 (fill / increase)
dz < 0 → raster2 is lower than raster1 (cut / decrease)

The workflow creates the following folders and files under the output directory:

- `aligned/`: aligned inputs (`*_raster1_aligned.tif`, `*_raster2_aligned.tif`)
- `rasters/`: difference rasters (`*_dz.tif`, `*_abs_dz.tif`)
- `report/`: Excel report (`*_Comparison_Report.xlsx`) when `excel: true`, plus alignment reports (`*_alignment_report.json`, `*_alignment_report.csv`)
- `vectors/`: exceedance polygons (`*_abs_dz_gt_<threshold>.geojson`) when `vector_threshold` is set
- `qgis/`: QML styles copied when `qgis_assets: true`

Estructura del output (eligiendo --outdir outputs/run1 y --name run1): 
```bash
´outputs/run1/
├─ aligned/
│  ├─ run1_raster1_aligned.tif
│  └─ run1_raster2_aligned.tif
├─ rasters/
│  ├─ run1_dz.tif
│  └─ run1_abs_dz.tif
├─ report/
│  ├─ run1_Comparison_Report.xlsx
│  ├─ run1_alignment_report.json
│  └─ run1_alignment_report.csv
├─ qgis/
│  ├─ dz_diverging.qml
│  └─ abs_dz_thresholds.qml
└─ vectors/
   └─ run1_abs_dz_gt_0.5.geojson
```

## QGIS recommended layer order

1. hillshade (optional) bottom
2. raster1 (optional)
3. dz (with dz style)
4. abs_dz (optional)
5. exceedance polygons (top)

## Notes / pitfalls

- Ensure rasters are comparable (same vertical datum/units).
- Resampling choice matters: use `nearest` for categorical inputs, `bilinear` for continuous surfaces.
- If the CRS is geographic (degrees), polygon areas are in degrees².
