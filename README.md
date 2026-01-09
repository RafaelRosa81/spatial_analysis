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
python scripts/compare_rasters.py \
  --raster1 data/dem_2020.tif \
  --raster2 data/dem_2022.tif \
  --outdir outputs \
  --name demo_run \
  --resampling bilinear \
  --excel \
  --qgis-assets \
  --vector-threshold 0.5
```

Config-based example:

```bash
python scripts/run_from_config.py --config config/example_config.yml
```

## Outputs

The workflow creates the following folders and files under the output directory:

- `aligned/`: aligned inputs (`*_raster1_aligned.tif`, `*_raster2_aligned.tif`)
- `rasters/`: difference rasters (`*_dz.tif`, `*_abs_dz.tif`)
- `report/`: Excel report (`*_Comparison_Report.xlsx`) when `excel: true`
- `vectors/`: exceedance polygons (`*_abs_dz_gt_<threshold>.geojson`) when `vector_threshold` is set
- `qgis/`: QML styles copied when `qgis_assets: true`

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
