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



---

# 4) Verificación inmediata (antes de commit)
```bat
python -m scripts.compare_rasters --help
