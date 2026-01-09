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

QGIS Integration
