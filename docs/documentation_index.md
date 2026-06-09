# Documentation Index

This document is the central navigation page for the repository documentation.

Use it to understand:

- what the repository can do;
- which pipeline you need;
- where to find configuration examples;
- where to find detailed parameter explanations;
- where to look for implementation details or troubleshooting.

---

# 1. Start here

## README

```text
README.md
```

Purpose:

- project overview;
- installation;
- environment setup;
- basic execution;
- QGIS integration concepts;
- quick examples.

Recommended for:

- first contact with the repository;
- environment installation;
- understanding the overall philosophy.

---

## Quick Start Guide

```text
docs/quick_start_guide.md
```

Purpose:

- fast overview of all pipelines;
- minimal YAML examples;
- quick execution examples;
- practical workflow guidance.

Recommended for:

- users who want to quickly run a workflow;
- understanding which pipeline to use;
- finding example YAML structures.

---

# 2. Canonical configuration reference

## Configuration Reference

```text
docs/config_reference.md
```

Purpose:

- canonical YAML reference;
- parameter descriptions;
- defaults;
- validation rules;
- configuration conventions.

Recommended for:

- writing or editing YAML configs;
- understanding required vs optional keys;
- checking default values.

This should be considered the authoritative configuration reference.

---

# 3. Pipeline-specific documentation

## Raster Diff

Main pipeline:

```text
pipeline: "raster_diff"
```

Primary docs:

- `README.md`
- `docs/quick_start_guide.md`
- `docs/config_reference.md`

Purpose:

- compare two rasters;
- compute signed and absolute differences;
- generate Excel reports;
- generate exceedance polygons;
- prepare QGIS visualization outputs.

Typical outputs:

```text
dz.tif
abs_dz.tif
Excel reports
GeoJSON exceedance polygons
```

---

## Polygon Mosaic

Main pipeline:

```text
pipeline: "polygon_mosaic"
```

Primary docs:

- `docs/polygon_mosaic_advanced.md`
- `docs/quick_start_guide.md`
- `docs/config_reference.md`

Purpose:

- merge/combine rasters;
- select raster inside/outside polygons;
- vertically align rasters;
- smooth borders with blending;
- define custom output extents.

Key concepts:

```text
selection polygon
inside/outside raster selection
vertical adjustment
border blending
extent polygon
output masking
```

Typical outputs:

```text
mosaic raster
adjusted raster
blend weights
overlap diagnostics
Excel report
```

---

## Sample Points From Raster Value Range

Main pipeline:

```text
pipeline: "sample_points_from_raster_value_range"
```

Primary docs:

- `docs/quick_start_guide.md`
- `docs/config_reference.md`

Purpose:

- generate sample points from raster values;
- filter raster cells by value range;
- export CSV/GPKG point layers.

Typical outputs:

```text
CSV points
GeoPackage points
```

---

## LandXML TIN To Mesh

Main pipeline:

```text
pipeline: "landxml_tin_to_mesh"
```

Primary docs:

- `docs/quick_start_guide.md`
- `docs/config_reference.md`
- `README.md`

Purpose:

- read LandXML TIN surfaces;
- preserve triangulation;
- export OBJ/PLY meshes;
- optionally rasterize TIN directly to DEM.

Key concepts:

```text
Pnts -> vertices
Faces -> triangles
TIN preservation
mesh export
optional DEM rasterization
```

Typical outputs:

```text
vertices.csv
faces.csv
OBJ/PLY meshes
DEM GeoTIFF
Excel report
```

---

## Raster Adapt Polygon

Main pipeline:

```text
pipeline: "raster_adapt_polygon"
```

Primary docs:

- `docs/raster_adapt_polygon.md`
- `docs/config_reference.md`
- `docs/quick_start_guide.md`

Purpose:

- reconstruct terrain inside polygons;
- remove or soften artificial terrain;
- interpolate new topography from surrounding terrain;
- generate adapted DEMs.

Key concepts:

```text
modify polygon
reference ring
adapted surface
boundary interpolation
surface fitting
border blending
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
adapted_dem.tif
adapted_surface.tif
reference_ring.tif
blend_weights.tif
Excel diagnostics
JSON summary
```

---

# 4. Repository architecture

## Architecture

```text
docs/architecture.md
```

Purpose:

- explain code organization;
- explain module structure;
- explain pipeline integration;
- explain repository conventions.

Recommended for:

- developers;
- contributors;
- users extending the repository.

---

## Pipeline Overview

```text
docs/pipeline_overview.md
```

Purpose:

- explain the pipeline philosophy;
- explain the configuration-driven execution model;
- explain the relationship between YAML and processing modules.

---

# 5. Troubleshooting

## Troubleshooting

```text
docs/troubleshooting.md
```

Purpose:

- common runtime errors;
- YAML mistakes;
- CRS problems;
- NoData issues;
- polygon/raster alignment problems;
- QGIS interpretation tips.

Recommended for:

- debugging failed runs;
- interpreting unexpected outputs.

---

# 6. Recommended reading order

## New users

Recommended order:

```text
README.md
↓
docs/quick_start_guide.md
↓
docs/config_reference.md
↓
specific pipeline documentation
```

---

## Users preparing real projects

Recommended order:

```text
quick_start_guide.md
↓
config_reference.md
↓
pipeline-specific advanced docs
↓
troubleshooting.md
```

---

## Developers extending the repository

Recommended order:

```text
architecture.md
↓
pipeline_overview.md
↓
config_reference.md
↓
existing pipeline modules
```

---

# 7. Example sanity checks

Useful commands:

```bash
python -m scripts.run_from_config --help
python scripts/polygon_mosaic_sanity.py
python -m scripts.landxml_tin_sanity
python -m scripts.raster_adapt_polygon_sanity
```

These are intended to validate that the environment and pipeline modules are functioning before real project execution.
