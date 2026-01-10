# Pipeline Overview

This pipeline compares two rasters by aligning them to a shared grid, computing
signed and absolute differences (`dz`), and generating summary reports and
optional QGIS assets.

## Pipeline flow

```mermaid
flowchart LR
  A[Input rasters]
  B[Alignment to raster1 grid]
  C[Compute dz (raster2 - raster1)]
  D[Compute abs(dz)]
  E[Statistics + histogram]
  F[Outputs: GeoTIFFs, Excel, GeoJSON, QGIS styles]

  A --> B --> C --> D --> E --> F
```

## Alignment logic

- **Reference grid:** raster1 is treated as the reference grid.
- **CRS decisions:** raster2 is reprojected to raster1’s CRS when needed.
- **Resampling:** `resampling` controls how raster2 is resampled (`nearest`,
  `bilinear`, or `cubic`). Use `nearest` for categorical rasters and `bilinear`
  for continuous surfaces.
- **Grid matching:** aligned rasters share width, height, CRS, and transform.
- **Nodata propagation:** nodata from each input raster is preserved; the combined
  mask ensures that nodata in either raster results in nodata in the output `dz`.

## Vertical transformations

- **Signed dz:** the pipeline computes `dz = raster2 - raster1`.
  - Positive dz means raster2 is higher (fill/increase).
  - Negative dz means raster2 is lower (cut/decrease).
- **Absolute dz:** `abs(dz)` is computed for magnitude-based analysis.

### Polygon mosaic vertical adjustment

For the `polygon_mosaic` pipeline, an optional vertical adjustment applies a
constant offset to raster2. The offset is the **median** of the overlap `dz`
values when:

- Overlap pixel count ≥ `vertical_adjustment.min_overlap_pixels`
- MAD (median absolute deviation) ≤ `vertical_adjustment.mad_threshold`

This is recorded in the mosaic report under `vertical_adjustment`.

## Borders, masks, and polygons

- **Masks:** nodata values (and NaNs) are treated as invalid/masked pixels.
- **Polygon masking (mosaic):** the polygon footprint is rasterized to create a
  binary mask. Outside the polygon is treated as no contribution from raster2.
- **Masked vs nodata vs zeros:**
  - *Masked* values are excluded from computations and propagated to outputs.
  - *Nodata* values are written to GeoTIFFs using the `nodata` value.
  - *Zero* is a valid data value and is not treated as nodata.
- **Transparency:** outputs are standard GeoTIFFs with nodata values, not alpha
  bands. QGIS interprets nodata as transparency.

## Histogram and statistics

- **Signed dz stats:** summary statistics are computed from valid `dz` pixels.
- **Absolute dz bins:** thresholds are applied to `abs(dz)` to compute counts and
  percentages by magnitude.
- **Histogram binning:** `bins` controls the number of histogram bins for `dz`.

See [config_reference.md](config_reference.md) for configuration options.
