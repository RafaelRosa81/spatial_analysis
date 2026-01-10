# Troubleshooting

## Module import errors

**Symptoms:** `ModuleNotFoundError` for rasterio, fiona, or shapely.

**Fixes:**
- Use the conda environment (`environment.yml`) for the full geospatial stack.
- Verify you activated the environment before running commands.

## CRS mismatch or unexpected reprojection

**Symptoms:** Alignment report shows CRS changes or large grid shifts.

**Fixes:**
- Confirm that both rasters use the same CRS and vertical datum.
- Inspect `outputs/<name>/report/*alignment_report.json` for CRS and grid shift
  metrics.
- If the CRS is geographic (degrees), area calculations are not in m².

## Nodata surprises

**Symptoms:** Large masked areas or unexpected gaps in dz.

**Fixes:**
- Ensure input rasters have a correct `nodata` value.
- Verify raster nodata in QGIS (Layer Properties → Information).
- Remember: nodata is propagated when either raster has nodata.

## Masked/transparent areas

**Symptoms:** Output appears transparent in QGIS.

**Fixes:**
- The outputs are GeoTIFFs with nodata values (no alpha band).
- QGIS renders nodata as transparency; check the layer’s nodata value.

## File path issues on Windows

**Symptoms:** File not found errors even though the file exists.

**Fixes:**
- Use forward slashes or double backslashes in YAML.
- Wrap paths in quotes.
- Use absolute paths if relative paths are ambiguous.

## Sample points: no points found in value range

**Symptoms:** Error like `No pixels found in range [value_min, value_max]`.

**Fixes:**
- Verify the raster value distribution (e.g., inspect with QGIS).
- Widen the range or check if nodata values dominate the raster.
- Confirm `value_min <= value_max`.

## Sample points: spacing looks wrong or too sparse

**Symptoms:** Regular sampling returns too few points or unexpected spacing.

**Fixes:**
- Ensure the raster CRS uses projected units if you expect meters.
- Remember that `sampling.spacing` is interpreted in **map units**, so degrees
  will produce very large spacing on geographic CRSs.

## Sample points: nodata filtering removes everything

**Symptoms:** No points are returned when `nodata_is_invalid: true`.

**Fixes:**
- Confirm the raster has a valid nodata value set.
- If nodata is unreliable, set `nodata_is_invalid: false` and re-run.

## Sample points: mask polygon not intersecting raster

**Symptoms:** No points found when using `mask_polygon`.

**Fixes:**
- Note: polygon masking is currently supported in the standalone CLI
  (`scripts/sample_points_from_raster.py`) but not applied in the YAML pipeline.
- Confirm the polygon overlaps the raster extent.
- Reproject the polygon to match the raster CRS before use.

## Sample points: output directory permission errors

**Symptoms:** `PermissionError` or `OSError` when writing outputs.

**Fixes:**
- Choose an output directory with write permissions.
- Avoid syncing folders that lock files (e.g., cloud sync).

## How to verify alignment correctness

- Compare `outputs/<name>/aligned/*_aligned.tif` against the reference raster in
  QGIS and confirm pixel alignment.
- Inspect `alignment_report.json` for CRS changes and grid origin shifts.
- For small samples, compute spot-check differences at known locations.
