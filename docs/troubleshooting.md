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

## How to verify alignment correctness

- Compare `outputs/<name>/aligned/*_aligned.tif` against the reference raster in
  QGIS and confirm pixel alignment.
- Inspect `alignment_report.json` for CRS changes and grid origin shifts.
- For small samples, compute spot-check differences at known locations.
