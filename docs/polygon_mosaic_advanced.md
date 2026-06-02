# Polygon Mosaic Advanced Configuration

This document explains the `polygon_mosaic` pipeline in detail, with emphasis on the
advanced options added for:

- selecting which raster is used inside and outside a polygon;
- choosing which raster receives the vertical adjustment;
- defining a separate polygon to control the final raster extent.

The pipeline is executed with:

```bash
python -m scripts.run_from_config --config config/your_config.yml
```

and selected in YAML with:

```yaml
pipeline: "polygon_mosaic"
```

---

## 1. Conceptual model

The `polygon_mosaic` pipeline creates a new raster by combining two input rasters:

- `raster1`
- `raster2`

A polygon defines the area where one raster is preferred over the other. The output
can also use an optional second polygon to define the final raster extent.

The pipeline can be configured so that, for example:

```text
inside polygon  -> raster1
outside polygon -> raster2
vertical target -> raster2
```

or using the legacy/default behavior:

```text
inside polygon  -> raster2
outside polygon -> raster1
vertical target -> raster2
```

This separation is important because the raster used inside/outside the polygon is
not necessarily the same raster that should be vertically adjusted.

---

## 2. Full YAML example

```yaml
pipeline: "polygon_mosaic"

name: "mosaic_custom"
outdir: "outputs/mosaic_custom"
resampling: "bilinear"
excel: true

polygon_mosaic:
  raster1: "D:/path/to/raster_inside_polygon.tif"
  raster2: "D:/path/to/raster_outside_polygon.tif"

  # Polygon used only to decide which raster is used inside/outside.
  polygon: "D:/path/to/polygon_mosaic_selection.shp"

  selection:
    inside_polygon: "raster1"
    outside_polygon: "raster2"

  output_grid:
    mode: "extent_polygon"
    reference: "raster2"
    extent_polygon: "D:/path/to/polygon_output_extent.shp"
    crop_to_extent_polygon: true
    mask_to_extent_polygon: true

  outputs:
    new_raster: "mosaic_dem.tif"
    excel_report: "mosaic_report.xlsx"
    save_intermediates: true

  vertical_adjustment:
    enabled: true
    method: "constant_offset"
    robust_stat: "median"
    target: "raster2"
    mad_threshold: 0.25
    min_overlap_pixels: 30000
    exclude_polygon_buffer_px: 5

  border_blending:
    enabled: true
    blend_width_px: 10
    weight_curve: "linear"

  alignment:
    resampling: "bilinear"

  nodata:
    use_raster1_nodata: true
```

---

## 3. Required keys

All required keys are placed under `polygon_mosaic`.

| Key | Type | Description |
| --- | --- | --- |
| `raster1` | path | First input raster. Also the legacy/default reference grid. |
| `raster2` | path | Second input raster. It is aligned to the selected output grid. |
| `polygon` | path | Polygon used to decide inside/outside raster selection. |

The following root-level keys are also commonly used and can be provided at the top
level or under the pipeline section depending on the project YAML style:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | required by runner | Output name prefix. |
| `outdir` | path | required by runner | Root output directory. |
| `excel` | bool | `true` | Enables Excel report generation. |
| `resampling` | string | `bilinear` | Shortcut for `alignment.resampling`. |

---

## 4. `selection`: choosing inside/outside raster values

```yaml
selection:
  inside_polygon: "raster2"
  outside_polygon: "raster1"
```

### Defaults

```yaml
selection:
  inside_polygon: "raster2"
  outside_polygon: "raster1"
```

These defaults preserve the original behavior of the pipeline.

### Allowed values

Both keys accept only:

```text
raster1
raster2
```

They must be different.

### Meaning

| Key | Meaning |
| --- | --- |
| `inside_polygon` | Raster used inside `polygon`. |
| `outside_polygon` | Raster used outside `polygon`. |

Example:

```yaml
selection:
  inside_polygon: "raster1"
  outside_polygon: "raster2"
```

This means:

```text
inside polygon  -> raster1
outside polygon -> raster2
```

This option only controls which raster values are used in the mosaic. It does not
control which raster defines the output grid, and it does not control which raster
is vertically adjusted.

---

## 5. `vertical_adjustment`: estimating and applying a vertical offset

```yaml
vertical_adjustment:
  enabled: true
  method: "constant_offset"
  robust_stat: "median"
  target: "raster2"
  mad_threshold: 0.25
  min_overlap_pixels: 30000
  exclude_polygon_buffer_px: 5
```

### Defaults

```yaml
vertical_adjustment:
  enabled: true
  method: "constant_offset"
  robust_stat: "median"
  target: "raster2"
  mad_threshold: 0.10
  min_overlap_pixels: 50000
  exclude_polygon_buffer_px: 5
```

### Parameters

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enables/disables vertical offset estimation and application. |
| `method` | string | `constant_offset` | Current method. A single constant vertical offset is estimated. |
| `robust_stat` | string | `median` | Current robust statistic. The median difference is used. |
| `target` | string | `raster2` | Raster to vertically adjust. Allowed: `raster1`, `raster2`. |
| `mad_threshold` | number | `0.10` | Maximum acceptable MAD of the overlap differences. |
| `min_overlap_pixels` | integer | `50000` | Minimum number of valid overlap pixels required to apply the offset. |
| `exclude_polygon_buffer_px` | integer | `5` | Pixel buffer near `polygon` boundary excluded from offset estimation. |

### How the offset is computed

If `target: raster2`, the pipeline computes:

```text
dz = raster2 - raster1
offset = median(dz)
raster2_adjusted = raster2 - offset
```

If `offset` is negative, `raster2` is raised. Example:

```text
offset = -0.74
raster2_adjusted = raster2 - (-0.74) = raster2 + 0.74
```

If `target: raster1`, the sign convention is reversed so that the selected target
raster is adjusted toward the other raster.

### Why an offset may not be applied

The Excel report includes `offset_reason`. Common values:

| Reason | Meaning | Typical fix |
| --- | --- | --- |
| `vertical adjustment disabled` | `enabled: false`. | Set `enabled: true`. |
| `overlap pixel count below threshold` | Not enough valid overlap pixels. | Lower `min_overlap_pixels` or increase overlap area. |
| `overlap MAD above threshold` | Difference is too spatially variable for a constant offset. | Increase `mad_threshold` only if justified, or avoid a global offset. |
| `applied constant offset to raster2` | Offset was applied to raster2. | No fix needed. |
| `applied constant offset to raster1` | Offset was applied to raster1. | No fix needed. |

### Practical advice

If the report shows:

```text
overlap_pixel_count = 34594
min_overlap_pixels = 50000
offset_reason = overlap pixel count below threshold
```

then the offset is not applied because `34594 < 50000`. In that case, a reasonable
configuration might be:

```yaml
vertical_adjustment:
  min_overlap_pixels: 30000
```

If the report shows:

```text
overlap_mad = 0.195
mad_threshold = 0.25
offset_applied = true
```

then the MAD criterion was satisfied and the offset was accepted.

---

## 6. `output_grid`: controlling final raster size and grid

The output raster is always a rectangular GeoTIFF grid. A polygon can define the
area of interest, but the underlying raster extent is still a rectangular bounding
box. Pixels outside a polygon can be set to NoData using `mask_to_extent_polygon`.

```yaml
output_grid:
  mode: "reference"
  reference: "raster1"
  extent_polygon: null
  crop_to_extent_polygon: false
  mask_to_extent_polygon: false
```

### Defaults

```yaml
output_grid:
  mode: "reference"
  reference: "raster1"
  extent_polygon: null
  crop_to_extent_polygon: false
  mask_to_extent_polygon: false
```

These defaults preserve the legacy behavior: the final raster uses the grid of
`raster1`.

### Parameters

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | string | `reference` | Either `reference` or `extent_polygon`. |
| `reference` | string | `raster1` | Raster whose CRS, pixel size, and grid alignment are used. Allowed: `raster1`, `raster2`. |
| `extent_polygon` | path/null | `null` | Polygon used to define the final output extent when `mode: extent_polygon`. |
| `crop_to_extent_polygon` | bool | `false` | Kept for YAML readability; `mode: extent_polygon` crops to the polygon bounding box. |
| `mask_to_extent_polygon` | bool | `false` | If `true`, pixels outside `extent_polygon` are set to NoData. |

### `mode: reference`

```yaml
output_grid:
  mode: "reference"
  reference: "raster1"
```

The output uses the full extent and grid of the chosen reference raster.

### `mode: extent_polygon`

```yaml
output_grid:
  mode: "extent_polygon"
  reference: "raster2"
  extent_polygon: "D:/path/to/polygon_output_extent.shp"
  mask_to_extent_polygon: true
```

The output uses:

- CRS from `reference`;
- pixel size from `reference`;
- grid alignment from `reference`;
- rectangular bounding box of `extent_polygon` as output extent;
- optional NoData mask outside the exact polygon geometry.

### Why the output may look rectangular

GeoTIFF rasters are rectangular grids. Even when `extent_polygon` is irregular, the
stored raster extent is the polygon bounding box. To make the visible raster match
the polygon shape, use:

```yaml
mask_to_extent_polygon: true
```

Then pixels outside the polygon geometry should be written as NoData. Depending on
QGIS symbology, NoData may appear transparent or white.

### Choosing `reference`

Use `reference: raster2` when the final raster should inherit the grid of the
outside/base raster:

```yaml
selection:
  inside_polygon: "raster1"
  outside_polygon: "raster2"

output_grid:
  mode: "extent_polygon"
  reference: "raster2"
```

Use `reference: raster1` when `raster1` is the desired output grid.

---

## 7. `border_blending`: smoothing the transition

```yaml
border_blending:
  enabled: true
  blend_width_px: 5
  weight_curve: "linear"
```

### Defaults

```yaml
border_blending:
  enabled: true
  blend_width_px: 5
  weight_curve: "linear"
```

### Parameters

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Enables/disables blending near the `polygon` boundary. |
| `blend_width_px` | integer | `5` | Width of transition band in pixels. |
| `weight_curve` | string | `linear` | Current option. The transition is linear. |

The blending is applied around the `polygon` used for inside/outside selection, not
around `output_grid.extent_polygon`.

Conceptually:

```text
new_raster = outside_raster * (1 - weight) + inside_raster * weight
```

where `weight` ranges from 0 outside the polygon to 1 inside the polygon, over the
specified blend width.

---

## 8. `alignment`: resampling rasters to the output grid

```yaml
alignment:
  resampling: "bilinear"
```

### Default

```yaml
alignment:
  resampling: "bilinear"
```

### Allowed values

Common values:

| Value | Use case |
| --- | --- |
| `nearest` | Categorical rasters/classes. |
| `bilinear` | Continuous DEM/MDT surfaces. Recommended for elevation rasters. |
| `cubic` | Smoother continuous rasters; slower and can introduce overshoot. |

A legacy top-level key is also supported:

```yaml
resampling: "bilinear"
```

When present, it is forwarded to `alignment.resampling` unless the nested key is
explicitly set.

---

## 9. `outputs`: generated file names

```yaml
outputs:
  new_raster: "new_raster.tif"
  excel_report: "polygon_mosaic_report.xlsx"
  save_intermediates: true
```

### Defaults

```yaml
outputs:
  new_raster: "new_raster.tif"
  excel_report: "polygon_mosaic_report.xlsx"
  save_intermediates: true
```

### Parameters

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `new_raster` | string | `new_raster.tif` | Final mosaic raster filename under `<outdir>/rasters/`. |
| `excel_report` | string | `polygon_mosaic_report.xlsx` | Excel report filename under `<outdir>/report/`. |
| `save_intermediates` | bool | `true` | Saves diagnostic rasters such as aligned/adjusted rasters and blend weights. |

### Intermediate outputs

When `save_intermediates: true`, outputs may include:

| Output | Meaning |
| --- | --- |
| `raster2_aligned.tif` | `raster2` resampled/aligned to the output grid. |
| `raster2_adjusted.tif` | `raster2_aligned` after optional vertical adjustment. |
| `raster1_adjusted.tif` | Only generated when `vertical_adjustment.target: raster1`. |
| `dz_overlap.tif` | Difference raster used for offset estimation. Sign depends on adjustment target. |
| `blend_weights.tif` | Raster of blending weights around the selection polygon. |

---

## 10. `nodata`

```yaml
nodata:
  use_raster1_nodata: true
```

### Default

```yaml
nodata:
  use_raster1_nodata: true
```

If `true`, the output uses the NoData value from `raster1` when available. If
`raster1` has no NoData, it falls back to `raster2` NoData or the internal default.

If `false`, it uses `raster2` NoData when available.

---

## 11. Validation and diagnostics

After running the pipeline, inspect the Excel report in:

```text
<outdir>/report/<excel_report>
```

Important fields:

| Field | Meaning |
| --- | --- |
| `offset_applied` | Whether vertical offset was applied. |
| `offset_value` | Median difference used as offset. |
| `offset_reason` | Why offset was or was not applied. |
| `overlap_pixel_count` | Number of valid pixels used for offset estimation. |
| `overlap_mad` | Median absolute deviation of overlap differences. |
| `selection.inside_polygon` | Raster used inside the selection polygon. |
| `selection.outside_polygon` | Raster used outside the selection polygon. |
| `vertical_adjustment.target` | Raster adjusted vertically. |
| `output_grid.reference` | Raster defining CRS, pixel size, and alignment. |
| `output_grid.extent_polygon` | Polygon controlling final output extent when enabled. |

Recommended QGIS checks:

1. Compare `raster2_adjusted - raster2_aligned`; if the offset was applied to
   raster2, this should be approximately constant.
2. Compare `new_raster - raster1` inside the selection polygon; it should be near
   zero except in the blend band when `inside_polygon: raster1`.
3. Compare `new_raster - raster2_adjusted` outside the selection polygon; it should
   be near zero except in the blend band when `outside_polygon: raster2`.
4. Load `blend_weights.tif` to verify where the transition band is located.
5. Confirm that pixels outside `output_grid.extent_polygon` are NoData when
   `mask_to_extent_polygon: true`.

---

## 12. Common pitfalls

### Windows paths in YAML

Prefer forward slashes:

```yaml
raster1: "D:/path/to/raster.tif"
```

or single quotes with backslashes:

```yaml
raster1: 'D:\path\to\raster.tif'
```

Do not use double-quoted Windows paths with unescaped backslashes, because YAML may
interpret sequences such as `\1` or `\D` as escape characters.

### The output raster is rectangular

A GeoTIFF is always rectangular. `extent_polygon` defines the rectangular bounding
box used for the raster. Use `mask_to_extent_polygon: true` to set pixels outside
the polygon geometry to NoData.

### Offset not applied even when a median exists

A median difference may be computed, but the offset is only applied if:

```text
overlap_pixel_count >= min_overlap_pixels
and
overlap_mad <= mad_threshold
```

Check `offset_reason` in the Excel report.

### The adjusted raster appears unchanged

If `offset_applied: false`, the adjusted raster will be effectively identical to
the aligned raster. Check `offset_reason` before assuming a code error.
