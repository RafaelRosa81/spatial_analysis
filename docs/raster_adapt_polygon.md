# Raster Adapt Polygon Pipeline

Pipeline name:

```yaml
pipeline: "raster_adapt_polygon"
```

This pipeline adapts an existing raster inside a user-defined polygon. It keeps the
original raster outside the polygon and reconstructs the topography inside the
polygon using reference elevations taken from the surrounding terrain.

The pipeline is designed for DEM/MDT correction workflows where a local area must
be removed, smoothed, reconstructed, or made coherent with the surrounding surface.

Typical use cases:

- remove or soften artificial fills/excavations;
- reconstruct terrain continuity inside a disturbed polygon;
- replace a noisy or unreliable DEM patch using surrounding terrain;
- create alternative terrain scenarios for hydraulic or drainage modelling;
- generate diagnostic rasters to compare possible terrain-adaptation criteria.

Run with:

```bash
python -m scripts.run_from_config --config config/raster_adapt_polygon_local.yml
```

---

## 1. Conceptual workflow

The pipeline works in this order:

```text
input raster + modify polygon
        ↓
rasterize polygon on the raster grid
        ↓
build an external reference ring around the polygon
        ↓
extract valid elevation samples from that ring
        ↓
interpolate / fit a replacement surface inside the polygon
        ↓
optionally blend the replacement surface with the original raster near the border
        ↓
write final adapted raster + diagnostic rasters + Excel/JSON report
```

The most important conceptual distinction is:

```text
modify_polygon      = area where raster values may be changed
reference_ring      = external zone used to learn the replacement topography
adapted_surface     = reconstructed surface inside the polygon
adapted_raster      = final raster after blending/replacement
```

Outside `modify_polygon`, the output raster is kept equal to the original input raster.

---

## 2. Basic YAML structure

```yaml
pipeline: "raster_adapt_polygon"

raster_adapt_polygon:
  name: "adapt_demo"
  outdir: "outputs/adapt_demo"
  excel: true

  raster: "D:/path/to/input_dem.tif"
  modify_polygon: "D:/path/to/modify_area.shp"

  outputs:
    adapted_raster: "adapted_dem.tif"
    excel_report: "adapt_report.xlsx"
    summary_json: "adapt_summary.json"
    save_intermediates: true

  reference_ring:
    outer_buffer_px: 20
    inner_buffer_px: 0
    min_reference_pixels: 500

  adaptation:
    method: "boundary_idw"
    idw_power: 2.0
    k_nearest: 32
    max_search_distance_px: null
    max_reference_points: 10000
    random_seed: 42
    polynomial_order: 2

  border_blending:
    enabled: true
    blend_width_px: 5

  nodata:
    preserve_nodata: true
```

---

## 3. Required inputs

### `raster`

Path to the input raster to adapt.

```yaml
raster: "D:/path/to/input_dem.tif"
```

Requirements:

- must exist;
- must be readable by `rasterio`;
- must have a valid CRS;
- should be a continuous raster, usually elevation.

### `modify_polygon`

Path to a polygon vector layer defining the zone to modify.

```yaml
modify_polygon: "D:/path/to/modify_area.shp"
```

Requirements:

- must exist;
- must contain at least one polygon geometry;
- may be in a different CRS than the raster; the pipeline reprojects geometries to the raster CRS;
- must overlap the raster grid.

The polygon should include the full area that you want to reconstruct. If artificial edges, banks, fills, or transition zones remain outside the polygon, the pipeline cannot modify them and may even use them as reference data.

---

## 4. Output files

With:

```yaml
outputs:
  adapted_raster: "adapted_dem.tif"
  save_intermediates: true
```

outputs are written under:

```text
<outdir>/rasters/
<outdir>/report/
<outdir>/metadata/
```

Typical structure:

```text
outputs/adapt_demo/
├─ rasters/
│  ├─ adapted_dem.tif
│  ├─ adapt_demo_adapted_surface.tif
│  ├─ adapt_demo_modify_mask.tif
│  ├─ adapt_demo_reference_ring.tif
│  └─ adapt_demo_blend_weights.tif
├─ report/
│  └─ adapt_report.xlsx
└─ metadata/
   └─ adapt_summary.json
```

### `adapted_dem.tif`

Final adapted raster.

```text
outside modify_polygon -> original raster
inside modify_polygon  -> adapted surface, optionally blended near the border
```

This is the main output to use as the corrected/adapted DEM.

### `<name>_adapted_surface.tif`

The reconstructed surface inside the polygon before final replacement/blending.

Use it to inspect what the selected adaptation method generated.

### `<name>_modify_mask.tif`

Rasterized version of `modify_polygon`.

Expected values:

```text
1 inside the modification polygon
0 outside the modification polygon
```

Use it to verify that the vector polygon was correctly rasterized on the input raster grid.

### `<name>_reference_ring.tif`

Raster mask showing the external reference pixels used to estimate the replacement surface.

Expected values:

```text
1 valid reference pixel
0 not used as reference
```

This is one of the most important diagnostic outputs. If this ring overlaps artificial terrain, talus, fill, or noisy data, the adapted result will inherit that problem.

### `<name>_blend_weights.tif`

Raster of blending weights.

Expected interpretation:

```text
0   -> original raster only
1   -> adapted surface only
0-1 -> transition between original and adapted surface
```

If `border_blending.enabled: false`, this raster is essentially the polygon mask.

### Excel report

The Excel report includes:

- `Summary`: execution summary;
- `Config`: flattened resolved configuration;
- `ReferenceRing`: statistics of the external reference samples;
- `Interpolation`: method parameters and fit diagnostics;
- `RasterStats`: original/adapted/difference statistics inside the polygon;
- `FileInventory`: list of generated files.

### JSON summary

Machine-readable copy of the main report information.

---

## 5. Reference ring configuration

```yaml
reference_ring:
  outer_buffer_px: 20
  inner_buffer_px: 0
  min_reference_pixels: 500
```

The reference ring is built outside `modify_polygon`.

### `outer_buffer_px`

Number of pixels outward from the polygon used to define the outer edge of the reference zone.

Example:

```yaml
outer_buffer_px: 60
```

If raster resolution is 0.5 m, this is approximately 30 m.

### `inner_buffer_px`

Number of pixels outward from the polygon to exclude before reference sampling starts.

Example:

```yaml
inner_buffer_px: 10
```

This means:

```text
0-10 px outside polygon   -> ignored
10-60 px outside polygon  -> used as reference, if outer_buffer_px=60
```

Use `inner_buffer_px` when the border of the polygon is contaminated by artificial slopes, edges, fill, or transition artifacts.

### `min_reference_pixels`

Minimum number of valid reference pixels required.

```yaml
min_reference_pixels: 1000
```

If the ring has fewer valid pixels, the pipeline fails with a clear error.

### Practical guidance

For small or clean polygons:

```yaml
reference_ring:
  outer_buffer_px: 20
  inner_buffer_px: 0
```

For polygons near disturbed terrain:

```yaml
reference_ring:
  outer_buffer_px: 50
  inner_buffer_px: 10
```

For heterogeneous terrain where the nearby edge is contaminated:

```yaml
reference_ring:
  outer_buffer_px: 60
  inner_buffer_px: 15
```

Always inspect `<name>_reference_ring.tif` in QGIS.

---

## 6. Adaptation methods

The method is selected with:

```yaml
adaptation:
  method: "boundary_idw"
```

Supported methods:

```text
boundary_idw
nearest_boundary
plane_fit
polynomial_fit
```

---

## 6.1 `boundary_idw`

```yaml
adaptation:
  method: "boundary_idw"
  idw_power: 2.0
  k_nearest: 32
  max_search_distance_px: null
  max_reference_points: 10000
  random_seed: 42
```

### What it does

For each pixel inside `modify_polygon`, the method estimates elevation using inverse-distance weighting from reference pixels in the external ring.

Conceptually:

```text
z(x,y) = sum(w_i * z_i) / sum(w_i)
w_i = 1 / d_i^p
```

where:

- `z_i` are elevations in the reference ring;
- `d_i` are distances from the target pixel to reference pixels;
- `p` is `idw_power`.

### Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `idw_power` | `2.0` | Controls how fast influence decreases with distance. Higher values make the interpolation more local. |
| `k_nearest` | `32` | Number of nearest reference pixels used for each target pixel. Lower values are more local. |
| `max_search_distance_px` | `null` | Optional maximum search distance in pixels. If no neighbours are found inside this distance, the method falls back to unrestricted nearest neighbours. |
| `max_reference_points` | `10000` | Maximum number of reference points kept from the ring. Larger values preserve more data but may be slower. |
| `random_seed` | `42` | Seed used if reference points are subsampled. |

### When to use

Use `boundary_idw` when the terrain around the polygon is locally variable and you want the adapted surface to respect local terrain patterns.

### Recommended variants

Global/smooth IDW:

```yaml
adaptation:
  method: "boundary_idw"
  idw_power: 2.0
  k_nearest: 32
  max_search_distance_px: null
```

More local IDW:

```yaml
adaptation:
  method: "boundary_idw"
  idw_power: 3.0
  k_nearest: 12
  max_search_distance_px: 20
```

If a visible ring appears near the polygon boundary, try:

```yaml
reference_ring:
  outer_buffer_px: 50
  inner_buffer_px: 10

border_blending:
  enabled: false
  blend_width_px: 0
```

or blending only slightly:

```yaml
border_blending:
  enabled: true
  blend_width_px: 2
```

---

## 6.2 `nearest_boundary`

```yaml
adaptation:
  method: "nearest_boundary"
```

### What it does

For each pixel inside the polygon, the method copies the value of the closest reference pixel from the external ring.

### Characteristics

Advantages:

- fast;
- stable;
- very local;
- useful as a diagnostic baseline.

Limitations:

- can create patchy or Voronoi-like regions;
- may need smoothing or blending for final production use;
- not ideal when the polygon is large.

### When to use

Use it to understand whether local boundary values are reasonable. It is also useful when preserving very local edge behaviour is more important than smoothness.

---

## 6.3 `plane_fit`

```yaml
adaptation:
  method: "plane_fit"
  polynomial_order: 1
  max_reference_points: 20000
  random_seed: 42
```

### What it does

Fits a first-order surface to the reference ring:

```text
z = a + b*x + c*y
```

Then evaluates that plane inside the polygon.

### Characteristics

Advantages:

- very smooth;
- robust for large disturbed areas;
- good when the surrounding terrain has a clear general slope;
- useful when IDW produces noisy or overly local artifacts.

Limitations:

- cannot represent curvature;
- may oversimplify complex terrain;
- may not respect local drainage details.

### Diagnostics

The Excel report includes:

```text
fit_rmse
fit_rank
fit_coefficients
```

A high `fit_rmse` means the reference ring is not well represented by a plane.

---

## 6.4 `polynomial_fit`

```yaml
adaptation:
  method: "polynomial_fit"
  polynomial_order: 2
  max_reference_points: 20000
  random_seed: 42
```

### What it does

Fits a polynomial surface to the reference ring. With `polynomial_order: 2`, the fitted model is:

```text
z = a + b*x + c*y + d*x² + e*x*y + f*y²
```

The coordinates are internally centered and scaled before fitting to improve numerical stability.

### Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `polynomial_order` | `2` | Supported values: `1` or `2`. Order `1` is equivalent to a plane. |
| `max_reference_points` | `10000` | Maximum ring samples used for fitting. |
| `random_seed` | `42` | Reproducible subsampling seed. |

### Characteristics

Advantages:

- smooth;
- can represent broad curvature;
- often works well when IDW leaves local artifacts.

Limitations:

- may overfit if the ring is noisy or heterogeneous;
- may produce unrealistic highs/lows inside large or irregular polygons;
- can create visible edge mismatch if the reference ring includes contaminated values.

### Diagnostics

Check in the Excel report:

```text
fit_rmse
fit_rank
fit_coefficients
```

If `fit_rmse` is large, try:

- larger and cleaner reference ring;
- `inner_buffer_px` greater than zero;
- `plane_fit` instead of `polynomial_fit`;
- subdividing the polygon into smaller regions.

---

## 7. Border blending

```yaml
border_blending:
  enabled: true
  blend_width_px: 5
```

Border blending controls the transition inside `modify_polygon` between the original raster and the adapted surface.

Conceptually:

```text
near polygon border -> more original raster
inside polygon      -> more adapted surface
```

The final equation is:

```text
adapted_raster = original * (1 - weight) + adapted_surface * weight
```

where `weight` ranges from 0 near the polygon edge to 1 farther inside.

### Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `true` | Enables/disables blending. |
| `blend_width_px` | `5` | Width of the transition band inside the polygon, in pixels. |

### Practical guidance

If a visible ring appears along the polygon boundary, test:

```yaml
border_blending:
  enabled: false
  blend_width_px: 0
```

If the result becomes too sharp, use a small blending width:

```yaml
border_blending:
  enabled: true
  blend_width_px: 2
```

Avoid large blending widths when the original raster inside the polygon contains the artifact you are trying to remove.

---

## 8. NoData handling

```yaml
nodata:
  preserve_nodata: true
```

If `preserve_nodata: true`, NoData pixels in the input raster remain NoData in the output.

This is the recommended setting.

---

## 9. Choosing a method

| Situation | Recommended method |
| --- | --- |
| Need local reconstruction that follows surrounding terrain | `boundary_idw` |
| Need a fast local diagnostic | `nearest_boundary` |
| Need a smooth planar replacement | `plane_fit` |
| Need a smooth curved replacement | `polynomial_fit` |
| IDW produces noisy or irregular surface | try `plane_fit` or `polynomial_fit` |
| Polynomial produces artificial curvature | try `plane_fit` or more local `boundary_idw` |
| A ring remains near polygon edge | reduce/disable blending and/or increase `inner_buffer_px` |
| Edge/reference terrain is contaminated | enlarge polygon or use `inner_buffer_px > 0` |
| Terrain varies strongly around polygon | subdivide the polygon and run the pipeline per sub-zone |

---

## 10. Recommended test matrix

For a new project, run the same input with several configurations:

### A. Local IDW

```yaml
reference_ring:
  outer_buffer_px: 50
  inner_buffer_px: 10

adaptation:
  method: "boundary_idw"
  idw_power: 3.0
  k_nearest: 12
  max_search_distance_px: 20

border_blending:
  enabled: false
  blend_width_px: 0
```

### B. Local IDW with small blending

```yaml
border_blending:
  enabled: true
  blend_width_px: 2
```

### C. Plane fit

```yaml
reference_ring:
  outer_buffer_px: 60
  inner_buffer_px: 10

adaptation:
  method: "plane_fit"
  polynomial_order: 1
```

### D. Polynomial fit

```yaml
reference_ring:
  outer_buffer_px: 60
  inner_buffer_px: 10

adaptation:
  method: "polynomial_fit"
  polynomial_order: 2
```

Compare results using:

```text
adapted_dem - original_dem
```

and elevation profiles in QGIS.

---

## 11. Reading the report

Important fields:

### `Summary`

- `modified_pixels`: pixels inside the polygon;
- `valid_modified_pixels`: valid input pixels inside the polygon;
- `reference_pixels`: valid pixels in the reference ring;
- `reference_pixels_used`: number used after optional subsampling;
- `method`: selected adaptation method.

### `ReferenceRing`

Check:

```text
reference_z_min
reference_z_max
reference_z_mean
reference_z_std
```

A very high `reference_z_std` means the ring is heterogeneous. In that case:

- use a more local method;
- increase `inner_buffer_px`;
- adjust `outer_buffer_px`;
- subdivide the polygon.

### `Interpolation`

For IDW:

```text
idw_power
k_nearest
max_search_distance_px
max_search_distance_units
```

For fitted methods:

```text
fit_order
fit_rank
fit_rmse
fit_coefficients
```

### `RasterStats`

The key scope is:

```text
adapted_minus_original_inside_polygon
```

This shows how much the terrain changed inside the polygon.

---

## 12. QGIS validation workflow

Load:

```text
adapted_dem.tif
<name>_adapted_surface.tif
<name>_modify_mask.tif
<name>_reference_ring.tif
<name>_blend_weights.tif
original raster
modify polygon
```

Check:

1. `modify_mask` matches the polygon.
2. `reference_ring` is outside the polygon and does not include contaminated terrain.
3. `blend_weights` does not preserve too much of the original artifact.
4. Profiles across the polygon are smooth enough for the project goal.
5. Raster calculator:

```text
adapted_dem - original_dem
```

Outside the polygon this should be zero or NoData-equivalent. Inside the polygon it should show the intended modification.

---

## 13. Common problems and fixes

### Visible ring around the polygon

Try:

```yaml
border_blending:
  enabled: false
  blend_width_px: 0
```

or:

```yaml
border_blending:
  enabled: true
  blend_width_px: 2
```

Also try:

```yaml
reference_ring:
  outer_buffer_px: 50
  inner_buffer_px: 10
```

### Interior remains too high or too low

Try a more local IDW:

```yaml
adaptation:
  method: "boundary_idw"
  idw_power: 3.0
  k_nearest: 12
  max_search_distance_px: 20
```

or use `plane_fit` if the desired replacement is a smooth slope.

### Reference ring is too variable

If the report shows large `reference_z_std`, the ring may be mixing different terrain units. Try:

- smaller `outer_buffer_px`;
- non-zero `inner_buffer_px`;
- a larger polygon that fully contains the disturbed area;
- splitting the polygon into several smaller polygons.

### Polynomial fit looks good but creates artificial curvature

Try:

```yaml
adaptation:
  method: "plane_fit"
```

or reduce the reference ring heterogeneity.

### IDW is too noisy

Try:

```yaml
idw_power: 2.0
k_nearest: 32
```

for a smoother result, or use `polynomial_fit`.

---

## 14. Limitations and future improvements

Current limitations:

- no automatic polygon subdivision;
- no automatic delta rasters yet;
- no smoothing post-process yet;
- no constraints on maximum raise/lower yet;
- no hydrologic conditioning constraints yet.

Possible future options:

```yaml
postprocess:
  smoothing:
    enabled: true
    sigma_px: 2

constraints:
  max_raise: 1.0
  max_lower: 1.0

outputs:
  save_delta_rasters: true
```

These are not part of the current implementation yet.
