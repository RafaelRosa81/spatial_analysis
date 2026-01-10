# CLI Reference

## Primary entrypoint

```bash
python -m scripts.run_from_config --config <path>
```

**Options**

- `--config` (required): Path to a YAML configuration file.

**Recommended commands**

```bash
python -m scripts.run_from_config --config config/minimal_raster_diff_example.yml
python -m scripts.run_from_config --config config/polygon_mosaic_example.yml
python -m scripts.run_from_config --config config/sample_points_example.yml
```

The help output includes examples:

```bash
python -m scripts.run_from_config --help
```

## Legacy CLI (ad hoc)

```bash
python -m scripts.compare_rasters --raster1 <path> --raster2 <path> --outdir <dir> --name <run>
```

**Options**

- `--raster1` (required): Path to raster 1 (reference grid).
- `--raster2` (required): Path to raster 2 (aligned to raster1).
- `--outdir` (required): Output directory.
- `--name` (required): Output name prefix.
- `--resampling` (optional): Resampling method for raster2 alignment (`bilinear` default).
- `--overwrite` (optional flag): Overwrite existing outputs.
- `--excel` (optional flag): Generate Excel report.
- `--thresholds` (optional): List of thresholds for `abs(dz)` binning.
- `--bins` (optional): Histogram bins for `dz` (default 60).
- `--qgis-assets` (optional flag): Copy QGIS QML styles into output.
- `--vector-threshold` (optional): Threshold for `abs(dz)` exceedance polygons.
- `--signed-vector-threshold` (optional): Threshold for signed dz exceedance polygons.

## Command cookbook

### Minimal raster diff run

```bash
python -m scripts.run_from_config --config config/minimal_raster_diff_example.yml
```

### Full-feature raster diff (Excel, QGIS, vectors)

```bash
python -m scripts.run_from_config --config config/full_raster_diff_example.yml
```

### Change bins or thresholds

Edit the config:

```yaml
bins: 80
thresholds: [0.05, 0.1, 0.25, 0.5, 1.0]
```

### Change output directory or run name

```yaml
outdir: "outputs"
name: "survey_2024_vs_2022"
```

### Enable/disable Excel and QGIS assets

```yaml
excel: false
qgis_assets: false
```

### Enable/disable vector outputs

```yaml
vector_threshold: 0.5
signed_vector_threshold: null
```

### Debugging tips

- Use `--help` to verify the CLI is available.
- Inspect the alignment report in `outputs/<name>/report/` for CRS/grid shifts.
- Start with `resampling: nearest` for categorical rasters and `bilinear` for DEMs.

## Testing

The regression test module `tests/run_regression.py` executes the pipelines in
`tests/configs/workspace_regression.yml`, including the sample points pipeline.
It compares output fingerprints against the golden data stored under
`tests/golden/`.

```bash
python tests/run_regression.py --mode compare
```

Use `--mode update` to regenerate golden fingerprints after intentional output
changes.
