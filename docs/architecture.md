# Architecture

## Components

- `scripts/run_from_config.py`: YAML-driven entrypoint.
- `raster_compare/core.py`: alignment and dz computation.
- `raster_compare/report.py`: statistics, histogram, and Excel report generation.
- `raster_compare/qgis.py`: QGIS assets and vector overlays.
- `raster_compare/polygon_mosaic.py`: polygon-based mosaic pipeline.

## Data flow + artifacts

```mermaid
flowchart LR
  A[Input rasters]
  B[Aligned rasters]
  C[dz / abs_dz rasters]
  D[Reports]
  E[QGIS assets]
  F[Vector overlays]

  A -->|align_to_reference| B
  B -->|compute_dz| C
  C -->|stats + histogram| D
  C -->|polygonize| F
  B -->|qgis styles| E

  subgraph Outputs
    B --> O1[outputs/<name>/aligned]
    C --> O2[outputs/<name>/rasters]
    D --> O3[outputs/<name>/report]
    E --> O4[outputs/<name>/qgis]
    F --> O5[outputs/<name>/vectors]
  end
```

## Output artifacts

- **GeoTIFFs**: aligned rasters, signed dz, absolute dz.
- **Excel**: summary statistics, area-by-threshold tables, histogram, alignment report.
- **GeoJSON**: exceedance polygons (abs or signed dz).
- **QGIS styles**: QML files for easy symbology.
- **Logs**: alignment report JSON/CSV under `report/`.

See [pipeline_overview.md](pipeline_overview.md) for detailed pipeline steps.
