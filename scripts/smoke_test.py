from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.run_from_config import resolve_raster_diff_config, run_raster_diff


def _write_raster(path: Path, data: np.ndarray) -> None:
    transform = from_origin(0.0, 10.0, 1.0, 1.0)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32633",
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raster1 = tmp_path / "raster1.tif"
        raster2 = tmp_path / "raster2.tif"
        outdir = tmp_path / "outputs"

        _write_raster(raster1, np.zeros((3, 3), dtype=np.float32))
        _write_raster(raster2, np.ones((3, 3), dtype=np.float32))

        raw_config = {
            "outdir": str(outdir),
            "name": "smoke",
            "raster_diff": {
                "raster1": str(raster1),
                "raster2": str(raster2),
                "excel": False,
                "qgis_assets": False,
                "thresholds": [0.5],
                "bins": 10,
                "vector_threshold": None,
                "signed_vector_threshold": None,
            },
        }

        config = resolve_raster_diff_config(raw_config)
        run_raster_diff(config)

        expected = [
            outdir / "aligned" / "smoke_raster1_aligned.tif",
            outdir / "aligned" / "smoke_raster2_aligned.tif",
            outdir / "rasters" / "smoke_dz.tif",
            outdir / "rasters" / "smoke_abs_dz.tif",
            outdir / "report" / "smoke_alignment_report.json",
            outdir / "report" / "smoke_alignment_report.csv",
        ]

        missing = [path for path in expected if not path.exists()]
        if missing:
            missing_list = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Smoke test outputs missing:\n{missing_list}")

    print("smoke_test ok")


if __name__ == "__main__":
    main()
