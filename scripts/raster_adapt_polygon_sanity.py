from __future__ import annotations

import numpy as np

from raster_compare.raster_adapt_polygon import _boundary_idw


def main() -> None:
    reference_coords = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    reference_values = np.array([100.0, 110.0, 90.0, 120.0], dtype=np.float64)

    target_coords = np.array(
        [
            [5.0, 5.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )

    result = _boundary_idw(
        target_coords=target_coords,
        reference_coords=reference_coords,
        reference_values=reference_values,
        idw_power=2.0,
        k_nearest=4,
    )

    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))
    assert result.min() >= 90.0
    assert result.max() <= 120.0

    print("raster_adapt_polygon_sanity ok")


if __name__ == "__main__":
    main()
