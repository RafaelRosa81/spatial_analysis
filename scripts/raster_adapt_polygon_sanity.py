from __future__ import annotations

import numpy as np

from raster_compare.raster_adapt_polygon import _boundary_idw


def main() -> None:
    reference_points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )

    reference_values = np.array([100.0, 110.0, 90.0, 120.0], dtype=np.float32)

    query_points = np.array(
        [
            [5.0, 5.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )

    result = _boundary_idw(
        reference_points=reference_points,
        reference_values=reference_values,
        query_points=query_points,
        power=2.0,
        k_nearest=4,
    )

    assert result.shape[0] == 2
    assert np.all(np.isfinite(result))
    assert result.min() >= 90.0
    assert result.max() <= 120.0

    print("raster_adapt_polygon_sanity ok")


if __name__ == "__main__":
    main()
