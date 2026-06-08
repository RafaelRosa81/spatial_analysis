from __future__ import annotations

import numpy as np

from raster_compare.raster_adapt_polygon import _compute_idw_weights


def main() -> None:
    distances = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    weights = _compute_idw_weights(distances, power=2.0)

    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1] > weights[2]

    print("raster_adapt_polygon_sanity ok")


if __name__ == "__main__":
    main()
