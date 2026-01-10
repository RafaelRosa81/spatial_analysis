from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from raster_compare.polygon_mosaic import _blend_weights


def main() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    weights = _blend_weights(mask, blend_width=1)
    assert weights.min() >= 0.0
    assert weights.max() <= 1.0

    r1 = np.zeros((5, 5), dtype=np.float32)
    r2 = np.full((5, 5), 10.0, dtype=np.float32)

    output = r1 * (1.0 - weights) + r2 * weights

    assert output[0, 0] == 0.0
    assert output[2, 2] >= 9.0

    print("polygon_mosaic_sanity ok")


if __name__ == "__main__":
    main()
