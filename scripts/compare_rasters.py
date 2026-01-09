"""CLI for raster alignment and difference products."""

from __future__ import annotations

import argparse
from pathlib import Path

from raster_compare.core import align_to_reference, compute_dz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align two rasters and compute signed/absolute dz products."
    )
    parser.add_argument("--raster1", required=True, help="Path to raster 1")
    parser.add_argument("--raster2", required=True, help="Path to raster 2")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--name", required=True, help="Output name prefix")
    parser.add_argument(
        "--resampling",
        default="bilinear",
        help="Resampling method for alignment (default: bilinear)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raster1 = Path(args.raster1)
    raster2 = Path(args.raster2)
    outdir = Path(args.outdir)
    name = args.name

    aligned_dir = outdir / "aligned"
    rasters_dir = outdir / "rasters"

    raster1_aligned = aligned_dir / f"{name}_raster1_aligned.tif"
    raster2_aligned = aligned_dir / f"{name}_raster2_aligned.tif"
    dz_path = rasters_dir / f"{name}_dz.tif"
    abs_dz_path = rasters_dir / f"{name}_abs_dz.tif"

    align_to_reference(
        raster1,
        raster1,
        raster1_aligned,
        resampling=args.resampling,
        overwrite=args.overwrite,
    )
    align_to_reference(
        raster2,
        raster1,
        raster2_aligned,
        resampling=args.resampling,
        overwrite=args.overwrite,
    )

    compute_dz(
        raster1_aligned,
        raster2_aligned,
        dz_path,
        abs_dz_path,
        overwrite=args.overwrite,
    )

    print("Generated outputs:")
    print(f"- {raster1_aligned}")
    print(f"- {raster2_aligned}")
    print(f"- {dz_path}")
    print(f"- {abs_dz_path}")


if __name__ == "__main__":
    main()
