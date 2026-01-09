from __future__ import annotations

import argparse
from pathlib import Path

from raster_compare.core import align_to_reference, compute_dz
from raster_compare.report import write_excel_report


def parse_args() -> argparse.Namespace:
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

    # Excel reporting (optional)
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Generate an Excel report with stats/histogram/threshold tables",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.10, 0.25, 0.50, 1.00],
        help="Thresholds for |dz| bins (default: 0.10 0.25 0.50 1.00)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Histogram bins for dz (default: 60)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raster1 = Path(args.raster1)
    raster2 = Path(args.raster2)
    outdir = Path(args.outdir)
    name = args.name
    resampling = args.resampling
    overwrite = args.overwrite

    aligned_dir = outdir / "aligned"
    rasters_dir = outdir / "rasters"
    report_dir = outdir / "report"

    aligned_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)

    raster1_aligned = aligned_dir / f"{name}_raster1_aligned.tif"
    raster2_aligned = aligned_dir / f"{name}_raster2_aligned.tif"

    # Use raster1 as reference grid
    align_to_reference(
        src_path=raster1,
        ref_path=raster1,
        out_path=raster1_aligned,
        resampling="nearest",
        overwrite=overwrite,
    )
    align_to_reference(
        src_path=raster2,
        ref_path=raster1,
        out_path=raster2_aligned,
        resampling=resampling,
        overwrite=overwrite,
    )

    dz_path = rasters_dir / f"{name}_dz.tif"
    abs_dz_path = rasters_dir / f"{name}_abs_dz.tif"
    compute_dz(
        raster1_aligned=raster1_aligned,
        raster2_aligned=raster2_aligned,
        out_dz=dz_path,
        out_abs_dz=abs_dz_path,
        overwrite=overwrite,
    )

    excel_path = None
    if args.excel:
        report_dir.mkdir(parents=True, exist_ok=True)
        excel_path = report_dir / f"{name}_Comparison_Report.xlsx"
        write_excel_report(
            dz_path=dz_path,
            abs_dz_path=abs_dz_path,
            raster1_path=raster1_aligned,
            raster2_path=raster2_aligned,
            out_xlsx=excel_path,
            thresholds=list(map(float, args.thresholds)),
            bins=int(args.bins),
        )

    print("Generated outputs:")
    print(f"- {raster1_aligned}")
    print(f"- {raster2_aligned}")
    print(f"- {dz_path}")
    print(f"- {abs_dz_path}")
    if excel_path is not None:
        print(f"- {excel_path}")


if __name__ == "__main__":
    main()
