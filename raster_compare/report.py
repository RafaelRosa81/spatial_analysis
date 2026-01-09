from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import rasterio
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font


def load_valid_values(raster_path: Path) -> np.ndarray:
    """
    Read band 1 and return a 1D array of valid values (nodata/NaN removed).
    """
    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    v = arr.reshape(-1)

    # Remove NaNs
    v = v[np.isfinite(v)]

    # Remove nodata if present
    if nodata is not None:
        v = v[v != nodata]

    return v.astype(np.float64, copy=False)


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    """
    Compute descriptive statistics + RMSE and selected percentiles.
    """
    keys = ["min", "max", "mean", "median", "std", "rmse", "p01", "p05", "p25", "p50", "p75", "p95", "p99"]
    if values.size == 0:
        return {k: float("nan") for k in keys}

    v = values.astype(np.float64, copy=False)
    rmse = float(np.sqrt(np.mean(v * v)))

    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=0)),
        "rmse": rmse,
        "p01": float(np.quantile(v, 0.01)),
        "p05": float(np.quantile(v, 0.05)),
        "p25": float(np.quantile(v, 0.25)),
        "p50": float(np.quantile(v, 0.50)),
        "p75": float(np.quantile(v, 0.75)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
    }


def threshold_table(abs_values: np.ndarray, thresholds: List[float]) -> pd.DataFrame:
    """
    Build bins: <=t1, (t1,t2], ..., >t_last
    """
    thresholds = sorted([float(t) for t in thresholds])
    edges = [0.0] + thresholds + [np.inf]

    labels: List[str] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi == np.inf:
            labels.append(f"> {lo:g}")
        elif lo == 0.0:
            labels.append(f"<= {hi:g}")
        else:
            labels.append(f"({lo:g}, {hi:g}]")

    counts: List[int] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi == np.inf:
            c = int(np.sum(abs_values > lo))
        elif lo == 0.0:
            c = int(np.sum(abs_values <= hi))
        else:
            c = int(np.sum((abs_values > lo) & (abs_values <= hi)))
        counts.append(c)

    total = int(abs_values.size)
    perc = [(c / total * 100.0) if total > 0 else float("nan") for c in counts]

    return pd.DataFrame({"Bin": labels, "Count": counts, "Percent": perc})


def histogram_table(values: np.ndarray, bins: int = 60) -> pd.DataFrame:
    if values.size == 0:
        return pd.DataFrame({"bin_left": [], "bin_right": [], "count": [], "percent": []})

    hist, edges = np.histogram(values, bins=int(bins))
    total = int(hist.sum())

    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": hist.astype(int),
            "percent": (hist / total * 100.0) if total > 0 else np.nan,
        }
    )


def _meta(path: Path) -> Dict[str, str]:
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "crs": str(src.crs),
            "width": str(src.width),
            "height": str(src.height),
            "pixel_x": str(src.transform.a),
            "pixel_y": str(abs(src.transform.e)),
            "nodata": str(src.nodata),
            "dtype": str(src.dtypes[0]),
            "bounds": str(src.bounds),
        }


def _crs_uses_meters(crs: rasterio.crs.CRS | None) -> bool:
    if crs is None:
        return False
    if crs.is_projected:
        return True
    return False


def _pixel_area_m2(raster_path: Path) -> Tuple[float | None, str | None]:
    with rasterio.open(raster_path) as src:
        crs = src.crs
        transform = src.transform

    if _crs_uses_meters(crs):
        return abs(transform.a * transform.e), None
    if crs is None:
        return None, "Pixel area not computed: raster has no CRS."
    if not crs.is_projected:
        return None, "Pixel area not computed: CRS is not projected."
    return None, "Pixel area not computed: CRS units are not meters."


def write_excel_report(
    dz_path: Path,
    abs_dz_path: Path,
    raster1_path: Path,
    raster2_path: Path,
    out_xlsx: Path,
    thresholds: List[float],
    bins: int = 60,
    run_config: Mapping[str, Any] | None = None,
) -> Path:
    """
    Create an Excel file with:
      - Stats_dz
      - Area_by_change_magnitude
      - Histogram_dz
      - Metadata
    """
    out_xlsx = Path(out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    dz_vals = load_valid_values(dz_path)
    abs_vals = np.abs(dz_vals)

    stats_df = pd.DataFrame([compute_stats(dz_vals)])
    th_df = threshold_table(abs_vals, thresholds)
    pixel_area_m2, area_warning = _pixel_area_m2(abs_dz_path)
    if pixel_area_m2 is None:
        th_df["PixelArea_m2"] = np.nan
        th_df["Area_m2"] = np.nan
    else:
        th_df["PixelArea_m2"] = float(pixel_area_m2)
        th_df["Area_m2"] = th_df["Count"] * float(pixel_area_m2)
    hist_df = histogram_table(dz_vals, bins=bins)

    meta_rows = []
    for layer_name, p in {
        "raster1": raster1_path,
        "raster2": raster2_path,
        "dz": dz_path,
        "abs_dz": abs_dz_path,
    }.items():
        md = _meta(Path(p))
        for k, v in md.items():
            meta_rows.append({"layer": layer_name, "field": k, "value": v})
    if area_warning:
        meta_rows.append({"layer": "report", "field": "area_units_warning", "value": area_warning})
    meta_df = pd.DataFrame(meta_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Stats_dz", index=False)
        th_df.to_excel(writer, sheet_name="Area_by_change_magnitude", index=False)
        hist_df.to_excel(writer, sheet_name="Histogram_dz", index=False)
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)
        if run_config:
            config_df = pd.DataFrame(
                [{"key": k, "value": v} for k, v in run_config.items()],
                columns=["key", "value"],
            )
            config_df.to_excel(writer, sheet_name="Config", index=False)

    # Light formatting
    wb = load_workbook(out_xlsx)
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"

        for col in ws.columns:
            col_letter = col[0].column_letter
            max_len = 0
            for c in col:
                if c.value is None:
                    continue
                max_len = max(max_len, len(str(c.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    if "Histogram_dz" in wb.sheetnames:
        ws = wb["Histogram_dz"]
        if ws.max_row >= 2:
            ws.cell(row=1, column=5, value="bin_label").font = Font(bold=True)
            ws.cell(row=1, column=5).alignment = Alignment(horizontal="center")
            for row in range(2, ws.max_row + 1):
                bin_left = ws.cell(row=row, column=1).value
                bin_right = ws.cell(row=row, column=2).value
                if bin_left is None or bin_right is None:
                    label = None
                else:
                    label = f"[{float(bin_left):.3g}, {float(bin_right):.3g})"
                ws.cell(row=row, column=5, value=label)
            ws.column_dimensions["E"].width = 18

            chart = BarChart()
            chart.type = "col"
            chart.title = "Histogram"
            chart.y_axis.title = "Count"
            chart.x_axis.title = "Bin"
            data = Reference(ws, min_col=3, max_col=3, min_row=1, max_row=ws.max_row)
            cats = Reference(ws, min_col=5, min_row=2, max_row=ws.max_row)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            chart.width = 20
            chart.height = 12
            ws.add_chart(chart, "F2")

    wb.save(out_xlsx)
    return out_xlsx
