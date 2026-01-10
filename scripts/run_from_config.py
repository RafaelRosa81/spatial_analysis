from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from pprint import pformat

import yaml

from raster_compare.core import align_to_reference, compute_dz
from raster_compare.polygon_mosaic import resolve_polygon_mosaic_config, run_polygon_mosaic
from raster_compare.qgis import (
    copy_qgis_assets,
    polygonize_exceedance,
    polygonize_signed_exceedance,
)
from raster_compare.report import (
    build_alignment_report,
    write_alignment_report_json_csv,
    write_excel_report,
    write_polygon_mosaic_excel,
)


ALLOWED_RESAMPLING = {"nearest", "bilinear", "cubic"}

# Raster diff: still requires these (but now we can provide them via the raster_diff section)
REQUIRED_KEYS = {"raster1", "raster2", "outdir", "name"}

DEFAULTS = {
    "resampling": "bilinear",
    "excel": True,
    "thresholds": [0.1, 0.25, 0.5, 1.0],
    "bins": 60,
    "qgis_assets": True,
    "vector_threshold": 0.5,
    "signed_vector_threshold": None,
}


def parse_args() -> argparse.Namespace:
    description = "Run workflows from a YAML config (supports multi-pipeline workspace YAML)."
    examples = """Examples:
  python -m scripts.run_from_config --config config/workspace.yml
"""
    parser = argparse.ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (workspace YAML).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping (YAML dictionary).")

    return config


def _validate_raster_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _select_pipeline_section(raw_config: dict, pipeline: str) -> dict:
    """
    Supports a single YAML that contains multiple pipeline sections.

    Top-level keys like name/outdir/resampling/excel can serve as shared defaults,
    but each section can override them.

    Returns: merged_config (dict) for the selected pipeline.
    """
    section = raw_config.get(pipeline)
    if section is None:
        raise ValueError(f"Pipeline section not found in YAML: '{pipeline}'")

    if not isinstance(section, dict):
        raise ValueError(f"Pipeline section '{pipeline}' must be a mapping (YAML dictionary).")

    # shared defaults at top level (only inject if missing in section)
    shared_keys = ["name", "outdir", "resampling", "excel", "qgis_assets"]
    merged = dict(section)
    for k in shared_keys:
        if k not in merged and raw_config.get(k) is not None:
            merged[k] = raw_config[k]

    # ensure pipeline is explicit in the merged config
    merged["pipeline"] = pipeline
    return merged


def resolve_raster_diff_config(raw_config: dict) -> dict:
    """
    raw_config here is already the selected section config for raster_diff,
    plus shared defaults injected by _select_pipeline_section().
    """
    raster_diff_section = raw_config.get("raster_diff") or {}
    if raster_diff_section and not isinstance(raster_diff_section, dict):
        raise ValueError("raster_diff section must be a mapping.")

    # Keep backward compatibility:
    # - if someone still nests raster_diff: {...} inside the raster_diff pipeline section
    # - or uses flat keys directly
    config = {**DEFAULTS, **raster_diff_section, **raw_config}

    missing = REQUIRED_KEYS - set(config)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Config missing required keys: {missing_list}")

    resampling = str(config["resampling"]).lower()
    if resampling not in ALLOWED_RESAMPLING:
        allowed = ", ".join(sorted(ALLOWED_RESAMPLING))
        raise ValueError(f"resampling must be one of: {allowed}")

    thresholds = config.get("thresholds", [])
    if not isinstance(thresholds, (list, tuple)) or not thresholds:
        raise ValueError("thresholds must be a non-empty list of numbers.")
    try:
        config["thresholds"] = [float(t) for t in thresholds]
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must contain numeric values.") from exc

    bins = config.get("bins", DEFAULTS["bins"])
    if int(bins) <= 0:
        raise ValueError("bins must be a positive integer.")
    config["bins"] = int(bins)

    vector_threshold = config.get("vector_threshold")
    if vector_threshold is not None:
        config["vector_threshold"] = float(vector_threshold)
        if config["vector_threshold"] <= 0:
            raise ValueError("vector_threshold must be greater than 0.")

    signed_vector_threshold = config.get("signed_vector_threshold")
    if signed_vector_threshold is not None:
        config["signed_vector_threshold"] = float(signed_vector_threshold)
        if config["signed_vector_threshold"] <= 0:
            raise ValueError("signed_vector_threshold must be greater than 0.")

    config["resampling"] = resampling
    config["pipeline"] = "raster_diff"
    return config


def run_raster_diff(config: dict) -> None:
    raster1 = Path(config["raster1"]).expanduser().resolve()
    raster2 = Path(config["raster2"]).expanduser().resolve()
    outdir = Path(config["outdir"]).expanduser().resolve()
    name = str(config["name"])
    resampling = config["resampling"]
    excel = bool(config["excel"])
    thresholds = list(map(float, config["thresholds"]))
    bins = int(config["bins"])
    qgis_assets = bool(config.get("qgis_assets", True))
    vector_threshold = config.get("vector_threshold")
    signed_vector_threshold = config.get("signed_vector_threshold")

    if vector_threshold is not None:
        vector_threshold = float(vector_threshold)
    if signed_vector_threshold is not None:
        signed_vector_threshold = float(signed_vector_threshold)

    _validate_raster_path(raster1, "raster1")
    _validate_raster_path(raster2, "raster2")

    resolved_config = {
        "pipeline": "raster_diff",
        "raster1": str(raster1),
        "raster2": str(raster2),
        "outdir": str(outdir),
        "name": name,
        "resampling": resampling,
        "excel": excel,
        "thresholds": thresholds,
        "bins": bins,
        "qgis_assets": qgis_assets,
        "vector_threshold": vector_threshold,
        "signed_vector_threshold": signed_vector_threshold,
    }

    print("Resolved configuration:")
    print(pformat(resolved_config))

    aligned_dir = outdir / "aligned"
    rasters_dir = outdir / "rasters"
    report_dir = outdir / "report"

    aligned_dir.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(parents=True, exist_ok=True)

    raster1_aligned = aligned_dir / f"{name}_raster1_aligned.tif"
    raster2_aligned = aligned_dir / f"{name}_raster2_aligned.tif"

    align_to_reference(
        src_path=raster1,
        ref_path=raster1,
        out_path=raster1_aligned,
        resampling="nearest",
        overwrite=False,
    )
    align_to_reference(
        src_path=raster2,
        ref_path=raster1,
        out_path=raster2_aligned,
        resampling=resampling,
        overwrite=False,
    )

    alignment_report = build_alignment_report(
        ref_path=raster1_aligned,
        src_path=raster2,
        aligned_path=raster2_aligned,
    )
    alignment_json, alignment_csv = write_alignment_report_json_csv(
        outdir=outdir,
        name=name,
        ref_path=raster1_aligned,
        src_path=raster2,
        aligned_path=raster2_aligned,
    )

    dz_path = rasters_dir / f"{name}_dz.tif"
    abs_dz_path = rasters_dir / f"{name}_abs_dz.tif"
    compute_dz(
        raster1_aligned=raster1_aligned,
        raster2_aligned=raster2_aligned,
        out_dz=dz_path,
        out_abs_dz=abs_dz_path,
        overwrite=False,
    )

    qgis_dir = None
    vector_path = None
    signed_vector_paths = None
    if qgis_assets:
        qgis_dir = copy_qgis_assets(outdir)

    if vector_threshold is not None:
        vectors_dir = outdir / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        threshold_str = f"{vector_threshold:g}"
        vector_path = vectors_dir / f"{name}_abs_dz_gt_{threshold_str}.geojson"
        polygonize_exceedance(
            abs_dz_path=abs_dz_path,
            out_vector_path=vector_path,
            threshold=vector_threshold,
            overwrite=False,
        )

    if signed_vector_threshold is not None:
        vectors_dir = outdir / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        threshold_str = f"{signed_vector_threshold:g}"
        signed_vector_paths = polygonize_signed_exceedance(
            dz_path=dz_path,
            out_positive_path=vectors_dir / f"{name}_dz_gt_{threshold_str}.geojson",
            out_negative_path=vectors_dir / f"{name}_dz_lt_-{threshold_str}.geojson",
            threshold=signed_vector_threshold,
            overwrite=False,
        )

    excel_path = None
    if excel:
        report_dir.mkdir(parents=True, exist_ok=True)
        excel_path = report_dir / f"{name}_Comparison_Report.xlsx"
        write_excel_report(
            dz_path=dz_path,
            abs_dz_path=abs_dz_path,
            raster1_path=raster1_aligned,
            raster2_path=raster2_aligned,
            out_xlsx=excel_path,
            thresholds=thresholds,
            bins=bins,
            run_config=resolved_config,
            alignment_report=alignment_report,
        )

    print("Generated outputs:")
    print(f"- {raster1_aligned}")
    print(f"- {raster2_aligned}")
    print(f"- {dz_path}")
    print(f"- {abs_dz_path}")
    if qgis_dir is not None:
        print(f"- {qgis_dir}")
    if vector_path is not None:
        print(f"- {vector_path}")
    if signed_vector_paths is not None:
        for signed_path in signed_vector_paths:
            print(f"- {signed_path}")
    if excel_path is not None:
        print(f"- {excel_path}")
    print(f"- {alignment_json}")
    print(f"- {alignment_csv}")


def run_polygon_mosaic_pipeline(selected_section_cfg: dict) -> None:
    """
    selected_section_cfg is the polygon_mosaic section merged with top-level overrides.
    We pass a dict that looks like the old 'raw_config' expected by resolve_polygon_mosaic_config().
    """
    config = resolve_polygon_mosaic_config(selected_section_cfg)
    print("Resolved configuration:")
    print(pformat(config))

    metrics = run_polygon_mosaic(config)

    outputs = metrics.get("outputs", {})
    report_path = outputs.get("report_path")
    if config.get("excel", True) and report_path:
        write_polygon_mosaic_excel(report_path, config, metrics)

    print("Generated outputs:")
    print(f"- {outputs.get('new_raster')}")
    if outputs.get("report_path"):
        print(f"- {outputs.get('report_path')}")


def run_sample_points_from_raster_value_range_pipeline(selected_section_cfg: dict) -> None:
    required = {"raster", "outdir", "name", "value_min", "value_max"}
    missing = required - set(selected_section_cfg.keys())
    if missing:
        raise ValueError(
            "Config missing required keys for sample_points_from_raster_value_range: "
            + ", ".join(sorted(missing))
        )

    raster = Path(selected_section_cfg["raster"]).expanduser().resolve()
    outdir = Path(selected_section_cfg["outdir"]).expanduser().resolve()
    name = str(selected_section_cfg["name"])
    value_min = float(selected_section_cfg["value_min"])
    value_max = float(selected_section_cfg["value_max"])

    _validate_raster_path(raster, "raster")
    outdir.mkdir(parents=True, exist_ok=True)

    sampling = selected_section_cfg.get("sampling") or {}
    if sampling and not isinstance(sampling, dict):
        raise ValueError("sampling section must be a mapping (YAML dictionary).")

    method = str(sampling.get("method", "random")).lower()
    if method not in {"random", "regular"}:
        raise ValueError("sampling.method must be one of: random, regular"
