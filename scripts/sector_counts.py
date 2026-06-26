from pathlib import Path

import yaml

from raster_compare.sector_counts import resolve_sector_counts_config, run_count_features_by_sector


def run(config_path: str) -> None:
    with Path(config_path).open(encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    config = resolve_sector_counts_config(raw_config)
    outputs = run_count_features_by_sector(config)
    for key, value in outputs.items():
        if key != "table":
            print(f"{key}: {value}")
