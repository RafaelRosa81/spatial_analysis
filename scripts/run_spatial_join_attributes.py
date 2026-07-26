from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

import yaml

from raster_compare.spatial_join_attributes import (
    resolve_spatial_join_attributes_config,
    run_spatial_join_attributes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join point or polygon features to polygon zones while preserving attributes."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config root must be a YAML mapping.")

    config = resolve_spatial_join_attributes_config(raw_config)
    print("Resolved configuration:")
    print(pformat(config))

    outputs = run_spatial_join_attributes(config)
    print("Generated outputs:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
