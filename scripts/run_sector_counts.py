import sys

from scripts.sector_counts import run


if len(sys.argv) != 3 or sys.argv[1] != "--config":
    raise SystemExit("Usage: python -m scripts.run_sector_counts --config path/to/config.yml")

run(sys.argv[2])
