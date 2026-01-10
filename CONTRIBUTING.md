# Contributing

Thanks for considering a contribution!

## Development setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate spatial_analysis
```

### Pip-only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Formatting

There is no enforced formatter at this time. Please keep changes consistent with
existing style, use descriptive names, and add docstrings for new functions.

## Tests

Minimal checks:

```bash
python -m scripts.run_from_config --help
python -m scripts.smoke_test
```

If you add new functionality, please include a small unit or smoke test that
runs quickly and does not require large datasets.
