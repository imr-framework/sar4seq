# SAR4seq Python

A Python implementation of the SAR4seq MATLAB toolbox for computing RF safety metrics for Pulseq sequences.

## Overview

SAR4seq computes RF safety metrics for Pulseq sequences including:
- Time averaged RF power for Siemens scanners
- Whole body SAR (Specific Absorption Rate) in W/kg for GE scanners (via TOPPE)

## Features

- **Global SAR computation**: Whole body and head SAR calculations
- **Local SAR computation**: Point-wise SAR calculations
- **VOP (Virtual Observation Points)**: Implementation of Eichfelder's VOP method
- **Multi-vendor support**: Siemens and GE scanner compatibility
- **Q-matrix generation**: Electromagnetic field-based SAR calculations

## Installation
If uv is not installed, install it via:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install package in editable mode
```bash
uv pip install -e .
```
However, the project structure shall be revised when integrating to pypulseq

## Run sar4seq
Run:
```bash
uv run src/sar4seq.py
uv sync
```

## Run pytest
Run the following command to run all tests at once:
```bash
uv run python -m pytest tests/ -v
```
or run single tests:
```bash
uv run python -m pytest tests/sar4seq.py -v
```

A note: The test warnings about deprecation of some usage syntax showed will be delt with later as I was only looking at functionality now. There's also an RF event warning brought about by a poorly timed generated sequence which I shall also work on later.

## Global SAR Calculation
Run:
```bash
uv run src/sar4seq.py
```
Provide correct paths to the QGlobal mat and pulseq files. Based on the number of spatial points and whether or not you have a GPU available, you can utilize the `requires_gpu` parameter to run the computation in host (cpu) or device (gpu):

```python
results = sar4seq(seq_path, Q_mat_path, requires_gpu=False, n_spatial=100)
```

### Dependencies

- numpy
- scipy
- matplotlib
- h5py (for .mat file compatibility)
- pypulseq (for Pulseq sequence handling)

## Documentation

- [API Reference](docs/api_reference.md)
- [Usage Examples](docs/examples.md)  
- [Theoretical Background](docs/theory.md)

## License

Copyright of the Board of Trustees of Johns Hopkins University

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Authors

- Original MATLAB implementation: Sairam Geethanath, Ph.D.
- Python translation, Leo Kinyera, B.S.
- Python code review, Sairam Geethanath, Ph.D.

