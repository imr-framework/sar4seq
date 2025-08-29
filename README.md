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

## SAR with VOP when `requires_gpu=False`
Run:
```bash
uv run src/vop_sar4seq.py
```

### Dependencies

- numpy
- scipy
- matplotlib
- h5py (for .mat file compatibility)
- pypulseq (for Pulseq sequence handling)

## Quick Start

```python
from sar4seq import SAR4seq
import pypulseq as pp

# Load sequence
seq = pp.Sequence()
seq.read('path/to/sequence.seq')

# Compute SAR
rf_power_avg, rf_head_avg, sar_predicted = SAR4seq(
    seq_path='path/to/sequence.seq',
    seq=seq,
    sample_weight=70.0  # kg
)

print(f"Time averaged RF power - Body: {rf_power_avg:.2f}W")
print(f"Time averaged RF power - Head: {rf_head_avg:.2f}W")
print(f"Predicted SAR: {sar_predicted:.2f}W/kg")
```

## Module Structure

```
sar4seq_python/
├── __init__.py              # Main package init
├── sar4seq.py               # Main SAR computation function
├── q_mat_gen.py             # Q-matrix generation
├── vop_qmatrices_v3.py      # VOP implementation
├── write_tse_500ms.py       # TSE sequence writer
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── calc_sar.py          # SAR calculation utilities
│   ├── gen_qpwr.py          # Q-matrix power generation
│   ├── gen_e12ptq.py        # 12-point Q generation
│   ├── get_coremat.py       # Core matrix computation
│   ├── get_emmodel.py       # EM model utilities
│   ├── read_qmat.py         # Q-matrix file reader
│   ├── write_qmat.py        # Q-matrix file writer
│   ├── chkcubair_global.py  # Air cube checker
│   └── do_sw_sar.py         # SAR switching utilities
├── data/                    # Data files directory
└── docs/                    # Documentation
    ├── api_reference.md     # API documentation
    ├── examples.md          # Usage examples
    └── theory.md            # Theoretical background
```

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

