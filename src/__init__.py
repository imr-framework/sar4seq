"""
SAR4seq source module

Contains the core implementation of SAR4seq functionality.
"""

# Import modules with error handling for optional dependencies
try:
    from .sar4seq import SAR4seq
    __all__ = ['SAR4seq']
except ImportError as e:
    print(f"Warning: Could not import SAR4seq: {e}")
    __all__ = []

try:
    from .q_mat_gen import Q_mat_gen
    __all__.append('Q_mat_gen')
except ImportError as e:
    print(f"Warning: Could not import Q_mat_gen: {e}")

# Export what was successfully imported
globals().update({name: globals()[name] for name in __all__ if name in globals()})