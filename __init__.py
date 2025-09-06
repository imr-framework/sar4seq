"""
SAR4seq Python Package

A Python implementation of the SAR4seq MATLAB toolbox for computing RF safety metrics 
for Pulseq sequences.

Main Functions:
- SAR4seq: Main function for computing SAR metrics
- Q_mat_gen: Q-matrix generation for SAR calculations
- VOP_Qmatrices_v3: Virtual Observation Points implementation

Copyright of the Board of Trustees of Columbia University in the City of New York
"""

__version__ = "1.0.0"
__author__ = ["Sairam Geethanath, Ph.D. (Original MATLAB)", "Leo Kinyera, Student. (Python Translation)"]

from .src.sar4seq import SAR4seq_legacy
from .src.q_mat_gen import Q_mat_gen
from .src.vop_qmatrices_v3 import VOP_Qmatrices_v3
from .src.write_tse_500ms import write_TSE_500ms

__all__ = ['SAR4seq_legacy', 'Q_mat_gen', 'VOP_Qmatrices_v3', 'write_TSE_500ms']
