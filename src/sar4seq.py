"""
SAR4seq: RF safety metrics computation for Pulseq sequences

This module computes RF safety metrics for Pulseq sequences:
1. Time-averaged RF power for Siemens scanners
2. Whole body SAR prediction for GE scanners (via TOPPE)

The module loads Q-matrices for electromagnetic modeling and processes
RF events in Pulseq sequences to calculate SAR values and verify
compliance with safety limits.

Copyright of the Board of Trustees of Columbia University in the City of New York
"""

from utils.calc_sar import SAR4seq_legacy

rf_power_body, rf_power_head, sar_prediction = SAR4seq_legacy(seq_path='custom_tse_1s.seq')
print("SAR computation completed successfully!")
