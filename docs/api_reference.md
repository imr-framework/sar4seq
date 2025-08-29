# API Reference

## Main Functions

### SAR4seq
```python
SAR4seq(seq_path=None, seq=None, sample_weight=None)
```
Main function for computing RF safety metrics for Pulseq sequences.

**Parameters:**
- `seq_path` (str, optional): Path to Pulseq sequence file
- `seq` (pypulseq.Sequence, optional): Pulseq sequence object
- `sample_weight` (float, optional): Weight of sample in kg (default: 40)

**Returns:**
- `RFwbg_tavg` (float): Time averaged RF power for whole body (W)
- `RFhg_tavg` (float): Time averaged RF power for head (W)  
- `SARwbg_pred` (float): Predicted whole body SAR (W/kg)

### Q_mat_gen
```python
Q_mat_gen(sar_type, model, qmat_write=False)
```
Generate Q-matrices for SAR calculations from electromagnetic field data.

**Parameters:**
- `sar_type` (str): Type of SAR calculation ('Global' or 'Local')
- `model` (dict): EM model data containing Ex, Ey, Ez, tissue_types, etc.
- `qmat_write` (bool): Whether to write Q-matrices to file

**Returns:**
- `Q` (dict): Q-matrix data structure

### VOP_Qmatrices_v3
```python
VOP_Qmatrices_v3(Q_local_file=None, Q_local_data=None, max_vops=500, Nc=8)
```
Implementation of VOP algorithm for efficient SAR calculations.

**Parameters:**
- `Q_local_file` (str, optional): Path to local Q-matrix file
- `Q_local_data` (dict, optional): Local Q-matrix data
- `max_vops` (int): Maximum number of VOPs (default: 500)
- `Nc` (int): Number of RF channels (default: 8)

**Returns:**
- `results` (dict): VOP data including matrices, indices, and maps

## Utility Functions

### calc_SAR
```python
calc_SAR(Q, I, weight)
```
Calculate SAR from Q-matrix and RF signal.

**Parameters:**
- `Q` (numpy.ndarray): Q-matrix for SAR calculation
- `I` (numpy.ndarray): RF signal array
- `weight` (float): Body weight in kg

**Returns:**
- `SAR` (float): SAR value in W/kg

### gen_Qpwr
```python
gen_Qpwr(Ex, Ey, Ez, tissue_types, sigma_by_rhox, mass_cell, sar_type, anatomy)
```
Generate Q-matrices for power calculation from EM field data.

### gen_E12ptQ
```python
gen_E12ptQ(Ex, Ey, Ez, X, sigma_by_rhox)
```
Generate Q-matrix using 12-point cube formulation.

### get_coremat
```python
get_coremat(Q_inds, ind, myu_per=0.01)
```
Compute core matrices for VOP algorithm.

### get_EMmodel
```python
get_EMmodel(dirname=None)
```
Load electromagnetic model data from directory.

### read_qmat / write_qmat
```python
read_qmat(filename, format='mat')
write_qmat(Q, sar_type, filename=None, format='mat')
```
Read and write Q-matrix data files.

## Data Structures

### EM Model Dictionary
```python
model = {
    'Ex': numpy.ndarray,        # Electric field x-component (M,N,P,Nc)
    'Ey': numpy.ndarray,        # Electric field y-component (M,N,P,Nc)  
    'Ez': numpy.ndarray,        # Electric field z-component (M,N,P,Nc)
    'tissue_types': numpy.ndarray,    # Tissue classification (M,N,P)
    'sigma_by_rhox': numpy.ndarray,   # Conductivity/density (M,N,P)
    'mass_cell': numpy.ndarray        # Mass per voxel (M,N,P)
}
```

### Q-Matrix Structure
```python
Q = {
    'Qtmf': numpy.ndarray,      # Whole body Q-matrix (Nc,Nc)
    'Qhmf': numpy.ndarray,      # Head Q-matrix (Nc,Nc)
}
```

### VOP Results Structure
```python
vop_results = {
    'VOP_matrices': numpy.ndarray,    # VOP matrices (Nvop,Nc,Nc)
    'VOP_spatial': numpy.ndarray,     # Spatial VOP array (M,N,P,Nc,Nc)
    'vop_indices': numpy.ndarray,     # Linear indices of VOPs
    'norms': numpy.ndarray,           # Spectral norms of VOPs
    'cluster_sizes': numpy.ndarray,   # Size of each cluster
    'vop_map': numpy.ndarray,         # Spatial VOP map (M,N,P)
    'num_vops': int,                  # Total number of VOPs
    'computation_time': float,        # Computation time in seconds
    'original_points': int            # Original number of observation points
}
```

## Constants

### SAR Limits (W/kg)
- **IEC 60601-2-33 Whole Body:** 4.0 (6 min), 8.0 (10 s)
- **IEC 60601-2-33 Head:** 3.2 (6 min), 6.4 (10 s)
- **IEC 60601-2-33 Extremity:** 10.0 (6 min), 20.0 (10 s)

### Scanner-Specific Factors
- **Siemens B1+ Factor:** 1.32
- **GE B1+ Factor:** 1.1725

### Anatomical Reference Masses (kg)
- **Visible Human Male Total:** 103.45
- **Visible Human Male Head:** 6.024
