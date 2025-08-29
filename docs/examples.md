# Usage Examples

## Basic SAR Calculation

### Simple SAR Check for a Sequence

```python
import pypulseq as pp
from sar4seq import SAR4seq

# Load a sequence
seq = pp.Sequence()
seq.read('path/to/sequence.seq')

# Calculate SAR for 70kg patient
rf_power_body, rf_power_head, sar_predicted = SAR4seq(
    seq_path='path/to/sequence.seq',
    seq=seq,
    sample_weight=70.0
)

print(f"RF Power - Body: {rf_power_body:.2f}W")
print(f"RF Power - Head: {rf_power_head:.2f}W") 
print(f"Predicted SAR: {sar_predicted:.2f}W/kg")
```

### SAR Calculation with Default Parameters

```python
from sar4seq import SAR4seq

# Use default parameters (40kg patient, default sequence)
rf_body, rf_head, sar = SAR4seq()

# Check if SAR is within limits
sar_limit = 4.0  # W/kg for whole body 6-minute average
if sar < sar_limit:
    print(f"SAR {sar:.2f}W/kg is within limits ({sar_limit}W/kg)")
else:
    print(f"WARNING: SAR {sar:.2f}W/kg exceeds limit ({sar_limit}W/kg)")
```

## Q-Matrix Generation

### Generate Global Q-Matrices

```python
from sar4seq import Q_mat_gen
from sar4seq.utils.get_emmodel import get_EMmodel

# Load EM model data
model = get_EMmodel('path/to/em_model_directory')

# Generate global Q-matrices
Q_global = Q_mat_gen('Global', model, qmat_write=True)

print(f"Whole body Q-matrix shape: {Q_global['Qtmf'].shape}")
print(f"Head Q-matrix shape: {Q_global['Qhmf'].shape}")
```

### Generate Local Q-Matrices

```python
# Generate local Q-matrices (computationally intensive)
Q_local = Q_mat_gen('Local', model, qmat_write=True)

print(f"Local matrices shape: {Q_local['local_matrices'].shape}")
print(f"Number of observation points: {Q_local['local_matrices'].shape[0]}")
```

### Create Dummy EM Model for Testing

```python
from sar4seq.utils.get_emmodel import create_dummy_em_model

# Create test model
model = create_dummy_em_model(grid_shape=(32, 32, 16), num_coils=8)

# Generate Q-matrices with test data
Q = Q_mat_gen('Global', model)
```

## VOP (Virtual Observation Points)

### Generate VOPs from Local Q-Matrices

```python
from sar4seq import VOP_Qmatrices_v3

# Generate VOPs from saved local Q-matrices
vop_results = VOP_Qmatrices_v3(
    Q_local_file='LocalQ.mat',
    max_vops=200,
    Nc=8
)

print(f"Generated {vop_results['num_vops']} VOPs")
print(f"Compression ratio: {vop_results['original_points']/vop_results['num_vops']:.1f}:1")
```

### VOP Statistics and Visualization

```python
from sar4seq.vop_qmatrices_v3 import plot_vop_statistics

# Plot VOP statistics
plot_vop_statistics(vop_results)

# Save VOP results
from sar4seq.vop_qmatrices_v3 import save_vop_results
save_vop_results(vop_results, 'vop_results.mat')
```

## Advanced SAR Calculations

### Multi-Channel SAR Calculation

```python
import numpy as np
from sar4seq.utils.calc_sar import calc_SAR_multichannel

# Multi-channel RF signal (8 channels, 1000 time points)
I_multichannel = np.random.complex128((8, 1000))

# Calculate SAR with phase offsets
tx_phases = np.linspace(0, 2*np.pi, 8)
sar_value = calc_SAR_multichannel(Q_global['Qtmf'], I_multichannel, 70.0, tx_phases)

print(f"Multi-channel SAR: {sar_value:.3f} W/kg")
```

### Local SAR Map Generation

```python
from sar4seq.utils.calc_sar import calc_local_SAR_map

# Generate local SAR map
if 'local_matrices' in Q_local:
    sar_map = calc_local_SAR_map(
        Q_local['local_matrices'], 
        I_multichannel, 
        model['mass_cell']
    )
    
    # Visualize maximum intensity projection
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(np.max(sar_map, axis=2), cmap='hot')
    plt.colorbar(label='SAR (W/kg)')
    plt.title('Local SAR Map (MIP)')
    plt.show()
```

## Sequence Creation

### Create TSE Sequence

```python
from sar4seq import write_TSE_500ms

# Create standard TSE sequence
seq = write_TSE_500ms('tse_500ms.seq')

# Calculate SAR for the created sequence  
rf_body, rf_head, sar = SAR4seq(seq_path='tse_500ms.seq', sample_weight=70)
```

### Create Custom TSE Sequence

```python
from sar4seq.write_tse_500ms import write_custom_TSE

# Create custom TSE with different parameters
custom_seq = write_custom_TSE(
    TR=1000e-3,      # 1 second TR
    TE_eff=80e-3,    # 80ms effective TE
    necho=8,         # 8 echo train length
    Nx=256, Ny=256,  # Higher resolution
    fov=300e-3,      # 30cm FOV
    output_filename='custom_tse.seq'
)
```

## SAR Monitoring and Compliance

### Real-time SAR Monitoring

```python
from sar4seq.utils.do_sw_sar import do_sw_sar, check_iec_compliance

# Simulate SAR time series
import numpy as np
time_points = np.linspace(0, 600, 1000)  # 10 minutes
sar_values = 2.0 + 0.5 * np.sin(time_points/60) + 0.2 * np.random.randn(1000)

# Define SAR limits
limits = {
    'ten_sec_wbg': 8.0,   # 10 second whole body limit
    'six_min_wbg': 4.0    # 6 minute whole body limit
}

# Check SAR compliance
results = do_sw_sar(sar_values, time_points, limits, [10, 360])

if results['compliance']:
    print("Sequence is SAR compliant")
else:
    print("SAR violations detected:")
    for violation in results['violations']:
        print(f"  {violation['type']}: {violation['value']:.2f} W/kg > {violation['limit']:.2f} W/kg")
```

### IEC Compliance Check

```python
# Check IEC 60601-2-33 compliance
sar_data = {
    '10s': 7.5,    # 10 second SAR
    '6min': 3.8    # 6 minute SAR
}

compliance = check_iec_compliance(sar_data, 'wholebody')

if compliance['compliant']:
    print("IEC compliant")
    for window, margin in compliance['margins'].items():
        print(f"  {window} margin: {margin:.2f} W/kg")
else:
    print("IEC violations:")
    for violation in compliance['violations']:
        print(f"  {violation['window']}: {violation['excess']:.2f} W/kg over limit")
```

## File I/O Operations

### Reading and Writing Q-Matrices

```python
from sar4seq.utils.read_qmat import read_qmat
from sar4seq.utils.write_qmat import write_qmat

# Read Q-matrix from MATLAB file
Q_loaded = read_qmat('Qglobal.mat', format='mat')

# Write Q-matrix to different formats
write_qmat(Q_loaded, 'Global', 'qmat_copy.mat', format='mat')
write_qmat(Q_loaded, 'Global', 'qmat_copy.hdf5', format='hdf5')
write_qmat(Q_loaded, 'Global', 'qmat_copy.npz', format='npz')
```

### Working with EM Model Data

```python
from sar4seq.utils.get_emmodel import save_em_model, validate_em_model

# Validate loaded model
try:
    validate_em_model(model)
    print("EM model is valid")
except ValueError as e:
    print(f"EM model validation failed: {e}")

# Save model to new location
save_em_model(model, 'backup_em_model/')
```

## Error Handling and Debugging

### Robust SAR Calculation

```python
try:
    rf_body, rf_head, sar = SAR4seq(
        seq_path='sequence.seq',
        sample_weight=70
    )
    
    # Check for reasonable values
    if sar > 20:  # Unreasonably high SAR
        print(f"Warning: Very high SAR detected ({sar:.2f} W/kg)")
    elif sar < 0:
        print("Error: Negative SAR calculated")
    else:
        print(f"SAR calculation successful: {sar:.2f} W/kg")
        
except FileNotFoundError:
    print("Sequence file not found")
except ValueError as e:
    print(f"SAR calculation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Memory Management for Large Datasets

```python
import gc

# For large EM models, manage memory carefully
try:
    print("Loading large EM model...")
    model = get_EMmodel('large_model_directory/')
    
    print("Generating Q-matrices...")
    Q = Q_mat_gen('Local', model)
    
    # Clear large model from memory
    del model
    gc.collect()
    
    print("Generating VOPs...")
    vop_results = VOP_Qmatrices_v3(Q_local_data=Q)
    
    print(f"Successfully processed {vop_results['num_vops']} VOPs")
    
except MemoryError:
    print("Insufficient memory for large dataset")
    print("Consider using a smaller model or more RAM")
```
