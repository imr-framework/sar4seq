"""
Q-Matrix File Reading Utilities

Functions for reading Q-matrix data from files.
"""

import numpy as np
import scipy.io as sio
import os
import h5py
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional


def read_qmat(filename, format='mat'):
    """
    Read Q-matrix data from file
    
    Parameters
    ----------
    filename : str
        Path to Q-matrix file
    format : str
        File format ('mat', 'hdf5', 'npz')
        
    Returns
    -------
    dict
        Q-matrix data
    """
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Q-matrix file not found: {filename}")
    
    if format.lower() == 'mat':
        return read_qmat_matlab(filename)
    elif format.lower() == 'hdf5':
        return read_qmat_hdf5(filename)
    elif format.lower() == 'npz':
        return read_qmat_numpy(filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


def read_qmat_matlab(filename):
    """
    Read Q-matrix from MATLAB .mat file
    
    Parameters
    ----------
    filename : str
        Path to .mat file
        
    Returns
    -------
    dict
        Q-matrix data
    """
    
    try:
        data = sio.loadmat(filename)
        
        # Handle different possible structures
        if 'Q' in data:
            return data['Q']
        elif 'Qavg' in data:
            return data['Qavg']
        else:
            # Return all non-private keys
            qmat_data = {key: value for key, value in data.items() 
                        if not key.startswith('__')}
            return qmat_data
            
    except Exception as e:
        print(f"Error reading MATLAB file {filename}: {e}")
        raise


def read_qmat_hdf5(filename):
    """
    Read Q-matrix from HDF5 file
    
    Parameters
    ----------
    filename : str
        Path to .hdf5 file
        
    Returns
    -------
    dict
        Q-matrix data
    """
    
    qmat_data = {}
    
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            qmat_data[key] = f[key][...]
            
            # Handle complex data
            if 'real' in f[key].attrs and 'imag' in f[key].attrs:
                real_part = f[key][...]
                imag_part = f[f[key].attrs['imag']][...]
                qmat_data[key] = real_part + 1j * imag_part
    
    return qmat_data


def read_qmat_numpy(filename):
    """
    Read Q-matrix from NumPy .npz file
    
    Parameters
    ----------
    filename : str
        Path to .npz file
        
    Returns
    -------
    dict
        Q-matrix data
    """
    
    data = np.load(filename)
    return dict(data)


def load_avgq_matrix(filename):
    """
    Load averaged Q-matrix data with index information
    
    Parameters
    ----------
    filename : str
        Path to Q-matrix file
        
    Returns
    -------
    dict
        Dictionary containing:
        - avg: Averaged Q-matrices
        - index: Spatial indices
        - other metadata
    """
    
    if filename.endswith('.mat'):
        data = sio.loadmat(filename)
        
        # Look for common structures
        if 'Qavg' in data:
            qdata = data['Qavg']
            if isinstance(qdata, np.ndarray) and qdata.dtype == object:
                # Handle MATLAB struct
                result = {}
                for field in qdata.dtype.names:
                    result[field] = qdata[field][0, 0]
                return result
            else:
                return {'avg': qdata}
        else:
            return data
    else:
        return read_qmat(filename)


def extract_q_point(qavg_data, coordinates):
    """
    Extract Q-matrix at specific coordinates
    
    Parameters
    ----------
    qavg_data : dict
        Q-matrix data with 'avg' and 'index' fields
    coordinates : tuple
        (x, y, z) coordinates
        
    Returns
    -------
    numpy.ndarray
        Q-matrix at specified point
    """
    
    if 'index' not in qavg_data or 'avg' not in qavg_data:
        raise ValueError("Q-matrix data must contain 'index' and 'avg' fields")
    
    x, y, z = coordinates
    indices = qavg_data['index']
    
    # Find matching index
    if indices.shape[0] >= 3:
        x_match = np.where(indices[0, :] == x)[0]
        y_match = np.where(indices[1, :] == y)[0]
        z_match = np.where(indices[2, :] == z)[0]
        
        # Find intersection
        xy_match = np.intersect1d(x_match, y_match)
        xyz_match = np.intersect1d(xy_match, z_match)
        
        if len(xyz_match) > 0:
            point_idx = xyz_match[0]
            return qavg_data['avg'][point_idx, :, :]
        else:
            raise ValueError(f"No Q-matrix found at coordinates {coordinates}")
    else:
        raise ValueError("Invalid index structure in Q-matrix data")


def validate_qmat_data(qmat_data):
    """
    Validate Q-matrix data structure
    
    Parameters
    ----------
    qmat_data : dict
        Q-matrix data to validate
        
    Returns
    -------
    bool
        True if valid
    """
    
    if not isinstance(qmat_data, dict):
        raise TypeError("Q-matrix data must be a dictionary")
    
    # Check for required fields
    if 'Qtmf' in qmat_data or 'Qhmf' in qmat_data:
        # Global Q-matrix format
        for key in ['Qtmf', 'Qhmf']:
            if key in qmat_data:
                Q = qmat_data[key]
                if not isinstance(Q, np.ndarray):
                    raise TypeError(f"{key} must be a numpy array")
                if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
                    raise ValueError(f"{key} must be a square matrix")
    
    elif 'avg' in qmat_data:
        # Local Q-matrix format
        Q_avg = qmat_data['avg']
        if not isinstance(Q_avg, np.ndarray):
            raise TypeError("avg field must be a numpy array")
        if Q_avg.ndim < 3:
            raise ValueError("avg field must have at least 3 dimensions")
    
    print("Q-matrix data validation passed")
    return True


def load_tissue_data():
    """
    Load tissue types data from MATLAB file
    
    Returns:
    --------
    tissue_data : numpy.ndarray
        3D tissue array with shape (x, y, z)
    tissue_properties : dict
        Dictionary mapping tissue values to electromagnetic properties
    """
    
    tissue_file = Path(__file__).parent.parent / "data" / "Tissue_types.mat"
    
    try:
        data = sio.loadmat(str(tissue_file))
        tissue_data = data['Tissue_types']
        
        print(f"📡 Loaded tissue data:")
        print(f"  Shape: {tissue_data.shape}")
        print(f"  Dtype: {tissue_data.dtype}")
        print(f"  Total voxels: {tissue_data.size:,}")
        
        # Define tissue electromagnetic properties
        # Based on typical values at 3T (128 MHz)
        tissue_properties = {
            # Air/background (conductivity ≈ 0, permittivity ≈ 1)
            0: {'conductivity': 0.0, 'permittivity': 1.0, 'density': 0.0, 'name': 'Air'},
            
            # Muscle (conductivity ≈ 0.5-0.8 S/m, permittivity ≈ 50-80)
            1: {'conductivity': 0.65, 'permittivity': 65.0, 'density': 1.05, 'name': 'Muscle'},
            
            # Fat (conductivity ≈ 0.04-0.08 S/m, permittivity ≈ 10-15)
            2: {'conductivity': 0.06, 'permittivity': 12.0, 'density': 0.92, 'name': 'Fat'},
            
            # Bone (conductivity ≈ 0.02-0.06 S/m, permittivity ≈ 10-20)
            3: {'conductivity': 0.04, 'permittivity': 15.0, 'density': 1.85, 'name': 'Bone'},
            
            # Brain (conductivity ≈ 0.4-0.7 S/m, permittivity ≈ 40-60)
            4: {'conductivity': 0.55, 'permittivity': 50.0, 'density': 1.04, 'name': 'Brain'},
            
            # Blood (conductivity ≈ 1.2-1.6 S/m, permittivity ≈ 60-80)
            5: {'conductivity': 1.4, 'permittivity': 70.0, 'density': 1.06, 'name': 'Blood'},
            
            # Lung (conductivity ≈ 0.2-0.4 S/m, permittivity ≈ 20-40)
            6: {'conductivity': 0.3, 'permittivity': 30.0, 'density': 0.5, 'name': 'Lung'},
            
            # Liver (conductivity ≈ 0.4-0.6 S/m, permittivity ≈ 40-60)
            7: {'conductivity': 0.5, 'permittivity': 50.0, 'density': 1.06, 'name': 'Liver'},
        }
        
        # Analyze tissue distribution
        unique_tissues = np.unique(np.real(tissue_data))
        print(f"  Unique tissue types found: {len(unique_tissues)}")
        for tissue_id in unique_tissues[:10]:
            tissue_id = int(tissue_id)
            count = np.sum(np.real(tissue_data) == tissue_id)
            percentage = count / tissue_data.size * 100
            tissue_name = tissue_properties.get(tissue_id, {}).get('name', f'Unknown_{tissue_id}')
            print(f"    Type {tissue_id} ({tissue_name}): {count:,} voxels ({percentage:.1f}%)")
        
        return tissue_data, tissue_properties
        
    except Exception as e:
        print(f"Could not load tissue data: {e}")
        print(f"Using synthetic tissue model instead")
        return None, None
    

def load_clinical_qmatrices(qmat=None):
    """
    Load Q-matrices for clinical SAR computation
    
    Parameters
    ----------
    qmat : str, optional
        Path to Q-matrix file. If None, will search default locations.
    
    Returns
    -------
    dict
        Q-matrices for different body regions and analysis types
    """
    
    print("\n🔬 Loading Q-matrices for clinical assessment...")
    
    # If specific Q-matrix path provided, use it first
    if qmat is not None:
        qmat_paths = [Path(qmat)]
    else:
        # Try to load existing Q-matrices from default locations
        qmat_paths = [
            Path(__file__).parent / 'test_qmat.mat',
            Path(__file__).parent.parent / 'data' / 'Qmat.mat',
            Path(__file__).parent.parent / 'data' / 'QGlobal.mat'
        ]
    
    Q_matrices = {}
    
    for qmat_path in qmat_paths:
        if qmat_path.exists():
            try:
                Q_data = sio.loadmat(str(qmat_path))
                print(f"  📁 Loaded Q-matrix from: {qmat_path.name}")
                
                # Extract relevant matrices
                if 'Qtmf' in Q_data and 'Qhmf' in Q_data:
                    Q_matrices['whole_body'] = Q_data['Qtmf']
                    Q_matrices['head'] = Q_data['Qhmf']
                    print(f"    Whole body Q: {Q_data['Qtmf'].shape}")
                    print(f"    Head Q: {Q_data['Qhmf'].shape}")
                    break
                elif 'QGlobal' in Q_data:
                    Q_matrices['global'] = Q_data['QGlobal']
                    print(f"    Global Q: {Q_data['QGlobal'].shape}")
                    break
                    
            except Exception as e:
                print(f"  ⚠️  Error loading {qmat_path.name}: {e}")
                continue
        else:
            if qmat is not None:
                # If specific path was provided but doesn't exist, raise error
                raise FileNotFoundError(f"Specified Q-matrix file not found: {qmat}")
            else:
                # If default path doesn't exist, just continue searching
                continue
    
    # Create default Q-matrices if none found
    if not Q_matrices:
        print("  ⚠️  No Q-matrices found. Creating default matrices...")
        n_channels = 8
        Q_matrices = {
            'whole_body': np.eye(n_channels, dtype=complex) * 1e-6,
            'head': np.eye(n_channels, dtype=complex) * 5e-7
        }
        print(f"    Created default Q-matrices ({n_channels}x{n_channels})")
    
    return Q_matrices


def load_q_matrix_with_validation(q_matrix_path: str,
                                 expected_channels: int = None,
                                 expected_spatial_points: int = None,
                                 **kwargs) -> Dict[str, Any]:
    """
    Load Q-matrix with comprehensive validation and metadata extraction.
    
    Args:
        q_matrix_path: Path to Q-matrix file
        expected_channels: Expected number of RF channels
        expected_spatial_points: Expected spatial resolution
        
    Returns:
        Dictionary containing Q-matrix data and metadata
    """
    if not os.path.exists(q_matrix_path):
        raise FileNotFoundError(f"Q-matrix file not found: {q_matrix_path}")
    
    # Determine file format from extension
    file_ext = Path(q_matrix_path).suffix.lower()
    
    try:
        if file_ext == '.mat':
            data = read_qmat_matlab(q_matrix_path)
        elif file_ext in ['.h5', '.hdf5']:
            data = read_qmat_hdf5(q_matrix_path)
        elif file_ext == '.npz':
            data = read_qmat_numpy(q_matrix_path)
        else:
            # Try to auto-detect format
            data = read_qmat(q_matrix_path)
    except Exception as e:
        raise ValueError(f"Failed to load Q-matrix from {q_matrix_path}: {e}")
    
    # Extract Q-matrix array
    q_matrix = _extract_q_matrix_array(data)
    
    # Validate Q-matrix properties
    validation_results = validate_q_matrix(q_matrix, expected_channels, expected_spatial_points)
    
    # Package results
    result = {
        'q_matrix': q_matrix,
        'file_path': q_matrix_path,
        'file_format': file_ext,
        'validation': validation_results,
        'metadata': _extract_q_matrix_metadata(data, q_matrix)
    }
    
    return result

def _extract_q_matrix_array(data: Dict[str, Any]) -> np.ndarray:
    """Extract Q-matrix array from loaded data structure."""
    # Common Q-matrix field names (including project-specific ones)
    q_field_names = [
        'Q', 'Qmat', 'Q_matrix', 'q_matrix', 'Q_uncompressed', 'Q_vop',
        'Qtmf', 'Qhmf'  # Project-specific field names
    ]
    
    q_matrix = None
    
    # Try to find Q-matrix in data
    for field_name in q_field_names:
        if field_name in data:
            candidate = data[field_name]
            if isinstance(candidate, np.ndarray):
                # For this project, prefer Qtmf if available (torso/body Q-matrix)
                if field_name == 'Qtmf':
                    q_matrix = candidate
                    break
                elif q_matrix is None:
                    q_matrix = candidate
    
    if q_matrix is None:
        # If no standard field found, look for the largest array
        for key, value in data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                if q_matrix is None or value.size > q_matrix.size:
                    q_matrix = value
    
    if q_matrix is None:
        available_keys = [k for k in data.keys() if not k.startswith('__')]
        raise ValueError(f"Could not locate Q-matrix array in data. Available keys: {available_keys}")
    
    # Handle different Q-matrix formats
    if q_matrix.ndim == 2:
        # 2D matrix - assume single channel, expand to 3D
        q_matrix = q_matrix[np.newaxis, :, :]
    elif q_matrix.ndim != 3:
        raise ValueError(f"Q-matrix must be 2D or 3D, got {q_matrix.ndim}D with shape {q_matrix.shape}")
    
    return q_matrix

def validate_q_matrix(q_matrix: np.ndarray,
                     expected_channels: int = None,
                     expected_spatial_points: int = None) -> Dict[str, Any]:
    """
    Validate Q-matrix array properties and structure.
    
    Args:
        q_matrix: Q-matrix array to validate
        expected_channels: Expected number of channels
        expected_spatial_points: Expected spatial resolution
        
    Returns:
        Dictionary containing validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'properties': {}
    }
    
    # Check basic properties
    if not isinstance(q_matrix, np.ndarray):
        validation['errors'].append("Q-matrix is not a numpy array")
        validation['is_valid'] = False
        return validation
    
    # Check dimensions
    if q_matrix.ndim != 3:
        validation['errors'].append(f"Q-matrix must be 3D, got {q_matrix.ndim}D")
        validation['is_valid'] = False
    else:
        n_channels, n_spatial1, n_spatial2 = q_matrix.shape
        
        # Check if square spatial dimensions
        if n_spatial1 != n_spatial2:
            validation['errors'].append(f"Q-matrix spatial dimensions not square: {n_spatial1} x {n_spatial2}")
            validation['is_valid'] = False
        
        validation['properties'].update({
            'n_channels': n_channels,
            'n_spatial_points': n_spatial1,
            'total_elements': q_matrix.size,
            'memory_gb': q_matrix.nbytes / (1024**3)
        })
        
        # Check expected dimensions
        if expected_channels is not None and n_channels != expected_channels:
            validation['warnings'].append(f"Channel count mismatch: expected {expected_channels}, got {n_channels}")
        
        if expected_spatial_points is not None and n_spatial1 != expected_spatial_points:
            validation['warnings'].append(f"Spatial resolution mismatch: expected {expected_spatial_points}, got {n_spatial1}")
    
    # Check data type
    if q_matrix.dtype.kind not in ['f', 'c']:  # float or complex
        validation['warnings'].append(f"Q-matrix has unusual dtype: {q_matrix.dtype}")
    
    validation['properties']['dtype'] = str(q_matrix.dtype)
    
    # Check for reasonable values
    if validation['is_valid']:
        finite_mask = np.isfinite(q_matrix)
        finite_ratio = np.sum(finite_mask) / q_matrix.size
        
        if finite_ratio < 0.9:
            validation['warnings'].append(f"Q-matrix contains many non-finite values ({(1-finite_ratio)*100:.1f}%)")
        
        if finite_ratio > 0:
            finite_values = q_matrix[finite_mask]
            validation['properties'].update({
                'min_value': np.min(finite_values),
                'max_value': np.max(finite_values),
                'mean_value': np.mean(finite_values),
                'finite_ratio': finite_ratio
            })
            
            # Check for symmetry (Hermitian property)
            if q_matrix.dtype.kind == 'c':  # Complex matrix
                for ch in range(min(n_channels, 3)):  # Check first few channels
                    Q_ch = q_matrix[ch]
                    if not np.allclose(Q_ch, Q_ch.conj().T, rtol=1e-10, atol=1e-10):
                        validation['warnings'].append(f"Q-matrix channel {ch} is not Hermitian")
                        break
    
    return validation

def _extract_q_matrix_metadata(data: Dict[str, Any], q_matrix: np.ndarray) -> Dict[str, Any]:
    """Extract metadata from Q-matrix file."""
    metadata = {
        'file_structure': list(data.keys()),
        'q_matrix_shape': q_matrix.shape,
        'q_matrix_dtype': str(q_matrix.dtype)
    }
    
    # Look for common metadata fields
    metadata_fields = [
        'frequency', 'field_strength', 'vop_matrix', 'tissue_mask',
        'resolution', 'phantom_type', 'creation_date', 'software_version'
    ]
    
    for field in metadata_fields:
        if field in data:
            metadata[field] = data[field]
    
    return metadata

def load_vop_compression_data(vop_path: str) -> Dict[str, Any]:
    """
    Load VOP (Virtual Observation Points) compression data.
    
    Args:
        vop_path: Path to VOP data file
        
    Returns:
        Dictionary containing VOP matrix and metadata
    """
    if not os.path.exists(vop_path):
        raise FileNotFoundError(f"VOP file not found: {vop_path}")
    
    file_ext = Path(vop_path).suffix.lower()
    
    try:
        if file_ext == '.mat':
            data = sio.loadmat(vop_path)
        elif file_ext in ['.h5', '.hdf5']:
            data = {}
            with h5py.File(vop_path, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
        elif file_ext == '.npz':
            data = dict(np.load(vop_path))
        else:
            raise ValueError(f"Unsupported VOP file format: {file_ext}")
    except Exception as e:
        raise ValueError(f"Failed to load VOP data from {vop_path}: {e}")
    
    # Extract VOP matrix
    vop_field_names = ['VOP', 'vop_matrix', 'V', 'compression_matrix']
    vop_matrix = None
    
    for field_name in vop_field_names:
        if field_name in data:
            vop_matrix = data[field_name]
            break
    
    if vop_matrix is None:
        raise ValueError("Could not locate VOP matrix in data")
    
    # Validate VOP matrix
    if vop_matrix.ndim != 2:
        raise ValueError(f"VOP matrix must be 2D, got {vop_matrix.ndim}D")
    
    result = {
        'vop_matrix': vop_matrix,
        'n_vop_points': vop_matrix.shape[0],
        'n_spatial_points': vop_matrix.shape[1],
        'compression_ratio': vop_matrix.shape[1] / vop_matrix.shape[0],
        'file_path': vop_path,
        'metadata': data
    }
    
    return result

# Add to __all__ exports
__all__ = getattr(__import__(__name__), '__all__', []) + [
    'load_q_matrix_with_validation',
    'validate_q_matrix', 
    'load_vop_compression_data'
]

if __name__ == '__main__':
    Q_matrices = load_clinical_qmatrices(qmat='/lhome/ext/i3m121/i3m1211/SAR/SAR4seq_python/data/Qmat.mat')