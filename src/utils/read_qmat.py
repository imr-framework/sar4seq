"""
Q-Matrix File Reading Utilities

Functions for reading Q-matrix data from files.
"""

import numpy as np
import scipy.io as sio
import os
import h5py


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
