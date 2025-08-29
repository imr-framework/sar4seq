"""
Q-Matrix File Writing Utilities

Functions for writing Q-matrix data to files.
"""

import numpy as np
import scipy.io as sio
import os
import h5py


def write_qmat(Q, sar_type, filename=None, format='mat'):
    """
    Write Q-matrix data to file
    
    Parameters
    ----------
    Q : dict or numpy.ndarray
        Q-matrix data to write
    sar_type : str
        Type of SAR calculation ('Global' or 'Local')
    filename : str, optional
        Output filename. If None, auto-generate based on sar_type
    format : str
        Output format ('mat', 'hdf5', 'npz')
        
    Returns
    -------
    bool
        Success status
    """
    
    if filename is None:
        if sar_type.lower() == 'global':
            filename = 'Qglobal'
        elif sar_type.lower() == 'local':
            filename = 'Qlocal'
        else:
            filename = 'Qmatrix'
    
    # Add extension if not present
    if format.lower() == 'mat' and not filename.endswith('.mat'):
        filename += '.mat'
    elif format.lower() == 'hdf5' and not filename.endswith(('.hdf5', '.h5')):
        filename += '.hdf5'
    elif format.lower() == 'npz' and not filename.endswith('.npz'):
        filename += '.npz'
    
    try:
        if format.lower() == 'mat':
            return write_qmat_matlab(Q, filename)
        elif format.lower() == 'hdf5':
            return write_qmat_hdf5(Q, filename)
        elif format.lower() == 'npz':
            return write_qmat_numpy(Q, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        print(f"Error writing Q-matrix file {filename}: {e}")
        return False


def write_qmat_matlab(Q, filename):
    """
    Write Q-matrix to MATLAB .mat file
    
    Parameters
    ----------
    Q : dict or numpy.ndarray
        Q-matrix data
    filename : str
        Output filename
        
    Returns
    -------
    bool
        Success status
    """
    
    try:
        if isinstance(Q, dict):
            sio.savemat(filename, Q)
        else:
            sio.savemat(filename, {'Q': Q})
        
        print(f"Q-matrix saved to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving MATLAB file: {e}")
        return False


def write_qmat_hdf5(Q, filename):
    """
    Write Q-matrix to HDF5 file
    
    Parameters
    ----------
    Q : dict or numpy.ndarray
        Q-matrix data
    filename : str
        Output filename
        
    Returns
    -------
    bool
        Success status
    """
    
    try:
        with h5py.File(filename, 'w') as f:
            if isinstance(Q, dict):
                for key, value in Q.items():
                    if np.iscomplexobj(value):
                        # Store complex data as real and imaginary parts
                        real_name = f"{key}_real"
                        imag_name = f"{key}_imag"
                        f.create_dataset(real_name, data=np.real(value))
                        f.create_dataset(imag_name, data=np.imag(value))
                        f[real_name].attrs['imag'] = imag_name
                    else:
                        f.create_dataset(key, data=value)
            else:
                if np.iscomplexobj(Q):
                    f.create_dataset('Q_real', data=np.real(Q))
                    f.create_dataset('Q_imag', data=np.imag(Q))
                    f['Q_real'].attrs['imag'] = 'Q_imag'
                else:
                    f.create_dataset('Q', data=Q)
        
        print(f"Q-matrix saved to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving HDF5 file: {e}")
        return False


def write_qmat_numpy(Q, filename):
    """
    Write Q-matrix to NumPy .npz file
    
    Parameters
    ----------
    Q : dict or numpy.ndarray
        Q-matrix data
    filename : str
        Output filename
        
    Returns
    -------
    bool
        Success status
    """
    
    try:
        if isinstance(Q, dict):
            np.savez_compressed(filename, **Q)
        else:
            np.savez_compressed(filename, Q=Q)
        
        print(f"Q-matrix saved to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving NumPy file: {e}")
        return False


def write_qmat_custom_format(Q, filename):
    """
    Write Q-matrix in custom binary format for fast loading
    
    Parameters
    ----------
    Q : dict
        Q-matrix data
    filename : str
        Output filename
        
    Returns
    -------
    bool
        Success status
    """
    
    try:
        # Custom format with header
        with open(filename, 'wb') as f:
            # Write header
            header = {
                'format_version': 1,
                'data_type': 'qmatrix',
                'num_matrices': len(Q) if isinstance(Q, dict) else 1
            }
            
            # Write header size and header
            header_str = str(header).encode('utf-8')
            f.write(len(header_str).to_bytes(4, byteorder='little'))
            f.write(header_str)
            
            # Write data
            if isinstance(Q, dict):
                for key, matrix in Q.items():
                    # Write key
                    key_bytes = key.encode('utf-8')
                    f.write(len(key_bytes).to_bytes(4, byteorder='little'))
                    f.write(key_bytes)
                    
                    # Write matrix shape
                    shape_bytes = np.array(matrix.shape, dtype=np.int32).tobytes()
                    f.write(len(shape_bytes).to_bytes(4, byteorder='little'))
                    f.write(shape_bytes)
                    
                    # Write matrix data
                    data_bytes = matrix.tobytes()
                    f.write(len(data_bytes).to_bytes(4, byteorder='little'))
                    f.write(data_bytes)
            else:
                # Single matrix
                key_bytes = b'Q'
                f.write(len(key_bytes).to_bytes(4, byteorder='little'))
                f.write(key_bytes)
                
                shape_bytes = np.array(Q.shape, dtype=np.int32).tobytes()
                f.write(len(shape_bytes).to_bytes(4, byteorder='little'))
                f.write(shape_bytes)
                
                data_bytes = Q.tobytes()
                f.write(len(data_bytes).to_bytes(4, byteorder='little'))
                f.write(data_bytes)
        
        print(f"Q-matrix saved in custom format to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving custom format file: {e}")
        return False


def backup_qmat_file(filename):
    """
    Create backup of existing Q-matrix file
    
    Parameters
    ----------
    filename : str
        Filename to backup
        
    Returns
    -------
    str or None
        Backup filename if created, None otherwise
    """
    
    if os.path.exists(filename):
        backup_name = filename + '.backup'
        counter = 1
        while os.path.exists(backup_name):
            backup_name = f"{filename}.backup{counter}"
            counter += 1
        
        try:
            import shutil
            shutil.copy2(filename, backup_name)
            print(f"Backup created: {backup_name}")
            return backup_name
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    
    return None
