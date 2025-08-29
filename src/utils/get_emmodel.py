"""
Electromagnetic Model Utilities

Functions for loading and processing electromagnetic field models.
"""

import numpy as np
import scipy.io as sio
import os
from pathlib import Path


def get_EMmodel(dirname=None):
    """
    Load electromagnetic model data from directory
    
    Parameters
    ----------
    dirname : str, optional
        Directory containing EM model files
        
    Returns
    -------
    dict
        Dictionary containing EM model data:
        - Ex, Ey, Ez: Electric field components
        - tissue_types: Tissue classification
        - sigma_by_rhox: Conductivity/density ratio
        - mass_cell: Mass per voxel
    """
    
    if dirname is None:
        # Try to find data in the package data directory
        data_dir = Path(__file__).parent.parent / 'data'
        if not data_dir.exists():
            raise FileNotFoundError("EM model directory not specified and default data directory not found")
        dirname = str(data_dir)
    
    model = {}
    
    # Load electric field components
    try:
        Ex_file = os.path.join(dirname, 'Ex.mat')
        Ey_file = os.path.join(dirname, 'Ey.mat')
        Ez_file = os.path.join(dirname, 'Ez.mat')
        
        if os.path.exists(Ex_file):
            Ex_data = sio.loadmat(Ex_file)
            model['Ex'] = Ex_data['Ex']
        else:
            raise FileNotFoundError(f"Ex.mat not found in {dirname}")
            
        if os.path.exists(Ey_file):
            Ey_data = sio.loadmat(Ey_file)
            model['Ey'] = Ey_data['Ey']
        else:
            raise FileNotFoundError(f"Ey.mat not found in {dirname}")
            
        if os.path.exists(Ez_file):
            Ez_data = sio.loadmat(Ez_file)
            model['Ez'] = Ez_data['Ez']
        else:
            raise FileNotFoundError(f"Ez.mat not found in {dirname}")
            
    except Exception as e:
        print(f"Error loading electric field files: {e}")
        raise
    
    # Load tissue types
    try:
        tissue_file = os.path.join(dirname, 'Tissue_types.mat')
        if os.path.exists(tissue_file):
            tissue_data = sio.loadmat(tissue_file)
            model['tissue_types'] = tissue_data['Tissue_types']
        else:
            raise FileNotFoundError(f"Tissue_types.mat not found in {dirname}")
    except Exception as e:
        print(f"Error loading tissue types: {e}")
        raise
    
    # Load conductivity/density data
    try:
        sigma_file = os.path.join(dirname, 'SigmabyRhox.mat')
        if os.path.exists(sigma_file):
            sigma_data = sio.loadmat(sigma_file)
            model['sigma_by_rhox'] = sigma_data['SigmabyRhox']
        else:
            raise FileNotFoundError(f"SigmabyRhox.mat not found in {dirname}")
    except Exception as e:
        print(f"Error loading conductivity data: {e}")
        raise
    
    # Load mass data
    try:
        mass_file = os.path.join(dirname, 'Mass_cell.mat')
        if os.path.exists(mass_file):
            mass_data = sio.loadmat(mass_file)
            model['mass_cell'] = mass_data['Mass_cell']
        else:
            raise FileNotFoundError(f"Mass_cell.mat not found in {dirname}")
    except Exception as e:
        print(f"Error loading mass data: {e}")
        raise
    
    # Validate model data
    validate_em_model(model)
    
    return model


def validate_em_model(model):
    """
    Validate electromagnetic model data consistency
    
    Parameters
    ----------
    model : dict
        EM model data dictionary
        
    Raises
    ------
    ValueError
        If model data is inconsistent
    """
    
    required_keys = ['Ex', 'Ey', 'Ez', 'Tissue_types', 'SigmabyRhox', 'Mass_cell']
    
    for key in required_keys:
        if key not in model:
            raise ValueError(f"Missing required model component: {key}")
    
    # Check spatial dimensions consistency
    Ex_shape = model['Ex'].shape[:3]
    
    for field in ['Ey', 'Ez']:
        if model[field].shape[:3] != Ex_shape:
            raise ValueError(f"Field {field} spatial dimensions don't match Ex")
    
    for array_name in ['Tissue_types', 'SigmabyRhox', 'Mass_cell']:
        if model[array_name].shape != Ex_shape:
            raise ValueError(f"{array_name} dimensions don't match field dimensions")
    
    # Check coil dimensions consistency
    if model['Ex'].shape[3:] != model['Ey'].shape[3:] or model['Ex'].shape[3:] != model['Ez'].shape[3:]:
        raise ValueError("Number of coils inconsistent between field components")
    
    print(f"EM model validation passed. Grid size: {Ex_shape}, Coils: {model['Ex'].shape[3]}")


def create_dummy_em_model(grid_shape=(64, 64, 32), num_coils=8):
    """
    Create dummy EM model for testing
    
    Parameters
    ----------
    grid_shape : tuple
        Shape of the spatial grid (Nx, Ny, Nz)
    num_coils : int
        Number of RF coils
        
    Returns
    -------
    dict
        Dummy EM model
    """
    
    print(f"Creating dummy EM model: {grid_shape} grid, {num_coils} coils")
    
    Nx, Ny, Nz = grid_shape
    
    # Create coordinate grids
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create dummy electric fields (dipole-like patterns)
    Ex = np.zeros((Nx, Ny, Nz, num_coils), dtype=complex)
    Ey = np.zeros((Nx, Ny, Nz, num_coils), dtype=complex)
    Ez = np.zeros((Nx, Ny, Nz, num_coils), dtype=complex)
    
    for coil in range(num_coils):
        # Create different field patterns for each coil
        phase = 2 * np.pi * coil / num_coils
        
        # Simple dipole-like patterns
        Ex[:, :, :, coil] = (X + 1j * Y) * np.exp(1j * phase) * np.exp(-(X**2 + Y**2 + Z**2))
        Ey[:, :, :, coil] = (Y + 1j * Z) * np.exp(1j * phase) * np.exp(-(X**2 + Y**2 + Z**2))
        Ez[:, :, :, coil] = (Z + 1j * X) * np.exp(1j * phase) * np.exp(-(X**2 + Y**2 + Z**2))
    
    # Create tissue types (simple ellipsoid body)
    tissue_types = np.zeros((Nx, Ny, Nz))
    body_mask = (X**2 + Y**2 + 2*Z**2) < 0.8
    head_mask = (X**2 + Y**2 + (Z-0.3)**2) < 0.3
    
    tissue_types[body_mask] = 2  # Torso
    tissue_types[head_mask] = 1  # Head
    
    # Create conductivity/density (tissue-dependent)
    sigma_by_rhox = np.ones((Nx, Ny, Nz)) * 1e-8  # Air default
    sigma_by_rhox[tissue_types == 1] = 0.6 / 1000  # Head tissue
    sigma_by_rhox[tissue_types == 2] = 0.8 / 1000  # Body tissue
    
    # Create mass data
    mass_cell = np.ones((Nx, Ny, Nz)) * 1.625e-7  # Air default
    mass_cell[tissue_types == 1] = 1.05e-3  # Head tissue density
    mass_cell[tissue_types == 2] = 1.03e-3  # Body tissue density
    
    model = {
        'Ex': Ex,
        'Ey': Ey,
        'Ez': Ez,
        'Tissue_types': tissue_types,
        'SigmabyRhox': sigma_by_rhox,
        'Mass_cell': mass_cell
    }
    
    return model


def save_em_model(model, dirname):
    """
    Save EM model to MATLAB files
    
    Parameters
    ----------
    model : dict
        EM model data
    dirname : str
        Directory to save files
    """
    
    os.makedirs(dirname, exist_ok=True)
    
    # Save each component
    sio.savemat(os.path.join(dirname, 'Ex.mat'), {'Ex': model['Ex']})
    sio.savemat(os.path.join(dirname, 'Ey.mat'), {'Ey': model['Ey']})
    sio.savemat(os.path.join(dirname, 'Ez.mat'), {'Ez': model['Ez']})
    sio.savemat(os.path.join(dirname, 'Tissue_types.mat'), {'Tissue_types': model['tissue_types']})
    sio.savemat(os.path.join(dirname, 'SigmabyRhox.mat'), {'SigmabyRhox': model['sigma_by_rhox']})
    sio.savemat(os.path.join(dirname, 'Mass_cell.mat'), {'Mass_cell': model['mass_cell']})
    
    print(f"EM model saved to {dirname}")
