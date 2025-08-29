"""
Air Cube Checker

Functions for checking if a cube's faces are in air for global SAR calculations.
"""

import numpy as np


def chkcubair_global(dim, mass_cell, mass_air, x, y, z):
    """
    Check if cube's faces are in air
    
    Parameters
    ----------
    dim : int
        Dimension of the cube check
    mass_cell : numpy.ndarray
        Mass per voxel array
    mass_air : float
        Mass threshold for air
    x, y, z : int
        Coordinates of the center voxel
        
    Returns
    -------
    int
        1 if valid (not in air), 0 if invalid (in air)
    """
    
    # Get array dimensions
    M, N, P = mass_cell.shape
    
    # Check bounds to avoid indexing errors
    if (x - dim < 0 or x + dim >= M or
        y - dim < 0 or y + dim >= N or
        z - dim < 0 or z + dim >= P):
        return 0  # Invalid if out of bounds
    
    try:
        # Extract faces of the cube
        xface0 = mass_cell[x - dim, y - dim:y + dim + 1, z - dim:z + dim + 1]
        yface0 = mass_cell[x - dim:x + dim + 1, y - dim, z - dim:z + dim + 1]
        zface0 = mass_cell[x - dim:x + dim + 1, y - dim:y + dim + 1, z - dim]
        
        xface1 = mass_cell[x + dim, y - dim:y + dim + 1, z - dim:z + dim + 1]
        yface1 = mass_cell[x - dim:x + dim + 1, y + dim, z - dim:z + dim + 1]
        zface1 = mass_cell[x - dim:x + dim + 1, y - dim:y + dim + 1, z + dim]
        
        chk = 1
        
        # Check if any face is entirely in air
        if (np.max(xface0) <= mass_air or np.max(yface0) <= mass_air or 
            np.max(zface0) <= mass_air or np.max(xface1) <= mass_air or 
            np.max(yface1) <= mass_air or np.max(zface1) <= mass_air):
            chk = 0
            
    except IndexError:
        # If indexing fails, return invalid
        chk = 0
    
    return chk


def chkcubair_local(mass_cell, mass_air, coordinates, cube_size=3):
    """
    Check air boundaries for multiple coordinates (local SAR)
    
    Parameters
    ----------
    mass_cell : numpy.ndarray
        Mass per voxel array
    mass_air : float
        Mass threshold for air
    coordinates : numpy.ndarray
        Array of coordinates with shape (N, 3)
    cube_size : int
        Size of the cube to check
        
    Returns
    -------
    numpy.ndarray
        Array of validity flags (1 for valid, 0 for invalid)
    """
    
    N = coordinates.shape[0]
    validity = np.zeros(N, dtype=int)
    
    for i, (x, y, z) in enumerate(coordinates):
        validity[i] = chkcubair_global(cube_size // 2, mass_cell, mass_air, x, y, z)
    
    return validity


def check_boundary_conditions(mass_cell, x, y, z, margin=2):
    """
    Check if coordinates are far enough from boundaries
    
    Parameters
    ----------
    mass_cell : numpy.ndarray
        Mass array
    x, y, z : int
        Coordinates to check
    margin : int
        Required margin from boundaries
        
    Returns
    -------
    bool
        True if coordinates are valid with margin
    """
    
    M, N, P = mass_cell.shape
    
    return (margin <= x < M - margin and 
            margin <= y < N - margin and 
            margin <= z < P - margin)


def find_air_tissue_boundary(mass_cell, mass_air_threshold=1.625e-7):
    """
    Find the boundary between air and tissue
    
    Parameters
    ----------
    mass_cell : numpy.ndarray
        Mass per voxel array
    mass_air_threshold : float
        Threshold for distinguishing air from tissue
        
    Returns
    -------
    tuple
        tissue_mask : numpy.ndarray
            Boolean mask of tissue voxels
        boundary_mask : numpy.ndarray
            Boolean mask of boundary voxels
    """
    
    tissue_mask = mass_cell > mass_air_threshold
    
    # Find boundary using morphological operations
    from scipy import ndimage
    
    # Erode tissue mask to find internal tissue
    eroded = ndimage.binary_erosion(tissue_mask)
    
    # Boundary is tissue that becomes air after erosion
    boundary_mask = tissue_mask & ~eroded
    
    return tissue_mask, boundary_mask
