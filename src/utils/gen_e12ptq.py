"""
12-Point Q-Matrix Generation

Functions for generating Q-matrices using the 12-point cube formulation.
"""

import numpy as np


def gen_E12ptQ(Ex, Ey, Ez, X, sigma_by_rhox):
    """
    Generate Q-matrix using 12-point cube formulation
    
    Parameters
    ----------
    Ex, Ey, Ez : numpy.ndarray
        Electric field components with shape (M, N, P, Nc) where Nc is number of coils
    X : list or numpy.ndarray
        Coordinates [x, y, z] of the voxel
    sigma_by_rhox : numpy.ndarray
        Conductivity divided by density array
        
    Returns
    -------
    numpy.ndarray
        Q-matrix with shape (Nc, Nc)
    """
    
    sigma_by_rhoy = sigma_by_rhox
    sigma_by_rhoz = sigma_by_rhox
    
    # Get coordinates
    if np.isscalar(X):
        M, N, P = Ex.shape[:3]
        x, y, z = np.unravel_index(X, (M, N, P))
        X = [x, y, z]
    else:
        x, y, z = X[0], X[1], X[2]
    
    # Handle boundary conditions
    M, N, P = Ex.shape[:3]
    
    # Define coordinates of the 12-point cube formulation
    # X direction points
    X1 = [x, min(y + 1, N - 1), z]
    X2 = [x, y, min(z + 1, P - 1)]
    X3 = [x, min(y + 1, N - 1), min(z + 1, P - 1)]
    
    # Y direction points
    Y1 = [min(x + 1, M - 1), y, z]
    Y2 = [x, y, min(z + 1, P - 1)]
    Y3 = [min(x + 1, M - 1), y, min(z + 1, P - 1)]
    
    # Z direction points
    Z1 = [min(x + 1, M - 1), y, z]
    Z2 = [x, min(y + 1, N - 1), z]
    Z3 = [min(x + 1, M - 1), min(y + 1, N - 1), z]
    
    # Get electric field contributions
    Ex1 = get_E(Ex[x, y, z, :], sigma_by_rhox[x, y, z])
    Ey1 = get_E(Ey[x, y, z, :], sigma_by_rhoy[x, y, z])
    Ez1 = get_E(Ez[x, y, z, :], sigma_by_rhoz[x, y, z])
    
    # X component power
    Expwr = (Ex1 + 
             get_E(Ex[X1[0], X1[1], X1[2], :], sigma_by_rhox[X1[0], X1[1], X1[2]]) +
             get_E(Ex[X2[0], X2[1], X2[2], :], sigma_by_rhox[X2[0], X2[1], X2[2]]) +
             get_E(Ex[X3[0], X3[1], X3[2], :], sigma_by_rhox[X3[0], X3[1], X3[2]]))
    
    # Y component power
    Eypwr = (Ey1 + 
             get_E(Ey[Y1[0], Y1[1], Y1[2], :], sigma_by_rhoy[Y1[0], Y1[1], Y1[2]]) +
             get_E(Ey[Y2[0], Y2[1], Y2[2], :], sigma_by_rhoy[Y2[0], Y2[1], Y2[2]]) +
             get_E(Ey[Y3[0], Y3[1], Y3[2], :], sigma_by_rhoy[Y3[0], Y3[1], Y3[2]]))
    
    # Z component power
    Ezpwr = (Ez1 + 
             get_E(Ez[Z1[0], Z1[1], Z1[2], :], sigma_by_rhoz[Z1[0], Z1[1], Z1[2]]) +
             get_E(Ez[Z2[0], Z2[1], Z2[2], :], sigma_by_rhoz[Z2[0], Z2[1], Z2[2]]) +
             get_E(Ez[Z3[0], Z3[1], Z3[2], :], sigma_by_rhoz[Z3[0], Z3[1], Z3[2]]))
    
    # Total power (0.125 = 0.25/2, where 0.25 is averaging factor and /2 accounts for density)
    Epwr = 0.125 * (Expwr + Eypwr + Ezpwr)
    
    return Epwr


def get_E(E, sigma_by_rho):
    """
    Calculate electric field contribution to Q-matrix
    
    Parameters
    ----------
    E : numpy.ndarray
        Electric field vector for all coils
    sigma_by_rho : float
        Conductivity divided by density at this location
        
    Returns
    -------
    numpy.ndarray
        Q-matrix contribution
    """
    
    # Ensure E is a column vector
    if E.ndim == 1:
        E = E.reshape(-1, 1)
    
    # Calculate outer product E * E^H (conjugate transpose)
    Epwr = sigma_by_rho * (E @ E.conj().T)
    
    return Epwr


def gen_E12ptQ_optimized(Ex, Ey, Ez, coordinates, sigma_by_rhox):
    """
    Optimized version for multiple coordinate calculations
    
    Parameters
    ----------
    Ex, Ey, Ez : numpy.ndarray
        Electric field components
    coordinates : numpy.ndarray
        Array of coordinates with shape (N, 3)
    sigma_by_rhox : numpy.ndarray
        Conductivity/density array
        
    Returns
    -------
    numpy.ndarray
        Q-matrices with shape (N, Nc, Nc)
    """
    
    N = coordinates.shape[0]
    Nc = Ex.shape[-1]  # Number of coils
    Q_matrices = np.zeros((N, Nc, Nc), dtype=complex)
    
    for i, coord in enumerate(coordinates):
        Q_matrices[i] = gen_E12ptQ(Ex, Ey, Ez, coord, sigma_by_rhox)
    
    return Q_matrices


def validate_field_coordinates(Ex, Ey, Ez, X):
    """
    Validate that coordinates are within field bounds
    
    Parameters
    ----------
    Ex, Ey, Ez : numpy.ndarray
        Electric field components
    X : list
        Coordinates [x, y, z]
        
    Returns
    -------
    bool
        True if coordinates are valid
    """
    
    M, N, P = Ex.shape[:3]
    x, y, z = X
    
    if not (0 <= x < M and 0 <= y < N and 0 <= z < P):
        raise IndexError(f"Coordinates ({x}, {y}, {z}) out of bounds for field shape ({M}, {N}, {P})")
    
    return True
