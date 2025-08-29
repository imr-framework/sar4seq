"""
SAR Calculation Utilities

Functions for calculating Specific Absorption Rate (SAR) from Q-matrices and RF signals.
"""

import numpy as np


def calc_SAR(Q, I, weight):
    """
    Calculate SAR from Q-matrix and RF signal
    
    Parameters
    ----------
    Q : numpy.ndarray
        Q-matrix for SAR calculation
    I : numpy.ndarray
        RF signal with Nc rows and Nt columns
    weight : float
        Body weight in kg
        
    Returns
    -------
    float
        SAR value in W/kg
    """
    
    # Calculate signal power (assuming single channel for now)
    I_exp = np.conj(I) * I  # Element-wise multiplication
    I_exp = np.sum(I_exp) / len(I_exp.flatten())
    I_fact = I_exp
    
    # Handle multi-dimensional Q matrices (local SAR)
    if Q.ndim > 2:
        SAR_temp = np.zeros_like(Q, dtype=complex)
        SAR_norm = np.zeros(Q.shape[0])
        
        for k in range(Q.shape[0]):
            Q_temp = Q[k, :, :]
            SAR_temp[k, :, :] = Q_temp * I_fact
            SAR_norm[k] = np.linalg.norm(SAR_temp[k, :, :])
        
        # Find maximum SAR location
        ind = np.argmax(SAR_norm)
        SAR_chosen = SAR_temp[ind, :, :]
        SAR = np.abs(np.sum(SAR_chosen))
    else:
        # Global SAR calculation
        SAR_temp = Q * I_fact
        SAR = np.abs(np.sum(SAR_temp))
        SAR = SAR / weight
    
    return SAR


def calc_SAR_multichannel(Q, I, weight, tx_phases=None):
    """
    Calculate SAR for multi-channel transmission
    
    Parameters
    ----------
    Q : numpy.ndarray
        Q-matrix for SAR calculation
    I : numpy.ndarray
        RF signal array with shape (Nc, Nt) where Nc is number of channels
    weight : float
        Body weight in kg
    tx_phases : numpy.ndarray, optional
        Transmission phases for each channel
        
    Returns
    -------
    float
        SAR value in W/kg
    """
    
    if I.ndim == 1:
        # Single channel case
        return calc_SAR(Q, I, weight)
    
    Nc = I.shape[0]  # Number of channels
    
    # Calculate cross-channel interference matrix
    I_fact = np.zeros((Nc, Nc), dtype=complex)
    
    for nc1 in range(Nc):
        for nc2 in range(Nc):
            I_fact[nc1, nc2] = np.mean(np.conj(I[nc1, :]) * I[nc2, :])
    
    # Apply transmission phases if provided
    if tx_phases is not None:
        if len(tx_phases) != Nc:
            raise ValueError("Number of phases must match number of channels")
        
        phase_matrix = np.exp(1j * np.array(tx_phases))
        phase_outer = np.outer(np.conj(phase_matrix), phase_matrix)
        I_fact = I_fact * phase_outer
    
    # Calculate SAR
    if Q.ndim > 2:
        # Local SAR with multiple observation points
        SAR_temp = np.zeros(Q.shape[0])
        
        for k in range(Q.shape[0]):
            Q_temp = Q[k, :, :]
            SAR_temp[k] = np.real(np.trace(Q_temp @ I_fact))
        
        SAR = np.max(SAR_temp)
    else:
        # Global SAR
        SAR = np.real(np.trace(Q @ I_fact))
        SAR = SAR / weight
    
    return SAR


def calc_local_SAR_map(Q_local, I, voxel_masses):
    """
    Calculate local SAR map from local Q-matrices
    
    Parameters
    ----------
    Q_local : numpy.ndarray
        Local Q-matrices with shape (Nx, Ny, Nz, Nc, Nc)
    I : numpy.ndarray
        RF signal array
    voxel_masses : numpy.ndarray
        Mass of each voxel with shape (Nx, Ny, Nz)
        
    Returns
    -------
    numpy.ndarray
        Local SAR map with shape (Nx, Ny, Nz)
    """
    
    if I.ndim == 1:
        I_fact = np.abs(I)**2 / len(I)
    else:
        # Multi-channel case
        I_fact = np.zeros((I.shape[0], I.shape[0]), dtype=complex)
        for nc1 in range(I.shape[0]):
            for nc2 in range(I.shape[0]):
                I_fact[nc1, nc2] = np.mean(np.conj(I[nc1, :]) * I[nc2, :])
    
    # Calculate SAR for each voxel
    SAR_map = np.zeros(Q_local.shape[:3])
    
    for i in range(Q_local.shape[0]):
        for j in range(Q_local.shape[1]):
            for k in range(Q_local.shape[2]):
                if voxel_masses[i, j, k] > 0:
                    Q_voxel = Q_local[i, j, k, :, :]
                    if I.ndim == 1:
                        SAR_map[i, j, k] = np.real(np.trace(Q_voxel)) * I_fact / voxel_masses[i, j, k]
                    else:
                        SAR_map[i, j, k] = np.real(np.trace(Q_voxel @ I_fact)) / voxel_masses[i, j, k]
    
    return SAR_map
