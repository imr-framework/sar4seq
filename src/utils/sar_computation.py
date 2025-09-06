#!/usr/bin/env python3
"""
Core SAR Computation Functions

Contains the fundamental SAR calculation algorithms for both full-resolution
and VOP-compressed Q-matrices, including GPU-optimized implementations.

Authors: Leo Kinyera, BS
Copyright: Board of Trustees of Columbia University in the City of New York
"""

import numpy as np
from typing import Dict, Any, Union, List, Tuple, Optional
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

def calculate_sar_uncompressed(seq_block_type: np.ndarray, 
                              seq_block_events: np.ndarray,
                              seq_block_durations: np.ndarray,
                              Q_uncompressed: np.ndarray, 
                              dt: float = 1e-6,
                              **kwargs) -> Dict[str, Any]:
    """
    Calculate SAR using full-resolution uncompressed Q-matrix.
    
    Args:
        seq_block_type: Sequence block types (RF=1, others=0)
        seq_block_events: RF pulse parameters
        seq_block_durations: Duration of each block
        Q_uncompressed: Full spatial resolution Q-matrix
        dt: Time step for SAR calculation
        
    Returns:
        Dictionary containing SAR values, timing, and analysis
    """
    start_time = time.time()
    
    # Extract parameters
    use_gpu = kwargs.get('use_gpu', False)
    clinical_mode = kwargs.get('clinical_mode', False)
    
    # Validate inputs
    if Q_uncompressed.ndim != 3:
        raise ValueError("Q_uncompressed must be 3D array (channels, spatial_points, spatial_points)")
    
    n_channels, n_spatial, _ = Q_uncompressed.shape
    
    if use_gpu and HAS_CUPY:
        Q_gpu = cp.asarray(Q_uncompressed)
        seq_gpu = cp.asarray(seq_block_events)
        sar_calc = _calculate_sar_gpu_uncompressed
    else:
        Q_gpu = Q_uncompressed
        seq_gpu = seq_block_events
        sar_calc = _calculate_sar_cpu_uncompressed
    
    # Calculate SAR for each time point
    sar_values = sar_calc(seq_gpu, Q_gpu, seq_block_type, seq_block_durations, dt)
    
    # Convert back to CPU if needed
    if use_gpu and HAS_CUPY:
        sar_values = cp.asnumpy(sar_values)
    
    computation_time = time.time() - start_time
    
    return {
        'sar_spatial': sar_values,
        'sar_max': np.max(sar_values, axis=1),  # Max over spatial points
        'sar_mean': np.mean(sar_values, axis=1),
        'computation_time': computation_time,
        'method': 'uncompressed',
        'gpu_used': use_gpu and HAS_CUPY,
        'spatial_resolution': n_spatial
    }

def _calculate_sar_cpu_uncompressed(seq_events: np.ndarray,
                                   Q_matrix: np.ndarray,
                                   block_types: np.ndarray,
                                   durations: np.ndarray,
                                   dt: float) -> np.ndarray:
    """CPU implementation of uncompressed SAR calculation."""
    n_channels, n_spatial, _ = Q_matrix.shape
    n_time_points = len(block_types)
    
    sar_spatial = np.zeros((n_time_points, n_spatial))
    
    for t in range(n_time_points):
        if block_types[t] == 1:  # RF block
            # Extract complex RF amplitudes for all channels
            rf_complex = seq_events[t, :n_channels] + 1j * seq_events[t, n_channels:2*n_channels]
            
            # Calculate SAR = Re(RF† * Q * RF) for each spatial point
            for spatial_idx in range(n_spatial):
                Q_spatial = Q_matrix[:, spatial_idx, spatial_idx]
                sar_spatial[t, spatial_idx] = np.real(
                    np.conj(rf_complex) @ Q_spatial @ rf_complex
                )
    
    return sar_spatial

def _calculate_sar_gpu_uncompressed(seq_events: Any,
                                   Q_matrix: Any,
                                   block_types: np.ndarray,
                                   durations: np.ndarray,
                                   dt: float) -> Any:
    """GPU implementation of uncompressed SAR calculation."""
    n_channels, n_spatial, _ = Q_matrix.shape
    n_time_points = len(block_types)
    
    sar_spatial = cp.zeros((n_time_points, n_spatial))
    
    # Find RF blocks
    rf_indices = np.where(block_types == 1)[0]
    
    if len(rf_indices) > 0:
        # Process all RF blocks in batch
        rf_events = seq_events[rf_indices]
        rf_complex = rf_events[:, :n_channels] + 1j * rf_events[:, n_channels:2*n_channels]
        
        # Vectorized SAR calculation across all spatial points
        for i, t in enumerate(rf_indices):
            Q_diag = cp.diagonal(Q_matrix, axis1=1, axis2=2)  # Extract diagonal elements
            sar_spatial[t] = cp.real(
                cp.sum(cp.conj(rf_complex[i:i+1]) * Q_diag * rf_complex[i:i+1], axis=1)
            )
    
    return sar_spatial

def calculate_sar_vop_compressed(seq_block_type: np.ndarray,
                                seq_block_events: np.ndarray,
                                seq_block_durations: np.ndarray,
                                Q_vop: np.ndarray,
                                vop_matrix: np.ndarray,
                                dt: float = 1e-6,
                                **kwargs) -> Dict[str, Any]:
    """
    Calculate SAR using VOP-compressed Q-matrix for computational efficiency.
    
    Args:
        seq_block_type: Sequence block types
        seq_block_events: RF pulse parameters
        seq_block_durations: Duration of each block
        Q_vop: VOP-compressed Q-matrix
        vop_matrix: VOP compression matrix
        dt: Time step
        
    Returns:
        Dictionary containing SAR analysis
    """
    start_time = time.time()
    
    use_gpu = kwargs.get('use_gpu', False)
    
    if use_gpu and HAS_CUPY:
        Q_gpu = cp.asarray(Q_vop)
        seq_gpu = cp.asarray(seq_block_events)
        vop_gpu = cp.asarray(vop_matrix)
        sar_calc = _calculate_sar_gpu_vop
    else:
        Q_gpu = Q_vop
        seq_gpu = seq_block_events
        vop_gpu = vop_matrix
        sar_calc = _calculate_sar_cpu_vop
    
    # Calculate VOP-space SAR
    sar_vop = sar_calc(seq_gpu, Q_gpu, seq_block_type, seq_block_durations, dt)
    
    # Reconstruct spatial SAR from VOP coefficients
    sar_spatial = _reconstruct_spatial_sar(sar_vop, vop_gpu, use_gpu)
    
    # Convert to CPU if needed
    if use_gpu and HAS_CUPY:
        sar_spatial = cp.asnumpy(sar_spatial)
        sar_vop = cp.asnumpy(sar_vop)
    
    computation_time = time.time() - start_time
    
    return {
        'sar_spatial': sar_spatial,
        'sar_vop': sar_vop,
        'sar_max': np.max(sar_spatial, axis=1),
        'sar_mean': np.mean(sar_spatial, axis=1),
        'computation_time': computation_time,
        'method': 'vop_compressed',
        'gpu_used': use_gpu and HAS_CUPY,
        'vop_count': Q_vop.shape[1]
    }

def _calculate_sar_cpu_vop(seq_events: np.ndarray,
                          Q_vop: np.ndarray,
                          block_types: np.ndarray,
                          durations: np.ndarray,
                          dt: float) -> np.ndarray:
    """CPU implementation of VOP-compressed SAR calculation."""
    n_channels, n_vop = Q_vop.shape
    n_time_points = len(block_types)
    
    sar_vop = np.zeros((n_time_points, n_vop))
    
    for t in range(n_time_points):
        if block_types[t] == 1:  # RF block
            rf_complex = seq_events[t, :n_channels] + 1j * seq_events[t, n_channels:2*n_channels]
            
            # VOP-space SAR calculation
            sar_vop[t] = np.real(np.conj(rf_complex) @ Q_vop)
    
    return sar_vop

def _calculate_sar_gpu_vop(seq_events: Any,
                          Q_vop: Any,
                          block_types: np.ndarray,
                          durations: np.ndarray,
                          dt: float) -> Any:
    """GPU implementation of VOP-compressed SAR calculation."""
    n_channels, n_vop = Q_vop.shape
    n_time_points = len(block_types)
    
    sar_vop = cp.zeros((n_time_points, n_vop))
    
    # Find RF blocks and process in batch
    rf_indices = np.where(block_types == 1)[0]
    
    if len(rf_indices) > 0:
        rf_events = seq_events[rf_indices]
        rf_complex = rf_events[:, :n_channels] + 1j * rf_events[:, n_channels:2*n_channels]
        
        # Batch matrix multiplication
        sar_batch = cp.real(cp.conj(rf_complex) @ Q_vop)
        
        for i, t in enumerate(rf_indices):
            sar_vop[t] = sar_batch[i]
    
    return sar_vop

def _reconstruct_spatial_sar(sar_vop: Union[np.ndarray, Any],
                            vop_matrix: Union[np.ndarray, Any],
                            use_gpu: bool) -> Union[np.ndarray, Any]:
    """Reconstruct spatial SAR from VOP coefficients."""
    if use_gpu and HAS_CUPY:
        # GPU matrix multiplication
        return sar_vop @ vop_matrix.T
    else:
        # CPU matrix multiplication
        return sar_vop @ vop_matrix.T

def calculate_time_averaged_sar(sar_spatial: np.ndarray,
                               durations: np.ndarray,
                               averaging_time: float = 6.0 * 60,  # 6 minutes
                               dt: float = 1e-6) -> Dict[str, np.ndarray]:
    """
    Calculate time-averaged SAR over specified window.
    
    Args:
        sar_spatial: Instantaneous SAR values [time, spatial]
        durations: Duration of each time block
        averaging_time: Averaging window in seconds
        dt: Time step
        
    Returns:
        Dictionary with averaged SAR values
    """
    n_time, n_spatial = sar_spatial.shape
    
    # Create time vector
    cumulative_time = np.cumsum(durations)
    total_time = cumulative_time[-1]
    
    # Number of averaging windows
    n_windows = int(np.ceil(total_time / averaging_time))
    
    sar_averaged = np.zeros((n_windows, n_spatial))
    window_times = np.zeros(n_windows)
    
    for window in range(n_windows):
        t_start = window * averaging_time
        t_end = min((window + 1) * averaging_time, total_time)
        window_times[window] = (t_start + t_end) / 2
        
        # Find time indices in this window
        in_window = (cumulative_time >= t_start) & (cumulative_time <= t_end)
        
        if np.any(in_window):
            # Weight by duration for proper averaging
            weights = durations[in_window]
            sar_window = sar_spatial[in_window]
            
            sar_averaged[window] = np.average(sar_window, axis=0, weights=weights)
    
    return {
        'sar_averaged': sar_averaged,
        'window_times': window_times,
        'averaging_time': averaging_time,
        'sar_max_averaged': np.max(sar_averaged, axis=1),
        'sar_peak_location': np.argmax(np.max(sar_averaged, axis=0))
    }

__all__ = [
    'calculate_sar_uncompressed',
    'calculate_sar_vop_compressed', 
    'calculate_time_averaged_sar'
]
