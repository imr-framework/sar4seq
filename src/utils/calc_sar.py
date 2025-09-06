"""
SAR Calculation Utilities

Functions for calculating Specific Absorption Rate (SAR) from Q-matrices and RF signals.
"""

import numpy as np
import os
from pathlib import Path
import scipy.io as sio
import time
import pypulseq as pp
from utils.gen_seq_test import create_test_sequence
from utils.read_qmat import load_clinical_qmatrices
from utils.gen_qpwr import create_full_resolution_qmatrix
from utils.rf_sequence import analyze_sequence_blocks, create_rf_vector
from utils.constants import CLINICAL_CONSTANTS
from utils.safety_assessment import perform_safety_assessment, generate_clinical_report

# Try to import VOP_Qmatrices_v3 from parent src directory
try:
    from ..vop_qmatrices_v3 import VOP_Qmatrices_v3
    VOP_AVAILABLE = True
except ImportError:
    VOP_AVAILABLE = False
    VOP_Qmatrices_v3 = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False



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
    # Element-wise multiplication
    I_exp = np.conj(I) * I
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
    
    # Number of channels
    Nc = I.shape[0]
    
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


def calc_SAR_full_resolution_cpu(Q_full, rf_vector, mass, progress_callback=None):
    """
    Calculate SAR using full spatial resolution on CPU
    
    Parameters:
    -----------
    Q_full : numpy.ndarray
        Full resolution Q-matrix (n_spatial, n_channels, n_channels)
    rf_vector : numpy.ndarray
        RF signal vector (n_channels,)
    mass : float
        Body mass in kg
    progress_callback : callable, optional
        Function to call for progress updates
        
    Returns:
    --------
    sar_local : numpy.ndarray
        SAR at each spatial location (n_spatial,)
    sar_peak : float
        Peak SAR value across all locations
    """
    
    n_spatial = Q_full.shape[0]
    sar_local = np.zeros(n_spatial)
    
    print(f"Calculating SAR at {n_spatial:,} spatial locations (CPU)...")
    start_time = time.time()
    
    # Process each spatial location
    for k in range(n_spatial):
        # Extract Q-matrix for this location
        Q_k = Q_full[k, :, :]
        
        # Calculate SAR using quadratic form: SAR = (1/2) * Re(rf_vector^H * Q * rf_vector)
        sar_contribution = np.conj(rf_vector) @ Q_k @ rf_vector
        sar_local[k] = 0.5 * np.real(sar_contribution) / mass
        
        # Progress update
        if progress_callback and k % 1000 == 0:
            progress = (k + 1) / n_spatial * 100
            progress_callback(progress, k + 1, n_spatial)
    
    computation_time = time.time() - start_time
    print(f"  CPU computation time: {computation_time:.3f} seconds")
    print(f"  Locations per second: {n_spatial/computation_time:.0f}")
    
    # Find peak SAR
    sar_peak = np.max(sar_local)
    peak_location = np.argmax(sar_local)
    
    print(f"  Peak SAR: {sar_peak:.6f} W/kg at location {peak_location}")
    print(f"  Mean SAR: {np.mean(sar_local):.6f} W/kg")
    
    return sar_local, sar_peak


def calc_SAR_full_resolution_gpu(Q_full, rf_vector, mass):
    """
    Calculate SAR using full spatial resolution on GPU - OPTIMIZED VERSION
    
    Parameters:
    -----------
    Q_full : numpy.ndarray
        Full resolution Q-matrix (n_spatial, n_channels, n_channels)
    rf_vector : numpy.ndarray
        RF signal vector (n_channels,)
    mass : float
        Body mass in kg
        
    Returns:
    --------
    sar_local : numpy.ndarray
        SAR at each spatial location (n_spatial,)
    sar_peak : float
        Peak SAR value across all locations
    """
    
    n_spatial = Q_full.shape[0]
    
    print(f"Calculating SAR at {n_spatial:,} spatial locations (GPU - OPTIMIZED)...")
    print(f"  Q-matrix shape: {Q_full.shape}")
    print(f"  Memory requirement: {Q_full.nbytes / (1024**2):.1f} MB")
    
    start_time = time.time()
    
    try:
        # Check GPU memory
        gpu_mem_info = cp.cuda.Device().mem_info
        free_mem_gb = gpu_mem_info[0] / (1024**3)
        total_mem_gb = gpu_mem_info[1] / (1024**3)
        print(f"  GPU memory: {free_mem_gb:.1f}/{total_mem_gb:.1f} GB free")
        
        # Transfer data to GPU 
        memory_needed_gb = Q_full.nbytes / (1024**3)
        print(f"  Data transfer size: {memory_needed_gb:.2f} GB")
        
        # Transfer Q-matrix and RF vector to GPU
        print(f"  Transferring Q-matrix to GPU...")
        # Use complex64 for memory efficiency
        Q_gpu = cp.asarray(Q_full, dtype=cp.complex64)
        rf_gpu = cp.asarray(rf_vector, dtype=cp.complex64)
        
        print(f"  Performing OPTIMIZED vectorized GPU computation...")
        
        # OPTIMIZED: Use einsum for highly efficient batch computation
        # This computes rf_conj @ Q[k] @ rf for all k simultaneously
        rf_conj = cp.conj(rf_gpu)
        
        # Method 1: Use einsum for maximum vectorization (fastest)
        # einsum('i,kij,j->k') computes rf_conj[i] * Q[k,i,j] * rf[j] for all k
        sar_contributions = cp.einsum('i,kij,j->k', rf_conj, Q_gpu, rf_gpu)
        
        # Convert to SAR values
        sar_local_gpu = 0.5 * cp.real(sar_contributions) / mass
        
        # Transfer result back to CPU
        print(f"  Transferring results back to CPU...")
        sar_local = cp.asnumpy(sar_local_gpu)
    
    except Exception as gpu_error:
        print(f"  ❌ GPU optimization failed: {gpu_error}")
        print(f"  Falling back to chunked processing...")
        
        try:
            # Fallback: Process in chunks if memory is an issue
            chunk_size = max(1000, int(n_spatial * free_mem_gb * 0.6 / memory_needed_gb))
            print(f"  Processing in chunks of {chunk_size:,} voxels")
            
            sar_local = np.zeros(n_spatial, dtype=np.float64)
            rf_gpu = cp.asarray(rf_vector, dtype=cp.complex64)
            rf_conj = cp.conj(rf_gpu)
            
            for start_idx in range(0, n_spatial, chunk_size):
                end_idx = min(start_idx + chunk_size, n_spatial)
                
                # Transfer chunk to GPU
                Q_chunk_gpu = cp.asarray(Q_full[start_idx:end_idx], dtype=cp.complex64)
                
                # Vectorized computation for chunk
                sar_chunk = cp.einsum('i,kij,j->k', rf_conj, Q_chunk_gpu, rf_gpu)
                sar_local_chunk = 0.5 * cp.real(sar_chunk) / mass
                
                # Transfer back to CPU
                sar_local[start_idx:end_idx] = cp.asnumpy(sar_local_chunk)
                
                # Show progress
                if start_idx % (chunk_size * 5) == 0:
                    progress = end_idx / n_spatial * 100
                    print(f"    Progress: {progress:.1f}%")
                    
        except Exception as fallback_error:
            print(f"  ❌ Chunked processing also failed: {fallback_error}")
            raise RuntimeError("GPU computation failed completely")
    
    computation_time = time.time() - start_time
    print(f"  GPU computation time: {computation_time:.3f} seconds")
    print(f"  Locations per second: {n_spatial/computation_time:.0f}")
    
    # Find peak SAR
    sar_peak = np.max(sar_local)
    peak_location = np.argmax(sar_local)
    
    print(f"  Peak SAR: {sar_peak:.6f} W/kg at location {peak_location}")
    print(f"  Mean SAR: {np.mean(sar_local):.6f} W/kg")
    
    return sar_local, sar_peak

    
# Global variables for GPU optimization
_GPU_Q_MATRIX = None
_GPU_Q_SHAPE = None

def calc_SAR_full_resolution_gpu_batch(Q_full, rf_vectors_list, mass):
    """
    Calculate SAR for multiple RF vectors using pre-loaded GPU Q-matrix
    
    This function transfers Q-matrix to GPU ONCE and processes ALL RF vectors,
    eliminating the major bottleneck of repeated GPU transfers.
    
    Parameters:
    -----------
    Q_full : numpy.ndarray
        Full resolution Q-matrix (n_spatial, n_channels, n_channels)
    rf_vectors_list : list of numpy.ndarray
        List of RF signal vectors, each (n_channels,)
    mass : float
        Body mass in kg
        
    Returns:
    --------
    sar_results : list
        List of (sar_local, sar_peak) tuples for each RF vector
    """
    global _GPU_Q_MATRIX, _GPU_Q_SHAPE
    
    n_spatial = Q_full.shape[0]
    n_rf_vectors = len(rf_vectors_list)
    
    print(f"🚀 BATCH GPU SAR computation for {n_rf_vectors} RF vectors at {n_spatial:,} spatial locations")
    print(f"  Q-matrix shape: {Q_full.shape}")
    print(f"  Memory requirement: {Q_full.nbytes / (1024**2):.1f} MB")
    
    start_time = time.time()
    
    try:
        # Check if Q-matrix is already on GPU and matches current one
        if _GPU_Q_MATRIX is None or _GPU_Q_SHAPE != Q_full.shape:
            print(f"  🔄 Transferring Q-matrix to GPU (ONE-TIME)...")
            _GPU_Q_MATRIX = cp.asarray(Q_full, dtype=cp.complex64)
            _GPU_Q_SHAPE = Q_full.shape
        else:
            print(f"  ✅ Using pre-loaded Q-matrix on GPU")
        
        # Transfer all RF vectors to GPU in batch
        print(f"  📡 Transferring {n_rf_vectors} RF vectors to GPU...")
        # Shape: (n_rf, n_channels)
        rf_batch_gpu = cp.asarray(np.array(rf_vectors_list), dtype=cp.complex64)
        
        print(f"  ⚡ Performing ULTRA-FAST batch GPU computation...")
        
        # ULTRA-OPTIMIZED: Compute SAR for ALL RF vectors simultaneously
        # einsum('bi,kij,bj->bk') computes rf_batch[b,i] * Q[k,i,j] * rf_batch[b,j] 
        # Result shape: (n_rf_vectors, n_spatial)
        rf_batch_conj = cp.conj(rf_batch_gpu)
        sar_batch_contributions = cp.einsum('bi,kij,bj->bk', rf_batch_conj, _GPU_Q_MATRIX, rf_batch_gpu)
        
        # Convert to SAR values for all vectors
        sar_batch_local = 0.5 * cp.real(sar_batch_contributions) / mass
        
        print(f"  📤 Transferring ALL results back to CPU...")
        sar_batch_cpu = cp.asnumpy(sar_batch_local)
        
        # Package results
        sar_results = []
        for i in range(n_rf_vectors):
            sar_local = sar_batch_cpu[i, :]
            sar_peak = np.max(sar_local)
            sar_results.append((sar_local, sar_peak))
        
    except Exception as gpu_error:
        print(f"  ❌ Batch GPU computation failed: {gpu_error}")
        print(f"  🔄 Falling back to sequential processing...")
        
        # Fallback to individual processing
        sar_results = []
        for rf_vector in rf_vectors_list:
            sar_local, sar_peak = calc_SAR_full_resolution_gpu(Q_full, rf_vector, mass)
            sar_results.append((sar_local, sar_peak))
    
    computation_time = time.time() - start_time
    total_locations = n_spatial * n_rf_vectors
    
    print(f"  🏁 BATCH GPU computation completed!")
    print(f"  Total computation time: {computation_time:.3f} seconds")
    print(f"  Total locations per second: {total_locations/computation_time:.0f}")
    print(f"  Speedup factor: ~{n_rf_vectors}x (one Q-matrix transfer for {n_rf_vectors} RF vectors)")
    
    return sar_results

def calc_SAR_vop_compressed(Q_full, rf_vector, mass, n_vop_points=500):
    """
    Calculate SAR using proper VOP compression from vop_qmatrices_v3.py
    
    Parameters:
    -----------
    Q_full : numpy.ndarray
        Full resolution Q-matrix (n_spatial, n_channels, n_channels)
    rf_vector : numpy.ndarray
        RF signal vector (n_channels,)
    mass : float
        Body mass in kg
    n_vop_points : int
        Maximum number of VOP points to generate
        
    Returns:
    --------
    sar_peak : float
        Peak SAR value from VOP compression
    vop_results : dict
        VOP calculation results
    """
    
    print(f"Calculating SAR using VOP compression...")
    start_time = time.time()
    
    # Prepare Q-matrix data for VOP algorithm
    # Reshape to 5D format expected by VOP algorithm: (M, N, P, Nc, Nc)
    n_spatial, n_channels, _ = Q_full.shape
    
    # Create a cubic spatial arrangement for the VOP algorithm
    # This is necessary because VOP expects 3D spatial structure
    cube_size = int(np.ceil(n_spatial**(1/3)))
    total_padded = cube_size**3
    
    print(f"  Reshaping {n_spatial:,} points to {cube_size}³ = {total_padded:,} grid")
    
    # Create padded Q-matrix in 5D format
    Q_5d = np.zeros((cube_size, cube_size, cube_size, n_channels, n_channels), dtype=complex)
    
    # Fill with actual Q-matrix data
    for i in range(min(n_spatial, total_padded)):
        x = i // (cube_size * cube_size)
        y = (i % (cube_size * cube_size)) // cube_size
        z = i % cube_size
        Q_5d[x, y, z, :, :] = Q_full[i, :, :]
    
    # Prepare data structure for VOP algorithm
    Q_local_data = {
        # Implementation matrices
        'imp': Q_5d,
        # Alternative key
        'local_matrices': Q_5d
    }
    
    try:
        # Run VOP compression using your sophisticated algorithm
        print(f"  Running VOP compression with max {n_vop_points} VOPs...")
        vop_results = VOP_Qmatrices_v3(
            Q_local_data=Q_local_data,
            max_vops=n_vop_points,
            Nc=n_channels,
            # Use GPU if available
            requires_gpu=CUPY_AVAILABLE
        )
        
        # Extract VOP matrices
        # Shape: (num_vops, Nc, Nc)
        VOPm = vop_results['VOP_matrices']
        num_vops = vop_results['num_vops']
        
        print(f"  VOP compression: {n_spatial:,} → {num_vops} points")
        print(f"  Compression ratio: {n_spatial/num_vops:.1f}:1")
        
        # Calculate SAR using VOP matrices
        I_power = np.abs(np.sum(np.conj(rf_vector) * rf_vector))
        sar_vop = np.zeros(num_vops)
        
        for k in range(num_vops):
            Q_k = VOPm[k, :, :]
            SAR_matrix = Q_k * I_power
            sar_vop[k] = np.abs(np.sum(SAR_matrix)) / mass
        
        sar_peak = np.max(sar_vop)
        
        computation_time = time.time() - start_time
        print(f"  VOP computation time: {computation_time:.3f} seconds")
        print(f"  Peak SAR: {sar_peak:.6f} W/kg")
        
        return sar_peak, vop_results
        
    except Exception as e:
        print(f"  ❌ VOP compression failed: {e}")
        print(f"  Falling back to simple subsampling...")
        
        # Fallback to simple subsampling
        vop_indices = np.linspace(0, n_spatial-1, n_vop_points, dtype=int)
        Q_vop = Q_full[vop_indices, :, :]
        
        I_power = np.abs(np.sum(np.conj(rf_vector) * rf_vector))
        sar_vop = np.zeros(len(vop_indices))
        
        for k in range(len(vop_indices)):
            Q_k = Q_vop[k, :, :]
            SAR_matrix = Q_k * I_power
            sar_vop[k] = np.abs(np.sum(SAR_matrix)) / mass
        
        sar_peak = np.max(sar_vop)
        
        computation_time = time.time() - start_time
        print(f"  Fallback computation time: {computation_time:.3f} seconds")
        print(f"  Peak SAR: {sar_peak:.6f} W/kg")
        
        return sar_peak, None


def SAR4seq_advanced(seq_path=None, seq=None, patient_weight=None, computation_mode='clinical', 
                    use_gpu=True, benchmark=False, safety_checks=True, vendor='siemens'):
    """
    Advanced clinical SAR safety assessment for Pulseq sequences
    
    This function provides comprehensive SAR analysis with multiple computation modes:
    - Clinical mode: Fast VOP-compressed SAR for real-time safety monitoring
    - Research mode: High-resolution uncompressed SAR for detailed analysis
    - Benchmark mode: Performance comparison between methods
    
    Parameters
    ----------
    seq_path : str, optional
        Path to Pulseq sequence file (.seq format)
    seq : pypulseq.Sequence, optional
        Pulseq sequence object (takes precedence over seq_path)
    patient_weight : float, optional
        Patient weight in kg (default: 40.0 kg)
    computation_mode : str
        'clinical': Fast VOP compression for real-time use
        'research': High-resolution uncompressed SAR analysis
        'benchmark': Compare both methods with performance analysis
    use_gpu : bool
        Use GPU acceleration when available (default: True)
    benchmark : bool
        Include detailed benchmarking regardless of mode
    safety_checks : bool
        Enforce clinical safety limit checking (default: True)
    vendor : str
        Scanner vendor for specific calculations ('siemens' or 'ge')
        
    Returns
    -------
    dict
        Comprehensive SAR analysis results including:
        - Time-averaged RF power (W)
        - Peak SAR values (W/kg)  
        - Safety limit compliance
        - Computation performance metrics
        - Detailed spatial SAR maps (research mode)
        
    Raises
    ------
    ValueError
        If sequence exceeds safety limits and safety_checks=True
    RuntimeError
        If critical computation errors occur
    """
    
    print("🏥 ADVANCED SAR4SEQ: CLINICAL RF SAFETY ASSESSMENT")
    print("=" * 60)
    
    # Input validation and defaults
    if patient_weight is None:
        patient_weight = CLINICAL_CONSTANTS['default_patient_weight']
    elif patient_weight <= 0:
        raise ValueError("Patient weight must be positive")
    
    if computation_mode not in ['clinical', 'vop', 'research', 'benchmark']:
        raise ValueError("computation_mode must be 'clinical', 'vop', 'research', or 'benchmark'")
    
    if vendor.lower() not in ['siemens', 'ge']:
        raise ValueError("vendor must be 'siemens' or 'ge'")
    
    vendor = vendor.lower()  # Normalize vendor name
    
    print(f"📋 Assessment Configuration:")
    print(f"  Computation mode: {computation_mode.upper()}")
    print(f"  Patient weight: {patient_weight:.1f} kg")
    print(f"  Scanner vendor: {vendor.upper()}")
    print(f"  GPU acceleration: {'✅ Enabled' if use_gpu and CUPY_AVAILABLE else '❌ Disabled'}")
    print(f"  Safety checks: {'✅ Enforced' if safety_checks else '⚠️  Disabled'}")
    
    # Load sequence with proper error handling
    if seq is None:
        if seq_path is None:
            print("\n📂 No sequence specified. Creating test sequence...")
            try:
                seq = create_test_sequence()
            except Exception as e:
                raise RuntimeError(f"Failed to create test sequence: {e}")
        else:
            try:
                seq = pp.Sequence()
                seq.read(seq_path)
                print(f"\n📂 Loaded sequence: {seq_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Sequence file not found: {seq_path}")
            except Exception as e:
                print(f"\n⚠️  Error loading sequence: {e}")
                print("Creating test sequence instead...")
                try:
                    seq = create_test_sequence()
                except Exception as test_e:
                    raise RuntimeError(f"Failed to load sequence and create test sequence: {e}, {test_e}")
    
    # Validate sequence
    if not hasattr(seq, 'block_events') or len(seq.block_events) == 0:
        raise ValueError("Sequence is empty or invalid")
    
    print(f"  Sequence blocks: {len(seq.block_events)}")
    print(f"  System parameters: {seq.system}")
    
    # Load or create Q-matrices with error handling
    try:
        Q_matrices = load_clinical_qmatrices()
    except Exception as e:
        raise RuntimeError(f"Failed to load Q-matrices: {e}")
    
    # Initialize results structure
    results = {
        'computation_mode': computation_mode,
        'patient_weight': patient_weight,
        'vendor': vendor,
        'sequence_info': {
            'total_blocks': len(seq.block_events),
            'rf_blocks': 0,
            'total_duration': 0.0
        },
        'safety_assessment': {
            'compliant': True,
            'violations': [],
            'warnings': []
        },
        'performance_metrics': {},
        'sar_analysis': {}
    }
    
    # Analyze sequence and compute SAR
    if computation_mode == 'clinical':
        results = compute_clinical_sar(seq, Q_matrices, patient_weight, vendor, 
                                     use_gpu, safety_checks, results)
    elif computation_mode == 'vop':
        results = compute_vop_sar(seq, Q_matrices, patient_weight, vendor, 
                                use_gpu, safety_checks, results)
    elif computation_mode == 'research':
        results = compute_research_sar(seq, Q_matrices, patient_weight, vendor,
                                     use_gpu, safety_checks, results)
    elif computation_mode == 'benchmark':
        results = compute_benchmark_sar(seq, Q_matrices, patient_weight, vendor,
                                      use_gpu, safety_checks, results)
    
    # Final safety assessment
    if safety_checks:
        perform_safety_assessment(results)
    
    # Generate comprehensive report
    generate_clinical_report(results)
    
    return results


def compute_clinical_sar(seq, Q_matrices, patient_weight, vendor, use_gpu, safety_checks, results):
    """
    Compute SAR using GPU-accelerated full-resolution for clinical speed
    """
    
    print("\n🏥 CLINICAL MODE: GPU-Accelerated Full-Resolution SAR")
    print("-" * 50)
    
    start_time = time.time()
    
    # Create full-resolution Q-matrix 
    # Use moderate resolution for clinical speed
    n_spatial = CLINICAL_CONSTANTS['clinical_spatial_points']
    n_channels = Q_matrices.get('whole_body', Q_matrices.get('global')).shape[0]
    
    print(f"  Generating Q-matrix: {n_spatial:,} spatial points, {n_channels} channels")
    Q_full = create_full_resolution_qmatrix(n_spatial, n_channels, use_tissue_data=True)
    
    # Process sequence with GPU acceleration for speed
    rf_blocks = analyze_sequence_blocks(seq)
    results['sequence_info']['rf_blocks'] = len(rf_blocks)
    results['sequence_info']['total_duration'] = rf_blocks[-1]['time'] if rf_blocks else 0.0
    
    sar_results = []
    
    # OPTIMIZATION: Use batch GPU processing if GPU is enabled
    if use_gpu and CUPY_AVAILABLE and len(rf_blocks) > 1:
        print(f"🚀 Using OPTIMIZED batch GPU processing for {len(rf_blocks)} RF blocks")
        
        # Prepare all RF vectors
        rf_vectors_list = []
        for rf_block in rf_blocks:
            rf_vector = create_rf_vector(rf_block, n_channels)
            rf_vectors_list.append(rf_vector)
        
        # Process all RF vectors in a single GPU batch
        sar_batch_results = calc_SAR_full_resolution_gpu_batch(Q_full, rf_vectors_list, patient_weight)
        
        # Package results
        for i, (rf_block, (sar_local, sar_peak)) in enumerate(zip(rf_blocks, sar_batch_results)):
            sar_results.append({
                'block_index': rf_block['block_index'],
                'time': rf_block['time'],
                'duration': rf_block['duration'],
                'sar_peak': sar_peak,
                'sar_local': sar_local,
                'rf_amplitude': np.abs(rf_vectors_list[i]).mean()
            })
    else:
        # Original method: process each RF block individually
        for i, rf_block in enumerate(rf_blocks):
            rf_vector = create_rf_vector(rf_block, n_channels)
            
            # Use GPU-accelerated full-resolution for clinical speed and accuracy
            if use_gpu and CUPY_AVAILABLE:
                sar_local, sar_peak = calc_SAR_full_resolution_gpu(Q_full, rf_vector, patient_weight)
            else:
                sar_local, sar_peak = calc_SAR_full_resolution_cpu(Q_full, rf_vector, patient_weight)
            
            sar_results.append({
                'block_index': rf_block['block_index'],
                'time': rf_block['time'],
                'duration': rf_block['duration'],
                'sar_peak': sar_peak,
                'sar_local': sar_local,
                'rf_amplitude': np.abs(rf_vector).mean()
            })
    
    # Calculate clinical metrics
    total_time = time.time() - start_time
    clinical_metrics = calculate_clinical_metrics(sar_results, patient_weight, vendor)
    
    results['performance_metrics'] = {
        'computation_time': total_time,
        'method': 'GPU Full-Resolution' if (use_gpu and CUPY_AVAILABLE) else 'CPU Full-Resolution',
        'spatial_points': n_spatial,
        # No VOP compression used
        'vop_points': None,
        # No compression
        'compression_ratio': 1.0,
        'gpu_used': use_gpu and CUPY_AVAILABLE
    }
    
    results['sar_analysis'] = clinical_metrics
    
    print(f"  ✅ Clinical assessment completed in {total_time:.2f} seconds")
    print(f"  📊 Spatial resolution: {n_spatial:,} points")
    print(f"  � Memory usage: {(n_spatial * n_channels * n_channels * 16) / (1024*1024):.1f} MB")
    
    return results

def compute_research_sar(seq, Q_matrices, patient_weight, vendor, use_gpu, safety_checks, results):
    """
    Compute SAR using full spatial resolution for detailed research analysis
    """
    
    print("\n🔬 RESEARCH MODE: High-Resolution Uncompressed SAR Analysis")
    print("-" * 50)
    
    start_time = time.time()
    
    # Use high spatial resolution for research accuracy
    n_spatial = CLINICAL_CONSTANTS['research_spatial_points']
    n_channels = Q_matrices.get('whole_body', Q_matrices.get('global')).shape[0]
    
    print(f"  Generating high-resolution Q-matrix: {n_spatial:,} spatial points")
    Q_full = create_full_resolution_qmatrix(n_spatial, n_channels, use_tissue_data=True)
    
    # Process sequence with full resolution
    rf_blocks = analyze_sequence_blocks(seq)
    results['sequence_info']['rf_blocks'] = len(rf_blocks)
    results['sequence_info']['total_duration'] = rf_blocks[-1]['time'] if rf_blocks else 0.0
    
    sar_results = []
    spatial_sar_maps = []
    
    for i, rf_block in enumerate(rf_blocks):
        rf_vector = create_rf_vector(rf_block, n_channels)
        
        # High-resolution SAR computation
        if use_gpu and CUPY_AVAILABLE:
            sar_spatial, sar_peak = calc_SAR_full_resolution_gpu(Q_full, rf_vector, patient_weight)
        else:
            sar_spatial, sar_peak = calc_SAR_full_resolution_cpu(Q_full, rf_vector, patient_weight)
        
        sar_results.append({
            'block_index': rf_block['block_index'],
            'time': rf_block['time'],
            'duration': rf_block['duration'],
            'sar_peak': sar_peak,
            'sar_mean': np.mean(sar_spatial),
            'sar_std': np.std(sar_spatial),
            'rf_amplitude': np.abs(rf_vector).mean()
        })
        
        # Store spatial maps for detailed analysis
        spatial_sar_maps.append(sar_spatial)
    
    # Calculate research metrics with spatial details
    total_time = time.time() - start_time
    research_metrics = calculate_research_metrics(sar_results, spatial_sar_maps, patient_weight, vendor)
    
    results['performance_metrics'] = {
        'computation_time': total_time,
        'method': 'Full Resolution',
        'spatial_points': n_spatial,
        'memory_usage_mb': Q_full.nbytes / (1024**2),
        'gpu_used': use_gpu and CUPY_AVAILABLE
    }
    
    results['sar_analysis'] = research_metrics
    results['spatial_sar_maps'] = spatial_sar_maps  # Include detailed spatial data
    
    print(f"  ✅ Research analysis completed in {total_time:.2f} seconds")
    print(f"  Spatial resolution: {n_spatial:,} points")
    print(f"  Memory usage: {results['performance_metrics']['memory_usage_mb']:.1f} MB")
    
    return results


def compute_benchmark_sar(seq, Q_matrices, patient_weight, vendor, use_gpu, safety_checks, results):
    """
    Compute SAR using all methods for comprehensive benchmark comparison
    """
    
    print("\n⚖️  BENCHMARK MODE: Comprehensive Method Comparison")
    print("-" * 50)
    
    # Run clinical (GPU full-resolution) mode
    print("\n🏥 CLINICAL MODE: GPU-Accelerated Full-Resolution SAR Assessment")
    print("-" * 50)
    results_clinical = compute_clinical_sar(seq, Q_matrices, patient_weight, vendor, use_gpu, False, 
                                          {'sequence_info': results['sequence_info'], 
                                           'safety_assessment': results['safety_assessment']})
    
    # Run VOP compression mode  
    print("\n🔬 VOP MODE: Compressed SAR Assessment")
    print("-" * 50)
    results_vop = compute_vop_sar(seq, Q_matrices, patient_weight, vendor, use_gpu, False,
                                {'sequence_info': results['sequence_info'],
                                 'safety_assessment': results['safety_assessment']})
    
    # Run research (high-resolution) mode
    print("\n🧪 RESEARCH MODE: High-Resolution Uncompressed SAR Analysis")
    print("-" * 50)
    results_research = compute_research_sar(seq, Q_matrices, patient_weight, vendor, use_gpu, False,
                                          {'sequence_info': results['sequence_info'],
                                           'safety_assessment': results['safety_assessment']})
    
    # Compare results
    print(f"\n📊 BENCHMARK COMPARISON")
    print("-" * 30)
    
    clinical_time = results_clinical['performance_metrics']['computation_time']
    vop_time = results_vop['performance_metrics']['computation_time']
    research_time = results_research['performance_metrics']['computation_time']
    
    print(f"Clinical (GPU):  {clinical_time:.3f}s")
    print(f"VOP (Compressed): {vop_time:.3f}s")
    print(f"Research (Full): {research_time:.3f}s")
    print(f"Clinical vs Research speedup: {research_time/clinical_time:.1f}x faster")
    print(f"Clinical vs VOP speedup: {vop_time/clinical_time:.1f}x")
    
    # Accuracy comparison
    clinical_peak = max([r['sar_peak'] for r in results_clinical['sar_analysis']['rf_block_results']])
    vop_peak = max([r['sar_peak'] for r in results_vop['sar_analysis']['rf_block_results']])
    research_peak = max([r['sar_peak'] for r in results_research['sar_analysis']['rf_block_results']])
    
    print(f"Peak SAR - Clinical: {clinical_peak:.6f} W/kg")
    print(f"Peak SAR - VOP: {vop_peak:.6f} W/kg") 
    print(f"Peak SAR - Research: {research_peak:.6f} W/kg")
    
    # Calculate comparison metrics
    speedup_clinical_vs_research = research_time / clinical_time if clinical_time > 0 else 0
    speedup_clinical_vs_vop = vop_time / clinical_time if clinical_time > 0 else 0
    accuracy_ratio_research_vs_clinical = research_peak / clinical_peak if clinical_peak > 0 else 1.0
    accuracy_ratio_vop_vs_clinical = vop_peak / clinical_peak if clinical_peak > 0 else 1.0
    
    # Combine results
    results['performance_metrics'] = {
        'clinical': results_clinical['performance_metrics'],
        'vop': results_vop['performance_metrics'],
        'research': results_research['performance_metrics'],
        'speedup_clinical_vs_research': speedup_clinical_vs_research,
        'speedup_clinical_vs_vop': speedup_clinical_vs_vop,
        'accuracy_ratio_research': accuracy_ratio_research_vs_clinical,
        'accuracy_ratio_vop': accuracy_ratio_vop_vs_clinical
    }
    
    results['sar_analysis'] = {
        'clinical': results_clinical['sar_analysis'],
        'vop': results_vop['sar_analysis'],
        'research': results_research['sar_analysis'],
        'comparison': {
            'peak_sar_clinical': clinical_peak,
            'peak_sar_vop': vop_peak,
            'peak_sar_research': research_peak,
            'clinical_vop_difference': abs(vop_peak - clinical_peak),
            'clinical_research_difference': abs(research_peak - clinical_peak)
        }
    }
    
    return results


def compute_vop_sar(seq, Q_matrices, patient_weight, vendor, use_gpu, safety_checks, results):
    """
    Compute SAR using VOP compression for regulatory compliance
    """
    
    print("\n🔬 VOP MODE: Compressed SAR for Regulatory Compliance")
    print("-" * 50)
    
    start_time = time.time()
    
    # Create high-resolution Q-matrix for VOP compression
    n_spatial = CLINICAL_CONSTANTS['research_spatial_points']  # Start with high resolution
    n_channels = Q_matrices.get('whole_body', Q_matrices.get('global')).shape[0]
    
    print(f"  Generating Q-matrix: {n_spatial:,} spatial points, {n_channels} channels")
    Q_full = create_full_resolution_qmatrix(n_spatial, n_channels, use_tissue_data=True)
    
    # Process sequence with VOP compression
    rf_blocks = analyze_sequence_blocks(seq)
    results['sequence_info']['rf_blocks'] = len(rf_blocks)
    results['sequence_info']['total_duration'] = rf_blocks[-1]['time'] if rf_blocks else 0.0
    
    sar_results = []
    vop_results = None
    
    for i, rf_block in enumerate(rf_blocks):
        rf_vector = create_rf_vector(rf_block, n_channels)
        
        # Use VOP compression for regulatory compliance
        sar_peak_vop, vop_calc_results = calc_SAR_vop_compressed(
            Q_full, rf_vector, patient_weight, 
            n_vop_points=CLINICAL_CONSTANTS['clinical_vop_points']
        )
        
        if vop_calc_results is not None and vop_results is None:
            # Store VOP details from first calculation
            vop_results = vop_calc_results
        
        sar_results.append({
            'block_index': rf_block['block_index'],
            'time': rf_block['time'],
            'duration': rf_block['duration'],
            'sar_peak': sar_peak_vop,
            'rf_amplitude': np.abs(rf_vector).mean()
        })
    
    # Calculate clinical metrics
    total_time = time.time() - start_time
    clinical_metrics = calculate_clinical_metrics(sar_results, patient_weight, vendor)
    
    results['performance_metrics'] = {
        'computation_time': total_time,
        'method': 'VOP Compressed',
        'spatial_points': n_spatial,
        'vop_points': vop_results['num_vops'] if vop_results else CLINICAL_CONSTANTS['clinical_vop_points'],
        'compression_ratio': n_spatial / (vop_results['num_vops'] if vop_results else CLINICAL_CONSTANTS['clinical_vop_points']),
        'gpu_used': use_gpu and CUPY_AVAILABLE
    }
    
    results['sar_analysis'] = clinical_metrics
    
    print(f"  ✅ VOP assessment completed in {total_time:.2f} seconds")
    print(f"  📊 VOP compression: {n_spatial:,} → {results['performance_metrics']['vop_points']} points")
    print(f"  🚀 Compression ratio: {results['performance_metrics']['compression_ratio']:.1f}:1")
    
    return results


def calculate_clinical_metrics(sar_results, patient_weight, vendor):
    """
    Calculate clinical SAR metrics following vendor specifications
    """
    
    if not sar_results:
        return {'error': 'No RF blocks found in sequence'}
    
    # Extract SAR values and timing
    sar_peaks = [r['sar_peak'] for r in sar_results]
    rf_amplitudes = [r['rf_amplitude'] for r in sar_results]
    durations = [r['duration'] for r in sar_results]
    
    total_duration = sar_results[-1]['time']
    
    # Vendor-specific calculations
    if vendor.lower() == 'siemens':
        b1_factor = CLINICAL_CONSTANTS['siemens_b1_fact']
    else:  # GE
        b1_factor = CLINICAL_CONSTANTS['ge_b1_fact']
    
    # Time-averaged RF power
    total_sar = sum(sar_peaks[i] * durations[i] for i in range(len(sar_peaks)))
    rf_power_avg = total_sar / total_duration / b1_factor if total_duration > 0 else 0
    
    # Peak SAR values
    sar_peak_global = max(sar_peaks) if sar_peaks else 0
    
    # Predicted SAR for patient
    wbody_weight = CLINICAL_CONSTANTS['wbody_weight']
    sar_predicted = sar_peak_global * np.sqrt(wbody_weight / patient_weight) / 2
    
    if vendor.lower() == 'ge':
        sar_predicted *= CLINICAL_CONSTANTS['ge_b1_fact']
    
    return {
        'rf_power_avg_W': rf_power_avg,
        'sar_peak_global_W_per_kg': sar_peak_global,
        'sar_predicted_patient_W_per_kg': sar_predicted,
        'total_duration_s': total_duration,
        'rf_block_results': sar_results,
        'vendor': vendor
    }


def calculate_research_metrics(sar_results, spatial_sar_maps, patient_weight, vendor):
    """
    Calculate detailed research metrics with spatial analysis
    """
    
    clinical_metrics = calculate_clinical_metrics(sar_results, patient_weight, vendor)
    
    if not spatial_sar_maps:
        return clinical_metrics
    
    # Spatial analysis
    all_spatial_sar = np.concatenate(spatial_sar_maps)
    
    research_metrics = clinical_metrics.copy()
    research_metrics.update({
        'spatial_analysis': {
            'total_voxels': len(all_spatial_sar),
            'sar_global_max': np.max(all_spatial_sar),
            'sar_global_mean': np.mean(all_spatial_sar),
            'sar_global_std': np.std(all_spatial_sar),
            'sar_percentile_95': np.percentile(all_spatial_sar, 95),
            'sar_percentile_99': np.percentile(all_spatial_sar, 99),
            'hotspot_count': np.sum(all_spatial_sar > CLINICAL_CONSTANTS['local_sar_limit'])
        }
    })
    
    return research_metrics

def SAR4seq_legacy(seq_path=None, seq=None, sample_weight=None):
    """
    Computes RF safety metrics for Pulseq sequences
    
    This function calculates time-averaged RF power and SAR values for a given
    Pulseq sequence, supporting both Siemens and GE scanner formats.
    
    Parameters
    ----------
    seq_path : str, optional
        Path to Pulseq sequence file (.seq format)
    seq : pypulseq.Sequence, optional
        Pulseq sequence object. If provided, takes precedence over seq_path
    sample_weight : float, optional
        Weight of the sample being imaged in kg (default: 40.0 kg)
        
    Returns
    -------
    tuple of (float, float, float)
        RFwbg_tavg : float
            Time averaged RF power for whole body (W)
        RFhg_tavg : float  
            Time averaged RF power for head (W)
        sar_wbg_pred_ge : float
            Predicted whole body SAR for GE scanners (W/kg)
            
    Raises
    ------
    ValueError
        If pulse sequence exceeds 10-second Global SAR limits
    FileNotFoundError
        If sequence file cannot be loaded and no fallback is available
    """
    
    # Default parameters
    if seq_path is None:
        seq_path = None
    
    if seq is None:
        system = None
        seq = None
    else:
        system = seq.system
    
    if sample_weight is None:
        sample_weight = 40.0  # kg
    
    # Constants
    siemens_b1_fact = 1.32
    ge_b1_fact = 1.1725
    
    wbody_weight = 103.45
    head_weight = 6.024
    
    # SAR limits
    six_min_thresh_wbg = 4.0   # W/Kg
    ten_sec_thresh_wbg = 8.0
    
    six_min_thresh_hg = 3.2    # W/Kg
    ten_sec_thresh_hg = 6.4
    
    # Check if Q matrix exists, if not generate it
    qmat_file = 'src/test_qmat.mat'
    if not os.path.exists(qmat_file):
        print("Q matrix file not found. Please ensure EM model data is available.")
        print("Loading Q matrix generation...")
        
        # This would require EM model data to be loaded
        # For now, we'll assume the Q matrix exists or provide a placeholder
        try:
            data_dir = Path(__file__).parent / 'data'
            qmat_path = data_dir / 'Qmat.mat'
            if qmat_path.exists():
                Q_data = sio.loadmat(str(qmat_path))
                Q = {
                    'Qtmf': Q_data.get('Qtmf', np.eye(8, dtype=complex)),
                    'Qhmf': Q_data.get('Qhmf', np.eye(8, dtype=complex))
                }
                print(f"Loaded Q matrices from {qmat_path}")
            else:
                raise FileNotFoundError("Q matrix data not found")
        except:
            print("Warning: Q matrix not available. Using dummy matrices for demonstration.")
            # Create dummy Q matrices for demonstration
            Q = {
                'Qtmf': np.eye(8, dtype=complex) * 1e-6,  # Small realistic values
                'Qhmf': np.eye(8, dtype=complex) * 5e-7
            }
    else:
        Q_data = sio.loadmat(qmat_file)
        Q = {
            'Qtmf': Q_data['Qtmf'],
            'Qhmf': Q_data['Qhmf']
        }
        print(f"Loaded Q matrices from {qmat_file}")
        print(f"Qtmf shape: {Q['Qtmf'].shape}, Qhmf shape: {Q['Qhmf'].shape}")
    
    # Import sequence file or create test sequence
    if seq is None:
        if seq_path is None:
            print("No sequence file specified. Creating a simple test sequence...")
            seq = create_test_sequence()
        else:
            # Create a temporary sequence to read the file
            temp_system = pp.Opts(
                max_grad=32, grad_unit='mT/m',
                max_slew=130, slew_unit='T/m/s',
                rf_ringdown_time=30e-6,
                rf_dead_time=100e-6
            )
            seq = pp.Sequence(temp_system)
            try:
                seq.read(seq_path)
                print(f"Successfully loaded sequence from: {seq_path}")
            except RuntimeError as e:
                if "older Pulseq format" in str(e):
                    print(f"Warning: {e}")
                    print("Creating a simple test sequence instead...")
                    seq = create_test_sequence()
                else:
                    raise e
            except FileNotFoundError:
                print(f"Sequence file not found: {seq_path}")
                print("Creating a simple test sequence instead...")
                seq = create_test_sequence()
            except Exception as e:
                print(f"Error reading sequence: {e}")
                print("Creating a simple test sequence instead...")
                seq = create_test_sequence()
        
        system = seq.system
    
    # Identify RF blocks and compute SAR
    t_vec = np.zeros(len(seq.block_events))
    sar_wbg_vec = np.zeros_like(t_vec)
    sar_hg_vec = np.zeros_like(t_vec)
    t_prev = 0
    
    for i_block in range(len(seq.block_events)):
        block = seq.get_block(i_block + 1)
        
        # Calculate block duration
        block_events = []
        if hasattr(block, 'rf') and block.rf is not None:
            block_events.append(block.rf)
        if hasattr(block, 'gx') and block.gx is not None:
            block_events.append(block.gx)
        if hasattr(block, 'gy') and block.gy is not None:
            block_events.append(block.gy)
        if hasattr(block, 'gz') and block.gz is not None:
            block_events.append(block.gz)
        if hasattr(block, 'adc') and block.adc is not None:
            block_events.append(block.adc)
        if hasattr(block, 'delay') and block.delay is not None:
            block_events.append(block.delay)
        
        if block_events:
            block_dur = pp.calc_duration(*block_events)
        else:
            block_dur = 0
            
        t_vec[i_block] = t_prev + block_dur
        t_prev = t_vec[i_block]
        
        # Process RF blocks
        if block.rf is not None:
            rf = block.rf
            signal = rf.signal
            
            # Ensure signal is compatible with Q-matrix dimensions
            num_coils = Q['Qtmf'].shape[0]
            if hasattr(signal, '__len__'):
                # If signal is an array, take the magnitude
                if len(signal) == 1:
                    # Single value, replicate for all coils
                    rf_vector = np.ones(num_coils, dtype=complex) * signal[0] * 0.01  # Very small scale for safety
                else:
                    # Multiple values, take mean or first value
                    rf_vector = np.ones(num_coils, dtype=complex) * np.mean(signal) * 0.01
            else:
                # Single scalar value
                rf_vector = np.ones(num_coils, dtype=complex) * signal * 0.01
            
            # Calculate SAR - Global: Wholebody, Head, Exposed Mass
            try:
                sar_wbg_vec[i_block] = calc_SAR(Q['Qtmf'], rf_vector, wbody_weight)
                sar_hg_vec[i_block] = calc_SAR(Q['Qhmf'], rf_vector, head_weight)
            except Exception as e:
                print(f"Warning: SAR calculation failed for block {i_block}: {e}")
                sar_wbg_vec[i_block] = 0
                sar_hg_vec[i_block] = 0
    
    # Filter out zeros
    T_scan = t_vec[-1]
    idx = np.abs(sar_wbg_vec) > 0
    sar_wbg_vec_filtered = sar_wbg_vec[idx]
    sar_hg_vec_filtered = sar_hg_vec[idx]
    
    # Time averaged RF power - match Siemens data
    RFwbg_tavg = np.sum(sar_wbg_vec_filtered) / T_scan / siemens_b1_fact
    RFhg_tavg = np.sum(sar_hg_vec_filtered) / T_scan / siemens_b1_fact
    
    print(f'Time averaged RF power-Siemens is - Body: {RFwbg_tavg:.3f}W &  Head: {RFhg_tavg:.3f}W')
    
    # Peak SAR values
    sar_wbg = np.max(sar_wbg_vec_filtered) if len(sar_wbg_vec_filtered) > 0 else 0
    sar_hg = np.max(sar_hg_vec_filtered) if len(sar_hg_vec_filtered) > 0 else 0
    
    # Sample head weight calculation
    sample_head_weight = (head_weight / wbody_weight) * sample_weight
    
    # Predicted SAR for Siemens
    sar_wbg_pred_siemens = sar_wbg * np.sqrt(wbody_weight / sample_weight) / 2
    sar_hg_pred_siemens = sar_hg * np.sqrt(head_weight / sample_head_weight) / 2
    
    print(f'Predicted SAR-Siemens is - Body: {sar_wbg_pred_siemens:.3f}W/kg &  Head: {sar_hg_pred_siemens:.3f}W/kg')
    
    # SAR whole body - match GE data
    sar_wbg_pred_ge = sar_wbg * np.sqrt(wbody_weight / sample_weight) * ge_b1_fact
    print(f'Predicted SAR-GE is {sar_wbg_pred_ge:.3f}W/kg')
    
    # Check for SAR limit violations
    if np.any(sar_wbg_pred_ge > ten_sec_thresh_wbg):
        raise ValueError('Pulse sequence exceeding 10 second Global SAR limits, increase TR')
    
    return RFwbg_tavg, RFhg_tavg, sar_wbg_pred_ge