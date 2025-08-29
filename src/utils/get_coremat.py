"""
Core Matrix Computation

Functions for computing core matrices in VOP (Virtual Observation Points) algorithm.
"""

import numpy as np
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def get_coremat_simple(Q_inds, ind):
    """
    Simplified version without parallel processing for small datasets
    
    Parameters
    ----------
    Q_inds : numpy.ndarray
        Q-matrices with shape (N, Nc, Nc)
    ind : numpy.ndarray
        Indices of matrices to consider
        
    Returns
    -------
    tuple
        Bstar : numpy.ndarray
            Core matrix
        ind_sort : numpy.ndarray
            Sorted indices
        vop_ind : int
            Index of the VOP
    """
    
    Q_ind = Q_inds[ind, :, :]
    
    # Calculate norms
    B = np.array([np.linalg.norm(Q_ind[k, :, :], ord=2) for k in range(len(ind))])
    
    # Find maximum
    max_k = np.argmax(B)
    vop_ind = ind[max_k]
    Bstar = Q_ind[max_k, :, :]
    
    # Calculate minimum eigenvalues
    lambda_min = np.zeros(len(ind))
    for k in range(len(ind)):
        Q_temp = Bstar - Q_ind[k, :, :]
        eigenvals = np.linalg.eigvals(Q_temp)
        lambda_min[k] = np.min(np.real(eigenvals))
    
    # Sort indices
    sort_indices = np.argsort(lambda_min)[::-1]
    ind_sort = ind[sort_indices]
    
    return Bstar, ind_sort, vop_ind

def get_coremat(Q_inds, ind, myu_per=0.01, requires_gpu=False, gpu_count=1):
    """
    Compute core matrix for VOP algorithm with optional GPU acceleration
    
    Parameters
    ----------
    Q_inds : numpy.ndarray or cupy.ndarray
        Q-matrices with shape (N, Nc, Nc)
    ind : numpy.ndarray or cupy.ndarray
        Indices of matrices to consider
    myu_per : float
        Percentage for myu definition
    requires_gpu : bool
        Whether to use GPU acceleration
    gpu_count : int
        Number of GPUs (currently unused)
        
    Returns
    -------
    tuple
        Bstar : array
            Core matrix
        ind_sort : array
            Sorted indices
        vop_ind : int
            Index of the VOP
        myu_def : float
            Myu definition value
    """
    
    if requires_gpu:
        try:
            # Convert inputs to CuPy arrays if not already
            if not isinstance(Q_inds, cp.ndarray):
                Q_inds = cp.asarray(Q_inds)
            if not isinstance(ind, cp.ndarray):
                ind = cp.asarray(ind)
            
            Q_ind = Q_inds[ind, :, :]
            
            # Vectorized norm calculation for GPU efficiency
            B = cp.linalg.norm(Q_ind, ord=2, axis=(1, 2))
            
            # Find maximum norm
            max_k = cp.argmax(B)
            max_B = B[max_k]
            vop_ind = int(ind[max_k])  # Convert to Python int
            myu_def = float(myu_per * max_B)  # Convert to Python float
            Bstar = Q_ind[max_k, :, :]
            
            # Vectorized eigenvalue computation
            # Create difference matrices for all k at once
            Bstar_expanded = cp.expand_dims(Bstar, 0)
            Q_diff = Bstar_expanded - Q_ind
            
            # Compute minimum eigenvalues for all matrices
            lambda_min = cp.zeros(len(ind))
            for k in range(len(ind)):
                eigenvals = cp.linalg.eigvalsh(Q_diff[k])
                lambda_min[k] = cp.min(cp.real(eigenvals))
            
            # Sort indices
            sort_indices = cp.argsort(lambda_min)[::-1]
            ind_sort = ind[sort_indices]
            
            return Bstar, ind_sort, vop_ind, myu_def
            
        except Exception as e:
            print(f"GPU computation failed: {e}")
            print("Falling back to CPU computation")
            # Convert back to numpy and continue with CPU
            if isinstance(Q_inds, cp.ndarray):
                Q_inds = cp.asnumpy(Q_inds)
            if isinstance(ind, cp.ndarray):
                ind = cp.asnumpy(ind)
    
    # CPU computation with parallel processing
    Q_ind = Q_inds[ind, :, :]
    
    def compute_norm(k):
        return np.linalg.norm(Q_ind[k, :, :], ord=2)
    
    # Parallel computation of norms
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        B = list(executor.map(compute_norm, range(len(ind))))
    
    B = np.array(B)
    
    # Find maximum norm
    max_k = np.argmax(B)
    max_B = B[max_k]
    vop_ind = int(ind[max_k])
    myu_def = float(myu_per * max_B)
    Bstar = Q_ind[max_k, :, :]
    
    # Generate difference matrices and find minimum eigenvalues
    def compute_min_eigenvalue(k):
        Q_temp = Bstar - Q_ind[k, :, :]
        eigenvals = np.linalg.eigvals(Q_temp)
        return np.min(np.real(eigenvals))
    
    # Parallel computation of minimum eigenvalues
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        lambda_min = list(executor.map(compute_min_eigenvalue, range(len(ind))))
    
    lambda_min = np.array(lambda_min)
    
    # Sort indices by minimum eigenvalue (descending)
    sort_indices = np.argsort(lambda_min)[::-1]
    ind_sort = ind[sort_indices]
    
    return Bstar, ind_sort, vop_ind, myu_def

def validate_q_matrices(Q_matrices, requires_gpu=False):
    """
    Validate Q-matrices for VOP computation with optional GPU support
    
    Parameters
    ----------
    Q_matrices : numpy.ndarray or cupy.ndarray
        Array of Q-matrices
    requires_gpu : bool
        Whether to use GPU acceleration
        
    Returns
    -------
    tuple
        (bool, str) - True if matrices are valid, and computation info
    """
    
    if requires_gpu:
        try:
            if not isinstance(Q_matrices, cp.ndarray):
                Q_matrices = cp.asarray(Q_matrices)

            if Q_matrices.ndim != 3:
                raise ValueError("Q_matrices must be 3D array (N, Nc, Nc)")
        
            N, Nc1, Nc2 = Q_matrices.shape
            if Nc1 != Nc2:
                raise ValueError("Q-matrices must be square")
            
            # Check for positive semi-definiteness using GPU
            for i in range(N):
                eigenvals = cp.linalg.eigvalsh(Q_matrices[i])
                if cp.any(cp.real(eigenvals) < -1e-10):  # Small tolerance for numerical errors
                    print(f"Warning: Q-matrix {i} may not be positive semi-definite")
            
            return True, "Validated on GPU"
            
        except Exception as e:
            print(f"GPU validation failed: {e}")
            print("Falling back to CPU validation")
            # Convert to numpy and continue with CPU
            if isinstance(Q_matrices, cp.ndarray):
                Q_matrices = cp.asnumpy(Q_matrices)
    
    # CPU validation
    if Q_matrices.ndim != 3:
        raise ValueError("Q_matrices must be 3D array (N, Nc, Nc)")
    
    N, Nc1, Nc2 = Q_matrices.shape
    if Nc1 != Nc2:
        raise ValueError("Q-matrices must be square")
    
    # Check for positive semi-definiteness
    for i in range(N):
        eigenvals = np.linalg.eigvals(Q_matrices[i])
        if np.any(np.real(eigenvals) < -1e-10):  # Small tolerance for numerical errors
            print(f"Warning: Q-matrix {i} may not be positive semi-definite")
    
    return True, "Validated on CPU"

def compute_matrix_statistics(Q_matrices, requires_gpu=False):
    """
    Compute statistics for Q-matrices with optional GPU acceleration

    Parameters
    ----------
    Q_matrices : numpy.ndarray or cupy.ndarray
        Array of Q-matrices
    requires_gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    tuple
        (dict, str) - Dictionary containing statistics and computation info
    """
    
    if requires_gpu:
        try:
            if not isinstance(Q_matrices, cp.ndarray):
                Q_matrices = cp.asarray(Q_matrices)
            
            N, Nc, _ = Q_matrices.shape

            # Vectorized norms computation
            norms = cp.linalg.norm(Q_matrices, ord=2, axis=(1, 2))

            # Improved condition number computation
            def compute_condition_number_gpu(matrix):
                try:
                    s = cp.linalg.svd(matrix, compute_uv=False)
                    return s[0] / s[-1] if s[-1] > 1e-15 else cp.inf
                except:
                    return cp.inf
            
            cond_nums = cp.array([compute_condition_number_gpu(Q_matrices[i]) for i in range(N)])

            # Vectorized traces and determinants
            traces = cp.trace(Q_matrices, axis1=1, axis2=2)
            dets = cp.array([cp.linalg.det(Q_matrices[i]) for i in range(N)])

            # Convert to Python scalars for serialization
            stats = {
                'norms': {
                    'mean': float(cp.mean(norms)), 'std': float(cp.std(norms)), 
                    'max': float(cp.max(norms)), 'min': float(cp.min(norms))
                },
                'condition_numbers': {
                    'mean': float(cp.mean(cond_nums[cp.isfinite(cond_nums)])), 
                    'std': float(cp.std(cond_nums[cp.isfinite(cond_nums)])), 
                    'max': float(cp.max(cond_nums[cp.isfinite(cond_nums)]))
                },
                'traces': {
                    'mean': float(cp.mean(traces)), 'std': float(cp.std(traces)), 
                    'max': float(cp.max(traces)), 'min': float(cp.min(traces))
                },
                'determinants': {
                    'mean': float(cp.mean(dets)), 'std': float(cp.std(dets)), 
                    'max': float(cp.max(dets)), 'min': float(cp.min(dets))
                }
            }
            
            return stats, "Computed on GPU"
            
        except Exception as e:
            print(f"GPU statistics computation failed: {e}")
            print("Falling back to CPU computation")
            # Convert to numpy and continue with CPU
            if isinstance(Q_matrices, cp.ndarray):
                Q_matrices = cp.asnumpy(Q_matrices)

    # CPU computation
    N, Nc, _ = Q_matrices.shape

    # Compute norms
    norms = np.array([np.linalg.norm(Q_matrices[i], ord=2) for i in range(N)])

    # Compute condition numbers with error handling
    def compute_condition_number_cpu(matrix):
        try:
            return np.linalg.cond(matrix)
        except:
            return np.inf
    
    cond_nums = np.array([compute_condition_number_cpu(Q_matrices[i]) for i in range(N)])

    # Compute traces
    traces = np.array([np.trace(Q_matrices[i]) for i in range(N)])

    # Compute determinants
    dets = np.array([np.linalg.det(Q_matrices[i]) for i in range(N)])

    # Filter out infinite condition numbers for statistics
    finite_cond_nums = cond_nums[np.isfinite(cond_nums)]
    
    stats = {
        'norms': {
            'mean': float(np.mean(norms)), 'std': float(np.std(norms)), 
            'max': float(np.max(norms)), 'min': float(np.min(norms))
        },
        'condition_numbers': {
            'mean': float(np.mean(finite_cond_nums)) if len(finite_cond_nums) > 0 else 0.0, 
            'std': float(np.std(finite_cond_nums)) if len(finite_cond_nums) > 0 else 0.0, 
            'max': float(np.max(finite_cond_nums)) if len(finite_cond_nums) > 0 else 0.0
        },
        'traces': {
            'mean': float(np.mean(traces)), 'std': float(np.std(traces)), 
            'max': float(np.max(traces)), 'min': float(np.min(traces))
        },
        'determinants': {
            'mean': float(np.mean(dets)), 'std': float(np.std(dets)), 
            'max': float(np.max(dets)), 'min': float(np.min(dets))
        }
    }

    return stats, "Computed on CPU"
  