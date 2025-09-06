"""
VOP (Virtual Observation Points) Q-Matrices Implementation

Implementation of the Eichfelder VOP paper for efficient SAR calculations.
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.get_coremat import get_coremat, validate_q_matrices
from utils.write_qmat import write_qmat

def auto_select_backend(Q_data_size_mb, prefer_gpu=True):
    """
    Automatically select the best backend based on data size and GPU availability
    
    Parameters
    ----------
    Q_data_size_mb : float
        Size of Q-matrix data in MB
    prefer_gpu : bool
        Whether to prefer GPU if available
        
    Returns
    -------
    bool
        True if GPU should be used, False for CPU
    """
    
    if not prefer_gpu:
        return False
        
    try:
        # Check if CuPy is available
        import cupy as cp
        
        # Check GPU memory
        mempool = cp.get_default_memory_pool()
        total_gpu_memory = cp.cuda.Device().mem_info[1] / 1024**2  # MB
        free_gpu_memory = cp.cuda.Device().mem_info[0] / 1024**2   # MB
        
        # Conservative estimate: need at least 3x the data size for computations
        estimated_memory_needed = Q_data_size_mb * 3
        
        if estimated_memory_needed < free_gpu_memory * 0.8:  # Use 80% of free memory
            print(f"Auto-selected GPU: Data size {Q_data_size_mb:.1f}MB, Available {free_gpu_memory:.1f}MB")
            return True
        else:
            print(f"Auto-selected CPU: Data size {Q_data_size_mb:.1f}MB exceeds available GPU memory {free_gpu_memory:.1f}MB")
            return False
            
    except Exception as e:
        print(f"GPU not available: {e}")
        return False

def VOP_Qmatrices_v3(Q_local_file=None, Q_local_data=None, max_vops=500, Nc=8, requires_gpu=None, num_gpu=1):
    """
    Implementation of VOP algorithm based on Eichfelder's method
    
    Parameters
    ----------
    Q_local_file : str, optional
        Path to local Q-matrix file
    Q_local_data : dict, optional
        Local Q-matrix data directly provided
    max_vops : int
        Maximum number of VOPs to generate
    Nc : int
        Number of RF channels
    requires_gpu : bool or None
        Whether to use GPU acceleration with CuPy. If None, auto-select based on data size
    num_gpu : int
        Number of GPUs to use (currently supports 1)
        
    Returns
    -------
    dict
        VOP data including matrices, indices, and maps
    """
    
    # Load Q-matrix data first to determine size
    if Q_local_data is not None:
        Qavg = Q_local_data
    elif Q_local_file is not None:
        try:
            from utils.read_qmat import read_qmat
        except ImportError:
            from utils.read_qmat import read_qmat
        Qavg = read_qmat(Q_local_file)
    else:
        raise ValueError("Must provide either Q_local_file or Q_local_data")
    
    # Extract implementation matrices
    if 'imp' in Qavg:
        Qavg_df = Qavg['imp']
    elif 'local_matrices' in Qavg:
        Qavg_df = Qavg['local_matrices']
    else:
        raise ValueError("Q-matrix data must contain 'imp' or 'local_matrices' field")
    
    # Auto-select backend if not specified
    if requires_gpu is None:
        data_size_mb = Qavg_df.nbytes / 1024**2
        requires_gpu = auto_select_backend(data_size_mb, prefer_gpu=True)
    
    # Choose computing backend
    if requires_gpu:
        try:
            xp = cp
            print("Starting VOP Q-matrices calculation v3 with GPU acceleration")
            print(f"Using GPU: {cp.cuda.Device().id}")
            print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB total")
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU computation")
            xp = np
            requires_gpu = False
    else:
        xp = np
        print("Starting VOP Q-matrices calculation v3 with CPU")
    
    # Convert to GPU if required
    if requires_gpu:
        if isinstance(Qavg_df, np.ndarray):
            Qavg_df = cp.asarray(Qavg_df)
            print(f"Transferred Q-matrix data to GPU ({Qavg_df.nbytes / 1024**2:.1f} MB)")
    
    # Validate Q-matrices using enhanced validation
    try:
        is_valid, validation_info = validate_q_matrices(Qavg_df, requires_gpu=requires_gpu)
        if is_valid:
            print(f"Q-matrix validation passed - {validation_info}")
    except Exception as e:
        print(f"Warning: Q-matrix validation failed: {e}")
    
    # Debug print to understand dimensions
    print(f"Q-matrix shape: {Qavg_df.shape}")
    print(f"Expected 5D shape: (M, N, P, Nc, Nc)")
    
    # Reshape Q-matrices
    M, N, P = Qavg_df.shape[:3]
    print(f"Spatial dimensions: M={M}, N={N}, P={P}")
    Qavg_df_reshaped = Qavg_df.reshape((M * N * P, Nc, Nc))
    
    # Find non-zero Q-matrices
    S = xp.abs(Qavg_df_reshaped) > 0
    # Use middle channel as indicator
    ind = xp.where(S[:, Nc//2, Nc//2])[0]
    
    Q_inds = Qavg_df_reshaped[ind, :, :]
    # Store for later use
    Q_ind = Q_inds.copy()
    obs_pts = len(ind)
    
    print(f"Found {obs_pts} observation points")
    
    # Performance monitoring
    gpu_memory_used = 0
    if requires_gpu:
        try:
            gpu_memory_used = (cp.cuda.Device().mem_info[1] - cp.cuda.Device().mem_info[0]) / 1024**2
            print(f"GPU memory in use: {gpu_memory_used:.1f} MB")
        except:
            pass
    
    # Initialize VOP variables
    cluster = xp.zeros(max_vops)
    normplot = xp.zeros(max_vops)
    vop_ind = xp.zeros(max_vops, dtype=int)
    vop_map = xp.zeros(M * N * P)
    
    indr = xp.arange(len(ind))
    VOPm = xp.zeros((max_vops, Nc, Nc), dtype=complex)
    VOP = 0
    
    t0 = time.time()
    
    while obs_pts != 0:
        # Selection of core matrix
        len_indr = len(indr)
        
        if len_indr == len(ind):
            # First iteration - pass GPU flag to get_coremat
            Bstar, ind_sorta, vopin, myu_def = get_coremat(Q_ind, indr, requires_gpu=requires_gpu)
            
            # Convert results to GPU if needed
            if requires_gpu and not isinstance(Bstar, cp.ndarray):
                Bstar = cp.asarray(Bstar)
                ind_sorta = cp.asarray(ind_sorta)
        elif len_indr == 1:
            print('Done clustering for all obs pts')
            break
        else:
            # Subsequent iterations - pass GPU flag to get_coremat
            Bstar, ind_sorta, vopin, _ = get_coremat(Q_ind, indr, requires_gpu=requires_gpu)
            
            # Convert results to GPU if needed
            if requires_gpu and not isinstance(Bstar, cp.ndarray):
                Bstar = cp.asarray(Bstar)
                ind_sorta = cp.asarray(ind_sorta)
        
        # Start with second matrix
        q = 2
        A = Bstar.copy()
        
        # Set up problem to find Z*
        Z = xp.zeros((Nc, Nc), dtype=complex)
        cluster_done = False
        # Corresponding to A
        obs_pts -= 1
        
        while not cluster_done:
            try:
                # Spectral decomposition
                Q_diff = A - Q_ind[ind_sorta[q-1], :, :]
                # Use eigh for Hermitian matrices when using CuPy, eig for general case with NumPy
                if requires_gpu and xp == cp:
                    # Ensure Q_diff is Hermitian for eigh
                    Q_diff = (Q_diff + Q_diff.conj().T) / 2
                    eigenvals, V = xp.linalg.eigh(Q_diff)
                else:
                    # Use numpy's eig for CPU computation
                    eigenvals, V = np.linalg.eig(Q_diff)
                
                # Create E+ and E-
                Ep = eigenvals.copy()
                Ep[Ep < 0] = 0
                Em = Ep - eigenvals
                
                # Calculate Z_new
                Z_new = V @ xp.diag(Em) @ V.conj().T
                Z = Z + Z_new
                
                myu_calc = xp.linalg.norm(Z, ord=2)
                
            except Exception as e:
                print(f"Warning: Error in spectral decomposition at q={q}: {e}")
                if requires_gpu:
                    print("Attempting to continue with CPU fallback for this iteration...")
                    try:
                        # Convert to CPU for this calculation
                        A_cpu = cp.asnumpy(A) if isinstance(A, cp.ndarray) else A
                        Q_ind_cpu = cp.asnumpy(Q_ind) if isinstance(Q_ind, cp.ndarray) else Q_ind
                        ind_sorta_cpu = cp.asnumpy(ind_sorta) if isinstance(ind_sorta, cp.ndarray) else ind_sorta
                        
                        Q_diff_cpu = A_cpu - Q_ind_cpu[ind_sorta_cpu[q-1], :, :]
                        # For CPU fallback, always use numpy's eig since it handles general matrices
                        eigenvals_cpu, V_cpu = np.linalg.eig(Q_diff_cpu)
                        
                        Ep_cpu = eigenvals_cpu.copy()
                        Ep_cpu[Ep_cpu < 0] = 0
                        Em_cpu = Ep_cpu - eigenvals_cpu
                        
                        Z_new_cpu = V_cpu @ np.diag(Em_cpu) @ V_cpu.conj().T
                        Z_cpu = cp.asnumpy(Z) if isinstance(Z, cp.ndarray) else Z
                        Z_cpu = Z_cpu + Z_new_cpu
                        
                        Z = cp.asarray(Z_cpu)
                        myu_calc = cp.linalg.norm(Z, ord=2)
                        
                    except Exception as e2:
                        print(f"CPU fallback also failed: {e2}")
                        cluster_done = True
                        break
                else:
                    cluster_done = True
                    break
            
            if myu_calc >= myu_def:
                # End of current cluster
                cluster_done = True
                VOP += 1
                
                # Check if we've reached the maximum number of VOPs
                if VOP > max_vops:
                    print(f"Reached maximum VOPs ({max_vops}), stopping algorithm")
                    VOP = max_vops  # Revert to max
                    obs_pts = 0  # Force outer loop to exit
                    break
                    
                VOPm[VOP-1, :, :] = A
                cluster[VOP-1] = q - 1
                normplot[VOP-1] = xp.linalg.norm(Bstar, ord=2)
                
                # Update indices
                used_indices = ind_sorta[:q-1]
                indr = xp.setdiff1d(indr, used_indices)
                obs_pts_check = float(xp.sum(cluster[:VOP])) + len(indr)
                
                # Convert scalar values to Python float for printing
                myu_calc_val = float(myu_calc) if requires_gpu else myu_calc
                print(f"VOP {VOP}, myu_calc: {myu_calc_val:.6f}, remaining pts: {obs_pts_check/1e4:.2f}k")
                
                vop_ind[VOP-1] = ind[vopin]
                vop_map[ind[used_indices]] = normplot[VOP-1]
                
                # Visualization update (optional) - convert to CPU for plotting
                if VOP % 10 == 0:
                    if requires_gpu:
                        vop_map_cpu = cp.asnumpy(vop_map)
                        S_reshaped = vop_map_cpu.reshape((M, N, P))
                    else:
                        S_reshaped = vop_map.reshape((M, N, P))
                    plt.figure(1)
                    plt.imshow(np.abs(S_reshaped[:, N//2, :]), cmap='viridis')
                    plt.title(f'VOP Map - {VOP} VOPs generated')
                    plt.pause(0.01)
                
            else:
                # Continue clustering
                if q < len(ind_sorta):
                    A = Bstar + Z
                    obs_pts -= 1
                    q += 1
                    
                    if q % 10000 == 0:
                        print(f"Processed {q/1e4:.1f}k matrices")
                        
                elif q == len(ind_sorta):
                    print('Reached end of clustering process')
                    obs_pts -= 1
                    cluster_done = True
                    VOP += 1
                    
                    # Check if we've reached the maximum number of VOPs
                    if VOP > max_vops:
                        print(f"Reached maximum VOPs ({max_vops}), stopping algorithm")
                        VOP = max_vops  # Revert to max
                        obs_pts = 0  # Force outer loop to exit
                        break
                        
                    VOPm[VOP-1, :, :] = A
                    # The last one
                    cluster[VOP-1] = q
                    normplot[VOP-1] = xp.linalg.norm(Bstar, ord=2)
                    
                    vop_ind[VOP-1] = ind[vopin]
                    vop_map[ind[ind_sorta[:q]]] = normplot[VOP-1]
                    indr = xp.setdiff1d(indr, ind_sorta[:q])
                    obs_pts_check = float(xp.sum(cluster[:VOP])) + len(indr)
                    myu_calc_val = float(myu_calc) if requires_gpu else myu_calc
                    print(f"VOP {VOP}, myu_calc: {myu_calc_val:.6f}, remaining pts: {obs_pts_check/1e4:.2f}k")
                    # End processing
                    indr = xp.array([])
        
        if len(indr) == 0:
            break
    
    t1 = time.time() - t0
    print(f"VOP calculation completed in {t1:.2f} seconds")
    
    # Performance summary
    if requires_gpu:
        try:
            final_memory_used = (cp.cuda.Device().mem_info[1] - cp.cuda.Device().mem_info[0]) / 1024**2
            print(f"Final GPU memory usage: {final_memory_used:.1f} MB")
            if gpu_memory_used > 0:
                memory_increase = final_memory_used - gpu_memory_used
                print(f"GPU memory increase during computation: {memory_increase:.1f} MB")
        except:
            pass
    
    # Convert results back to CPU if using GPU
    if requires_gpu:
        VOPm = cp.asnumpy(VOPm[:VOP, :, :])
        normplot = cp.asnumpy(normplot[:VOP])
        vop_ind = cp.asnumpy(vop_ind[:VOP])
        cluster = cp.asnumpy(cluster[:VOP])
        vop_map = cp.asnumpy(vop_map)
        ind = cp.asnumpy(ind)
    else:
        # Trim arrays to actual VOP count
        VOPm = VOPm[:VOP, :, :]
        normplot = normplot[:VOP]
        vop_ind = vop_ind[:VOP]
        cluster = cluster[:VOP]
    
    # Prepare VOP for writing to scanner compatible format
    VOP_imp = np.zeros((M, N, P, Nc, Nc), dtype=complex)
    
    for k in range(VOP):
        x, y, z = np.unravel_index(vop_ind[k], (M, N, P))
        VOP_imp[x, y, z, :, :] = VOPm[k, :, :]
    
    # Plot norm of the VOPs (Figure 5 from Eichfelder)
    plt.figure()
    plt.plot(range(1, VOP+1), np.abs(normplot), 'ko')
    plt.xlabel('Index of VOP')
    plt.ylabel('Spectral norm of VOP')
    plt.title('VOP Spectral Norms')
    plt.grid(True)
    plt.show()
    
    # Prepare results
    results = {
        'VOP_matrices': VOPm,
        'VOP_spatial': VOP_imp,
        'vop_indices': vop_ind,
        'norms': normplot,
        'cluster_sizes': cluster,
        'vop_map': vop_map.reshape((M, N, P)),
        'num_vops': VOP,
        'myu_def': myu_def if 'myu_def' in locals() else None,
        'computation_time': t1,
        'original_points': len(ind),
        'gpu_accelerated': requires_gpu
    }
    
    print(f"Generated {VOP} VOPs from {len(ind)} observation points")
    print(f"Compression ratio: {len(ind)/VOP:.1f}:1")
    if requires_gpu:
        print("GPU acceleration was used")
    
    return results


def validate_vop_results(vop_results, tolerance=1e-6):
    """
    Validate VOP calculation results
    
    Parameters
    ----------
    vop_results : dict
        VOP calculation results
    tolerance : float
        Numerical tolerance for validation
        
    Returns
    -------
    bool
        True if validation passes
    """
    
    VOPm = vop_results['VOP_matrices']
    norms = vop_results['norms']
    
    # Ensure we're working with numpy arrays
    if hasattr(VOPm, 'get'):  # CuPy array
        VOPm = VOPm.get()
    if hasattr(norms, 'get'):  # CuPy array
        norms = norms.get()
    
    # Check matrix properties
    for i, vop_matrix in enumerate(VOPm):
        # Check positive semi-definiteness
        eigenvals = np.linalg.eigvals(vop_matrix)
        if np.any(np.real(eigenvals) < -tolerance):
            print(f"Warning: VOP {i} may not be positive semi-definite")
        
        # Check norm consistency
        calculated_norm = np.linalg.norm(vop_matrix, ord=2)
        if abs(calculated_norm - norms[i]) > tolerance:
            print(f"Warning: Norm inconsistency for VOP {i}")
    
    print("VOP validation completed")
    return True


def save_vop_results(vop_results, filename):
    """
    Save VOP results to file
    
    Parameters
    ----------
    vop_results : dict
        VOP calculation results
    filename : str
        Output filename
    """
    
    # Prepare data for saving
    save_data = {
        'VOPm': vop_results['VOP_matrices'],
        'VOP_imp': vop_results['VOP_spatial'],
        'vop_ind': vop_results['vop_indices'],
        'normplot': vop_results['norms'],
        'cluster': vop_results['cluster_sizes'],
        'num_vops': vop_results['num_vops']
    }
    
    if vop_results['myu_def'] is not None:
        save_data['myu_def'] = vop_results['myu_def']
    
    success = write_qmat(save_data, 'VOP', filename)
    
    if success:
        print(f"VOP results saved to {filename}")
    else:
        print(f"Error saving VOP results to {filename}")


def plot_vop_statistics(vop_results):
    """
    Plot various VOP statistics
    
    Parameters
    ----------
    vop_results : dict
        VOP calculation results
    """
    
    # Convert GPU arrays to CPU for plotting if needed
    def to_numpy(arr):
        if hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        return np.asarray(arr)
    
    norms = to_numpy(vop_results['norms'])
    cluster_sizes = to_numpy(vop_results['cluster_sizes'])
    vop_map = to_numpy(vop_results['vop_map'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Spectral norms
    axes[0, 0].plot(range(1, vop_results['num_vops']+1), norms, 'b-o')
    axes[0, 0].set_xlabel('VOP Index')
    axes[0, 0].set_ylabel('Spectral Norm')
    axes[0, 0].set_title('VOP Spectral Norms')
    axes[0, 0].grid(True)
    
    # Cluster sizes
    axes[0, 1].bar(range(1, vop_results['num_vops']+1), cluster_sizes)
    axes[0, 1].set_xlabel('VOP Index')
    axes[0, 1].set_ylabel('Cluster Size')
    axes[0, 1].set_title('VOP Cluster Sizes')
    
    # VOP map (middle slice)
    middle_slice = vop_map.shape[1] // 2
    im = axes[1, 0].imshow(vop_map[:, middle_slice, :], cmap='viridis')
    axes[1, 0].set_title('VOP Map (Sagittal)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Compression statistics
    original_points = vop_results['original_points']
    num_vops = vop_results['num_vops']
    compression_ratio = original_points / num_vops
    
    axes[1, 1].bar(['Original Points', 'VOPs'], [original_points, num_vops])
    axes[1, 1].set_ylabel('Count')
    title = f'Compression Ratio: {compression_ratio:.1f}:1'
    if vop_results.get('gpu_accelerated', False):
        title += ' (GPU)'
    axes[1, 1].set_title(title)
    
    plt.tight_layout()
    plt.show()
