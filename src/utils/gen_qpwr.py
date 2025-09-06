"""
Q-matrix Power Generation Utilities

Functions for generating Q-matrices for SAR calculations from electromagnetic field data.
"""

import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Fix import to use relative import
try:
    from .read_qmat import load_tissue_data
except ImportError:
    # Fallback for when run as script
    try:
        from read_qmat import load_tissue_data
    except ImportError:
        # Define a stub function if import fails
        def load_tissue_data():
            print("Warning: Could not import load_tissue_data, using placeholder")
            return None, None

try:
    from .gen_e12ptq import gen_E12ptQ
    from .chkcubair_global import chkcubair_global
except ImportError:
    # Fallback for when module is run directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from gen_e12ptq import gen_E12ptQ
    from chkcubair_global import chkcubair_global


def gen_Qpwr(Ex, Ey, Ez, tissue_types, sigma_by_rhox, mass_cell, sar_type, anatomy):
    """
    Generate Q-matrices for power calculation
    
    Parameters
    ----------
    Ex, Ey, Ez : numpy.ndarray
        Electric field components
    tissue_types : numpy.ndarray
        Tissue type classification array
    sigma_by_rhox : numpy.ndarray
        Conductivity divided by density array
    mass_cell : numpy.ndarray
        Mass per voxel array
    sar_type : str
        Type of SAR calculation ('global' or 'local')
    anatomy : str
        Anatomical region ('wholebody', 'head', 'torso', 'extremities')
        
    Returns
    -------
    tuple
        Qpwr_df : numpy.ndarray
            Q-matrix for power calculation
        tissue_types : numpy.ndarray
            Tissue types array
        sigma_by_rhox : numpy.ndarray
            Conductivity/density array
        mass_cell : numpy.ndarray
            Mass cell array
        mass_corr : float
            Corrected total mass
        Qpwr2 : numpy.ndarray, optional
            5D Q-matrix array (for local SAR)
    """
    
    D = tissue_types.shape
    mass_corr = 0
    mass_air = 1.625e-7 + 1e-9
    
    if sar_type.lower() == 'global':
        # Based on body segmentation
        if anatomy.lower() == 'wholebody':
            R = np.where(tissue_types > 0)
        elif anatomy.lower() == 'head':
            R = np.where(tissue_types == 1)
        elif anatomy.lower() == 'torso':
            R = np.where(tissue_types == 2)
        elif anatomy.lower() == 'extremities':
            # TODO: Implement extremities segmentation
            raise NotImplementedError("Extremities segmentation not implemented yet")
        else:
            raise ValueError(f"Unknown anatomy type: {anatomy}")
        
        X, Y, Z = R
        
        # Determine number of coils from field dimensions
        num_coils = Ex.shape[3] if len(Ex.shape) > 3 else 1
        Qpwr = np.zeros((num_coils, num_coils), dtype=complex)
        mass_corr = 0
        
        print('Performing global Q calculations....')
        dim = 2
        t0_gQ = time.time()
        
        for r in range(len(X)):
            Cr = [X[r], Y[r], Z[r]]
            M = mass_cell[X[r], Y[r], Z[r]]
            
            chk = chkcubair_global(dim, mass_cell, mass_air, X[r], Y[r], Z[r])
            
            if (M > mass_air) and (chk == 1):
                Q_contrib = gen_E12ptQ(Ex, Ey, Ez, Cr, sigma_by_rhox)
                Qpwr = Qpwr + (M * Q_contrib)
                mass_corr = mass_corr + M
                
                if (r + 1) % 1000 == 0:
                    print(f"Processed {r + 1} voxels")
        
        t1_gQ = time.time() - t0_gQ
        print(f'Global Q matrices calculated in {t1_gQ:.2f} seconds')
        print(f"Total voxels processed: {len(X)}")
        
        Qpwr_df = Qpwr
        return Qpwr_df, tissue_types, sigma_by_rhox, mass_cell, mass_corr, None
        
    elif sar_type.lower() == 'local':
        # Prepare for parallel processing
        M, N, P = D
        
        print('Preparing for parallel loop... making indices')
        # Store indices only which have mass > air
        ind = mass_cell > mass_air
        ms = np.where(ind.flatten())[0]
        
        print('Calculating Qpwr now')
        Qpwr = np.zeros((len(ms), 8, 8), dtype=complex)
        
        t0_lQ = time.time()
        
        # Parallel computation of Q-matrices
        def compute_q_matrix(k):
            idx = ms[k]
            m, n, p = np.unravel_index(idx, D)
            return gen_E12ptQ(Ex, Ey, Ez, [m, n, p], sigma_by_rhox)
        
        # Use multiprocessing for parallel computation
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(compute_q_matrix, range(len(ms))))
        
        for k, result in enumerate(results):
            Qpwr[k, :, :] = result
        
        print('Creating the 5D Qpwr matrix...')
        Qpwr2 = np.zeros((M * N * P, 8, 8), dtype=complex)
        Qpwr2[ms, :, :] = Qpwr
        Qpwr2 = Qpwr2.reshape((M, N, P, 8, 8))
        
        t1_lQ = time.time() - t0_lQ
        print(f'Local Q matrices calculated in {t1_lQ:.2f} seconds')
        
        print('Cleared variables to make space....')
        
        # Average Q over 10g volumes
        print('Calculating Mass-averaged local Q matrices...')
        Mdef = 0.01  # kg for IEC required mass per arbitrary volume V
        Qpwr_df = get_Qavg(mass_cell, Mdef, Qpwr2, ms)
        
        return Qpwr_df, tissue_types, sigma_by_rhox, mass_cell, mass_corr, Qpwr2
    
    else:
        raise ValueError(f"Unknown SAR type: {sar_type}")


def get_Qavg(mass_cell, Mdef, Qpwr2, ms):
    """
    Calculate mass-averaged Q-matrices over specified volumes
    
    Parameters
    ----------
    mass_cell : numpy.ndarray
        Mass per voxel
    Mdef : float
        Target mass for averaging (0.01 kg for 10g)
    Qpwr2 : numpy.ndarray
        5D Q-matrix array
    ms : numpy.ndarray
        Indices of non-air voxels
        
    Returns
    -------
    numpy.ndarray
        Mass-averaged Q-matrices
    """
    
    print("Computing mass-averaged Q-matrices...")
    
    # This is a simplified implementation
    # The full implementation would require complex spatial averaging
    # over 10g volumes according to IEC standards
    
    M, N, P, _, _ = Qpwr2.shape
    Qavg = np.zeros_like(Qpwr2)
    
    # Simple box averaging as placeholder
    # Real implementation would use more sophisticated volume selection
    kernel_size = 3  # Approximate kernel for 10g averaging
    
    for i in range(kernel_size//2, M - kernel_size//2):
        for j in range(kernel_size//2, N - kernel_size//2):
            for k in range(kernel_size//2, P - kernel_size//2):
                
                # Extract local neighborhood
                i_start, i_end = i - kernel_size//2, i + kernel_size//2 + 1
                j_start, j_end = j - kernel_size//2, j + kernel_size//2 + 1
                k_start, k_end = k - kernel_size//2, k + kernel_size//2 + 1
                
                local_mass = mass_cell[i_start:i_end, j_start:j_end, k_start:k_end]
                local_Q = Qpwr2[i_start:i_end, j_start:j_end, k_start:k_end, :, :]
                
                total_mass = np.sum(local_mass)
                
                if total_mass >= Mdef:
                    # Mass-weighted average
                    weights = local_mass / total_mass
                    for ch1 in range(8):
                        for ch2 in range(8):
                            Qavg[i, j, k, ch1, ch2] = np.sum(
                                weights * local_Q[:, :, :, ch1, ch2]
                            )
    
    return Qavg


def validate_em_fields(Ex, Ey, Ez):
    """
    Validate electromagnetic field data
    
    Parameters
    ----------
    Ex, Ey, Ez : numpy.ndarray
        Electric field components
        
    Returns
    -------
    bool
        True if fields are valid
    """
    
    # Check dimensions match
    if not (Ex.shape == Ey.shape == Ez.shape):
        raise ValueError("Electric field components must have same dimensions")
    
    # Check for complex data
    if not (np.iscomplexobj(Ex) and np.iscomplexobj(Ey) and np.iscomplexobj(Ez)):
        print("Warning: Electric fields should be complex-valued")
    
    # Check for reasonable field magnitudes
    max_field = max(np.max(np.abs(Ex)), np.max(np.abs(Ey)), np.max(np.abs(Ez)))
    if max_field > 1e6:  # Arbitrary large value check
        print(f"Warning: Very large field values detected (max: {max_field:.2e})")
    
    return True


def create_full_resolution_qmatrix(n_spatial_points, n_channels, base_coupling=1e-6, use_tissue_data=True):
    """
    Create a full spatial resolution Q-matrix for SAR computation
    
    Parameters:
    -----------
    n_spatial_points : int
        Number of spatial points (voxels)
    n_channels : int
        Number of RF channels
    base_coupling : float
        Base electromagnetic coupling strength
    use_tissue_data : bool
        Whether to use realistic tissue data from .mat file
        
    Returns:
    --------
    Q_full : numpy.ndarray
        Full resolution Q-matrix with shape (n_spatial_points, n_channels, n_channels)
    """
    
    print(f"Creating full-resolution Q-matrix...")
    print(f"  Spatial points: {n_spatial_points}")
    print(f"  RF channels: {n_channels}")
    print(f"  Total elements: {n_spatial_points * n_channels * n_channels:,}")
    
    # Initialize Q-matrix
    Q_full = np.zeros((n_spatial_points, n_channels, n_channels), dtype=complex)
    
    if use_tissue_data:
        # Load realistic tissue data
        tissue_data, tissue_properties = load_tissue_data()
        
        if tissue_data is not None and tissue_properties is not None:
            print(f"  Using realistic tissue electromagnetic properties")
            
            # Flatten tissue data for sampling
            tissue_flat = tissue_data.flatten()
            total_voxels = len(tissue_flat)
            
            # Sample voxels from tissue data
            if n_spatial_points <= total_voxels:
                # Subsample tissue data
                indices = np.random.choice(total_voxels, n_spatial_points, replace=False)
                sampled_tissues = tissue_flat[indices]
            else:
                # Repeat tissue data if we need more points
                repeat_factor = (n_spatial_points // total_voxels) + 1
                extended_tissues = np.tile(tissue_flat, repeat_factor)
                sampled_tissues = extended_tissues[:n_spatial_points]
            
            # Create Q-matrix based on tissue properties
            for k in range(n_spatial_points):
                tissue_type = int(np.real(sampled_tissues[k]))
                
                # Get tissue properties or use default for unknown types
                if tissue_type in tissue_properties:
                    props = tissue_properties[tissue_type]
                    conductivity = props['conductivity']
                    permittivity = props['permittivity']
                else:
                    # Default to air properties for unknown tissue types
                    conductivity = 0.0
                    permittivity = 1.0
                
                # Calculate coupling strength based on tissue properties
                # Higher conductivity and permittivity lead to stronger coupling
                coupling_strength = base_coupling * (1 + conductivity * 10 + permittivity * 0.1)
                
                # Create Q-matrix for this voxel with realistic channel coupling
                for i in range(n_channels):
                    for j in range(n_channels):
                        if i == j:
                            # Self-coupling (stronger)
                            Q_full[k, i, j] = coupling_strength * (1.0 + 0.1j)
                        else:
                            # Cross-coupling (weaker, varies by channel separation)
                            separation_factor = abs(i - j) / n_channels
                            cross_coupling = coupling_strength * (0.3 + 0.05j) * (1 - separation_factor * 0.5)
                            Q_full[k, i, j] = cross_coupling
                
                # Progress reporting
                if (k + 1) % (n_spatial_points // 20) == 0:  # Report every 5%
                    progress = (k + 1) / n_spatial_points * 100
                    tissue_name = tissue_properties.get(tissue_type, {}).get('name', f'Type_{tissue_type}')
                    print(f"    Progress: {progress:.1f}% - Processing tissue type {tissue_type}")
            
        else:
            print(f"  Tissue data not available, using synthetic model")
            use_tissue_data = False
    
    if not use_tissue_data:
        # Fall back to synthetic Q-matrix model
        print(f"  Using synthetic electromagnetic coupling model")
        
        # Create realistic spatial variation
        np.random.seed(42)  # For reproducible results
        
        for k in range(n_spatial_points):
            # Simulate spatial variation in tissue properties
            spatial_factor = 1.0 + 0.5 * np.sin(k * 2 * np.pi / n_spatial_points)
            coupling_strength = base_coupling * spatial_factor
            
            # Create Q-matrix for this spatial location
            for i in range(n_channels):
                for j in range(n_channels):
                    if i == j:
                        # Self-coupling
                        Q_full[k, i, j] = coupling_strength * (1.0 + 0.1j)
                    else:
                        # Cross-coupling decreases with channel separation
                        separation = abs(i - j)
                        cross_coupling = coupling_strength * (0.5 + 0.05j) / (1 + separation * 0.2)
                        Q_full[k, i, j] = cross_coupling
            
            # Progress reporting for synthetic model
            if (k + 1) % (n_spatial_points // 10) == 0:  # Report every 10%
                progress = (k + 1) / n_spatial_points * 100
                print(f"    Progress: {progress:.1f}%")
    
    # Report memory usage
    memory_mb = Q_full.nbytes / (1024 * 1024)
    print(f"  Q-matrix memory usage: {memory_mb:.1f} MB")
    
    return Q_full