"""
Q-Matrix Generation

Main function for generating Q-matrices for SAR calculations from electromagnetic field data.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from utils.gen_qpwr import gen_Qpwr
from utils.write_qmat import write_qmat
from utils.read_qmat import read_qmat


def Q_mat_gen(sar_type, model, qmat_write=False):
    """
    Generate Q-matrices for SAR calculations
    
    Parameters
    ----------
    sar_type : str
        Type of SAR calculation ('Global' or 'Local')
    model : dict
        Dictionary containing EM model data:
        - Ex, Ey, Ez: Electric field components
        - Tissue_types: Tissue classification array
        - SigmabyRhox: Conductivity/density array
        - Mass_cell: Mass per voxel array
    qmat_write : bool, optional
        Whether to write Q-matrices to file
        
    Returns
    -------
    dict
        Q-matrix data structure
    """
    
    print(f"Q-matrix generation started for {sar_type} SAR")
    
    # Extract model data
    Ex = model['Ex']
    Ey = model['Ey']
    Ez = model['Ez']
    tissue_types = model['Tissue_types']
    sigma_by_rhox = model['SigmabyRhox']
    mass_cell = model['Mass_cell']
    
    t0 = time.time()
    
    if sar_type.lower() == 'global':
        
        print('SAR type: GLOBAL')
        
        # Whole body calculation
        print('Q - Whole body calculation started ....')
        Qavg_df, tissue_types, SbRx, mass_cell, mass_body, _ = gen_Qpwr(
            Ex, Ey, Ez, tissue_types, sigma_by_rhox, mass_cell, 'global', 'wholebody'
        )
        Qavg_tm = Qavg_df / mass_body
        
        # Visualization
        plt.figure(1)
        plt.imshow(np.abs(Qavg_tm), cmap='viridis')
        plt.colorbar()
        plt.title('Implemented - Mass normalized BODY')
        plt.show()
        
        # Head calculation
        print('Q - Head calculation started ....')
        Qavg_df_head, _, _, _, mass_head, _ = gen_Qpwr(
            Ex, Ey, Ez, tissue_types, sigma_by_rhox, mass_cell, 'global', 'head'
        )
        Qavg_hm = Qavg_df_head / mass_head
        
        # Visualization
        plt.figure(2)
        plt.imshow(np.abs(Qavg_hm), cmap='viridis')
        plt.colorbar()
        plt.title('Implemented - Mass normalized HEAD')
        plt.show()
        
        # Create Q structure
        Q = {
            'Qtmf': Qavg_tm,  # Whole body Q-matrix (mass normalized)
            'Qhmf': Qavg_hm   # Head Q-matrix (mass normalized)
            # 'Qemf': Qavg_tom  # TODO: Extremity Q-matrix not implemented
        }
        
        if qmat_write:
            status = write_qmat(Q, 'Global')
            if status:
                print("Global Q-matrices written to file successfully")
            else:
                print("Error writing Q-matrices to file")
    
    elif sar_type.lower() == 'local':
        
        print('SAR type: LOCAL')
        
        # Load existing Q-matrix for comparison (if available)
        try:
            Qavg = read_qmat('avgQMatrix.mat')
            print("Loaded reference Q-matrix for comparison")
            
            # Extract point Q-matrix for comparison
            if 'index' in Qavg and 'avg' in Qavg:
                S = Qavg['index'][:3, :]
                
                # Find specific coordinates (example: 60, 49, 42)
                x_match = np.where(S[0, :] == 60)[0]
                y_match = np.where(S[1, :] == 49)[0]
                z_match = np.where(S[2, :] == 42)[0]
                
                # Find intersection
                xy_match = np.intersect1d(x_match, y_match)
                xyz_match = np.intersect1d(xy_match, z_match)
                
                if len(xyz_match) > 0:
                    s = xyz_match[0]
                    Qpt = Qavg['avg'][s, :, :]
                    
                    plt.figure()
                    plt.imshow(np.abs(Qpt), cmap='viridis')
                    plt.colorbar()
                    plt.title('Reference Q-matrix at point (60,49,42)')
                    plt.show()
        
        except FileNotFoundError:
            print("Reference Q-matrix not found, proceeding with calculation only")
            Qavg = None
        
        # Calculate local Q-matrices
        print('Starting calculation of local Q matrices.....')
        t0_local = time.time()
        
        Qavg_df, tissue_types, SbRx, mass_cell, mass_body, Qpwr2 = gen_Qpwr(
            Ex, Ey, Ez, tissue_types, sigma_by_rhox, mass_cell, 'local', 'wholebody'
        )
        
        t1_local = time.time() - t0_local
        print(f'Done calculating local Q-matrices in {t1_local:.2f} seconds')
        
        # Compare with reference if available
        if Qavg is not None and Qpwr2 is not None:
            try:
                # Extract comparison point
                Qpt_design = Qpwr2[60, 49, 42, :, :]
                
                plt.figure()
                plt.imshow(np.abs(Qpt_design), cmap='viridis')
                plt.colorbar()
                plt.title('Calculated Q-matrix at point (60,49,42)')
                plt.show()
                
                if 'Qpt' in locals():
                    # Comparison plots
                    plt.figure()
                    plt.imshow(np.abs(Qpt) / np.abs(Qpt_design), cmap='viridis')
                    plt.colorbar()
                    plt.title('Ratio: Reference / Calculated')
                    plt.show()
                    
                    plt.figure()
                    plt.imshow(np.abs(np.abs(Qpt) - np.abs(Qpt_design)), cmap='viridis')
                    plt.colorbar()
                    plt.title('Absolute Difference')
                    plt.show()
                
            except (IndexError, KeyError) as e:
                print(f"Error in comparison: {e}")
        
        # Calculate RMSE if reference is available
        if Qavg is not None:
            try:
                RMSE_map, Qtri_map, Qimp_map, NRMSE_map = get_RMSE(Qavg, Qavg_df, mass_cell)
                
                # Display comparison maps
                slice_idx = min(193, Qtri_map.shape[2] - 1)
                
                plt.figure()
                plt.imshow(Qtri_map[:, :, slice_idx], cmap='jet', vmin=0, vmax=2)
                plt.colorbar()
                plt.title('Reference Q-map (slice 193)')
                plt.show()
                
                plt.figure()
                plt.imshow(Qimp_map[:, :, slice_idx], cmap='jet', vmin=0, vmax=2)
                plt.colorbar()
                plt.title('Calculated Q-map (slice 193)')
                plt.show()
                
                diff_map = np.abs(Qtri_map - Qimp_map)
                plt.figure()
                plt.imshow(diff_map[:, :, slice_idx], cmap='jet', vmin=0, vmax=0.1)
                plt.colorbar()
                plt.title('Difference map (slice 193)')
                plt.show()
                
            except Exception as e:
                print(f"Error calculating RMSE: {e}")
        
        # Prepare output structure
        Q = {
            'local_matrices': Qavg_df,
            'tissue_types': tissue_types,
            'mass_cell': mass_cell
        }
        
        if Qavg is not None:
            Q['reference'] = Qavg
            Q['imp'] = Qavg_df  # Implementation matrices
        
        if qmat_write:
            status = write_qmat(Q, 'Local', filename='LocalQ.mat')
            if status:
                print("Local Q-matrices written to file successfully")
            else:
                print("Error writing Q-matrices to file")
    
    else:
        raise ValueError(f"Unknown SAR type: {sar_type}. Must be 'Global' or 'Local'")
    
    t1 = time.time() - t0
    print(f'Q matrices calculated in {t1:.2f} seconds')
    
    return Q


def get_RMSE(Qavg_ref, Qavg_calc, mass_cell):
    """
    Calculate RMSE between reference and calculated Q-matrices
    
    Parameters
    ----------
    Qavg_ref : dict
        Reference Q-matrix data
    Qavg_calc : numpy.ndarray
        Calculated Q-matrices
    mass_cell : numpy.ndarray
        Mass per voxel
        
    Returns
    -------
    tuple
        RMSE_map, Qtri_map, Qimp_map, NRMSE_map
    """
    
    # This is a simplified implementation
    # The full implementation would require detailed comparison
    
    try:
        if 'avg' in Qavg_ref:
            ref_matrices = Qavg_ref['avg']
        else:
            ref_matrices = Qavg_ref
        
        # Calculate norms for visualization
        Qtri_map = np.linalg.norm(ref_matrices, axis=(-2, -1))
        Qimp_map = np.linalg.norm(Qavg_calc, axis=(-2, -1))
        
        # Calculate RMSE
        diff = ref_matrices - Qavg_calc
        RMSE_map = np.sqrt(np.mean(np.abs(diff)**2, axis=(-2, -1)))
        
        # Normalized RMSE
        ref_norm = np.linalg.norm(ref_matrices, axis=(-2, -1))
        NRMSE_map = np.divide(RMSE_map, ref_norm, 
                             out=np.zeros_like(RMSE_map), 
                             where=ref_norm!=0)
        
        return RMSE_map, Qtri_map, Qimp_map, NRMSE_map
        
    except Exception as e:
        print(f"Error in RMSE calculation: {e}")
        # Return dummy maps
        shape = mass_cell.shape
        return (np.zeros(shape), np.zeros(shape), 
                np.zeros(shape), np.zeros(shape))


def validate_q_generation_inputs(sar_type, model):
    """
    Validate inputs for Q-matrix generation
    
    Parameters
    ----------
    sar_type : str
        SAR type
    model : dict
        EM model data
        
    Raises
    ------
    ValueError
        If inputs are invalid
    """
    
    if sar_type.lower() not in ['global', 'local']:
        raise ValueError("sar_type must be 'Global' or 'Local'")
    
    required_fields = ['Ex', 'Ey', 'Ez', 'Tissue_types', 'SigmabyRhox', 'Mass_cell']
    for field in required_fields:
        if field not in model:
            raise ValueError(f"Missing required model field: {field}")
    
    # Check dimensions consistency
    Ex_shape = model['Ex'].shape
    for field in ['Ey', 'Ez']:
        if model[field].shape != Ex_shape:
            raise ValueError(f"Field {field} shape doesn't match Ex")
    
    spatial_shape = Ex_shape[:3]
    for field in ['Tissue_types', 'SigmabyRhox', 'Mass_cell']:
        if model[field].shape != spatial_shape:
            raise ValueError(f"Field {field} spatial dimensions don't match")
    
    print("Q-matrix generation input validation passed")
