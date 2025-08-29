#!/usr/bin/env python3
"""
Test Q-matrix generation functionality

These tests validate that the Q-matrix generation from the Python translation
matches expected behavior from the original MATLAB implementation.
"""

import pytest
import numpy as np
import scipy.io as sio
import os
from pathlib import Path

# Import the modules to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.q_mat_gen import Q_mat_gen
from src.utils.get_emmodel import create_dummy_em_model, get_EMmodel
from src.utils.gen_qpwr import gen_Qpwr


class TestQMatrixGeneration:
    """Test suite for Q-matrix generation functionality"""
    
    @pytest.fixture
    def small_dummy_model(self):
        """Create a small dummy model for fast testing"""
        return create_dummy_em_model(grid_shape=(20, 20, 10), num_coils=2)
    
    @pytest.fixture
    def medium_dummy_model(self):
        """Create a medium dummy model for more comprehensive testing"""
        return create_dummy_em_model(grid_shape=(16, 16, 8), num_coils=4)
    
    def test_q_matrix_structure(self, small_dummy_model):
        """Test that Q-matrices have correct structure and properties"""
        Q = Q_mat_gen('Global', small_dummy_model, qmat_write=False)
        
        # Check that required keys exist
        assert 'Qtmf' in Q, "Qtmf matrix missing"
        assert 'Qhmf' in Q, "Qhmf matrix missing"
        
        # Check matrix dimensions
        num_coils = small_dummy_model['Ex'].shape[3]
        assert Q['Qtmf'].shape == (num_coils, num_coils), f"Qtmf wrong shape: {Q['Qtmf'].shape}"
        assert Q['Qhmf'].shape == (num_coils, num_coils), f"Qhmf wrong shape: {Q['Qhmf'].shape}"
        
        # Check that matrices are complex
        assert np.iscomplexobj(Q['Qtmf']), "Qtmf should be complex"
        assert np.iscomplexobj(Q['Qhmf']), "Qhmf should be complex"
        
        # Check that matrices are Hermitian (Q = Q*)
        assert np.allclose(Q['Qtmf'], Q['Qtmf'].conj().T, rtol=1e-10), "Qtmf should be Hermitian"
        assert np.allclose(Q['Qhmf'], Q['Qhmf'].conj().T, rtol=1e-10), "Qhmf should be Hermitian"
    
    def test_q_matrix_positive_definite(self, small_dummy_model):
        """Test that Q-matrices are positive semi-definite"""
        Q = Q_mat_gen('Global', small_dummy_model, qmat_write=False)
        
        # Check eigenvalues are non-negative (positive semi-definite)
        eig_tmf = np.linalg.eigvals(Q['Qtmf'])
        eig_hmf = np.linalg.eigvals(Q['Qhmf'])
        
        assert np.all(eig_tmf.real >= -1e-10), f"Qtmf not positive semi-definite: {eig_tmf}"
        assert np.all(eig_hmf.real >= -1e-10), f"Qhmf not positive semi-definite: {eig_hmf}"
    
    def test_scaling_properties(self, small_dummy_model):
        """Test that Q-matrices scale correctly with field strength"""
        # Generate Q-matrices with original fields
        Q1 = Q_mat_gen('Global', small_dummy_model, qmat_write=False)
        
        # Scale fields by factor of 2
        scaled_model = small_dummy_model.copy()
        scale_factor = 2.0
        scaled_model['Ex'] = scaled_model['Ex'] * scale_factor
        scaled_model['Ey'] = scaled_model['Ey'] * scale_factor
        scaled_model['Ez'] = scaled_model['Ez'] * scale_factor
        
        Q2 = Q_mat_gen('Global', scaled_model, qmat_write=False)
        
        # Q-matrices should scale by factor^2 (power scales as |E|^2)
        expected_scale = scale_factor**2
        
        assert np.allclose(Q2['Qtmf'], Q1['Qtmf'] * expected_scale, rtol=1e-10), \
            "Qtmf scaling incorrect"
        assert np.allclose(Q2['Qhmf'], Q1['Qhmf'] * expected_scale, rtol=1e-10), \
            "Qhmf scaling incorrect"
    
    def test_tissue_segmentation(self, medium_dummy_model):
        """Test that whole body vs head Q-matrices are different"""
        Q = Q_mat_gen('Global', medium_dummy_model, qmat_write=False)
        
        # Whole body should include more tissue than head
        # So Qtmf (whole body) should generally be larger than Qhmf (head)
        qtmf_norm = np.linalg.norm(Q['Qtmf'])
        qhmf_norm = np.linalg.norm(Q['Qhmf'])
        
        assert qtmf_norm > qhmf_norm, \
            f"Whole body Q-matrix should be larger than head: {qtmf_norm} vs {qhmf_norm}"
    
    def test_different_coil_counts(self):
        """Test Q-matrix generation with different numbers of coils"""
        coil_counts = [2, 4, 8]
        
        for num_coils in coil_counts:
            model = create_dummy_em_model(grid_shape=(8, 8, 4), num_coils=num_coils)
            Q = Q_mat_gen('Global', model, qmat_write=False)
            
            assert Q['Qtmf'].shape == (num_coils, num_coils), \
                f"Wrong Qtmf shape for {num_coils} coils"
            assert Q['Qhmf'].shape == (num_coils, num_coils), \
                f"Wrong Qhmf shape for {num_coils} coils"
    
    def test_reproducibility(self, small_dummy_model):
        """Test that Q-matrix generation is reproducible"""
        Q1 = Q_mat_gen('Global', small_dummy_model, qmat_write=False)
        Q2 = Q_mat_gen('Global', small_dummy_model, qmat_write=False)
        
        assert np.allclose(Q1['Qtmf'], Q2['Qtmf']), "Qtmf not reproducible"
        assert np.allclose(Q1['Qhmf'], Q2['Qhmf']), "Qhmf not reproducible"
    
    def test_mass_conservation(self, medium_dummy_model):
        """Test that mass calculations are consistent"""
        # Test the gen_Qpwr function directly
        Ex = medium_dummy_model['Ex']
        Ey = medium_dummy_model['Ey'] 
        Ez = medium_dummy_model['Ez']
        tissue_types = medium_dummy_model['Tissue_types']
        sigma_by_rhox = medium_dummy_model['SigmabyRhox']
        mass_cell = medium_dummy_model['Mass_cell']
        
        # Calculate whole body and head masses
        _, _, _, _, mass_wb, _ = gen_Qpwr(Ex, Ey, Ez, tissue_types, sigma_by_rhox, 
                                         mass_cell, 'global', 'wholebody')
        _, _, _, _, mass_head, _ = gen_Qpwr(Ex, Ey, Ez, tissue_types, sigma_by_rhox,
                                           mass_cell, 'global', 'head')
        
        # Head mass should be less than whole body mass
        assert mass_head < mass_wb, f"Head mass {mass_head} should be < whole body mass {mass_wb}"
        
        # Check that masses are reasonable (not zero or negative)
        assert mass_wb > 0, f"Whole body mass should be positive: {mass_wb}"
        assert mass_head > 0, f"Head mass should be positive: {mass_head}"
    
    def test_field_magnitude_consistency(self, small_dummy_model):
        """Test that Q-matrices have reasonable magnitudes"""
        Q = Q_mat_gen('Global', small_dummy_model, qmat_write=False)
        
        # Q-matrix elements should have reasonable magnitudes
        # They shouldn't be too large or too small
        qtmf_max = np.max(np.abs(Q['Qtmf']))
        qhmf_max = np.max(np.abs(Q['Qhmf']))
        
        assert qtmf_max > 1e-12, f"Qtmf values too small: {qtmf_max}"
        assert qtmf_max < 1e6, f"Qtmf values too large: {qtmf_max}"
        assert qhmf_max > 1e-12, f"Qhmf values too small: {qhmf_max}"
        assert qhmf_max < 1e6, f"Qhmf values too large: {qhmf_max}"


class TestQMatrixFileIO:
    """Test Q-matrix file input/output functionality"""
    
    def test_qmat_write_read_cycle(self, tmp_path):
        """Test that Q-matrices can be written and read back correctly"""
        # Create test data
        test_Q = {
            'Qtmf': np.random.random((4, 4)).astype(np.complex128) + 
                   1j * np.random.random((4, 4)).astype(np.complex128),
            'Qhmf': np.random.random((4, 4)).astype(np.complex128) + 
                   1j * np.random.random((4, 4)).astype(np.complex128)
        }
        
        # Make matrices Hermitian
        test_Q['Qtmf'] = test_Q['Qtmf'] + test_Q['Qtmf'].conj().T
        test_Q['Qhmf'] = test_Q['Qhmf'] + test_Q['Qhmf'].conj().T
        
        # Write to file
        from src.utils.write_qmat import write_qmat
        from src.utils.read_qmat import read_qmat
        
        test_file = tmp_path / "test_qmatrix.mat"
        success = write_qmat(test_Q, 'Global', str(test_file))
        assert success, "Failed to write Q-matrix file"
        
        # Read back
        Q_loaded = read_qmat(str(test_file))
        assert Q_loaded is not None, "Failed to read Q-matrix file"
        
        # Compare
        assert 'Qtmf' in Q_loaded, "Qtmf missing from loaded data"
        assert 'Qhmf' in Q_loaded, "Qhmf missing from loaded data"
        
        assert np.allclose(Q_loaded['Qtmf'], test_Q['Qtmf']), "Qtmf data mismatch"
        assert np.allclose(Q_loaded['Qhmf'], test_Q['Qhmf']), "Qhmf data mismatch"


if __name__ == "__main__":
    # Run with: python -m pytest test_q_matrix_generation.py -v
    pytest.main([__file__, "-v"])
