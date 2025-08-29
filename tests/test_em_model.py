#!/usr/bin/env python3
"""
Test electromagnetic model utilities

These tests validate the EM model creation, loading, and validation functions
to ensure they produce physically consistent electromagnetic field data.
"""

import pytest
import numpy as np
import os
import sys

# Import modules to test
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.get_emmodel import (
    create_dummy_em_model, 
    validate_em_model,
    get_EMmodel
)


class TestEMModelCreation:
    """Test electromagnetic model creation and validation"""
    
    def test_dummy_model_basic_structure(self):
        """Test that dummy models have correct basic structure"""
        model = create_dummy_em_model(grid_shape=(8, 8, 4), num_coils=2)
        
        # Check required keys
        required_keys = ['Ex', 'Ey', 'Ez', 'Tissue_types', 'SigmabyRhox', 'Mass_cell']
        for key in required_keys:
            assert key in model, f"Missing required key: {key}"
        
        # Check spatial dimensions
        Nx, Ny, Nz = 8, 8, 4
        assert model['Ex'].shape[:3] == (Nx, Ny, Nz), f"Ex wrong spatial dims: {model['Ex'].shape}"
        assert model['Ey'].shape[:3] == (Nx, Ny, Nz), f"Ey wrong spatial dims: {model['Ey'].shape}"
        assert model['Ez'].shape[:3] == (Nx, Ny, Nz), f"Ez wrong spatial dims: {model['Ez'].shape}"
        
        # Check coil dimension
        assert model['Ex'].shape[3] == 2, f"Ex wrong coil dim: {model['Ex'].shape[3]}"
        assert model['Ey'].shape[3] == 2, f"Ey wrong coil dim: {model['Ey'].shape[3]}"
        assert model['Ez'].shape[3] == 2, f"Ez wrong coil dim: {model['Ez'].shape[3]}"
        
        # Check tissue arrays
        assert model['Tissue_types'].shape == (Nx, Ny, Nz), "Tissue_types wrong shape"
        assert model['SigmabyRhox'].shape == (Nx, Ny, Nz), "SigmabyRhox wrong shape"
        assert model['Mass_cell'].shape == (Nx, Ny, Nz), "Mass_cell wrong shape"
    
    def test_dummy_model_field_properties(self):
        """Test electromagnetic field properties"""
        model = create_dummy_em_model(grid_shape=(10, 10, 6), num_coils=4)
        
        Ex, Ey, Ez = model['Ex'], model['Ey'], model['Ez']
        
        # Fields should be complex
        assert np.iscomplexobj(Ex), "Ex should be complex"
        assert np.iscomplexobj(Ey), "Ey should be complex"
        assert np.iscomplexobj(Ez), "Ez should be complex"
        
        # Fields should have non-zero magnitudes
        assert np.max(np.abs(Ex)) > 0, "Ex should have non-zero values"
        assert np.max(np.abs(Ey)) > 0, "Ey should have non-zero values"
        assert np.max(np.abs(Ez)) > 0, "Ez should have non-zero values"
        
        # Check field magnitude ranges (reasonable for dummy model)
        for field, name in [(Ex, 'Ex'), (Ey, 'Ey'), (Ez, 'Ez')]:
            max_mag = np.max(np.abs(field))
            assert 0.001 < max_mag < 10, f"{name} magnitude seems unreasonable: {max_mag}"
    
    def test_dummy_model_tissue_segmentation(self):
        """Test tissue segmentation in dummy models"""
        model = create_dummy_em_model(grid_shape=(16, 16, 8), num_coils=2)
        
        tissue_types = model['Tissue_types']
        
        # Should have air (0), head (1), and body (2) tissues
        unique_tissues = np.unique(tissue_types)
        expected_tissues = {0, 1, 2}  # air, head, body
        
        assert set(unique_tissues).issubset(expected_tissues), \
            f"Unexpected tissue types: {unique_tissues}"
        
        # Should have some of each tissue type
        assert np.sum(tissue_types == 0) > 0, "Should have air voxels"
        assert np.sum(tissue_types == 1) > 0, "Should have head voxels"
        assert np.sum(tissue_types == 2) > 0, "Should have body voxels"
        
        # Head should be smaller volume than body + air
        head_voxels = np.sum(tissue_types == 1)
        total_voxels = tissue_types.size
        assert head_voxels < total_voxels / 2, "Head should be smaller than half the volume"
    
    def test_dummy_model_tissue_properties(self):
        """Test tissue property assignments"""
        model = create_dummy_em_model(grid_shape=(12, 12, 6), num_coils=2)
        
        tissue_types = model['Tissue_types']
        sigma_by_rhox = model['SigmabyRhox']
        mass_cell = model['Mass_cell']
        
        # Different tissue types should have different properties
        air_mask = tissue_types == 0
        head_mask = tissue_types == 1
        body_mask = tissue_types == 2
        
        if np.any(air_mask) and np.any(head_mask):
            air_sigma = np.mean(sigma_by_rhox[air_mask])
            head_sigma = np.mean(sigma_by_rhox[head_mask])
            assert air_sigma < head_sigma, "Air should have lower conductivity than tissue"
        
        if np.any(air_mask) and np.any(body_mask):
            air_mass = np.mean(mass_cell[air_mask])
            body_mass = np.mean(mass_cell[body_mask])
            assert air_mass < body_mass, "Air should have lower density than tissue"
    
    def test_model_validation(self):
        """Test model validation function"""
        # Valid model should pass
        model = create_dummy_em_model(grid_shape=(8, 8, 4), num_coils=2)
        validate_em_model(model)  # Should not raise exception
        
        # Invalid model should fail
        invalid_model = model.copy()
        del invalid_model['Ex']
        
        with pytest.raises(ValueError, match="Missing required model component"):
            validate_em_model(invalid_model)
        
        # Mismatched dimensions should fail
        invalid_model2 = model.copy()
        invalid_model2['Tissue_types'] = np.zeros((4, 4, 2))  # Wrong size
        
        with pytest.raises(ValueError, match="dimensions don't match"):
            validate_em_model(invalid_model2)
    
    def test_different_grid_sizes(self):
        """Test dummy model creation with various grid sizes"""
        grid_sizes = [
            (4, 4, 2),
            (8, 8, 4),
            (16, 12, 8),
            (20, 20, 10)
        ]
        
        for grid_shape in grid_sizes:
            model = create_dummy_em_model(grid_shape=grid_shape, num_coils=2)
            validate_em_model(model)  # Should pass validation
            
            # Check dimensions
            assert model['Ex'].shape[:3] == grid_shape
            assert model['Tissue_types'].shape == grid_shape
    
    def test_different_coil_counts(self):
        """Test dummy model creation with various coil counts"""
        coil_counts = [1, 2, 4, 8, 16]
        
        for num_coils in coil_counts:
            model = create_dummy_em_model(grid_shape=(8, 8, 4), num_coils=num_coils)
            validate_em_model(model)  # Should pass validation
            
            # Check coil dimension
            assert model['Ex'].shape[3] == num_coils
            assert model['Ey'].shape[3] == num_coils
            assert model['Ez'].shape[3] == num_coils
    
    def test_field_orthogonality_properties(self):
        """Test some basic field properties for the dummy model"""
        model = create_dummy_em_model(grid_shape=(8, 8, 4), num_coils=2)
        
        Ex, Ey, Ez = model['Ex'], model['Ey'], model['Ez']
        
        # Fields from different coils should be different
        if Ex.shape[3] > 1:
            coil1_ex = Ex[:, :, :, 0]
            coil2_ex = Ex[:, :, :, 1]
            
            # They shouldn't be identical
            assert not np.allclose(coil1_ex, coil2_ex), \
                "Different coils should produce different fields"
    
    def test_model_consistency_across_calls(self):
        """Test that model creation is deterministic"""
        model1 = create_dummy_em_model(grid_shape=(8, 8, 4), num_coils=2)
        model2 = create_dummy_em_model(grid_shape=(8, 8, 4), num_coils=2)
        
        # Models should be identical (assuming deterministic generation)
        for key in model1.keys():
            if isinstance(model1[key], np.ndarray):
                assert np.allclose(model1[key], model2[key]), \
                    f"Model creation not consistent for {key}"
    
    def test_physical_units_and_scales(self):
        """Test that physical quantities have reasonable scales"""
        model = create_dummy_em_model(grid_shape=(10, 10, 5), num_coils=2)
        
        # Conductivity/density should be reasonable for biological tissue
        sigma_by_rhox = model['SigmabyRhox']
        tissue_mask = model['Tissue_types'] > 0
        
        if np.any(tissue_mask):
            tissue_sigma = sigma_by_rhox[tissue_mask]
            # Should be in reasonable range for S/m per kg/m³
            assert np.all(tissue_sigma > 1e-10), "Conductivity/density too low"
            assert np.all(tissue_sigma < 10), "Conductivity/density too high"
        
        # Mass should be reasonable
        mass_cell = model['Mass_cell']
        tissue_mass = mass_cell[tissue_mask]
        
        if np.any(tissue_mask):
            # Should be reasonable mass per voxel
            assert np.all(tissue_mass > 1e-10), "Tissue mass too low"
            assert np.all(tissue_mass < 1), "Tissue mass too high"


class TestEMModelFileOperations:
    """Test file I/O operations for EM models"""
    
    def test_model_data_types(self):
        """Test that model arrays have correct data types"""
        model = create_dummy_em_model(grid_shape=(6, 6, 3), num_coils=2)
        
        # Field components should be complex
        assert np.iscomplexobj(model['Ex']), "Ex should be complex"
        assert np.iscomplexobj(model['Ey']), "Ey should be complex"
        assert np.iscomplexobj(model['Ez']), "Ez should be complex"
        
        # Other arrays should be real
        assert np.isrealobj(model['Tissue_types']), "Tissue_types should be real"
        assert np.isrealobj(model['SigmabyRhox']), "SigmabyRhox should be real"
        assert np.isrealobj(model['Mass_cell']), "Mass_cell should be real"


if __name__ == "__main__":
    # Run with: python -m pytest test_em_model.py -v
    pytest.main([__file__, "-v"])
