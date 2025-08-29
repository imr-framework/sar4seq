#!/usr/bin/env python3
"""
Test SAR calculation functionality

These tests validate the SAR calculation functions to ensure they produce
physically meaningful and mathematically correct results.
"""

import pytest
import numpy as np
import os
import sys

# Import modules to test
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.calc_sar import calc_SAR
from src.utils.get_emmodel import create_dummy_em_model
from src.q_mat_gen import Q_mat_gen


class TestSARCalculation:
    """Test suite for SAR calculation functionality"""
    
    @pytest.fixture
    def test_model_and_q(self):
        """Create test model with Q-matrices"""
        # Create a dummy EM model with larger grid for valid calculations
        model = create_dummy_em_model(grid_shape=(20, 20, 10), num_coils=4)
        
        # Generate Q-matrices
        Q = Q_mat_gen('Global', model, qmat_write=False)
        
        return model, Q
    
    def test_sar_basic_calculation(self, test_model_and_q):
        """Test basic SAR calculation functionality"""
        model, Q = test_model_and_q
        
        # Create test RF signal
        num_coils = Q['Qtmf'].shape[0]
        rf_signal = np.ones(num_coils, dtype=complex) * 0.1
        
        # Calculate SAR
        sar_wb = calc_SAR(Q['Qtmf'], rf_signal, 70.0)  # 70kg patient
        sar_head = calc_SAR(Q['Qhmf'], rf_signal, 6.0)  # 6kg head
        
        # Check that SAR values are reasonable
        assert 0 <= sar_wb <= 100, f"Whole body SAR out of range: {sar_wb} W/kg"
        assert 0 <= sar_head <= 100, f"Head SAR out of range: {sar_head} W/kg"
        
        # SAR should be real and positive
        assert np.isreal(sar_wb), "Whole body SAR should be real"
        assert np.isreal(sar_head), "Head SAR should be real"
        assert sar_wb >= 0, "Whole body SAR should be non-negative"
        assert sar_head >= 0, "Head SAR should be non-negative"
    
    def test_sar_scaling_with_rf_amplitude(self, test_model_and_q):
        """Test that SAR scales quadratically with RF amplitude"""
        model, Q = test_model_and_q
        
        num_coils = Q['Qtmf'].shape[0]
        rf_base = np.ones(num_coils, dtype=complex) * 0.1
        
        # Calculate SAR for base amplitude
        sar_base = calc_SAR(Q['Qtmf'], rf_base, 70.0)
        
        # Scale RF by factor of 2
        rf_scaled = rf_base * 2.0
        sar_scaled = calc_SAR(Q['Qtmf'], rf_scaled, 70.0)
        
        # SAR should scale by factor^2 (quadratic relationship)
        expected_sar = sar_base * 4.0
        assert np.isclose(sar_scaled, expected_sar, rtol=1e-10), \
            f"SAR scaling incorrect: {sar_scaled} vs expected {expected_sar}"
    
    def test_sar_mass_scaling(self, test_model_and_q):
        """Test that SAR scales inversely with mass"""
        model, Q = test_model_and_q
        
        num_coils = Q['Qtmf'].shape[0]
        rf_signal = np.ones(num_coils, dtype=complex) * 0.1
        
        # Calculate SAR for different masses
        mass1 = 70.0
        mass2 = 35.0  # Half the mass
        
        sar1 = calc_SAR(Q['Qtmf'], rf_signal, mass1)
        sar2 = calc_SAR(Q['Qtmf'], rf_signal, mass2)
        
        # SAR should be inversely proportional to mass
        expected_ratio = mass1 / mass2
        actual_ratio = sar2 / sar1
        
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10), \
            f"SAR mass scaling incorrect: ratio {actual_ratio} vs expected {expected_ratio}"
    
    def test_sar_with_complex_rf(self, test_model_and_q):
        """Test SAR calculation with complex RF signals"""
        model, Q = test_model_and_q
        
        num_coils = Q['Qtmf'].shape[0]
        
        # Test with different phase patterns
        phases = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
        amplitudes = [0.1] * num_coils
        
        for phase in phases:
            rf_signal = np.array(amplitudes) * np.exp(1j * phase)
            sar = calc_SAR(Q['Qtmf'], rf_signal, 70.0)
            
            assert np.isreal(sar), f"SAR should be real for phase {phase}"
            assert sar >= 0, f"SAR should be non-negative for phase {phase}"
    
    def test_sar_with_random_rf_patterns(self, test_model_and_q):
        """Test SAR with various random RF excitation patterns"""
        model, Q = test_model_and_q
        
        num_coils = Q['Qtmf'].shape[0]
        
        for trial in range(10):
            # Random amplitude and phase
            amplitudes = np.random.uniform(0, 0.2, num_coils)
            phases = np.random.uniform(0, 2*np.pi, num_coils)
            rf_signal = amplitudes * np.exp(1j * phases)
            
            sar_wb = calc_SAR(Q['Qtmf'], rf_signal, 70.0)
            sar_head = calc_SAR(Q['Qhmf'], rf_signal, 6.0)
            
            assert np.isreal(sar_wb), f"WB SAR should be real (trial {trial})"
            assert np.isreal(sar_head), f"Head SAR should be real (trial {trial})"
            assert sar_wb >= 0, f"WB SAR should be non-negative (trial {trial})"
            assert sar_head >= 0, f"Head SAR should be non-negative (trial {trial})"
    
    def test_sar_zero_rf(self, test_model_and_q):
        """Test SAR calculation with zero RF input"""
        model, Q = test_model_and_q
        
        num_coils = Q['Qtmf'].shape[0]
        rf_signal = np.zeros(num_coils, dtype=complex)
        
        sar_wb = calc_SAR(Q['Qtmf'], rf_signal, 70.0)
        sar_head = calc_SAR(Q['Qhmf'], rf_signal, 6.0)
        
        assert sar_wb == 0, f"SAR should be zero for zero RF: {sar_wb}"
        assert sar_head == 0, f"Head SAR should be zero for zero RF: {sar_head}"
    
    def test_sar_single_coil_vs_multi_coil(self):
        """Compare single coil vs multi-coil SAR calculations"""
        # Create models with different coil counts
        model_1coil = create_dummy_em_model(grid_shape=(16, 16, 8), num_coils=1)
        model_4coil = create_dummy_em_model(grid_shape=(16, 16, 8), num_coils=4)
        
        Q_1coil = Q_mat_gen('Global', model_1coil, qmat_write=False)
        Q_4coil = Q_mat_gen('Global', model_4coil, qmat_write=False)
        
        # Single coil excitation
        rf_1coil = np.array([0.1], dtype=complex)
        sar_1coil = calc_SAR(Q_1coil['Qtmf'], rf_1coil, 70.0)
        
        # Multi-coil with only first coil active
        rf_4coil = np.array([0.1, 0.0, 0.0, 0.0], dtype=complex)
        sar_4coil_first = calc_SAR(Q_4coil['Qtmf'], rf_4coil, 70.0)
        
        # Both should be positive
        assert sar_1coil > 0, "Single coil SAR should be positive"
        assert sar_4coil_first >= 0, "Multi-coil SAR should be non-negative"
    
    def test_sar_conservation_properties(self, test_model_and_q):
        """Test fundamental SAR conservation properties"""
        model, Q = test_model_and_q
        
        num_coils = Q['Qtmf'].shape[0]
        
        # Test superposition: SAR(a+b) related to SAR(a) + SAR(b)
        rf_a = np.random.random(num_coils).astype(complex) * 0.05
        rf_b = np.random.random(num_coils).astype(complex) * 0.05
        
        sar_a = calc_SAR(Q['Qtmf'], rf_a, 70.0)
        sar_b = calc_SAR(Q['Qtmf'], rf_b, 70.0)
        sar_sum = calc_SAR(Q['Qtmf'], rf_a + rf_b, 70.0)
        
        # For positive semi-definite Q, triangle inequality should hold
        # SAR(a+b) <= SAR(a) + SAR(b) + 2*sqrt(SAR(a)*SAR(b))
        max_expected = sar_a + sar_b + 2*np.sqrt(sar_a * sar_b)
        
        assert sar_sum <= max_expected + 1e-10, \
            "SAR superposition violates triangle inequality"
    
    def test_sar_units_and_magnitudes(self, test_model_and_q):
        """Test that SAR values have reasonable magnitudes for typical scenarios"""
        model, Q = test_model_and_q
        
        num_coils = Q['Qtmf'].shape[0]
        
        # Typical MRI RF amplitude (scaled for our dummy model)
        typical_rf = np.ones(num_coils, dtype=complex) * 0.01  # Small amplitude
        
        sar_wb = calc_SAR(Q['Qtmf'], typical_rf, 70.0)
        sar_head = calc_SAR(Q['Qhmf'], typical_rf, 6.0)
        
        # For typical RF, SAR should be in reasonable range
        # (Note: actual values depend on dummy model characteristics)
        assert 0 <= sar_wb <= 10, f"WB SAR seems unreasonable: {sar_wb} W/kg"
        assert 0 <= sar_head <= 10, f"Head SAR seems unreasonable: {sar_head} W/kg"


class TestSARPhysicalConsistency:
    """Test physical consistency of SAR calculations"""
    
    def test_sar_energy_conservation(self):
        """Test that SAR conserves energy principles"""
        model = create_dummy_em_model(grid_shape=(16, 16, 8), num_coils=2)
        Q = Q_mat_gen('Global', model, qmat_write=False)
        
        # Power deposited should equal RF^H * Q * RF
        rf_signal = np.array([0.1 + 0.05j, 0.08 - 0.03j])
        mass = 70.0
        
        sar = calc_SAR(Q['Qtmf'], rf_signal, mass)
        
        # Direct calculation: P = RF^H * Q * RF, SAR = P / mass
        power_direct = np.real(np.conj(rf_signal).T @ Q['Qtmf'] @ rf_signal)
        sar_direct = power_direct / mass
        
        assert np.isclose(sar, sar_direct, rtol=1e-12), \
            f"SAR calculation inconsistent: {sar} vs {sar_direct}"
    
    def test_sar_hermitian_property(self):
        """Test that Q-matrix Hermitian property is preserved in SAR calculation"""
        model = create_dummy_em_model(grid_shape=(16, 16, 8), num_coils=3)
        Q = Q_mat_gen('Global', model, qmat_write=False)
        
        # Q should be Hermitian
        assert np.allclose(Q['Qtmf'], Q['Qtmf'].conj().T), "Q-matrix should be Hermitian"
        
        # For Hermitian Q, RF^H * Q * RF should always be real
        for _ in range(5):
            rf_signal = (np.random.random(3) + 1j * np.random.random(3)) * 0.1
            
            result = np.conj(rf_signal).T @ Q['Qtmf'] @ rf_signal
            assert np.abs(np.imag(result)) < 1e-12, \
                f"Result should be real for Hermitian Q: {result}"


if __name__ == "__main__":
    # Run with: python -m pytest test_sar_calculation.py -v
    pytest.main([__file__, "-v"])
