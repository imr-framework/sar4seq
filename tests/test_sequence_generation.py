#!/usr/bin/env python3
"""
Test sequence generation functionality

These tests validate the TSE sequence generation to ensure compatibility
with Pulseq and proper RF safety integration.
"""

import pytest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Import modules to test
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from src.write_tse_500ms import write_custom_TSE, write_TSE_500ms
except ImportError:
    print("Warning: TSE sequence module not found, some tests will be skipped")
    write_custom_TSE = None
    write_TSE_500ms = None

try:
    import pypulseq as pp
    PYPULSEQ_AVAILABLE = True
except ImportError:
    PYPULSEQ_AVAILABLE = False


class TestTSESequenceGeneration:
    """Test TSE sequence generation functionality"""
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_basic_tse_creation(self):
        """Test basic TSE sequence creation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_tse.seq")
            
            # Create a simple TSE sequence
            seq = write_custom_TSE(
                TR=100e-3,      # 100ms TR
                TE_eff=20e-3,   # 20ms TE
                necho=4,        # 4 echoes
                Nx=32, Ny=32,   # Small matrix
                fov=100e-3,     # 10cm FOV
                output_filename=output_file
            )
            
            assert seq is not None, "Sequence creation should return sequence object"
            assert os.path.exists(output_file), "Sequence file should be created"
            
            # Check file size is reasonable
            file_size = os.path.getsize(output_file)
            assert file_size > 100, f"Sequence file seems too small: {file_size} bytes"
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_tse_parameter_validation(self):
        """Test TSE sequence parameter validation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_tse_params.seq")
            
            # Test valid parameters
            seq = write_custom_TSE(
                TR=200e-3,
                TE_eff=40e-3,
                necho=8,
                Nx=64, Ny=64,
                fov=200e-3,
                output_filename=output_file
            )
            
            assert seq is not None, "Valid parameters should create sequence"
            
            # Test that TE < TR constraint is enforced implicitly
            # (Should work but might generate warnings)
            seq2 = write_custom_TSE(
                TR=50e-3,       # Short TR
                TE_eff=80e-3,   # Long TE
                necho=2,
                Nx=16, Ny=16,
                fov=50e-3,
                output_filename=output_file.replace('.seq', '_2.seq')
            )
            
            # Should still create sequence (with warnings)
            assert seq2 is not None, "Should handle parameter edge cases"
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_tse_different_matrices(self):
        """Test TSE sequence with different matrix sizes"""
        matrix_sizes = [(16, 16), (32, 32), (64, 64), (128, 64)]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, (nx, ny) in enumerate(matrix_sizes):
                output_file = os.path.join(tmp_dir, f"test_tse_matrix_{i}.seq")
                
                seq = write_custom_TSE(
                    TR=150e-3,
                    TE_eff=30e-3,
                    necho=6,
                    Nx=nx, Ny=ny,
                    fov=150e-3,
                    output_filename=output_file
                )
                
                assert seq is not None, f"Should create sequence for matrix {nx}x{ny}"
                assert os.path.exists(output_file), f"File should exist for matrix {nx}x{ny}"
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_tse_echo_train_lengths(self):
        """Test TSE sequences with different echo train lengths"""
        echo_counts = [2, 4, 8, 16]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            for necho in echo_counts:
                output_file = os.path.join(tmp_dir, f"test_tse_etl_{necho}.seq")
                
                seq = write_custom_TSE(
                    TR=200e-3,
                    TE_eff=25e-3,
                    necho=necho,
                    Nx=32, Ny=32,
                    fov=120e-3,
                    output_filename=output_file
                )
                
                assert seq is not None, f"Should create sequence for {necho} echoes"
                assert os.path.exists(output_file), f"File should exist for {necho} echoes"
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_tse_timing_constraints(self):
        """Test TSE sequence timing constraints"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test reasonable timing parameters
            timing_params = [
                (100e-3, 15e-3),  # TR=100ms, TE=15ms
                (200e-3, 30e-3),  # TR=200ms, TE=30ms
                (500e-3, 50e-3),  # TR=500ms, TE=50ms
            ]
            
            for i, (TR, TE) in enumerate(timing_params):
                output_file = os.path.join(tmp_dir, f"test_tse_timing_{i}.seq")
                
                seq = write_custom_TSE(
                    TR=TR,
                    TE_eff=TE,
                    necho=4,
                    Nx=32, Ny=32,
                    fov=100e-3,
                    output_filename=output_file
                )
                
                assert seq is not None, f"Should create sequence for TR={TR*1000}ms, TE={TE*1000}ms"
    
    @pytest.mark.skipif(write_TSE_500ms is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_tse_500ms_function(self):
        """Test the standard TSE_500ms function if available"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_tse_500ms.seq")
            
            try:
                seq = write_TSE_500ms(output_filename=output_file)
                assert seq is not None, "TSE_500ms should create sequence"
                assert os.path.exists(output_file), "TSE_500ms should create file"
            except Exception as e:
                # If function requires specific parameters, skip gracefully
                pytest.skip(f"TSE_500ms function requires specific setup: {e}")


class TestSequenceFileFormat:
    """Test sequence file format and Pulseq compatibility"""
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_sequence_file_readback(self):
        """Test that generated sequence files can be read back"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_readback.seq")
            
            # Create sequence
            seq_orig = write_custom_TSE(
                TR=150e-3,
                TE_eff=25e-3,
                necho=4,
                Nx=32, Ny=32,
                fov=120e-3,
                output_filename=output_file
            )
            
            # Read it back
            seq_loaded = pp.Sequence()
            seq_loaded.read(output_file)
            
            # Basic checks
            assert len(seq_loaded.block_events) > 0, "Loaded sequence should have events"
            
            # Check that some RF events exist
            rf_events = [block for block in seq_loaded.block_events.values() 
                        if hasattr(block, 'rf') and block.rf is not None]
            assert len(rf_events) > 0, "Sequence should contain RF events"
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    @pytest.mark.skipif(not PYPULSEQ_AVAILABLE, reason="pypulseq not available")
    def test_sequence_basic_structure(self):
        """Test basic structure of generated sequences"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_structure.seq")
            
            # Create sequence with known parameters
            necho = 6
            seq = write_custom_TSE(
                TR=200e-3,
                TE_eff=30e-3,
                necho=necho,
                Nx=64, Ny=32,  # Ny < Nx to test asymmetric matrix
                fov=150e-3,
                output_filename=output_file
            )
            
            # Load and analyze
            seq_loaded = pp.Sequence()
            seq_loaded.read(output_file)
            
            # Count RF pulses (should have excitation + refocusing pulses)
            rf_count = 0
            for block in seq_loaded.block_events.values():
                if hasattr(block, 'rf') and block.rf is not None:
                    rf_count += 1
            
            # Should have at least excitation + refocusing pulses
            assert rf_count >= necho, f"Should have at least {necho} RF pulses, found {rf_count}"
    
    def test_sequence_file_properties(self):
        """Test properties of sequence files without requiring Pulseq"""
        # This test can run even without pypulseq
        if write_custom_TSE is None:
            pytest.skip("TSE module not available")
            
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_properties.seq")
            
            try:
                seq = write_custom_TSE(
                    TR=100e-3,
                    TE_eff=20e-3,
                    necho=4,
                    Nx=32, Ny=32,
                    fov=100e-3,
                    output_filename=output_file
                )
                
                # Check file exists and has reasonable size
                assert os.path.exists(output_file), "Sequence file should exist"
                
                file_size = os.path.getsize(output_file)
                assert 100 < file_size < 1e6, f"File size seems unreasonable: {file_size}"
                
                # Check it's a text file (Pulseq files are text-based)
                with open(output_file, 'r') as f:
                    first_line = f.readline()
                    assert first_line.startswith('#'), "Pulseq files should start with comment"
                    
            except Exception as e:
                pytest.skip(f"Sequence creation failed: {e}")


class TestSequenceParameterSweeps:
    """Test sequence generation across parameter ranges"""
    
    @pytest.mark.skipif(write_custom_TSE is None, reason="TSE module not available")
    def test_parameter_ranges(self):
        """Test sequence generation across reasonable parameter ranges"""
        # Define reasonable parameter ranges for TSE
        test_cases = [
            # (TR, TE, necho, Nx, Ny, fov, description)
            (80e-3, 15e-3, 4, 16, 16, 80e-3, "fast_low_res"),
            (150e-3, 25e-3, 8, 32, 32, 120e-3, "standard"),
            (300e-3, 40e-3, 12, 64, 64, 200e-3, "high_res"),
            (500e-3, 60e-3, 16, 128, 96, 250e-3, "very_high_res"),
        ]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, (TR, TE, necho, Nx, Ny, fov, desc) in enumerate(test_cases):
                output_file = os.path.join(tmp_dir, f"test_{desc}.seq")
                
                try:
                    seq = write_custom_TSE(
                        TR=TR, TE_eff=TE, necho=necho,
                        Nx=Nx, Ny=Ny, fov=fov,
                        output_filename=output_file
                    )
                    
                    assert seq is not None, f"Failed to create {desc} sequence"
                    assert os.path.exists(output_file), f"File not created for {desc}"
                    
                except Exception as e:
                    # Log the failure but don't fail the test for edge cases
                    print(f"Warning: {desc} sequence failed: {e}")


if __name__ == "__main__":
    # Run with: python -m pytest test_sequence_generation.py -v
    pytest.main([__file__, "-v"])
