"""
Test sequence generation for SAR4seq

This module provides functions to generate simple Pulseq sequences for testing
SAR calculation functionality when no sequence file is available.
"""

import numpy as np
import pypulseq as pp


def create_test_sequence(system=None):
    """
    Create a simple test TSE sequence for demonstration
    
    Parameters
    ----------
    system : pypulseq.Opts, optional
        System limits. If None, default system parameters will be used.
        
    Returns
    -------
    pypulseq.Sequence
        Simple TSE sequence for testing
    """
    if system is None:
        # Create default system parameters
        system = pp.Opts(
            max_grad=32, grad_unit='mT/m',
            max_slew=130, slew_unit='T/m/s',
            rf_ringdown_time=30e-6,
            rf_dead_time=100e-6
        )
    
    seq = pp.Sequence(system)
    
    # Create simple RF pulses
    # Simple rectangular pulse for testing (low amplitude for safety demo)
    rf_duration = 1e-3  # 1 ms
    
    rf_simple = pp.make_block_pulse(
        flip_angle=10 * np.pi / 180,  # Small flip angle for low SAR
        duration=rf_duration,
        system=system
    )
    
    # Create delays to make a simple sequence
    delay1 = pp.make_delay(10e-3)  # 10 ms
    delay2 = pp.make_delay(20e-3)  # 20 ms
    
    # Add blocks to sequence
    seq.add_block(rf_simple)
    seq.add_block(delay1)
    seq.add_block(rf_simple)  # Second RF pulse
    seq.add_block(delay2)
    
    print("Created simple test sequence with rectangular RF pulses")
    return seq


def create_advanced_test_sequence(system=None):
    """
    Create a more complex test sequence with multiple RF pulses
    
    Parameters
    ----------
    system : pypulseq.Opts, optional
        System limits. If None, default system parameters will be used.
        
    Returns
    -------
    pypulseq.Sequence
        More complex test sequence
    """
    if system is None:
        # Create default system parameters
        system = pp.Opts(
            max_grad=32, grad_unit='mT/m',
            max_slew=130, slew_unit='T/m/s',
            rf_ringdown_time=30e-6,
            rf_dead_time=100e-6
        )
    
    seq = pp.Sequence(system)
    
    # Create different types of RF pulses
    rf_90 = pp.make_block_pulse(
        flip_angle=90 * np.pi / 180,
        duration=1e-3,
        system=system
    )
    
    rf_180 = pp.make_block_pulse(
        flip_angle=180 * np.pi / 180,
        duration=2e-3,
        system=system
    )
    
    # Create delays
    te_delay = pp.make_delay(10e-3)  # TE delay
    tr_delay = pp.make_delay(100e-3)  # TR delay
    
    # Build a simple spin echo sequence
    seq.add_block(rf_90)
    seq.add_block(te_delay)
    seq.add_block(rf_180)
    seq.add_block(te_delay)
    seq.add_block(tr_delay)
    
    print("Created advanced test sequence with 90° and 180° pulses")
    return seq


if __name__ == "__main__":
    # Example usage
    print("Creating test sequences...")
    
    # Create simple test sequence
    simple_seq = create_test_sequence()
    print(f"Simple sequence has {len(simple_seq.block_events)} blocks")
    
    # Create advanced test sequence
    advanced_seq = create_advanced_test_sequence()
    print(f"Advanced sequence has {len(advanced_seq.block_events)} blocks")
    
    # Optionally save sequences
    simple_seq.write("simple_test_sequence.seq")
    advanced_seq.write("advanced_test_sequence.seq")
    print("Test sequences saved to files")
