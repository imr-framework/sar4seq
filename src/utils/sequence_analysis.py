#!/usr/bin/env python3
"""
Pulse Sequence Analysis Functions

Contains functions for processing and analyzing MRI pulse sequences,
including RF pulse extraction, timing analysis, and sequence validation.

Authors: Leo Kinyera, BS
Copyright: Board of Trustees of Columbia University in the City of New York
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

try:
    import pypulseq as pp
    HAS_PYPULSEQ = True
except ImportError:
    HAS_PYPULSEQ = False
    pp = None

def parse_sequence_file(seq_path: str) -> Dict[str, Any]:
    """
    Parse a pypulseq sequence file and extract relevant parameters.
    
    Args:
        seq_path: Path to .seq file
        
    Returns:
        Dictionary containing sequence information and RF events
    """
    if not HAS_PYPULSEQ:
        raise ImportError("pypulseq is required for sequence parsing. Install with: pip install pypulseq")
    
    try:
        seq = pp.Sequence()
        seq.read(seq_path)
    except Exception as e:
        raise ValueError(f"Failed to read sequence file {seq_path}: {e}")
    
    # Extract sequence blocks
    seq_blocks = []
    block_events = []
    block_durations = []
    block_types = []
    
    for block_idx in range(1, seq.num_blocks + 1):
        block = seq.get_block(block_idx)
        duration = pp.calc_duration(block)
        
        # Initialize block event array
        event_data = np.zeros(16)  # Enough for complex RF data
        block_type = 0  # Default: no RF
        
        # Check for RF events
        if hasattr(block, 'rf') and block.rf is not None:
            block_type = 1  # RF block
            rf = block.rf
            
            # Extract RF parameters
            if hasattr(rf, 'signal') and rf.signal is not None:
                # Complex RF signal (real and imaginary parts)
                rf_signal = rf.signal
                if len(rf_signal) > 0:
                    # Use first sample or average for block representation
                    rf_complex = np.mean(rf_signal) if len(rf_signal) > 1 else rf_signal[0]
                    event_data[0] = np.real(rf_complex)  # Real part
                    event_data[1] = np.imag(rf_complex)  # Imaginary part
            
            # RF timing and frequency
            if hasattr(rf, 'freq_offset'):
                event_data[2] = rf.freq_offset
            if hasattr(rf, 'phase_offset'):
                event_data[3] = rf.phase_offset
            if hasattr(rf, 'flip_angle'):
                event_data[4] = rf.flip_angle
        
        seq_blocks.append(block)
        block_events.append(event_data)
        block_durations.append(duration)
        block_types.append(block_type)
    
    # Convert to numpy arrays
    seq_block_events = np.array(block_events)
    seq_block_durations = np.array(block_durations)
    seq_block_types = np.array(block_types)
    
    # Extract sequence metadata
    sequence_info = {
        'num_blocks': seq.num_blocks,
        'total_duration': np.sum(seq_block_durations),
        'num_rf_blocks': np.sum(seq_block_types == 1),
        'rf_duty_cycle': np.sum(seq_block_durations[seq_block_types == 1]) / np.sum(seq_block_durations),
        'seq_blocks': seq_blocks,
        'seq_block_events': seq_block_events,
        'seq_block_durations': seq_block_durations,
        'seq_block_types': seq_block_types
    }
    
    # Add sequence parameters if available
    if hasattr(seq, 'definitions'):
        sequence_info['definitions'] = seq.definitions
    
    return sequence_info

def analyze_rf_characteristics(seq_block_events: np.ndarray,
                              seq_block_types: np.ndarray,
                              seq_block_durations: np.ndarray) -> Dict[str, Any]:
    """
    Analyze RF pulse characteristics from sequence data.
    
    Args:
        seq_block_events: RF event parameters
        seq_block_types: Block type indicators
        seq_block_durations: Duration of each block
        
    Returns:
        Dictionary containing RF analysis results
    """
    # Find RF blocks
    rf_indices = np.where(seq_block_types == 1)[0]
    
    if len(rf_indices) == 0:
        return {
            'num_rf_pulses': 0,
            'total_rf_time': 0.0,
            'rf_duty_cycle': 0.0,
            'warning': 'No RF pulses found in sequence'
        }
    
    rf_events = seq_block_events[rf_indices]
    rf_durations = seq_block_durations[rf_indices]
    
    # Extract complex RF amplitudes
    rf_real = rf_events[:, 0]
    rf_imag = rf_events[:, 1]
    rf_complex = rf_real + 1j * rf_imag
    rf_magnitude = np.abs(rf_complex)
    rf_phase = np.angle(rf_complex)
    
    # RF statistics
    total_rf_time = np.sum(rf_durations)
    total_sequence_time = np.sum(seq_block_durations)
    rf_duty_cycle = total_rf_time / total_sequence_time if total_sequence_time > 0 else 0
    
    # Power statistics (proportional to |RF|²)
    rf_power = rf_magnitude ** 2
    avg_rf_power = np.mean(rf_power)
    peak_rf_power = np.max(rf_power)
    
    # Temporal analysis
    rf_intervals = np.diff(np.cumsum(seq_block_durations)[rf_indices]) if len(rf_indices) > 1 else []
    avg_rf_interval = np.mean(rf_intervals) if len(rf_intervals) > 0 else 0
    
    analysis = {
        'num_rf_pulses': len(rf_indices),
        'total_rf_time': total_rf_time,
        'total_sequence_time': total_sequence_time,
        'rf_duty_cycle': rf_duty_cycle,
        'rf_statistics': {
            'avg_magnitude': np.mean(rf_magnitude),
            'peak_magnitude': np.max(rf_magnitude),
            'std_magnitude': np.std(rf_magnitude),
            'avg_power': avg_rf_power,
            'peak_power': peak_rf_power,
            'rms_power': np.sqrt(np.mean(rf_power))
        },
        'temporal_statistics': {
            'avg_rf_duration': np.mean(rf_durations),
            'min_rf_duration': np.min(rf_durations),
            'max_rf_duration': np.max(rf_durations),
            'avg_rf_interval': avg_rf_interval
        },
        'phase_statistics': {
            'avg_phase': np.mean(rf_phase),
            'phase_range': np.max(rf_phase) - np.min(rf_phase),
            'phase_std': np.std(rf_phase)
        }
    }
    
    # Add frequency analysis if available
    if rf_events.shape[1] > 2:
        freq_offsets = rf_events[:, 2]
        phase_offsets = rf_events[:, 3]
        
        analysis['frequency_statistics'] = {
            'avg_freq_offset': np.mean(freq_offsets),
            'freq_range': np.max(freq_offsets) - np.min(freq_offsets),
            'avg_phase_offset': np.mean(phase_offsets),
            'phase_offset_range': np.max(phase_offsets) - np.min(phase_offsets)
        }
    
    return analysis

def validate_sequence_for_sar(sequence_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate sequence parameters for SAR calculation compatibility.
    
    Args:
        sequence_info: Sequence information from parse_sequence_file
        
    Returns:
        Dictionary containing validation results and recommendations
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Check for RF blocks
    num_rf = sequence_info.get('num_rf_blocks', 0)
    if num_rf == 0:
        validation['errors'].append("No RF blocks found - SAR calculation not possible")
        validation['is_valid'] = False
    elif num_rf < 10:
        validation['warnings'].append(f"Very few RF blocks ({num_rf}) - results may be limited")
    
    # Check duty cycle
    duty_cycle = sequence_info.get('rf_duty_cycle', 0)
    if duty_cycle > 0.5:
        validation['warnings'].append(f"High RF duty cycle ({duty_cycle:.1%}) - verify sequence parameters")
    elif duty_cycle < 0.01:
        validation['warnings'].append(f"Very low RF duty cycle ({duty_cycle:.1%}) - check for missing RF events")
    
    # Check sequence duration
    total_duration = sequence_info.get('total_duration', 0)
    if total_duration > 1800:  # 30 minutes
        validation['warnings'].append(f"Long sequence duration ({total_duration/60:.1f} min) - consider computational time")
    elif total_duration < 1:  # 1 second
        validation['warnings'].append(f"Very short sequence ({total_duration:.1f} s) - verify completeness")
    
    # Check for temporal resolution
    durations = sequence_info.get('seq_block_durations', np.array([]))
    if len(durations) > 0:
        min_duration = np.min(durations[durations > 0])
        if min_duration < 1e-6:  # Less than 1 microsecond
            validation['warnings'].append("Very fine temporal resolution - may impact computation time")
    
    # Generate recommendations
    if validation['is_valid']:
        if len(validation['warnings']) == 0:
            validation['recommendations'].append("✓ Sequence appears suitable for SAR calculation")
        else:
            validation['recommendations'].append("⚠️ Sequence can be processed but review warnings")
    else:
        validation['recommendations'].append("✗ Sequence cannot be processed - fix errors first")
    
    # Add computational recommendations
    if num_rf > 10000:
        validation['recommendations'].append("• Consider GPU acceleration for large sequences")
    
    if duty_cycle > 0.2:
        validation['recommendations'].append("• Use VOP compression for high duty cycle sequences")
    
    return validation

def extract_multichannel_rf(seq_block_events: np.ndarray,
                           seq_block_types: np.ndarray,
                           n_channels: int = 8) -> np.ndarray:
    """
    Extract multi-channel RF data from sequence events.
    
    Args:
        seq_block_events: Event data array
        seq_block_types: Block types
        n_channels: Number of RF channels
        
    Returns:
        Multi-channel RF array [time_blocks, channels*2] (real and imaginary parts)
    """
    n_blocks = len(seq_block_types)
    multichannel_rf = np.zeros((n_blocks, n_channels * 2))
    
    # Find RF blocks
    rf_indices = np.where(seq_block_types == 1)[0]
    
    for rf_idx in rf_indices:
        # Extract available RF data
        rf_data = seq_block_events[rf_idx]
        
        # Map to channels (assuming simple replication for demo)
        for ch in range(min(n_channels, len(rf_data)//2)):
            if ch * 2 + 1 < len(rf_data):
                multichannel_rf[rf_idx, ch] = rf_data[ch * 2]      # Real part
                multichannel_rf[rf_idx, ch + n_channels] = rf_data[ch * 2 + 1]  # Imaginary part
    
    return multichannel_rf

def calculate_sequence_timing(seq_block_durations: np.ndarray,
                             dt: float = 1e-6) -> Dict[str, Any]:
    """
    Calculate detailed timing information for the sequence.
    
    Args:
        seq_block_durations: Duration of each block
        dt: Desired time step for interpolation
        
    Returns:
        Dictionary containing timing analysis
    """
    cumulative_time = np.cumsum(seq_block_durations)
    total_time = cumulative_time[-1]
    
    # Create high-resolution time vector if needed
    if dt < np.min(seq_block_durations[seq_block_durations > 0]) / 10:
        n_time_points = int(total_time / dt)
        time_vector = np.linspace(0, total_time, n_time_points)
        
        # Map blocks to high-resolution time
        block_mapping = np.searchsorted(cumulative_time, time_vector, side='right')
        
        timing_info = {
            'total_time': total_time,
            'time_step': dt,
            'n_time_points': n_time_points,
            'time_vector': time_vector,
            'block_mapping': block_mapping,
            'high_resolution': True
        }
    else:
        timing_info = {
            'total_time': total_time,
            'cumulative_time': cumulative_time,
            'block_durations': seq_block_durations,
            'n_blocks': len(seq_block_durations),
            'high_resolution': False
        }
    
    return timing_info

__all__ = [
    'parse_sequence_file',
    'analyze_rf_characteristics',
    'validate_sequence_for_sar',
    'extract_multichannel_rf',
    'calculate_sequence_timing'
]
