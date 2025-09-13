import numpy as np
import pypulseq as pp

def analyze_sequence_blocks(seq):
    """
    Analyze Pulseq sequence to extract RF block information
    """
    
    rf_blocks = []
    t_prev = 0
    
    for i_block in range(len(seq.block_events)):
        block = seq.get_block(i_block + 1)
        
        # Calculate block duration
        block_events = []
        if hasattr(block, 'rf') and block.rf is not None:
            block_events.append(block.rf)
        if hasattr(block, 'gx') and block.gx is not None:
            block_events.append(block.gx)
        if hasattr(block, 'gy') and block.gy is not None:
            block_events.append(block.gy)
        if hasattr(block, 'gz') and block.gz is not None:
            block_events.append(block.gz)
        if hasattr(block, 'adc') and block.adc is not None:
            block_events.append(block.adc)
        if hasattr(block, 'delay') and block.delay is not None:
            block_events.append(block.delay)
        
        if block_events:
            block_dur = pp.calc_duration(*block_events)
        else:
            block_dur = 0
            
        # Store RF blocks
        if block.rf is not None:
            rf_blocks.append({
                'block_index': i_block,
                'time': t_prev + block_dur,
                'duration': block_dur,
                'rf_signal': block.rf.signal,
                'rf_amplitude': np.abs(block.rf.signal).max() if hasattr(block.rf.signal, '__len__') else abs(block.rf.signal)
            })
        
        t_prev += block_dur
    
    return rf_blocks

def create_rf_vector(rf_block, n_channels):
    """
    Create RF vector for SAR calculation from RF block data
    """
    
    signal = rf_block['rf_signal']
    
    rf_scale = 1.0
    
    if hasattr(signal, '__len__'):
        if len(signal) == 1:
            rf_vector = np.ones(n_channels, dtype=complex) * signal[0] * rf_scale
        else:
            rf_vector = np.ones(n_channels, dtype=complex) * np.mean(signal) * rf_scale
    else:
        rf_vector = np.ones(n_channels, dtype=complex) * signal * rf_scale
    
    return rf_vector
