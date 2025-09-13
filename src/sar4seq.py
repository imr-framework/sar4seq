import numpy as np
import pypulseq as pp
import scipy.io as sio
import time
from utils.calc_sar import (calc_SAR_full_resolution_gpu, 
                           calc_SAR_batch_gpu,
                           calc_SAR_batch_cpu)
from utils.rf_sequence import analyze_sequence_blocks, create_rf_vector


def sar4seq(seq_path: str, Q_mat_path: str, requires_gpu: bool = False, n_spatial: int = 50000):

    seq = pp.Sequence()
    seq.read(seq_path)

    rf_blocks = analyze_sequence_blocks(seq)
    print(f"Found {len(rf_blocks)} RF blocks")

    n_channels = 8
    sar_results = []

    Q_data = sio.loadmat(Q_mat_path)
    Q_struct = Q_data['Q']
    val = Q_struct[0, 0]
    QGlobal = val['Qtmf'].astype(np.complex128)

    print(f"Q-matrix shape: {QGlobal.shape}")
    print(f"Q-matrix dtype: {QGlobal.dtype}")

    # Create full spatial resolution Q-matrix
    print(f"Creating spatial Q-matrix with {n_spatial:,} points...")

    Q_full = np.tile(QGlobal[np.newaxis, :, :], (n_spatial, 1, 1))
    memory_usage = Q_full.nbytes / (1024**2)
    print(f"Q_full shape: {Q_full.shape}")
    print(f"Memory usage: {memory_usage:.1f} MB")

    # Calculate SAR using TRUE parallel batch processing for ALL RF blocks
    print(f"\nCalculating SAR for all {len(rf_blocks)} RF blocks using BATCH processing...")
    print("This will be much faster than sequential processing...")

    start_time = time.time()

    # Prepare all RF vectors first
    print("Preparing RF vectors...")
    rf_vectors = []
    rf_metadata = []

    for idx, rf_block in enumerate(rf_blocks):
        rf_vector = create_rf_vector(rf_block, n_channels)
        rf_vectors.append(rf_vector)
        rf_metadata.append({
            'block_index': idx,
            'time': rf_block.get('time', 0),
            'rf_amplitude': np.max(np.abs(rf_vector))
        })

    # Convert to batch format for true parallel processing
    rf_batch = np.array(rf_vectors)
    print(f"RF batch shape: {rf_batch.shape}")

    # TRUE BATCH PROCESSING: Process ALL RF vectors simultaneously
    processing_success = False
    
    if requires_gpu:
        print("Processing ALL RF blocks simultaneously using batch GPU processing...")
        try:
            # Use the batch GPU function for maximum parallelization
            sar_local_batch, sar_peaks_batch = calc_SAR_batch_gpu(Q_full, rf_batch)
            
            # Extract results for compatibility with existing analysis code
            for idx in range(len(rf_blocks)):
                sar_results.append({
                    'block_index': rf_metadata[idx]['block_index'],
                    'time': rf_metadata[idx]['time'],
                    'sar_local': sar_local_batch[idx],
                    'sar_peak': sar_peaks_batch[idx],
                    'rf_amplitude': rf_metadata[idx]['rf_amplitude']
                })
            
            print(f"✅ GPU batch processing completed successfully!")
            processing_success = True
            
        except Exception as e:
            print(f"❌ Batch GPU processing failed: {e}")
            print("Falling back to batch CPU processing...")
    
    # Try CPU batch processing if GPU wasn't requested or failed
    if not processing_success:
        try:
            # Try batch CPU processing
            sar_local_batch, sar_peaks_batch = calc_SAR_batch_cpu(Q_full, rf_batch)
                
            # Extract results
            for idx in range(len(rf_blocks)):
                sar_results.append({
                    'block_index': rf_metadata[idx]['block_index'],
                    'time': rf_metadata[idx]['time'],
                    'sar_local': sar_local_batch[idx],
                    'sar_peak': sar_peaks_batch[idx],
                    'rf_amplitude': rf_metadata[idx]['rf_amplitude']
                })
                    
            print(f"✅ CPU batch processing completed successfully!")
            processing_success = True
                
        except Exception as e2:
            print(f"❌ Batch CPU processing also failed: {e2}")
            print("Falling back to sequential processing...")
    # Final fallback to sequential processing only if batch processing failed
    if not processing_success:
        print("Using sequential processing as final fallback...")
        for idx, rf_block in enumerate(rf_blocks):
            if idx % 20 == 0:
                print(f"Processing RF block {idx+1}/{len(rf_blocks)}")
                
            rf_vector = create_rf_vector(rf_block, n_channels)
                
            try:
                sar_local, sar_peak = calc_SAR_full_resolution_gpu(Q_full, rf_vector)
                sar_results.append({
                    'block_index': idx,
                    'time': rf_block.get('time', 0),
                    'sar_local': sar_local,
                    'sar_peak': sar_peak,
                    'rf_amplitude': np.max(np.abs(rf_vector))
                })
            except Exception as e3:
                print(f"SAR calculation failed for block {idx}: {e3}")
                continue

    total_time = time.time() - start_time
    print(f"\nAll RF blocks processed in {total_time:.1f} seconds")

    # Analyze results across ALL RF blocks
    if sar_results:
        all_peaks = [result['sar_peak'] for result in sar_results]
        all_rf_amps = [result['rf_amplitude'] for result in sar_results]
        
        print(f"\nCOMPLETE SAR ANALYSIS RESULTS:")
        print(f"  Total RF blocks processed: {len(sar_results)}")
        print(f"  Peak SAR (max across all blocks): {np.max(all_peaks):.6f} W/kg")
        print(f"  Mean SAR (across all blocks): {np.mean(all_peaks):.6f} W/kg")
        print(f"  SAR standard deviation: {np.std(all_peaks):.6f} W/kg")
        print(f"  Min SAR: {np.min(all_peaks):.6f} W/kg")
        print(f"  Max SAR: {np.max(all_peaks):.6f} W/kg")
        
        print(f"\nRF Amplitude Analysis:")
        print(f"  Max RF amplitude: {np.max(all_rf_amps):.3f} A")
        print(f"  Min RF amplitude: {np.min(all_rf_amps):.3f} A")
        print(f"  Mean RF amplitude: {np.mean(all_rf_amps):.3f} A")
        
        # Time-averaged SAR calculation
        if sar_results:
            total_duration = sar_results[-1]['time']
            total_sar_energy = sum(result['sar_peak'] * 0.003 for result in sar_results)  # Assuming 3ms per block
            time_averaged_sar = total_sar_energy / total_duration if total_duration > 0 else 0
            
            print(f"\nTime-Averaged SAR:")
            print(f"  Total sequence duration: {total_duration:.3f} s")
            print(f"  Time-averaged SAR: {time_averaged_sar:.6f} W/kg")
            
            # Safety assessment
            safety_limit = 4.0  # W/kg whole body
            max_sar = np.max(all_peaks)
            if max_sar > safety_limit:
                print(f"\nSAFETY WARNING: Peak SAR exceeds limit!")
                print(f"    Peak SAR: {max_sar:.3f} W/kg > {safety_limit} W/kg")
            else:
                print(f"\nSAFETY: All SAR values within limits")
                print(f"    Peak SAR: {max_sar:.3f} W/kg ≤ {safety_limit} W/kg")
        
    else:
        print("No SAR results to display")
        
    return {
        'sar_results': sar_results,
        'analysis': {
            'peak_sar': np.max(all_peaks) if sar_results else 0,
            'mean_sar': np.mean(all_peaks) if sar_results else 0,
            'std_sar': np.std(all_peaks) if sar_results else 0,
            'min_sar': np.min(all_peaks) if sar_results else 0,
            'max_sar': np.max(all_peaks) if sar_results else 0,
            'total_duration': sar_results[-1]['time'] if sar_results else 0,
            'time_averaged_sar': time_averaged_sar if sar_results else 0,
            'safety_compliant': np.max(all_peaks) <= 4.0 if sar_results else True
        },
        'processing_time': total_time,
        'n_blocks_processed': len(sar_results)
    }

if __name__ == '__main__':
    seq_path = '/home/maxi/Desktop/sar4seq/tse_50s.seq'
    Q_mat_path = '/home/maxi/Desktop/sar4seq/data/QGlobal.mat'

    results = sar4seq(seq_path, Q_mat_path, requires_gpu=False, n_spatial=100000)