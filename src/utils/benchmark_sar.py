import numpy as np
from utils.gen_qpwr import create_full_resolution_qmatrix
from utils.calc_sar import (
    calc_SAR_full_resolution_cpu, 
    calc_SAR_full_resolution_gpu,
    calc_SAR_vop_compressed,
)
from utils.gen_seq_test import create_test_sequence
# GPU acceleration support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def benchmark_sar_methods(n_spatial_points=10000, n_vop_points=100, n_channels=8):
    """
    Benchmark different SAR calculation methods
    
    Parameters:
    -----------
    n_spatial_points : int
        Number of spatial points for full resolution
    n_vop_points : int
        Number of VOP points for compressed method
    n_channels : int
        Number of RF channels
    """
    
    print("=" * 70)
    print("SAR COMPUTATION BENCHMARK")
    print("=" * 70)
    
    # Create test RF signal
    rf_amplitude = 0.01  # Tesla
    rf_vector = np.ones(n_channels, dtype=complex) * rf_amplitude
    mass = 70.0  # kg
    
    print(f"\nTest parameters:")
    print(f"  RF amplitude: {rf_amplitude} T")
    print(f"  Number of channels: {n_channels}")
    print(f"  Body mass: {mass} kg")
    
    # 1. Create full-resolution Q-matrix
    print(f"\n" + "=" * 50)
    print("1. FULL RESOLUTION SAR (NO COMPRESSION)")
    print("=" * 50)
    
    Q_full = create_full_resolution_qmatrix(n_spatial_points, n_channels, use_tissue_data=True)
    
    # Progress callback for CPU calculation
    def progress_update(progress, current, total):
        if current % 2000 == 0:  # Update every 2000 iterations
            print(f"    Progress: {progress:.1f}% ({current:,}/{total:,})")
    
    # CPU calculation
    sar_local_cpu, sar_peak_cpu = calc_SAR_full_resolution_cpu(
        Q_full, rf_vector, mass, progress_update
    )
    
    # GPU calculation (if available)
    if CUPY_AVAILABLE:
        try:
            print(f"  Attempting GPU computation (forcing CUDA initialization)...")
            
            # Try to force GPU availability check
            try:
                # Check if we can create a simple array on GPU
                test_array = cp.array([1, 2, 3])
                cp.cuda.Device(0).synchronize()
                print(f"    ✅ GPU test successful, proceeding with SAR calculation")
                
                sar_local_gpu, sar_peak_gpu = calc_SAR_full_resolution_gpu(
                    Q_full, rf_vector, mass
                )
                
                # Verify GPU vs CPU results
                difference = abs(sar_peak_gpu - sar_peak_cpu)
                print(f"  GPU vs CPU difference: {difference:.2e} W/kg")
                
            except Exception as gpu_init_error:
                print(f"    ⚠️  GPU initialization failed: {gpu_init_error}")
                print(f"    Trying alternative GPU approach...")
                
                # Alternative: try with explicit device selection
                try:
                    cp.cuda.Device(0).use()
                    sar_local_gpu, sar_peak_gpu = calc_SAR_full_resolution_gpu(
                        Q_full, rf_vector, mass
                    )
                    difference = abs(sar_peak_gpu - sar_peak_cpu)
                    print(f"  GPU vs CPU difference: {difference:.2e} W/kg")
                except Exception as alt_error:
                    print(f"    ❌ Alternative GPU approach failed: {alt_error}")
                    sar_peak_gpu = None
            
        except Exception as e:
            print(f"  GPU calculation failed: {e}")
            sar_peak_gpu = None
    else:
        print(f"  GPU calculation skipped (CuPy not available)")
        sar_peak_gpu = None
    
    # 2. VOP compressed comparison
    print(f"\n" + "=" * 50)
    print("2. VOP COMPRESSED SAR (REFERENCE)")
    print("=" * 50)
    
    # Use proper VOP compression from vop_qmatrices_v3.py
    sar_peak_vop, vop_results = calc_SAR_vop_compressed(Q_full, rf_vector, mass, n_vop_points)
    
    if vop_results is not None:
        print(f"VOP algorithm successfully generated {vop_results['num_vops']} VOPs")
        print(f"Original points: {vop_results['original_points']:,}")
        print(f"Compression ratio: {vop_results['original_points']/vop_results['num_vops']:.1f}:1")
        if vop_results.get('gpu_accelerated', False):
            print(f"VOP calculation used GPU acceleration")
    
    # 3. Performance comparison
    print(f"\n" + "=" * 50)
    print("3. PERFORMANCE COMPARISON")
    print("=" * 50)
    
    print(f"{'Method':<25} {'SAR (W/kg)':<15} {'Ratio':<10} {'Notes'}")
    print("-" * 70)
    
    # Full resolution CPU
    ratio_cpu = sar_peak_cpu / sar_peak_vop if sar_peak_vop > 0 else 1.0
    print(f"{'Full Resolution (CPU)':<25} {sar_peak_cpu:<15.6f} {ratio_cpu:<10.3f} Most accurate")
    
    # Full resolution GPU
    if sar_peak_gpu is not None:
        ratio_gpu = sar_peak_gpu / sar_peak_vop if sar_peak_vop > 0 else 1.0
        print(f"{'Full Resolution (GPU)':<25} {sar_peak_gpu:<15.6f} {ratio_gpu:<10.3f} Fastest + accurate")
    
    # VOP compressed
    print(f"{'VOP Compressed':<25} {sar_peak_vop:<15.6f} {'1.000':<10} Reference")
    
    # 4. Memory usage comparison
    print(f"\n" + "=" * 50)
    print("4. MEMORY USAGE COMPARISON")
    print("=" * 50)
    
    full_memory = Q_full.nbytes / (1024 * 1024)
    
    if vop_results is not None:
        # Calculate VOP memory usage from actual VOP matrices
        vop_memory = vop_results['VOP_matrices'].nbytes / (1024 * 1024)
        memory_reduction = full_memory / vop_memory
        print(f"Full resolution Q-matrix: {full_memory:.1f} MB")
        print(f"VOP compressed Q-matrix:  {vop_memory:.1f} MB")
        print(f"Memory reduction factor:  {memory_reduction:.1f}x")
    else:
        # Estimate memory reduction from compression ratio
        estimated_vop_memory = full_memory / (n_spatial_points / n_vop_points)
        print(f"Full resolution Q-matrix: {full_memory:.1f} MB")
        print(f"VOP compressed Q-matrix:  {estimated_vop_memory:.1f} MB (estimated)")
        print(f"Memory reduction factor:  {n_spatial_points/n_vop_points:.1f}x")
    
    # 5. Recommendations
    print(f"\n" + "=" * 50)
    print("5. RECOMMENDATIONS")
    print("=" * 50)
    
    print(f"For routine clinical use:")
    print(f"  → Use VOP compression ({n_vop_points} points)")
    print(f"  → Fast computation with conservative SAR estimates")
    
    print(f"\nFor research/optimization:")
    if CUPY_AVAILABLE:
        print(f"  → Use GPU full resolution ({n_spatial_points:,} points)")
        print(f"  → Most accurate SAR with reasonable speed")
    else:
        print(f"  → Consider GPU implementation for {n_spatial_points:,} points")
        print(f"  → Current CPU implementation may be too slow")
    
    print(f"\nFor sequence optimization:")
    print(f"  → Start with VOP for initial screening")
    print(f"  → Use full resolution for final validation")


def test_with_pulseq_sequence():
    """Test uncompressed SAR calculation with a real Pulseq sequence"""
    
    print(f"\n" + "=" * 70)
    print("TESTING WITH PULSEQ SEQUENCE")
    print("=" * 70)
    
    # Create or load a test sequence
    try:
        seq = create_test_sequence()
        print(f"Created test sequence with {len(seq.block_events)} blocks")
    except Exception as e:
        print(f"Error creating test sequence: {e}")
        return
    
    # Create smaller Q-matrix for this test
    n_spatial = 1000  # Smaller for faster testing
    n_channels = 8
    Q_full = create_full_resolution_qmatrix(n_spatial, n_channels, base_coupling=1e-6, use_tissue_data=True)
    
    # Process sequence blocks
    mass = 70.0
    total_sar_full = 0
    total_sar_vop = 0
    
    print(f"\nProcessing {len(seq.block_events)} sequence blocks...")
    
    for i_block in range(min(5, len(seq.block_events))):  # Test first 5 blocks
        block = seq.get_block(i_block + 1)
        
        if hasattr(block, 'rf') and block.rf is not None:
            rf = block.rf
            signal = rf.signal
            
            # Create RF vector
            if hasattr(signal, '__len__'):
                rf_amplitude = np.mean(np.abs(signal)) * 0.01
            else:
                rf_amplitude = abs(signal) * 0.01
            
            rf_vector = np.ones(n_channels, dtype=complex) * rf_amplitude
            
            print(f"\n  Block {i_block + 1}: RF amplitude = {rf_amplitude:.6f}")
            
            # Full resolution calculation
            _, sar_peak_full = calc_SAR_full_resolution_cpu(Q_full, rf_vector, mass)
            total_sar_full += sar_peak_full
            
            # VOP compressed calculation (use proper VOP algorithm)
            sar_peak_vop, _ = calc_SAR_vop_compressed(Q_full, rf_vector, mass, n_vop_points=100)
            total_sar_vop += sar_peak_vop
            
            print(f"    Full resolution SAR: {sar_peak_full:.6f} W/kg")
            print(f"    VOP compressed SAR:  {sar_peak_vop:.6f} W/kg")
            print(f"    Difference: {abs(sar_peak_full - sar_peak_vop):.6f} W/kg")
    
    print(f"\nSequence totals:")
    print(f"  Full resolution total: {total_sar_full:.6f} W/kg")
    print(f"  VOP compressed total:  {total_sar_vop:.6f} W/kg")


def test_large_scale_sar(n_spatial_points=100000, n_vop_points=500):
    """
    Test SAR computation with large-scale spatial resolution
    
    Parameters:
    -----------
    n_spatial_points : int
        Number of spatial points (default: 100,000 for realistic body model)
    n_vop_points : int
        Number of VOP points for comparison
    """
    
    print("=" * 80)
    print(f"🚀 LARGE-SCALE SAR COMPUTATION TEST: {n_spatial_points:,} VOXELS")
    print("=" * 80)
    
    # Estimate memory requirements
    n_channels = 8
    total_elements = n_spatial_points * n_channels * n_channels
    memory_gb = (total_elements * 16) / (1024**3)  # 16 bytes per complex128
    
    print(f"\nMemory requirements:")
    print(f"  Spatial points: {n_spatial_points:,}")
    print(f"  RF channels: {n_channels}")
    print(f"  Total Q-matrix elements: {total_elements:,}")
    print(f"  Estimated memory: {memory_gb:.2f} GB")
    
    if memory_gb > 4.0:
        print(f"  ⚠️  Large memory requirement! Consider reducing spatial points.")
        response = input(f"Continue with {n_spatial_points:,} points? (y/N): ")
        if response.lower() != 'y':
            print("Test cancelled.")
            return
    
    try:
        benchmark_sar_methods(
            n_spatial_points=n_spatial_points,
            n_vop_points=n_vop_points,
            n_channels=n_channels
        )
        
        # Performance analysis
        print(f"\n" + "=" * 50)
        print("LARGE-SCALE PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        estimated_time_per_pulse = 0.6  # seconds (rough estimate from scaling)
        pulses_per_sequence = 100  # typical MRI sequence
        
        print(f"Estimated performance for {n_spatial_points:,} voxels:")
        print(f"  Time per RF pulse: ~{estimated_time_per_pulse:.1f} seconds")
        print(f"  Full sequence ({pulses_per_sequence} pulses): ~{estimated_time_per_pulse * pulses_per_sequence:.0f} seconds")
        print(f"  VOP equivalent ({n_vop_points} points): ~{pulses_per_sequence * 0.001:.1f} seconds")
        
        print(f"\nConclusion:")
        print(f"  → Full resolution: Research/validation use")
        print(f"  → VOP compression: Real-time clinical use")
        
    except Exception as e:
        print(f"Large-scale test failed: {e}")
        print(f"Consider reducing spatial points or using GPU acceleration")
