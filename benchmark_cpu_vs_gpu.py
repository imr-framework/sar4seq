#!/usr/bin/env python3
"""
CPU vs GPU Benchmark for SAR4seq Clinical Assessment

This script runs identical SAR computations on CPU and GPU to demonstrate
the performance improvements of the optimized GPU implementation.

The benchmark uses:
- Same number of spatial points (user configurable)
- Same clinical sequence (tse_500ms.seq)
- Same patient parameters
- Identical Q-matrix and tissue data

Expected results:
- GPU should be significantly faster for large spatial resolutions
- Results should be numerically identical between CPU and GPU
- Memory usage should be optimized on GPU
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
import argparse

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_dir)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU benchmarking")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy not available - only CPU benchmarking will be performed")

from uncompressed_qmatrix_sar import (
    compute_clinical_sar,
    load_tissue_data,
    create_full_resolution_qmatrix,
    clear_gpu_memory
)
import pypulseq as pp

def load_sequence(seq_path):
    """Load the clinical sequence"""
    if not os.path.exists(seq_path):
        print(f"❌ Sequence file not found: {seq_path}")
        return None
    
    print(f"📂 Loading sequence: {seq_path}")
    try:
        seq = pp.Sequence()
        seq.read(seq_path)
        print(f"✅ Sequence loaded: {len(seq.block_events)} blocks")
        return seq
    except Exception as e:
        print(f"❌ Error loading sequence: {e}")
        return None

def run_cpu_benchmark(seq, patient_weight, vendor, n_spatial_points):
    """Run CPU benchmark with specified spatial resolution"""
    print(f"\n🖥️  CPU BENCHMARK")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Spatial points: {n_spatial_points:,}")
    print(f"  Patient weight: {patient_weight} kg")
    print(f"  Vendor: {vendor}")
    print(f"  GPU acceleration: ❌ Disabled")
    
    start_time = time.time()
    
    try:
        results = compute_clinical_sar(
            seq=seq,
            patient_weight=patient_weight,
            vendor=vendor,
            use_gpu=False,  # Force CPU
            n_spatial_points=n_spatial_points
        )
        
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU computation completed successfully")
        print(f"⏱️  Total computation time: {cpu_time:.3f} seconds")
        print(f"📊 Performance metrics:")
        print(f"   RF blocks processed: {results['sequence_info']['rf_blocks']}")
        print(f"   Peak SAR: {results['clinical_assessment']['peak_sar']:.6f} W/kg")
        print(f"   Patient SAR: {results['clinical_assessment']['patient_sar']:.6f} W/kg")
        print(f"   Time per RF block: {cpu_time/results['sequence_info']['rf_blocks']:.3f} seconds")
        
        return results, cpu_time
        
    except Exception as e:
        print(f"❌ CPU benchmark failed: {e}")
        return None, float('inf')

def run_gpu_benchmark(seq, patient_weight, vendor, n_spatial_points):
    """Run GPU benchmark with specified spatial resolution"""
    if not CUPY_AVAILABLE:
        print(f"\n⚠️  GPU BENCHMARK SKIPPED - CuPy not available")
        return None, float('inf')
    
    print(f"\n🚀 GPU BENCHMARK (Optimized)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Spatial points: {n_spatial_points:,}")
    print(f"  Patient weight: {patient_weight} kg")
    print(f"  Vendor: {vendor}")
    print(f"  GPU acceleration: ✅ Enabled")
    
    start_time = time.time()
    
    try:
        results = compute_clinical_sar(
            seq=seq,
            patient_weight=patient_weight,
            vendor=vendor,
            use_gpu=True,  # Force GPU
            n_spatial_points=n_spatial_points
        )
        
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU computation completed successfully")
        print(f"⏱️  Total computation time: {gpu_time:.3f} seconds")
        print(f"📊 Performance metrics:")
        print(f"   RF blocks processed: {results['sequence_info']['rf_blocks']}")
        print(f"   Peak SAR: {results['clinical_assessment']['peak_sar']:.6f} W/kg")
        print(f"   Patient SAR: {results['clinical_assessment']['patient_sar']:.6f} W/kg")
        print(f"   Time per RF block: {gpu_time/results['sequence_info']['rf_blocks']:.3f} seconds")
        
        return results, gpu_time
        
    except Exception as e:
        print(f"❌ GPU benchmark failed: {e}")
        return None, float('inf')

def validate_results(cpu_results, gpu_results, tolerance=1e-6):
    """Validate that CPU and GPU give identical results"""
    print(f"\n🔍 RESULT VALIDATION")
    print("=" * 60)
    
    if cpu_results is None or gpu_results is None:
        print("❌ Cannot validate - one or both benchmarks failed")
        return False
    
    try:
        cpu_peak = cpu_results['clinical_assessment']['peak_sar']
        gpu_peak = gpu_results['clinical_assessment']['peak_sar']
        
        cpu_patient = cpu_results['clinical_assessment']['patient_sar']
        gpu_patient = gpu_results['clinical_assessment']['patient_sar']
        
        peak_diff = abs(cpu_peak - gpu_peak)
        patient_diff = abs(cpu_patient - gpu_patient)
        
        print(f"Peak SAR comparison:")
        print(f"  CPU:    {cpu_peak:.8f} W/kg")
        print(f"  GPU:    {gpu_peak:.8f} W/kg")
        print(f"  Diff:   {peak_diff:.2e} W/kg")
        
        print(f"\nPatient SAR comparison:")
        print(f"  CPU:    {cpu_patient:.8f} W/kg")
        print(f"  GPU:    {gpu_patient:.8f} W/kg")
        print(f"  Diff:   {patient_diff:.2e} W/kg")
        
        if peak_diff < tolerance and patient_diff < tolerance:
            print(f"\n✅ Results match within tolerance ({tolerance:.0e})")
            return True
        else:
            print(f"\n❌ Results differ beyond tolerance ({tolerance:.0e})")
            return False
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def print_performance_comparison(cpu_time, gpu_time, n_spatial_points, n_rf_blocks):
    """Print detailed performance comparison"""
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print(f"Test Configuration:")
    print(f"  Spatial points: {n_spatial_points:,}")
    print(f"  RF blocks: {n_rf_blocks}")
    print(f"  Total computations: {n_spatial_points * n_rf_blocks:,}")
    
    print(f"\nTiming Results:")
    print(f"  CPU time:  {cpu_time:.3f} seconds")
    
    if gpu_time != float('inf'):
        print(f"  GPU time:  {gpu_time:.3f} seconds")
        
        if gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            print(f"\n🚀 GPU is {speedup:.1f}x FASTER than CPU")
        else:
            slowdown = gpu_time / cpu_time
            print(f"\n🐌 GPU is {slowdown:.1f}x SLOWER than CPU")
            
        # Detailed analysis
        print(f"\nDetailed Analysis:")
        print(f"  CPU locations/second:  {(n_spatial_points * n_rf_blocks) / cpu_time:,.0f}")
        print(f"  GPU locations/second:  {(n_spatial_points * n_rf_blocks) / gpu_time:,.0f}")
        print(f"  CPU time per RF block: {cpu_time / n_rf_blocks:.3f} seconds")
        print(f"  GPU time per RF block: {gpu_time / n_rf_blocks:.3f} seconds")
    else:
        print(f"  GPU time:  Not available")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='CPU vs GPU SAR Benchmark')
    parser.add_argument('--spatial', type=int, default=50000,
                      help='Number of spatial points (default: 50000)')
    parser.add_argument('--weight', type=float, default=70.0,
                      help='Patient weight in kg (default: 70.0)')
    parser.add_argument('--vendor', type=str, default='siemens',
                      choices=['siemens', 'ge'], help='Scanner vendor')
    parser.add_argument('--sequence', type=str, 
                      default='/lhome/ext/i3m121/i3m1211/SAR/SAR4seq_python/tse_500ms.seq',
                      help='Path to sequence file')
    parser.add_argument('--cpu-only', action='store_true',
                      help='Run only CPU benchmark')
    parser.add_argument('--gpu-only', action='store_true',
                      help='Run only GPU benchmark')
    
    args = parser.parse_args()
    
    print("🏥 SAR4seq CPU vs GPU Benchmark")
    print("=" * 60)
    print(f"Benchmark Configuration:")
    print(f"  Spatial points: {args.spatial:,}")
    print(f"  Patient weight: {args.weight} kg") 
    print(f"  Scanner vendor: {args.vendor}")
    print(f"  Sequence file: {args.sequence}")
    print("=" * 60)
    
    # Load sequence
    seq = load_sequence(args.sequence)
    if seq is None:
        print("❌ Cannot proceed without sequence file")
        return
    
    # Run benchmarks
    cpu_results, cpu_time = None, float('inf')
    gpu_results, gpu_time = None, float('inf')
    
    if not args.gpu_only:
        cpu_results, cpu_time = run_cpu_benchmark(
            seq, args.weight, args.vendor, args.spatial
        )
    
    if not args.cpu_only and CUPY_AVAILABLE:
        gpu_results, gpu_time = run_gpu_benchmark(
            seq, args.weight, args.vendor, args.spatial
        )
        
        # Clean up GPU memory
        clear_gpu_memory()
    
    # Validate results
    if cpu_results and gpu_results:
        validate_results(cpu_results, gpu_results)
    
    # Print performance comparison
    n_rf_blocks = len([b for b in seq.block_events if hasattr(b, 'rf') and b.rf is not None])
    print_performance_comparison(cpu_time, gpu_time, args.spatial, n_rf_blocks)
    
    print(f"\n🏁 Benchmark completed!")

if __name__ == "__main__":
    main()
