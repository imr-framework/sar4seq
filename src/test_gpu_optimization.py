#!/usr/bin/env python3
"""
Test script to demonstrate the optimized GPU SAR computation performance

This script compares:
1. Original GPU implementation (transfer Q-matrix for each RF vector)
2. Optimized batch GPU implementation (transfer Q-matrix once, process all RF vectors)
3. CPU implementation for reference

Expected performance improvements:
- 10-50x speedup for batch GPU vs original GPU
- Elimination of memory transfer bottleneck
- Better GPU utilization
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU testing")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy not available - GPU tests will be skipped")

from src.experiment import (
    calc_SAR_full_resolution_cpu,
    calc_SAR_full_resolution_gpu,
    calc_SAR_full_resolution_gpu_batch,
    clear_gpu_memory
)

def create_test_data(n_spatial=10000, n_channels=4, n_rf_vectors=50):
    """Create test Q-matrix and RF vectors"""
    print(f"Creating test data: {n_spatial} spatial points, {n_channels} channels, {n_rf_vectors} RF vectors")
    
    # Create realistic Q-matrix (Hermitian positive definite)
    Q_full = np.zeros((n_spatial, n_channels, n_channels), dtype=np.complex64)
    
    for k in range(n_spatial):
        # Create random Hermitian matrix
        A = np.random.randn(n_channels, n_channels) + 1j * np.random.randn(n_channels, n_channels)
        Q_full[k] = (A @ A.conj().T) * 1e-6  # Scale to realistic SAR values
    
    # Create realistic RF vectors
    rf_vectors = []
    for i in range(n_rf_vectors):
        # Realistic RF amplitudes and phases
        amplitudes = np.random.uniform(0.1, 2.0, n_channels) * 1e-3
        phases = np.random.uniform(0, 2*np.pi, n_channels)
        rf_vector = amplitudes * np.exp(1j * phases)
        rf_vectors.append(rf_vector)
    
    mass = 70.0  # kg
    
    return Q_full, rf_vectors, mass

def test_cpu_performance(Q_full, rf_vectors, mass):
    """Test CPU performance"""
    print(f"\n🖥️  CPU Performance Test")
    print("-" * 50)
    
    start_time = time.time()
    
    cpu_results = []
    for i, rf_vector in enumerate(rf_vectors):
        sar_local, sar_peak = calc_SAR_full_resolution_cpu(Q_full, rf_vector, mass)
        cpu_results.append((sar_local, sar_peak))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(rf_vectors)} RF vectors")
    
    cpu_time = time.time() - start_time
    
    print(f"✅ CPU computation completed")
    print(f"⏱️  Total time: {cpu_time:.3f} seconds")
    print(f"📊 Time per RF vector: {cpu_time/len(rf_vectors):.3f} seconds")
    
    return cpu_results, cpu_time

def test_original_gpu_performance(Q_full, rf_vectors, mass):
    """Test original GPU performance (transfer Q-matrix each time)"""
    if not CUPY_AVAILABLE:
        print("⚠️  Skipping original GPU test - CuPy not available")
        return None, float('inf')
    
    print(f"\n🐌 Original GPU Performance Test (Inefficient)")
    print("-" * 50)
    
    start_time = time.time()
    
    gpu_results = []
    for i, rf_vector in enumerate(rf_vectors):
        sar_local, sar_peak = calc_SAR_full_resolution_gpu(Q_full, rf_vector, mass)
        gpu_results.append((sar_local, sar_peak))
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(rf_vectors)} RF vectors")
    
    gpu_time = time.time() - start_time
    
    print(f"✅ Original GPU computation completed")
    print(f"⏱️  Total time: {gpu_time:.3f} seconds")
    print(f"📊 Time per RF vector: {gpu_time/len(rf_vectors):.3f} seconds")
    
    return gpu_results, gpu_time

def test_optimized_gpu_performance(Q_full, rf_vectors, mass):
    """Test optimized batch GPU performance"""
    if not CUPY_AVAILABLE:
        print("⚠️  Skipping optimized GPU test - CuPy not available")
        return None, float('inf')
    
    print(f"\n🚀 Optimized Batch GPU Performance Test")
    print("-" * 50)
    
    start_time = time.time()
    
    # Single batch call for all RF vectors
    batch_results = calc_SAR_full_resolution_gpu_batch(Q_full, rf_vectors, mass)
    
    batch_time = time.time() - start_time
    
    print(f"✅ Batch GPU computation completed")
    print(f"⏱️  Total time: {batch_time:.3f} seconds")
    print(f"📊 Time per RF vector: {batch_time/len(rf_vectors):.3f} seconds")
    
    return batch_results, batch_time

def validate_results(cpu_results, gpu_results, batch_results, tolerance=1e-6):
    """Validate that all methods give the same results"""
    print(f"\n🔍 Validating Results")
    print("-" * 50)
    
    if cpu_results is None:
        print("❌ No CPU results to validate")
        return False
    
    all_valid = True
    
    if gpu_results is not None:
        print("Comparing CPU vs Original GPU...")
        for i, ((cpu_local, cpu_peak), (gpu_local, gpu_peak)) in enumerate(zip(cpu_results, gpu_results)):
            if abs(cpu_peak - gpu_peak) > tolerance:
                print(f"❌ Mismatch in RF vector {i}: CPU peak={cpu_peak:.6f}, GPU peak={gpu_peak:.6f}")
                all_valid = False
                break
        else:
            print("✅ CPU and Original GPU results match")
    
    if batch_results is not None:
        print("Comparing CPU vs Batch GPU...")
        for i, ((cpu_local, cpu_peak), (batch_local, batch_peak)) in enumerate(zip(cpu_results, batch_results)):
            if abs(cpu_peak - batch_peak) > tolerance:
                print(f"❌ Mismatch in RF vector {i}: CPU peak={cpu_peak:.6f}, Batch GPU peak={batch_peak:.6f}")
                all_valid = False
                break
        else:
            print("✅ CPU and Batch GPU results match")
    
    if all_valid:
        print("🎉 All methods produce consistent results!")
    
    return all_valid

def print_performance_summary(cpu_time, gpu_time, batch_time, n_rf_vectors):
    """Print performance comparison summary"""
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Test Configuration:")
    print(f"  RF vectors: {n_rf_vectors}")
    print(f"  Spatial points: {Q_full.shape[0]:,}")
    print(f"  Channels: {Q_full.shape[1]}")
    print()
    
    print(f"Results:")
    print(f"  CPU time:           {cpu_time:.3f} seconds")
    if gpu_time != float('inf'):
        print(f"  Original GPU time:  {gpu_time:.3f} seconds")
    if batch_time != float('inf'):
        print(f"  Batch GPU time:     {batch_time:.3f} seconds")
    print()
    
    if gpu_time != float('inf') and batch_time != float('inf'):
        speedup_gpu_vs_batch = gpu_time / batch_time
        print(f"Speedup Analysis:")
        print(f"  Batch GPU vs Original GPU: {speedup_gpu_vs_batch:.1f}x faster")
        
        if batch_time < cpu_time:
            speedup_cpu_vs_batch = cpu_time / batch_time
            print(f"  Batch GPU vs CPU:          {speedup_cpu_vs_batch:.1f}x faster")
        else:
            slowdown_cpu_vs_batch = batch_time / cpu_time
            print(f"  Batch GPU vs CPU:          {slowdown_cpu_vs_batch:.1f}x slower")
    
    print("=" * 60)

if __name__ == "__main__":
    print("🧪 SAR4seq GPU Optimization Performance Test")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {"n_spatial": 5000, "n_channels": 4, "n_rf_vectors": 20, "name": "Small Test"},
        {"n_spatial": 10000, "n_channels": 4, "n_rf_vectors": 50, "name": "Medium Test"},
        {"n_spatial": 20000, "n_channels": 4, "n_rf_vectors": 100, "name": "Large Test"},
    ]
    
    for config in test_configs:
        print(f"\n🎯 Running {config['name']}")
        print("=" * 60)
        
        # Create test data
        Q_full, rf_vectors, mass = create_test_data(
            config["n_spatial"], 
            config["n_channels"], 
            config["n_rf_vectors"]
        )
        
        # Run tests
        cpu_results, cpu_time = test_cpu_performance(Q_full, rf_vectors, mass)
        gpu_results, gpu_time = test_original_gpu_performance(Q_full, rf_vectors, mass)
        batch_results, batch_time = test_optimized_gpu_performance(Q_full, rf_vectors, mass)
        
        # Validate results
        validate_results(cpu_results, gpu_results, batch_results)
        
        # Print summary
        print_performance_summary(cpu_time, gpu_time, batch_time, config["n_rf_vectors"])
        
        # Clear GPU memory between tests
        if CUPY_AVAILABLE:
            clear_gpu_memory()
    
    print(f"\n🏁 All performance tests completed!")
    
    if CUPY_AVAILABLE:
        clear_gpu_memory()
