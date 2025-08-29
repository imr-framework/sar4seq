#!/usr/bin/env python3
"""
Performance analysis for SAR4seq deployment scenarios

This script analyzes the computational performance of SAR4seq
under different realistic deployment conditions.
"""

import time
import numpy as np
import pypulseq as pp
from src.sar4seq import SAR4seq
from src.gen_seq_test import create_test_sequence, create_advanced_test_sequence


def create_large_sequence(num_rf_pulses=100):
    """Create a larger sequence for performance testing"""
    system = pp.Opts(
        max_grad=32, grad_unit='mT/m',
        max_slew=130, slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6
    )
    seq = pp.Sequence(system)
    
    # Create different RF pulses
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
    short_delay = pp.make_delay(5e-3)
    long_delay = pp.make_delay(20e-3)
    
    # Build a sequence with many RF pulses (simulating a real TSE or similar)
    for i in range(num_rf_pulses):
        if i % 10 == 0:
            seq.add_block(rf_180)  # Refocusing pulse every 10 blocks
        else:
            seq.add_block(rf_90)
        
        if i % 5 == 0:
            seq.add_block(long_delay)
        else:
            seq.add_block(short_delay)
    
    print(f"Created large sequence with {len(seq.block_events)} blocks ({num_rf_pulses} RF pulses)")
    return seq


def benchmark_sar_calculation():
    """Benchmark SAR calculation performance"""
    print("=== SAR4seq Performance Benchmark ===\n")
    
    # Test scenarios
    scenarios = [
        ("Small sequence (4 blocks)", create_test_sequence()),
        ("Advanced sequence (5 blocks)", create_advanced_test_sequence()),
        ("Medium sequence (50 RF pulses)", create_large_sequence(50)),
        ("Large sequence (200 RF pulses)", create_large_sequence(200)),
        ("Very large sequence (500 RF pulses)", create_large_sequence(500))
    ]
    
    results = []
    
    for name, seq in scenarios:
        print(f"Testing: {name}")
        print(f"  Blocks: {len(seq.block_events)}")
        
        # Count RF blocks
        rf_count = 0
        for i in range(len(seq.block_events)):
            block = seq.get_block(i + 1)
            if block.rf is not None:
                rf_count += 1
        print(f"  RF blocks: {rf_count}")
        
        # Benchmark computation time
        start_time = time.time()
        try:
            rf_body, rf_head, sar_ge = SAR4seq(seq=seq, sample_weight=60.0)
            end_time = time.time()
            
            computation_time = end_time - start_time
            results.append({
                'name': name,
                'blocks': len(seq.block_events),
                'rf_blocks': rf_count,
                'time': computation_time,
                'sar': sar_ge,
                'success': True
            })
            
            print(f"  Time: {computation_time:.3f} seconds")
            print(f"  SAR: {sar_ge:.3f} W/kg")
            print(f"  Performance: {rf_count/computation_time:.1f} RF blocks/second")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'name': name,
                'blocks': len(seq.block_events),
                'rf_blocks': rf_count,
                'time': None,
                'sar': None,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    return results


def analyze_scaling():
    """Analyze computational scaling with sequence size"""
    print("=== Scaling Analysis ===\n")
    
    rf_counts = [10, 25, 50, 100, 200, 400]
    times = []
    
    for rf_count in rf_counts:
        print(f"Testing {rf_count} RF pulses...")
        seq = create_large_sequence(rf_count)
        
        start_time = time.time()
        try:
            rf_body, rf_head, sar_ge = SAR4seq(seq=seq, sample_weight=60.0)
            end_time = time.time()
            computation_time = end_time - start_time
            times.append(computation_time)
            print(f"  Time: {computation_time:.3f}s, Rate: {rf_count/computation_time:.1f} RF/s")
        except Exception as e:
            print(f"  Error: {e}")
            times.append(None)
    
    # Analyze scaling
    print("\nScaling Analysis:")
    valid_data = [(rf, t) for rf, t in zip(rf_counts, times) if t is not None]
    
    if len(valid_data) > 1:
        # Linear fit to estimate scaling
        rf_vals = np.array([x[0] for x in valid_data])
        time_vals = np.array([x[1] for x in valid_data])
        
        # Simple linear regression
        A = np.vstack([rf_vals, np.ones(len(rf_vals))]).T
        slope, intercept = np.linalg.lstsq(A, time_vals, rcond=None)[0]
        
        print(f"  Estimated scaling: ~{slope*1000:.2f} ms per RF pulse")
        print(f"  Base overhead: ~{intercept*1000:.1f} ms")
        
        # Predict time for realistic scenarios
        realistic_scenarios = [1000, 5000, 10000]
        print("\nPredicted times for realistic sequences:")
        for rf_pulses in realistic_scenarios:
            predicted_time = slope * rf_pulses + intercept
            print(f"  {rf_pulses:5d} RF pulses: ~{predicted_time:.2f} seconds")


def memory_analysis():
    """Analyze memory usage patterns"""
    print("\n=== Memory Analysis ===")
    
    # Test with different Q-matrix sizes (simulating real deployment scenarios)
    print("\nCurrent Q-matrix dimensions: 4x4 (test matrices)")
    print("Real deployment scenarios might have:")
    print("- Multi-channel arrays: 8x8, 16x16, 32x32 Q-matrices")
    print("- Higher resolution models: larger Q-matrices")
    print("- Multiple tissue types: separate Q-matrices for different regions")
    
    # Estimate memory for different scenarios
    matrix_sizes = [4, 8, 16, 32, 64]
    for size in matrix_sizes:
        # Each complex number is 16 bytes (8 for real, 8 for imaginary)
        matrix_memory = size * size * 16 * 2  # 2 matrices (Qtmf, Qhmf)
        print(f"- {size}x{size} Q-matrices: ~{matrix_memory/1024:.1f} KB")


def deployment_recommendations():
    """Provide deployment recommendations"""
    print("\n=== Deployment Recommendations ===\n")
    
    print("1. PERFORMANCE CHARACTERISTICS:")
    print("   - Current implementation: O(N) scaling with RF blocks")
    print("   - Q-matrix operations: O(1) per RF block (small matrices)")
    print("   - File I/O: Minimal impact for typical sequences")
    
    print("\n2. POTENTIAL BOTTLENECKS:")
    print("   - Large sequences (>1000 RF pulses): May take seconds")
    print("   - Complex RF waveforms: More signal processing overhead")
    print("   - Large Q-matrices: Increased memory and computation")
    print("   - Real-time applications: Current implementation may be too slow")
    
    print("\n3. OPTIMIZATION OPPORTUNITIES:")
    print("   - Vectorization: Process multiple RF blocks simultaneously")
    print("   - Caching: Pre-compute frequently used Q-matrix operations")
    print("   - Parallel processing: Multi-threading for large sequences")
    print("   - Memory optimization: Sparse matrix representations")
    
    print("\n4. DEPLOYMENT SCENARIOS:")
    print("   - Research/Development: Current performance adequate")
    print("   - Clinical Pre-planning: May need optimization for large sequences")
    print("   - Real-time monitoring: Requires significant optimization")
    print("   - Batch processing: Current implementation suitable")


def main():
    """Run complete performance analysis"""
    try:
        # Run benchmarks
        results = benchmark_sar_calculation()
        
        # Analyze scaling
        analyze_scaling()
        
        # Memory analysis
        memory_analysis()
        
        # Deployment recommendations
        deployment_recommendations()
        
        print("\n=== Summary ===")
        successful_tests = [r for r in results if r['success']]
        if successful_tests:
            avg_time = np.mean([r['time'] for r in successful_tests])
            max_time = np.max([r['time'] for r in successful_tests])
            max_blocks = np.max([r['rf_blocks'] for r in successful_tests])
            
            print(f"Average computation time: {avg_time:.3f} seconds")
            print(f"Maximum computation time: {max_time:.3f} seconds")
            print(f"Largest test: {max_blocks} RF blocks")
            print(f"Performance: Current implementation suitable for research use")
            if max_time > 1.0:
                print(f"Warning: Large sequences may require optimization for production")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")


if __name__ == "__main__":
    main()
