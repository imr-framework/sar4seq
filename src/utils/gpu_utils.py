#!/usr/bin/env python3
"""
GPU Acceleration Utilities

Contains GPU optimization functions, memory management, and performance
utilities for accelerated SAR computation using CuPy.

Authors: Leo Kinyera, BS
Copyright: Board of Trustees of Columbia University in the City of New York
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import time
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None
    cp_sparse = None

def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and capabilities.
    
    Returns:
        Dictionary containing GPU information and capabilities
    """
    gpu_info = {
        'gpu_available': HAS_CUPY,
        'cupy_installed': HAS_CUPY,
    }
    
    if not HAS_CUPY:
        gpu_info.update({
            'gpu_count': 0,
            'error': 'CuPy not installed. Install with: pip install cupy',
            'recommendation': 'Install CuPy for GPU acceleration'
        })
        return gpu_info
    
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        gpu_info['gpu_count'] = gpu_count
        
        if gpu_count == 0:
            gpu_info.update({
                'error': 'No CUDA-capable GPU found',
                'recommendation': 'Use CPU computation or check GPU drivers'
            })
            return gpu_info
        
        # Get current device info
        device = cp.cuda.Device()
        gpu_info.update({
            'current_device': device.id,
            'device_name': device.attributes['Name'].decode() if 'Name' in device.attributes else 'Unknown',
            'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            'total_memory_gb': device.mem_info[1] / (1024**3),
            'free_memory_gb': device.mem_info[0] / (1024**3),
        })
        
        # Test basic GPU operation
        try:
            test_array = cp.random.random((1000, 1000))
            cp.cuda.Stream.null.synchronize()
            gpu_info['gpu_functional'] = True
        except Exception as e:
            gpu_info.update({
                'gpu_functional': False,
                'error': f'GPU test failed: {e}',
                'recommendation': 'Check GPU drivers and CUDA installation'
            })
        
    except Exception as e:
        gpu_info.update({
            'error': f'Failed to access GPU: {e}',
            'recommendation': 'Check CUDA installation and drivers'
        })
    
    return gpu_info

def optimize_gpu_memory(data_size_gb: float, 
                       available_memory_gb: float = None) -> Dict[str, Any]:
    """
    Optimize GPU memory usage for large SAR computations.
    
    Args:
        data_size_gb: Estimated data size in GB
        available_memory_gb: Available GPU memory in GB
        
    Returns:
        Dictionary containing memory optimization strategy
    """
    if not HAS_CUPY:
        return {'strategy': 'cpu_only', 'reason': 'CuPy not available'}
    
    if available_memory_gb is None:
        try:
            device = cp.cuda.Device()
            available_memory_gb = device.mem_info[0] / (1024**3)
        except:
            available_memory_gb = 4.0  # Conservative default
    
    # Memory optimization strategy
    memory_ratio = data_size_gb / available_memory_gb
    
    if memory_ratio < 0.3:
        strategy = {
            'strategy': 'full_gpu',
            'description': 'Process entire dataset on GPU',
            'batch_size': None,
            'expected_speedup': '5-20x vs CPU'
        }
    elif memory_ratio < 0.8:
        strategy = {
            'strategy': 'gpu_with_streaming',
            'description': 'Use GPU with memory streaming',
            'batch_size': None,
            'expected_speedup': '3-10x vs CPU'
        }
    elif memory_ratio < 2.0:
        # Calculate optimal batch size
        batch_factor = int(np.ceil(memory_ratio / 0.7))
        strategy = {
            'strategy': 'gpu_batched',
            'description': f'Process in {batch_factor} batches on GPU',
            'batch_size': batch_factor,
            'expected_speedup': '2-8x vs CPU'
        }
    else:
        strategy = {
            'strategy': 'hybrid_cpu_gpu',
            'description': 'Use CPU with GPU for critical operations',
            'batch_size': int(np.ceil(memory_ratio)),
            'expected_speedup': '1.5-3x vs CPU'
        }
    
    strategy.update({
        'data_size_gb': data_size_gb,
        'available_memory_gb': available_memory_gb,
        'memory_ratio': memory_ratio,
        'recommendations': _get_memory_recommendations(strategy)
    })
    
    return strategy

def _get_memory_recommendations(strategy: Dict[str, Any]) -> List[str]:
    """Generate memory optimization recommendations."""
    recommendations = []
    
    memory_ratio = strategy['memory_ratio']
    
    if memory_ratio > 1.5:
        recommendations.append("• Consider using VOP compression to reduce memory requirements")
        recommendations.append("• Use lower spatial resolution for initial testing")
    
    if memory_ratio > 0.8:
        recommendations.append("• Close other GPU applications to free memory")
        recommendations.append("• Consider using mixed precision (float32 instead of float64)")
    
    if strategy['strategy'] == 'gpu_batched':
        recommendations.append("• Monitor GPU memory usage during computation")
        recommendations.append("• Adjust batch size if out-of-memory errors occur")
    
    return recommendations

def setup_gpu_computation(Q_matrix: np.ndarray,
                         sequence_data: np.ndarray,
                         use_gpu: bool = True) -> Dict[str, Any]:
    """
    Set up GPU computation environment and transfer data.
    
    Args:
        Q_matrix: Q-matrix for SAR calculation
        sequence_data: Sequence event data
        use_gpu: Whether to attempt GPU setup
        
    Returns:
        Dictionary containing setup results and GPU arrays
    """
    setup_result = {
        'gpu_setup_successful': False,
        'use_gpu': False,
        'setup_time': 0,
        'memory_used_gb': 0
    }
    
    if not use_gpu or not HAS_CUPY:
        setup_result.update({
            'Q_matrix': Q_matrix,
            'sequence_data': sequence_data,
            'reason': 'GPU not requested or CuPy not available'
        })
        return setup_result
    
    start_time = time.time()
    
    try:
        # Check GPU availability
        gpu_info = check_gpu_availability()
        if not gpu_info.get('gpu_functional', False):
            setup_result.update({
                'Q_matrix': Q_matrix,
                'sequence_data': sequence_data,
                'reason': 'GPU not functional',
                'error': gpu_info.get('error', 'Unknown GPU error')
            })
            return setup_result
        
        # Estimate memory requirements
        q_memory = Q_matrix.nbytes / (1024**3)
        seq_memory = sequence_data.nbytes / (1024**3)
        total_memory = q_memory + seq_memory
        
        # Check if data fits in GPU memory
        available_memory = gpu_info.get('free_memory_gb', 0)
        if total_memory > available_memory * 0.8:  # Leave 20% buffer
            setup_result.update({
                'Q_matrix': Q_matrix,
                'sequence_data': sequence_data,
                'reason': f'Insufficient GPU memory: need {total_memory:.1f}GB, available {available_memory:.1f}GB'
            })
            return setup_result
        
        # Transfer data to GPU
        Q_gpu = cp.asarray(Q_matrix)
        seq_gpu = cp.asarray(sequence_data)
        
        # Synchronize to ensure transfer is complete
        cp.cuda.Stream.null.synchronize()
        
        setup_time = time.time() - start_time
        
        setup_result.update({
            'gpu_setup_successful': True,
            'use_gpu': True,
            'Q_matrix': Q_gpu,
            'sequence_data': seq_gpu,
            'setup_time': setup_time,
            'memory_used_gb': total_memory,
            'gpu_device': gpu_info.get('current_device', 0),
            'gpu_name': gpu_info.get('device_name', 'Unknown')
        })
        
    except Exception as e:
        setup_result.update({
            'Q_matrix': Q_matrix,
            'sequence_data': sequence_data,
            'reason': f'GPU setup failed: {e}',
            'error': str(e)
        })
    
    return setup_result

def benchmark_gpu_performance(Q_matrix: np.ndarray,
                             sequence_data: np.ndarray,
                             n_trials: int = 3) -> Dict[str, Any]:
    """
    Benchmark GPU vs CPU performance for SAR calculation.
    
    Args:
        Q_matrix: Test Q-matrix
        sequence_data: Test sequence data
        n_trials: Number of benchmark trials
        
    Returns:
        Dictionary containing benchmark results
    """
    benchmark_results = {
        'cpu_times': [],
        'gpu_times': [],
        'gpu_available': HAS_CUPY
    }
    
    if not HAS_CUPY:
        benchmark_results['error'] = 'CuPy not available for GPU benchmarking'
        return benchmark_results
    
    # Simplified benchmark operation (matrix multiplication)
    def benchmark_operation_cpu(Q, seq):
        """Simple CPU benchmark operation."""
        start = time.time()
        result = np.real(np.conj(seq) @ Q @ seq)
        return time.time() - start
    
    def benchmark_operation_gpu(Q, seq):
        """Simple GPU benchmark operation."""
        start = time.time()
        # Convert to GPU arrays
        Q_gpu = cp.asarray(Q)
        seq_gpu = cp.asarray(seq)
        result = cp.real(cp.conj(seq_gpu) @ Q_gpu @ seq_gpu)
        cp.cuda.Stream.null.synchronize()  # Wait for completion
        return time.time() - start
    
    # Prepare test data
    if Q_matrix.ndim == 3:
        # Use first channel for benchmark
        Q_test = Q_matrix[0]
        if sequence_data.ndim == 2:
            seq_test = sequence_data[0, :Q_matrix.shape[0]]
        else:
            seq_test = sequence_data[:Q_matrix.shape[0]]
    else:
        Q_test = Q_matrix
        seq_test = sequence_data
    
    # Ensure we have complex sequence data
    if seq_test.dtype.kind != 'c':  # Not complex
        seq_test = seq_test[:len(seq_test)//2] + 1j * seq_test[len(seq_test)//2:]
    
    try:
        # CPU benchmark
        for trial in range(n_trials):
            cpu_time = benchmark_operation_cpu(Q_test, seq_test)
            benchmark_results['cpu_times'].append(cpu_time)
        
        # GPU benchmark
        for trial in range(n_trials):
            gpu_time = benchmark_operation_gpu(Q_test, seq_test)
            benchmark_results['gpu_times'].append(gpu_time)
        
        # Calculate statistics
        cpu_avg = np.mean(benchmark_results['cpu_times'])
        gpu_avg = np.mean(benchmark_results['gpu_times'])
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        benchmark_results.update({
            'cpu_avg_time': cpu_avg,
            'gpu_avg_time': gpu_avg,
            'speedup_factor': speedup,
            'gpu_faster': speedup > 1,
            'recommendation': 'Use GPU' if speedup > 1.5 else 'Use CPU'
        })
        
    except Exception as e:
        benchmark_results['error'] = f'Benchmark failed: {e}'
    
    return benchmark_results

def optimize_batch_processing(total_data_size: int,
                             available_memory_gb: float,
                             dtype: np.dtype = np.float32) -> Dict[str, Any]:
    """
    Optimize batch size for memory-efficient processing.
    
    Args:
        total_data_size: Total number of elements to process
        available_memory_gb: Available GPU memory in GB
        dtype: Data type for memory estimation
        
    Returns:
        Dictionary containing batch optimization parameters
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_memory_gb = (total_data_size * bytes_per_element) / (1024**3)
    
    if total_memory_gb <= available_memory_gb * 0.8:
        return {
            'use_batching': False,
            'batch_size': total_data_size,
            'num_batches': 1,
            'memory_per_batch_gb': total_memory_gb
        }
    
    # Calculate optimal batch size
    target_memory_per_batch = available_memory_gb * 0.7  # Leave 30% buffer
    elements_per_batch = int((target_memory_per_batch * (1024**3)) / bytes_per_element)
    num_batches = int(np.ceil(total_data_size / elements_per_batch))
    
    return {
        'use_batching': True,
        'batch_size': elements_per_batch,
        'num_batches': num_batches,
        'memory_per_batch_gb': target_memory_per_batch,
        'total_memory_gb': total_memory_gb,
        'memory_reduction_factor': total_memory_gb / target_memory_per_batch
    }

def cleanup_gpu_memory():
    """Clean up GPU memory and reset memory pool."""
    if HAS_CUPY:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass


__all__ = [
    'check_gpu_availability',
    'optimize_gpu_memory',
    'setup_gpu_computation',
    'benchmark_gpu_performance',
    'optimize_batch_processing',
    'cleanup_gpu_memory'
]
