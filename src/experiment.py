#!/usr/bin/env python3

import gc
import sys

# Local imports
from utils.calc_sar import SAR4seq_advanced
from utils.benchmark_sar import test_large_scale_sar, benchmark_sar_methods, test_with_pulseq_sequence
from colorama import Fore, Style

# GPU acceleration support
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Global variables for GPU optimization
_GPU_Q_MATRIX = None
_GPU_Q_SHAPE = None


def clear_gpu_memory():
    """
    Clear GPU Q-matrix from memory to free up GPU resources
    """
    global _GPU_Q_MATRIX, _GPU_Q_SHAPE
    
    if CUPY_AVAILABLE and _GPU_Q_MATRIX is not None:
        print(f"{Fore.GREEN}Clearing GPU Q-matrix from memory...")
        _GPU_Q_MATRIX = None
        _GPU_Q_SHAPE = None
        cp.cuda.Device().synchronize()
        # Force garbage collection on GPU
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        print(f"{Fore.CYAN}{Style.BRIGHT}Fore.GPU memory cleared{Style.RESET_ALL}")


def run_clinical_interface():
    """
    Interactive command-line interface for clinical SAR assessment
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}ADVANCED SAR4SEQ: CLINICAL RF SAFETY ASSESSMENT{Style.RESET_ALL}")
    print("Research-Grade SAR Analysis with Clinical Validation")
    print("=" * 70)
    
    # Clinical interface for practical healthcare use
    print("\nSelect assessment mode:")
    print("1. Clinical Mode (GPU-accelerated full-resolution - Fast & accurate)")
    print("2. VOP Mode (Compressed SAR - Regulatory compliance)")
    print("3. Research Mode (High-resolution analysis - Detailed SAR maps)")
    print("4. Benchmark Mode (Compare all methods)")
    
    try:
        choice = input("\nEnter choice (1-4) or press Enter for Clinical mode: ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    
    # Get additional parameters for clinical use
    try:
        patient_weight = input("Patient weight (kg) [40.0]: ").strip()
        patient_weight = float(patient_weight) if patient_weight else 40.0
    except ValueError:
        patient_weight = 40.0
    
    try:
        vendor = input("Scanner vendor (siemens/ge) [siemens]: ").strip().lower()
        vendor = vendor if vendor in ['siemens', 'ge'] else 'siemens'
    except:
        vendor = 'siemens'
    
    try:
        seq_path = input("Sequence file path (optional): ").strip()
        seq_path = seq_path if seq_path else None
    except:
        seq_path = None
    
    use_gpu = CUPY_AVAILABLE
    if CUPY_AVAILABLE:
        try:
            gpu_choice = input("Use GPU acceleration? (Y/n) [Y]: ").strip().lower()
            use_gpu = gpu_choice != 'n'
        except:
            use_gpu = True
    
    return choice, patient_weight, vendor, seq_path, use_gpu


if __name__ == "__main__":
    choice, patient_weight, vendor, seq_path, use_gpu = run_clinical_interface()
    
    print("\n" + "=" * 70)
    
    if choice == "1" or choice == "":
        # Clinical mode - GPU-accelerated full-resolution
        try:
            results = SAR4seq_advanced(
                seq_path=seq_path,
                patient_weight=patient_weight,
                computation_mode='clinical',
                use_gpu=use_gpu,
                vendor=vendor,
                safety_checks=True
            )
            
            print(f"\n{Fore.CYAN}{Style.BRIGHT}Clinical assessment completed successfully!{Style.RESET_ALL}")
            print(f"Computation time: {results['performance_metrics']['computation_time']:.2f} seconds")
            print(f"Safety compliance: {'PASS' if results['safety_assessment']['compliant'] else '❌ FAIL'}")
            
        except ValueError as e:
            print(f"\n{Fore.RED}SAFETY VIOLATION: {e}")
            print("Sequence requires modification before clinical use.")
        except Exception as e:
            print(f"\n{Fore.RED}Error in clinical assessment: {e}")
    
    elif choice == "2":
        # VOP mode - Compressed SAR for regulatory compliance
        try:
            results = SAR4seq_advanced(
                seq_path=seq_path,
                patient_weight=patient_weight,
                computation_mode='vop',
                use_gpu=use_gpu,
                vendor=vendor,
                safety_checks=True
            )
            
            print(f"\n{Fore.CYAN}{Style.BRIGHT}VOP assessment completed successfully!{Style.RESET_ALL}")
            print(f"Computation time: {results['performance_metrics']['computation_time']:.2f} seconds")
            print(f"Compression ratio: {results['performance_metrics']['compression_ratio']:.1f}:1")
            print(f"Safety compliance: {'{Fore.CYAN}{Style.BRIGHT}PASS' if results['safety_assessment']['compliant'] else '❌ FAIL'}{Style.RESET_ALL}")
            
        except ValueError as e:
            print(f"\n{Fore.RED}SAFETY VIOLATION: {e}")
            print("Sequence requires modification before regulatory submission.")
        except Exception as e:
            print(f"\n{Fore.RED}Error in VOP assessment: {e}")
    
    elif choice == "3":
        # Research mode - High-resolution analysis
        try:
            results = SAR4seq_advanced(
                seq_path=seq_path,
                patient_weight=patient_weight,
                computation_mode='research',
                use_gpu=use_gpu,
                vendor=vendor,
                safety_checks=True
            )
            
            print(f"\n{Fore.CYAN}{Style.BRIGHT}Research analysis completed successfully!{Style.RESET_ALL}")
            if 'spatial_analysis' in results['sar_analysis']:
                spatial = results['sar_analysis']['spatial_analysis']
                print(f"Analyzed {spatial['total_voxels']:,} spatial points")
                print(f"Peak SAR: {spatial['sar_global_max']:.3f} W/kg")
                print(f"Hotspots detected: {spatial['hotspot_count']}")
        
        except Exception as e:
            print(f"\n{Fore.RED}Error in research analysis: {e}")
    
    elif choice == "4":
        # Benchmark mode - Compare methods
        try:
            results = SAR4seq_advanced(
                seq_path=seq_path,
                patient_weight=patient_weight,
                computation_mode='benchmark',
                use_gpu=use_gpu,
                vendor=vendor,
                safety_checks=True
            )
            
            print(f"\n{Fore.CYAN}{Style.BRIGHT}Benchmark comparison completed!{Style.RESET_ALL}")
            perf = results['performance_metrics']
            print(f"Clinical method: {perf['clinical']['computation_time']:.3f}s")
            print(f"Research method: {perf['research']['computation_time']:.3f}s")
            print(f"Clinical speedup: {perf['speedup_factor']:.1f}x faster")
        
        except Exception as e:
            print(f"\n{Fore.RED}Error in benchmark: {e}")
    
    elif choice == "4":
        # Legacy testing mode
        print("\nLEGACY TESTING MODE")
        print("Select test type:")
        print("1. Quick tests (1K-10K voxels)")
        print("2. Large-scale test (100K voxels)")
        print("3. Custom test")
        
        try:
            test_choice = input("Enter choice (1-3): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        
        if test_choice == "2":
            test_large_scale_sar(n_spatial_points=100000, n_vop_points=500)
        elif test_choice == "3":
            try:
                n_spatial = int(input("Enter number of spatial points: "))
                n_vop = int(input("Enter number of VOP points: "))
                test_large_scale_sar(n_spatial_points=n_spatial, n_vop_points=n_vop)
            except ValueError:
                print("Invalid input. Running default tests.")
                test_choice = "1"
        
        if test_choice in ["1", ""]:
            test_cases = [
                {"n_spatial": 1000, "n_vop": 50, "name": "Small test"},
                {"n_spatial": 5000, "n_vop": 100, "name": "Medium test"},
                {"n_spatial": 10000, "n_vop": 200, "name": "Large test"},
            ]
            
            for i, test_case in enumerate(test_cases):
                print(f"\n{'='*20} TEST CASE {i+1}: {test_case['name']} {'='*20}")
                
                try:
                    benchmark_sar_methods(
                        n_spatial_points=test_case["n_spatial"],
                        n_vop_points=test_case["n_vop"]
                    )
                except KeyboardInterrupt:
                    print(f"\nTest interrupted by user")
                    break
                except Exception as e:
                    print(f"Error in test case: {e}")
                    continue
            
            # Test with actual Pulseq sequence
            try:
                test_with_pulseq_sequence()
            except Exception as e:
                print(f"Pulseq sequence test failed: {e}")
    
    print(f"\n" + "=" * 70)
    print(f"{Fore.CYAN}{Style.BRIGHT}SAR4seq Clinical Assessment Complete{Style.RESET_ALL}")
    
    # Clean up GPU memory
    if CUPY_AVAILABLE:
        clear_gpu_memory()
        print(f"{Fore.CYAN}{Style.BRIGHT}GPU acceleration available and utilized{Style.RESET_ALL}")
    else:
        print("Install CuPy for GPU acceleration: pip install cupy")
    
    print("For clinical use, ensure proper validation and regulatory compliance")
    print("For research applications, consider high-resolution mode for detailed analysis")
    print("=" * 70)
