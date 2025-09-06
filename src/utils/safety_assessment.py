import time
from utils.constants import CLINICAL_CONSTANTS

def perform_safety_assessment(results):
    """
    Perform comprehensive safety assessment against clinical limits
    """
    
    print("\n🛡️  CLINICAL SAFETY ASSESSMENT")
    print("-" * 40)
    
    violations = []
    warnings = []
    
    # Extract SAR values based on computation mode
    if results['computation_mode'] == 'benchmark':
        sar_analysis = results['sar_analysis']['research']  # Use more conservative research values
    else:
        sar_analysis = results['sar_analysis']
    
    if 'error' in sar_analysis:
        warnings.append(f"SAR analysis error: {sar_analysis['error']}")
        results['safety_assessment']['compliant'] = False
        return
    
    sar_predicted = sar_analysis['sar_predicted_patient_W_per_kg']
    sar_peak = sar_analysis['sar_peak_global_W_per_kg']
    
    # Check against safety limits
    if sar_predicted > CLINICAL_CONSTANTS['ten_sec_thresh_wbg']:
        violations.append(f"10-second whole body SAR limit exceeded: {sar_predicted:.2f} > {CLINICAL_CONSTANTS['ten_sec_thresh_wbg']} W/kg")
    
    if sar_predicted > CLINICAL_CONSTANTS['six_min_thresh_wbg']:
        violations.append(f"6-minute whole body SAR limit exceeded: {sar_predicted:.2f} > {CLINICAL_CONSTANTS['six_min_thresh_wbg']} W/kg")
    
    # Check for research mode spatial hotspots
    if 'spatial_analysis' in sar_analysis:
        hotspot_count = sar_analysis['spatial_analysis']['hotspot_count']
        if hotspot_count > 0:
            warnings.append(f"Local SAR hotspots detected: {hotspot_count} voxels > {CLINICAL_CONSTANTS['local_sar_limit']} W/kg")
    
    # Print assessment results
    if violations:
        print("  ❌ SAFETY VIOLATIONS DETECTED:")
        for violation in violations:
            print(f"    - {violation}")
        results['safety_assessment']['compliant'] = False
    else:
        print("  ✅ All safety limits within acceptable ranges")
        results['safety_assessment']['compliant'] = True
    
    if warnings:
        print("  ⚠️  WARNINGS:")
        for warning in warnings:
            print(f"    - {warning}")
    
    results['safety_assessment']['violations'] = violations
    results['safety_assessment']['warnings'] = warnings
    
    # Raise exception if violations found and safety checks enabled
    if violations and results.get('safety_checks', True):
        raise ValueError(f"Sequence violates clinical safety limits: {'; '.join(violations)}")


def generate_clinical_report(results):
    """
    Generate comprehensive clinical assessment report
    """
    
    print("\n" + "=" * 60)
    print("📋 CLINICAL SAR ASSESSMENT REPORT")
    print("=" * 60)
    
    # Header information
    print(f"Patient Weight: {results['patient_weight']:.1f} kg")
    print(f"Scanner Vendor: {results['vendor'].upper()}")
    print(f"Computation Mode: {results['computation_mode'].upper()}")
    print(f"Assessment Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sequence information
    seq_info = results['sequence_info']
    print(f"\nSequence Analysis:")
    print(f"  Total blocks: {seq_info['total_blocks']}")
    print(f"  RF blocks: {seq_info['rf_blocks']}")
    print(f"  Duration: {seq_info['total_duration']:.3f} seconds")
    
    # SAR results
    if 'error' not in results['sar_analysis']:
        if results['computation_mode'] == 'benchmark':
            # Show both clinical and research results
            clinical_sar = results['sar_analysis']['clinical']
            research_sar = results['sar_analysis']['research']
            
            print(f"\nSAR Analysis (Clinical VOP):")
            print(f"  Peak SAR: {clinical_sar['sar_peak_global_W_per_kg']:.3f} W/kg")
            print(f"  Predicted Patient SAR: {clinical_sar['sar_predicted_patient_W_per_kg']:.3f} W/kg")
            print(f"  Time-averaged RF Power: {clinical_sar['rf_power_avg_W']:.3f} W")
            
            print(f"\nSAR Analysis (Research Full-Resolution):")
            print(f"  Peak SAR: {research_sar['sar_peak_global_W_per_kg']:.3f} W/kg")  
            print(f"  Predicted Patient SAR: {research_sar['sar_predicted_patient_W_per_kg']:.3f} W/kg")
            
            if 'spatial_analysis' in research_sar:
                spatial = research_sar['spatial_analysis']
                print(f"  Spatial Analysis ({spatial['total_voxels']:,} voxels):")
                print(f"    Global max SAR: {spatial['sar_global_max']:.3f} W/kg")
                print(f"    95th percentile: {spatial['sar_percentile_95']:.3f} W/kg")
                print(f"    Hotspots (>{CLINICAL_CONSTANTS['local_sar_limit']} W/kg): {spatial['hotspot_count']}")
        else:
            sar_analysis = results['sar_analysis']
            print(f"\nSAR Analysis:")
            print(f"  Peak SAR: {sar_analysis['sar_peak_global_W_per_kg']:.3f} W/kg")
            print(f"  Predicted Patient SAR: {sar_analysis['sar_predicted_patient_W_per_kg']:.3f} W/kg")
            print(f"  Time-averaged RF Power: {sar_analysis['rf_power_avg_W']:.3f} W")
            
            if 'spatial_analysis' in sar_analysis:
                spatial = sar_analysis['spatial_analysis']
                print(f"  Spatial Analysis ({spatial['total_voxels']:,} voxels):")
                print(f"    Global max SAR: {spatial['sar_global_max']:.3f} W/kg")
                print(f"    Mean SAR: {spatial['sar_global_mean']:.3f} W/kg")
                print(f"    Hotspots: {spatial['hotspot_count']}")
    
    # Performance metrics  
    perf = results['performance_metrics']
    if isinstance(perf, dict) and 'clinical' in perf:
        # Benchmark mode
        print(f"\nPerformance Metrics:")
        print(f"  Clinical (VOP): {perf['clinical']['computation_time']:.3f}s")
        print(f"  Research (Full): {perf['research']['computation_time']:.3f}s")
        print(f"  Speedup factor: {perf['speedup_factor']:.1f}x")
        print(f"  Compression ratio: {perf['clinical']['compression_ratio']:.1f}:1")
    else:
        print(f"\nPerformance Metrics:")
        print(f"  Computation time: {perf['computation_time']:.3f} seconds")
        print(f"  Method: {perf['method']}")
        if 'compression_ratio' in perf:
            print(f"  Compression ratio: {perf['compression_ratio']:.1f}:1")
        print(f"  GPU acceleration: {'Yes' if perf['gpu_used'] else 'No'}")
    
    # Safety assessment
    safety = results['safety_assessment']
    print(f"\nSafety Assessment:")
    if safety['compliant']:
        print("  ✅ COMPLIANT - All safety limits within acceptable ranges")
    else:
        print("  ❌ NON-COMPLIANT - Safety violations detected")
        
    if safety['violations']:
        print("  Violations:")
        for violation in safety['violations']:
            print(f"    - {violation}")
            
    if safety['warnings']:
        print("  Warnings:")
        for warning in safety['warnings']:
            print(f"    - {warning}")
    
    print("\n" + "=" * 60)
    
    # Clinical recommendations
    print("📋 CLINICAL RECOMMENDATIONS:")
    if results['computation_mode'] == 'clinical':
        print("  • Use for real-time clinical SAR monitoring")
        print("  • Fast VOP compression suitable for scanner integration")
    elif results['computation_mode'] == 'research':
        print("  • Use for detailed SAR analysis and sequence optimization")
        print("  • High spatial resolution reveals SAR hotspots")
    else:  # benchmark
        print("  • Clinical mode recommended for routine use")
        print("  • Research mode recommended for sequence development")
    
    if not safety['compliant']:
        print("  • ⚠️  SEQUENCE MODIFICATION REQUIRED before clinical use")
        print("  • Consider increasing TR or reducing flip angles")
    
    print("=" * 60)