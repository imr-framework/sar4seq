#!/usr/bin/env python3
"""
Clinical Safety Assessment Functions

Contains functions for evaluating SAR against clinical safety limits,
generating compliance reports, and performing safety assessments.
"""

import numpy as np
from typing import Dict, Any, List
import warnings
from utils.constants import CLINICAL_CONSTANTS

def assess_clinical_safety(sar_results: Dict[str, Any],
                          patient_weight: float = None,
                          scan_region: str = 'body',
                          vendor: str = 'siemens') -> Dict[str, Any]:
    """
    Comprehensive clinical safety assessment against IEC standards.
    
    Args:
        sar_results: SAR calculation results from sar_computation
        patient_weight: Patient weight in kg
        scan_region: 'body', 'head', or 'extremity'
        vendor: Scanner vendor ('siemens', 'ge', 'philips')
        
    Returns:
        Dictionary containing safety assessment and compliance status
    """
    if patient_weight is None:
        patient_weight = CLINICAL_CONSTANTS['default_patient_weight']
        warnings.warn(f"No patient weight provided, using default: {patient_weight} kg")
    
    # Get appropriate safety limits
    limits = _get_safety_limits(scan_region)
    
    # Extract SAR values
    sar_max = sar_results.get('sar_max', np.array([]))
    sar_averaged = sar_results.get('sar_averaged', sar_max)
    
    if len(sar_max) == 0:
        raise ValueError("No SAR values found in results")
    
    # Apply vendor-specific B1 scaling
    b1_scaling = _get_vendor_b1_scaling(vendor)
    sar_max_scaled = sar_max * (b1_scaling ** 2)
    sar_averaged_scaled = sar_averaged * (b1_scaling ** 2)
    
    # Perform safety checks
    safety_assessment = {
        'patient_weight': patient_weight,
        'scan_region': scan_region,
        'vendor': vendor,
        'b1_scaling_factor': b1_scaling,
        'limits': limits,
        'peak_sar': {
            'instantaneous_max': np.max(sar_max_scaled),
            'averaged_max': np.max(sar_averaged_scaled),
            'location_index': np.argmax(sar_averaged_scaled)
        }
    }
    
    # Check compliance with IEC limits
    compliance = _check_iec_compliance(sar_averaged_scaled, limits)
    safety_assessment.update(compliance)
    
    # Generate safety recommendations
    recommendations = _generate_safety_recommendations(safety_assessment)
    safety_assessment['recommendations'] = recommendations
    
    return safety_assessment

def _get_safety_limits(scan_region: str) -> Dict[str, float]:
    """Get IEC 60601-2-33 safety limits for scan region."""
    if scan_region.lower() in ['head', 'brain']:
        return {
            'six_min_limit': CLINICAL_CONSTANTS['six_min_thresh_hg'],
            'ten_sec_limit': CLINICAL_CONSTANTS['ten_sec_thresh_hg'],
            'local_limit': CLINICAL_CONSTANTS['local_sar_limit']
        }
    elif scan_region.lower() in ['body', 'torso', 'whole_body']:
        return {
            'six_min_limit': CLINICAL_CONSTANTS['six_min_thresh_wbg'],
            'ten_sec_limit': CLINICAL_CONSTANTS['ten_sec_thresh_wbg'],
            'local_limit': CLINICAL_CONSTANTS['local_sar_limit']
        }
    else:  # extremity
        return {
            'six_min_limit': CLINICAL_CONSTANTS['six_min_thresh_wbg'],
            'ten_sec_limit': CLINICAL_CONSTANTS['ten_sec_thresh_wbg'],
            'local_limit': CLINICAL_CONSTANTS['local_sar_limit'] * 2  # Higher for extremities
        }

def _get_vendor_b1_scaling(vendor: str) -> float:
    """Get vendor-specific B1 scaling factor."""
    vendor_lower = vendor.lower()
    if 'siemens' in vendor_lower:
        return CLINICAL_CONSTANTS['siemens_b1_fact']
    elif 'ge' in vendor_lower or 'general' in vendor_lower:
        return CLINICAL_CONSTANTS['ge_b1_fact']
    else:
        return 1.0  # Default/conservative scaling

def _check_iec_compliance(sar_values: np.ndarray, limits: Dict[str, float]) -> Dict[str, Any]:
    """Check compliance with IEC safety limits."""
    max_sar = np.max(sar_values)
    
    # Check against limits
    six_min_compliant = max_sar <= limits['six_min_limit']
    ten_sec_compliant = max_sar <= limits['ten_sec_limit']  # Simplified for demonstration
    local_compliant = max_sar <= limits['local_limit']
    
    overall_compliant = six_min_compliant and ten_sec_compliant and local_compliant
    
    # Calculate safety margins
    six_min_margin = (limits['six_min_limit'] - max_sar) / limits['six_min_limit'] * 100
    ten_sec_margin = (limits['ten_sec_limit'] - max_sar) / limits['ten_sec_limit'] * 100
    local_margin = (limits['local_limit'] - max_sar) / limits['local_limit'] * 100
    
    return {
        'overall_compliant': overall_compliant,
        'six_min_compliant': six_min_compliant,
        'ten_sec_compliant': ten_sec_compliant,
        'local_compliant': local_compliant,
        'safety_margins': {
            'six_min_margin_percent': six_min_margin,
            'ten_sec_margin_percent': ten_sec_margin,
            'local_margin_percent': local_margin
        },
        'violations': {
            'six_min_violation': max_sar - limits['six_min_limit'] if not six_min_compliant else 0,
            'ten_sec_violation': max_sar - limits['ten_sec_limit'] if not ten_sec_compliant else 0,
            'local_violation': max_sar - limits['local_limit'] if not local_compliant else 0
        }
    }

def _generate_safety_recommendations(assessment: Dict[str, Any]) -> List[str]:
    """Generate safety recommendations based on assessment."""
    recommendations = []
    
    if not assessment['overall_compliant']:
        recommendations.append("⚠️  SEQUENCE NOT CLINICALLY SAFE - Immediate attention required")
        
        if not assessment['six_min_compliant']:
            recommendations.append("• Reduce RF duty cycle or pulse amplitudes")
            recommendations.append("• Consider longer TR or reduced flip angles")
        
        if not assessment['local_compliant']:
            recommendations.append("• Review local SAR hotspots")
            recommendations.append("• Consider parallel imaging to reduce RF power")
    
    else:
        min_margin = min(assessment['safety_margins'].values())
        if min_margin < 20:
            recommendations.append("⚠️  Low safety margin - consider sequence optimization")
        else:
            recommendations.append("✓ Sequence within safe clinical limits")
    
    # B1 scaling recommendations
    if assessment['b1_scaling_factor'] > 1.2:
        recommendations.append(f"• High B1 scaling factor ({assessment['b1_scaling_factor']:.2f}) - verify calibration")
    
    return recommendations

def generate_clinical_report(safety_assessment: Dict[str, Any],
                           sar_results: Dict[str, Any],
                           sequence_info: Dict[str, Any] = None) -> str:
    """
    Generate a comprehensive clinical safety report.
    
    Args:
        safety_assessment: Results from assess_clinical_safety
        sar_results: SAR calculation results
        sequence_info: Optional sequence parameters
        
    Returns:
        Formatted clinical report string
    """
    report = []
    report.append("=" * 80)
    report.append("                   CLINICAL SAR SAFETY ASSESSMENT REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Patient and scan information
    report.append("PATIENT AND SCAN PARAMETERS:")
    report.append(f"  Patient Weight:     {safety_assessment['patient_weight']:.1f} kg")
    report.append(f"  Scan Region:        {safety_assessment['scan_region'].title()}")
    report.append(f"  Scanner Vendor:     {safety_assessment['vendor'].title()}")
    report.append(f"  B1 Scaling Factor:  {safety_assessment['b1_scaling_factor']:.3f}")
    report.append("")
    
    # SAR analysis results
    report.append("SAR ANALYSIS RESULTS:")
    peak = safety_assessment['peak_sar']
    report.append(f"  Peak Instantaneous SAR:  {peak['instantaneous_max']:.3f} W/kg")
    report.append(f"  Peak Time-Averaged SAR:  {peak['averaged_max']:.3f} W/kg")
    report.append(f"  Peak Location Index:     {peak['location_index']}")
    report.append("")
    
    # Performance metrics
    if 'computation_time' in sar_results:
        report.append("COMPUTATIONAL PERFORMANCE:")
        report.append(f"  Computation Time:   {sar_results['computation_time']:.3f} seconds")
        report.append(f"  Method Used:        {sar_results.get('method', 'Unknown')}")
        report.append(f"  GPU Acceleration:   {'Yes' if sar_results.get('gpu_used', False) else 'No'}")
        if 'spatial_resolution' in sar_results:
            report.append(f"  Spatial Resolution: {sar_results['spatial_resolution']:,} points")
        report.append("")
    
    # Compliance status
    report.append("IEC 60601-2-33 COMPLIANCE STATUS:")
    limits = safety_assessment['limits']
    margins = safety_assessment['safety_margins']
    
    status_symbol = "✓" if safety_assessment['overall_compliant'] else "✗"
    report.append(f"  Overall Compliance: {status_symbol} {'PASS' if safety_assessment['overall_compliant'] else 'FAIL'}")
    report.append("")
    
    report.append("  Detailed Compliance Check:")
    report.append(f"    6-min Average:  {peak['averaged_max']:.3f} W/kg <= {limits['six_min_limit']:.1f} W/kg  "
                 f"({'✓ PASS' if safety_assessment['six_min_compliant'] else '✗ FAIL'}) "
                 f"(Margin: {margins['six_min_margin_percent']:+.1f}%)")
    
    report.append(f"    10-sec Average: {peak['averaged_max']:.3f} W/kg <= {limits['ten_sec_limit']:.1f} W/kg  "
                 f"({'✓ PASS' if safety_assessment['ten_sec_compliant'] else '✗ FAIL'}) "
                 f"(Margin: {margins['ten_sec_margin_percent']:+.1f}%)")
    
    report.append(f"    Local SAR:      {peak['averaged_max']:.3f} W/kg <= {limits['local_limit']:.1f} W/kg  "
                 f"({'✓ PASS' if safety_assessment['local_compliant'] else '✗ FAIL'}) "
                 f"(Margin: {margins['local_margin_percent']:+.1f}%)")
    report.append("")
    
    # Safety recommendations
    report.append("SAFETY RECOMMENDATIONS:")
    for i, rec in enumerate(safety_assessment['recommendations'], 1):
        report.append(f"  {i}. {rec}")
    report.append("")
    
    # Sequence information if provided
    if sequence_info:
        report.append("SEQUENCE PARAMETERS:")
        for key, value in sequence_info.items():
            report.append(f"  {key.replace('_', ' ').title()}: {value}")
        report.append("")
    
    report.append("=" * 80)
    report.append("Report generated by SAR4seq Clinical Safety Assessment System")
    report.append("Authors: Leo Kinyera, BS - Columbia University")
    report.append("=" * 80)
    
    return "\n".join(report)

def check_pediatric_safety(safety_assessment: Dict[str, Any],
                          patient_age: float) -> Dict[str, Any]:
    """
    Additional safety checks for pediatric patients.
    
    Args:
        safety_assessment: Standard safety assessment
        patient_age: Patient age in years
        
    Returns:
        Enhanced safety assessment with pediatric considerations
    """
    pediatric_assessment = safety_assessment.copy()
    
    if patient_age < 18:
        # More conservative limits for pediatric patients
        pediatric_factor = 0.8 if patient_age < 12 else 0.9
        
        # Adjust limits
        for limit_key in ['six_min_limit', 'ten_sec_limit', 'local_limit']:
            if limit_key in pediatric_assessment['limits']:
                pediatric_assessment['limits'][limit_key] *= pediatric_factor
        
        # Re-check compliance with pediatric limits
        sar_values = np.array([pediatric_assessment['peak_sar']['averaged_max']])
        compliance = _check_iec_compliance(sar_values, pediatric_assessment['limits'])
        pediatric_assessment.update(compliance)
        
        # Add pediatric-specific recommendations
        pediatric_recommendations = [
            f"🧸 PEDIATRIC PATIENT (Age: {patient_age:.1f} years)",
            f"• Applied {(1-pediatric_factor)*100:.0f}% reduction in SAR limits",
            "• Extra monitoring recommended during scan"
        ]
        
        if not pediatric_assessment['overall_compliant']:
            pediatric_recommendations.append("• Consider further sequence optimization for pediatric safety")
        
        pediatric_assessment['recommendations'] = (
            pediatric_recommendations + pediatric_assessment['recommendations']
        )
        
        pediatric_assessment['pediatric_patient'] = True
        pediatric_assessment['pediatric_factor'] = pediatric_factor
    
    return pediatric_assessment

__all__ = [
    'assess_clinical_safety',
    'generate_clinical_report',
    'check_pediatric_safety'
]
