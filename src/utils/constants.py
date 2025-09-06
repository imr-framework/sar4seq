"""
SAR Calculation Constants

Clinical constants and safety limits for SAR calculations.
"""

CLINICAL_CONSTANTS = {
    # Vendor-specific B1 scaling factors
    'siemens_b1_fact': 1.32,
    'ge_b1_fact': 1.1725,
    
    # Standard reference masses (kg)
    'wbody_weight': 103.45,
    'head_weight': 6.024,
    'default_patient_weight': 40.0,
    
    # SAR safety limits (W/kg) - IEC 60601-2-33
    'six_min_thresh_wbg': 4.0,    # Whole body global, 6-minute average
    'ten_sec_thresh_wbg': 8.0,    # Whole body global, 10-second average
    'six_min_thresh_hg': 3.2,     # Head global, 6-minute average  
    'ten_sec_thresh_hg': 6.4,     # Head global, 10-second average
    'local_sar_limit': 20.0,      # Local SAR limit (W/kg)
    
    # Spatial resolution settings
    'clinical_vop_points': 500,    # Standard VOP count for clinical use
    'clinical_spatial_points': 50000,  # Moderate resolution for clinical speed
    'research_spatial_points': 100000,  # High-resolution research points
}

# Tissue electromagnetic properties at 3T (128 MHz)
TISSUE_PROPERTIES = {
    # Air/background (conductivity ≈ 0, permittivity ≈ 1)
    0: {'conductivity': 0.0, 'permittivity': 1.0, 'density': 0.0, 'name': 'Air'},
    
    # Muscle (conductivity ≈ 0.5-0.8 S/m, permittivity ≈ 50-80)
    1: {'conductivity': 0.65, 'permittivity': 65.0, 'density': 1.05, 'name': 'Muscle'},
    
    # Fat (conductivity ≈ 0.04-0.08 S/m, permittivity ≈ 10-15)
    2: {'conductivity': 0.06, 'permittivity': 12.0, 'density': 0.92, 'name': 'Fat'},
    
    # Bone (conductivity ≈ 0.02-0.06 S/m, permittivity ≈ 10-20)
    3: {'conductivity': 0.04, 'permittivity': 15.0, 'density': 1.85, 'name': 'Bone'},
    
    # Brain (conductivity ≈ 0.4-0.7 S/m, permittivity ≈ 40-60)
    4: {'conductivity': 0.55, 'permittivity': 50.0, 'density': 1.04, 'name': 'Brain'},
    
    # Blood (conductivity ≈ 1.2-1.6 S/m, permittivity ≈ 60-80)
    5: {'conductivity': 1.4, 'permittivity': 70.0, 'density': 1.06, 'name': 'Blood'},
    
    # Lung (conductivity ≈ 0.2-0.4 S/m, permittivity ≈ 20-40)
    6: {'conductivity': 0.3, 'permittivity': 30.0, 'density': 0.5, 'name': 'Lung'},
    
    # Liver (conductivity ≈ 0.4-0.6 S/m, permittivity ≈ 40-60)
    7: {'conductivity': 0.5, 'permittivity': 50.0, 'density': 1.06, 'name': 'Liver'},
}
