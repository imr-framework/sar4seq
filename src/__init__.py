"""
SAR4seq - Comprehensive SAR Computation and Safety Assessment

A complete toolkit for MRI sequence SAR calculation, clinical safety assessment,
and GPU-accelerated computation.

Authors: Leo Kinyera, BS
Copyright: Board of Trustees of Columbia University in the City of New York
"""

# Utility modules
from .utils.sar_computation import (
    calculate_sar_uncompressed,
    calculate_sar_vop_compressed,
    calculate_time_averaged_sar
)

from .utils.clinical_assessment import (
    assess_clinical_safety,
    generate_clinical_report,
    check_pediatric_safety
)

from .utils.sequence_analysis import (
    parse_sequence_file,
    analyze_rf_characteristics,
    validate_sequence_for_sar
)

from .utils.gpu_utils import (
    check_gpu_availability,
    benchmark_gpu_performance,
    setup_gpu_computation
)

from .utils.read_qmat import (
    load_q_matrix_with_validation,
    validate_q_matrix,
    load_vop_compression_data
)

from utils.constants import CLINICAL_CONSTANTS, TISSUE_PROPERTIES

# Legacy import with error handling
try:
    from .sar4seq import SAR4seq
except ImportError:
    # Legacy module not available
    SAR4seq = None

# Version info
__version__ = "2.0.0"
__author__ = "Leo Kinyera, BS"
__email__ = "leokinyera81@gmail.com"

__all__ = [
    # Main interface
    'SAR4SeqProcessor',
    'main_sar_computation',
    
    # SAR computation
    'calculate_sar_uncompressed',
    'calculate_sar_vop_compressed',
    'calculate_time_averaged_sar',
    
    # Clinical assessment
    'assess_clinical_safety',
    'generate_clinical_report',
    'check_pediatric_safety',
    
    # Sequence analysis
    'parse_sequence_file',
    'analyze_rf_characteristics',
    'validate_sequence_for_sar',
    
    # GPU utilities
    'check_gpu_availability',
    'benchmark_gpu_performance',
    'setup_gpu_computation',
    
    # Data loading
    'load_q_matrix_with_validation',
    'validate_q_matrix',
    'load_vop_compression_data',
    
    # Constants
    'CLINICAL_CONSTANTS',
    'TISSUE_PROPERTIES',
    
    # Legacy
    'SAR4seq'
]