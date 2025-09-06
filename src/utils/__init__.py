"""
Utils package for SAR4seq

Contains utility functions for SAR calculations and Q-matrix operations.
"""

# Import new modular utilities (safe imports)
try:
    from .clinical_constants import CLINICAL_CONSTANTS, TISSUE_PROPERTIES
    from .sar_computation import (
        calculate_sar_uncompressed,
        calculate_sar_vop_compressed,
        calculate_time_averaged_sar
    )
    from .clinical_assessment import (
        assess_clinical_safety,
        generate_clinical_report,
        check_pediatric_safety
    )
    from .sequence_analysis import (
        parse_sequence_file,
        analyze_rf_characteristics,
        validate_sequence_for_sar
    )
    from .gpu_utils import (
        check_gpu_availability,
        benchmark_gpu_performance,
        setup_gpu_computation
    )
    from .read_qmat import (
        load_q_matrix_with_validation,
        validate_q_matrix,
        load_vop_compression_data
    )
    
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    NEW_MODULES_AVAILABLE = False
    print(f"Warning: Could not import new modules: {e}")

# Import legacy utilities with error handling
try:
    from .read_qmat import read_qmat
    from .write_qmat import write_qmat
    LEGACY_MODULES_AVAILABLE = True
except ImportError:
    LEGACY_MODULES_AVAILABLE = False

# Try importing other legacy modules (with error handling)
legacy_imports = [
    ('calc_sar', 'calc_SAR'),
    ('gen_qpwr', 'gen_Qpwr'),
    ('gen_e12ptq', 'gen_E12ptQ'),
    ('get_coremat', 'get_coremat'),
    ('get_emmodel', 'get_EMmodel'),
    ('chkcubair_global', 'chkcubair_global'),
    ('do_sw_sar', 'do_sw_sar')
]

available_legacy = []
for module_name, func_name in legacy_imports:
    try:
        # Use importlib for cleaner relative imports
        import importlib
        module = importlib.import_module(f'.{module_name}', package=__name__)
        globals()[func_name] = getattr(module, func_name)
        available_legacy.append(func_name)
    except (ImportError, AttributeError):
        pass  # Skip modules that can't be imported

# Define exports based on what's available
__all__ = []

if NEW_MODULES_AVAILABLE:
    __all__.extend([
        'CLINICAL_CONSTANTS', 'TISSUE_PROPERTIES',
        'calculate_sar_uncompressed', 'calculate_sar_vop_compressed', 'calculate_time_averaged_sar',
        'assess_clinical_safety', 'generate_clinical_report', 'check_pediatric_safety',
        'parse_sequence_file', 'analyze_rf_characteristics', 'validate_sequence_for_sar',
        'check_gpu_availability', 'benchmark_gpu_performance', 'setup_gpu_computation',
        'load_q_matrix_with_validation', 'validate_q_matrix', 'load_vop_compression_data'
    ])

if LEGACY_MODULES_AVAILABLE:
    __all__.extend(['read_qmat', 'write_qmat'])

__all__.extend(available_legacy)
