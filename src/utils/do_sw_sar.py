"""
SAR Switching Utilities

Functions for handling SAR calculations with switching conditions.
"""

import numpy as np
import time


def do_sw_sar(sar_values, time_vector, limits, window_times):
    """
    Perform SAR checking with sliding window averaging
    
    Parameters
    ----------
    sar_values : numpy.ndarray
        Array of instantaneous SAR values
    time_vector : numpy.ndarray
        Time points corresponding to SAR values
    limits : dict
        SAR limits with keys like 'six_min_wbg', 'ten_sec_wbg', etc.
    window_times : list
        Time windows for averaging (e.g., [10, 360] for 10s and 6min)
        
    Returns
    -------
    dict
        Results including violations and time-averaged values
    """
    
    results = {
        'violations': [],
        'max_sar': {},
        'time_averaged': {},
        'compliance': True
    }
    
    # Check each time window
    for window_time in window_times:
        window_name = f"{window_time}s"
        
        # Calculate sliding window average
        window_sar = sliding_window_average(sar_values, time_vector, window_time)
        max_window_sar = np.max(window_sar) if len(window_sar) > 0 else 0
        
        results['max_sar'][window_name] = max_window_sar
        results['time_averaged'][window_name] = np.mean(window_sar) if len(window_sar) > 0 else 0
        
        # Check against limits
        if window_time == 10:  # 10 second limit
            if 'ten_sec_wbg' in limits and max_window_sar > limits['ten_sec_wbg']:
                results['violations'].append({
                    'type': 'ten_sec_wholebody',
                    'value': max_window_sar,
                    'limit': limits['ten_sec_wbg'],
                    'window': window_time
                })
                results['compliance'] = False
            
            if 'ten_sec_hg' in limits and max_window_sar > limits['ten_sec_hg']:
                results['violations'].append({
                    'type': 'ten_sec_head',
                    'value': max_window_sar,
                    'limit': limits['ten_sec_hg'],
                    'window': window_time
                })
                results['compliance'] = False
                
        elif window_time == 360:  # 6 minute limit
            if 'six_min_wbg' in limits and max_window_sar > limits['six_min_wbg']:
                results['violations'].append({
                    'type': 'six_min_wholebody',
                    'value': max_window_sar,
                    'limit': limits['six_min_wbg'],
                    'window': window_time
                })
                results['compliance'] = False
            
            if 'six_min_hg' in limits and max_window_sar > limits['six_min_hg']:
                results['violations'].append({
                    'type': 'six_min_head',
                    'value': max_window_sar,
                    'limit': limits['six_min_hg'],
                    'window': window_time
                })
                results['compliance'] = False
    
    return results


def sliding_window_average(values, time_points, window_duration):
    """
    Calculate sliding window average of SAR values
    
    Parameters
    ----------
    values : numpy.ndarray
        SAR values
    time_points : numpy.ndarray
        Time points for each SAR value
    window_duration : float
        Duration of averaging window in seconds
        
    Returns
    -------
    numpy.ndarray
        Time-averaged SAR values
    """
    
    if len(values) == 0 or len(time_points) == 0:
        return np.array([])
    
    # Ensure same length
    min_length = min(len(values), len(time_points))
    values = values[:min_length]
    time_points = time_points[:min_length]
    
    # Calculate window averages
    window_averages = []
    
    for i, t_end in enumerate(time_points):
        t_start = t_end - window_duration
        
        # Find indices within window
        window_mask = (time_points >= t_start) & (time_points <= t_end)
        window_values = values[window_mask]
        window_times = time_points[window_mask]
        
        if len(window_values) > 0:
            # Time-weighted average
            if len(window_times) > 1:
                dt = np.diff(window_times)
                dt = np.append(dt, dt[-1])  # Assume last interval same as previous
                time_weighted_avg = np.sum(window_values * dt) / np.sum(dt)
            else:
                time_weighted_avg = window_values[0]
            
            window_averages.append(time_weighted_avg)
        else:
            window_averages.append(0.0)
    
    return np.array(window_averages)


def check_iec_compliance(sar_values, sar_type='wholebody'):
    """
    Check compliance with IEC 60601-2-33 SAR limits
    
    Parameters
    ----------
    sar_values : dict
        Dictionary with SAR values for different time windows
    sar_type : str
        Type of SAR ('wholebody', 'head', 'extremity')
        
    Returns
    -------
    dict
        Compliance results
    """
    
    # IEC limits (W/kg)
    iec_limits = {
        'wholebody': {
            '10s': 4.0,    # 10 second limit
            '6min': 2.0    # 6 minute limit
        },
        'head': {
            '10s': 10.0,   # 10 second limit
            '6min': 3.2    # 6 minute limit
        },
        'extremity': {
            '10s': 20.0,   # 10 second limit
            '6min': 10.0   # 6 minute limit
        }
    }
    
    if sar_type not in iec_limits:
        raise ValueError(f"Unknown SAR type: {sar_type}")
    
    limits = iec_limits[sar_type]
    compliance = {
        'compliant': True,
        'violations': [],
        'margins': {}
    }
    
    for window, limit in limits.items():
        if window in sar_values:
            sar_value = sar_values[window]
            margin = limit - sar_value
            compliance['margins'][window] = margin
            
            if sar_value > limit:
                compliance['compliant'] = False
                compliance['violations'].append({
                    'window': window,
                    'value': sar_value,
                    'limit': limit,
                    'excess': sar_value - limit
                })
    
    return compliance


def estimate_safe_tr(current_sar, current_tr, sar_limit, safety_factor=0.9):
    """
    Estimate safe TR to meet SAR limits
    
    Parameters
    ----------
    current_sar : float
        Current SAR value (W/kg)
    current_tr : float
        Current TR (seconds)
    sar_limit : float
        Target SAR limit (W/kg)
    safety_factor : float
        Safety factor (0.9 = 10% margin)
        
    Returns
    -------
    float
        Suggested TR (seconds)
    """
    
    if current_sar <= 0:
        return current_tr
    
    # SAR is roughly proportional to 1/TR for same pulse energy
    target_sar = sar_limit * safety_factor
    tr_scaling = current_sar / target_sar
    
    suggested_tr = current_tr * tr_scaling
    
    return suggested_tr


def sar_monitoring_report(sar_history, time_history, limits):
    """
    Generate SAR monitoring report
    
    Parameters
    ----------
    sar_history : numpy.ndarray
        Historical SAR values
    time_history : numpy.ndarray
        Time points
    limits : dict
        SAR limits
        
    Returns
    -------
    dict
        Monitoring report
    """
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': time_history[-1] - time_history[0] if len(time_history) > 1 else 0,
        'total_pulses': len(sar_history),
        'statistics': {
            'max_sar': np.max(sar_history) if len(sar_history) > 0 else 0,
            'mean_sar': np.mean(sar_history) if len(sar_history) > 0 else 0,
            'std_sar': np.std(sar_history) if len(sar_history) > 0 else 0
        },
        'compliance': True,
        'warnings': []
    }
    
    # Check compliance
    max_sar = report['statistics']['max_sar']
    
    for limit_name, limit_value in limits.items():
        if max_sar > limit_value:
            report['compliance'] = False
            report['warnings'].append({
                'type': 'limit_exceeded',
                'limit': limit_name,
                'value': max_sar,
                'threshold': limit_value
            })
        elif max_sar > 0.8 * limit_value:  # Warning at 80%
            report['warnings'].append({
                'type': 'approaching_limit',
                'limit': limit_name,
                'value': max_sar,
                'threshold': limit_value,
                'percentage': (max_sar / limit_value) * 100
            })
    
    return report
