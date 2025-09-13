"""
TSE (Turbo Spin Echo) Sequence Writer

Creates TSE sequences using PyPulseq and exports them for scanner execution.
"""

import numpy as np
import pypulseq as pp


def write_TSE_50s(output_filename='tse_50s.seq'):
    """
    Create a TSE sequence and export for execution
    
    This function creates a Turbo Spin Echo sequence similar to the MATLAB version
    with the following specifications:
    - TR = 3.2s
    - TE_eff = 60ms  
    - Echo train length = 16
    - Matrix size = 128x256
    - FOV = 256mm
    - Slice thickness = 5mm
    - Total scan time = ~50s
    
    Parameters
    ----------
    output_filename : str
        Output filename for the sequence
        
    Returns
    -------
    pypulseq.Sequence
        The created sequence object
    """
    
    print("Creating TSE sequence with ~50s total scan time")
    
    # System gradient limits
    dG = 250e-6
    system = pp.Opts(
        max_grad=30, grad_unit='mT/m',
        max_slew=170, slew_unit='T/m/s',
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6
    )
    
    # Create sequence object
    seq = pp.Sequence(system)
    
    # Sequence parameters
    fov = 256e-3  # Field of view (m)
    Nx = 128      # Matrix size x
    Ny = 256      # Matrix size y (increased for longer scan)
    necho = 16    # Echo train length
    Nslices = 1   # Number of slices
    
    rf_flip = 120  # Flip angle (degrees)
    slice_thickness = 5e-3  # Slice thickness (m)
    TE = 12e-3    # Echo spacing (s)
    TR = 3.2      # Repetition time (s) - increased for ~50s total scan time
    TE_eff = 60e-3  # Effective echo time (s)
    
    # RF flip angles for echo train
    if isinstance(rf_flip, (int, float)):
        rf_flip = np.full(necho, rf_flip)
    
    # Calculate readout and phase encoding parameters
    delta_k = 1 / fov
    k_width = Nx * delta_k
    readout_time = 6.4e-3  # Readout duration
    
    # Create RF pulses
    # 90-degree excitation pulse
    rf_ex, gz_ex, _ = pp.make_sinc_pulse(
        flip_angle=90 * np.pi / 180,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True
    )
    
    # 180-degree refocusing pulses
    rf_ref, gz_ref, _ = pp.make_sinc_pulse(
        flip_angle=180 * np.pi / 180,
        duration=3e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True
    )
    
    # Create readout gradient
    gr_ro = pp.make_trapezoid(
        channel='x',
        flat_area=k_width,
        flat_time=readout_time,
        system=system
    )
    
    # ADC event
    adc = pp.make_adc(
        num_samples=Nx,
        duration=gr_ro.flat_time,
        delay=gr_ro.rise_time,
        system=system
    )
    
    # Phase encoding gradient
    phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
    
    # Slice select rephasing gradient
    gz_reph = pp.make_trapezoid(
        channel='z',
        area=-gz_ex.area / 2,
        duration=2e-3,
        system=system
    )
    
    # Calculate timing
    TE_delay = TE / 2 - pp.calc_duration(gz_ex) / 2 - pp.calc_duration(gz_ref) / 2
    TR_delay = TR - pp.calc_duration(gz_ex) - necho * TE - pp.calc_duration(gr_ro)
    
    if TE_delay < 0:
        raise ValueError("TE too short, increase TE or reduce RF/gradient durations")
    if TR_delay < 0:
        raise ValueError("TR too short, increase TR")
    
    # Create delay events
    delay_TE = pp.make_delay(TE_delay)
    delay_TR = pp.make_delay(TR_delay)
    
    print(f"Sequence timing:")
    print(f"  TE delay: {TE_delay*1000:.2f} ms")
    print(f"  TR delay: {TR_delay*1000:.2f} ms")
    print(f"  Total scan time: {TR * Ny / necho:.1f} s")
    
    # Build sequence
    for slice_idx in range(Nslices):
        for pe_group in range(0, Ny, necho):
            
            # Excitation
            seq.add_block(rf_ex, gz_ex)
            seq.add_block(gz_reph)
            seq.add_block(delay_TE)
            
            # Echo train
            for echo in range(necho):
                pe_idx = pe_group + echo
                if pe_idx >= Ny:
                    break
                
                # Phase encoding gradient
                gy_pe = pp.make_trapezoid(
                    channel='y',
                    area=phase_areas[pe_idx],
                    duration=2e-3,
                    system=system
                )
                
                # Pre-readout crusher
                gy_pre = pp.make_trapezoid(
                    channel='y',
                    area=-gy_pe.area / 2,
                    duration=1e-3,
                    system=system
                )
                
                # Readout prephasing
                gr_pre = pp.make_trapezoid(
                    channel='x',
                    area=-gr_ro.area / 2,
                    duration=1e-3,
                    system=system
                )
                
                # Post-readout crusher
                gy_post = pp.make_trapezoid(
                    channel='y',
                    area=-gy_pe.area / 2,
                    duration=1e-3,
                    system=system
                )
                
                # Refocusing pulse
                seq.add_block(rf_ref, gz_ref)
                
                # Pre-readout gradients
                seq.add_block(gr_pre, gy_pre)
                seq.add_block(delay_TE)
                
                # Readout
                seq.add_block(gr_ro, adc, gy_pe)
                
                # Post-readout
                seq.add_block(gy_post)
                
                if echo < necho - 1:
                    seq.add_block(delay_TE)
            
            # TR delay
            seq.add_block(delay_TR)
    
    # Check sequence timing and validity
    ok, error_report = seq.check_timing()
    if not ok:
        print("Sequence timing check failed!")
        print(error_report)
        return None
    
    # Calculate and display sequence statistics
    print("\nSequence Statistics:")
    print(f"  Number of blocks: {len(seq.block_events)}")
    print(f"  Total duration: {seq.duration()[0]:.3f} s")
    print(f"  Number of ADC samples: {seq.duration()[1]}")
    
    # Write sequence to file
    seq.write(output_filename)
    print(f"Sequence written to: {output_filename}")
    
    return seq


def write_custom_TSE(TR=500e-3, TE_eff=60e-3, necho=16, Nx=128, Ny=128, 
                     fov=256e-3, slice_thickness=5e-3, output_filename='custom_tse.seq'):
    """
    Create a customizable TSE sequence
    
    Parameters
    ----------
    TR : float
        Repetition time (s)
    TE_eff : float
        Effective echo time (s)
    necho : int
        Echo train length
    Nx, Ny : int
        Matrix dimensions
    fov : float
        Field of view (m)
    slice_thickness : float
        Slice thickness (m)
    output_filename : str
        Output filename
        
    Returns
    -------
    pypulseq.Sequence
        The created sequence object
    """
    
    print(f"Creating custom TSE sequence:")
    print(f"  TR = {TR*1000:.0f} ms")
    print(f"  TE_eff = {TE_eff*1000:.0f} ms")
    print(f"  Echo train length = {necho}")
    print(f"  Matrix = {Nx}x{Ny}")
    print(f"  FOV = {fov*1000:.0f} mm")
    
    # Calculate echo spacing
    TE = TE_eff / (necho // 2)
    
    # System limits
    system = pp.Opts(
        max_grad=30, grad_unit='mT/m',
        max_slew=170, slew_unit='T/m/s',
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6
    )
    
    seq = pp.Sequence(system)
    
    # Create RF pulses with optimized durations
    rf_ex, gz_ex, _ = pp.make_sinc_pulse(
        flip_angle=90 * np.pi / 180,
        duration=2.5e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True
    )
    
    rf_ref, gz_ref, _ = pp.make_sinc_pulse(
        flip_angle=180 * np.pi / 180,
        duration=2.5e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True
    )
    
    # Calculate k-space parameters
    delta_k = 1 / fov
    k_width = Nx * delta_k
    readout_time = 4e-3  # Shorter readout for custom sequence
    
    # Create readout gradient
    gr_ro = pp.make_trapezoid(
        channel='x',
        flat_area=k_width,
        flat_time=readout_time,
        system=system
    )
    
    # ADC
    adc = pp.make_adc(
        num_samples=Nx,
        duration=gr_ro.flat_time,
        delay=gr_ro.rise_time,
        system=system
    )
    
    # Phase encoding
    phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
    
    # Build sequence with simplified timing
    for pe_idx in range(0, Ny, necho):
        
        # Excitation
        seq.add_block(rf_ex, gz_ex)
        
        # Echo train
        for echo in range(necho):
            if pe_idx + echo >= Ny:
                break
                
            # Refocusing
            seq.add_block(rf_ref, gz_ref)
            
            # Phase encoding
            gy_pe = pp.make_trapezoid(
                channel='y',
                area=phase_areas[pe_idx + echo],
                duration=1.5e-3,
                system=system
            )
            
            # Readout prephasing
            gr_pre = pp.make_trapezoid(
                channel='x',
                area=-gr_ro.area / 2,
                duration=1e-3,
                system=system
            )
            
            seq.add_block(gr_pre, gy_pe)
            seq.add_block(gr_ro, adc)
    
    # Write sequence
    seq.write(output_filename)
    print(f"Custom TSE sequence written to: {output_filename}")
    
    return seq


if __name__ == "__main__":
    # Create default TSE sequence
    seq = write_TSE_50s()
    
    # Create custom TSE sequence
    custom_seq = write_custom_TSE(
        TR=1000e-3,
        TE_eff=80e-3,
        necho=8,
        Nx=256,
        Ny=256,
        output_filename='custom_tse_1s.seq'
    )
