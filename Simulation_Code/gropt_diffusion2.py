import numpy as np
from time import perf_counter as timer
import gropt
from safe_vec_generatory import *

import numpy as np
from time import perf_counter as timer
import gropt
from safe_vec_generatory import *
from scipy.interpolate import interp1d

def match_length(vec, N):
    """Resample vector to length N for gropt."""
    x_old = np.linspace(0, 1, len(vec))
    x_new = np.linspace(0, 1, N)
    return interp1d(x_old, vec, kind='linear')(x_new)


def optimize_diffusion_waveform_minTE(
        T_90, T_180, T_readout,
        TE_min, TE_max, dt, 
        b_target, 
        smax=200,
        T_180_center=2.5e-3,
        pns_max=0.75, 
        cns_max=0.75, 
        gmax=0.2,
        custom_vec=False,
        safe_params=None, safe_params_cardiac=None,
        tol=1e-6, max_iter=200, mmt=0, TE_start=None,
        timings_path=None, waveforms_path=None,
        iteration_logging=False):
    """
    Find minimum TE for a waveform with specified b-value while enforcing SAFE/CNS.
    """
    print('Optimizing diffusion waveform for min TE between {:.2f} ms and {:.2f} ms'.format(TE_min*1e3, TE_max*1e3))

    best_result = None
    best_TE = None
    best_bval = None
    best_solve_time = None
    gparams_final = None
    stim_vec_pns_final = None
    stim_vec_cns_final = None

    # Compute initial low/high bounds
    low = max([
        np.ceil((T_90 + T_180_center) / dt) * dt,
        np.ceil((T_180 - T_180_center + T_readout) / dt) * dt
    ]) * 2
    high = TE_max
    iter_count = 0
    trial_log = {}

    while high - low > tol and iter_count < max_iter:
        TE = (low + high) / 2
        if TE_start and iter_count == 0:
            TE = TE_start

        iter_count += 1

        # Initialize gradient parameters
        gparams = gropt.GroptParams()
        gparams.diff_init(T_90=T_90, T_180=T_180, T_readout=T_readout, TE=TE, dt=dt)

        # --- Prepare stimulus vectors ---
        if not custom_vec:
            stim_vec_pns = [(pns_max if pns_max is not None else 1e12) * np.ones(gparams.N) for _ in range(3)]
            stim_vec_cns = [(cns_max if cns_max is not None else 1e12) * np.ones(gparams.N) for _ in range(3)]
        else:
            seq = PNSCNS_SequenceBuilder(
                timing_file=timings_path,
                waveform_file=waveforms_path,
                dt_in=1e-5,
                dt_out=dt,
                safe_params=safe_params,
                safe_params_cardiac=safe_params_cardiac,
                TE=TE,
                n_repeats=5
            )

            pns_x = (1 - seq.safe_gx_out) * pns_max
            pns_y = (1 - seq.safe_gy_out) * pns_max
            pns_z = (1 - seq.safe_gz_out) * pns_max
            stim_vec_pns = [pns_x, pns_y, pns_z]

            if safe_params_cardiac is not None:
                cns_x = (1 - seq.safe_cardiac_gx_out) * cns_max
                cns_y = (1 - seq.safe_cardiac_gy_out) * cns_max
                cns_z = (1 - seq.safe_cardiac_gz_out) * cns_max
                stim_vec_cns = [cns_x, cns_y, cns_z]

        # --- Add hardware & b-value constraints ---
        gparams.add_gmax(gmax)
        gparams.add_smax(smax)
        gparams.add_moment(0, 0.0)
        if mmt >= 1:
            gparams.add_moment(1, 0.0)
        if mmt >= 2:
            gparams.add_moment(2, 0.0)
        if mmt >=3:
            gparams.add_moment(3, 0.0)
        if mmt >=4:
            gparams.add_moment(4, 0.0)
        gparams.add_bvalue(b_target, mode='setval')

        # --- Add SAFE/CNS constraints ---
        if safe_params is not None:
            for axis in range(3):
                gparams.add_SAFE_vec(match_length(stim_vec_pns[axis], gparams.N),
                                     safe_params=safe_params,
                                     new_first_axis=axis)
        if safe_params_cardiac is not None:
            for axis in range(3):
                gparams.add_SAFE_vec(match_length(stim_vec_cns[axis], gparams.N),
                                     safe_params=safe_params_cardiac,
                                     new_first_axis=axis)

        # Solve waveform
        start_t = timer()
        solver = gropt.SolverGroptSDMM()
        solver.set_general_params(min_iter=6000, max_iter=10000, max_feval=100000)
        solver.solve(gparams)
        stop_t = timer()

        out = gparams.get_out()
        final_good = gparams.final_good > 0
        bval = gparams.get_output_bvalue()
        solve_time_ms = 1000 * (stop_t - start_t)

        print(f"Iter {iter_count}: TE={TE*1e3:.2f} ms | Final Good={final_good} | b={bval:.2f} | Solve={solve_time_ms:.2f} ms")

        if iteration_logging:
            trial_log[f"trial_{iter_count}"] = {
                "TE": TE,
                "bval": bval,
                "final_good": final_good,
                "solve_time_ms": solve_time_ms,
                "out": out.copy(),
                "stim_vec_pns": stim_vec_pns,
                "stim_vec_cns": stim_vec_cns,
                "gparams": gparams
            }

        # Update TE bounds
        if final_good:
            best_result = out.copy()
            best_TE = TE
            best_bval = bval
            best_solve_time = solve_time_ms
            gparams_final = gparams
            stim_vec_pns_final = stim_vec_pns
            stim_vec_cns_final = stim_vec_cns
            high = TE  # try smaller TE
        else:
            low = TE  # TE too short, increase

    if best_result is None:
        return None, None, None, False, None, None, None, None, trial_log
    else:
        return (best_TE, best_result, best_bval, True,
                best_solve_time, gparams_final, stim_vec_pns_final, stim_vec_cns_final, trial_log)

import numpy as np
from time import perf_counter as timer
import gropt

def make_gropt_waveform_for_TE(T_90, T_180, T_readout, TE, dt, b_target,
                               smax=200, gmax=0.2,
                               pns_max=0.75, cns_max=0.75, 
                               safe_params=None, safe_params_cardiac=None,
                               mmt=0, use_cns = True,custom_vec = False, timings_path = None, waveforms_path = None):
    """
    Generate a GRopt diffusion gradient waveform for a given TE.
    
    Returns:
        TE: actual TE used
        waveform: numpy array of gradient waveform
        bval: resulting b-value
        success: True/False if solver converged
        solve_time_ms: time in ms
        gparams: gropt.GroptParams object
        stim_vec_pns: PNS vector
        stim_vec_cns: CNS vector
    """

    # Initialize gradient parameters
    gparams = gropt.GroptParams()
    gparams.diff_init(T_90=T_90, T_180=T_180, T_readout=T_readout, TE=TE, dt=dt)

    # Stim vectors
    stim_vec_pns = (pns_max if pns_max is not None else 1e12) * np.ones(gparams.N)
    stim_vec_cns = (cns_max if cns_max is not None else 1e12) * np.ones(gparams.N)

     # Stim vectors
    if not custom_vec:
        stim_vec_pns = (pns_max if pns_max is not None else 1e12) * np.ones(gparams.N)
        stim_vec_cns = (cns_max if cns_max is not None else 1e12) * np.ones(gparams.N)
    
    if custom_vec:
        print('----- Custom Vec Mode -----')
        
        seq = PNSCNS_SequenceBuilder(
            timing_file=timings_path,
            waveform_file=waveforms_path,
            dt_in=1e-5,
            dt_out=dt,
            safe_params=safe_params,
            safe_params_cardiac=safe_params_cardiac,
            TE=TE,
            n_repeats=5
        )

        # Access outputs
        pns_x = (1-seq.safe_gx_out) * pns_max
        pns_y = (1-seq.safe_gy_out) * pns_max
        pns_z = (1-seq.safe_gz_out) * pns_max

        stim_vec_pns = [pns_x, pns_y,pns_z]

        #print(np.nanmax(stim_vec_pns[0]), np.nanmax(stim_vec_pns[1]), np.nanmax(stim_vec_pns[2]) )

        if safe_params_cardiac is not None:
            cns_x  = (1-seq.safe_cardiac_gx_out) * cns_max
            cns_y  = (1-seq.safe_cardiac_gy_out) * cns_max
            cns_z  = (1-seq.safe_cardiac_gz_out) * cns_max

            stim_vec_cns = [cns_x, cns_y,cns_z]
            #print(np.nanmax(stim_vec_cns[0]), np.nanmax(stim_vec_cns[1]), np.nanmax(stim_vec_cns[2]) )

        ## Plot
        #seq.plot_all()

        print(gparams.N, cns_x.shape, pns_x.shape)

    
    

    # Hardware & b-value constraints
    gparams.add_gmax(gmax)
    gparams.add_smax(smax)
    gparams.add_moment(0, 0.0)
    if mmt >= 1:
        gparams.add_moment(1, 0.0)
    if mmt >= 2:
        gparams.add_moment(2, 0.0)
    gparams.add_bvalue(b_target, mode='setval')

    # SAFE constraints
    if safe_params is not None:
        for axis in range(3):
            gparams.add_SAFE_vec(stim_vec_pns, safe_params=safe_params, new_first_axis=axis)
    if safe_params_cardiac is not None and use_cns is not False:
        for axis in range(3):
            gparams.add_SAFE_vec(stim_vec_cns, safe_params=safe_params_cardiac, new_first_axis=axis)

    # Solve
    start_t = timer()
    solver = gropt.SolverGroptSDMM()
    solver.set_general_params(min_iter=6000, max_iter=10000, max_feval=100000)
    solver.solve(gparams)
    stop_t = timer()

    out = gparams.get_out()
    final_good = gparams.final_good > 0
    bval = gparams.get_output_bvalue()
    solve_time_ms = 1000 * (stop_t - start_t)

    print(f"TE={TE*1e3:.2f} ms | Final Good={final_good} | b={bval:.2f} | Solve={solve_time_ms:.2f} ms")

    return TE, out, bval, final_good, solve_time_ms, gparams, stim_vec_pns, stim_vec_cns




import numpy as np


def make_stimulus_vector_dual(
    gparams, T_90, T_180, T_180_center, TE, T_readout,
    base_val=1.0,
    min_val_180=0.4,
    min_val_90=0.4,
    min_val_end=0.4,
    transition_frac_90=0.1,
    transition_frac_180=0.1,
    end_dip_frac=0.12
):
    """
    SAFE stimulus vector with smooth dips and robust handling for short TE cases.
    """
    
    N = gparams.N
    t = np.linspace(0, TE - T_readout, N)
    stim = np.ones_like(t) * base_val

    # --- Dip 1: T90 ---
    mask1 = t <= T_90
    stim[mask1] = min_val_90

    # --- Smooth transition out of T90 ---
    trans_len_90 = max(2, int(transition_frac_90 * N))
    trans_dur_90 = trans_len_90 * (TE - T_readout) / N
    trans_idx_90 = np.where((t > T_90) & (t <= T_90 + trans_dur_90))[0]
    if len(trans_idx_90) > 0:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(trans_idx_90))))
        stim[trans_idx_90] = min_val_90 + (base_val - min_val_90) * w

    # --- Dip 2: 180° pulse ---
    dip2_start = TE/2 - T_180_center
    dip2_end   = TE/2 + (T_180 - T_180_center)
    mask2 = (t >= dip2_start) & (t <= dip2_end)
    stim[mask2] = min_val_180

    # --- Smooth transitions around 180° dip ---
    trans_len_180 = max(2, int(transition_frac_180 * N))
    trans_dur_180 = trans_len_180 * (TE - T_readout) / N

    # Pre-180 transition
    pre_idx = np.where((t >= dip2_start - trans_dur_180) & (t < dip2_start))[0]
    if len(pre_idx) > 0:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(pre_idx))))
        stim[pre_idx] = min_val_180 + (base_val - min_val_180) * (1 - w)

    # Post-180 transition
    post_idx = np.where((t > dip2_end) & (t <= dip2_end + trans_dur_180))[0]
    if len(post_idx) > 0:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(post_idx))))
        stim[post_idx] = min_val_180 + (base_val - min_val_180) * w

    # --- Smooth ramp from end of post-180 to final end dip ---
    if len(post_idx) > 0:
        ramp_start_idx = min(post_idx[-1] + 1, N - 1)
    else:
        ramp_start_idx = min(np.searchsorted(t, dip2_end), N - 1)
    ramp_end_idx = N
    ramp_len = max(0, ramp_end_idx - ramp_start_idx)

    if ramp_len > 1:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))
        stim[ramp_start_idx:ramp_end_idx] = stim[ramp_start_idx] * (1 - w) + min_val_end * w
    else:
        stim[-1] = min_val_end  # Always ensure final point set

    # --- Always enforce last few samples approach min_val_end ---
    tail_len = max(2, int(end_dip_frac * N))
    stim[-tail_len:] = np.linspace(stim[-tail_len], min_val_end, tail_len)

    return stim



def make_stimulus_vector_dual_old(
    gparams, T_90, T_180, T_180_center, TE, T_readout,
    base_val=1.0,
    min_val_180=0.4,
    min_val_90=0.4,
    min_val_end=0.4,
    transition_frac_90=0.1,
    transition_frac_180=0.1,
    end_dip_frac=0.12
):
    """
    Create a SAFE stimulus vector with smooth dips:
      - Flat min_val_90 during T_90 (start)
      - Flat min_val_180 during 180° pulse window around TE/2
      - Smooth transitions (cosine) between base and dip levels
      - Smooth ramp to min_val_end at the vector end

    Parameters
    ----------
    gparams : object
        Must contain .N (number of samples)
    T_90 : float
        Duration of 90° pulse (s)
    T_180 : float
        Duration of 180° pulse (s)
    T_180_center : float
        Offset of 180° pulse center relative to TE/2 (s)
    TE : float
        Echo time (s)
    T_readout : float
        Duration of readout (vector ends before this)
    base_val : float
        Maximum threshold value
    min_val_180 : float
        Minimum value during 180° pulse
    min_val_90 : float
        Minimum value during 90° pulse
    min_val_end : float
        Minimum value at the end dip
    transition_frac_90 : float
        Fraction of vector for transition after T90
    transition_frac_180 : float
        Fraction of vector for transitions around 180° dip
    end_dip_frac : float
        Fraction of vector for final end dip
    """
    N = gparams.N
    t = np.linspace(0, TE - T_readout, N)
    stim = np.ones_like(t) * base_val

    # --- Dip 1: T90 ---
    mask1 = t <= T_90
    stim[mask1] = min_val_90

    # --- Smooth transition out of T90 ---
    trans_len_90 = max(2, int(transition_frac_90 * N))
    trans_dur_90 = trans_len_90 * (TE - T_readout) / N
    trans_idx_90 = np.where((t > T_90) & (t <= T_90 + trans_dur_90))[0]
    if len(trans_idx_90) > 0:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(trans_idx_90))))
        stim[trans_idx_90] = min_val_90 + (base_val - min_val_90) * w

    # --- Dip 2: 180° pulse ---
    dip2_start = TE/2 - T_180_center
    dip2_end   = TE/2 + (T_180 - T_180_center)
    mask2 = (t >= dip2_start) & (t <= dip2_end)
    stim[mask2] = min_val_180

    # --- Smooth transitions around 180° dip ---
    trans_len_180 = max(2, int(transition_frac_180 * N))
    trans_dur_180 = trans_len_180 * (TE - T_readout) / N

    # Pre-180 transition
    pre_idx = np.where((t >= dip2_start - trans_dur_180) & (t < dip2_start))[0]
    if len(pre_idx) > 0:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(pre_idx))))
        stim[pre_idx] = min_val_180 + (base_val - min_val_180) * (1 - w)

    # Post-180 transition
    post_idx = np.where((t > dip2_end) & (t <= dip2_end + trans_dur_180))[0]
    if len(post_idx) > 0:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, len(post_idx))))
        stim[post_idx] = min_val_180 + (base_val - min_val_180) * w

    # --- Smooth ramp from end of post-180 to final end dip ---
    ramp_start_idx = post_idx[-1] + 1 if len(post_idx) > 0 else np.searchsorted(t, dip2_end)
    ramp_end_idx = N
    ramp_len = ramp_end_idx - ramp_start_idx
    if ramp_len > 1:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))
        stim[ramp_start_idx:ramp_end_idx] = stim[ramp_start_idx] * (1 - w) + min_val_end * w
    elif ramp_len == 1:
        stim[-1] = min_val_end

    return stim



def smooth_transition(start_val, end_val, n):
    if n <= 1:
        return np.full(n, end_val)
    x = np.linspace(-3, 3, n)  # smooth sigmoid over [-3,3]
    y = 1/(1 + np.exp(-x))      # 0 → 1
    return start_val + (end_val - start_val)*y



def adjust_stimulus_vector_keep_pulses_piecewise(
    stim_orig, dt_old, N_new, 
    T_90, T_180, T_180_center, TE_old, T_readout,
    TE_new=None, stim_dic = None, T_readout_init = None
):
    """
    Piecewise resample stim_orig -> length N_new while keeping 90° and 180°
    pulse shapes unchanged and placing the 180° center so that:
        center_180 + T_180_center = TE_new/2
    Only the delays between pulses and after the 180° are stretched/compressed.

    Parameters
    ----------
    stim_orig : array_like, length N_old
    dt_old : float
        sample period of stim_orig (s)
    N_new : int
        desired output length
    T_90, T_180, T_180_center : floats (s)
    TE_old : float (s)  -- original TE (full TE, not TE-T_readout)
    T_readout : float (s)
    TE_new : float (s) or None
        new desired TE. If None, TE_new is inferred from N_new*dt_old + T_readout.

    Returns
    -------
    stim_new, tt_new, info
    """
    
    
    # ======================================================
    # Interpolate stim_orig if original dt differs
    # ======================================================
    stim_orig = np.asarray(stim_orig)
    if T_readout_init is None:
        T_readout_init = T_readout

        
    if stim_dic is not None and 'dt' in stim_dic:
        dt_src = stim_dic['dt']
        if not np.isclose(dt_src, dt_old):
            # -- Create time axis for original waveform
            tt_src = np.arange(0, len(stim_orig) * dt_src, dt_src)
            # Match the intended total duration (to TE_old - T_readout)
            total_old = TE_old - T_readout_init
            # Clip or pad if time axes mismatch slightly
            if tt_src[-1] < total_old:
                tt_src = np.append(tt_src, total_old)
                stim_orig = np.append(stim_orig, stim_orig[-1])
            elif tt_src[-1] > total_old:
                mask = tt_src <= total_old
                tt_src = tt_src[mask]
                stim_orig = stim_orig[:len(tt_src)]

            # -- Interpolate onto new uniform dt_old grid
            tt_uniform = np.arange(0, total_old, dt_old)
            stim_interp = np.interp(tt_uniform, tt_src, stim_orig)

            stim_orig = stim_interp
            N_old = len(stim_orig)
            tt_old = tt_uniform
        else:
            # no resampling needed
            N_old = len(stim_orig)
            total_old = TE_old - T_readout_init
            tt_old = np.linspace(0, total_old, N_old, endpoint=False)
    else:
        # no stim_dic provided
        N_old = len(stim_orig)
        total_old = TE_old - T_readout_init
        tt_old = np.linspace(0, total_old, N_old, endpoint=False)


    # ======================================================
    # compute new total and time vector
    # ======================================================
    
    total_new = N_new * dt_old
    if TE_new is None:
        TE_new = total_new + T_readout
    else:
        # ensure consistency between TE_new and N_new*dt_old:
        # user-provided TE_new takes precedence for locating the 180, but we still
        # set total_new to N_new*dt_old so output vector length matches N_new.
        total_new = N_new * dt_old

    tt_new = np.linspace(0, total_new, N_new, endpoint=False)

    # ======================================================
    # --- compute knot times (old and new) ---
    # ======================================================
    # end of 90 pulse
    t90_end_old = T_90
    t90_end_new = T_90  # we keep it unchanged (absolute time)

    # 180 start/end times (relative to full TE)
    t180_center_old = TE_old / 2.0 
    t180_start_old = t180_center_old - T_180_center
    t180_end_old = t180_start_old + T_180

    t180_center_new = TE_new / 2.0 
    t180_start_new = t180_center_new - T_180_center
    t180_end_new = t180_start_new + T_180

    # ======================================================
    # clamp starts/ends into [0, total_*] (If pulses would lie outside the [0, total] range, clamp but keep durations)
    # ======================================================
    def clamp_pair(start, end, total):
        s = max(0.0, start)
        e = min(total, end)
        # if clamped smaller than pulse duration, we still keep the requested start/end
        return s, e

    # But we will rely on mapping even if old/new pulses go slightly beyond - use unclamped
    # However we must compute valid durations for delay segments (can't be negative)
    dur_delay1_old = max(0.0, t180_start_old - t90_end_old)   # between 90 and 180 (old)
    dur_delay2_old = max(0.0, total_old - t180_end_old)       # after 180 (old)

    # For new we compute exactly from TE_new
    dur_delay1_new = max(0.0, t180_start_new - t90_end_new)
    dur_delay2_new = max(0.0, total_new - t180_end_new)

    # --- Make knot vectors (new timeline knots -> old timeline knots) ---
    # Old knots (in time of the original vector)
    old_knots = np.array([0.0,
                          t90_end_old,
                          t180_start_old,
                          t180_end_old,
                          total_old], dtype=float)

    # New knots (desired positions in the new timeline)
    new_knots = np.array([0.0,
                          t90_end_new,
                          t180_start_new,
                          t180_end_new,
                          total_new], dtype=float)

    # ======================================================
    # If any old_knot is outside old time range, clamp them into [0, total_old]
    # This avoids producing NaNs when mapping.
    # ======================================================
    old_knots_clamped = np.clip(old_knots, 0.0, total_old)

    # Ensure monotonic non-decreasing knots for interpolation (tiny eps to enforce strictness)
    eps = 1e-12
    for arr in (old_knots_clamped, new_knots):
        for i in range(1, len(arr)):
            if arr[i] <= arr[i-1]:
                arr[i] = arr[i-1] + eps

    # --- Map each new time sample to an old time via piecewise linear mapping ---
    # Invert mapping new_knots -> old_knots_clamped
    tt_map_old = np.interp(tt_new, new_knots, old_knots_clamped)

    #safe_map_old = np.interp(tt_new, new_knots, old_knots_clamped)
    # ======================================================
    # Interpolate stim_orig at mapped times (outside old range: use edge values)
    # np.interp requires x to be within tt_old range; provide left/right fill via edge values
    # So use np.interp with tt_old and stim_orig.
    # ======================================================
    stim_new = np.interp(tt_map_old, tt_old, stim_orig, left=stim_orig[0], right=stim_orig[-1])

    info = dict(
        N_old=N_old,
        N_new=N_new,
        dt_old=dt_old,
        total_old=total_old,
        total_new=total_new,
        t90_end_old=t90_end_old,
        t180_start_old=t180_start_old,
        t180_end_old=t180_end_old,
        t180_start_new=t180_start_new,
        t180_end_new=t180_end_new,
        dur_delay1_old=dur_delay1_old,
        dur_delay2_old=dur_delay2_old,
        dur_delay1_new=dur_delay1_new,
        dur_delay2_new=dur_delay2_new,
        old_knots=old_knots,
        new_knots=new_knots
    )

    return stim_new, tt_new, info





def M2_helper(T, r, delay):
    # T is time of the right (smaller) time range, r is ramp time, and delay is the time between the two waveforms
    # The time between the two will be T180 + T1 - T2 (assuming T=T2)
    f1a = T + delay/2 - 2*r - np.sqrt(2*T**2 + 2*T*delay - 2*T*r + delay**2)/2
    f1b = T + delay/2 - 2*r + np.sqrt(2*T**2 + 2*T*delay - 2*T*r + delay**2)/2
    
    f2a = T - f1a - 4*r
    f2b = T - f1b - 4*r
    
    # print(f"f1a: {f1a}, f2a: {f2a}")
    # print(f"f1b: {f1b}, f2b: {f2b}")

    if f1a > 0 and f2a > 0:
        return f1a, f2a
    elif f1b > 0 and f2b > 0:
        return f1b, f2b 
    else:
        # print("No valid M2 solution")
        return 0, 0
    
def find_M2_satisfied_f1f2(T_target, r, delay, tol=1, max_iter=50):
    """
    Iteratively find f1, f2 so that:
      (1) M2 = 0   (from M2_helper)
      (2) f1 + f2 + 4r ≈ T_target
    """
    T = T_target
    for _ in range(max_iter):
        f1, f2 = M2_helper(T, r, delay)
        if f1 <= 0 or f2 <= 0:
            return 0, 0, False

        total_len = f1 + f2 + 4 * r
        diff = total_len - T_target

        if abs(diff) <= tol:
            return f1, f2, True

        # Adjust T slightly toward the desired total duration
        T -= diff * 0.5  # relaxation step

    return f1, f2, False

import numpy as np
from scipy.interpolate import interp1d

def interp_grad(t, g, tt_target):
    """Linear interpolation to target time grid"""
    f = interp1d(t, g, kind='linear', fill_value=0, bounds_error=False)
    return f(tt_target)