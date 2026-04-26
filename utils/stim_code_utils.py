from safe_vec_generator import PNSCNS_SequenceBuilder
import gropt
import numpy as np
from scipy.interpolate import interp1d


def diff_min_TE_base(params, target_bval=0, TE0=10e-3, TE1=120e-3, stop_dt=0.1e-3, 
                     waveforms_file= None, timings_file = None, seq_repeats = 5,
                     **kwargs):

    if target_bval == 0:
        target_bval = params['bvalue']

    stop_dt = max(stop_dt, 2 * params['dt'])

    min_TE = params['T_90'] + params['T_180'] + params['T_readout']
    min_TE = max(min_TE, 2 * (params['T_180'] + params['T_readout']))

    #TE0 += min_TE

    low = max([
    np.ceil((params['T_90'] + params['T_180'] / 2) / params['dt']) * params['dt'],
    np.ceil((params['T_180'] / 2 + params['T_readout'] / 2) / params['dt']) * params['dt']
        ]) * 2
    
    TE0 = low

    #result0 = diff_solve_TE(TE0, params, bval_min=target_bval / 2, **kwargs)
    result1 = diff_solve_TE(TE1, params, bval_min=target_bval / 2, 
                            timings_file=timings_file, waveforms_file=waveforms_file, 
                            seq_repeats=seq_repeats,  **kwargs)
    
    
    
    print(f'starting TE search: {TE0}, {TE1}\n')
    for _i in range(3):
        if not result1.converged or result1.bvalue < target_bval:
            result1 = diff_solve_TE(TE1, params, bval_min=4 ** (_i + 1) * target_bval, 
                                    timings_file=timings_file, waveforms_file=waveforms_file, 
                                    seq_repeats=seq_repeats,  **kwargs)

    if not result1.converged or result1.bvalue < target_bval:
        print(
            f'ERROR: Max TE {TE1} did not converge or was too low bvalue: {result1.converged}  {result1.bvalue}'
        )

    print('Searching TE: ', end='', flush=True)
    while TE1 - TE0 > stop_dt:
        _TE = TE0 + (TE1 - TE0) / 2
        _result = diff_solve_TE(_TE, params, bval_min=target_bval / 2, 
                                timings_file=timings_file, waveforms_file=waveforms_file, 
                                seq_repeats=seq_repeats, **kwargs)

        if not _result.converged:
            print(f'!{_TE * 1000:.2f}', end=' ', flush=True)
        else:
            print(f'{_TE * 1000:.2f}', end=' ', flush=True)

        if _result.converged:
            if _result.bvalue > target_bval:
                TE1 = _TE
                result1 = _result
            else:
                TE0 = _TE
                result0 = _result
        else:
            TE0 = _TE
            result0 = _result

    print('Done!', flush=True)

    if result0.converged and result0.bvalue > target_bval:
        return TE0, result0
    if _result.converged and _result.bvalue > target_bval:
        return _TE, _result
    if result1.converged and result1.bvalue > target_bval:
        return TE1, result1

    return None, None


def diff_solve_TE(TE, params, bval_min=100.0, refine=False, waveforms_file=None, timings_file=None, **kwargs):
    result = _diff_solve_TE(TE, params, bval_min=bval_min, waveforms_file=waveforms_file, timings_file=timings_file, **kwargs)

    if refine:
        if not result.converged:
            result2 = _diff_solve_TE(TE, params, bval_min=bval_min / 2, 
                                     waveforms_file = waveforms_file, timings_file = timings_file,  
                                     **kwargs)
        else:
            bval0 = result.bvalue
            if bval0 < bval_min:
                result2 = _diff_solve_TE(TE, params, bval_min=0.8 * bval0,
                                          waveforms_file = waveforms_file, timings_file = timings_file,  
                                            **kwargs)
            else:
                result2 = _diff_solve_TE(TE, params, bval_min=0.8 * bval0,
                                          waveforms_file = waveforms_file, timings_file = timings_file,   **kwargs)

        if (result2.converged and not result.converged) or (
            result2.converged and result2.bvalue > result.bvalue
        ):
            result = result2

    return result


def _diff_solve_TE(
    TE,
    params,
    bval_min=100.0,
    dt=0.0,
    extra_iters=1000,
    ils_max_iter=24,
    moment_tol=1e-5,
    bval_scale=1.02,
    waveforms_file=None,
    timings_file=None,
    seq_repeats = 5,
    **kwargs,
):

    if dt == 0:
        dt = params['dt']

    if 'diff_mode' in params:
        diff_mode = params['diff_mode']
    else:
        diff_mode = 'gropt'

    # Set up the GrOpt problem
    gparams = gropt.GroptParams()

    start_idx = 0
    if diff_mode == 'gropt':
        gparams.diff_init(
            dt=dt,
            TE=TE,
            T_90=params['T_90'],
            T_180=params['T_180'],
            T_readout=params['T_readout'],
        )
    elif diff_mode == 'conventional':
        gparams.diff_init_deadtime(
            dt=dt,
            TE=TE,
            T_90=params['T_90'],
            T_180=params['T_180'],
            T_readout=params['T_readout'],
        )
    elif diff_mode == 'preencode':
        start_idx = gparams.diff_init_preencode(
            dt=dt,
            TE=TE,
            T_90=params['T_90'],
            T_180=params['T_180'],
            T_readout=params['T_readout'],
            T_pre=params['T_pre'],
        )
        params['start_idx'] = start_idx
    else:
        msg = f'Unknown diff_mode: {diff_mode}'
        raise ValueError(msg)

    gparams.add_gmax(params['gmax'])
    gparams.add_smax(params['smax'])
    for _i_moment in range(params['MMT'] + 1):
        gparams.add_moment(_i_moment, 0.0, start_idx=start_idx, tol=moment_tol)

    # dEfault to first axis if you dont specify which pns or cns idx
    if params.get('cns_idx') is None:
        params['cns_idx'] = [0]

    if params.get('pns_idx') is None:
        params['pns_idx'] = [0]

    if 'pns_lim' in params:
        if 'pns_params' in params:
            # check if pns_lim is a number or an array
            # if array, assume vector constraint
            if not isinstance(params['pns_lim'], (int, float)):
                # if a vector make sure to resize to match time base
                pns_lim = match_constraint_to_timebase(params['pns_lim'], dt, TE, params['T_readout'])
                for i in params['pns_idx']:
                    gparams.add_SAFE_vec(pns_lim, safe_params=params['pns_params'],new_first_axis = i)
            
            
            # add constant value constraint if pns_lim is a scalar or custom envelope based on sequence timings
            else:
                # check if waveforms_file and timings_file are provided, if so use them to construct a custom envelope constraint
                if waveforms_file is not None and timings_file is not None:
                    seq = PNSCNS_SequenceBuilder(
                        timing_file=timings_file,
                        waveform_file=waveforms_file,
                        dt_in=1e-5,
                        dt_out=dt,
                        safe_params=params['pns_params'],
                        safe_params_cardiac=params['cns_params'],
                        TE=TE,
                        n_repeats=seq_repeats,
                    )
                    pns_x = (1 - seq.safe_gx_out) * params['pns_lim']
                    pns_y = (1 - seq.safe_gy_out) * params['pns_lim']
                    pns_z = (1 - seq.safe_gz_out) * params['pns_lim']
                    stim_vec_pns = [pns_x, pns_y, pns_z]

                    for i in params['pns_idx']:
                        gparams.add_SAFE_vec(match_length(stim_vec_pns[i], gparams.N ), safe_params=params['pns_params'],new_first_axis = i)
              
                # add constant value constraint if pns_lim is a scalar
                else:
                    for i in params['pns_idx']:
                        gparams.add_SAFE(params['pns_lim'], safe_params=params['pns_params'],new_first_axis = i)
        else:
            # Generate random safe params if not provided
            pns_params, cns_params = gropt.get_random_safe_params()
            params['pns_params'] = pns_params
            for i in params['pns_idx']:
                gparams.add_SAFE(params['pns_lim'], safe_params=pns_params, new_first_axis = i)

    if 'cns_lim' in params:
        if 'cns_params' in params:
            # check if cns_lim is a number or an array
            # if array, assume vector constraint
            if not isinstance(params['cns_lim'], (int, float)):
                # if a vector make sure to resize to match time base
                cns_lim = match_constraint_to_timebase(params['cns_lim'], dt, TE, params['T_readout'])
                for i in params['cns_idx']:
                    gparams.add_SAFE_vec(cns_lim, safe_params=params['cns_params'],new_first_axis = i)
            # add constant value constraint if cns_lim is a scalar
            else:
                # check if waveforms_file and timings_file are provided, if so use them to construct a custom envelope constraint
                if waveforms_file is not None and timings_file is not None:
                    seq = PNSCNS_SequenceBuilder(
                        timing_file=timings_file,
                        waveform_file=waveforms_file,
                        dt_in=1e-5,
                        dt_out=dt,
                        safe_params=params['pns_params'],
                        safe_params_cardiac=params['cns_params'],
                        TE=TE,
                        n_repeats=seq_repeats,
                    )
                    cns_x = (1 - seq.safe_cardiac_gx_out) * params['cns_lim']
                    cns_y = (1 - seq.safe_cardiac_gy_out) * params['cns_lim']
                    cns_z = (1 - seq.safe_cardiac_gz_out) * params['cns_lim']
                    stim_vec_cns = [cns_x, cns_y, cns_z]

                    for i in params['cns_idx']:
                        gparams.add_SAFE_vec(match_length(stim_vec_cns[i], gparams.N), safe_params=params['cns_params'],new_first_axis = i)
                
                # add constant value constraint if cns_lim is a scalar
                else:
                    for i in params['cns_idx']:
                        gparams.add_SAFE(params['cns_lim'], safe_params=params['cns_params'],new_first_axis = i)
        else:
            pns_params, cns_params = gropt.get_random_safe_params()
            params['cns_params'] = cns_params
            for i in params['cns_idx']:
                gparams.add_SAFE(params['cns_lim'], safe_params=cns_params,new_first_axis = i)

    if 'concomitant' in params:
        gparams.add_concomitant(start_idx=start_idx, weight_mod=1e4)

    if 'eddy_lam' in params:
        gparams.add_eddy(params['eddy_lam'])

    gparams.add_bvalue(bval_min, mode='minval_max', start_idx0=start_idx, max_scale=bval_scale)

    gparams.prepare()

    result = diff_solve(gparams, extra_iters=extra_iters, ils_max_iter=ils_max_iter)

    return result


def diff_solve(gparams, extra_iters=2000, ils_max_iter=30):
    solver = gropt.SolverGroptSDMM()
    solver.set_general_params(max_feval=200000, max_iter=20000, gamma_x=1.6, extra_iters=extra_iters)
    solver.set_ils_params(ils_max_iter=ils_max_iter, ils_tol=1e-12, ils_sigma=0.0001, ils_tik_lam=0.0001)
    solver.set_sdmm_params(rw_interval=16, grw_interval=41)
    result = solver.solve(gparams)
    return result




def match_constraint_to_timebase(stim, dt, TE, T_readout):
    """
    Ensure stim matches the rasterized time base [0, TE - T_readout]
    """
    # target time axis
    t_target = np.arange(0, TE - T_readout, dt)
    n_target = len(t_target)

    # scalar → expand
    if np.isscalar(stim):
        return np.full(n_target, stim)

    stim = np.asarray(stim)
    n_stim = len(stim)

    # already correct size
    if n_stim == n_target:
        return stim

    # resample via interpolation
    t_stim = np.linspace(0, TE - T_readout, n_stim)

    stim_resampled = np.interp(t_target, t_stim, stim)

    return stim_resampled




def make_stim_envelope(params,
                       TE=120e-3,
                       start_val=0.7,
                       plateau_val=1.0,
                       end_val=0.5,
                       T_ramp=4e-3):
    """
    Create stimulation envelope vector.

    Envelope:
    - Starts at start_val during RF (T_90)
    - Ramps up to plateau_val
    - Holds plateau
    - Ramps down to end_val at the end

    Parameters
    ----------
    params : dict
        Must contain 'T_90', 'T_readout', 'dt'
    TE : float
        Echo time (default 120 ms)
    start_val : float
        Initial value during RF
    plateau_val : float
        Main plateau value
    end_val : float
        Final value at end of waveform
    T_ramp : float
        Ramp duration

    Returns
    -------
    stim_vec : np.ndarray
        Envelope of length N = (TE - T_readout)/dt
    """

    dt = params['dt']

    # key indices
    N_90 = int(np.round(params['T_90'] / dt)) - 2
    N_ramp = int(np.round(T_ramp / dt))

    # total length
    N = int(np.round((TE - params['T_readout']) / dt))

    stim_vec = plateau_val * np.ones(N)

    # beginning (RF region)
    stim_vec[:N_90] = start_val

    # ramp up
    end_ramp_up = min(N_90 + N_ramp, N)
    stim_vec[N_90:end_ramp_up] = np.linspace(start_val, plateau_val, end_ramp_up - N_90)

    # ramp down (end)
    if N_ramp > 0:
        stim_vec[-N_ramp:] = np.linspace(plateau_val, end_val, N_ramp)

    return stim_vec


def match_length(vec, N):
    """Resample vector to length N for gropt."""
    x_old = np.linspace(0, 1, len(vec))
    x_new = np.linspace(0, 1, N)
    return interp1d(x_old, vec, kind='linear')(x_new)
