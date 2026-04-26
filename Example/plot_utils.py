import matplotlib.pyplot as plt
import numpy as np
import gropt
from safe_vec_generator import PNSCNS_SequenceBuilder
from scipy.interpolate import interp1d


def get_moments(g, dt, inv_vec=None, start_idx=0, scale_to_one=True):
    if g.squeeze().ndim == 2:
        g = g[0]  # TODO: 3-axis case, right now just assumes 1 axis

    Nm = 5
    tt = np.arange(g.size) * dt
    _m = np.zeros((Nm, g.size))

    for mm in range(Nm):
        _m[mm] = tt**mm

    if inv_vec is not None:
        mm = dt * _m * (g * inv_vec)[np.newaxis, :]
    else:
        mm = dt * _m * g[np.newaxis, :]

    mm[:, :start_idx] = 0

    moments = [np.cumsum(x) for x in mm]

    if scale_to_one:
        moments = [x / np.abs(x).max() for x in moments]

    return moments


def get_concomitant(g, dt, inv_vec, start_idx=0):
    g_start = g[start_idx:]
    inv_vec_start = inv_vec[start_idx:]
    pos = np.sum(dt * g_start[inv_vec_start > 0] ** 2.0)
    neg = np.sum(dt * g_start[inv_vec_start < 0] ** 2.0)

    ratio = pos / neg
    if ratio < 1:
        ratio = 1 / ratio

    return ratio


def get_bval(g, dt, inv_vec=None, TE=0, start_idx=0):
    if g.squeeze().ndim == 2:
        g = g[0]  # TODO: 3-axis case, right now just assumes 1 axis

    if inv_vec is None:
        inv_vec = np.ones(g.size)
        tINV = int(np.floor(TE / dt / 2.0))
        inv_vec[tINV:] = -1

    GAMMA = 42.58e3

    Gt = 0
    bval = 0
    for i in range(start_idx, g.size):
        Gt += inv_vec[i] * g[i] * dt
        bval += Gt * Gt * dt

    bval *= (GAMMA * 2 * np.pi) ** 2

    return bval


def plot_diff(*args, mode='diff', **kwargs):

    plot_waves(*args, mode=mode, **kwargs)


def plot_waves(
    g,
    dt,
    inv_vec=None,
    TE=0,
    gmax=0,
    smax=0,
    start_idx=0,
    eddy_lam=0.0,
    plot_eddy=False,
    plot_pns=False,
    plot_cns=False,
    stim_vec=None,
    stim_vec_cns = None,
    pns_lim=0,
    cns_lim=0,
    N_cols=2,
    mode='regular',
    params={},
    highlight_rf=True,
    pns_params=None,    
    cns_params=None,
    timings_file = None,
    waveforms_file = None,
    seq_repeats = 5,
):

    if start_idx == 0:
        start_idx = params.get('start_idx', 0)
    if eddy_lam == 0:
        eddy_lam = params.get('eddy_lam', 0.0)
    if stim_vec is None:
        stim_vec = params.get('stim_vec', None)



    
    if not isinstance(params['pns_lim'], (int, float)): 
        pns_lim = np.max(params['pns_lim'])
        stim_vec = params['pns_lim']
        # ensure stim_vec is same length as time base
        stim_vec = match_constraint_to_timebase(stim_vec, dt, TE, params['T_readout'])

    if not isinstance(params['cns_lim'], (int, float)): 
        cns_lim = np.max(params['cns_lim'])
        stim_vec_cns = params['cns_lim']
        # ensure stim_vec_cns is same length as time base
        stim_vec_cns = match_constraint_to_timebase(stim_vec_cns, dt, TE, params['T_readout'])
    
    if isinstance(params['pns_lim'], (int, float)) and timings_file is not None and waveforms_file is not None:
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
        stim_vec = []
        for i in range(3):
            stim_vec.append(match_constraint_to_timebase(stim_vec_pns[i], dt, TE, params['T_readout']))
        stim_vec = np.array(stim_vec)
        
    if isinstance(params['cns_lim'], (int, float)) and timings_file is not None and waveforms_file is not None:
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
        stim_vec_cns_tmp = [cns_x, cns_y, cns_z]
        stim_vec_cns = []
        for i in range(3):
            stim_vec_cns.append(match_constraint_to_timebase(stim_vec_cns_tmp[i], dt, TE, params['T_readout'])) 
        stim_vec_cns = np.array(stim_vec_cns)

    if pns_lim == 0:
        pns_lim = params.get('pns_lim', 0.0)
    if cns_lim == 0:
        cns_lim = params.get('cns_lim', 0.0)


    if pns_params is None or cns_params is None:
        _pns_params, _cns_params = gropt.get_random_safe_params()
        if pns_params is None:
            pns_params = params.get('pns_params', _pns_params)
        if cns_params is None:
            cns_params = params.get('cns_params', _cns_params)

    pns = None
    cns = None

    pns = []
    cns = []


    if plot_pns or pns_lim > 0:
        if pns_lim == 0:
            pns_lim = 1.0
        for i in range(3):
            pns_i = gropt.get_SAFE(g, dt, safe_params=pns_params, new_first_axis=i)
            pns.append(pns_i)
        pns = np.stack(pns, axis=1)

    if plot_cns or cns_lim > 0:
        if cns_lim == 0:
            cns_lim = 1.0
        for i in range(3):
            cns_i = gropt.get_SAFE(g, dt, safe_params=cns_params, new_first_axis=i)
            cns.append(cns_i)
        cns = np.stack(cns, axis=1)

    if inv_vec is None:
        inv_vec = np.ones(g.size)
        tINV = start_idx + int(np.floor(TE / dt / 2.0))
        inv_vec[tINV:] = -1

    tt_ms = np.arange(g.size) * dt * 1e3

    to_plot = ['gradient', 'slew', 'moments']
    to_plot = ['gradient', 'slew', 'moments']

    if plot_pns or pns_lim > 0:
        to_plot.append('pns')

    if plot_cns or cns_lim > 0:
        to_plot.append('cns')
    if plot_eddy or eddy_lam > 0:
        to_plot.append('eddy')

    N_plots = len(to_plot)
    N_rows = (N_plots + N_cols - 1) // N_cols
    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(10, N_rows * 3.0), layout='tight')

    # Diffusion Title String
    # ======================================
    if mode == 'diff':
        label = ''

        if TE > 0:
            label += f'TE: {1000 * TE:.2f} ms  ---  '

        bval = get_bval(g, dt, inv_vec, start_idx=start_idx)
        label += f'b-value: {bval:.0f} $mm^2/s$  ---  '

        c_ratio = get_concomitant(g, dt, inv_vec, start_idx=start_idx)
        label += f'concomitant ratio: {c_ratio:.2f}'

        f.suptitle(label)

        # Get the span locations for plotting 0's
        if TE > 0 and 'T_180' in params:
            if 'T_pre' in params:
                t_start = params['T_pre']
            else:
                t_start = 0.0

            t_inv = t_start + TE / 2.0
            t_180_start = t_inv - params['T_180'] / 2.0
            t_180_end = t_inv + params['T_180'] / 2.0

            if 'T_90' in params:
                t_90_start = t_start
                t_90_end = t_start + params['T_90']

    for i_ax, ax in enumerate(axarr.flatten()):
        plot_type = to_plot[i_ax] if i_ax < N_plots else None
        if plot_type is None:
            ax.set_visible(False)
        elif plot_type == 'gradient':
            ax.axhline(linestyle='--', color='0.7')
            if gmax == 0:
                gmax = params.get('gmax', 0)
            if gmax > 0:
                ax.axhline(1000 * gmax, linestyle=':', color='r', alpha=0.7)
                ax.axhline(-1000 * gmax, linestyle=':', color='r', alpha=0.7)

            if highlight_rf and t_90_end > 0:
                ax.axvspan(t_90_start * 1e3, t_90_end * 1e3, color='coral', alpha=0.3)
            if highlight_rf and t_180_end > 0:
                ax.axvspan(t_180_start * 1e3, t_180_end * 1e3, color='coral', alpha=0.3)
            ax.plot(tt_ms, g * 1000)
            ax.set_title('Gradient')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('G [mT/m]')
        elif plot_type == 'slew':
            ax.axhline(linestyle='--', color='0.7')
            if smax == 0:
                smax = params.get('smax', 0)
            if smax > 0:
                ax.axhline(smax, linestyle=':', color='r', alpha=0.7)
                ax.axhline(-smax, linestyle=':', color='r', alpha=0.7)
            ax.plot(tt_ms[:-1], np.diff(g) / dt)
            ax.set_title('Slew')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('dG/dt [T/m/s]')
        elif plot_type == 'moments':
            mm = get_moments(g, dt, inv_vec, start_idx=start_idx)

            ax.axhline(linestyle='--', color='0.7')
            for im in range(params.get('MMT', 4) + 1):
                ax.plot(tt_ms, mm[im], label=f'{im}')
            ax.legend(loc='upper left')
            ax.set_title('Moments')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('Moment [a.u.]')
     

        elif plot_type == 'pns':
            ax.axhline(linestyle='--', color='0.7')

            cmap = plt.get_cmap('tab10')  # or 'tab20' if >10 channels

            if pns is not None:
                n_channels = pns.shape[1]
                for i in range(n_channels):
                    color = cmap(i)
                    ax.plot(tt_ms, pns[:, i], color=color, label=f'PNS$_{i}$')

            if stim_vec is not None:
                if np.ndim(stim_vec) == 2 and stim_vec.shape[0] >= 3:
                    for i in params['pns_idx']:
                        color = cmap(i)
                        ax.plot(tt_ms, stim_vec[i], linestyle='--',
                                color=color, alpha=1, label=f'E$_{i}$')
                else:
                    ax.plot(tt_ms, stim_vec, linestyle='-', color='r', alpha=0.7)

            if pns_lim > 0:
                ax.axhline(pns_lim, linestyle=':', color='r', alpha=0.7)

            ax.set_title('PNS (SAFE)')
            ax.set_xlabel('t [ms]')
            ax.legend(loc='lower left')


        elif plot_type == 'cns':
            ax.axhline(linestyle='--', color='0.7')

            cmap = plt.get_cmap('tab10')  # or 'tab20' if >10 channels

            if cns is not None:
                n_channels = cns.shape[1]
                for i in range(n_channels):
                    color = cmap(i)
                    ax.plot(tt_ms, cns[:, i], color=color, label=f'CNS$_{i}$')

            if stim_vec_cns is not None:
                if np.ndim(stim_vec_cns) == 2 and stim_vec_cns.shape[0] >= 3:
                    for i in params['cns_idx']:
                        color = cmap(i)
                        ax.plot(tt_ms, stim_vec_cns[i], linestyle='--',
                                color=color, alpha=1, label=f'E$_{i}$')
                else:
                    ax.plot(tt_ms, stim_vec_cns, linestyle=':', color='r', alpha=0.7)

            if cns_lim > 0:
                ax.axhline(cns_lim, linestyle=':', color='m', alpha=0.7)

            ax.set_title('CNS (SAFE)')
            ax.set_xlabel('t [ms]')
            ax.legend(loc='lower left')
        elif plot_type == 'eddy':
            all_lam = np.linspace(0.1, 120, 1000)
            all_e = []
            for lam in all_lam:
                _lam = lam * 1.0e-3
                r = np.diff(np.exp(-np.arange(g.size + 1) * dt / _lam))[::-1]
                all_e.append(100 * r @ g)
            all_e = np.array(all_e)

            if all_e.min() < -all_e.max():
                all_e = -all_e

            ax.axhline(linestyle='--', color='0.7')
            ax.plot(all_lam, all_e)

            if eddy_lam > 0:
                ax.axvline(eddy_lam * 1e3, linestyle='--', color='r', alpha=0.7)

            ax.set_title('Eddy Spectrum')
            ax.set_xlabel('lambda [ms]')
            ax.set_ylabel('eddy spectrum [a.u.]')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.show()



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


def match_length(vec, N):
    """Resample vector to length N for gropt."""
    x_old = np.linspace(0, 1, len(vec))
    x_new = np.linspace(0, 1, N)
    return interp1d(x_old, vec, kind='linear')(x_new)
