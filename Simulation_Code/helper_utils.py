import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import savemat
import seaborn as sns

def process_gradient_waveform(out, dt_old, dt_new, TE, save_path=None, threshold=1e-3):
    """
    Interpolate gradient waveform, split into non-zero segments,
    and save full / inverted gradients.

    Parameters
    ----------
    out : array_like
        Original gradient waveform (T/m).
    dt_old : float
        Original dwell time [s].
    dt_new : float
        Desired dwell time [s].
    TE : float
        Echo time [s].
    save_path : str, optional
        Path to save .mat file (default: None, no save).
    threshold : float, optional
        Threshold in mT/m for detecting non-zero segments (default=1e-3).

    Returns
    -------
    gDiffs : dict
        Dictionary containing:
            - G_full : interpolated full gradient [mT/m]
            - G_full_inv : inverted after TE/2 [mT/m]
            - gDiff{i} : dict with "time", "gradient", "TE"
    """

    # Original time axis
    t_old = np.arange(len(out)) * dt_old

    # Interpolation to new dwell time
    interp_fn = interp1d(t_old, out, kind='linear', fill_value='extrapolate')
    t_new = np.arange(t_old[0], t_old[-1] + dt_new/2, dt_new)
    out_new = interp_fn(t_new)

    # Convert to mT/m
    tt_ms = np.arange(out_new.size) * dt_new * 1e3
    G = out_new * 1e3

    # Detect non-zero regions
    is_nonzero = np.abs(G) > threshold
    diffs = np.diff(is_nonzero.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends   = np.where(diffs == -1)[0] + 1

    # Dictionary to store results
    gDiffs = {}
    gDiffs["G_full"] = G

    # Add inverted version (after TE/2)
    N_180 = int((TE / 2) / dt_new)
    G_copy = G.copy()
    G_copy[N_180:] *= -1
    gDiffs["G_full_inv"] = G_copy

    # ----------------------
    # Plotting with styling
    # ----------------------
    sns.set_context("talk")  # larger fonts
    cb_colors = sns.color_palette("colorblind")  # colorblind-friendly
    line_width = 3.5

    plt.figure(figsize=(8, 4))
    
    for i, (start, end) in enumerate(zip(starts, ends), 1):
        seg_start = max(start - 1, 0)
        seg_end   = min(end + 1, len(G))
        seg_t = tt_ms[seg_start:seg_end]
        seg_g = G[seg_start:seg_end].copy()

        gDiffs[f"gDiff{i}"] = {
            "time": seg_t,
            "gradient": seg_g,
            "TE": TE,
        }

        plt.plot(seg_t, seg_g, label=f'gDiff{i}', color=cb_colors[i % len(cb_colors)],
                 lw=line_width)

    plt.xlabel('Time [ms]', fontsize=14)
    plt.ylabel('Gradient [mT/m]', fontsize=14)
    plt.title('Segments of GrOpt Waveform', fontsize=16)
    #plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=12)
    
    # Hide top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Save to .mat if requested
    if save_path is not None:
        savemat(save_path, gDiffs)
        print(f"Saved gDiffs to {save_path}")

    return gDiffs


import gropt 
def plot_safe_full(
    g,
    gparams,
    safe_params,
    safe_params_cardiac,
    stim_vec=None,
    plot_cns=True,
    plot_moments=False,
    T_readout=0.0,
    Nm=5,orientation = 'h',save_path = None,
    GAMMA=42.58e3,
    xlabel_font=22, ylabel_font=22, xtick_font=20, ytick_font=20,
    title_font=23, legend_font=13, smax=200, gmax=200):
    


    axes_labels = ['X-axis', 'Y-axis', 'Z-axis']
    cb_colors = sns.color_palette('colorblind')
    line_width = 3.25
    line_styles = ['-', '-.', ':']

    first_grad = g * 1e3  # mT/m
    slew_rate = np.diff(g*1e-3, axis=0) / gparams.dt * 1e3  # T/m/s
    dt = gparams.dt  # s



    with sns.plotting_context("poster"):
        ncols = 4 if plot_cns else 3
        if plot_moments:
            ncols += 1  # add a column for moments


        if orientation == 'h':
            fig, axs = plt.subplots(1, ncols,
                                figsize=(6.5*ncols, 4.5),
                                sharex=True,
                                gridspec_kw={'wspace': 0.3, 'hspace': 0.},dpi=300)
        
        elif orientation == 'v':
            fig, axs = plt.subplots(ncols, 1,
                                figsize=(7, 4.5*ncols),
                                sharex=True,
                                gridspec_kw={'wspace': 0., 'hspace': 0.3},dpi=300)
        
        axs = np.atleast_1d(axs)  # always array

        t = np.arange(g.shape[0]) * gparams.dt * 1e3  # ms
        
        # --------- 1. Gradient -----------
        axs[0].plot(t, first_grad, color='k', lw=line_width)
        axs[0].set_title('$G(t)$', fontsize=title_font)
        axs[0].set_xlabel('Time [ms]', fontsize=xlabel_font)
        axs[0].set_ylabel('mT/m', fontsize=ylabel_font)
        #axs[0].grid(True, linestyle='--', alpha=0.2)
        axs[0].tick_params(axis='x', labelsize=xtick_font)
        axs[0].tick_params(axis='y', labelsize=ytick_font)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].plot(t, gmax*np.ones_like(t), color='gray', alpha=0.5, ls='--', lw=line_width, zorder=0)
        axs[0].plot(t, -gmax*np.ones_like(t), color='gray', alpha=0.5, ls='--', lw=line_width, zorder=0)

        # --------- 2. Slew -----------
        axs[1].plot(t[:-1], slew_rate, color='k', lw=line_width)
        axs[1].set_title('$S(t)$', fontsize=title_font)
        axs[1].set_xlabel('Time [ms]', fontsize=xlabel_font)
        axs[1].set_ylabel('T/m/s', fontsize=ylabel_font)
        #axs[1].grid(True, linestyle='--', alpha=0.2)
        axs[1].tick_params(axis='x', labelsize=xtick_font)
        axs[1].tick_params(axis='y', labelsize=ytick_font)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].plot(t, smax*np.ones_like(t), color='gray', alpha=0.5, ls='--', lw=line_width, zorder=0)
        axs[1].plot(t, -smax*np.ones_like(t), color='gray', alpha=0.5, ls='--', lw=line_width, zorder=0)
        axs[1].set_ylim([-1.1*200, 1.1*200])
        axs[1].set_yticks(np.arange(-200, 201, 100))


        # --------- 3. SAFE-PNS -----------
        for i, ax_idx in enumerate(range(3)):
            safe = gropt.gropt_wrapper.get_SAFE(g, gparams.dt,
                                                safe_params=safe_params,
                                                new_first_axis=ax_idx)
            axs[2].plot(t, safe, label=axes_labels[i],
                        color=cb_colors[i], lw=line_width,
                        linestyle=line_styles[i])
        if stim_vec is not None:
            axs[2].plot(t[:stim_vec.shape[0]], stim_vec,
                        color='gray', alpha=1, ls='--', lw=line_width)
        axs[2].set_title('SAFE-PNS', fontsize=title_font)
        axs[2].set_xlabel('Time [ms]', fontsize=xlabel_font)
        axs[2].set_ylabel('% Threshold', fontsize=ylabel_font)
        #axs[2].grid(True, linestyle='--', alpha=0.2)
        
        axs[2].tick_params(axis='x', labelsize=xtick_font)
        axs[2].tick_params(axis='y', labelsize=ytick_font)
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].legend(fontsize=legend_font, loc='lower right')
        axs[2].set_ylim([0, 1])
        

        idx_cns = 3
        if plot_cns:
            for i, ax_idx in enumerate(range(3)):
                safe_c = gropt.gropt_wrapper.get_SAFE(g, gparams.dt,
                                                      safe_params=safe_params_cardiac,
                                                      new_first_axis=ax_idx)
                axs[idx_cns].plot(t, safe_c,
                                  label=axes_labels[i],
                                  color=cb_colors[i], lw=line_width,
                                  linestyle=line_styles[i])
            if stim_vec is not None:
                axs[idx_cns].plot(t[:stim_vec.shape[0]], stim_vec,
                                  color='gray', alpha=0.5, ls='-', lw=line_width)
            axs[idx_cns].set_title('SAFE-CNS', fontsize=title_font)
            axs[idx_cns].set_xlabel('Time [ms]', fontsize=xlabel_font)
            axs[idx_cns].set_ylabel('unitless', fontsize=ylabel_font)
            #axs[idx_cns].grid(True, linestyle='--', alpha=0.2)
            axs[idx_cns].tick_params(axis='x', labelsize=xtick_font)
            axs[idx_cns].tick_params(axis='y', labelsize=ytick_font)
            axs[idx_cns].spines['top'].set_visible(False)
            axs[idx_cns].spines['right'].set_visible(False)
            axs[idx_cns].legend(fontsize=legend_font, loc='lower right')
            axs[idx_cns].set_ylim([0, 1])

        # --------- 5. Moments (optional) -----------
        if plot_moments:
            # compute moments
            line_styles = ['-', '-', '-']
            cb_colors = sns.color_palette('Set1')
            G = g
            TE = G.size*dt*1e3 + T_readout*1e3
            tINV = int(np.floor(TE/dt/1.0e3/2.0))
            INV = np.ones(G.size)
            #INV[tINV:] = -1
            tvec = np.arange(G.size)*dt
            tMat = np.zeros((Nm, G.size))
            for mm in range(Nm):
                tMat[mm] = tvec**mm
            mm = GAMMA*dt*tMat * (G*INV)[np.newaxis,:]

            idx_mom = ncols-1
            for order in range(3):  # plot first 3 orders
                mmt = np.cumsum(mm[order])
                axs[idx_mom].plot(t, mmt/np.abs(mmt).max(),
                                  lw=line_width,
                                  color=cb_colors[order],
                                  linestyle=line_styles[order],
                                  label=f'M{order}')
            axs[idx_mom].axhline(0, color='k', lw=1)
            axs[idx_mom].set_title('Moments', fontsize=title_font)
            axs[idx_mom].set_xlabel('Time [ms]', fontsize=xlabel_font)
            axs[idx_mom].set_ylabel('Normalized', fontsize=ylabel_font)
            #axs[idx_mom].grid(True, linestyle='--', alpha=0.2)
            axs[idx_mom].tick_params(axis='x', labelsize=xtick_font)
            axs[idx_mom].tick_params(axis='y', labelsize=ytick_font)
            axs[idx_mom].spines['top'].set_visible(False)
            axs[idx_mom].spines['right'].set_visible(False)
            axs[idx_mom].legend(fontsize=legend_font, loc='lower right')
        
        for i in range(ncols):
            axs[i].set_xlim([t[0], t[-1]+1])  # set x-limits for all subplots
        
        
        
        plt.tight_layout()
        plt.show()

        # print out max G and max S
        print(f"Max Gradient: {np.max(np.abs(first_grad)):.2f} mT/m")
        print(f"Max Slew Rate: {np.max(np.abs(slew_rate)):.2f} T/m/s")

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        return fig, axs
    


def plot_safe_full_invert(
    g,
    gparams,
    safe_params,
    safe_params_cardiac,
    stim_vec=None,
    stim_vec_cns=None,  
    plot_gradient=True,
    plot_slew=True,
    plot_pns=True,
    plot_cns=True,
    plot_moments=False,
    T_readout=0.0,
    Nm=5,
    orientation='h',
    save_path=None,
    GAMMA=42.58e3,
    xlabel_font=22, ylabel_font=22, xtick_font=20, ytick_font=20,
    title_font=23, legend_font=13,
    smax=200, gmax=200,line_width = 3.0,
    max_pns=90, max_cns=90,
    dark_mode=True, xMax = 45,
    epi = None
):
    """
    Plot gradient, slew rate, and SAFE-PNS/CNS with adjustable grid intensity and color mode.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import gropt

    axes_labels = ['X-axis', 'Y-axis', 'Z-axis']
    cb_colors = sns.color_palette('colorblind')
    
    line_styles = ['-', '-.', ':']

    first_grad = g * 1e3  # mT/m
    slew_rate = np.diff(g * 1e-3, axis=0) / gparams.dt * 1e3  # T/m/s
    dt = gparams.dt  # s
    t = np.arange(g.shape[0]) * gparams.dt * 1e3  # ms

    # --- Styling based on mode ---
    if dark_mode:
        plt.style.use('dark_background')
        text_color = 'white'
        back_color = 'dimgray'
        grid_color = (1, 1, 1, 0.08)   # faint white grid
        bg_color = 'black'
    else:
        plt.style.use('default')
        text_color = 'black'
        back_color = 'lightgray'
        grid_color = (0, 0, 0, 0.08)   # faint black grid
        bg_color = 'white'

    # --- Determine number of plots dynamically ---
    plot_count = 0
    if plot_gradient:
        plot_count += 1
    if plot_slew:
        plot_count += 1
    if plot_cns:
        plot_count += 1
    plot_count += 1  # Always include SAFE-PNS
    if plot_moments:
        plot_count += 1

    # --- Layout setup ---
    figsize = (9 * plot_count, 4.5) if orientation == 'h' else (8, 4.5 * plot_count)
    fig, axs = plt.subplots(
        1 if orientation == 'h' else plot_count,
        plot_count if orientation == 'h' else 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'wspace': 0.3, 'hspace': 0.3},
        dpi=300,
        facecolor=bg_color
    )
    axs = np.atleast_1d(axs)
    ax_idx = 0
    cb_colors = sns.color_palette('colorblind')

    # --- Add horizontal line at y = 0 for all subplots ---
    for ax in axs:
        ax.axhline(0, color=back_color, alpha=0.8, ls='-', lw=1.5)

    # --- Gradient plot ---
    print(epi)
    if plot_gradient:
        axs[ax_idx].plot(t, first_grad, color=text_color, lw=line_width)
        axs[ax_idx].fill_between(t , 0, first_grad, color=cb_colors[0], alpha=0.75)

        if epi is not None:
            print(epi)
            t_epi = epi["t"]*1e3        # already in ms
            t_shift = t[-1] - t_epi[0]    # small offset so it’s visible
            t_epi_shifted = t_epi + t_shift
            gy_epi = epi['gx'] # use gx to show epi
            axs[ax_idx].plot(t_epi_shifted, gy_epi * 1e3,color=text_color, lw=line_width, alpha = 0.7)
            axs[ax_idx].legend(fontsize=legend_font, loc='upper right')
            # vline at TE 
            axs[ax_idx].axvline(t[-1] + T_readout, color='black', alpha=1, ls='--', lw=line_width)

        
        
        axs[ax_idx].set_title('$G(t)$', fontsize=title_font, color=text_color)
        axs[ax_idx].set_xlabel('Time [ms]', fontsize=xlabel_font, color=text_color)
        axs[ax_idx].set_ylabel('mT/m', fontsize=ylabel_font, color=text_color)
        #axs[ax_idx].grid(True, linestyle='--', color=grid_color, alpha=0.3)
        # Threshold lines that always span full x-range
        axs[ax_idx].axhline(gmax, color=text_color, alpha=0.2, ls='--', lw=line_width)
        axs[ax_idx].axhline(-gmax, color=text_color, alpha=0.2, ls='--', lw=line_width)
        axs[ax_idx].axhline(0, color=text_color, alpha=1, ls='-', lw=line_width)
        
        # Set yticks from -gmax to gmax in steps of 100
        step = 100
        yticks = np.arange(-gmax, gmax + step, step)
        axs[ax_idx].set_yticks(yticks)
        axs[ax_idx].set_ylim([-gmax + 1, gmax + 1])
        axs[ax_idx].tick_params(axis='y', labelsize=ytick_font)

        ax_idx += 1


    # --- Slew rate plot ---
    if plot_slew:
        axs[ax_idx].plot(t[:-1], slew_rate, color=text_color, lw=line_width)
        axs[ax_idx].set_title('$S(t)$', fontsize=title_font, color=text_color)
        axs[ax_idx].set_xlabel('Time [ms]', fontsize=xlabel_font, color=text_color)
        axs[ax_idx].set_ylabel('T/m/s', fontsize=ylabel_font, color=text_color)
        #axs[ax_idx].grid(True, linestyle='--', color=grid_color, alpha=0.3)
        # Use axhline for full-width threshold lines
        axs[ax_idx].axhline(smax, color=text_color, alpha=0.2, ls='--', lw=line_width)
        axs[ax_idx].axhline(-smax, color=text_color, alpha=0.2, ls='--', lw=line_width)
        axs[ax_idx].axhline(0, color=text_color, alpha=1, ls='-', lw=line_width)
        axs[ax_idx].set_ylim([-201, 201])
        axs[ax_idx].set_yticks(np.arange(-200, 201, 100))

         # Set yticks from -smax to smax in steps of 100
        step = 100
        yticks = np.arange(-smax, smax + step, step)
        axs[ax_idx].set_yticks(yticks)
        axs[ax_idx].tick_params(axis='y', labelsize=ytick_font)

        ax_idx += 1

    # --- SAFE-PNS plot ---
    if plot_pns:
        pns_norm = np.empty((0, len(t)))  # initialize outside if looping multiple times

        safe_all =[]
        for i in range(1,2):
            safe = gropt.gropt_wrapper.get_SAFE(g, gparams.dt, safe_params=safe_params, new_first_axis=i)
            axs[ax_idx].plot(t, safe, label=axes_labels[i],
                            color='black', lw=line_width, linestyle='-') #line_styles[i]
            safe_all.append(safe)
            
        pns_comp = np.array([gropt.gropt_wrapper.get_SAFE(g, gparams.dt,
                                                        safe_params=safe_params,
                                                        new_first_axis=i)
                            for i in range(3)])
        
        #pns_norm = np.vstack([pns_norm, np.linalg.norm(pns_comp[-3:, :] * 0.01, axis=0)])
        #print(pns_norm.shape)
        #axs[ax_idx].plot(t, pns_norm[-1] * 100, color=text_color, lw=line_width, ls='-', label='NRM')
    
        if stim_vec is None:
            axs[ax_idx].axhline(max_pns, color=text_color, alpha=0.3, ls='-', lw=2)
        axs[ax_idx].set_title('SAFE-PNS', fontsize=title_font, color=text_color)
        axs[ax_idx].set_xlabel('Time [ms]', fontsize=xlabel_font, color=text_color)
        axs[ax_idx].set_ylabel('unitless', fontsize=ylabel_font, color=text_color)
        #axs[ax_idx].grid(True, linestyle='--', color=grid_color, alpha=0.5)
        axs[ax_idx].legend(fontsize=legend_font, loc='lower right', facecolor=bg_color, framealpha=0.2)
        axs[ax_idx].set_ylim([0, 1])
        axs[ax_idx].axhline(0, color=text_color, alpha=1, ls='-', lw=line_width)
        if stim_vec is not None:
            for ii in range(1, 2):
                y = stim_vec[ii] 
                x = t[:y.shape[0]]
                # Make sure x and y are numpy arrays
                x = np.asarray(x)
                y = np.asarray(y)

                # If lengths mismatch, interpolate or truncate y to match x
                if len(x) != len(y):
                    from scipy.interpolate import interp1d
                    f = interp1d(np.linspace(0, 1, len(y)), y, kind='linear', fill_value="extrapolate")
                    y = f(np.linspace(0, 1, len(x)))

                axs[ax_idx].plot(
                    x,
                    y,
                    color='red',       # solid black line
                    linewidth=6,     # adjust thickness as needed
                    zorder=2 ,       # ensure it’s on top
                    ls = '--',
                )
        
        
        axs[ax_idx].axhline(0.7, color='red',  ls=':', lw=5,alpha = 0.5)
        

        ax_idx += 1

    # --- SAFE-CNS plot ---
    if plot_cns:
        pns_norm = np.empty((0, len(t)))  # initialize outside if looping multiple times

        safe_all =[]
        for i in range(1,2):
            safe_c = gropt.gropt_wrapper.get_SAFE(g, gparams.dt,
                                                  safe_params=safe_params_cardiac,
                                                  new_first_axis=i)
            axs[ax_idx].plot(t, safe_c , label=axes_labels[i],
                             color='black', lw=line_width,
                             linestyle='-')#line_styles[i] #cb_colors[i]
            safe_all.append(safe_c)
        axs[ax_idx].set_title('SAFE-CNS', fontsize=title_font, color=text_color)
        axs[ax_idx].set_xlabel('Time [ms]', fontsize=xlabel_font, color=text_color)
        axs[ax_idx].set_ylabel('unitless', fontsize=ylabel_font, color=text_color)
        #axs[ax_idx].grid(True, linestyle='--', color=grid_color, alpha=0.5)
        axs[ax_idx].legend(fontsize=legend_font, loc='lower right', facecolor=bg_color, framealpha=0.2)
        axs[ax_idx].set_ylim([0, 1])
        axs[ax_idx].axhline(0, color=text_color, alpha=1, ls='-', lw=line_width)
        if stim_vec_cns is None:
            axs[ax_idx].axhline(max_cns, color=text_color, alpha=0.3, ls='-', lw=2)
        if stim_vec_cns is not None:
            for ii in range(1, 2):
                y = stim_vec_cns[ii] 
                x = t[:y.shape[0]]

                # Make sure x and y are numpy arrays
                x = np.asarray(x)
                y = np.asarray(y)

                # If lengths mismatch, interpolate or truncate y to match x
                if len(x) != len(y):
                    from scipy.interpolate import interp1d
                    f = interp1d(np.linspace(0, 1, len(y)), y, kind='linear', fill_value="extrapolate")
                    y = f(np.linspace(0, 1, len(x)))

                    
                axs[ax_idx].plot(
                    x,
                    y,
                    color='purple',       # solid black line
                    linewidth=6,     # adjust thickness as needed
                    zorder=2 ,        # ensure it’s on top
                    ls = '--',
                )
            axs[ax_idx].axhline(max_cns, color='purple',  ls='--', lw=3)
        
        pns_comp = np.array([gropt.gropt_wrapper.get_SAFE(g, gparams.dt,
                                                        safe_params=safe_params_cardiac,
                                                        new_first_axis=i)
                            for i in range(3)])
        
        #pns_norm = np.vstack([pns_norm, np.linalg.norm(pns_comp[-3:, :] * 0.01, axis=0)])
        #print(pns_norm.shape)
        #axs[ax_idx].plot(t, pns_norm[-1] * 100, color=text_color, lw=line_width, ls='-', label='NRM')
    
        
        axs[ax_idx].axhline(0.7, color='purple',  ls=':', lw=5,alpha = 0.5)
        
        ax_idx += 1
        

    # --- Optional Moments plot ---
    if plot_moments:
        line_styles = ['-', '-', '-']
        cb_colors = sns.color_palette('Set1')
        G = g
        TE = G.size * dt * 1e3 + T_readout * 1e3
        tINV = int(np.floor(TE / dt / 1.0e3 / 2.0))
        INV = np.ones(G.size)
        #INV[tINV:] = -1
        tvec = np.arange(G.size) * dt
        tMat = np.array([tvec**mm for mm in range(Nm)])
        mm = GAMMA * dt * tMat * (G * INV)[np.newaxis, :]

        for order in range(3):
            mmt = np.cumsum(mm[order])
            axs[ax_idx].plot(t, mmt / np.abs(mmt).max(),
                             lw=line_width,
                             color=cb_colors[order],
                             linestyle=line_styles[order],
                             label=f'M{order}')
        axs[ax_idx].axhline(0, color=text_color, lw=1, alpha=0.5)
        axs[ax_idx].set_title('Moments', fontsize=title_font, color=text_color)
        axs[ax_idx].set_xlabel('Time [ms]', fontsize=xlabel_font, color=text_color)
        axs[ax_idx].set_ylabel('Normalized', fontsize=ylabel_font, color=text_color)
        #axs[ax_idx].grid(True, linestyle='--', color=grid_color, alpha=0.5)
        axs[ax_idx].axhline(0, color=text_color, alpha=1, ls='-', lw=line_width)
        axs[ax_idx].legend(fontsize=legend_font, loc='lower right', facecolor=bg_color, framealpha=0.2)

    # --- Final styling ---
    for ax in axs:
        ax.set_facecolor(bg_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(text_color)
        ax.spines['bottom'].set_color(text_color)
        ax.tick_params(axis='x', labelsize=xtick_font, colors=text_color)
        ax.tick_params(axis='y', labelsize=ytick_font, colors=text_color)
        ax.set_xlim([t[0], t[-1] + 1])

    # set background transparent if dark mode
    if dark_mode:
        fig.patch.set_alpha(0.0)
    
    # set xlim for all subplots
    for ax in axs:
        # if first axis set to xMax 
        ax.set_xlim([t[0], xMax])
        ax.set_xticks(np.arange(0, xMax, 25))
        ax.tick_params(axis='x', labelsize=xtick_font)
        ax.tick_params(axis='y', labelsize=ytick_font)
    
    
    
    axs[0].set_ylim([-201,201])
    axs[0].set_yticks(np.arange(-200, 201, 100))
     

    plt.tight_layout()
    plt.show()

    

    if plot_gradient:
        print(f"Max Gradient: {np.max(np.abs(first_grad)):.2f} mT/m")
    if plot_slew:
        print(f"Max Slew Rate: {np.max(np.abs(slew_rate)):.2f} T/m/s")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        print(f"Saved figure to {save_path}")

    
    return fig, axs

def plot_safe_combined(
    g,
    gparams,
    safe_params,
    safe_params_cardiac,
    stim_vec=None,
    stim_vec_cns=None,
    GAMMA=42.58e3,
    title_font=20,
    legend_font=13,
    xtick_font=18,
    ytick_font=18,
    xlabel_font=20,
    ylabel_font=20,
    smax=200,
    gmax=200,
    max_pns=100,
    max_cns=100,
    dark_mode=True,
    xMax=None,
    save_path=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gropt

    # -------------------------------------------------
    # Prep
    # -------------------------------------------------
    cb = sns.color_palette("colorblind")
    dt = gparams.dt
    t = np.arange(g.shape[0]) * dt * 1e3
    first_grad = g * 1e3                     # mT/m
    slew = np.diff(g * 1e-3, axis=0) / dt * 1e3   # T/m/s
    t_slew = t[:-1]

    safe_pns = gropt.gropt_wrapper.get_SAFE(g, dt, safe_params=safe_params, new_first_axis=1)
    safe_cns = gropt.gropt_wrapper.get_SAFE(g, dt, safe_params=safe_params_cardiac, new_first_axis=1)

    if dark_mode:
        plt.style.use("dark_background")
        bg = "black"
    else:
        plt.style.use("default")
        bg = "white"

    # -------------------------------------------------
    # Create 3 subplots
    # -------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), dpi=300, sharex=True)
    fig.patch.set_facecolor(bg)

    # =================================================
    # 1) GRADIENT (left)  +  SLEW (right)
    # =================================================
    axL = axs[0]
    axR = axL.twinx()

    axL.plot(t, first_grad, lw=3, color=cb[0], label="Gradient")
    axL.fill_between(t, 0, first_grad, color=cb[0], alpha=0.3)

    axR.plot(t_slew, slew, lw=2.5, color=cb[1], label="Slew")

    axL.set_ylabel("Gradient [mT/m]", fontsize=ylabel_font)
    axR.set_ylabel("Slew [T/m/s]", fontsize=ylabel_font, color=cb[1])

    axL.set_ylim([-gmax, gmax])
    axR.set_ylim([-smax, smax])

    axL.tick_params(labelsize=ytick_font)
    axR.tick_params(labelsize=ytick_font, colors=cb[1])

    axL.axhline(0, color="gray", lw=1)

    # =================================================
    # 2) SLEW (left)  +  SAFE-PNS (right)
    # =================================================
    axL2 = axs[1]
    axR2 = axL2.twinx()

    # left = slew
    axL2.plot(t_slew, slew, lw=2.5, color=cb[1])
    axL2.set_ylabel("Slew [T/m/s]", fontsize=ylabel_font, color=cb[1])
    axL2.set_ylim([-smax, smax])
    axL2.tick_params(labelsize=ytick_font, colors=cb[1])

    # right = SAFE-PNS
    axR2.plot(t, safe_pns , lw=3, color=cb[2], label="SAFE-PNS")
    axR2.axhline(max_pns, ls="--", lw=1.5, color=cb[2], alpha=0.3)
    axR2.set_ylabel("unitless", fontsize=ylabel_font, color=cb[2])
    axR2.set_ylim([0, 1])
    axR2.tick_params(labelsize=ytick_font, colors=cb[2])

    # =================================================
    # 3) SLEW (left)  +  SAFE-CNS (right)
    # =================================================
    axL3 = axs[2]
    axR3 = axL3.twinx()

    # left = slew
    axL3.plot(t_slew, slew, lw=2.5, color=cb[1])
    axL3.set_ylabel("Slew [T/m/s]", fontsize=ylabel_font, color=cb[1])
    axL3.set_ylim([-smax, smax])
    axL3.tick_params(labelsize=ytick_font, colors=cb[1])

    # right = SAFE-CNS
    axR3.plot(t, safe_cns , lw=3, color=cb[3], label="SAFE-CNS")
    axR3.axhline(max_cns, ls="--", lw=1.5, color=cb[3], alpha=0.3)
    axR3.set_ylabel("unitless", fontsize=ylabel_font, color=cb[3])
    axR3.set_ylim([0, 1])
    axR3.tick_params(labelsize=ytick_font, colors=cb[3])

    # -------------------------------------------------
    # X labels only on bottom
    # -------------------------------------------------
    axL3.set_xlabel("Time [ms]", fontsize=xlabel_font)
    # set xticks color black
    axL3.tick_params(axis='x', labelsize=xtick_font, colors='black')

    if xMax is not None:
        axL.set_xlim(0, xMax)

    for ax in plt.gcf().axes:
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', colors='black')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig