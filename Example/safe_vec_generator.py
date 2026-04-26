import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
import pypulseq as pp
import gropt


# --------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------

def interp_lin(t, g, tt):
    """Safe linear interpolation."""
    if t.size == 0 or g.size == 0:
        return np.zeros_like(tt)
    f = interp1d(t, g, kind='linear', fill_value=0, bounds_error=False)
    return f(tt)

def time_to_index(t, t_ref):
    """Nearest index to a reference time."""
    return int(np.argmin(np.abs(t - t_ref)))

def resample_to_raster(t_old, y_old, raster):
    """Resample y_old(t_old) onto a new uniform time axis."""
    t_new = np.arange(t_old[0], t_old[-1], raster)
    y_new = np.interp(t_new, t_old, y_old)
    return t_new, y_new

# ==============================================================
# MAIN CLASS
# ==============================================================

class PNSCNS_SequenceBuilder:
    """
    Build interpolated waveforms, repeated sequences, compute PNS/CNS responses,
    truncate from 90° RF center to end of zeros_delayTE2, and resample.
    Includes optional fat-sat block before RF90.
    """

    def __init__(self, timing_file, waveform_file,
                 dt_in, dt_out, safe_params, safe_params_cardiac, TE, n_repeats=5):

        self.timing_file  = timing_file
        self.waveform_file = waveform_file
        self.dt_in   = dt_in
        self.dt_out  = dt_out
        self.safe_params = safe_params
        self.safe_params_cardiac = safe_params_cardiac
        self.TE = TE
        self.n_repeats = n_repeats

        # -----------------------------
        # Load data and interpolate
        # -----------------------------
        self._load_timing()
        self._load_waveforms()
        self._interpolate_RF_blocks()
        self._build_diffusion_encoding()
        self._assemble_sequence()
        
        # -----------------------------
        # Repeat sequence and compute SAFE/CNS
        # -----------------------------
        self._repeat_sequence()
        self._compute_safe()

        # -----------------------------
        # Truncate 90° center → end of zeros_delayTE2
        # -----------------------------
        self._truncate_center90_to_TE2()

        # -----------------------------
        # Resample SAFE/PNS & CNS
        # -----------------------------
        self._resample_safe_cns()


    # ----------------------------------------------------------
    # Load timing file
    # ----------------------------------------------------------
    def _load_timing(self):
        data = loadmat(self.timing_file)

        self.rf90_dur   = data['rf_90_duration'][0][0]
        self.rf180_dur  = data['rf_180_duration'][0][0]
        self.center90   = data['rf_90_rfCenterInclDelay'][0][0]
        self.center180  = data['rf_180_rfCenterInclDelay'][0][0]
        self.nav_dur    = data['nav_dur'][0][0]
        self.timeToTE   = data['timeToTE'][0][0]
        self.T90        = (self.rf90_dur - self.center90 + self.nav_dur)
        self.T180       = self.rf180_dur


    # ----------------------------------------------------------
    # Load waveform file (including optional fatsat)
    # ----------------------------------------------------------
    def _load_waveforms(self):
        mat = loadmat(self.waveform_file)
        self.w90   = mat["wave_data_rf90"]
        self.w180  = mat["wave_data_rf180"]
        self.wEPI  = mat["wave_data_epi"]
        self.wFAT  = mat.get("wave_data_fatsat", None)  # optional fatsat block


    # ----------------------------------------------------------
    # Extract and interpolate RF/EPI/FAT blocks
    # ----------------------------------------------------------
    def _extract(self, W):
        t_x, gx = W[0,0][0,:], W[0,0][1,:]
        t_y, gy = W[0,1][0,:], W[0,1][1,:]
        t_z, gz = W[0,2][0,:], W[0,2][1,:]
        t_rf, rf = W[0,3][0,:], W[0,3][1,:]

        gx = pp.convert.convert(gx, from_unit = 'Hz/m', to_unit='mT/m') * 1e-3
        gy = pp.convert.convert(gy, from_unit = 'Hz/m', to_unit='mT/m') * 1e-3
        gz = pp.convert.convert(gz, from_unit = 'Hz/m', to_unit='mT/m') * 1e-3

        if rf.size > 0:
            rf = np.abs(rf)/np.max(np.abs(rf))

        return dict(t_x=t_x, gx=gx,
                    t_y=t_y, gy=gy,
                    t_z=t_z, gz=gz,
                    t_rf=t_rf, rf=rf)

    def _interpolate_block(self, block, duration):
        tt = np.arange(0, duration + self.dt_in, self.dt_in)
        gx = interp_lin(block["t_x"], block["gx"], tt)
        gy = interp_lin(block["t_y"], block["gy"], tt)
        gz = interp_lin(block["t_z"], block["gz"], tt)
        rf = interp_lin(block["t_rf"], block["rf"], tt)
        return dict(gx=gx, gy=gy, gz=gz, rf=rf, t=tt)

    def _interpolate_RF_blocks(self):
        self.RF90_raw  = self._extract(self.w90)
        self.RF180_raw = self._extract(self.w180)
        self.EPI_raw   = self._extract(self.wEPI)

        self.RF90_interp  = self._interpolate_block(self.RF90_raw,  self.rf90_dur)
        self.RF180_interp = self._interpolate_block(self.RF180_raw, self.rf180_dur)

        # Interpolate fatsat if available
        if self.wFAT is not None:
            t_arrays = [self.wFAT[0,0][0,:], self.wFAT[0,1][0,:],
                        self.wFAT[0,2][0,:], self.wFAT[0,3][0,:]]
            dur_fat = max([t.max() for t in t_arrays if t.size>0])
            self.FAT_interp = self._interpolate_block(self._extract(self.wFAT), dur_fat)
        else:
            self.FAT_interp = dict(gx=np.array([]), gy=np.array([]), gz=np.array([]), rf=np.array([]), t=np.array([]))

        # EPI flexible
        t_arrays = [self.EPI_raw["t_x"], self.EPI_raw["t_y"], self.EPI_raw["t_z"], self.EPI_raw["t_rf"]]
        tmax = max([t.max() for t in t_arrays if t.size>0])
        tt = np.arange(0, tmax + self.dt_in, self.dt_in)
        self.EPI_interp = dict(
            gx=interp_lin(self.EPI_raw["t_x"], self.EPI_raw["gx"], tt),
            gy=interp_lin(self.EPI_raw["t_y"], self.EPI_raw["gy"], tt),
            gz=interp_lin(self.EPI_raw["t_z"], self.EPI_raw["gz"], tt),
            rf=interp_lin(self.EPI_raw["t_rf"], self.EPI_raw["rf"], tt),
            t=tt
        )


    # ----------------------------------------------------------
    # Split RF180 into pre/post, create delays
    # ----------------------------------------------------------
    def _build_diffusion_encoding(self):
        self.i_center_90  = time_to_index(self.RF90_interp["t"],  self.center90)
        self.i_center_180 = time_to_index(self.RF180_interp["t"], self.center180)

        R = self.RF180_interp
        self.RF180_pre  = {k:R[k][:self.i_center_180+1] for k in ["gx","gy","gz","rf"]}
        self.RF180_post = {k:R[k][self.i_center_180:]   for k in ["gx","gy","gz","rf"]}

        dt = self.dt_in
        delayTE1 = np.ceil((self.TE/2 - self.T90 - self.center180)/dt)*dt
        delayTE2 = np.ceil((self.TE/2 - self.T180 + self.center180 - self.timeToTE)/dt)*dt

        self.z1 = np.zeros(int(delayTE1/dt))
        self.z2 = np.zeros(int(delayTE2/dt))


    # ----------------------------------------------------------
    # Assemble sequence (one repetition)
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # Assemble sequence (one repetition)
    # ----------------------------------------------------------
    def _assemble_sequence(self):
        def cat(keys): return np.concatenate(keys)
        self.Gx_seq = cat([
            self.FAT_interp["gx"],    # Fat-sat
            self.RF90_interp["gx"],   # 90° RF
            self.z1,                  # delayTE1
            self.RF180_interp["gx"],  # Full RF180
            self.z2,                  # delayTE2
            self.EPI_interp["gx"]     # EPI readout
        ])
        self.Gy_seq = cat([
            self.FAT_interp["gy"],
            self.RF90_interp["gy"],
            self.z1,
            self.RF180_interp["gy"],
            self.z2,
            self.EPI_interp["gy"]
        ])
        self.Gz_seq = cat([
            self.FAT_interp["gz"],
            self.RF90_interp["gz"],
            self.z1,
            self.RF180_interp["gz"],
            self.z2,
            self.EPI_interp["gz"]
        ])

        # set so RF of fat is 0.25, RF 90 is 0.5, RF 180 peak is 1.0, and EPI is 0.5 (for visualization)
        self.FAT_interp["rf"] = self.FAT_interp["rf"] * 0.25
        self.RF90_interp["rf"] = self.RF90_interp["rf"] * 0.5
        self.RF180_interp["rf"] = self.RF180_interp["rf"] * 1.0
        self.EPI_interp["rf"] = self.EPI_interp["rf"] * 0.5

        self.RF_seq = cat([
            self.FAT_interp["rf"],
            self.RF90_interp["rf"],
            self.z1,
            self.RF180_interp["rf"],
            self.z2,
            self.EPI_interp["rf"]
        ])


        n = len(self.Gx_seq)
        self.t_seq = np.arange(0, n*self.dt_in, self.dt_in)


    # ----------------------------------------------------------
    # Repeat sequence and store full repeated arrays
    # ----------------------------------------------------------
    def _repeat_sequence(self):
        n = self.n_repeats
        self.Gx_seq_full = np.tile(self.Gx_seq, n)
        self.Gy_seq_full = np.tile(self.Gy_seq, n)
        self.Gz_seq_full = np.tile(self.Gz_seq, n)
        self.RF_seq_full = np.tile(self.RF_seq, n)
        self.t_seq_full  = np.tile(self.t_seq, n)


    # ----------------------------------------------------------
    # Compute SAFE / CNS using gropt wrapper
    # ----------------------------------------------------------
    def _compute_safe(self):
        import gropt.gropt_wrapper as gw

        dt = self.dt_in
        self.safe_gx = np.array(gw.get_SAFE(self.Gx_seq_full, dt, safe_params=self.safe_params, new_first_axis=0))
        self.safe_gy = np.array(gw.get_SAFE(self.Gy_seq_full, dt, safe_params=self.safe_params, new_first_axis=1))
        self.safe_gz = np.array(gw.get_SAFE(self.Gz_seq_full, dt, safe_params=self.safe_params, new_first_axis=2))

        if self.safe_params_cardiac is not None:
            self.safe_cardiac_gx = np.array(gw.get_SAFE(self.Gx_seq_full, dt, safe_params=self.safe_params_cardiac, new_first_axis=0))
            self.safe_cardiac_gy = np.array(gw.get_SAFE(self.Gy_seq_full, dt, safe_params=self.safe_params_cardiac, new_first_axis=1))
            self.safe_cardiac_gz = np.array(gw.get_SAFE(self.Gz_seq_full, dt, safe_params=self.safe_params_cardiac, new_first_axis=2))



    # ----------------------------------------------------------
    # Truncate from RF90 center → end of sequence (including EPI)
    # ----------------------------------------------------------
    def _truncate_center90_to_TE2(self):
        # Indices of sequence components
        len_FAT     = len(self.FAT_interp["rf"])
        len_RF90    = len(self.RF90_interp["rf"])
        len_z1      = len(self.z1)
        len_RF180   = len(self.RF180_interp["rf"])
        len_z2      = len(self.z2)
        len_EPI     = len(self.EPI_interp["rf"])

        # Total length up to end of zeros_delayTE2
        idx_end_TE2 = len_FAT + len_RF90 + len_z1 + len_RF180 + len_z2 #+ len_EPI

        # Index of RF90 center relative to full sequence
        start = len_FAT + self.i_center_90
        end   = idx_end_TE2

        # Truncate sequences
        self.Gx_trunc = self.Gx_seq_full[start:end]
        self.Gy_trunc = self.Gy_seq_full[start:end]
        self.Gz_trunc = self.Gz_seq_full[start:end]
        self.RF_trunc = self.RF_seq_full[start:end]

        # Reset time array so 0 = start of RF90
        self.t_trunc  = self.t_seq_full[start:end] - self.t_seq_full[start]

        # Truncate SAFE/PNS & CNS
        self.safe_gx_trunc = self.safe_gx[start:end]
        self.safe_gy_trunc = self.safe_gy[start:end]
        self.safe_gz_trunc = self.safe_gz[start:end]

        if self.safe_params_cardiac is not None:
            self.safe_cardiac_gx_trunc = self.safe_cardiac_gx[start:end]
            self.safe_cardiac_gy_trunc = self.safe_cardiac_gy[start:end]
            self.safe_cardiac_gz_trunc = self.safe_cardiac_gz[start:end]


    # ----------------------------------------------------------
    # Resample truncated SAFE/PNS & CNS responses to dt_out
    # ----------------------------------------------------------
    def _resample_safe_cns(self):
        dt_out = self.dt_out
        t_old = self.t_trunc

        self.t_out, self.Gx_out = resample_to_raster(t_old, self.Gx_trunc, dt_out)
        _, self.Gy_out = resample_to_raster(t_old, self.Gy_trunc, dt_out)
        _, self.Gz_out = resample_to_raster(t_old, self.Gz_trunc, dt_out)

        _, self.safe_gx_out = resample_to_raster(t_old, self.safe_gx_trunc, dt_out)
        _, self.safe_gy_out = resample_to_raster(t_old, self.safe_gy_trunc, dt_out)
        _, self.safe_gz_out = resample_to_raster(t_old, self.safe_gz_trunc, dt_out)

        if self.safe_params_cardiac is not None:
            _, self.safe_cardiac_gx_out = resample_to_raster(t_old, self.safe_cardiac_gx_trunc, dt_out)
            _, self.safe_cardiac_gy_out = resample_to_raster(t_old, self.safe_cardiac_gy_trunc, dt_out)
            _, self.safe_cardiac_gz_out = resample_to_raster(t_old, self.safe_cardiac_gz_trunc, dt_out)


    # ----------------------------------------------------------
    # Optional plotting function
    # ----------------------------------------------------------
    def plot_all(self):
        fig, axs = plt.subplots(3,1,figsize=(16,12), sharex=True)

        # -----------------------------
        # Top: Gradients
        # -----------------------------
        axs[0].plot(self.t_trunc, self.Gx_trunc, label='Gx')
        axs[0].plot(self.t_trunc, self.Gy_trunc, label='Gy')
        axs[0].plot(self.t_trunc, self.Gz_trunc, label='Gz')
        axs[0].set_ylabel('Gradient (T/m)')
        axs[0].set_title('Truncated Gradients')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # -----------------------------
        # Middle: SAFE/PNS
        # -----------------------------
        axs[1].plot(self.t_trunc, self.safe_gx_trunc, label='SAFE Gx', color='blue')
        axs[1].plot(self.t_trunc, self.safe_gy_trunc, label='SAFE Gy', color='orange')
        axs[1].plot(self.t_trunc, self.safe_gz_trunc, label='SAFE Gz', color='green')
        axs[1].plot(self.t_trunc, 1 - self.safe_gx_trunc, '--', color='blue', label='1 - SAFE Gx')
        axs[1].plot(self.t_trunc, 1 - self.safe_gy_trunc, '--', color='orange', label='1 - SAFE Gy')
        axs[1].plot(self.t_trunc, 1 - self.safe_gz_trunc, '--', color='green', label='1 - SAFE Gz')
        axs[1].set_ylabel('SAFE (PNS)')
        axs[1].set_title('SAFE (PNS) Response')
        axs[1].legend(ncol=3, fontsize=9)
        axs[1].grid(True, alpha=0.3)

        # -----------------------------
        # Bottom: SAFE-CNS
        # -----------------------------
        if self.safe_params_cardiac is not None:
            axs[2].plot(self.t_trunc, self.safe_cardiac_gx_trunc, label='SAFE-CNS Gx', color='blue')
            axs[2].plot(self.t_trunc, self.safe_cardiac_gy_trunc, label='SAFE-CNS Gy', color='orange')
            axs[2].plot(self.t_trunc, self.safe_cardiac_gz_trunc, label='SAFE-CNS Gz', color='green')
            axs[2].plot(self.t_trunc, 1 - self.safe_cardiac_gx_trunc, '--', color='blue', label='1 - SAFE-CNS Gx')
            axs[2].plot(self.t_trunc, 1 - self.safe_cardiac_gy_trunc, '--', color='orange', label='1 - SAFE-CNS Gy')
            axs[2].plot(self.t_trunc, 1 - self.safe_cardiac_gz_trunc, '--', color='green', label='1 - SAFE-CNS Gz')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('SAFE-CNS')
        axs[2].set_title('Cardiac SAFE-CNS Response')
        axs[2].legend(ncol=3, fontsize=9)
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    # ==============================================================
    # Full Sequence Plot
    # ==============================================================
    def plot_full_sequence(self):

        plt.figure(figsize=(16,12))
        time_plot = np.arange(0, len(self.Gx_seq) * self.n_repeats * self.dt_in, self.dt_in)
        # Gradients
        plt.subplot(4,1,1)
        plt.plot(time_plot, self.Gx_seq_full, label='Gx')
        plt.plot(time_plot, self.Gy_seq_full, label='Gy')
        plt.plot(time_plot, self.Gz_seq_full, label='Gz')
        plt.ylabel('Gradient (T/m)')
        plt.title('Full Repeated Gradient Waveforms')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # RF
        plt.subplot(4,1,2)
        plt.plot(time_plot, self.RF_seq_full, label='RF', color='purple')
        plt.ylabel('Normalized RF')
        plt.title('Full RF Waveform')
        plt.grid(True, alpha=0.3)

        # SAFE / PNS
        plt.subplot(4,1,3)
        plt.plot(time_plot, self.safe_gx, label='SAFE Gx', color='blue')
        plt.plot(time_plot, self.safe_gy, label='SAFE Gy', color='orange')
        plt.plot(time_plot, self.safe_gz, label='SAFE Gz', color='green')
        plt.plot(time_plot, 1-self.safe_gx, '--', color='blue', label='1-SAFE Gx')
        plt.plot(time_plot, 1-self.safe_gy, '--', color='orange', label='1-SAFE Gy')
        plt.plot(time_plot, 1-self.safe_gz, '--', color='green', label='1-SAFE Gz')
        plt.ylabel('SAFE (PNS)')
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=9)

        # SAFE-CNS / Cardiac

        plt.subplot(4,1,4)
        if self.safe_params_cardiac is None:
            plt.plot(time_plot, self.safe_cardiac_gx, label='SAFE-CNS Gx', color='blue')
            plt.plot(time_plot, self.safe_cardiac_gy, label='SAFE-CNS Gy', color='orange')
            plt.plot(time_plot, self.safe_cardiac_gz, label='SAFE-CNS Gz', color='green')
            plt.plot(time_plot, 1-self.safe_cardiac_gx, '--', color='blue', label='1-SAFE-CNS Gx')
            plt.plot(time_plot, 1-self.safe_cardiac_gy, '--', color='orange', label='1-SAFE-CNS Gy')
            plt.plot(time_plot, 1-self.safe_cardiac_gz, '--', color='green', label='1-SAFE-CNS Gz')
        plt.xlabel('Time (s)')
        plt.ylabel('SAFE-CNS')
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=9)

        plt.tight_layout()
        plt.show()