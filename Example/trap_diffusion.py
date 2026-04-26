import numpy as np
import gropt  # for b-value and SAFE/PNS evaluation
import gc

"""
Updated Version should use this one now 
"""

def make_trapezoid(amplitude=0.1, rise_time=1e-3, flat_time=2e-3, fall_time=1e-3, dt=1e-5):
    """
    Generate a trapezoidal gradient waveform.

    Parameters
    ----------
    amplitude : float
        Peak gradient amplitude (T/m)
    rise_time : float
        Rise time to peak amplitude (s)
    flat_time : float
        Flat top duration (s)
    fall_time : float
        Fall time to zero (s)
    dt : float
        Dwell time (s)

    Returns
    -------
    g : np.ndarray
        Gradient waveform (T/m)
    t : np.ndarray
        Time axis (s)
    """
    n_rise = int(np.round(rise_time / dt))
    n_flat = int(np.round(flat_time / dt))
    n_fall = int(np.round(fall_time / dt))

    rise = np.linspace(0, amplitude, n_rise, endpoint=False)
    flat = amplitude * np.ones(n_flat)
    fall = np.linspace(amplitude, 0, n_fall)


    g = np.concatenate([rise, flat, fall])
    # if flat is 0 duration, g will be just rise and fall
    if flat_time == 0:
        g = np.concatenate([rise, fall])

    t = np.arange(g.size) * dt
    return g, t



def compute_bvalue(G, dt, TE, gamma=267.522190e6):
    """
    Compute b-value from gradient waveform G (T/m) with timestep dt (s).
    Flips the gradient sign after TE/2 before integration.
    Returns b-value in s/mm^2.
    """
    G = np.asarray(G)
    # ensure shape (N,1) for single axis or (N,3) for 3 axes
    if G.ndim == 1:
        G = G[:, None]
    elif G.ndim == 2 and G.shape[1] in (1,3):
        pass
    else:
        raise ValueError("G must be (N,), (N,1) or (N,3)")

    # convert gamma to rad/s/mm
    gamma = gamma / 1000.0

    # --- flip the sign after TE/2 ---
    idx_flip = int(np.floor((TE/2)/dt))
    if 0 < idx_flip < G.shape[0]:
        G = G.copy()
        G[idx_flip:] *= -1

    # cumulative integral
    G_int = np.cumsum(G, axis=0) * dt
    # b-value calculation
    bval = gamma**2 * dt * np.sum(G_int**2)

    
    return bval



import numpy as np
import gropt  # for b-value and SAFE/PNS evaluation
import gc

class GetMinTE_Trap:
    """
    Get minimum TE for trapezoidal diffusion waveforms (m0, m1, or m2 nulled)
    with given target b-value, RF timings, readout time, and hardware limits.
    Uses gropt for b-value and PNS/CNS safety evaluation.
    """

    def __init__(self, 
                 params = None,
                 gmaxRange=[0.100, 0.200], 
                 smaxRange=[100, 50], 
                 maxTE=150e-3, ):
        
        # -------------------------
        # defaults
        # -------------------------
        defaults = {
            'T_90': 2.835e-3,
            'T_180': 7.07e-3,
            'T_readout': 13.734e-3,
            'T_180_center': 0.00354075,
            'MMT': 0,
            'dt': 10e-6,
            'bvalue': 1000,
            'pns_lim': None,
            'cns_lim': None,
            'pns_params': None,
            'cns_params': None,
            'cns_idx':[0],
            'pns_idx':[0],
        }

        # ensure params exists
        if params is None:
            params = {}

        # fill missing fields individually
        params = params.copy() if params is not None else {}
        for k, v in defaults.items():
            params[k] = params.get(k, v)

        # -------------------------
        # assign to class
        # -------------------------
        self.targetBval = params['bvalue']
        self.rf_90_duration = params['T_90']
        self.rf_180_duration = params['T_180']
        self.timeToTE = params['T_readout']

        self.pnsThresh = params['pns_lim']
        self.cnsThresh = params['cns_lim']

        self.safe_params = params['pns_params']
        self.safe_params_cardiac = params['cns_params']

        self.pns_idx = params['pns_idx']
        self.cns_idx = params['cns_idx']


        if self.safe_params_cardiac is None:
            self.safe_params_cardiac = self.safe_params
            print("Warning: cns_params not provided, using pns_params for both PNS and CNS checks.")

            
        self.gmax_range = gmaxRange
        self.smax_range = smaxRange
        self.dt = params['dt']
        self.rf_180_rfCenterInclDelay = params['T_180_center']
        self.maxTE = maxTE
        self.mmt = params['MMT']  # moment order to null  up to(0,1,2)

        self.pnsThresh_value = self.pnsThresh
        self.cnsThresh_value = self.cnsThresh

        
        # Put all timings on the raster 
        self.rf_90_duration = np.ceil(self.rf_90_duration / self.dt) * self.dt
        self.rf_180_duration = np.ceil(self.rf_180_duration / self.dt) * self.dt
        self.rf_180_rfCenterInclDelay = np.ceil(self.rf_180_rfCenterInclDelay / self.dt) * self.dt
        self.timeToTE = np.ceil(self.timeToTE / self.dt) * self.dt
        self.maxTE = np.ceil(self.maxTE / self.dt) * self.dt


    def build_waveform(self, diffGrad, delay1_total, delay2_total, timings=None, idle_pre=0.0, idle_post_diff=0.0):
        """
        Build waveform array:
          RF90 zeros + idle_pre (after RF90) + diffusion + idle_post_diff (gap before RF180)
          + RF180 zeros + flipped diffusion + delay2
        """
        n90 = int(np.round(self.rf_90_duration / self.dt))
        n180 = int(np.round(self.rf_180_duration / self.dt))
        n_idle_post_diff = int(np.round(idle_post_diff / self.dt))

        n_lobe = diffGrad.size
        lobe_time = n_lobe * self.dt

        diffGrad_flip = np.flip(diffGrad) if abs(self.mmt) == 2 else diffGrad
        
        g = np.concatenate([
            np.zeros(n90),
            diffGrad,
            np.zeros(n_idle_post_diff),
            np.zeros(n180),
            diffGrad_flip,
        ])

        if self.mmt == 2:
            n_idle_pre = int(np.round(idle_pre / self.dt))
            g = np.concatenate([
            np.zeros(n90),
            np.zeros(n_idle_pre),
            diffGrad,
            np.zeros(n_idle_post_diff),
            np.zeros(n180),
            diffGrad_flip,
                ])
            
        if self.mmt == -2:
            n_idle_pre = int(np.round(idle_pre / self.dt))
            g = np.concatenate([
            np.zeros(n90),
            diffGrad,
            np.zeros(n_idle_post_diff),
            np.zeros(n180),
            diffGrad_flip,
                ])


        return g
    
    def _check_safe(self, g):
        """
        Compute PNS and CNS safety checks for a waveform g.

        Uses axis-specific time-varying thresholds:
        self.pnsThresh = [pns_x, pns_y, pns_z]
        self.cnsThresh = [cns_x, cns_y, cns_z]

        Returns:
            (pns_ok, cns_ok): booleans indicating if all axes pass.
        """
        pns_ok_all = []
        cns_ok_all = []


        # -------------------------
        # PNS CHECK
        # -------------------------
        pns_ok_all = []

        for axis in self.pns_idx:
            safe = np.array(gropt.gropt_wrapper.get_SAFE(
                g, self.dt,
                safe_params=self.safe_params,
                new_first_axis=axis
            ))

            pns_thresh = np.asarray(self.pnsThresh[axis])

            # interpolate if needed
            if pns_thresh.size != safe.size:
                pns_thresh = np.interp(
                    np.linspace(0, 1, safe.size),
                    np.linspace(0, 1, pns_thresh.size),
                    pns_thresh
                )

            pns_ok_all.append(np.all(safe <= pns_thresh))


        # -------------------------
        # CNS CHECK
        # -------------------------
        cns_ok_all = []

        for axis in self.cns_idx:
            safe_cardiac = np.array(gropt.gropt_wrapper.get_SAFE(
                g, self.dt,
                safe_params=self.safe_params_cardiac,
                new_first_axis=axis
            ))

            cns_thresh = np.asarray(self.cnsThresh[axis])

            # interpolate if needed
            if cns_thresh.size != safe_cardiac.size:
                cns_thresh = np.interp(
                    np.linspace(0, 1, safe_cardiac.size),
                    np.linspace(0, 1, cns_thresh.size),
                    cns_thresh
                )

            cns_ok_all.append(np.all(safe_cardiac <= cns_thresh))

        # Overall OK if all axes pass
        return np.all(pns_ok_all), np.all(cns_ok_all)


    def evaluate_waveform(self, g, TE):
        """
        Evaluate waveform g for b-value and PNS/CNS safety.
        Returns (bval, pns_ok, cns_ok).
        """
        bval = compute_bvalue(g, self.dt, TE)
        if self.pnsThresh is not None:
            pns_ok, cns_ok = self._check_safe(g)
        else:
            pns_ok, cns_ok = True, True
        return bval, pns_ok, cns_ok


    def construct_diffGrad(self, gmax, smax, delayTE2_min,delayTE1_min = None):
        """
        Construct a trapezoidal diffusion gradient waveform that fits within
        the given maximum gradient amplitude (gmax), maximum slew rate (smax),
        and matches or slightly exceeds delayTE2_min (rounded up to raster).

        Parameters
        ----------
        gmax : float
            Gradient amplitude (T/m).
        smax : float
            Slew rate (T/m/s).
        delayTE2_min : float
            Desired minimum duration (s).

        Returns
        -------
        diffGrad : np.ndarray
            1D array of the gradient waveform on the raster (self.dt).
        timings : dict
            Dictionary with keys:
            - "rise_time" : float
            - "flat_time" : float
            - "fall_time" : float
            - "total_time": float
        """
        # --- Rise/fall time from slew rate ---
        rise_time = gmax / smax
        rise_points = int(np.ceil(rise_time / self.dt))

        flag = True
        # Rasterized ramps
        ramp_up = np.linspace(0, gmax, rise_points, endpoint=False)
        ramp_down = ramp_up[::-1]

        # Flat portion estimate
        if self.mmt == 0:
            flat_time_est = delayTE2_min - 2 * rise_time
            flat_points = int(np.ceil(flat_time_est / self.dt)) if flat_time_est > 0 else 0
            flat = np.full(flat_points, gmax)

        elif self.mmt == 1:
            flat_time_est = (delayTE2_min - 4 * rise_time) / 2
            flat_points = int(np.ceil(flat_time_est / self.dt)) if flat_time_est > 0 else 0
            flat = np.full(flat_points, gmax)
        
        elif self.mmt == 2:
            # --- First lobe (positive) ---
            flat1_est = (delayTE2_min - 5 * rise_time) / 3
            flat1_points = int(np.ceil(flat1_est / self.dt)) if flat1_est > 0 else 0
            flat1 = np.full(flat1_points, gmax)
            # Duration of first lobe
            
            pos_lobe = np.concatenate([ramp_up, flat1, ramp_down])
            # --- Second lobe (negative) ---
            # flat1_points is already an int
            flat2_points = 2 * flat1_points + rise_points  # total time in seconds
            #print(remaining_time)
            flat2 = np.full(flat2_points, gmax)
            neg_lobe = np.concatenate([ramp_up, flat2, ramp_down]) * -1

        elif self.mmt == -2:
            T_180_points = int(np.ceil(self.rf_180_duration / self.dt))
            r = rise_points

            delayTE2_points = int(np.ceil(delayTE2_min / self.dt))
            delayTE1_points = int(np.ceil(delayTE1_min / self.dt))
            T = delayTE2_points
            delay = delayTE1_points - delayTE2_points + T_180_points

            # Solve for M2-nulled flat durations
            f1, f2 = M2_helper(T, r, delay)
            arb_amp0 = gmax

            # Handle invalid or negative durations
            if f1 <= 0 or f2 <= 0:
                f1, f2 = max(f1, 0), max(f2, 0)
                flag = False

            # -----------------------------------------
            # Enforce total duration = delayTE2_points
            # -----------------------------------------
            f1 = int(np.ceil(f1))
            f2 = int(np.ceil(f2))

            total_len = f1 + f2 + 4 * r
            target_len = T

            diff = total_len - target_len
            if diff != 0 and (f1 + f2) > 0:
                # Remove or add proportionally from both lobes
                frac1 = f1 / (f1 + f2)
                frac2 = f2 / (f1 + f2)

                f1_adj = f1 - np.round(diff * frac1)
                f2_adj = f2 - np.round(diff * frac2)

                # Prevent negatives due to rounding
                f1 = max(int(f1_adj), 0)
                f2 = max(int(f2_adj), 0)

            # Recheck total duration after adjustment
            total_len = f1 + f2 + 4 * r
            if total_len != target_len:
                # Small correction if still off due to rounding
                correction = target_len - total_len
                f2 = max(f2 + correction, 0)

            # -----------------------------------------
            # Construct lobes on raster
            # -----------------------------------------
            f1 = np.full(f1, arb_amp0)
            f2 = np.full(f2, arb_amp0)

            pos_lobe = np.concatenate([ramp_up, f1, ramp_down])
            neg_lobe = np.concatenate([ramp_up, f2, ramp_down]) * -1

        # Initial waveform
        if self.mmt < 2 and self.mmt >=0:
            diffGrad = np.concatenate([ramp_up, flat, ramp_down])

        if self.mmt == 1:
            diffGrad = np.concatenate([diffGrad, diffGrad * -1])

        if abs(self.mmt) == 2:
            diffGrad = np.concatenate([pos_lobe, neg_lobe])

        # --- Adjust to match delayTE2_min on raster ---
        if self.mmt == 0:
            # monopolar
            target_points = int(np.ceil(delayTE2_min / self.dt))
            current_points = diffGrad.size
            excess_points = current_points - target_points

            if flat_points > 0:
                # adjust flat only
                flat_points -= excess_points
                flat_points = max(flat_points, 0)
                flat = flat[:flat_points] if flat_points > 0 else np.array([])
                diffGrad = np.concatenate([ramp_up, flat, ramp_down])
            else:
                # scale ramps proportionally if flat == 0
                total_ramp_points = ramp_up.size + ramp_down.size
                scale = target_points / total_ramp_points
                new_rise_points = max(int(np.ceil(ramp_up.size * scale)), 1)
                ramp_up_scaled = np.linspace(0, gmax * scale, new_rise_points, endpoint=False)
                ramp_down_scaled = ramp_up_scaled[::-1]
                diffGrad = np.concatenate([ramp_up_scaled, ramp_down_scaled])

        elif self.mmt == 1:
            # bipolar (M1 nulling) -> treat each lobe as half the target duration
            total_target_points = int(np.ceil(delayTE2_min / self.dt))
            half_target_points = total_target_points // 2

            # original positive lobe points
            ramp_points = ramp_up.size + ramp_down.size
            current_half_points = ramp_points + flat.size

            if current_half_points > half_target_points:
                # scale ramps and flat proportionally
                scale = half_target_points / current_half_points
                new_ramp_up_points = max(int(np.ceil(ramp_up.size * scale)), 1)
                new_ramp_down_points = max(int(np.ceil(ramp_down.size * scale)), 1)
                new_flat_points = max(half_target_points - new_ramp_up_points - new_ramp_down_points, 0)

                ramp_up_scaled = np.linspace(0, gmax * scale, new_ramp_up_points, endpoint=False)
                ramp_down_scaled = np.linspace(gmax * scale, 0, new_ramp_down_points, endpoint=False)
                flat_scaled = np.full(new_flat_points, gmax * scale) if new_flat_points > 0 else np.array([])

                pos_lobe = np.concatenate([ramp_up_scaled, flat_scaled, ramp_down_scaled])

            elif current_half_points < half_target_points:
                # pad flat if needed
                pad_points = half_target_points - current_half_points
                pos_lobe = np.concatenate([ramp_up, flat, ramp_down, np.full(pad_points, gmax)])

            else:
                # already fits exactly
                pos_lobe = np.concatenate([ramp_up, flat, ramp_down])

            # mirror to negative lobe
            diffGrad = np.concatenate([pos_lobe, -pos_lobe])

            # update ramps for timings
            ramp_up = ramp_up[:ramp_up.size] if ramp_up.size > 0 else np.array([])
            ramp_down = ramp_down[:ramp_down.size] if ramp_down.size > 0 else np.array([])
            flat = pos_lobe[ramp_up.size : ramp_up.size + flat.size] if flat.size > 0 else np.array([])


            
        # Final timings (all rasterized)
        rise_time = ramp_up.size * self.dt
        flat_time = max(diffGrad.size - 2 * ramp_up.size, 0) * self.dt
        
        if self.mmt == 1:
            ramp_up = ramp_up[:ramp_up.size] if ramp_up.size > 0 else np.array([])
            ramp_down = ramp_down[:ramp_down.size] if ramp_down.size > 0 else np.array([])
            flat = pos_lobe[ramp_up.size : ramp_up.size + flat.size] if flat.size > 0 else np.array([])

            rise_time = ramp_up.size * self.dt
            flat_time = flat.size * self.dt

        if self.mmt == 2:
            # use actual lengths of the rasterized flat portions
            flat_time = flat1.size * self.dt
            flat_time2 = flat2.size * self.dt

        
        fall_time = ramp_down.size * self.dt
        total_time = diffGrad.size * self.dt

        timings = {
            "rise_time": rise_time,
            "flat_time": flat_time,
            'flat_time2': flat_time2 if self.mmt==2 else None,
            "fall_time": fall_time,
            "total_time": total_time,

        }

        return diffGrad, timings, flag

   
    def compute(self,terminate_early=False, start_TE = None):
        

        print('Making trapezoid waveform...')
        gradRasterTime = self.dt
        TE = max([
            np.ceil((self.rf_90_duration + self.rf_180_rfCenterInclDelay) / gradRasterTime) * gradRasterTime,
            np.ceil((self.rf_180_duration - self.rf_180_rfCenterInclDelay + self.timeToTE) / gradRasterTime) * gradRasterTime
        ]) * 2
        
        if start_TE is not None:
            TE = max(TE, start_TE)
        
        iteration = -1
        while TE < self.maxTE:
            iteration += 1
            
            
            
            delayTE1_min = np.ceil((TE / 2 - self.rf_90_duration - self.rf_180_rfCenterInclDelay) / gradRasterTime) * gradRasterTime
            delayTE2_min = np.ceil((TE / 2 - self.rf_180_duration + self.rf_180_rfCenterInclDelay - self.timeToTE) / gradRasterTime) * gradRasterTime
            
            if iteration % 20 == 0:
                print('Trying TE={:.2f} ms: delayTE1_min={:.2f} ms, delayTE2_min={:.2f} ms, PNS={:.2f} ms'.format(TE*1e3, delayTE1_min*1e3, delayTE2_min*1e3, self.pnsThresh_value*1e3))
            
            # Timing Checks 
            if np.ceil(np.max(self.gmax_range) / np.max(self.smax_range) / self.dt) * self.dt * 2 > delayTE2_min:
                #print('Skipping TE={:.2f} ms because minimum rise time ({:.2f} ms) exceeds delayTE2_min ({:.2f} ms)'.format(TE*1e3, np.ceil(np.max(self.gmax_range) / np.max(self.smax_range) / self.dt) * self.dt * 2, delayTE2_min))
                # increment TE by the difference in min rise time
                TE += (np.ceil(np.max(self.gmax_range) / np.max(self.smax_range) / self.dt) * self.dt * 2 - delayTE2_min)
                if self.mmt == 1:
                    TE += (np.ceil(np.max(self.gmax_range) / np.max(self.smax_range) / self.dt) * self.dt * 2)
                continue

            if self.mmt == 2:
                tmp_rise = np.ceil(np.max(self.gmax_range) / np.max(self.smax_range) / self.dt)
                tmp_flat = (delayTE2_min - 5 * tmp_rise * self.dt) / 3 
                tmp_grad_dur = tmp_rise * 4 + tmp_flat + tmp_flat *2 + tmp_rise            
                
                if delayTE2_min < 5 * tmp_rise + 3*(self.rf_180_duration - 2*tmp_rise):
                    #print('Skipping TE={:.2f} ms because minimum delayTE2 ({:.2f} ms) is less than minimum required for M2 ({:.2f} ms)'.format(TE*1e3, delayTE2_min*1e3, (5 * tmp_rise + 3* self.rf_180_duration - 2*tmp_rise)*self.dt*1e3))
                    TE += (5 * tmp_rise + 3* self.rf_180_duration - 2*tmp_rise)*self.dt - delayTE2_min
                    continue
                if tmp_grad_dur < 0: # This TE is too short for M2
                    #print('Skipping TE={:.2f} ms because minimum gradient duration ({:.2f} ms) is negative for M2'.format(TE*1e3, tmp_grad_dur*1e3))
                    # add the difference to TE
                    TE += -tmp_grad_dur
                    continue

                if 2 * tmp_rise + tmp_flat < self.rf_180_duration:
                    #print('Skipping TE={:.2f} ms because minimum gradient duration ({:.2f} ms) is less than RF180 duration ({:.2f} ms) for M2'.format(TE*1e3, (2 * tmp_rise + tmp_flat)*1e3, self.rf_180_duration*1e3))
                    TE += self.rf_180_duration - (2 * tmp_rise + tmp_flat)
                    
                    continue

  
            diffGrad, timings, flag = self.construct_diffGrad(np.max(self.gmax_range), np.max(self.smax_range), delayTE2_min, delayTE1_min = delayTE1_min)

            
            
            #delayTE1_min, delayTE2_min, __, adjusted, idle_pre, idle_post_diff = self._ensure_delays_for_lobe(diffGrad, delayTE1_min, delayTE2_min, TE, gradRasterTime, timings)
            #idle_post_diff = np.ceil((delayTE1_min - diffGrad.size * self.dt) / gradRasterTime) * gradRasterTime 
            idel_pre= 0
            
            grad_dur = diffGrad.size * self.dt
            # compute idle_post_diff to cover remaining time, then ceil to raster
            idle_post_diff = delayTE1_min - grad_dur - idel_pre 
            idle_post_diff = np.round(idle_post_diff / gradRasterTime) * gradRasterTime

            # for mmt special calculation: 
            if self.mmt == 2:
                gap = timings['rise_time']*2 + timings['flat_time']
                idle_post_diff = np.ceil(max(gap - self.rf_180_duration, 0.0) / gradRasterTime) * gradRasterTime
                idel_pre = np.ceil(max(delayTE1_min - grad_dur - idle_post_diff, 0.0) / gradRasterTime) * gradRasterTime
            
            if self.mmt <2 and self.mmt > 0:
                assert abs(idle_post_diff + grad_dur - delayTE1_min) <= gradRasterTime, \
                                            f"idle_post_diff calculation error: should be {delayTE1_min - grad_dur}, is {idle_post_diff}"
            
            
            g = self.build_waveform(diffGrad, delayTE1_min, delayTE2_min, timings=timings, idle_pre=idel_pre, idle_post_diff=idle_post_diff)
            t = np.arange(g.size) * self.dt
            TE = g.size * self.dt + self.timeToTE
            b, pns_ok, cns_ok = self.evaluate_waveform(g, TE)
            
            if terminate_early:
                # return the first waveform regardless of anything else
                return TE, g, t, b, {'TE': TE, 'idle_pre': idel_pre, 'idle_post_diff': idle_post_diff, 'gap': timings.get('gap', None)}


            if b < self.targetBval and flag is True:
                del g, t, diffGrad  # free memory
                TE += gradRasterTime 
                continue
            
            if pns_ok and cns_ok and b >= self.targetBval and flag is True:
                if timings['flat_time'] <= 0 and self.mmt == 2:
                    print('Skipping TE={:.2f} ms because flat time is zero for M2'.format(TE*1e3))
                    TE += gradRasterTime
                    continue

                return TE, g, t, b, {'TE': TE, 'idle_pre': idel_pre, 'idle_post_diff': idle_post_diff, 'gap': timings.get('gap', None)}
            

            if b > self.targetBval and flag is True:
                # Scale amplitude to hit target b-value
                b_fixed = compute_bvalue(g, self.dt, TE)
                scale_factor = np.sqrt(self.targetBval / b_fixed)
                g_scaled = g * scale_factor
                new_b = compute_bvalue(g_scaled, self.dt, TE)

                # Check safety
                pns_ok, cns_ok = self._check_safe(g_scaled)
                if pns_ok and cns_ok:
                    t = np.arange(g_scaled.size) * self.dt
                    print(f"Pre-check successful at fixed TE={TE*1e3:.2f} ms with scaled b-value={self.targetBval}")
                    
                    timings_out = {
                    'TE': TE,
                    'rise_time': timings['rise_time'],
                    'fall_time': timings['fall_time'],
                    'flat_time': timings['flat_time'],
                    'pns_thresh': self.pnsThresh,
                    'cns_thresh': self.cnsThresh,
                    'gmax': np.max(g_scaled),
                    'smax': np.max(self.smax_range),
                    'idle_pre': idel_pre,
                    'n_idle1_seconds': idle_post_diff,
                    'gap': idle_post_diff,
                    }
                    print(timings_out.keys())
                    
                    return TE, g_scaled, t, b_fixed, timings_out
                
                if not (pns_ok and cns_ok):
                    #print(f"Tried fixed TE and scaling b_fixed = {new_b:.2f} but safety check FAILED")
                    # Check if variables exist and delete them
                    for var_name in ['g', 't', 'diffGrad', 'diffGrad_cand', 'g_test', 'g_best', 'g_scaled']:
                        if var_name in locals():
                            del locals()[var_name]

 
                               
            if iteration % 20 == 0:
                print('Starting TE sweep with TE={:.2f} ms'.format(TE*1e3))
            
            valid_results = []

            for gmax in self.gmax_range:
                for smax in self.smax_range:
                    diffGrad_cand, timings_cand, flag = self.construct_diffGrad(gmax, smax, delayTE2_min, delayTE1_min=delayTE1_min)
                    grad_dur_cand = diffGrad_cand.size * self.dt
                    idle_post_diff_cand = np.ceil((delayTE1_min - diffGrad_cand.size * self.dt) / gradRasterTime) * gradRasterTime 
            
                    
                    idle_pre_cand = 0.0
                    if self.mmt == 2:
                        gap = timings_cand['rise_time']*2 + timings_cand['flat_time'] 
                        idle_post_diff_cand = np.ceil(max(gap  - self.rf_180_duration, 0.0) / gradRasterTime) * gradRasterTime
                        idle_pre_cand = np.ceil(max(delayTE1_min - grad_dur_cand - idle_post_diff_cand, 0.0) / gradRasterTime) * gradRasterTime

                        if timings_cand['flat_time'] <= 0:
                            continue
  
                    if self.mmt <2 and flag == False:
                        continue
                    
                    g_test = self.build_waveform(diffGrad_cand, delayTE1_min, delayTE2_min, timings=timings_cand, idle_pre=idle_pre_cand, idle_post_diff=idle_post_diff_cand)
                    TE_test = g_test.size * self.dt + self.timeToTE
                    
                    
                    b_test = compute_bvalue(g_test, self.dt, TE_test)
                    if b_test < self.targetBval: 
                        continue
                    
                    pns_ok, cns_ok = self._check_safe(g_test)
                    
                    
                    if pns_ok and cns_ok:
                        valid_results.append((TE_test, gmax, smax, g_test, b_test, timings_cand, idle_pre_cand, idle_post_diff_cand))
                        print(f"\tValid candidate gmax={gmax:.3f} smax={smax:.3f}  b={b_test:.2f} TE={TE_test*1e3:.2f} ms")

                    if not (pns_ok and cns_ok):
                        # scale g to see if it hits target bval safely
                        scale_factor = np.sqrt(self.targetBval / b_test)
                        g_scaled = g_test * scale_factor
                        new_b = compute_bvalue(g_scaled, self.dt, TE_test)
                        pns_ok, cns_ok = self._check_safe(g_scaled)
                        if pns_ok and cns_ok:
                            valid_results.append((TE_test, gmax, smax, g_scaled, new_b, timings_cand, idle_pre_cand, idle_post_diff_cand))
                            print(f"\tValid candidate (after scaling) gmax={gmax:.3f} smax={smax:.3f}  b={new_b:.2f} TE={TE_test*1e3:.2f} ms")

                    
                    for var_name in ['g', 't', 'diffGrad', 'diffGrad_cand', 'g_test', 'g_best', 'g_scaled']:
                        if var_name in locals():
                            del locals()[var_name]

            if valid_results:
                best = min(valid_results, key=lambda x: (x[0], abs(x[4] - self.targetBval)))  # choose min TE and then closest bval
                TE_best, gmax_b, smax_b, g_best, b_best, timings_best, idle_pre_best, idle_post_best = best
                t_best = np.arange(g_best.size) * self.dt
                print(f"Selected candidate: TE={TE_best*1e3:.2f} ms b={b_best:.0f} with gmax={gmax_b} smax={smax_b}")
                
                # Scale factor: b ~ amplitude^2 for trapezoids, so g_new = g_old * sqrt(targetB / currentB)
                scale_factor = np.sqrt(self.targetBval / b_best)
                g_adjusted = g_best * scale_factor

                # Recompute b-value to verify
                b_adjusted = compute_bvalue(g_adjusted, self.dt, TE_best)
                print(f"Adjusted gmax to hit exact B-value: {b_adjusted:.2f}")
                b_best = b_adjusted
                g_best = g_adjusted
                
                
                timings_out = {
                    'TE': TE_best,
                    'rise_time': timings_best['rise_time'],
                    'fall_time': timings_best['fall_time'],
                    'flat_time': timings_best['flat_time'],
                    'pns_thresh': self.pnsThresh,
                    'cns_thresh': self.cnsThresh,
                    'delayTE1'
                    'gmax': gmax_b,
                    'smax': smax_b,
                    'idle_pre': idle_pre_best,
                    'n_idle1_seconds': idle_post_best,
                    'gap': timings_best.get('gap', None)
                }

                print(timings_out.keys())
                return TE_best, g_best, t_best, b_best, timings_out

           
            # Check if variables exist and delete them
            for var_name in ['g', 't', 'diffGrad', 'diffGrad_cand', 'g_test', 'g_best', 'g_scaled']:
                if var_name in locals():
                    del locals()[var_name]

            
            gc.collect()        # force garbage collection
            TE += gradRasterTime*2 

        raise RuntimeError("No valid TE found within maxTE.")




    def compute_binary(self, terminate_early=False, start_TE=None, tol = 1e-6,max_iter= 10000, pns_vec_dict = None):
        """
        Compute the minimum TE and corresponding gradient waveform using binary search.
        Inputs:
            - terminate_early: if True, return the first valid waveform found without searching for minimum TE
            - start_TE: optional starting TE for the search (in seconds)
            - tol: tolerance for convergence (in seconds)
            - max_iter: maximum number of iterations to prevent infinite loops
        Outputs:
            - TE_best: minimum TE found (in seconds)
            - g_best: corresponding gradient waveform (numpy array)
            - t_best: time vector for the gradient waveform (numpy array)
            - b_best: b-value of the best waveform
            - timings_best_out: dictionary of timing parameters for the best waveform
        """
        print('Making trapezoid waveform (binary search)...')
        gradRasterTime = self.dt

        # Initial minimum TE based on RF durations
        TE_min = max([
            np.ceil((self.rf_90_duration + self.rf_180_rfCenterInclDelay) / gradRasterTime) * gradRasterTime,
            np.ceil((self.rf_180_duration - self.rf_180_rfCenterInclDelay + self.timeToTE) / gradRasterTime) * gradRasterTime
        ]) * 2
        if start_TE is not None:
            TE_min = max(TE_min, start_TE)
        print('Start TE is {:.2f} ms'.format(TE_min*1e3))
        TE_low = TE_min
        TE_high = self.maxTE

        TE_best = None
        g_best = None
        t_best = None
        timings_best_out = None
        b_best = None

        iteration  = 0

        print('Searching TE: ', end=' ')
        # Binary search for minimum TE
        while TE_high - TE_low > tol and iteration < max_iter :  # precision threshold
            TE = (TE_low + TE_high) / 2

            iteration += 1
            delayTE1_min = np.ceil((TE / 2 - self.rf_90_duration - self.rf_180_rfCenterInclDelay) / gradRasterTime) * gradRasterTime
            delayTE2_min = np.ceil((TE / 2 - self.rf_180_duration + self.rf_180_rfCenterInclDelay - self.timeToTE) / gradRasterTime) * gradRasterTime


            gparams = gropt.GroptParams()
            gparams.diff_init(T_90=self.rf_90_duration - self.rf_180_rfCenterInclDelay, 
                                T_180=self.rf_180_rfCenterInclDelay, T_readout=self.timeToTE, TE=TE, dt=self.dt)
            N = gparams.N

            self.pnsThresh = [self.pnsThresh_value * np.ones(N), self.pnsThresh_value* np.ones(N), self.pnsThresh_value* np.ones(N)]
            self.cnsThresh = [self.pnsThresh_value * np.ones(N), self.pnsThresh_value * np.ones(N), self.pnsThresh_value * np.ones(N)]

            if iteration % 1 == 0:
                #print('Trying TE={:.2f} ms: delayTE1_min={:.2f} ms, delayTE2_min={:.2f} ms'.format(TE*1e3, delayTE1_min*1e3, delayTE2_min*1e3))
                print('{:.2f}'.format(TE*1e3), end=' ')

            
            if self.mmt == 2:
                tmp_rise = np.ceil(np.max(self.gmax_range) / np.max(self.smax_range) / self.dt)
                tmp_flat = (delayTE2_min - 5 * tmp_rise * self.dt) / 3
                tmp_grad_dur = tmp_rise * 4 + tmp_flat + tmp_flat * 2 + tmp_rise

                if delayTE2_min < 5 * tmp_rise + 3*(self.rf_180_duration - 2*tmp_rise):
                    TE_low = TE + gradRasterTime
                    continue
                if tmp_grad_dur < 0:
                    TE_low = TE + gradRasterTime
                    continue
                if 2 * tmp_rise + tmp_flat < self.rf_180_duration:
                    TE_low = TE + gradRasterTime
                    continue

            # -------------------------------------
            # Evaluate all gmax/smax combinations
            # -------------------------------------
            valid_results = []

            for gmax in self.gmax_range:
                for smax in self.smax_range:
                    diffGrad_cand, timings_cand, flag = self.construct_diffGrad(gmax, smax, delayTE2_min, delayTE1_min=delayTE1_min)
                    grad_dur_cand = diffGrad_cand.size * self.dt
                    idle_post_diff_cand = np.ceil((delayTE1_min - grad_dur_cand) / gradRasterTime) * gradRasterTime
                    idle_pre_cand = 0.0

                    #print(gmax,smax, grad_dur_cand, delayTE2_min, idle_post_diff_cand)

                    if self.mmt == 2:
                        gap = timings_cand['rise_time']*2 + timings_cand['flat_time']
                        idle_post_diff_cand = np.ceil(max(gap - self.rf_180_duration, 0.0) / gradRasterTime) * gradRasterTime
                        idle_pre_cand = np.ceil(max(delayTE1_min - grad_dur_cand - idle_post_diff_cand, 0.0) / gradRasterTime) * gradRasterTime

                        if timings_cand['flat_time'] <= 0:
                            continue

                    if self.mmt == -2 and flag is False:
                        continue
                    

                    g_test = self.build_waveform(diffGrad_cand, delayTE1_min, delayTE2_min,
                                                timings=timings_cand, idle_pre=idle_pre_cand, idle_post_diff=idle_post_diff_cand)
                    TE_test = g_test.size * self.dt + self.timeToTE
                    b_test = compute_bvalue(g_test, self.dt, TE_test)
                    if b_test < self.targetBval:
                        continue

                    pns_ok, cns_ok = self._check_safe(g_test)
                    if pns_ok and cns_ok:
                        valid_results.append((TE_test, gmax, smax, g_test, b_test, timings_cand, idle_pre_cand, idle_post_diff_cand))
                    else:
                        # scale g to see if it hits target bval safely
                        scale_factor = np.sqrt(self.targetBval / b_test)
                        g_scaled = g_test * scale_factor
                        new_b = compute_bvalue(g_scaled, self.dt, TE_test)
                        pns_ok, cns_ok = self._check_safe(g_scaled)
                        if pns_ok and cns_ok:
                            valid_results.append((TE_test, gmax, smax, g_scaled, new_b, timings_cand, idle_pre_cand, idle_post_diff_cand))

                    # cleanup
                    for var_name in ['g_test', 'diffGrad_cand', 'g_scaled']:
                        if var_name in locals():
                            del locals()[var_name]

            # -------------------------------------
            # Pick best candidate for this TE
            # -------------------------------------
            if valid_results:
                best = min(valid_results, key=lambda x: (x[0], abs(x[4] - self.targetBval)))
                TE_candidate, gmax_b, smax_b, g_candidate, b_candidate, timings_candidate, idle_pre_candidate, idle_post_candidate = best

                TE_best = TE_candidate
                g_best = g_candidate
                t_best = np.arange(g_best.size) * self.dt
                b_best = b_candidate
                timings_best_out = {
                    'TE': TE_candidate,
                    'rise_time': timings_candidate['rise_time'],
                    'fall_time': timings_candidate['fall_time'],
                    'flat_time': timings_candidate['flat_time'],
                    'pns_thresh': self.pnsThresh,
                    'cns_thresh': self.cnsThresh,
                    'diff_time': timings_candidate['total_time'],
                    'gmax': gmax_b,
                    'smax': smax_b,
                    'idle_pre': idle_pre_candidate,
                    'n_idle1_seconds': idle_post_candidate,
                    'gap': timings_candidate.get('gap', None)
                }

                # Binary search: try smaller TE
                TE_high = TE - gradRasterTime
            else:
                # TE too small: increase
                TE_low = TE + gradRasterTime

            # cleanup memory
            gc.collect()

        if TE_best is None:
            raise RuntimeError("No valid TE found within maxTE.")

        # Scale to exact target b-value
        b_fixed = compute_bvalue(g_best, self.dt, TE_best)
        scale_factor = np.sqrt(self.targetBval / b_fixed)
        g_scaled = g_best * scale_factor
        t_best = np.arange(g_scaled.size) * self.dt
        b_scaled = compute_bvalue(g_scaled, self.dt, TE_best)

        print(f"\nBest TE found with binary search: {TE_best*1e3:.3f} ms, b-value: {b_scaled}")
        return TE_best, g_scaled, t_best, b_scaled, timings_best_out
    

import numpy as np



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


def interp_grad(t, g, tt_target):
    """Linear interpolation to target time grid"""
    f = interp1d(t, g, kind='linear', fill_value=0, bounds_error=False)
    return f(tt_target)