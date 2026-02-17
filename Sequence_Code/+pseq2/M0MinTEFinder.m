classdef M0MinTEFinder
    properties
        timingParamsFile
        mriSystem
        targetBval
        pnsThresh
        gmaxRange
        smaxRange
        sys
        rf_90_duration
        rf_90_rfCenterInclDelay
        rf_180_duration
        rf_180_rfCenterInclDelay
        nav_dur
    end

    methods
        function obj = M0MinTEFinder(timingParamsFile, mriSystem, targetBval, pnsThresh, gmaxRange, smaxRange)
            obj.timingParamsFile = timingParamsFile;
            obj.mriSystem = mriSystem;
            obj.targetBval = targetBval;
            obj.pnsThresh = pnsThresh;
            obj.gmaxRange = gmaxRange;
            obj.smaxRange = smaxRange;

            obj.sys = mr.opts('MaxGrad', max(gmaxRange), 'GradUnit', 'mT/m', 'MaxSlew', max(smaxRange), 'SlewUnit', 'T/m/s', 'riseTime', 0);

            % Load timing parameters
            s = load(timingParamsFile);
            obj.rf_90_duration = s.rf_90_duration;
            obj.rf_90_rfCenterInclDelay = s.rf_90_rfCenterInclDelay;
            obj.rf_180_duration = s.rf_180_duration;
            obj.rf_180_rfCenterInclDelay = s.rf_180_rfCenterInclDelay;
            obj.nav_dur = s.nav_dur;
        end

        function [final_bval, final_seq, diffGrad_final, TE, valid_g, valid_s, pns_out_final] = compute(obj)
            gradRasterTime = obj.sys.gradRasterTime;

            minTE_A = ceil((obj.rf_90_duration - obj.rf_90_rfCenterInclDelay + obj.nav_dur + obj.rf_180_rfCenterInclDelay)/gradRasterTime) * gradRasterTime;
            minTE_B = ceil((obj.rf_180_duration + obj.nav_dur + obj.rf_90_rfCenterInclDelay)/gradRasterTime) * gradRasterTime;
            TE = max([minTE_A, minTE_B]);

            % Sweep over gradient and slew rate ranges
            cnt = 1;
            for g = mr.convert(obj.gmaxRange, 'mT/m', 'Hz/m')
                for s = mr.convert(obj.smaxRange, 'T/m/s', 'Hz/m/s')
                    local_sys = mr.opts('MaxGrad', g, 'GradUnit', 'Hz/m', 'MaxSlew', s, 'SlewUnit', 'Hz/m/s', 'gradRasterTime', gradRasterTime);
                    trap_obj = pseq2.Monopolar(local_sys);
                    trap_obj.makeDummySeq(TE, obj.rf_90_duration, obj.rf_90_rfCenterInclDelay, obj.rf_180_duration);
                    b_tmp(cnt) = trap_obj.computeBValue();
                    diff_tmp(cnt) = trap_obj.gDiff;
                    seq_tmp(cnt) = trap_obj.seq;
                    g_list(cnt) = g;
                    s_list(cnt) = s;

                    try
                        pns_out(cnt) = mr.checkForPNS(trap_obj.seq, 'system', local_sys);
                    catch
                        pns_out(cnt) = struct('peak', NaN);
                    end

                    cnt = cnt + 1;
                end
            end

            [~, idx_valid] = min(abs([b_tmp] - obj.targetBval));

            final_bval = b_tmp(idx_valid);
            final_seq = seq_tmp(idx_valid);
            diffGrad_final = diff_tmp(idx_valid);
            valid_g = g_list(idx_valid);
            valid_s = s_list(idx_valid);
            pns_out_final = pns_out(idx_valid);
        end
    end
end
