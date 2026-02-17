classdef getMinTE_m0
    properties
        timingParamsFile
        targetBval
        pnsThresh
        mriSystemLabel
        gmaxRange
        smaxRange

        sys
        ascName

        rf_90_duration
        rf_90_rfCenterInclDelay
        rf_180_duration
        rf_180_rfCenterInclDelay
        nav_dur
        timeToTE

        gx
        gxPre
        nNav
        Ny_meas
        gz_180
        mrSystemLabel

        epi_mode % include epi timings

    end

    methods
        function obj = getMinTE_m0(timingParamsFile, mriSystemLabel, targetBval, pnsThresh, gmaxRange, smaxRange,epi_mode)
            obj.timingParamsFile = timingParamsFile;
            obj.mriSystemLabel = mriSystemLabel;
            obj.targetBval = targetBval;
            obj.pnsThresh = pnsThresh;
            obj.gmaxRange = gmaxRange;
            obj.smaxRange = smaxRange;

            % Load timing params
            s = load(timingParamsFile);
            obj.rf_90_duration = s.rf_90_duration;
            obj.rf_90_rfCenterInclDelay = s.rf_90_rfCenterInclDelay;
            obj.rf_180_duration = s.rf_180_duration;
            obj.rf_180_rfCenterInclDelay = s.rf_180_rfCenterInclDelay;
            obj.nav_dur = s.nav_dur;
            obj.timeToTE = s.timeToTE;
            
            obj.gxPre = s.gxPre;
            obj.gx = s.gx;
            obj.nNav = s.nNav;
            obj.Ny_meas = s.Ny_meas;
            obj.gz_180 = s.gz_180;
            obj.mrSystemLabel = mriSystemLabel;
            obj.epi_mode = epi_mode;

            if obj.nav_dur == 0
                obj.nNav = 0;
            end

           

            % MRI System parameters
            switch mriSystemLabel
                case 'C'
                    obj.sys = mr.opts('MaxGrad', 200, 'GradUnit', 'mT/m', ...
                        'MaxSlew', 200, 'SlewUnit', 'T/m/s', ...
                        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6, ...
                        'adcDeadTime', 10e-6, 'B0', 2.89);
                    obj.ascName = '/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_P034_X60.asc';
                case 'P'
                    obj.sys = mr.opts('MaxGrad', 80, 'GradUnit', 'mT/m', ...
                        'MaxSlew', 200, 'SlewUnit', 'T/m/s', ...
                        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6, ...
                        'adcDeadTime', 10e-6, 'B0', 2.89);
                    obj.ascName = '/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_K2309_2250V_951A_XQ_GC04XQ.asc';
                case 'V'
                    obj.sys = mr.opts('MaxGrad', 45, 'GradUnit', 'mT/m', ...
                        'MaxSlew', 200, 'SlewUnit', 'T/m/s', ...
                        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6, ...
                        'adcDeadTime', 10e-6, 'B0', 2.89);
                    obj.ascName = '/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_K2309_2250V_951A_XR_AS82.asc';
                otherwise
                    error('Unknown MRI system label');
            end
        end

        function [final_bval, final_seq, diffGrad_final, TE, valid_g, valid_s, pns_out_final,waveform_seq] = compute(obj)
            gradRasterTime = obj.sys.gradRasterTime;
            TE = max([...
                ceil((obj.rf_90_duration - obj.rf_90_rfCenterInclDelay + obj.nav_dur + obj.rf_180_rfCenterInclDelay) / gradRasterTime) * gradRasterTime,
                ceil((obj.rf_180_duration - obj.rf_180_rfCenterInclDelay + obj.timeToTE) / gradRasterTime) * gradRasterTime
            ]) * 2;

            maxTE = 150e-3;
            niter = -1;
            while TE < maxTE
                niter = niter + 1;

                delayTE1_min = ceil((TE / 2 - obj.rf_90_duration + obj.rf_90_rfCenterInclDelay - obj.nav_dur - obj.rf_180_rfCenterInclDelay) / gradRasterTime) * gradRasterTime;
                delayTE2_min = ceil((TE / 2 - obj.rf_180_duration + obj.rf_180_rfCenterInclDelay - obj.timeToTE) / gradRasterTime) * gradRasterTime;

                [bval, tmp_seq, diffGrad, delayTE1_min, delayTE2_min, adjust_time] = pseq2.Monopolar.computeMaxBValue(delayTE1_min, delayTE2_min, gradRasterTime, obj.rf_90_duration + obj.nav_dur, obj.rf_90_rfCenterInclDelay, obj.rf_180_duration, obj.sys.maxGrad, obj.sys.maxSlew, obj.sys);

                if adjust_time ~= 0
                    fprintf('Increased TE to accommodate invalid waveform, delayTE1 and delayTE2 also updated\n')
                    TE = TE + adjust_time * 2;
                    continue
                end

                tmp_seq = obj.buildSequence(diffGrad, delayTE1_min, delayTE2_min);
                [pns_out, ~, pns_comp, ~] = tmp_seq.calcPNS(obj.ascName, 0);

                switch obj.mriSystemLabel
                    case "C"
                    pns_pass = all([max(pns_comp(3,:)) <= obj.pnsThresh, max(pns_comp(2,:)) <= obj.pnsThresh, max(pns_comp(1,:)) <= obj.pnsThresh, max(pns_comp(6,:)) <= obj.pnsThresh, max(pns_comp(5,:)) <= obj.pnsThresh, max(pns_comp(4,:)) <= obj.pnsThresh]);
                    case "V"
                    pns_pass = all([max(pns_comp(3,:)) <= obj.pnsThresh, max(pns_comp(2,:)) <= obj.pnsThresh, max(pns_comp(1,:)) <= obj.pnsThresh]);
                    case "P"
                    pns_pass = all([max(pns_comp(3,:)) <= obj.pnsThresh, max(pns_comp(2,:)) <= obj.pnsThresh, max(pns_comp(1,:)) <= obj.pnsThresh]);


                end

                if pns_pass && bval > obj.targetBval
                    final_bval = bval;
                    final_seq = tmp_seq;
                    diffGrad_final = diffGrad;
                    TE = TE;
                    valid_g = mr.convert(obj.sys.maxGrad, 'Hz/m', 'mT/m');
                    valid_s = mr.convert(obj.sys.maxSlew, 'Hz/m/s','T/m/s');
                    pns_out_final = pns_out;
                    
                    break

                
                
                elseif bval > obj.targetBval
                    fprintf('B-value exceeds target, but PNS is violated, checking if any viable hardware configurations\n')

                    [valid_b, valid_g, valid_s] = obj.sweepHardwareConfigs(delayTE1_min, delayTE2_min);

                    if ~isempty(valid_b)
                        [final_bval, final_seq, diffGrad_final, ~, ~, ~] = pseq2.Monopolar.computeMaxBValue(delayTE1_min, delayTE2_min, gradRasterTime, obj.rf_90_duration + obj.nav_dur, obj.rf_90_rfCenterInclDelay, obj.rf_180_duration, mr.convert(valid_g, 'mT/m', 'Hz/m'), mr.convert(valid_s, 'T/m/s', 'Hz/m/s'), obj.sys);
                        [pns_out_final, ~, ~, ~] = final_seq.calcPNS(obj.ascName, 1);
                        TE = TE;
                        return
                    else
                        fprintf('No viable combinations found ... increasing TE ... \n');
                        TE = TE + gradRasterTime * 10;
                    end
                else
                    TE = TE + gradRasterTime * 10;
                end
            end

            [final_bval, final_seq, diffGrad_final, ~, ~, ~] = pseq2.Monopolar.computeMaxBValue(delayTE1_min, delayTE2_min, gradRasterTime, obj.rf_90_duration+ obj.nav_dur, obj.rf_90_rfCenterInclDelay, obj.rf_180_duration, obj.sys.maxGrad, obj.sys.maxSlew, obj.sys);
            [pns_out_final, ~, ~, ~] = final_seq.calcPNS(obj.ascName, 1);

            % Make Sequence to get waveforms 
            waveform_seq = mr.Sequence(obj.sys);
            waveform_seq.addBlock(mr.makeDelay(obj.rf_90_duration+ obj.nav_dur  - obj.rf_90_rfCenterInclDelay));
            waveform_seq.addBlock(mr.makeDelay(delayTE1_min),diffGrad_final)
            waveform_seq.addBlock(obj.rf_180_duration)
            waveform_seq.addBlock(mr.makeDelay(delayTE2_min),diffGrad_final)
            
        end
    
        function seq = buildSequence(obj, diffGrad, delay1, delay2)
            diffGrad_y = diffGrad; diffGrad_y.channel = 'y';
            diffGrad_x = diffGrad; diffGrad_x.channel = 'x';
            t_90 = obj.rf_90_duration - obj.rf_90_rfCenterInclDelay;
            t_180 = obj.rf_180_duration;
            seq = mr.Sequence(obj.sys);
            seq.addBlock(mr.makeDelay(t_90));
            
            if obj.nav_dur > 0
                seq.addBlock(obj.gxPre)
                gx_start = obj.gx;
                for ii = 1:obj.nNav
                    seq.addBlock(gx_start)
                    gx_start = mr.scaleGrad(gx_start,-1); % flip to go back
                end
    
                gxPost = obj.gxPre;
                gxPost.delay = 0;
                seq.addBlock(gxPost)
            end
            
            seq.addBlock(mr.makeDelay(delay1), diffGrad, diffGrad_y, diffGrad_x);
            seq.addBlock(obj.gz_180);
            seq.addBlock(mr.makeDelay(delay2), mr.scaleGrad(diffGrad, -1), mr.scaleGrad(diffGrad_x, -1), mr.scaleGrad(diffGrad_y, -1));
            
            if obj.epi_mode 
                gxPreMain = obj.gxPre;
                gxPreMain.delay = 0;
                seq.addBlock(mr.scaleGrad(gxPreMain,-1));
                for ii = 1:obj.Ny_meas
                    seq.addBlock(gx_start)
                    gx_start = mr.scaleGrad(gx_start,-1); % flip to go back
                end
            end
     

        end

        

        function [valid_b, valid_g, valid_s] = sweepHardwareConfigs(obj, delayTE1, delayTE2)
            g_vals = mr.convert(obj.gmaxRange, 'mT/m', 'Hz/m');
            s_vals = mr.convert(obj.smaxRange, 'T/m/s', 'Hz/m/s');
            Ng = numel(g_vals); Ns = numel(s_vals);
            numComb = Ng * Ns;
            b_flat = NaN(numComb, 1);
            pns_flat = NaN(numComb, 1);
            ig_vec = zeros(numComb, 1);
            is_vec = zeros(numComb, 1);
            [G_idx, S_idx] = ndgrid(1:Ng, 1:Ns);
            ig_all = G_idx(:);
            is_all = S_idx(:);

            parfor idx = 1:numComb
                ig = ig_all(idx);
                is = is_all(idx);
                g = g_vals(ig);
                s = s_vals(is);
                [tmp_bval, tmp_seq, diffGrad, ~, ~, adjust_time] = pseq2.Monopolar.computeMaxBValue(delayTE1, delayTE2, obj.sys.gradRasterTime, obj.rf_90_duration+ obj.nav_dur, obj.rf_90_rfCenterInclDelay, obj.rf_180_duration, g, s, obj.sys);
                if adjust_time == 0
                    tmp_seq = obj.buildSequence(diffGrad, delayTE1, delayTE2);
                    [tmp_pns_out, ~, pns_comp, ~] = tmp_seq.calcPNS(obj.ascName, 0);
                    %pass = all([max(pns_comp(3,:)) <= obj.pnsThresh, max(pns_comp(2,:)) <= obj.pnsThresh, max(pns_comp(1,:)) <= obj.pnsThresh, max(pns_comp(6,:)) <= obj.pnsThresh, max(pns_comp(5,:)) <= obj.pnsThresh, max(pns_comp(4,:)) <= obj.pnsThresh]);
                    pass = 0;
                    switch obj.mriSystemLabel
                        case "C"
                            pass = all([max(pns_comp(3,:)) <= obj.pnsThresh, max(pns_comp(2,:)) <= obj.pnsThresh, max(pns_comp(1,:)) <= obj.pnsThresh, max(pns_comp(6,:)) <= obj.pnsThresh, max(pns_comp(5,:)) <= obj.pnsThresh, max(pns_comp(4,:)) <= obj.pnsThresh]);
                        case "P"
                            pass = all([max(pns_comp(3,:)) <= obj.pnsThresh, max(pns_comp(2,:)) <= obj.pnsThresh, max(pns_comp(1,:)) <= obj.pnsThresh]);

                        
                        case "V"
                            pass = all([max(pns_comp(3,:)) <= obj.pnsThresh, max(pns_comp(2,:)) <= obj.pnsThresh, max(pns_comp(1,:)) <= obj.pnsThresh]);

                    end
                    
                    
                    b_flat(idx) = tmp_bval;
                    pns_flat(idx) = pass;
                end
                ig_vec(idx) = ig;
                is_vec(idx) = is;
            end

            b_tmp = NaN(Ng, Ns);
            pns_tmp = NaN(Ng, Ns);
            for idx = 1:numComb
                b_tmp(ig_vec(idx), is_vec(idx)) = b_flat(idx);
                pns_tmp(ig_vec(idx), is_vec(idx)) = pns_flat(idx);
            end

            valid_mask = (pns_tmp == 1) & (b_tmp >= obj.targetBval);
            if any(valid_mask, 'all')
                [~, linear_idx] = min(b_tmp(valid_mask));
                [i_all, j_all] = find(valid_mask);
                i_valid = i_all(linear_idx);
                j_valid = j_all(linear_idx);
                valid_b = b_tmp(i_valid, j_valid);
                valid_g = obj.gmaxRange(i_valid);
                valid_s = obj.smaxRange(j_valid);
                fprintf('Valid combo found: G = %.2f mT/m, S = %.2f T/m/s, b = %.0f s/mm^2\n', valid_g, valid_s, valid_b);
            else
                valid_b = [];
                valid_g = [];
                valid_s = [];
            end


        end
        
    
    end
end