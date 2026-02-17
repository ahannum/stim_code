classdef Asymm
    %Asymm Class for generating Asymm diffusion gradients and computing b-value

    properties
        gDiff          % Diffusion gradient waveform (trapezoid)      
        gDiff2              % Diffusion gradient waveform (trapezoid)
        smallDelta     % Effective gradient duration (s)
        bigDelta       % Time between diffusion gradient centers (s)
        bValueFinal   % Computed b-value (s/mm^2)
        gAmplitude     % Final gradient amplitude (T/m)
        wave_data      % Interpolated waveform
        moments       % moments of waveform 
    end

    methods
        function obj = Asymm(target_bval, delayTE1, delayTE2, rf180, gz180, rf90_duration, rf90_centerInclDelay, rf180_duration, gradRasterTime, diffGradMax, maxSlew, lims)
            % Constructor for Asymm diffusion gradient generator
            if target_bval == 0 
                gDiff = mr.makeTrapezoid('z', 'amplitude', 0,'riseTime',delayTE2/4, 'flatTime', delayTE2/4, 'system',lims);
                b_initial = 0;
                b_actual = 0;
                obj.wave_data = [0];
                obj.smallDelta = [0];
                obj.bigDelta = [0];
                obj.bValueFinal = 0;
                obj.gAmplitude = mr.convert(0,'Hz/m','mT/m');
                obj.gDiff = gDiff;
            else
                g = diffGradMax;
                gr = ceil(g / maxSlew / gradRasterTime) * gradRasterTime;
                
                
                flat_time1 = ceil((delayTE2 - 6*gr) / 4 / gradRasterTime) * gradRasterTime; 
                big_delta = delayTE1 + mr.calcDuration(rf180, gz180);
                
                flat_time2 = ceil((3*flat_time1 + 2*gr) / gradRasterTime) * gradRasterTime; 
                
                
                times = [0, gr, gr + flat_time1, gr*2 + flat_time1,  gr*3 + flat_time1,  gr*3 + flat_time2*2, gr*4 + flat_time2*2 ];
                amplitudes = [0, g, g, 0,-g, -g, 0];
                gDiff = mr.makeExtendedTrapezoid('z', 'system',lims, ...
                             'times', times, ...
                             'amplitudes', amplitudes) ;

                times2 = [0, gr, gr + flat_time2, gr*2 + flat_time2,  gr*3 + flat_time2,  gr*3 + flat_time1*2, gr*4 + flat_time1*2 ];
                amplitudes2 = [0, -g, -g, 0,g, g, 0];
                gDiff2 = mr.makeExtendedTrapezoid('z', 'system',lims, ...
                             'times', times2, ...
                             'amplitudes', amplitudes2) ;

              
                assert(mr.calcDuration(gDiff) <= delayTE1, 'Gradient too long for delayTE1.');
                assert(mr.calcDuration(gDiff) <= delayTE2, 'Gradient too long for delayTE2.');
    
                dt = gradRasterTime;
                dummy_seq = obj.makeDummySeq(gDiff, gDiff2, delayTE1, delayTE2, rf90_duration, rf90_centerInclDelay, rf180_duration,lims);
                [g_array_Tpm, b_actual] = obj.simulateWaveformAndBval(dummy_seq, dt);
                b_initial = b_actual;
                
                g_scaled = g * sqrt(target_bval / b_actual)
                assert(g_scaled <= diffGradMax, 'Gradient too high, increase TE.');
    
                %gDiff = mr.makeTrapezoid('z', 'amplitude', g_scaled, 'riseTime', gr, 'flatTime', small_delta - gr,'system',lims);
                times = [0, gr, gr + flat_time, gr*2 + flat_time,  gr*3 + flat_time,  gr*3 + flat_time*2, gr*4 + flat_time*2 ];
                amplitudes = [0, g_scaled, g_scaled, 0,-g_scaled, -g_scaled, 0];
                gDiff = mr.makeExtendedTrapezoid('z', 'system',lims, ...
                             'times', times, ...
                             'amplitudes', amplitudes) ;


                times2 = [0, gr, gr + flat_time2, gr*2 + flat_time2,  gr*3 + flat_time2,  gr*3 + flat_time1*2, gr*4 + flat_time1*2 ];
                amplitudes2 = [0, -g_scaled, -g_scaled, 0,g_scaled, g_scaled, 0];
                gDiff2 = mr.makeExtendedTrapezoid('z', 'system',lims, ...
                             'times', times2, ...
                             'amplitudes', amplitudes2) ;
           
                dummy_seq = obj.makeDummySeq(gDiff, gDiff2, delayTE1, delayTE2, rf90_duration, rf90_centerInclDelay, rf180_duration, lims);
                [g_array_Tpm, b_actual] = obj.simulateWaveformAndBval(dummy_seq, dt);
    
                obj.wave_data = g_array_Tpm;
                obj.smallDelta = small_delta;
                obj.bigDelta = big_delta;
                obj.bValueFinal = b_actual;
                obj.gAmplitude = mr.convert(g_scaled,'Hz/m','mT/m');
                obj.gDiff = gDiff;
                obj.gDiff2 = gDiff2;
    
            end
            fprintf('Asymm gradient prepared: Gmax = %.3f mT/m | Initial b = %.1f s/mm^2 | Final b = %.1f s/mm^2\n', obj.gAmplitude , b_initial, b_actual);
            
        end

        function dummy_seq = makeDummySeq(obj, gDiff, gDiff2, delayTE1, delayTE2, rf90_duration, rf90_centerInclDelay, rf180_duration,lims)
            t_90 = rf90_duration - rf90_centerInclDelay;
            t_180 = rf180_duration;
            dummy_seq = mr.Sequence(lims);
            dummy_seq.addBlock(mr.makeDelay(t_90));
            dummy_seq.addBlock(mr.makeDelay(delayTE1), gDiff);
            dummy_seq.addBlock(mr.makeDelay(t_180));
            dummy_seq.addBlock(mr.makeDelay(delayTE2), mr.scaleGrad(gDiff2, -1));
        end

        function [g_array_Tpm, bval] = simulateWaveformAndBval(obj, seq, dt)
            [wave_data, ~, ~, ~, ~] = seq.waveforms_and_times(true);
            tmin = 0;
            tmax = max(wave_data{3}(1, :));
            tt = tmin:dt:tmax;

            g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
            g_interp(isnan(g_interp)) = 0;

            g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;

            bval = obj.get_bvalue(g_array_Tpm, dt);
        end
    end

    methods (Static)
        function bval = get_bvalue(G, dt, gamma)
            if nargin < 3
                gamma = 267.522190e6;  % rad/s/T
            end

            gamma = gamma / 1000;  % convert to mm (s/mm^2 units)

            if isrow(G)
                G = G(:);
            end

            if size(G,2) > 3
                error('G must be N x 1 or N x 3');
            end

            G_int = cumsum(G) * dt;

            if size(G,2) == 1
                bval = gamma^2 * dt * sum(G_int.^2);
            else
                bval = 0;
                for d = 1:3
                    bval = bval + sum(G_int(:,d).^2);
                end
                bval = gamma^2 * dt * bval;
            end
        end



        function [bval, dummy_seq, gDiff, delayTE1,delayTE2,adjust_time] = computeMaxBValue(delayTE1, delayTE2, ...
                      gradRasterTime, rf90_duration, rf90_centerInclDelay,rf180_duration, ...
                      diffGradMax, maxSlew, lims)
            % Compute Max B-value for a given TE 
           
            % Set adjust_time to 0
            adjust_time= 0;
            
            % Form your waveform
            g = diffGradMax;
            gr = ceil(g / maxSlew / gradRasterTime) * gradRasterTime;
    
            small_delta = delayTE2 - gr;
            flat_time =  ceil((delayTE2 - 4 * gr) / 2 / gradRasterTime) * gradRasterTime;

            if  ceil((delayTE2 - 4 * gr) / 2 / gradRasterTime) * gradRasterTime < 0
                adjust_time = abs(((delayTE2 - 4 * gr) ));   % How much we need to absorb
                % Push delays to absorb the truncation
                delayTE1 = delayTE1 + adjust_time;
                delayTE2 = delayTE2 + adjust_time;

                flat_time =  ceil((delayTE2 - 4 * gr) / 2 / gradRasterTime) * gradRasterTime; 
               
                %small_delta = gr;  % effective smallDelta is now just ramp time
            end
            
            %gDiff = mr.makeTrapezoid('z', 'amplitude', g, 'riseTime', gr, 'flatTime',flat_time,'system',lims);
            if flat_time > 0 
                times = [0, gr, gr + flat_time, gr*2 + flat_time,  gr*3 + flat_time,  gr*3 + flat_time*2, gr*4 + flat_time*2 ];
                amplitudes = [0, g, g, 0,-g, -g, 0];
            
            else
                times = [0, gr, gr*2,  gr*3,  gr*4];
                amplitudes = [0, g,  0,-g,  0];

            end   
            times = ceil(times / lims.gradRasterTime) * lims.gradRasterTime; 

             % Snap times to gradient raster
            times_snapped = ceil(times / lims.gradRasterTime) * lims.gradRasterTime;
            if any(diff(times_snapped) <= 0)
                time_diffs = diff(times_snapped);
                min_diff = lims.gradRasterTime;
                needed_steps = max(0, ceil((min_diff - time_diffs) / min_diff));
                cumulative_adjust = cumsum([0, needed_steps]) * min_diff;
                times_adjusted = times_snapped + cumulative_adjust;
                
                adjustments = times_adjusted - times_snapped;
                total_adjustment = sum(adjustments);
                delayTE1 = delayTE1 + total_adjustment;
                delayTE2 = delayTE2 + total_adjustment;
                adjust_time = adjust_time + total_adjustment;
                times = times_adjusted;
            else
                times = times_snapped;
            end
            
           
            gDiff = mr.makeExtendedTrapezoid('z', 'system',lims, ...
                             'times', times, ...
                             'amplitudes', amplitudes) ;
           
            dt = gradRasterTime;
            
            % Make Dummy Seq
            t_90 = rf90_duration - rf90_centerInclDelay;
            t_180 = rf180_duration;
            dummy_seq = mr.Sequence(lims);
            dummy_seq.addBlock(mr.makeDelay(t_90));
            dummy_seq.addBlock(mr.makeDelay(delayTE1), gDiff);
            dummy_seq.addBlock(mr.makeDelay(t_180));
            dummy_seq.addBlock(mr.makeDelay(delayTE2), mr.scaleGrad(gDiff, -1));

            % Get B-value 
            [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
            tmin = 0;
            tmax = max(wave_data{3}(1, :));
            tt = tmin:dt:tmax;
            g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
            g_interp(isnan(g_interp)) = 0;
            g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;
            bval = pseq2.Asymm.get_bvalue(g_array_Tpm, dt);

             % Compute Moments
            moments = pseq2.Asymm.get_moments(g_array_Tpm', dt,0);
            
  
        
        end

        function [bval,moments] = getRealBValue(gDiff, gradRasterTime, rf90_duration, rf90_centerInclDelay, ...
                               rf180_duration, delayTE1, delayTE2, lims)
        %GETREALBVALUE Compute the actual b-value using a dummy sequence
        %
        % Inputs:
        %   gDiff                - Gradient object (from mr.makeTrapezoid or similar)
        %   gradRasterTime       - Gradient raster time (in seconds)
        %   rf90_duration        - Duration of the 90° RF pulse (in seconds)
        %   rf90_centerInclDelay - Center delay of the 90° RF pulse (in seconds)
        %   rf180_duration       - Duration of the 180° RF pulse (in seconds)
        %   delayTE1             - Delay before the first diffusion gradient (in seconds)
        %   delayTE2             - Delay before the second diffusion gradient (in seconds)
        %   lims                 - Scanner hardware limits structure
        %
        % Output:
        %   bval                 - Computed b-value (in s/mm²)
        
            % Compute delays relative to RF pulses
            t_90 = rf90_duration - rf90_centerInclDelay;
            t_180 = rf180_duration;
            dt = gradRasterTime;
        
            % Create dummy sequence
            dummy_seq = mr.Sequence(lims);
            dummy_seq.addBlock(mr.makeDelay(t_90));
            dummy_seq.addBlock(mr.makeDelay(delayTE1), gDiff);
            dummy_seq.addBlock(mr.makeDelay(t_180));
            dummy_seq.addBlock(mr.makeDelay(delayTE2), mr.scaleGrad(gDiff, -1));
        
            % Extract waveform data
            [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
        
            % Interpolate gradient waveform over time
            tmin = 0;
            tmax = max(wave_data{3}(1, :));
            tt = tmin:dt:tmax;
            g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
            g_interp(isnan(g_interp)) = 0;
        
            % Convert to Tesla/meter
            g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;
        
            % Calculate b-value
            bval = pseq2.Asymm.get_bvalue(g_array_Tpm, dt);

            % Compute Moments
            moments = pseq2.Asymm.get_moments(g_array_Tpm', dt,1);
        end

        
        function [bval, moments] = compute_bval_rotations(Gr, delayTE1, delayTE2, ...
            rf_90_duration, rf_90_rfCenterInclDelay, rf_180_duration, sys)
        
        % Inputs:
        %   Gr: cell array of gradient waveforms {Gx, Gy, Gz} (mr.grad structs)
        %   delayTE1, delayTE2: delays before and after 180 pulse (s)
        %   rf_90_duration: duration of 90° RF pulse (s)
        %   rf_90_rfCenterInclDelay: time from start to center of 90° RF (s)
        %   rf_180_duration: duration of 180° RF pulse (s)
        %   sys: system limits (mr.opts structure)
        
        % Output:
        %   bval: computed b-value in s/mm²
        
        % Compute delays relative to RF pulses
        t_90 = rf_90_duration - rf_90_rfCenterInclDelay;
        t_180 = rf_180_duration;
        dt = sys.gradRasterTime;
        
        % Flip gradients for refocusing
        Gr_flip = Gr;
        for i = 1:3
            Gr_flip{i} = mr.scaleGrad(Gr_flip{i}, -1);
        end
        
        % Build dummy sequence
        dummy_seq = mr.Sequence(sys);
        dummy_seq.addBlock(mr.makeDelay(t_90));
        dummy_seq.addBlock(mr.makeDelay(delayTE1), Gr{1}, Gr{2}, Gr{3});
        dummy_seq.addBlock(mr.makeDelay(t_180));
        dummy_seq.addBlock(mr.makeDelay(delayTE2), Gr_flip{1}, Gr_flip{2}, Gr_flip{3});
        
        % Extract gradient waveforms
        [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
        
        % Create common time vector
        tmin = 0;
        tmax = max([wave_data{1}(1,:), wave_data{2}(1,:), wave_data{3}(1,:)]);
        tt = tmin:dt:tmax;
        
        % Interpolate gradients onto uniform time grid
        g_interp_all = zeros(3, length(tt));
        for i = 1:3
            g_interp = interp1(wave_data{i}(1,:), wave_data{i}(2,:), tt, 'linear');
            g_interp(isnan(g_interp)) = 0;
            g_interp_all(i, :) = g_interp;
        end
        
        % Convert gradients to T/m
        g_array_Tpm = mr.convert(g_interp_all, 'Hz/m', 'mT/m') * 1e-3;
        
        % Compute b-value
        bval = pseq2.Asymm.get_bvalue(g_array_Tpm', dt);  % units: s/mm²

        % Compute Moments
        moments = pseq2.Asymm.get_moments(g_array_Tpm', dt,0);
        
        end

        function mmt_norm = get_moments(G,  dt, do_plot)
        % G          : gradient waveform (1D vector, already inverted if needed)
        % T_readout  : readout time in ms
        % dt         : dwell time in seconds
        
        G = G(:)';  % ensure row vector
        GAMMA = 42.58e3;  % Hz/mT = 42.58 MHz/T
        
        Nm = 5;
        N = numel(G);
        tvec = (0:N-1) * dt;  % time in seconds
        
        % Time matrix: each row is t^0, t^1, ..., t^4
        tMat = zeros(Nm, N);
        for mm = 1:Nm
            tMat(mm, :) = tvec .^ (mm-1);
        end
        
        % Compute moments: each row corresponds to one order (M0 to M4)
        mm = GAMMA * dt * tMat .* G;
        
        % Plot cumulative (integrated) normalized moments for M0, M1, M2
        if do_plot
            figure; hold on;
            colors = lines(3);
           legend_entries = cell(1, 3);  % Preallocate legend labels
    
            for i = 1:3
                mmt = cumsum(mm(i, :));  % unnormalized cumulative moment
                mmt_norm = mmt / max(abs(mmt));  % normalize for plotting
                plot(mmt_norm, 'Color', colors(i,:), 'LineWidth', 1.5);
            
                % Store formatted legend entry with final unnormalized value
                final_val = mmt(end);
                legend_entries{i} = sprintf('M_%d = %.2e', i-1, final_val);
            end
            
            yline(0, 'k');
            legend(legend_entries, 'Location', 'best');
            
            yline(0, 'k');
            xlabel('Timepoint'); ylabel('Normalized Cumulative Moment');
            title('Cumulative Gradient Moments (normalized)');
            grid on;
        else
            for i = 1:3
                mmt = cumsum(mm(i, :));  % unnormalized cumulative moment
                mmt_norm = mmt / max(abs(mmt));  % normalize for plotting
            end
        end
        end
        
    end
end













% classdef Asymm
%     %BIPOLAR Class for generating Asymm diffusion gradients and computing b-value
% 
%     properties
%         gDiff          % Diffusion gradient waveform (trapezoid)
%         gDiff2         % Diffusion Gradient negative trapezoid
%         smallDelta     % Effective gradient duration (s)
%         bigDelta       % Time between diffusion gradient centers (s)
%         bValueFinal   % Computed b-value (s/mm^2)
%         gAmplitude     % Final gradient amplitude (T/m)
%         wave_data      % Interpolated waveform
%     end
% 
%     methods
%         function obj = Asymm(target_bval, delayTE1, delayTE2, rf180, gz180, rf90_duration, rf90_centerInclDelay, rf180_duration, gradRasterTime, diffGradMax, maxSlew, lims)
%             % Constructor for Asymm diffusion gradient generator
%             if target_bval == 0 
%                 gDiff = mr.makeTrapezoid('z', 'amplitude', 0,'riseTime',delayTE2/4, 'flatTime', delayTE2/4, 'system',lims);
%                 b_initial = 0;
%                 b_actual = 0;
%                 obj.wave_data = [0];
%                 obj.smallDelta = [0];
%                 obj.bigDelta = [0];
%                 obj.bValueFinal = 0;
%                 obj.gAmplitude = mr.convert(0,'Hz/m','mT/m');
%                 obj.gDiff = gDiff;
%             else
%                 g = diffGradMax;
%                 gr = ceil(g / maxSlew / gradRasterTime) * gradRasterTime;
% 
%                 flat_time = ceil ((delayTE2 - 4 * gr) / 2) * gradRasterTime / gradRasterTime; 
%                 %small_delta = delayTE2 - gr;
%                 %big_delta = delayTE1 + mr.calcDuration(rf180, gz180);
% 
%                 gDiff = mr.makeTrapezoid('z', 'amplitude', g, 'riseTime', gr, 'flatTime', flat_time,'system',lims);
%                 gDiff2 = mr.scaleGrad(gDiff,-1);
% 
%                 assert(mr.calcDuration(gDiff) <= delayTE1, 'Gradient too long for delayTE1.');
%                 assert(mr.calcDuration(gDiff) <= delayTE2, 'Gradient too long for delayTE2.');
% 
%                 dt = gradRasterTime;
%                 dummy_seq = obj.makeDummySeq(gDiff, gDiff2, delayTE1, delayTE2, rf90_duration, rf90_centerInclDelay, rf180_duration,lims);
%                 [g_array_Tpm, b_actual] = obj.simulateWaveformAndBval(dummy_seq, dt);
%                 b_initial = b_actual;
% 
%                 g_scaled = g * sqrt(target_bval / b_actual)
%                 assert(g_scaled <= diffGradMax, 'Gradient too high, increase TE.');
% 
%                 gDiff = mr.makeTrapezoid('z', 'amplitude', g_scaled, 'riseTime', gr, 'flatTime', flat_time,'system',lims);
%                 gDiff2 = mr.scaleGrad(gDiff,-1);
% 
%                 dummy_seq = obj.makeDummySeq(gDiff, gDiff2, delayTE1, delayTE2, rf90_duration, rf90_centerInclDelay, rf180_duration, lims);
%                 [g_array_Tpm, b_actual] = obj.simulateWaveformAndBval(dummy_seq, dt);
% 
%                 obj.wave_data = g_array_Tpm;
%                 obj.smallDelta = small_delta;
%                 obj.bigDelta = big_delta;
%                 obj.bValueFinal = b_actual;
%                 obj.gAmplitude = mr.convert(g_scaled,'Hz/m','mT/m');
%                 obj.gDiff = gDiff;
% 
%             end
%             fprintf('Asymm gradient prepared: Gmax = %.3f mT/m | Initial b = %.1f s/mm^2 | Final b = %.1f s/mm^2\n', obj.gAmplitude , b_initial, b_actual);
% 
%         end
% 
%         function dummy_seq = makeDummySeq(obj, gDiff, gDiff2, delayTE1, delayTE2, rf90_duration, rf90_centerInclDelay, rf180_duration,lims)
%             t_90 = rf90_duration - rf90_centerInclDelay;
%             t_180 = rf180_duration;
%             dummy_seq = mr.Sequence(lims);
%             dummy_seq.addBlock(mr.makeDelay(t_90));
%             dummy_seq.addBlock(gDiff)
%             dummy_seq.addBlock(mr.makeDelay(delayTE1 - mr.calcDuration(gDiff)), gDiff2);
%             dummy_seq.addBlock(mr.makeDelay(t_180));
%             dummy_seq.addBlock(mr.scaleGrad(gDiff,-1))
%             dummy_seq.addBlock(mr.makeDelay(delayTE2 - mr.calcDuration(gDiff)), mr.scaleGrad(gDiff2,-1));
% 
% 
%         end
% 
%         function [g_array_Tpm, bval] = simulateWaveformAndBval(obj, seq, dt)
%             [wave_data, ~, ~, ~, ~] = seq.waveforms_and_times(true);
%             tmin = 0;
%             tmax = max(wave_data{3}(1, :));
%             tt = tmin:dt:tmax;
% 
%             g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
%             g_interp(isnan(g_interp)) = 0;
% 
%             g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;
% 
%             bval = obj.get_bvalue(g_array_Tpm, dt);
%         end
%     end
% 
%     methods (Static)
%         function bval = get_bvalue(G, dt, gamma)
%             if nargin < 3
%                 gamma = 267.522190e6;  % rad/s/T
%             end
% 
%             gamma = gamma / 1000;  % convert to mm (s/mm^2 units)
% 
%             if isrow(G)
%                 G = G(:);
%             end
% 
%             if size(G,2) > 3
%                 error('G must be N x 1 or N x 3');
%             end
% 
%             G_int = cumsum(G) * dt;
% 
%             if size(G,2) == 1
%                 bval = gamma^2 * dt * sum(G_int.^2);
%             else
%                 bval = 0;
%                 for d = 1:3
%                     bval = bval + sum(G_int(:,d).^2);
%                 end
%                 bval = gamma^2 * dt * bval;
%             end
%         end
% 
% 
% 
%         function [bval, dummy_seq, gDiff, delayTE1,delayTE2,adjust_time] = computeMaxBValue(delayTE1, delayTE2, ...
%                       gradRasterTime, rf90_duration, rf90_centerInclDelay,rf180_duration, ...
%                       diffGradMax, maxSlew, lims)
%             % Compute Max B-value for a given TE 
% 
%             % Set adjust_time to 0
%             adjust_time= 0;
% 
%             % Form your waveform
%             g = diffGradMax;
%             gr = ceil(g / maxSlew / gradRasterTime) * gradRasterTime;
% 
%             %small_delta = delayTE2 - gr;
%             flat_time =  ((delayTE2 - 4 * gr) / 2) * gradRasterTime / gradRasterTime; 
% 
%             if  ((delayTE2 - 4 * gr) / 2)  < 0
%                 adjust_time = abs(((delayTE2 - 4 * gr) ));   % How much we need to absorb
%                 flat_time = 0;
% 
%                 % Push delays to absorb the truncation
%                 delayTE1 = delayTE1 + adjust_time;
%                 delayTE2 = delayTE2 + adjust_time;
% 
%                 flat_time =  ((delayTE2 - 4 * gr) / 2) * gradRasterTime / gradRasterTime; 
% 
%                 %small_delta = gr;  % effective smallDelta is now just ramp time
%             end
% 
%             gDiff = mr.makeTrapezoid('z', 'amplitude', g, 'riseTime', gr, 'flatTime',flat_time,'system',lims);
%             gDiff2 = mr.scaleGrad(gDiff,-1);
% 
%             dt = gradRasterTime;
% 
%             % Make Dummy Seq
%             t_90 = rf90_duration - rf90_centerInclDelay;
%             t_180 = rf180_duration;
%             dummy_seq = mr.Sequence(lims);
%             dummy_seq.addBlock(mr.makeDelay(t_90));
%             dummy_seq.addBlock(gDiff)
%             dummy_seq.addBlock(mr.makeDelay(delayTE1 - mr.calcDuration(gDiff)), gDiff2);
%             dummy_seq.addBlock(mr.makeDelay(t_180));
%             dummy_seq.addBlock(mr.scaleGrad(gDiff,-1))
%             dummy_seq.addBlock(mr.makeDelay(delayTE2 - mr.calcDuration(gDiff)), mr.scaleGrad(gDiff2,-1));
%             %dummy_seq.addBlock(mr.makeDelay(delayTE2), mr.scaleGrad(gDiff, -1), mr.scaleGrad(gDiff2, -1));
% 
%             % Get B-value 
%             [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
%             tmin = 0;
%             tmax = max(wave_data{3}(1, :));
%             tt = tmin:dt:tmax;
%             g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
%             g_interp(isnan(g_interp)) = 0;
%             g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;
%             bval = pseq2.Asymm.get_bvalue(g_array_Tpm, dt);
% 
% 
% 
%         end
% 
%         function bval = getRealBValue(gDiff, gradRasterTime, rf90_duration, rf90_centerInclDelay, ...
%                                rf180_duration, delayTE1, delayTE2, lims)
%         %GETREALBVALUE Compute the actual b-value using a dummy sequence
%         %
%         % Inputs:
%         %   gDiff                - Gradient object (from mr.makeTrapezoid or similar)
%         %   gradRasterTime       - Gradient raster time (in seconds)
%         %   rf90_duration        - Duration of the 90° RF pulse (in seconds)
%         %   rf90_centerInclDelay - Center delay of the 90° RF pulse (in seconds)
%         %   rf180_duration       - Duration of the 180° RF pulse (in seconds)
%         %   delayTE1             - Delay before the first diffusion gradient (in seconds)
%         %   delayTE2             - Delay before the second diffusion gradient (in seconds)
%         %   lims                 - Scanner hardware limits structure
%         %
%         % Output:
%         %   bval                 - Computed b-value (in s/mm²)
% 
%             gDiff2 = mr.scaleGrad(gDiff,-1);
%             % Compute delays relative to RF pulses
%             t_90 = rf90_duration - rf90_centerInclDelay;
%             t_180 = rf180_duration;
%             dt = gradRasterTime;
% 
%             % Create dummy sequence
%             dummy_seq = mr.Sequence(lims);
%             dummy_seq.addBlock(mr.makeDelay(t_90));
%             dummy_seq.addBlock(gDiff)
%             dummy_seq.addBlock(mr.makeDelay(delayTE1 - mr.calcDuration(gDiff)), gDiff2);
%             dummy_seq.addBlock(mr.makeDelay(t_180));
%             dummy_seq.addBlock(mr.scaleGrad(gDiff,-1))
%             dummy_seq.addBlock(mr.makeDelay(delayTE2 - mr.calcDuration(gDiff)), mr.scaleGrad(gDiff2,-1));
% 
%             % Extract waveform data
%             [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
% 
%             % Interpolate gradient waveform over time
%             tmin = 0;
%             tmax = max(wave_data{3}(1, :));
%             tt = tmin:dt:tmax;
%             g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
%             g_interp(isnan(g_interp)) = 0;
% 
%             % Convert to Tesla/meter
%             g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;
% 
%             % Calculate b-value
%             bval = pseq2.Asymm.get_bvalue(g_array_Tpm, dt);
%         end
% 
% 
%         function bval = compute_bval_rotations(Gr, delayTE1, delayTE2, ...
%             rf_90_duration, rf_90_rfCenterInclDelay, rf_180_duration, sys)
% 
%         % Inputs:
%         %   Gr: cell array of gradient waveforms {Gx, Gy, Gz} (mr.grad structs)
%         %   delayTE1, delayTE2: delays before and after 180 pulse (s)
%         %   rf_90_duration: duration of 90° RF pulse (s)
%         %   rf_90_rfCenterInclDelay: time from start to center of 90° RF (s)
%         %   rf_180_duration: duration of 180° RF pulse (s)
%         %   sys: system limits (mr.opts structure)
% 
%         % Output:
%         %   bval: computed b-value in s/mm²
% 
%         % Compute delays relative to RF pulses
%         t_90 = rf_90_duration - rf_90_rfCenterInclDelay;
%         t_180 = rf_180_duration;
%         dt = sys.gradRasterTime;
% 
%         % Flip gradients for refocusing
%         Gr_flip = Gr;
%         for i = 1:3
%             Gr_flip{i} = mr.scaleGrad(Gr_flip{i}, -1);
%         end
% 
%         % Build dummy sequence
%         dummy_seq = mr.Sequence(sys);
%         dummy_seq.addBlock(mr.makeDelay(t_90));
%         dummy_seq.addBlock(Gr{1}, Gr{2}, Gr{3});
%         dummy_seq.addBlock(mr.makeDelay(delayTE1 - mr.calcDuration(Gr{1})),Gr_flip{1}, Gr_flip{2}, Gr_flip{3})
%         dummy_seq.addBlock(mr.makeDelay(t_180));
%         dummy_seq.addBlock(Gr_flip{1}, Gr_flip{2}, Gr_flip{3});
%         dummy_seq.addBlock(mr.makeDelay(delayTE2 - mr.calcDuration(Gr{1})),Gr{1}, Gr{2}, Gr{3})
% 
% 
%         % Extract gradient waveforms
%         [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
% 
%         % Create common time vector
%         tmin = 0;
%         tmax = max([wave_data{1}(1,:), wave_data{2}(1,:), wave_data{3}(1,:)]);
%         tt = tmin:dt:tmax;
% 
%         % Interpolate gradients onto uniform time grid
%         g_interp_all = zeros(3, length(tt));
%         for i = 1:3
%             g_interp = interp1(wave_data{i}(1,:), wave_data{i}(2,:), tt, 'linear');
%             g_interp(isnan(g_interp)) = 0;
%             g_interp_all(i, :) = g_interp;
%         end
% 
%         % Convert gradients to T/m
%         g_array_Tpm = mr.convert(g_interp_all, 'Hz/m', 'mT/m') * 1e-3;
% 
%         % Compute b-value
%         bval = pseq2.Asymm.get_bvalue(g_array_Tpm', dt);  % units: s/mm²
% 
%         end
% 
% 
% 
% 
% 
% 
%     end
% end