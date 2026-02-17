function [g3_segment_cell, M0_cell, M0_TE, all_pts] = get_moment_check(...
    wave_data, tfp_excitation, tfp_refocusing, t_adc, ...
    fp_adc, Nadc, TRdelay_perSlice, tr_to_plot, do_plot)

    % Extract real part of RF waveform
    tt_rf = squeeze(real(wave_data{1,4}(1,:)));  % Python index 3 is MATLAB index 4

    % Find the starting indices of the RF pulses
    rf_starts = find(diff(tt_rf) > 100e-6) + 1;
    rf_starts = [1, rf_starts];
    tt_rf_start = tt_rf(1, rf_starts);

    % Extract excitation and refocusing times
    tt_excite = tfp_excitation(1,:);
    tt_refocusing = tfp_refocusing(1,:);
    
    % Define inversion times
    tt_inv = tt_refocusing(1:end);
    tt_rf_start = tt_excite(1:end);

    % Find ADC start times
    adc_starts = find(diff(t_adc) > 200e-6) + 1;
    adc_starts = [1, adc_starts];
    tt_adc_start = t_adc(adc_starts);

    % Calculate ADC stop times
    tt_adc_stop = t_adc(adc_starts + Nadc - 1);

    % Calculate TR times
    if length(tt_rf_start) > 1
        tt_TR = diff(tt_rf_start);
        tt_TR = cumsum([tt_TR, tt_TR(end)]);
    else
        tt_TR = tt_rf_start + TRdelay_perSlice;
    end

    % Segment ADC times
    seg_t_adc = cell(length(adc_starts), 1);
    for i = 1:length(adc_starts)
        seg_t_adc{i} = t_adc(adc_starts(i):adc_starts(i) + Nadc - 1);
    end
    seg_t_adc = vertcat(seg_t_adc{:});

    % Step 1: Interpolation time grid
    g3 = cell(1, 3);
    dt = 1e-6;
    tmin_all = 0;
    tmax_all = -inf;
    for i = 1:3
        tmax_all = max(tmax_all, max(wave_data{i}(1,:)));
    end
    tt_target = tmin_all:dt:tmax_all;

    for i = 1:3
        g_interp = interp1(wave_data{i}(1,:), wave_data{i}(2,:), tt_target, 'linear');
        g3{i} = [tt_target; g_interp];
    end

    % Step 2: Segment g3 by TR
    numTR = length(tt_excite);
    g3_segment_cell = cell(1, numTR);
    M0_cell = cell(1, numTR);
    M0_TE = {};
    all_pts = {};

    idx_adc= 2;


    for j = 1:numTR
        [~, idx_start] = min(abs(tt_target - tt_excite(j)));
        [~, idx_end]   = min(abs(tt_target - tt_TR(j)));
        [~, idx_inv]   = min(abs(tt_target - tt_inv(j)));
        time_inv = tt_target(idx_inv);
        time_start = tt_target(idx_start);

        g3_segment = zeros(3, idx_end - idx_start + 1);
        for i = 1:3
            tmp = g3{i};
            tmp(:, idx_inv:end) = tmp(:, idx_inv:end) * -1;
            g_segment = tmp(:, idx_start:idx_end);
            g3_segment(i, :) = g_segment(2, :);
        end

        tt_segment = tt_target(idx_start:idx_end);
        g3_segment_cell{j} = {tt_segment, g3_segment};
        M0_cell{j} = dt * cumsum(g3_segment, 2, "omitnan");

        t_TE = time_inv + (time_inv - time_start);
        [~, idx_TE] = min(abs(tt_segment - t_TE));
        M0_TE{j} = M0_cell{j}(:, idx_TE);

        [~, idx] = min(abs(tt_segment - tt_adc_start(idx_adc)));
        all_pts{j} = M0_cell{j}(:, idx:end);

        idx_adc = idx_adc + 1;
    end

    % Optional plotting
    if do_plot
        figure;
        subplot(121);
        title('Moments'); hold on;
        for ii = 1:3
            plot(g3_segment_cell{tr_to_plot}{1}, M0_cell{tr_to_plot}(ii, :)); hold on;
        end
        ylabel('M'); xlabel('Time [s]');

        subplot(122);
        title('TE Trajectory Check'); hold on;
        scatter(all_pts{tr_to_plot}(1,:), all_pts{tr_to_plot}(2,:), '.');
        scatter(0, 0, 200, '+', 'g', 'LineWidth', 2);
        scatter(M0_TE{tr_to_plot}(1), M0_TE{tr_to_plot}(2), 100, 'x', 'r', 'LineWidth', 2);
        ylabel('k_y'); xlabel('k_x'); grid on;
    end
end