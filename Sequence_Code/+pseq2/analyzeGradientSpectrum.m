function analyzeGradientSpectrum(seq, sys, ascName)
% ANALYZEGRADIENTSPECTRUM Computes and plots the acoustic spectrum of gradients
%
% Usage:
%   analyzeGradientSpectrum(seq, sys)
%   analyzeGradientSpectrum(seq, sys, ascName)
%
% Inputs:
%   seq      - Pulseq sequence object
%   sys      - System object (with .gradRasterTime)
%   ascName  - Optional path to Siemens .asc file to overlay resonance bands

% Parameters
dt = sys.gradRasterTime;     % time raster
fmax = 10000;                % max frequency to display (Hz)
nwin = 5000;                 % window length (samples)
os = 3;                      % oversampling

if nargin < 3
    ascName = [];
end

% Load .asc resonance data if available
if ischar(ascName) && exist(ascName, 'file')
    ascData = mr.Siemens.readasc(ascName);
else
    ascData = [];
end

% Frequency axis
faxis = (0:(nwin/2 - 1)) / (nwin * dt * os);
nfmax = sum(faxis <= fmax);

% Interpolate gradient waveforms to raster time
wave_data = seq.waveforms_and_times();
tmax = max([wave_data{1}(1,end), wave_data{2}(1,end), wave_data{3}(1,end)]);
nt = ceil(tmax / dt);
tmax = nt * dt;

gw = zeros(3, nt);
for i = 1:3
    gw(i,:) = interp1(wave_data{i}(1,:), wave_data{i}(2,:), ((1:nt) - 0.5) * dt, 'linear', 0);
end

% Compute segmented FFT
gs = [];
for g = 1:3
    x = gw(g,:);
    nx = ceil(length(x) / nwin) * nwin;
    x = [x, zeros(1, nx - length(x))]; % zero pad

    nseg1 = nx / nwin;
    xseg = zeros(nseg1 * 2 - 1, nwin * os);
    xseg(1:2:end, 1:nwin) = reshape(x, [nwin, nseg1])';

    if nseg1 > 1
        xseg(2:2:end, 1:nwin) = reshape(x(1 + nwin/2:end - nwin/2), [nwin, nseg1 - 1])';
    end

    xseg = xseg - mean(xseg, 2); % remove DC
    if nseg1 > 1
        win = 0.5 * (1 - cos(2 * pi * (1:nwin) / nwin));
        xseg(:, 1:nwin) = xseg(:, 1:nwin) .* win;
    end

    fseg = abs(fft(xseg, [], 2));
    fseg = fseg(:, 1:end/2);

    if nseg1 > 1
        gs = [gs; sqrt(mean(fseg.^2))]; % root-mean-square
    else
        gs = [gs; abs(fseg)];
    end
end

% Plot gradient spectrum
figure; hold on;
plot(faxis(1:nfmax), gs(:,1:nfmax), 'LineWidth', 1.1);
plot(faxis(1:nfmax), sqrt(sum(gs(:,1:nfmax).^2, 1)), 'k--', 'LineWidth', 1.5);

xlabel('Frequency (Hz)');
ylabel('Amplitude (a.u.)');
title('Gradient Spectrum');
legend({'Gx', 'Gy', 'Gz', 'G_{total}'}, 'Location', 'northeast');
grid on;
xlim([0 fmax]);

% Highlight acoustic resonance bands
if ~isempty(ascData)
    resFreqs = ascData.asGPAParameters(1).sGCParameters.aflAcousticResonanceFrequency;
    resBands = ascData.asGPAParameters(1).sGCParameters.aflAcousticResonanceBandwidth;
    
    y_limits = ylim;

    for i = 1:length(resFreqs)
        if resFreqs(i) > 0 && resBands(i) > 0
            f1 = resFreqs(i) - resBands(i)/2;
            f2 = resFreqs(i) + resBands(i)/2;
            fill([f1, f2, f2, f1], [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], ...
                [1, 0.7, 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
            line([resFreqs(i) resFreqs(i)], y_limits, 'Color', [1 0 0], 'LineStyle', '-', 'LineWidth', 1.2);
        end
    end
end

end