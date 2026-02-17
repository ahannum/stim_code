function [sl_th_90, sl_th_180] = computeRFSliceThickness(rf_ex, gz, rf_ref, g_ref)
% computeRFSliceThickness
% Computes the slice thickness of 90° excitation and 180° refocusing pulses
% using RF simulation and profile flank detection.
%
% Inputs:
%   rf_ex  - RF pulse structure for excitation
%   gz     - Gradient structure for excitation (must include amplitude)
%   rf_ref - RF pulse structure for refocusing
%   g_ref  - Gradient structure for refocusing (must include amplitude)
%
% Outputs:
%   sl_th_90   - Excitation slice thickness (in meters)
%   sl_th_180  - Refocusing slice thickness (in meters)

% Simulate excitation pulse
[M_z90, M_xy90, F2_90] = mr.simRf(rf_ex);
sl_th_90 = mr.aux.findFlank(F2_90(end:-1:1)/gz.amplitude, abs(M_xy90(end:-1:1)), 0.5) ...
         - mr.aux.findFlank(F2_90/gz.amplitude, abs(M_xy90), 0.5);

% Simulate refocusing pulse
[M_z180, M_xy180, F2_180, ref_eff] = mr.simRf(rf_ref);
sl_th_180 = mr.aux.findFlank(F2_180(end:-1:1)/g_ref.amplitude, abs(ref_eff(end:-1:1)), 0.5) ...
          - mr.aux.findFlank(F2_180/g_ref.amplitude, abs(ref_eff), 0.5);

% Plot RF profiles
figure;
plot(F2_90/gz.amplitude*1e3, abs(M_xy90), 'b', 'LineWidth', 1.2); hold on;
plot(F2_180/g_ref.amplitude*1e3, abs(ref_eff), 'r', 'LineWidth', 1.2);
title('RF Slice Profiles');
xlabel('Through-slice position (mm)');
ylabel('Signal amplitude');
legend('Excitation (M_{xy})', 'Refocusing (ref-eff)');
grid on;

% Print slice thicknesses
fprintf('Slice thickness (90° excitation pulse): %.3f mm\n', sl_th_90 * 1e3);
fprintf('Slice thickness (180° refocusing pulse): %.3f mm\n', sl_th_180 * 1e3);

end