% QC: the calibration part: the number of whole echo train should be the
% same as the undersampling part. 
% QC: dummy scans?
% same calibration scans for multiple repetitions?
% AJH: I increase rep in callibraiton rn if done in 2 shots, 

clear all;close all;

addpath('utils/pulseq-master/matlab/') % For Pulseq
addpath('utils/pulseq-master/matlab/demoUnsorted') % For PNS/CNS Calculation
addpath('utils/safe_pns_prediction-master/') % For PNS/CNS Calculation

%%  Enter User-Defined parameters here that will change! 
dummy_mode = 0;
gmax = 200; % Select max gradient amplitude as 200 or 45
gmax_diffusion = 200; % max amplitude for diffusion 
timings_only = false; % Stop after computing sequence parameters to save diffusion timings only
mmt = 0; % moment nulling level for diffusion, supports mmt = [0,1]
pns_python = 70; % percent PNS threshold for python design
bval = 3000; % target b-value 
target_bval = 3000; % If scaling down to 1000 for DTI

gropt_mode = 0; % Load GrOpt waveforms instead of Trap
Nslices = 1; %Number of slices
Nrep =1 %10; % Number of averages 
vec_mode = 1;

RSegment = 1; % Whether we split up the callibration or not 
rf_type = "slr-ls";
min_TE_mode = 1;


phs = 'external';
ssgr = 1;
if Nslices == 1
    TR = 3000e-3;
else
    TR = 115e-3
end

noRotDiff= 0;
bval_real = target_bval;


%% Initialize System 
if gmax == 200
    sys = mr.opts('MaxGrad',200,'GradUnit','mT/m',...
        'MaxSlew',200,'SlewUnit','T/m/s',...
        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T, Gmax for Cima is 200
    );
    
    sys_rf = mr.opts('MaxGrad',45,'GradUnit','mT/m',...
        'MaxSlew',90,'SlewUnit','T/m/s',...
        'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6, 'B0', 2.89);  % 50/130
    % set the slew to 90 for the DTI
    
    sys_epi = mr.opts('MaxGrad',68,'GradUnit','mT/m',...
        'MaxSlew',150, ...
        'SlewUnit','T/m/s',...
        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T % Max Slew for EPI Cima is 38, 180
    );
    
    sys_diff = mr.opts('MaxGrad',200,'GradUnit','mT/m',...
        'MaxSlew',100,'SlewUnit','T/m/s',...
        'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T
    );

    ascName = '/Users/ariel/Desktop/MP_GradSys_P034_X60.asc';
    readoutTime = 610e-6%590e-6; %590e-6;     % default value 590e-6, 680e-6,720e-6


elseif gmax == 80
    sys = mr.opts('MaxGrad',80,'GradUnit','mT/m',...
        'MaxSlew',200,'SlewUnit','T/m/s',...
        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T, Gmax for Cima is 200
    );
    
    sys_rf = mr.opts('MaxGrad',45,'GradUnit','mT/m',...
        'MaxSlew',100,'SlewUnit','T/m/s',...
        'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6, 'B0', 2.89);  % 50/130
    
    sys_epi = mr.opts('MaxGrad',68,'GradUnit','mT/m',...
        'MaxSlew',150, ...
        'SlewUnit','T/m/s',...
        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T % Max Slew for EPI Cima is 38, 180
    );
    
    sys_diff = mr.opts('MaxGrad',80,'GradUnit','mT/m',...
        'MaxSlew',100,'SlewUnit','T/m/s',...
        'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T
    );

    
    ascName = '/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_K2309_2250V_951A_XQ_GC04XQ.asc';
    readoutTime = 610e-6; %590e-6;     % default value 590e-6, 680e-6,720e-6




elseif gmax == 45
    sys = mr.opts('MaxGrad',45,'GradUnit','mT/m',...
        'MaxSlew',200,'SlewUnit','T/m/s',...
        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T, Gmax for Cima is 200
    );
    
    sys_rf = mr.opts('MaxGrad',45,'GradUnit','mT/m',...
        'MaxSlew',70,'SlewUnit','T/m/s',...
        'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6, 'B0', 2.89);  % 50/130
    
    sys_epi = mr.opts('MaxGrad',33,'GradUnit','mT/m',...
        'MaxSlew',150, ...
        'SlewUnit','T/m/s',...
        'rfRingdownTime', 30e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T % Max Slew for EPI Cima is 38, 180
    );
    
    sys_diff = mr.opts('MaxGrad',44,'GradUnit','mT/m',...
        'MaxSlew',100,'SlewUnit','T/m/s',...
        'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6,...
        'adcDeadTime', 10e-6, 'B0', 2.89 ... % this is Siemens' 3T
    );
    
    ascName = '/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_K2309_2250V_951A_XR_AS82.asc';
    readoutTime = 720e-6; %590e-6;     % default value 590e-6, 680e-6,720e-6

    % Test Spoiler Amp Area so can use the same area as 200/200 System 
    sys_spoiler = mr.opts('MaxGrad',45,'GradUnit','mT/m',...
        'MaxSlew',100,'SlewUnit','T/m/s',...
        'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6, 'B0', 2.89);  % 50/130
    spoiler_amp=3*8*42.58*10e2;
    est_rise=500e-6; % ramp time 280 us
    est_flat=2500e-6; %duration 600 us
    gp_tmp=mr.makeTrapezoid('x','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_spoiler);
    spoiler_area = gp_tmp.area; 
end

% %ascName = '/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_K2309_2250V_951A_XQ_GC04XQ.asc'
% ascName = '/Users/ariel/Documents/PhD/Projects/asc_files/MP_GradSys_K2309_2250V_951A_XR_AS82.asc';

%% Set Variable Inputs, These are consistent between systems and protocols
seq=mr.Sequence(sys);      % Create a new sequence object

% TR Between excitations 


fov=220e-3; Nx=146; Ny=Nx;  % 96 Define FOV and resolution, 156
pe_enable=1;               % a flag to quickly disable phase encoding (1/0) as needed for the delay calibration
ro_os=2;                   % oversampling factor (in contrast to the product sequence we don't really need it)
partFourierFactor= 6/8;      % partial Fourier factor: 1: full sampling 0: start with ky=0

NCalib = 30; % Number of calibration lines 
nNav = 3;

% Identify if using min TE or provided TE (for TRAP only)
if timings_only
    min_TE_mode = 0; % Identify if using min TE or provided TE (for TRAP only)
else
    min_TE_mode = 1;
end
if min_TE_mode > 0
    if ~ vec_mode
        diff_file_name = sprintf('diffusion_trap_parameters_m%d_gmax%d_bval%d_%s_pns%d.mat',mmt, gmax_diffusion, bval,rf_type,pns_python);
    elseif vec_mode
        diff_file_name = sprintf('diffusion_trap_parameters_m%d_gmax%d_bval%d_%s_pns%d_vec.mat',mmt, gmax_diffusion, bval,rf_type,pns_python);
    end
    if isfile(diff_file_name)
        load(diff_file_name);
    else
        %timings_only = true;
        warning('File not found: %s. Proceeding with timings_only = true.', diff_file_name);
        TE = 55e-3; % Echo Time of protocol
    end
else
    TE = 55e-3; % Echo Time of protocol
end

switch phs
    case 'external'
        nav_factor = 0;
    case 'internal'
        nav_factor = 1;
end


if gropt_mode
    if ~ vec_mode
        load(sprintf('diffusion_gropt_parameters_m%d_gmax%d_b%d_%s_pns%d.mat',mmt,gmax_diffusion,bval,rf_type,pns_python))
    TE = gDiff1.TE;
    elseif vec_mode
        load(sprintf('diffusion_gropt_parameters_m%d_gmax%d_b%d_%s_pns%d_vec.mat',mmt,gmax_diffusion,bval,rf_type,pns_python))
        TE = gDiff1.TE;
    end
end


% Identify number of slices, slice gap, and averages 
sliceThickness = 1.5e-3; % Slice Thickness
slice_gap=0e-3; % Gap between slices 

R=2; % Acceleration 

if R>1
    incl_calib = 1; % Whether to include the callibration or not in the sequence 
else
    incl_calib = 0;
end
calib_only = 0;

crusher_mode = 0; % Crushers 180 turn on/off


%% Define Diffusion Vector 
load('diffusion_directions_DTI.mat')
if target_bval == 1000
    diff_dir = b1000;
elseif target_bval == 3000
    diff_dir = b3000;
end




%diff_dir = [[0,0,0];
%     [ -0.668583, -0.668583, 0.325568 ]*-1;
%     [ -0.662784, 0.337216, -0.668583 ]*-1;
%     [ 0.337216, -0.662784, -0.668583 ]*-1];

Ndir = size(diff_dir,1);

%% Seq Name Generate
% Define additional variables
Ndirections = Ndir;  


% Determine base name for diffusion gradient mode
if gropt_mode
    mode_str = 'Gropt';
else
    mode_str = 'Trap';
end

% Add moment nulling label
mmt_str = sprintf('m%d', mmt);

% Add TE mode
if min_TE_mode
    te_str = 'minTE';
    
else
    te_str = 'fixTE';
end


switch rf_type
    case "sinc"
        sth_correction_90 = 1.35; % Correct for slice profile 
        sth_correction_180 = 1.5; % Correct for slice profile, 1.5 for sinc, 1.25 for SLR
        rfDur1 = 4e-3; % Duration of 90 RF Pulse 
        rfDur2 = 4.e-3; % Duration of 180 RF Pulse 8.8
        tbw_180 = 6;

    case "slr-min"
        sth_correction_90 = 1.35; % Correct for slice profile % want gradient on 90 and 180 the same, want the amplitude of both to be the same 
        sth_correction_180 = 0.9; % Correct for slice profile, 1.5 for sinc, 1.25 for SLR
        
        % Working combo for 180 is tbw_180 = 6, rfDur2 = 4e-3, correction
        % 1.2
        rfDur1 = 4.4e-3; % Duration of 90 RF Pulse 
        rfDur2 = 4.4e-3; % Duration of 180 RF Pulse ,5 and 6 good, but then flip angle adjust
        tbw_180 = 3;

    case "slr-ls"
        sth_correction_90 = 1.43; % Correct for slice profile % want gradient on 90 and 180 the same, want the amplitude of both to be the same 
        sth_correction_180 = 1.08 ; % Correct for slice profile, 1.5 for sinc, 1.25 for SLR
        % 1.25 and 1.5 work well 
        % Working combo for 180 is tbw_180 = 6, rfDur2 = 4e-3, correction
        % 1.2
        rfDur1 = 3.45e-3; % Duration of 90 RF Pulse 
        rfDur2 = 5.75e-3; % Duration of 180 RF Pulse ,5 and 6 good, but then flip angle adjust
        tbw_180 = 4.75;

    case "slr-ls-test"
        sth_correction_90 = 1.43; % Correct for slice profile % want gradient on 90 and 180 the same, want the amplitude of both to be the same 
        sth_correction_180 = 1.1 ; % Correct for slice profile, 1.5 for sinc, 1.25 for SLR
        % 1.25 and 1.5 work well 
        % Working combo for 180 is tbw_180 = 6, rfDur2 = 4e-3, correction
        % 1.2
        rfDur1 = 3.6e-3; % Duration of 90 RF Pulse 
        rfDur2 = 4.75e-3; % Duration of 180 RF Pulse ,5 and 6 good, but then flip angle adjust
        tbw_180 = 4.05;

    case "slr-ls-test2"
        sth_correction_90 = 1.5; % Correct for slice profile % want gradient on 90 and 180 the same, want the amplitude of both to be the same 
        sth_correction_180 = 1.1 ; % Correct for slice profile, 1.5 for sinc, 1.25 for SLR
        % 1.25 and 1.5 work well 
        % Working combo for 180 is tbw_180 = 6, rfDur2 = 4e-3, correction
        % 1.2
        rfDur1 = 4.5e-3; % Duration of 90 RF Pulse 
        rfDur2 = 4.75e-3; % Duration of 180 RF Pulse ,5 and 6 good, but then flip angle adjust
        tbw_180 = 4.05;

end

% Combine everything
seq_name = sprintf('Gmax%d_DTI_%s_%s_b%d_%s_Nsl%d_Ndir%d_Nrep%d_Calib%dshot_pns%d', ...
                   gmax_diffusion, mmt_str, mode_str, bval, rf_type, Nslices, Ndirections, Nrep, RSegment,pns_python);
if vec_mode
    seq_name = sprintf('Gmax%d_DTI_%s_%s_b%d_%s_Nsl%d_Ndir%d_Nrep%d_Calib%dshot_pns%d_vec', ...
                   gmax_diffusion, mmt_str, mode_str, bval, rf_type, Nslices, Ndirections, Nrep, RSegment,pns_python);
end

if target_bval ~= bval
    seq_name = sprintf('%s_realb%d', seq_name, target_bval);
end

seq_waveform_name = seq_name;

% Append ssgr if enabled
if ssgr == 1
    seq_name = sprintf('%s_ssgr', seq_name);
end

if noRotDiff == 1
    seq_name = sprintf('%s_NOROT',seq_name);
end

% Add file extension
seq_name = [seq_name, '.seq'];
seq_waveform_name = [seq_waveform_name, '.seq'];


%% Initialization -- RF Pulses
% Fat Sat
sat_ppm=-3.45;
rf_fs = mr.makeGaussPulse(110*pi/180,'system',sys,'Duration',8e-3,...
    'bandwidth',abs(sat_ppm*1e-6*sys.B0*sys.gamma),'freqPPM',sat_ppm,'use','saturation');
rf_fs.phasePPM=-2*pi*rf_fs.freqPPM*rf_fs.center; % compensate for the frequency-offset induced phase    


% Spoilers from Liu et al., MRM, 2024
spoiler_amp=3*8*42.58*10e2;
est_rise=500e-6; % ramp time 280 us
est_flat=2500e-6; %duration 600 us


gp_r=mr.makeTrapezoid('x','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_rf);
gp_p=mr.makeTrapezoid('y','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_rf);
gp_s=mr.makeTrapezoid('z','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_rf);

gn_r=mr.makeTrapezoid('x','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',sys_rf);
gn_p=mr.makeTrapezoid('y','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',sys_rf);
gn_s=mr.makeTrapezoid('z','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',sys_rf);


fatsat_timing = mr.calcDuration(gp_r,gp_p,gp_s) + mr.calcDuration(rf_fs,gn_r,gn_p,gn_s);

% Design 90-degree and 180-degree RF Pulses 
switch rf_type
    case "sinc"
        % RF 90
        [rf_90, gz, gzReph] = mr.makeSincPulse(pi/2,'system',sys_rf,'duration',rfDur1,...
            'sliceThickness',sliceThickness*sth_correction_90,'apodization',0.5,'timeBwProduct',4,'use','excitation');

        rf_90_duration = mr.calcDuration(rf_90,gz) +  mr.calcDuration(gzReph);
        rf_90_rfCenterInclDelay=rf_90.delay + mr.calcRfCenter(rf_90);
        
        % RF 180
        [rf_180, gz_180] = mr.makeSincPulse(pi,'system',sys_rf,'duration',rfDur2,...
            'sliceThickness',sliceThickness*sth_correction_180,'apodization', 0.5, ...
            'timeBwProduct',tbw_180,'PhaseOffset',pi/2,'use','refocusing');
        
    case "slr-ls-relaxed"
          [rf_90, gz, gzReph] = mr.makeSLRpulse(pi/2,'duration',rfDur1,'sliceThickness',sliceThickness*sth_correction_90,...
            'timeBwProduct',4.1,'dwell',rfDur1/500,'passbandRipple',3e-3,'stopbandRipple',1e-3,...
            'filterType','ls','system',sys_rf,'use','excitation', 'PhaseOffset' ,pi/2); 
        rf_90_duration = mr.calcDuration(rf_90,gz) +  mr.calcDuration(gzReph);
        rf_90_rfCenterInclDelay=rf_90.delay + mr.calcRfCenter(rf_90);
            
        
        % RF 180
        [rf_180, gz_180] =  mr.makeSLRpulse(pi,'duration',rfDur2,'sliceThickness',sliceThickness*sth_correction_180,...
                'timeBwProduct',tbw_180,'dwell',rfDur2/500,'passbandRipple',0.9e-3,'stopbandRipple',6e-3,...
                'filterType','ls','system',sys_rf,'use','refocusing', 'PhaseOffset' ,0); %passband oiginally 1

    case "slr-ls"
        % RF 90
        [rf_90, gz, gzReph] = mr.makeSLRpulse(pi/2,'duration',rfDur1,'sliceThickness',sliceThickness*sth_correction_90,...
            'timeBwProduct',4.1,'dwell',rfDur1/500,'passbandRipple',5e-3,'stopbandRipple',1e-3,...
            'filterType','ls','system',sys_rf,'use','excitation', 'PhaseOffset' ,pi/2); 
        rf_90_duration = mr.calcDuration(rf_90,gz) +  mr.calcDuration(gzReph);
        rf_90_rfCenterInclDelay=rf_90.delay + mr.calcRfCenter(rf_90);
            
        
        % RF 180
        [rf_180, gz_180] =  mr.makeSLRpulse(pi,'duration',rfDur2,'sliceThickness',sliceThickness*sth_correction_180,...
                'timeBwProduct',tbw_180,'dwell',rfDur2/500,'passbandRipple',1.2e-3,'stopbandRipple',6e-3,...
                'filterType','ls','system',sys_rf,'use','refocusing', 'PhaseOffset' ,0); %passband oiginally 1

    case "slr-ls-test"
        % RF 90
        [rf_90, gz, gzReph] = mr.makeSLRpulse(pi/2,'duration',rfDur1,'sliceThickness',sliceThickness*sth_correction_90,...
            'timeBwProduct',4.1,'dwell',rfDur1/500,'passbandRipple',5e-3,'stopbandRipple',1e-3,...
            'filterType','ls','system',sys_rf,'use','excitation', 'PhaseOffset' ,pi/2); 
        rf_90_duration = mr.calcDuration(rf_90,gz) +  mr.calcDuration(gzReph);
        rf_90_rfCenterInclDelay=rf_90.delay + mr.calcRfCenter(rf_90);
            
        
        % RF 180
        [rf_180, gz_180] =  mr.makeSLRpulse(pi,'duration',rfDur2,'sliceThickness',sliceThickness*sth_correction_180,...
                'timeBwProduct',tbw_180,'dwell',rfDur2/500,'passbandRipple',1.2e-3,'stopbandRipple',6e-3,...
                'filterType','ls','system',sys_rf,'use','refocusing', 'PhaseOffset' ,0); %passband oiginally 1
    case "slr-ls-test2"
        % RF 90
        [rf_90, gz, gzReph] = mr.makeSLRpulse(pi/2,'duration',rfDur1,'sliceThickness',sliceThickness*sth_correction_90,...
            'timeBwProduct',3.75,'dwell',rfDur1/500,'passbandRipple',5e-3,'stopbandRipple',1e-3,...
            'filterType','ls','system',sys_rf,'use','excitation', 'PhaseOffset' ,pi/2); 
        rf_90_duration = mr.calcDuration(rf_90,gz) +  mr.calcDuration(gzReph);
        rf_90_rfCenterInclDelay=rf_90.delay + mr.calcRfCenter(rf_90);
            
        
        % RF 180
        [rf_180, gz_180] =  mr.makeSLRpulse(pi,'duration',rfDur2,'sliceThickness',sliceThickness*sth_correction_180,...
                'timeBwProduct',tbw_180,'dwell',rfDur2/500,'passbandRipple',1.2e-3,'stopbandRipple',6e-3,...
                'filterType','ls','system',sys_rf,'use','refocusing', 'PhaseOffset' ,0); %passband oiginally 1

end

% Confirm Slice Thickness
[sl90, sl180] = pseq2.computeRFSliceThickness(rf_90, gz, rf_180,gz_180);

% Print amplitudes of gz and gz_180
fprintf('Amplitudes of RF gradeints gz = %0.2f mT/m, and gz 180 = %0.2f \n',mr.convert(gz.amplitude,'Hz/m','mT/m'),mr.convert(gz_180.amplitude,'Hz/m','mT/m') )
fprintf('Amplitude of B1 of RF_180 is: %0.2f mT, System max b1 is %0.2f mT \n',mr.convert(max(rf_180.signal(:)),'Hz/m','mT/m')*1e3, mr.convert(sys.maxB1,'Hz/m','mT/m')*1e3)
fprintf('Amplitude of B1 of RF_90 is: %0.2f mT, System max b1 is %0.2f mT \n',mr.convert(max(rf_90.signal(:)),'Hz/m','mT/m')*1e3, mr.convert(sys.maxB1,'Hz/m','mT/m')*1e3)


%%
% Spoilers Around 180 from Liu et al., MRM, 2024
crusher_d=0.95e-3;
gz180_crusher_1=mr.makeTrapezoid('z',sys_rf,'Amplitude',19.05*42.58*10e2,'Duration',crusher_d); % that's what we used for Siemens
gz180_crusher_2=mr.makeTrapezoid('y',sys_rf,'Amplitude',19.05*42.58*10e2,'Duration',crusher_d); % that's what we used for Siemens
gz180_crusher_3=mr.makeTrapezoid('x',sys_rf,'Amplitude',19.05*42.58*10e2,'Duration',crusher_d); % that's what we used for Siemens

[~, gzr1_t, gzr1_a]=mr.makeExtendedTrapezoidArea('z',0, gz_180.amplitude, gz180_crusher_1.area ,sys_rf);
[~, gzr2_t, gzr2_a]=mr.makeExtendedTrapezoidArea('z', gz_180.amplitude,0, gz180_crusher_1.area,sys_rf);

rf_180_crush_dur = mr.calcDuration(gz180_crusher_1,gz180_crusher_2,gz180_crusher_3);

if gz_180.delay>(gzr1_t(3)-gz_180.riseTime)
    gz_180.delay=gz_180.delay-(gzr1_t(3)-gz_180.riseTime);
else
    rf_180.delay=rf_180.delay+(gzr1_t(3)-gz_180.riseTime)-gz_180.delay;
    gz_180.delay=0;
end

gz180n=mr.makeExtendedTrapezoid('z','system',sys_rf,'times',[gzr1_t gzr1_t(3)+gz_180.flatTime+gzr2_t]+gz_180.delay, 'amplitudes', [gzr1_a gzr2_a]);
gz_180 = gz180n;
rf_180_duration = mr.calcDuration(rf_180,gz_180);
rf_180_rfCenterInclDelay = rf_180.delay + mr.calcRfCenter(rf_180);


% define the output trigger to play out with every slice excitation
trig=mr.makeDigitalOutputPulse('osc0','duration', 100e-6); % possible channels: 'osc0','osc1','ext1'

%% Initialization -- EPI
% Define other gradients and ADC events
deltak=1/fov;
deltaky = R*deltak; % Rsegement*R
kWidth = Nx*deltak;

disp(['deltak = ',num2str(deltak),' deltaky = ', num2str(deltaky)]);
blip_dur = ceil(2*sqrt(deltaky/sys_epi.maxSlew)/10e-6/2)*10e-6*2; % we round-up the duration to 2x the gradient raster time
gy = mr.makeTrapezoid('y',sys_epi,'Area',-deltaky,'Duration',blip_dur); % we use negative blips to save one k-space line on our way towards the k-space center

extra_area=blip_dur/2*blip_dur/2*sys_epi.maxSlew; % check unit!;
gx = mr.makeTrapezoid('x',sys_epi,'Area',kWidth+extra_area,'duration',readoutTime+blip_dur);
% actual sampled area = whole area - rampup deadarea - rampdown deadarea
actual_area=gx.area-gx.amplitude/gx.riseTime*blip_dur/2*blip_dur/2/2-gx.amplitude/gx.fallTime*blip_dur/2*blip_dur/2/2;
gx.amplitude=gx.amplitude/actual_area*kWidth; % rescale amplitude to make sampled area = kWidth
gx.area = gx.amplitude*(gx.flatTime + gx.riseTime/2 + gx.fallTime/2); % udpate parameters relative to amplitude
gx.flatArea = gx.amplitude*gx.flatTime;
assert(gx.amplitude<=sys_epi.maxGrad);
ESP = 1e3 * mr.calcDuration(gx) ; % echo spacing, ms
disp(['echo spacing = ', num2str(ESP), ' ms']) ;

assert(ro_os>=2) ;
adcSamples=Nx*ro_os ;
adcDwell=floor(readoutTime/adcSamples*1e7)*1e-7;
disp(['ADC bandwidth = ', num2str(1/adcDwell/1000), ' kHz']) ;
fprintf('Actual RO oversampling factor is %g, Siemens recommends it to be above 1.3\n', deltak/gx.amplitude/adcDwell)
fprintf('Blip Dur %g \n', blip_dur)
adc = mr.makeAdc(adcSamples,'Dwell',adcDwell,'Delay',blip_dur/2,'phaseModulation',0.1*rand(1,adcSamples),'system',sys_epi);
% realign the ADC with respect to the gradient
time_to_center=adc.dwell*((adcSamples-1)/2+0.5); % I've been told that Siemens samples in the center of the dwell period
adc.delay=round((gx.riseTime+gx.flatTime/2-time_to_center)*1e6)*1e-6; % we adjust the delay to align the trajectory with the gradient. We have to aligh the delay to 1us 

% split the blip into two halves and produce a combined synthetic gradient
gy_parts = mr.splitGradientAt(gy, blip_dur/2, sys_epi);
[gy_blipup, gy_blipdown,~]=mr.align('right',gy_parts(1),'left',gy_parts(2),gx);
gy_blipdownup=mr.addGradients({gy_blipdown, gy_blipup}, sys_epi);

% pe_enable support
gy_blipup.waveform=gy_blipup.waveform*pe_enable;
gy_blipdown.waveform=gy_blipdown.waveform*pe_enable;
gy_blipdownup.waveform=gy_blipdownup.waveform*pe_enable;

% phase encoding and partial Fourier
Ny_pre=round(Ny*partFourierFactor - floor(Ny/2)-1); % PE steps prior to ky=0, excluding the central line
Ny_pre=round(Ny_pre/1/R);

Ny_post=round(Ny/2+1); % PE lines after the k-space center including the central line
Ny_post=round(Ny_post/1/R);
Ny_meas=Ny_pre+Ny_post;

% Pre-phasing gradients
gxPre = mr.makeTrapezoid('x',sys_epi,'Area',-gx.area/2);
gyPre = mr.makeTrapezoid('y',sys_epi,'Area',Ny_pre*deltaky);
[gxPre,gyPre,gzReph]=mr.align('right',gxPre,'left',gyPre,gzReph);
% relax the PE prepahser to reduce stimulation
gyPre = mr.makeTrapezoid('y',sys_epi,'Area',gyPre.area,'Duration',mr.calcDuration(gxPre,gyPre,gzReph));
gyPre.amplitude=gyPre.amplitude*pe_enable;

disp(['BW/Pixel = ', num2str(1/adc.duration), ' Hz/Px'])
disp('EPI gradients and ADC prepared.');

% Rewind to Center K-space
gx_post = mr.makeTrapezoid('x',sys_rf,'Area',-gx.area/2);
gy_post = mr.makeTrapezoid('y',sys_rf,'Area',(Ny_post-1)*deltaky);
[gx_post,gy_post]=mr.align('right',gx_post,'left',gy_post);
gy_post = mr.makeTrapezoid('y',sys_rf,'Area',gy_post.area,'Duration',mr.calcDuration(gx_post,gy_post));


% EPI Timings 
gxPre_adj = gxPre;
gxPre_adj.delay = 0;
nav_dur = (nNav * mr.calcDuration(gx) + mr.calcDuration(gxPre_adj)) * nav_factor ; % Add prewind and rewinder time 
nav_dur_actual = nNav * mr.calcDuration(gx) + mr.calcDuration(gxPre_adj); % Add prewind and rewinder time 
timeToTE = Ny_pre*mr.calcDuration(gx) + mr.calcDuration(gx)/2;       
timeToTE = timeToTE + mr.calcDuration(gyPre,gxPre) - adc.dwell/2;
totalReadout = Ny_pre*mr.calcDuration(gx) + mr.calcDuration(gx) + mr.calcDuration(gyPre,gxPre) - adc.dwell/2;  

start_gx = gx;


%% Initialization -- Calibration if R>1
if R > 1
    deltak_calib = 1 / fov; % compute deltak
    deltaky_calib = deltak_calib;
    blip_dur_calib = ceil(2*sqrt(deltaky_calib/sys.maxSlew)/10e-6/2)*10e-6*2; % we round-up the duration to 2x the gradient raster time
    gy_calib = mr.makeTrapezoid('y',sys,'Area',-deltaky_calib,'Duration',blip_dur_calib); % we use negative blips to save one k-space line on our way towards the k-space center

    gy_parts_calib = mr.splitGradientAt(gy_calib, blip_dur_calib/2, sys);
    [gy_blipup_calib, gy_blipdown_calib,~]=mr.align('right',gy_parts_calib(1),'left',gy_parts_calib(2),gx);
    gy_blipdownup_calib=mr.addGradients({gy_blipdown_calib, gy_blipup_calib}, sys);
    
    Ny_calib = Ny_pre * 2 + 1;
    if NCalib > Ny_calib
         warning('NCalib (%d) is greater than maximum number of callibrations lines (%d). NCalib set to Ny_calib.', NCalib, Ny_calib);
    end
    
    
    Ny_pre_calib=round(Ny_calib/2-1); % PE steps prior to ky=0, excluding the central line
    Ny_pre_calib=round(Ny_pre_calib); 

    % phase encoding and partial Fourier
    Ny_pre_calib=round(Ny_calib/2-1); % PE steps prior to ky=0, excluding the central line

    
    Ny_post_calib=round(Ny_calib/2+1); % PE lines after the k-space center including the central line
    Ny_meas_calib=Ny_pre_calib+Ny_post_calib;

    % Pre-phasing gradients
    gyPre_calib = mr.makeTrapezoid('y',sys,'Area',Ny_pre_calib*deltak);
    [gxPre_calib,gyPre_calib]=mr.align('right',gxPre,'left',gyPre_calib);
    % relax the PE prepahser to reduce stimulation
    gyPre_calib = mr.makeTrapezoid('y',sys,'Area',gyPre_calib.area,'Duration',mr.calcDuration(gxPre,gyPre_calib));
    
    calib_R = 1;
    if RSegment > 1
        calib_R = 2;
    end

    gyPre_cell_calib = cell(1,RSegment);
    gyPre_cell_calib{1,1} = gyPre_calib;

    
    
    gy_post_calib = mr.makeTrapezoid('y',sys_rf,'Area',(Ny_post_calib-1)*deltaky_calib);
    [gx_post_calib,gy_post_calib]=mr.align('right',gx_post,'left',gy_post_calib);
    gy_post_calib = mr.makeTrapezoid('y',sys_rf,'Area',gy_post_calib.area,'Duration',mr.calcDuration(gx_post_calib,gy_post_calib));

    gyPost_cell_calib = cell(1,RSegment);
    gyPost_cell_calib{1,1} = gy_post_calib;
    

    
    if RSegment > 1
        gyPre_cell_calib{1,1} = gyPre;
        gyPost_cell_calib{1,1} = gy_post;
        
        for rr = 2:RSegment
            %gyPre_next = gyPre;
            gyPre_next = mr.makeTrapezoid('y',sys,'Area',gyPre.area - deltak,'Duration',mr.calcDuration(gxPre,gyPre));
    
            gyPre_cell_calib{1,rr} = gyPre_next;

            
            gy_post_next = mr.makeTrapezoid('y',sys_rf,'Area',gy_post.area + deltak,'Duration',mr.calcDuration(gx_post,gy_post));
            gyPost_cell_calib{1,rr} = gy_post_next;
            

        end

        Ny_meas_calib = Ny_meas; % We acquire in segments 

        gy_blipup_calib = gy_blipup;
        gy_blipdownup_calib = gy_blipdownup;
        gy_blipdown_calib = gy_blipdown;


        actual_esp=gx.riseTime+gx.flatTime+gx.fallTime;
        TEShift=actual_esp./RSegment;
        TEShift=(TEShift/ sys.gradRasterTime) * sys.gradRasterTime; % Put it on the Raster 



    end

    disp('EPI Calibration Prepared');

    
end

%% Initialization  -- Get timings to construct diffusion waveforms 
delayTE1 = ceil((TE / 2 - rf_90_duration + rf_90_rfCenterInclDelay - nav_dur - ...
    rf_180_rfCenterInclDelay ) / sys.gradRasterTime) * sys.gradRasterTime;
delayTE2 = ceil((TE / 2 - rf_180_duration  + rf_180_rfCenterInclDelay - timeToTE) / sys.gradRasterTime) * sys.gradRasterTime;

assert(delayTE1 >= 0, 'Delay TE1 must be non-negative');
assert(delayTE2 >= 0, 'Delay TE2 must be non-negative');


% Save Timing Parameters
save_name_timings = sprintf('diffusion_timing_parameters_Gmax%d_%s.mat', gmax,rf_type);
save(save_name_timings, ...
    'rf_90_duration', 'rf_90_rfCenterInclDelay', ...
    'rf_180_duration', 'rf_180_rfCenterInclDelay', ...
    'nav_dur', 'timeToTE', ...
    'gz_180','gz', 'gxPre', 'gx', 'nNav', 'Ny_meas', ...
    'sys');  

if timings_only
    % 90 RF Pulse
    test_seq=mr.Sequence(sys);      % Create a new sequence object
    test_seq.addBlock(rf_90,gz);
    test_seq.addBlock(gzReph);
    [wave_data_rf90, tfp_excitation, ~, ~, ~]  =  test_seq.waveforms_and_times(true);
    %[pns_result, pns_norm_90, pns_comp_90, t_axis_90] = test_seq.calcPNS(ascName);
    
    % Fat Sat: 
    test_seq= mr.Sequence(sys);
    test_seq.addBlock(gp_r,gp_p,gp_s);
    test_seq.addBlock(rf_fs, gn_r,gn_p,gn_s);
    [wave_data_fatsat, ~, ~, ~, ~]  =  test_seq.waveforms_and_times(true);
    

    % Crushers + 180
    test_seq=mr.Sequence(sys);      % Create a new sequence object
    gx_180 = gz_180;
    gx_180.channel = 'x';
    gy_180 = gz_180;
    gy_180.channel = 'y';
    if size(gy_180.waveform,1) == 6
                gy_180.waveform(3:4) = 0;
                gx_180.waveform(3:4) = 0;
    elseif size(gy_180.waveform,1) == 8
        gy_180.waveform(4:5) = 0;
        gx_180.waveform(4:5) = 0;
    end
    test_seq.addBlock(rf_180,gz_180,gy_180,gx_180);
    [wave_data_rf180, ~, tfp_refocusing, ~, ~]  =  test_seq.waveforms_and_times(true);
    %[pns_result, pns_norm_180, pns_comp_180, t_axis_180] = test_seq.calcPNS(ascName);
    
    % EPI
    test_seq=mr.Sequence(sys);      % Create a new sequence object
    test_seq.addBlock(gyPre, gxPre );% lin/nav/avg reset
    for i = 1:Ny_meas
        if i == 1
            test_seq.addBlock(gx, gy_blipup);
        elseif i == Ny_meas
            test_seq.addBlock(gx, gy_blipdown);
        else
            test_seq.addBlock(gx, gy_blipdownup);
        end
        % Reverse polarity of read gradient
        gx = mr.scaleGrad(gx, -1);
    end
    [wave_data_epi, ~, ~, ~, ~]  =  test_seq.waveforms_and_times(true);
    %[pns_result, pns_norm_epi, pns_comp_epi, t_axis_epi] = test_seq.calcPNS(ascName);

    save_name_timings = sprintf('diffusion_timing_parameters_Gmax%d_%s_waveforms.mat', gmax,rf_type);
    save(save_name_timings, ...
    'wave_data_rf90', 'wave_data_rf180', 'wave_data_fatsat',...
    'tfp_excitation','tfp_refocusing','wave_data_epi');  

    return
end

%% Initialization -- Diffusion

if gropt_mode == 0
    if min_TE_mode > 0
        gDiff = mr.makeTrapezoid('z', 'amplitude', mr.convert(amplitude*1e3,'mT/m','Hz/m') ,'riseTime',rise_time, 'flatTime', flat_time,'system',sys);
        
         if mmt == 1
            gDiff_extended = mr.makeExtendedTrapezoid('z','system',sys_diff, ...
                                    'amplitudes',[0, gDiff.amplitude, gDiff.amplitude,0, -gDiff.amplitude, -gDiff.amplitude,0,], ...
                                    'times', [0, rise_time, rise_time+flat_time, mr.calcDuration(gDiff), 3*rise_time+flat_time, 3*rise_time+2*flat_time, 4*rise_time+2*flat_time ]);
            gDiff = gDiff_extended;
         end
        
        
        
        [actual_bval] = pseq2.Monopolar.getRealBValue(gDiff, sys_diff.gradRasterTime, rf_90_duration + nav_dur, rf_90_rfCenterInclDelay, ...
                                   rf_180_duration, delayTE1, delayTE2, sys_diff);
        %elseif mmt==1
        %    [actual_bval, moment] = pseq2.Bipolar.getRealBValue(gDiff, sys_diff.gradRasterTime, rf_90_duration + nav_dur, rf_90_rfCenterInclDelay, ...
        %                           rf_180_duration, delayTE1, delayTE2, sys_diff);
        %end
        fprintf('The computed b-value is %.2f s/mm^2\n', actual_bval);
    
    
        if actual_bval > target_bval 
            gDiff  = mr.scaleGrad(gDiff, sqrt(target_bval / actual_bval));
            if mmt == 0
                actual_bval = pseq2.Monopolar.getRealBValue(gDiff, sys_diff.gradRasterTime, rf_90_duration + nav_dur, rf_90_rfCenterInclDelay, ...
                                   rf_180_duration, delayTE1, delayTE2, sys_diff);
                fprintf('Scaled G, updated b-value is %.2f s/mm^2\n', actual_bval);
            elseif mmt ==1
                actual_bval = pseq2.Bipolar.getRealBValue(gDiff, sys_diff.gradRasterTime, rf_90_duration + nav_dur, rf_90_rfCenterInclDelay, ...
                                   rf_180_duration, delayTE1, delayTE2, sys_diff);
                fprintf('Scaled G, updated b-value is %.2f s/mm^2\n', actual_bval);
            end
        end
    else
        if mmt == 0
            diff = pseq2.Monopolar(bval, delayTE1, delayTE2, rf_180, gz_180, rf_90_duration, rf_90_rfCenterInclDelay, rf_180_duration, sys_diff.gradRasterTime, sys_diff.maxGrad, sys_diff.maxSlew, sys_diff);
        elseif mmt == 1
            diff = pseq2.Bipolar(bval, delayTE1, delayTE2, rf_180, gz_180, rf_90_duration, rf_90_rfCenterInclDelay, rf_180_duration, sys_diff.gradRasterTime, sys_diff.maxGrad, sys_diff.maxSlew, sys_diff);
        end
        gDiff = diff.gDiff;
    end

else
    %diff1 = mr.makeExtendedTrapezoid('z', 'system', sys, 'times', (gDiff1.time - gDiff1.time(1)) * 1e-3, ...
    %    'amplitudes', mr.convert(gDiff1.gradient,'mT/m','Hz/m'));
   % diff2 = mr.makeExtendedTrapezoid('z', 'system', sys, 'times', (gDiff2.time - gDiff2.time(1)) * 1e-3, ...
    %    'amplitudes', mr.convert(gDiff2.gradient,'mT/m','Hz/m'));

    diff1 = mr.makeArbitraryGrad('z',mr.convert(gDiff1.gradient,'mT/m','Hz/m'),sys,'oversampling',false,'first',0,'last',0);
    diff2 = mr.makeArbitraryGrad('z',mr.convert(gDiff2.gradient,'mT/m','Hz/m'),sys,'oversampling',false,'first',0,'last',0);
    
    t_90 = rf_90_duration - rf_90_rfCenterInclDelay;
    t_180 = rf_180_duration;
    dt = sys.gradRasterTime;
    dummy_seq = mr.Sequence(sys);
    dummy_seq.addBlock(mr.makeDelay(t_90));
    dummy_seq.addBlock(mr.makeDelay(delayTE1), diff1);
    dummy_seq.addBlock(mr.makeDelay(t_180));
    dummy_seq.addBlock(mr.makeDelay(delayTE2), mr.scaleGrad(diff2, -1));
    

    % Extract waveform data
    [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
     tmin = 0;
    tmax = max(wave_data{3}(1, :));
    tt = tmin:dt:tmax;  
    g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
    g_interp(isnan(g_interp)) = 0;
    % Convert to Tesla/meter
    g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;
    % Calculate b-value
    bval_tmp = pseq2.Monopolar.get_bvalue(g_array_Tpm, dt);

    %actual_bval = pseq2.Monopolar.get_bvalue(G_full_inv*1e-3, sys.gradRasterTime);
    actual_bval = bval_tmp;
    fprintf('B-value of GrOpt waveform is %0.2f s/mm^2\n', bval_tmp)

    if bval_tmp > target_bval
        diff1 = mr.scaleGrad(diff1, sqrt(target_bval / actual_bval));
        diff2 = mr.scaleGrad(diff2, sqrt(target_bval / actual_bval));
        %G_full2 = G_full * sqrt(target_bval/actual_bval);

        t_90 = rf_90_duration - rf_90_rfCenterInclDelay;
        t_180 = rf_180_duration;
        dt = sys.gradRasterTime;
        dummy_seq = mr.Sequence(sys);
        dummy_seq.addBlock(mr.makeDelay(t_90));
        dummy_seq.addBlock(mr.makeDelay(delayTE1), diff1);
        dummy_seq.addBlock(mr.makeDelay(t_180));
        dummy_seq.addBlock(mr.makeDelay(delayTE2), mr.scaleGrad(diff2, -1));
        
        % Extract waveform data
        [wave_data, ~, ~, ~, ~] = dummy_seq.waveforms_and_times(true);
         tmin = 0;
        tmax = max(wave_data{3}(1, :));
        tt = tmin:dt:tmax;  
        g_interp = interp1(wave_data{3}(1,:), wave_data{3}(2,:), tt, 'linear');
        g_interp(isnan(g_interp)) = 0;
        % Convert to Tesla/meter
        g_array_Tpm = mr.convert(g_interp, 'Hz/m', 'mT/m') * 1e-3;
        % Calculate b-value
        actual_bval = pseq2.Monopolar.get_bvalue(g_array_Tpm, dt);         
         %actual_bval = pseq2.Monopolar.get_bvalue(G_full2*1e-3, sys.gradRasterTime);
         fprintf('Updated B-value of GrOpt waveform is %0.2f s/mm^2\n',actual_bval)
    end

    if mr.calcDuration(diff1) > delayTE1
        TE = ceil(2 * (delayTE1 + rf_90_duration - rf_90_rfCenterInclDelay + nav_dur + rf_180_rfCenterInclDelay) / sys.gradRasterTime) * sys.gradRasterTime;
        delayTE2 = ceil((TE / 2 - rf_180_duration  + rf_180_rfCenterInclDelay - timeToTE) / sys.gradRasterTime) * sys.gradRasterTime;
    end

end

%% Initialization -- Slice Positiongs 
 %slice_positions = (params.thickness + params.sliceGap) * ((0:(params.Nslices-1)) - (params.Nslices-1)/2);
 slice_positions = (sliceThickness + slice_gap) * ((0:(Nslices-1)) - (Nslices-1)/2);

% Reorder slices for interleaved acquisition 
slice_positions = slice_positions([1:2:Nslices, 2:2:Nslices]);

[slice_close_iso, idxClosest] = min(abs(slice_positions));

%% Calculate -- Delay TR
delayTR =  ceil((TR - fatsat_timing - rf_90_duration - nav_dur - mr.calcDuration(gy_post,gx_post) - ...
                        totalReadout - rf_180_duration - delayTE1 - delayTE2)/sys.gradRasterTime)*sys.gradRasterTime;
delayTR = ceil(delayTR / sys.blockDurationRaster) * sys.blockDurationRaster; 

TR_slc_min = ceil((fatsat_timing + rf_90_duration + nav_dur +  mr.calcDuration(gy_post,gx_post) + ...
                        totalReadout + rf_180_duration + delayTE1 + delayTE2)/sys.gradRasterTime)*sys.gradRasterTime;

fprintf('Min Slice TR is %.2f ms\n', TR_slc_min*1e3);
fprintf('TE is %.2f ms\n', TE*1e3);
ROpolarity = sign(gx.amplitude);  % Store original polarity   
gz_180_start = gz_180;
rf_180_start= rf_180;

%% Construct Sequence -- Noise scan for GRAPPA recon 
seq.addBlock(mr.makeLabel('SET', 'LIN', 0),mr.makeLabel('SET','SLC', 0),mr.makeLabel('SET','REP', 0)) ;
seq.addBlock(mr.makeDelay(ceil(mr.calcDuration(adc)/sys.blockDurationRaster)*sys.blockDurationRaster),adc, mr.makeLabel('SET', 'NOISE', true)) ;
seq.addBlock(mr.makeLabel('SET', 'NOISE', false)) ;


%% Include 3 dummy scans at beginning for stabilization: 

if dummy_mode == 1
for  dummy = 1:3    %seq.addBlock(mr.makeLabel('SET','REF', true)) ; % set up Mdh.setPATRefScan header for GRAPPA
    for Nmulti = 1:RSegment
        for s = 1:1
            slicePosition = slice_positions(s);
            
            % Add Fat Sat 
            seq.addBlock(gp_r,gp_p,gp_s);
            seq.addBlock(rf_fs, gn_r,gn_p,gn_s);
        
            % Excitation 
            rf_90.freqOffset = gz.amplitude * slicePosition;
            rf_90.phaseOffset = -2 * pi * rf_90.freqOffset * mr.calcRfCenter(rf_90);

            seq.addBlock(rf_90, gz, trig)
        
            % Add Navigator 
            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
            gx = start_gx;
            gx = mr.scaleGrad(gx, -1);
            
            start_nav = Ny/2  ;
            
            if partFourierFactor < 1
                start_nav = round(Ny * (1- partFourierFactor)) ;
            end
        
            seq.addBlock(gxPre_calib, gzReph)
           
            % Reverse gxPre back
            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
            for n = 1:nNav
              

                seq.addBlock(gx);
        
         
                % Reverse polarity of read gradient
                gx = mr.scaleGrad(gx, -1);
        
                
            end
        
            gxPost_calib = mr.scaleGrad(gxPre_calib, -1);
            gxPost_calib.delay = 0;
            seq.addBlock(gxPost_calib); % k-space center line
        
            % Add Delay 
            if nav_factor == 0
                seq.addBlock(mr.makeDelay(delayTE1 - nav_dur_actual));
            else
                seq.addBlock(mr.makeDelay(delayTE1));
            end

            gz_180 = gz_180_start;
            
            rf_180 = rf_180_start;
            % Add Refocusing
            if isfield(gz_180, 'amplitude')
                rf_180.freqOffset = gz_180.amplitude * slicePosition;
            else
                amplitude = gz_180.waveform(4);
                rf_180.freqOffset = amplitude * slicePosition;
            end 

            rf_180.phaseOffset = pi/2 -2 * pi * rf_180.freqOffset * mr.calcRfCenter(rf_180);
        
            if ssgr > 0
                gz_180 = mr.scaleGrad(gz_180_start,-1);
                rf_180.freqOffset = rf_180.freqOffset * -1;
                rf_180.phaseOffset =  rf_180.phaseOffset * -1;
            end
            
            gz_180_add = gz_180;
            gy_180_add = gz_180;
            gx_180_add = gz_180;

            gy_180_add.channel = 'y';
            gx_180_add.channel = 'x';

            if size(gy_180_add.waveform,1) == 6
                gy_180_add.waveform(3:4) = 0;
                gx_180_add.waveform(3:4) = 0;
            elseif size(gy_180_add.waveform,1) == 8
                gy_180_add.waveform(4:5) = 0;
                gx_180_add.waveform(4:5) = 0;
            end
        
            seq.addBlock(rf_180, gz_180_add,gy_180_add,gx_180_add);
        
            % Add delay TE2
            seq.addBlock(mr.makeDelay(delayTE2));
            
            %seq.addBlock(mr.makeLabel('SET','REF', true)) ; 
            % Add EPI readout 
            gx = start_gx;
            gx = mr.scaleGrad(gx, -1);
            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
            
            start_line = start_nav - floor(Ny_calib/2) - 1 ;
            % If RSegment is > 1, we are using the regular gyPre
            if RSegment > 1
                start_line = 0;
            end
            
            if Nmulti > 1
                start_line = start_line + Nmulti -  1; 
                seq.addBlock(mr.makeDelay(ceil(TEShift * (Nmulti-1)/sys.blockDurationRaster)*sys.blockDurationRaster));
            end
            
            real_calib = [(start_nav - floor(NCalib/2) ): (start_nav + floor(NCalib/2) -1 )];
            gyPre_calib = gyPre_cell_calib{Nmulti};

            seq.addBlock(gyPre_calib, gxPre_calib );% lin/nav/avg reset
        
            % Loop over phase encoding steps
            %llin = mr.makeLabel('INC', 'LIN', 1);
            track_lines = start_line + 1;
            for i = 1:Ny_meas_calib
                lrev = mr.makeLabel('SET', 'REV', sign(gx.amplitude) == ROpolarity);
                lseg = mr.makeLabel('SET', 'SEG', sign(gx.amplitude) == ROpolarity);
                
                if i == 1
                    if ismember(track_lines, real_calib)
                        seq.addBlock(gx, gy_blipup_calib);
                    else
                        seq.addBlock(gx, gy_blipup_calib);
                    end

                elseif i == Ny_meas_calib
                    if ismember(track_lines, real_calib)
                        seq.addBlock(gx, gy_blipdown_calib);
                    else
                       seq.addBlock(gx, gy_blipdown_calib);
                    end
                else
                    
                    if ismember(track_lines, real_calib)
                        seq.addBlock(gx, gy_blipdownup_calib );
                    else
                       seq.addBlock(gx, gy_blipdownup_calib,  lrev);
                    end
                end
                
                %adc_lbl = [adc_lbl sign(gx.amplitude) == ROpolarity];
                
                % Reverse polarity of read gradient
                gx = mr.scaleGrad(gx, -1);
                
                
                %llin = mr.makeLabel('INC', 'LIN', calib_R);
                %track_lines = track_lines + calib_R; % Increment line tracking 
        
            end
            
            if RSegment > 1
                seq.addBlock(gyPost_cell_calib{Nmulti}, mr.scaleGrad(gx_post,-1));
            else
                seq.addBlock(gy_post_calib, gx_post_calib);
            end 

            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
        

            seq.addBlock(mr.makeDelay(delayTR));
            seq.addBlock(mr.makeLabel('INC', 'SLC', 1));

        end
        
        % if RSegment > 1
        %     seq.addBlock(mr.makeLabel('INC', 'REP', 1));
        % end
        
        [ktraj_adc, t_adc, ktraj, t_ktraj, ~, ~, slicepos, t_slicepos] = seq.calculateKspacePP();
        figure; plot(ktraj(1,:), ktraj(2,:), 'b'); hold on;
        plot(ktraj_adc(1,:), ktraj_adc(2,:), 'r.'); 
        title('k-space trajectory (k_x/k_y)');
        
    end
    %seq.addBlock(mr.makeLabel('SET','REF', false)) ; % QC: turn off refscan for undersampling part
end

end
%% Construct Sequence -- Calibration 
calib_count = 0;
if incl_calib > 0
    %seq.addBlock(mr.makeLabel('SET','REF', true)) ; % set up Mdh.setPATRefScan header for GRAPPA
    
    for Nmulti = 1:RSegment
        
        seq.addBlock(mr.makeLabel('SET', 'SLC', 0));
        for s = 1:Nslices
            slicePosition = slice_positions(s);
            calib_count = calib_count+1;
            
            % Add Fat Sat 
            seq.addBlock(gp_r,gp_p,gp_s);
            seq.addBlock(rf_fs, gn_r,gn_p,gn_s);
        
            % Excitation 
            rf_90.freqOffset = gz.amplitude * slicePosition;
            rf_90.phaseOffset = -2 * pi * rf_90.freqOffset * mr.calcRfCenter(rf_90);

            seq.addBlock(rf_90, gz, trig)
        
            % Add Navigator 
            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
            gx = start_gx;
            gx = mr.scaleGrad(gx, -1);
            
            start_nav = Ny/2  ;
            
            if partFourierFactor < 1
                start_nav = round(Ny * (1- partFourierFactor)) ;
            end
        
            seq.addBlock(gxPre_calib, gzReph, ...
            mr.makeLabel('SET', 'NAV', 1), ...
            mr.makeLabel('SET', 'LIN', start_nav)); % k-space center line, 0 indexing
            seq.addBlock(mr.makeLabel('SET','REF', false)) ; 
            % Reverse gxPre back
            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
            adc_lbl = [];
            for n = 1:nNav
                seq.addBlock(mr.makeLabel('SET', 'REV', sign(gx.amplitude) == ROpolarity), ...
                    mr.makeLabel('SET', 'SEG', sign(gx.amplitude) == ROpolarity), ...
                    mr.makeLabel('SET', 'AVG', n == nNav));
        
                seq.addBlock(gx, adc);
        
                adc_lbl = [adc_lbl sign(gx.amplitude) == ROpolarity];
        
                % Reverse polarity of read gradient
                gx = mr.scaleGrad(gx, -1);
        
                
            end
        
            gxPost_calib = mr.scaleGrad(gxPre_calib, -1);
            gxPost_calib.delay = 0;
            seq.addBlock(gxPost_calib,mr.makeLabel('SET', 'NAV', 1), ...
                    mr.makeLabel('SET', 'LIN', start_nav)); % k-space center line
        
            % Add Delay 
            if nav_factor == 0
                seq.addBlock(mr.makeDelay(delayTE1 - nav_dur_actual));
            else
                seq.addBlock(mr.makeDelay(delayTE1));
            end

            gz_180 = gz_180_start;
            
            rf_180 = rf_180_start;
            % Add Refocusing
            if isfield(gz_180, 'amplitude')
                rf_180.freqOffset = gz_180.amplitude * slicePosition;
            else
                amplitude = gz_180.waveform(4);
                rf_180.freqOffset = amplitude * slicePosition;
            end 

            rf_180.phaseOffset = pi/2 -2 * pi * rf_180.freqOffset * mr.calcRfCenter(rf_180);
        
            if ssgr > 0
                gz_180 = mr.scaleGrad(gz_180_start,-1);
                rf_180.freqOffset = rf_180.freqOffset * -1;
                rf_180.phaseOffset =  rf_180.phaseOffset * -1;
            end
            
            gz_180_add = gz_180;
            gy_180_add = gz_180;
            gx_180_add = gz_180;

            gy_180_add.channel = 'y';
            gx_180_add.channel = 'x';

            if size(gy_180_add.waveform,1) == 6
                gy_180_add.waveform(3:4) = 0;
                gx_180_add.waveform(3:4) = 0;
            elseif size(gy_180_add.waveform,1) == 8
                gy_180_add.waveform(4:5) = 0;
                gx_180_add.waveform(4:5) = 0;
            end
        
            seq.addBlock(rf_180, gz_180_add,gy_180_add,gx_180_add);
        
            % Add delay TE2
            seq.addBlock(mr.makeDelay(delayTE2));
            
            seq.addBlock(mr.makeLabel('SET','REF', true)) ; 
            % Add EPI readout 
            gx = start_gx;
            gx = mr.scaleGrad(gx, -1);
            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
            
            start_line = start_nav - floor(Ny_calib/2) - 1 ;
            % If RSegment is > 1, we are using the regular gyPre
            if RSegment > 1
                start_line = 0;
            end
            
            if Nmulti > 1
                start_line = start_line + Nmulti -  1; 
                seq.addBlock(mr.makeDelay(ceil(TEShift * (Nmulti-1)/sys.blockDurationRaster)*sys.blockDurationRaster));
            end
            
            real_calib = [(start_nav - floor(NCalib/2) ): (start_nav + floor(NCalib/2) -1 )];
            gyPre_calib = gyPre_cell_calib{Nmulti};

            seq.addBlock(gyPre_calib, gxPre_calib, ...
                    mr.makeLabel('SET','LIN', start_line), ... % increment LIN by -1 for "llin" performs before adc below
                    mr.makeLabel('SET','NAV', 0), ...
                    mr.makeLabel('SET','AVG', 0) );% lin/nav/avg reset
        
            % Loop over phase encoding steps
            llin = mr.makeLabel('INC', 'LIN', 1);
            track_lines = start_line + 1;
            for i = 1:Ny_meas_calib
                lrev = mr.makeLabel('SET', 'REV', sign(gx.amplitude) == ROpolarity);
                lseg = mr.makeLabel('SET', 'SEG', sign(gx.amplitude) == ROpolarity);
                
                if i == 1
                    if ismember(track_lines, real_calib)
                        seq.addBlock(gx, gy_blipup_calib, adc, lrev, lseg, llin);
                    else
                        seq.addBlock(gx, gy_blipup_calib, lrev, lseg, llin);
                    end

                elseif i == Ny_meas_calib
                    if ismember(track_lines, real_calib)
                        seq.addBlock(gx, gy_blipdown_calib, adc, lrev, lseg, llin);
                    else
                       seq.addBlock(gx, gy_blipdown_calib,  lrev, lseg, llin);
                    end
                else
                    
                    if ismember(track_lines, real_calib)
                        seq.addBlock(gx, gy_blipdownup_calib, adc, lrev, lseg, llin);
                    else
                       seq.addBlock(gx, gy_blipdownup_calib,  lrev, lseg, llin);
                    end
                end
                
                adc_lbl = [adc_lbl sign(gx.amplitude) == ROpolarity];
                
                % Reverse polarity of read gradient
                gx = mr.scaleGrad(gx, -1);
                
                llin = mr.makeLabel('INC', 'LIN', calib_R);
                track_lines = track_lines + calib_R; % Increment line tracking 
        
            end
            
            if RSegment > 1
                seq.addBlock(gyPost_cell_calib{Nmulti}, mr.scaleGrad(gx_post,-1));
            else
                seq.addBlock(gy_post_calib, gx_post_calib);
            end 

            gxPre_calib = mr.scaleGrad(gxPre_calib, -1);
        

            seq.addBlock(mr.makeDelay(delayTR));
            seq.addBlock(mr.makeLabel('INC', 'SLC', 1));

        end
        
        % if RSegment > 1
        %     seq.addBlock(mr.makeLabel('INC', 'REP', 1));
        % end
        
        [ktraj_adc, t_adc, ktraj, t_ktraj, ~, ~, slicepos, t_slicepos] = seq.calculateKspacePP();
        figure; plot(ktraj(1,:), ktraj(2,:), 'b'); hold on;
        plot(ktraj_adc(1,:), ktraj_adc(2,:), 'r.'); 
        title('k-space trajectory (k_x/k_y)');

        
        
    end
    seq.addBlock(mr.makeLabel('SET','REF', false)) ; % QC: turn off refscan for undersampling part
end




%% Construct Sequence -- Main
if nav_factor == 0
    nNav = 0;
end

if calib_only == 0 
% Iterate over reps and slices to construct sequence 
for r = 1:Nrep
    
    for d = 1:Ndir
        if gropt_mode == 0
            if isfield(gDiff, 'amplitude') 
                % gDiff.amplitude exists
                amp = gDiff.amplitude;
            else
                % Field does not exist — do something else
                amp = gDiff.waveform(3);
            end
            % Prepare Diffusion
            diff_gx = diff_dir(d,1) * amp;
            diff_gy = diff_dir(d,2) * amp;
            diff_gz = diff_dir(d,3) * amp;
        else
            amp1 =  diff_dir(d,1) ;
            amp2 =  diff_dir(d,2) ;
            amp3 =  diff_dir(d,3) ;

            diff_gx = amp1;
            diff_gy = amp2;
            diff_gz = amp3;
            gDiff = diff1;
        end

        %rise_time = gDiff.riseTime;
        %flat_time = gDiff.flatTime;
        %fall_time = rise_time;
        
        % If diffusion 1 axis or 0
        if ((sum((diff_dir(d,:)).^2)==0)||(abs(sum(diff_dir(d,:)))==1))            
            if isfield(gDiff, 'amplitude')
                rise_time = gDiff.riseTime;
                flat_time = gDiff.flatTime;
                fall_time = rise_time; 
                
                gDiff_x = mr.makeTrapezoid('x', 'amplitude',diff_gx,'riseTime',rise_time, 'flatTime',flat_time,'fallTime',fall_time,'system',sys_diff);
                gDiff_y = mr.makeTrapezoid('y', 'amplitude',diff_gy,'riseTime',rise_time, 'flatTime',flat_time,'fallTime',fall_time,'system',sys_diff);
                gDiff_z = mr.makeTrapezoid('z', 'amplitude',diff_gz,'riseTime',rise_time, 'flatTime',flat_time,'fallTime',fall_time,'system',sys_diff);
            else
                if gropt_mode == 0
                    amplitudes_x = gDiff.waveform .*  diff_dir(d,1);
                    amplitudes_y = gDiff.waveform .*  diff_dir(d,2);
                    amplitudes_z = gDiff.waveform .*  diff_dir(d,3);
    
                    gDiff_x = mr.makeExtendedTrapezoid('x', 'system',sys_diff, ...
                                 'times', gDiff.tt, ...
                                 'amplitudes', amplitudes_x) ;
                    gDiff_y = mr.makeExtendedTrapezoid('y', 'system',sys_diff, ...
                                 'times', gDiff.tt, ...
                                 'amplitudes', amplitudes_y) ;
                    gDiff_z = mr.makeExtendedTrapezoid('z', 'system',sys_diff, ...
                                 'times', gDiff.tt, ...
                                 'amplitudes', amplitudes_z) ;
                else
                    amplitudes_x1 = diff1.waveform .* diff_gx;
                    amplitudes_y1 = diff1.waveform .* diff_gy;
                    amplitudes_z1 = diff1.waveform .* diff_gz;
    
                    % gDiff_x1 = mr.makeExtendedTrapezoid('x', 'system',sys_diff, ...
                    %              'times', diff1.tt, ...
                    %              'amplitudes', amplitudes_x1) ;
                    % gDiff_y1 = mr.makeExtendedTrapezoid('y', 'system',sys_diff, ...
                    %              'times', diff1.tt, ...
                    %              'amplitudes', amplitudes_y1) ;
                    % gDiff_z1 = mr.makeExtendedTrapezoid('z', 'system',sys_diff, ...
                    %              'times', diff1.tt, ...
                    %              'amplitudes', amplitudes_z1) ;

                    gDiff_x1 = mr.makeArbitraryGrad('x',amplitudes_x1,sys,'oversampling',false,'first',0,'last',0);
                    gDiff_y1 = mr.makeArbitraryGrad('y',amplitudes_y1,sys,'oversampling',false,'first',0,'last',0);
                    gDiff_z1 = mr.makeArbitraryGrad('z',amplitudes_z1,sys,'oversampling',false,'first',0,'last',0);


                    amplitudes_x2 = diff2.waveform .* diff_gx;
                    amplitudes_y2 = diff2.waveform .* diff_gy;
                    amplitudes_z2 = diff2.waveform .* diff_gz;
    
                    % gDiff_x2 = mr.makeExtendedTrapezoid('x', 'system',sys_diff, ...
                    %              'times', diff2.tt, ...
                    %              'amplitudes', amplitudes_x2) ;
                    % gDiff_y2 = mr.makeExtendedTrapezoid('y', 'system',sys_diff, ...
                    %              'times', diff2.tt, ...
                    %              'amplitudes', amplitudes_y2) ;
                    % gDiff_z2 = mr.makeExtendedTrapezoid('z', 'system',sys_diff, ...
                    %              'times', diff2.tt, ...
                    %              'amplitudes', amplitudes_z2) ;

                    gDiff_x2 = mr.makeArbitraryGrad('x',amplitudes_x2,sys,'oversampling',false,'first',0,'last',0);
                    gDiff_y2 = mr.makeArbitraryGrad('y',amplitudes_y2,sys,'oversampling',false,'first',0,'last',0);
                    gDiff_z2 = mr.makeArbitraryGrad('z',amplitudes_z2,sys,'oversampling',false,'first',0,'last',0);

                end
            end
        % If diffusion off-axis, rotate
        else

            % Rotation matrix given gradient is designed on Z and
            % rotated to diff_dir
            dir0 = diff_dir(d,:)';                      % 3×1
            dir_unit = dir0 / norm(dir0);               % normalize
            dir_norm = 1; % norm(dir0);                     % magnitude

            % Compute rotation matrix from [0 0 1] to unit direction
            Rot_Matrix = vectorToRotationMatrixFromTo([0,0,1], dir_unit);
            
        
            if gropt_mode == 0
              
                % Rotate gDiff (assumes it's z-directed)
                Gr = mr.rotate3D(Rot_Matrix, gDiff);
    
                % Assign diffusion gradients in x, y, z
                gDiff_x=Gr{1};
                gDiff_y=Gr{2};
                gDiff_z=Gr{3};
                
                if isfield(gDiff, 'amplitude')
                    gDiff_x = mr.makeTrapezoid('x','amplitude',gDiff_x.amplitude * dir_norm,'riseTime',gDiff_x.riseTime, 'flatTime',gDiff_x.flatTime,'fallTime',gDiff_x.fallTime,'system',sys_diff);
                    gDiff_y = mr.makeTrapezoid('y','amplitude',gDiff_y.amplitude * dir_norm,'riseTime',gDiff_y.riseTime, 'flatTime',gDiff_y.flatTime,'fallTime',gDiff_y.fallTime,'system',sys_diff);
                    gDiff_z = mr.makeTrapezoid('z','amplitude',gDiff_z.amplitude * dir_norm,'riseTime',gDiff_z.riseTime, 'flatTime',gDiff_z.flatTime,'fallTime',gDiff_z.fallTime,'system',sys_diff);
                    
                    Gr{1}=gDiff_x;
                    Gr{2}=gDiff_y;
                    Gr{3}=gDiff_z;
                end
                
                if mmt == 0
                    bval_rot = pseq2.Monopolar.compute_bval_rotations(Gr, delayTE1, delayTE2, ...
                        rf_90_duration, rf_90_rfCenterInclDelay, rf_180_duration, sys);
                elseif mmt == 1
                   bval_rot = pseq2.Bipolar.compute_bval_rotations(Gr, delayTE1, delayTE2, ...
                    rf_90_duration, rf_90_rfCenterInclDelay, rf_180_duration, sys);
    
                end
            else

                % Rotate gDiff (assumes it's z-directed)
                Gr_1 = mr.rotate3D(Rot_Matrix, diff1);
    
                % Assign diffusion gradients in x, y, z
                gDiff_x1=Gr_1{1};
                gDiff_y1=Gr_1{2};
                gDiff_z1=Gr_1{3};

                % Rotate gDiff (assumes it's z-directed)
                Gr_2 = mr.rotate3D(Rot_Matrix, diff2);
    
                % Assign diffusion gradients in x, y, z
                gDiff_x2=Gr_2{1};
                gDiff_y2=Gr_2{2};
                gDiff_z2=Gr_2{3};


            end


        end

        seq.addBlock(mr.makeLabel('SET', 'SLC', 0));

        for s = 1:Nslices

            slicePosition = slice_positions(s);

            % Add Fat Sat 
            seq.addBlock(gp_r,gp_p,gp_s);
            seq.addBlock(rf_fs, gn_r,gn_p,gn_s);

            % Excitation 
            rf_90.freqOffset = gz.amplitude * slicePosition;
            rf_90.phaseOffset = -2 * pi * rf_90.freqOffset * mr.calcRfCenter(rf_90);
            seq.addBlock(rf_90, gz, trig)

           
            if partFourierFactor < 1
                start_nav = round(Ny * (1- partFourierFactor)) ;
            end
    
            % Add Navigator 
            if nNav > 0
                gxPre = mr.scaleGrad(gxPre, -1);
                gx = start_gx;
                gx = mr.scaleGrad(gx, -1);
                start_nav = Ny/2  ;
                seq.addBlock(gxPre, gzReph, ...
                mr.makeLabel('SET', 'NAV', 1), ...
                mr.makeLabel('SET', 'LIN', start_nav)); % k-space center line, 0 indexing
    
                % Reverse gxPre back
                gxPre = mr.scaleGrad(gxPre, -1);
                adc_lbl = [];
                for n = 1:nNav
                    seq.addBlock(mr.makeLabel('SET', 'REV', sign(gx.amplitude) == ROpolarity), ...
                        mr.makeLabel('SET', 'SEG', sign(gx.amplitude) == ROpolarity), ...
                        mr.makeLabel('SET', 'AVG', n == nNav));
    
                    seq.addBlock(gx, adc);
    
                    adc_lbl = [adc_lbl sign(gx.amplitude) == ROpolarity];
    
                    % Reverse polarity of read gradient
                    gx = mr.scaleGrad(gx, -1);
    
    
                end

                gxPost = mr.scaleGrad(gxPre, -1);
                gxPost.delay = 0;
                seq.addBlock(gxPost,mr.makeLabel('SET', 'NAV', 1), ...
                        mr.makeLabel('SET', 'LIN', start_nav)); % k-space center line
            else
                seq.addBlock(gzReph, ...
                mr.makeLabel('SET', 'NAV', 0), ...
                mr.makeLabel('SET', 'LIN', start_nav)); % k-space center line, 0 indexing
            end
            
            % Add Delay 
            
            if gropt_mode == 0
                seq.addBlock(mr.makeDelay(delayTE1),gDiff_x,gDiff_y,gDiff_z  );
            else
                seq.addBlock(mr.makeDelay(delayTE1),gDiff_x1,gDiff_y1,gDiff_z1);
            end




           
            % Add Refocusing
            gz_180 = gz_180_start;
            rf_180 = rf_180_start;
            
            if isfield(gz_180, 'amplitude')
                rf_180.freqOffset = gz_180.amplitude * slicePosition;
            else
                amplitude = gz_180.waveform(4);
                rf_180.freqOffset = amplitude * slicePosition;
            end 
            
            rf_180.phaseOffset = pi/2 -2 * pi * rf_180.freqOffset * mr.calcRfCenter(rf_180);

            if ssgr > 0
                gz_180 = mr.scaleGrad(gz_180_start,-1);
                rf_180.freqOffset = rf_180.freqOffset * -1;
                rf_180.phaseOffset =  rf_180.phaseOffset * -1;
            end

            gz_180_add = gz_180;
            gy_180_add = gz_180;
            gx_180_add = gz_180;

            gy_180_add.channel = 'y';
            gx_180_add.channel = 'x';
            
            if size(gy_180_add.waveform,1) == 6
                gy_180_add.waveform(3:4) = 0;
                gx_180_add.waveform(3:4) = 0;
            elseif size(gy_180_add.waveform,1) == 8
                gy_180_add.waveform(4:5) = 0;
                gx_180_add.waveform(4:5) = 0;
            end



            if sum(abs(diff_dir(d,:))) > 0
                % Zero the spoilers
                % gz_180_add.waveform(1:2) = 0;
                % gz_180_add.waveform(5:6) = 0;
                % 
                % gy_180_add.waveform(1:2) = 0;
                % gy_180_add.waveform(5:6) = 0;
                % 
                % gx_180_add.waveform(1:2) = 0;
                % gx_180_add.waveform(5:6) = 0;

            end

            seq.addBlock(rf_180, gz_180_add,gy_180_add,gx_180_add);

            % Add delay TE2
            if gropt_mode == 0
                seq.addBlock(mr.makeDelay(delayTE2),gDiff_x,gDiff_y,gDiff_z);
            else
                seq.addBlock(mr.makeDelay(delayTE2),gDiff_x2,gDiff_y2,gDiff_z2);
            end


            % Add EPI readout 
            gx = start_gx;
            gx = mr.scaleGrad(gx, -1);
            gxPre = mr.scaleGrad(gxPre, -1);
            start_line = -1;
            if R > 1
                start_line = 0;
            end
            seq.addBlock(gyPre, gxPre, ...
                    mr.makeLabel('SET','LIN', start_line), ... % increment LIN by -1 for "llin" performs before adc below
                    mr.makeLabel('SET','NAV', 0), ...
                    mr.makeLabel('SET','AVG', 0) );% lin/nav/avg reset

            % Loop over phase encoding steps
            llin = mr.makeLabel('INC', 'LIN', 1);
            for i = 1:Ny_meas
                lrev = mr.makeLabel('SET', 'REV', sign(gx.amplitude) == ROpolarity);
                lseg = mr.makeLabel('SET', 'SEG', sign(gx.amplitude) == ROpolarity);

                if i == 1
                    seq.addBlock(gx, gy_blipup, adc, lrev, lseg, llin);
                elseif i == Ny_meas
                    seq.addBlock(gx, gy_blipdown, adc, lrev, lseg, llin);
                else
                    seq.addBlock(gx, gy_blipdownup, adc, lrev, lseg, llin);
                end

                adc_lbl = [adc_lbl sign(gx.amplitude) == ROpolarity];

                % Reverse polarity of read gradient
                gx = mr.scaleGrad(gx, -1);

                llin = mr.makeLabel('INC', 'LIN', R);

            end
            seq.addBlock(mr.scaleGrad(gx_post,-1),gy_post);
            gxPre = mr.scaleGrad(gxPre, -1);
            seq.addBlock(mr.makeDelay(delayTR));
            seq.addBlock(mr.makeLabel('INC', 'SLC', 1));
        end
        seq.addBlock(mr.makeLabel('INC', 'REP', 1));


    end
end
end

%% check whether the timing of the sequence is correct
[ok, error_report]=seq.checkTiming;

if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end



%% Write Sequence
if partFourierFactor < 1
    center  = round(Ny * ( 1 - partFourierFactor));
else
    center = Ny/2;
end

phaseResolution =Ny_meas/Ny *R;
seq.setDefinition('Name', 'epi');
seq.setDefinition('FOV', [fov fov max(slice_positions)-min(slice_positions)+sliceThickness]);
seq.setDefinition('SlicePositions', slice_positions);
seq.setDefinition('SliceThickness', sliceThickness);
seq.setDefinition('SliceGap', slice_gap);
% seq.setDefinition('PhaseResolution', phaseResolution ) ;
seq.setDefinition('PhaseResolution', 1 ) ;
seq.setDefinition('ReceiverGainHigh', 1);
seq.setDefinition('kSpaceCenterLine', center ) ;
seq.setDefinition('ReadoutOversamplingFactor', ro_os);
seq.setDefinition('TargetGriddedSamples', Nx*ro_os);
seq.setDefinition('TrapezoidGriddingParameters', ...
    [gx.riseTime gx.flatTime gx.fallTime adc.delay-gx.delay adc.duration]);
seq.setDefinition('AccelerationFactorPE', 2) ;
% seq.setDefinition('AccelerationFactor3D', 1) ;
seq.setDefinition('FirstRefLine', 22) ;
seq.setDefinition('FirstFourierLine', 1) ;
seq.setDefinition('PhaseEncodingLine_DIFF', Nx) ;
seq.setDefinition( 'RefLinesPE', 30) ;
seq.write(seq_name);

return ;

[ktraj_adc, t_adc, ktraj, t_ktraj, ~, ~, slicepos, t_slicepos] = seq.calculateKspacePP();
figure; plot(ktraj(1,:), ktraj(2,:), 'b'); hold on;
plot(ktraj_adc(1,:), ktraj_adc(2,:), 'r.'); 
title('k-space trajectory (k_x/k_y)');

[wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc]  =  seq.waveforms_and_times(true);
%save(seq_name_waveform, 'wave_data');

[g3_segment_cell, M0_cell, M0_TE, all_pts] = pseq2.get_moment_check(  wave_data, tfp_excitation, tfp_refocusing, t_adc, ...
    fp_adc, adc.numSamples, delayTR, 4, true);







pns_result = seq.calcPNS(ascName)




%% Visualization and Safety Checks
%pns_result = seq.calcPNS(ascName);
fprintf('PNS check is %s.\n', ifelse(all(pns_result == 1), 'OK', 'FAILED'));



% seq.plot('stacked',1);
% labels =seq.evalLabels('evolution','adc') ; % exact ADC labels

% [ktraj_adc, t_adc, ktraj, t_ktraj, ~, ~, slicepos, t_slicepos] = seq.calculateKspacePP();
% figure; plot(ktraj(1,:), ktraj(2,:), 'b'); hold on;
% plot(ktraj_adc(1,:), ktraj_adc(2,:), 'r.'); 
% title('k-space trajectory (k_x/k_y)');
% 
% % plot label evolution
% figure ; hold on ;
% plot(labels.LIN) ;
% plot(labels.AVG) ;
% plot(labels.SEG) ;
% plot(labels.REP) ;
% %plot(labels.REF) ;
% %plot(labels.NOISE) ;
% legend('LIN', 'AVG', 'SEG', 'REP', 'REF', 'NOISE') ;




%% Moment and Frequency Checks 
% Analyze Frequency Spectrum
%pseq2.analyzeGradientSpectrum(seq, sys, ascName)

% Assess that zero moment is sustained
[wave_data, tfp_excitation, tfp_refocusing, t_adc, fp_adc]  =  seq.waveforms_and_times(true);
%save(seq_name_waveform, 'wave_data');

[g3_segment_cell, M0_cell, M0_TE, all_pts] = pseq2.get_moment_check(  wave_data, tfp_excitation, tfp_refocusing, t_adc, ...
    fp_adc, adc.numSamples, delayTR, 2, true);
[g3_segment_cell, M0_cell, M0_TE, all_pts] = pseq2.get_moment_check(  wave_data, tfp_excitation, tfp_refocusing, t_adc, ...
     fp_adc, adc.numSamples, delayTR, 5, true);






%% Helper Function 

% Converting diffusion vector to a rotation matrix
function R = vectorToRotationMatrixFromTo(v1, v2)
    v1 = v1(:) / norm(v1);
    v2 = v2(:) / norm(v2);
    v = cross(v1, v2);
    s = norm(v);
    c = dot(v1, v2);
    if s == 0
        if c > 0
            R = eye(3);
        else
            % 180° rotation: pick an orthogonal axis
            orth = null(v1.');
            axis = orth(:,1);
            vx = [   0    -axis(3)  axis(2);
                   axis(3)   0    -axis(1);
                  -axis(2)  axis(1)   0 ];
            R = eye(3) + 2*vx^2;  % rotation by 180°
        end
    else
        vx = [   0   -v(3)  v(2);
               v(3)   0   -v(1);
              -v(2)  v(1)   0 ];
        R = eye(3) + vx + vx^2 * ((1 - c) / s^2);
    end
end


% boolean logic print statements
function out = ifelse(cond, valTrue, valFalse)
    if cond
        out = valTrue;
    else
        out = valFalse;
    end
end