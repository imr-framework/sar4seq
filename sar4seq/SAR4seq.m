%% File details


%% Paths 
addpath(genpath('/Users/sairamgeethanath/Documents/Columbia/Github-SG/pulseq-master/'));
addpath(genpath('.'));
%% SAR limits
SixMinThresh_wbg =4; %W/Kg
TenSecThresh_wbg = 8;

SixMinThresh_hg =3.2;%W/Kg
TenSecThresh_hg = 6.4;
%% Load relevant model - ONLY once per test
dirname = uigetdir(''); %Load dir with files for EM model
cdir = pwd;
cd(dirname)
load Ex.mat; model.Ex =Ex; clear Ex; 
load Ey.mat; model.Ey =Ey; clear Ey;  
load Ez.mat; model.Ez =Ez; clear Ez;  
load Tissue_types.mat; model.Tissue_types =Tissue_types; clear Tissue_types;  
load SigmabyRhox.mat; model.SigmabyRhox =SigmabyRhox; clear SigmabyRhox;  
load Mass_cell.mat; model.Mass_cell =Mass_cell; clear Mass_cell; 
cd(cdir)

%% Compute and store Q matrices once- if not done already - per model - will write relevant qmat files
tic;
Q = Q_mat_gen('Global', model,1);
toc;
%% Import a seq file to compute SAR for
system = mr.opts('MaxGrad', 32, 'GradUnit', 'mT/m', ...
    'MaxSlew', 130, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 30e-6, ...
    'rfDeadTime', 100e-6);
seq=mr.Sequence(system);
seq.read('tse.seq'); %Import the seq file you want to check; alternately you can perform the check in the sequence code.

%% Identify RF blocks and compute SAR - 10 seconds must be less than twice and 6 minutes must be less than 4 (WB) and 3.2 (head-20)
obj = seq;
t_vec = zeros(1, length(obj.blockEvents));
SARwbg_vec = zeros(size(t_vec));
SARhg_vec =SARwbg_vec;
t_prev=0;

  for iB=1:length(obj.blockEvents)
                block = obj.getBlock(iB);
                ev={block.rf, block.gx, block.gy, block.gz, block.adc, block.delay};
                ind=~cellfun(@isempty,ev);
                block_dur=mr.calcDuration(ev{ind});
                t_vec(iB) = t_prev + block_dur;
                t_prev = t_vec(iB);
                
                     if ~isempty(block.rf)
                                rf=block.rf;
                                t=rf.t; signal = rf.signal;
                               %Calculate SAR - Global: Wholebody, Head, Exposed Mass
                               SARwbg_vec(iB) = calc_SAR(Q.Qtmf,signal); %This rf could be parallel transmit as well
                               SARhg_vec(iB) = calc_SAR(Q.Qhmf,signal); %This rf could be parallel transmit as well
%                                if((SARwbg_vec(iB)  > 4) || (SAR_head > 3.2))
%                                    error('Pulse exceeding Global SAR limits');
%                                end                       
                       %% Incorporate time averaged SAR
                     end
  end
  
 
 %% Interpolate SAR - Pictorial representation only 
 tsec = 1:t_vec(end);
[SARwbg_lim_s] = interp1(t_vec, SARwbg_vec, tsec,'spline'); %< 2 SARmax
[SARhg_lim_s] = interp1(t_vec, SARhg_vec, tsec,'spline'); %< 2 SARmax

figure(101); plot(t_vec, SARwbg_vec); hold on;
SARwbg_lim_s(SARwbg_lim_s < 0) = 0;
SARhg_lim_s(SARhg_lim_s < 0) = 0;

plot(tsec, SARwbg_lim_s); hold on;
plot(tsec, SARhg_lim_s); hold on;
legend('Whole body', 'Head only');
%% Calculate time averaged SAR 1 -N, better to do -N/2 to N/2
 SAR_wbg_tensec = do_sw_sar(SARwbg_lim_s,tsec, 10);%< 2 SARmax
 SAR_wbg_sixmin = do_sw_sar(SARwbg_lim_s,tsec, 600);
 
  SAR_hg_tensec = do_sw_sar(SARhg_lim_s,tsec, 10);%< 2 SARmax
 SAR_hg_sixmin = do_sw_sar(SARhg_lim_s,tsec, 600);
 
plot(tsec, SAR_wbg_tensec); hold on;
plot(tsec, SAR_hg_tensec); hold on;
legend('Whole body', 'Head only');

 %% Check for each instant of the time averaged SAR with appropriate time limits
 if((SAR_wbg_tensec > TenSecThresh_wbg)|| (SAR_hg_tensec > TenSecThresh_hg))
         error('Pulse exceeding 10 second Global SAR limits, increase TR');
 end
 
  if((SAR_hg_sixmin > SixMinThresh_wbg)|| (SAR_hg_sixmin > SixMinThresh_hg))
         error('Pulse exceeding 10 second Global SAR limits, increase TR');
 end
 
 
 



