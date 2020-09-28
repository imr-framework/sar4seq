%% File details
%
%     1. Computes RF safety metrics for Pulseq sequences 
%     a. For Pulseq sequences for deployment on Siemens scanners - 
%     computes time averaged RF power for the sequence
%     b. For Pulseq sequences for deployment on GE scanners (via TOPPE) -
%     computes the whole body SAR in W/kg
%     
% 
%    Parameters
%    ----------
%       seq_path : Path to Pulseq sequence file - string
%       seq : Pulseq sequence object determining system parameters - seq
%       object
%       Sample_weight : weight of the sample being imaged - double
% 
%     Returns
%     -------
%       Time averaged RF power : double
%       Whole body SAR : double
%            
%       
% 
% Copyright of the Board of Trustees of Columbia University in the City of New York

function [RFwbg_tavg,RFhg_tavg,SARwbg_pred] = SAR4seq(seq_path,seq,Sample_weight)

%% Paths 
addpath(genpath('/Users/sairamgeethanath/Documents/Columbia/Github-SG/pulseq-master/'));
addpath(genpath('.'));


if(nargin < 1)
    seq_path = './seq_files/180_tse500ms.seq';
    system = mr.opts('MaxGrad', 32, 'GradUnit', 'mT/m', ...
    'MaxSlew', 130, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 30e-6, ...
    'rfDeadTime', 100e-6);
    seq=mr.Sequence(system);
    Sample_weight = 40;% kg
elseif(nargin <2)
    system = mr.opts('MaxGrad', 32, 'GradUnit', 'mT/m', ...
    'MaxSlew', 130, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 30e-6, ...
    'rfDeadTime', 100e-6);
    seq=mr.Sequence(system);
    Sample_weight = 10;% kg
elseif(nargin <3)
    Sample_weight = 10;% kg
end


%% Constants
SiemensB1fact = 1.32;%need to explore this further - B1+ factor
GEB1fact = 1.1725;% need to explore this further - B1+ factor

Wbody_weight = 103.45; %kg - from Visible Human Male
Head_weight = 6.024;  %kg - from Visible Human Male

%% SAR limits
SixMinThresh_wbg =4; %W/Kg
TenSecThresh_wbg = 8;

SixMinThresh_hg =3.2;%W/Kg
TenSecThresh_hg = 6.4;

if(~(exist('Qmat.mat','file')))
    %% Load relevant model - ONLY once per site if somebody needs to explore the Q matrix formulation
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
    Q = Q_mat_gen('Global', model,0);
    save('Qmat.mat','Q');
    toc;
else
    load('Qmat.mat','Q'); %Loads Q
end

%% Import a seq file to compute SAR for
seq.read(seq_path); %Import the seq file you want to check; alternately you can perform the check in the sequence code.

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
                               SARwbg_vec(iB) = calc_SAR(Q.Qtmf,signal,Wbody_weight); %This rf could be parallel transmit as well
                               SARhg_vec(iB) = calc_SAR(Q.Qhmf,signal,Head_weight); %This rf could be parallel transmit as well
%                                if((SARwbg_vec(iB)  > 4) || (SAR_head > 3.2))
%                                    error('Pulse exceeding Global SAR limits');
%                                end                       
                       %% Incorporate time averaged SAR
                     end
  end
  
%% Filter out zeros in iB
T_scan = t_vec(end); %find a better way to get to end of scan time
idx = find(abs(SARwbg_vec) > 0);
SARwbg_vec = squeeze(SARwbg_vec(idx));
SARhg_vec = squeeze(SARhg_vec(idx));
  
  
%% Time averaged RF power - match Siemens data
RFwbg_tavg = sum(SARwbg_vec)./T_scan./SiemensB1fact;
RFhg_tavg = sum(SARhg_vec)./T_scan./SiemensB1fact;
disp(['Time averaged RF power-Siemens is - Body: ', num2str(RFwbg_tavg),'W &  Head: ', num2str(RFhg_tavg), 'W']);

SARwbg = max(SARwbg_vec); %W/kg but Siemens reports W/lb
SARhg = max(SARhg_vec);

Sample_head_weight = (Head_weight/Wbody_weight)*Sample_weight;

SARwbg_predSiemens = SARwbg.* sqrt(Wbody_weight/Sample_weight)/2;
SARhg_predSiemens = SARhg.* sqrt(Head_weight/Sample_head_weight)/2;


% SARwbg_predSiemens = SARwbg_predSiemens/2.20462; %Siemens reports W/lb
disp(['Predicted SAR-Siemens is - Body:', num2str(SARwbg_predSiemens), 'W/kg &  Head: ',num2str(SARhg_predSiemens), 'W/kg'])
%% SAR whole body - match GE data
SARwbg_predGE = SARwbg.* sqrt(Wbody_weight/Sample_weight).*GEB1fact;
disp(['Predicted SAR-GE is ', num2str(SARwbg_predGE), 'W/kg'])

 %% Check for each instant of the time averaged SAR with appropriate time limits
 if(sum(SARwbg_predGE > TenSecThresh_wbg))
         error('Pulse sequence exceeding 10 second Global SAR limits, increase TR');
 end





