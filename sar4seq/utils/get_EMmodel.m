function [Ex, Ey, Ez, Tissue_types,SigmabyRhox,Mass_cell] = get_EMmodel(dirname)

%% Currently for ICL data but this file can be replaced to read mat files


disp('Reading deadface files...');
t0_df = cputime;
% Air_seg = df_read(fullfile(dirname,'airSeg.df')); %Tissue density
Tissue_types= df_read(fullfile(dirname,'BodySegmentation.df')); %Tissue density
S = df_read(fullfile(dirname,'material.df')); 
% D = size(Tissue_types);
SigmabyRhox = real(S);
Mass_cell =imag(S);
% Mass_corr=0;
% Mass_air = 1.625e-7 + 1e-9;
clear S S2;

Ex = zeros([size(SigmabyRhox),8]); %dir, vector, tx channels 
Ey=Ex;
Ez=Ex;



%% Lot of hardcoding for now, make it better later

parfor k=1:8 %num_channels
    Ex(:,:,:,k) = df_read(fullfile(dirname,['multix_coil',num2str(k),'_ex.df'])); 
    Ey(:,:,:,k) = df_read(fullfile(dirname,['multix_coil',num2str(k),'_ey.df'])); 
    Ez(:,:,:,k) = df_read(fullfile(dirname,['multix_coil',num2str(k),'_ez.df'])); 
end

clear k  dirname;
fclose('all');
t1_df = cputime - t0_df;
disp(['Deadface files read in ',num2str(t1_df),' seconds']);
