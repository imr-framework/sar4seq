function Q= Q_mat_gen(SAR_type,model,qmat_write)

%% Read the required data in
% Author: Sairam Geethanath, Ph.D.
% 01/2012
% phantom data has e fields files named as_Ex while the VHM has _ex
addpath(genpath('.'));
close all;


%% Load required EM model data
Ex = model.Ex;
Ey= model.Ey;
Ez = model.Ez;
Tissue_types = model.Tissue_types;
SigmabyRhox = model.SigmabyRhox;
Mass_cell = model.Mass_cell;

% if(parpool('local') == 0)
%   parpool local 8;
% end

%%
t0 = cputime;
switch SAR_type 

    case 'Global'
            clc;disp('SAR type: GLOBAL');
           
            %% Read from df files and check local computation, using index - whole body
            clc;disp('Q - Whole body calculation started ....');
            [Qavg_df,Tissue_types,SbRx,Mass_cell,Mass_body]  = gen_Qpwr(Ex,Ey,Ez,Tissue_types,SigmabyRhox, Mass_cell,'global','wholebody');%SbR stands for SigmabyRho
            Qavg_tm = Qavg_df./Mass_body;
            figure(1);imagesc(abs((Qavg_tm)));colorbar;title('Implemented - Mass normalized BODY');

            %% Head
            disp('Q -Head calculation started ....');
            [Qavg_df,~,~,~,Mass_head] = gen_Qpwr(Ex,Ey,Ez,Tissue_types,SigmabyRhox, Mass_cell,'global','head');%SbR stands for SigmabyRho
            Qavg_hm = Qavg_df./Mass_head;
            figure(2);imagesc(abs(Qavg_hm));colorbar;title('Implemented - Mass normalized HEAD');
            
             %% Torso
%             disp('Q - torso calculation started ....');
%             [Qavg_df,~,~,~,Mass_torso] = gen_Q(dirname,'global','torso');%SbR stands for SigmabyRho
%             Qavg_tom = Qavg_df./Mass_torso;
%             figure(2);imagesc(abs(Qavg_tom));colorbar;title('Implemented - Mass normalized TORSO');
            
            
            %% Make structure for writing it to a file
            Q.Qtmf = Qavg_tm;
            Q.Qhmf = Qavg_hm;
%             Q.Qemf = Qavg_tom; %TODO have not implemented this at all, need to figure out
            if (qmat_write)
                 [status] = write_qmat(Q,'Global');
            end
    
    case 'Local'
            disp('SAR type: LOCAL');
            [Qavg] = read_qmat('avgQMatrix');
            %%
            S = squeeze(Qavg.index(1:3,:));
            
            x =find(S(1,:)== 60);
            y =find(S(2,:)==49);
            z =find(S(3,:)==42);
            
            s = intersect(x,y);
            s = intersect(s,z);
            
            Qpt = squeeze(Qavg.avg(s,:,:));
            figure;imagesc(abs(Qpt));

            dirname = uigetdir('');   
            disp('Starting calculation of local Q matrices.....');
            t0_local = cputime;
                 [Qavg_df,Tissue_types,SbRx,Mass_cell,Mass_body] = gen_Qpwr(dirname,'local','wholebody');%SbR stands for SigmabyRho
            t1_local = cputime - t0_local;
            clc;disp(['Done calculating local Q-matrices in ',num2str(t1_local),' seconds']);
           
            Qpt_design = squeeze(Qavg_df(61,50,42,:,:));
            figure;imagesc(abs((Qpt_design)));
            figure;imagesc(abs(Qpt)./abs((Qpt_design)));
            figure;imagesc(abs(abs(Qpt) -abs((Qpt_design))));
            %%
            [RMSE_map, Qtri_map, Qimp_map,NRMSE_map] = get_RMSE(Qavg,Qavg_df,Mass_cell);
            
            %% Global Display of local Q
                figure;imshow(squeeze(Qtri_map(:,:,193)),'InitialMagnification',200);colorbar;colormap(jet);caxis([0 2]);
                figure;imshow(squeeze(Qimp_map(:,:,193)),'InitialMagnification',200);colorbar;colormap(jet);caxis([0 2]);
                diff_map = abs(Qtri_map - Qimp_map);
                figure;imshow(squeeze(diff_map(:,:,193)),'InitialMagnification',200);colorbar;colormap(jet);caxis([0 0.1])

                %% Save
                Qavg.imp= Qavg_df;
                if(qmat_write)
                    save('LocalQ','Qavg','-v7.3');
                end


end
% parpool close;
t1 = cputime - t0;
disp(['Q matrices calculated in ',num2str(t1),' seconds']);



