function [SAR] = calc_SAR(Q,I,weight)
%% I has Nc rows and Nt columns
% Ifact = zeros(size(I,1));

% for nc = 1:size(I,1)
%     for nc2 = 1:size(I,1)
%         Ifact(nc, nc2) = I(nc,:)'.*I(nc2,:)./size(I,2);
%     end
% end

Iexp = conj(I).*I; %this is assuming only 1 channel at the moment
Iexp = sum(Iexp(:))./length(Iexp); 
Ifact = Iexp;

% if(size(Q,1) > 1)
%     ph = 0:2*pi/size(Q,1):(size(Q,1)-1)*(pi/4);
%     tx_ph = exp(1i*ph).';
%     tx_ph = repmat(tx_ph, [1 size(I,1)]);
%     I = repmat((I).', [size(Q,1), 1]);
%     I = I.*tx_ph;
% end



if(ndims(Q) > 2)
        SAR_temp = zeros(size(Q));
        SAR_norm = zeros(size(Q,1));
        for k=1:size(Q,1)
              Qtemp = squeeze(Q(k,:,:));
              SAR_temp(k,:,:) = Qtemp.*Ifact;
              SAR_norm(k) = norm(squeeze(SAR_temp(k,:,:)));
        end
        [~,ind] = max(SAR_norm);
        SAR_chosen = SAR_temp(ind,:,:);
        SAR = abs(sum(SAR_chosen(:)));
else
       SAR_temp = Q.*Ifact;
       SAR = abs(sum(SAR_temp(:)));
       SAR = SAR./weight;
end