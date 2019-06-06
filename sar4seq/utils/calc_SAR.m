function [SAR] = calc_SAR(Q,I)
%% I has Nc rows and Nt columns
% Ifact = zeros(size(I,1));

% for nc = 1:size(I,1)
%     for nc2 = 1:size(I,1)
%         Ifact(nc, nc2) = I(nc,:)'.*I(nc2,:)./size(I,2);
%     end
% end

if(size(Q,1) > 1)
    I = repmat(I.', [size(Q,1), 1]);
end
Ifact = (I*I')./size(I,2);
if(ndims(Q) > 2)
        for k=1:size(Q,1)
              Qtemp = squeeze(Q(k,:,:));
              SAR_temp(k) = Qtemp.*Ifact;
        end
        SAR = max(abs(SAR_temp));
        SAR = abs(sum(SAR(:)));
else
       SAR_temp = Q.*Ifact;
       SAR = abs(sum(SAR_temp(:)));
end