function  SAR_timeavg = do_sw_sar(SARwbg_lim_s, tsec,t)

SAR_timeavg = zeros(1, length(tsec));
for instant=1:(tsec(end) - t) %better to go from -sw/2:sw/2
    SAR_timeavg(instant) = sum(SARwbg_lim_s(instant:instant+t-1))./t;
end

