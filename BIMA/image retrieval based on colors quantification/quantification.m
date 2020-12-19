function iv = quantification(v,K)
if v==1
    iv = K;
end
if v<1
    iv  = floor(v*K) +1;
end
end
