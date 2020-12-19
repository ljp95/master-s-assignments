function J = normaliseImage(I,k1,k2)
m = min(min(I));
n = max(max(I));
J = round((I-m)*double((k2-k1)/(n-m))+k1);