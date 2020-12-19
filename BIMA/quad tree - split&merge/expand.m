function J = expand(I)
m = max(size(I));
a = ceil(log2(m));
J = ones(2^a,2^a)*floor(mean2(I));
J(1:size(I,1),1:size(I,2)) = I;
end
