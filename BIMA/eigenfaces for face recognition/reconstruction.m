function x_r = reconstruction(z,x_moy,W,K)
Wk = W(:,1:K);
x_r = x_moy + Wk * z;
end