function z = calculeProj(x,x_moy,K,W)
Wk = W(:,1:K);
z = transpose(Wk) * (x - x_moy);
end
