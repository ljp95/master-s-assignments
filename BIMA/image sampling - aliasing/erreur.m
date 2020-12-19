function eps = erreur(xr, xd, A)
eps = (1/(2*A*size(xr,1)^2))*sum(sum(abs(xr-xd)));
end
