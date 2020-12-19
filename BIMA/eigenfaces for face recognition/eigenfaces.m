function [U,lambdas] = eigenfaces(Xc)
[U,s,v] = svd(Xc,0);
lambdas = diag(s).^2;
end