function D = calculMatDist(Xc_train,Xc_test,W,K)
n = size(Xc_test,2);
m = size(Xc_train,2);
x_moy = moyen(Xc_train);

%Matrix of projected test face
M_test = zeros(K,n);
for i = 1:n
    x_test = Xc_test(:,i);
    z_test = calculeProj(x_test,x_moy,K,W);
    M_test(:,i) = z_test;
end
%Matrix of projected train face
M_train = zeros(K,m);
for i = 1:m
    x_train = Xc_train(:,i);    
    z_train = calculeProj(x_train,x_moy,K,W);
    M_train(:,i) = z_train;
end

%Matrix of distance
D = zeros(n,m);
for i = 1:n
    for j = 1:m
        D(i,j) = sqrt(sum((M_test(:,i) - M_train(:,j)).^2));
    end
end


