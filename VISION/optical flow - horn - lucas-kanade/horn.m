function [u,v] = horn(I1,I2,alpha,N)
[n,m] = size(I1);
[Ix,Iy,It] = gradhorn(I1,I2);
A = [1 2 1; 2 0 2; 1 2 1]/12;
u = zeros(n,m); v = zeros(n,m);
for i=1:N
    u = conv2(u,A,'same');
    v = conv2(v,A,'same');
    tmp = (Ix.*u + Iy.*v + It)./(alpha + Ix.^2 + Iy.^2);
    u = u - Ix.*tmp;
    v = v - Iy.*tmp;
end
end