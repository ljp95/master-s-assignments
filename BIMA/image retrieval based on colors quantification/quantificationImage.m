function [Iq,histo] = quantificationImage(I,nH,nS,nV)
[n,m] = size(I(:,:,1));
B = zeros(n,m);
Iq = cat(3,B,B,B);

for i=1:n
    for j=1:m
        Iq(i,j,1) = quantification(I(i,j,1),nH);
        Iq(i,j,2) = quantification(I(i,j,2),nS);
        Iq(i,j,3) = quantification(I(i,j,3),nV);
    end
end

histo = zeros(nH,nS,nV);
for i=1:n
    for j=1:m
        histo(Iq(i,j,1),Iq(i,j,2),Iq(i,j,3)) = histo(Iq(i,j,1),Iq(i,j,2),Iq(i,j,3))+1;
    end
end
end