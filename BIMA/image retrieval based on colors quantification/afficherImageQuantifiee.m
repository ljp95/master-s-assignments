function afficherImageQuantifiee(Iq,nH,nS,nV,palette)
[n,m] = size(Iq(:,:,1));
Iq2 = zeros(size(Iq));

for i=1:n
    for j=1:m
        % assign the right color from the palette 
        % it depends on the indices in Iq in each dimension for each pixel
        Iq2(i,j,:) = palette(Iq(i,j,1) + (Iq(i,j,2)-1)*nH + (Iq(i,j,3)-1)*nH*nS,:);
    end
end
figure();imagesc(Iq2);
end


