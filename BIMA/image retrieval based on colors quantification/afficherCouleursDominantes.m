function afficherCouleursDominantes(histo,palette,k)
% Sort by max
[valeurs,indices] = sort(histo,'descend'); 
I = zeros(k,k,3);
figure();

for i=1:k
    I(i,i,:) = palette(indices(i),:);
    subplot(1,k,i);
    imagesc(I(i,i,:));
end
end
    

