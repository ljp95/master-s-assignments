function palette = calculerPalette(nH,nS,nV)
% Out : (nH*nS*nV) RGB colors
counter = 1;
palette = zeros(nH*nS*nV,3);    

for i=1:nV
    for j=1:nS
        for k=1:nH
            % Evenly spaced numbers from 0 to 
            % (nH-1)/nH,(nS-1)/nS,(nV-1)/nV
            % -0.75 to avoid 0 and 1
            palette(counter,:) = [(k-0.75)/nH, (j-0.75)/nS, (i-0.75)/nV];
            counter = counter +1;
        end
    end
end
palette = hsv2rgb(palette);
end
