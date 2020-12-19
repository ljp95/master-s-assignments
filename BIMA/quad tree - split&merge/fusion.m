function B = fusion(S,J,thresh)
B = zeros(size(J));
kmax = full(max(S(:)));

for k = 2:kmax    
    [vals,i,j] = qtgetblk(J,S,k);
    if(~isempty(vals))
        for l=1:length(i)
            if(std2(vals(:,:,l)) < thresh)
                B(i(l):i(l)+k-1,j(l):j(l)+k-1) = 1;
            end
        end
    end
end
B = bwlabel(B);
end