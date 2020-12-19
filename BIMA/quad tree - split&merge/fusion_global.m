function B = fusion_global(S,J,thresh)
B = zeros(size(J));
kmax = full(max(S(:)));
moy = 0;
var = 0;
n = 0;
for k = 1:kmax    
    [vals,i,j] = qtgetblk(J,S,k);
    if(~isempty(vals))
        for l=1:length(i)
            n1 = n + (k^2);
            moy1 = (moy*n + (k^2)*mean2(vals(:,:,l)))/n1;
            var1 = (n*(var + moy^2) + (k^2)*(std2(vals(:,:,l))^2 + mean2(vals(:,:,l))^2))/n1 - moy1^2;
            if(var1^(1/2) < thresh)
                B(i(l):i(l)+k-1,j(l):j(l)+k-1) = 1;
                n = n1;
                moy = moy1;
                var = var1;
            end
        end
    end
end
B = bwlabel(B);
end