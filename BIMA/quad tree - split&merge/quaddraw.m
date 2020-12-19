function Q = quaddraw(I,S)
kmax = full(max(S(:)));
Q = cat(3,I,I,I);  % matrix 3d initialized at I

for k = 1:kmax    
    [vals,i,j] = qtgetblk(I,S,k);
    if(~isempty(vals))
        for l=1:length(i)
            Q(i(l)+k-1,j(l):j(l)+k-1,1) = 255;
            Q(i(l):i(l)+k-1,j(l)+k-1,1) = 255;
            Q(i(l)+k-1,j(l):j(l)+k-1,2:3) = 0;
            Q(i(l):i(l)+k-1,j(l)+k-1,2:3) = 0;
        end
    end
end
end



         
          