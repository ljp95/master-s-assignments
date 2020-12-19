function J = passage(Il,s)
J = zeros(size(Il,1),size(Il,2));
for i = 2:size(Il,1)-1
    for j = 2:size(Il,2)-1
        m = max(max(Il(i-1:i+1,j-1:j+1)));
        n = min(min(Il(i-1:i+1,j-1:j+1)));
        if((m>0) && (n<0) && (m-n)>s)
            J(i,j) = 255;
        end
    end
end
