function [d_min,d_max] = minAndMax(M,class1,class2)
d_min = max(max(M));
d_max = min(min(M));
% To avoid comparing the same face from the same class
% Adding 1 in the loop if that's the case, nothing if not
add = 0;
if(class1 == class2) 
    add = 1;
end
    
for i = 1:size(class1,2)
    for j = i + add:size(class1,2)
        if(d_min > M(class1(i),class2(j)))
            d_min = M(class1(i),class2(j));
        else
            if(d_max < M(class1(i),class2(j)))
                d_max = M(class1(i),class2(j));
            end
        end
    end
end