function Rnms = nms(R, Rb)
Rnms = Rb;

for i = 2:size(R,1)-1
    for j = 2:size(R,2)-1
        if(R(i,j) < max(max(R(i-1:i+1, j-1:j+1))))
                Rnms(i,j)= 0;
        end
    end
end
end
            