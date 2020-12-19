function K = nms(Ig, Ior)
%K = zeros(size(Ig,1),size(Ig,2));
K = Ig;
%K(K>0)=1;
for i = 2:size(Ig,1)-1
    for j = 2:size(Ig,2)-1
        if (Ior(i,j)==1)
            if(Ig(i,j)<Ig(i,j-1) || Ig(i,j) < Ig(i,j+1))
                K(i,j)= 0;
            end
        end
        if (Ior(i,j)==2)
            if(Ig(i,j)<Ig(i+1,j+1) || Ig(i,j) < Ig(i-1,j-1))
                K(i,j)= 0;
            end
        end
        if (Ior(i,j)==3)
            if(Ig(i,j)<Ig(i+1,j) || Ig(i,j) < Ig(i-1,j))
                K(i,j)= 0;
            end
        end
        if (Ior(i,j)==4)
            if(Ig(i,j)<Ig(i+1,j-1) || Ig(i,j) < Ig(i-1,j+1))
                K(i,j)= 0;
            end
        end
    end
end
            

