function Ff = filtrePasseBasIdeal(n,m,fc)
Ff = zeros(n,m);
centre = [round(n/2),round(m/2)];
for i = -centre(1):centre(1)
    for j = -centre(2):centre(2)
        if (sqrt(i^2+j^2)<fc)
            Ff(i+centre(1),j+centre(2)) = 1;
        end
    end
end
end

        
        