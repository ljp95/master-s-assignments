function h = calculerHisto(I)
h = zeros(1,256);
for i = 0:255
    h(i+1) = compterPixels(I,i);
end


