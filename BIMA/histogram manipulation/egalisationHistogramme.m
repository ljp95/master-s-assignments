function J = egalisationHistogramme(I,h)
H = cumsum(h);
J = I;
pixels = size(I,1)*size(I,2);
for i = 0:255
    J(find(I==i)) = round(256*H(i+1)/pixels);
end
