function If = convolution(I,h)
[n,m] = size(I);
[c,d] = size(h);
g = rot90(h,2);
g = double(g);
I = double(I);
J = imagePad(I,g);
If = zeros(n,m);
for i = 1:n
    for j = 1:m
        If(i,j) = sum(sum(J(i:i+(c-1),j:j+(d-1)).*g));
    end
end
end