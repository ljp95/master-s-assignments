function J = remplacerPixels(I,k1,k2)
x = find(I==k1);
J = I;
J(x) = k2;