function [] = split(filename)
I = imread(filename);
thresh = std2(I)/2;
J = expand(I);
S = qtdecomp(J, thresh);
J2 = quaddraw(J,S);
imagesc(J2);