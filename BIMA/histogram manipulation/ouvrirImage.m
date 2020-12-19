function [I,n,m] = ouvrirImage(nom)
I = imread(nom);
I = double(I);
n = size(I,1);
m = size(I,2);