function [I,n,m] = ouvrirImage(nom)
    I = double(imread(nom));
    [n,m] = size(I);
end
