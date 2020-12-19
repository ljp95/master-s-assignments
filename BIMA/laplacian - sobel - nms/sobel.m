function [Ix,Iy] = sobel(I)
Gx = [1,0,-1;2,0,-2;1,0,-1];
Gy = [1,2,1;0,0,0;-1,-2,-1];
Ix = convolution(I,Gx);
Iy = convolution(I,Gy);