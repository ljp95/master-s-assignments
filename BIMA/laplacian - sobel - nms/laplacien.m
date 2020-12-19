function Il = laplacien(I)
L = [0,1,0;1,-4,1;0,1,0];
Il = convolution(I,L);