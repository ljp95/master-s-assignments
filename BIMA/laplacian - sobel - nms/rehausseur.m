function Il = rehausseur(I)
L = [0,-1,0;-1,6,-1;0,-1,0];
Il = convolution(I,L);