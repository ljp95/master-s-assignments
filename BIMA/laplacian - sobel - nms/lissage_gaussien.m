function J = lissage_gaussien(I,sigma)

J = convolution(I,gauss(sigma));