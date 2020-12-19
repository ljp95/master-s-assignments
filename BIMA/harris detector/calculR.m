function R = calculR(I,echelle)
hx = [1,0, -1];
hy = [0,1,0];
Ix = convolution_separable(I,hx,hy.');
hx = [0,1, 0];
hy = [1,0,-1];
Iy = convolution_separable(I,hx,hy.');
w = gauss1d(echelle);

Ix2 = conv2(Ix.*Ix,w,'same');
Iy2 = conv2(Iy.*Iy,w,'same');
IxIy = conv2(Ix.*Iy,w,'same');

detM = Ix2.*Iy2 - IxIy.*IxIy;
trM = Ix2 + Iy2;
R = detM - 0.04.*trM.*trM;
end