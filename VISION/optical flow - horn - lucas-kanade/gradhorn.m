function [Ix,Iy,It] = gradhorn(I1,I2)
I1 = double(I1);
I2 = double(I2);
I3 = I1+I2;

Ix = (conv2(I3,[1 -1;1 -1],'same'))/4;
Iy = (conv2(I3,[1 1;-1 -1],'same'))/4; 
It = (conv2(I2-I1,[1 1;1 1],'same'))/4;
end








