function I3 = blend(I1,I2,alpha)
I1=double(I1);
I2=double(I2);
I3=alpha*I1+(1-alpha)*I2;