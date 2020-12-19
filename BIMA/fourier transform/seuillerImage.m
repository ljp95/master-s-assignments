function I = seuillerImage(I,s)
x=find(I>=s);
y=find(I<s);
I(x)=255;
I(y)=0;