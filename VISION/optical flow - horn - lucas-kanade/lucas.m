function [u,v] = lucas(I1,I2,n)
I1 = double(I1);    I2 = double(I2);
[mx,my] = size(I1); mid = floor(n/2);
u = zeros(mx,my);   v = zeros(mx,my);

[Ix,Iy,It] = gradhorn(I1,I2);

for i = mid+1 : mx-mid
    for j = mid+1 : my-mid
        B = -It(i-mid:i+mid , j-mid:j+mid)';
        X =  Ix(i-mid:i+mid , j-mid:j+mid)';
        Y =  Iy(i-mid:i+mid , j-mid:j+mid)';
        A = [X(:),Y(:)];
        w = pinv(A'*A)*A'*B(:);
        u(i,j) = w(1);
        v(i,j) = w(2);
    end
end
end