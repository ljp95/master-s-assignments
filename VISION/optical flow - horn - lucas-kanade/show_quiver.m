function show_quiver(u,v,scale)
[n,m] = size(u);
x = zeros(size(u));
y = zeros(size(v));
for i=1:10:n
    for j=1:10:m
        x(i,j) = u(i,j);
        y(i,j) = v(i,j);
    end
end
quiver(x,y,scale);
set(gca,'Ydir','reverse');
end




