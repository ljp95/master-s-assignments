function Rb = seuilleR(R,S)
Rb = zeros(size(R));
x = find(R>S);
Rb(x) = 100;
end