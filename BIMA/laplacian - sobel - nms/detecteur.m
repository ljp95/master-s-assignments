function J = detecteur(M,s)
 J = zeros(size(M,1),size(M,2));
 a = find(M>s);
 J(a) = 255;
 