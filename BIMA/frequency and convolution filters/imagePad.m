function J = imagePad(I,h)

% [n_f, m_f] = size(h);
% [n_I, m_I] = size(I);
% J1 = [zeros(floor(n_f/2), m_I); I;  zeros(floor(n_f/2), m_I)];
% [n_3, m_3] = size(J1);
% J = [zeros(n_3, floor(n_f/2)), J1, zeros(n_3, floor(n_f/2))];

[m,n] = size(I);
[c,d] = size(h);
J = zeros((c-1)+m,(d-1)+n);
for i = 1:size(I,1)
    for j = 1:size(I,2)
        J(i+(c-1)/2,j+(d-1)/2) = I(i,j);
    end
end
end

