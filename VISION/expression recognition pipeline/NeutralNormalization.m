function [ feato ] = NeutralNormalization( feati )

N = size(feati,1);
feato = feati;
feato(2:2:N) = feato(2:2:N) - feato(1:2:N);

end
