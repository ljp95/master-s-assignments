function histon = normalisehisto(histo)
s = sqrt(sum(histo.^2));
histon = histo/s;
end
