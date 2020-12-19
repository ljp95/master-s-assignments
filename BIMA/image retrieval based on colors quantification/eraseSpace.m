function goodFileName = eraseSpace(fileName)
n = size(fileName,2);
for i=1:n
    if fileName(n+1-i) ~= ' '
        break
    end
end
if i==1
    goodFileName = fileName;
else
    goodFileName = fileName(1:n+1-i);
end
end