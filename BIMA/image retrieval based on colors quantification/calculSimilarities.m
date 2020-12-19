function sim = calculSimilarities(listHist)
n = size(listHist,1);
sim = zeros(n,n);
for i=1:n
    for j=1:n
        sim(i,j) = sum(listHist(i,:).*listHist(j,:));
    end
end
end
