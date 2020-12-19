function id_test_hat = identification(D,id_train)
[Y,indices] = min(D,[],2); %indice of most close train face for each face test
id_test_hat = zeros(1,size(D,1));
for i = 1:size(D,1)
    %Getting the indice of the person behind the train face
    id_test_hat(i) = id_train(indices(i));
end
end
