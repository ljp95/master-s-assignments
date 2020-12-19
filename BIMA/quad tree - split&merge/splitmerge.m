function B = splitmerge(filename,thresh)
I = imread(filename);
J = expand(I);
S = qtdecomp(J,thresh);
B = fusion(S,J,thresh);
figure();imagesc(label2rgb(B(1:size(I,1),1:size(I,2))));title('splitmerge');
end