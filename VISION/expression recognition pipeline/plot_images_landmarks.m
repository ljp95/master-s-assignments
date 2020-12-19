number = 18;
fig = figure();
imshow(cell2mat(images(number)))
n_points = size(landmarks,2)/2;
hold on
for i = 1:n_points
    x = round(landmarks(number,i));
    y = round(landmarks(number,i+n_points));
    plot(x,y,'r+','Markersize',10)
end 
% saveas(fig,"landmarks.jpeg");
