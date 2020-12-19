function [distances]=extractDistances(landmarks)

%disp('Distance feature extraction...');

featurePoints_lefteyeint = 26;
featurePoints_righteyeint = 23;
featurePoints_lefteyeext = 29;
featurePoints_righteyeext = 20;

%% TODO
[N,m] = size(landmarks);
N_points = m/2;
distances = zeros(N,N_points*(N_points-1)/2);

%Distances
for i = 1:N
    tmp = [];
    for j = 1:N_points-1
        %One point
        x = landmarks(i,j);
        y = landmarks(i,j+N_points);
        %Other points
        other_x = landmarks(i,j+1:N_points);
        other_y = landmarks(i,j+1+N_points:end);
        %Compute distances
        tmp = [tmp,sqrt(sum([x-other_x ; y-other_y].^2))];
    end
    %Affectation
    distances(i,:) = tmp;
end 

%Normalization
%Compute all left and right eyes coordinates
left  = [landmarks(:,26)+landmarks(:,29) ,landmarks(:,26+N_points)+landmarks(:,29+N_points)]/2;
right = [landmarks(:,23)+landmarks(:,20) ,landmarks(:,23+N_points)+landmarks(:,20+N_points)]/2;
%Distance between eyes
iod = sqrt(sum((left-right).^2,2));
%Normalize
distances = distances ./ iod;

end
