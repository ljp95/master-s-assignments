function [textures]=extractAppearanceFeatures(images, landmarks)

%disp('Appearance feature extraction...');

featurePoints_lefteyeint = 26;
featurePoints_righteyeint = 23;
featurePoints_lefteyeext = 29;
featurePoints_righteyeext = 20;

numExamples=size(landmarks,1);
numPoints=size(landmarks,2)/2;

% TODO
% Set the HOG parameters
blockSize = [2 2];
numBins = 9;
CellSizeRatio = 0.2;
descSize = numBins * blockSize(1)* blockSize(2) * numPoints;
textures = zeros(numExamples, descSize);

margin=200;

for i=1:numExamples
    if(mod(i,25) == 0)
        disp(['Image #' num2str(i)]);
    end
    % TODO
    % Compute the interocular distance
    left  = [landmarks(i,26)+landmarks(i,29) ,landmarks(i,26+numPoints)+landmarks(i,29+numPoints)]/2;
    right = [landmarks(i,23)+landmarks(i,20) ,landmarks(i,23+numPoints)+landmarks(i,20+numPoints)]/2;
    iod = sqrt(sum((left-right).^2));
    
    cellSize = [int16(CellSizeRatio*iod) int16(CellSizeRatio*iod)];
    
    im1=images{i};
    
    %pad image with zeros in order to avoid exceeding image dimensions
    szy=size(im1,1);
    szx=size(im1,2);
    impad=zeros(szy+2*margin,szx+2*margin);
    impad(margin+1:margin+szy,margin+1:margin+szx)=im1;
    
    %we also need to add an offset to the landmarks
    pts=reshape(landmarks(i,:), 49, 2)+margin*ones(49,2);
    
    % TODO
    % Compute the HOG Descriptor for padded version of the image i
    [features,validPoints,visualization] = extractHOGFeatures(impad,pts, 'numBins',numBins, 'BlockSize',blockSize, 'CellSize',cellSize);
    % TODO
    % Store the descriptor in the texture matrix
    tmp = features'; % as tmp(:) concatenates each column
    textures(i,:) = tmp(:); 
%     plot(visualization)    
end

end