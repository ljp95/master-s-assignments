function [ features ] = extractFeat(images,landmarks,featuresOptions)

switch (featuresOptions.type)
    case 'distances'
        features = extractDistances(landmarks);
    case 'HOG' 
        features = extractAppearanceFeatures(images,landmarks);
    case 'distAndHOG'
        features = cat(2,extractDistances(landmarks),extractAppearanceFeatures(images,landmarks));
    otherwise
        error('undefined features type' );
end