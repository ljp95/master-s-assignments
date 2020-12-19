%load database
if(1)
    clear, close all
    load 'images.mat'
    load 'landmarks.mat'
    load 'labels.mat'
    labelsStr = labelNum2Str(labels);
    load 'ids.mat'
    load 'mdist.mat'
    load 'mHOG.mat'
end

%for reproducibility
rng(1);

%Global parameters
numTrees=100;

numData = size(images,1);
numSubjects=ids(numData);
numClasses=max(labels)+1;

featuresOptions.type = 'distances';
features = extractFeat(images,landmarks,featuresOptions);

% % Neutral normalization of the features (see section 5.3)
% features = NeutralNormalization( features );

% Mean normalization of the features (see section 5.4)
% switch (featuresOptions.type)
%     case 'distances'
%         features = MeanNormalization( features, meandistfeat );
%     case 'HOG' 
%         features = MeanNormalization( features, meanappfeat );
%     case 'distAndHOG'
%         meancatfeat = cat(2,meandistfeat,meanappfeat);
%         features = MeanNormalization( features, meancatfeat );
%     otherwise
%         error('undefined features type' );
% end     


%generate subject-independant k-fold cross-validation
k=10;
cvids = crossvalind('Kfold', numSubjects, k);
cvdata = zeros(numData,1);
for s=1:numData
    cvdata(s)=cvids(int16(ids(s)));
end

%Confusion matrix for cross-validated models 
conf=zeros(numClasses,numClasses);
predLabels = [];

for i = 1:k
    disp(['Test fold ' num2str(i)]);
    testIdx = (cvdata == i);
    trainIdx = ~testIdx;

    trainLabels = labelsStr(trainIdx);
    trainFeatures = features(trainIdx,:);
    
    testLabels = labelsStr(testIdx);
    testFeatures = features(testIdx,:);
    % TODO
    % train a treeBagger over training instances 
    trees = TreeBagger(numTrees,trainFeatures,trainLabels);

    % TODO
    % test using test instances
    Y = predict(trees,testFeatures)
    
    predLabels=[predLabels;Y];
    confcurr=confusionmat(testLabels,Y);
    conf=conf+confcurr;
end

%normalize confusion matrix
for c=1:numClasses
    conf(c,:)=conf(c,:)/sum(conf(c,:));
end
imagesc(conf);
disp(['Accuracy: ' num2str(trace(conf/numClasses))]);
colorbar
xlabel('predictions')
ylabel('labels')

save('resRF.mat','trees','featuresOptions');