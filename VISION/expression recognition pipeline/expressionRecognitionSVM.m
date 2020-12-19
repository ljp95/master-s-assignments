%% load database
if(1)
    clear, close all
    load 'images.mat'
    load 'landmarks.mat'
    load 'labels.mat'
    labelsStr = labels+1;
    load 'ids.mat'
    load 'mdist.mat'
    load 'mHOG.mat'
end

%for reproducibility
rng(1);

numData = size(images,1);
numSubjects=ids(numData);
numClasses=max(labels)+1;

featuresOptions.type = 'distances';
features = extractFeat(images,landmarks,featuresOptions);

% % Neutral normalization of the features (section 5.3)
% features = NeutralNormalization( features );

% Mean normalization of the features (section 5.4)
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


%% generate subject-independant k-fold cross-validation
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
    classesPred = [];

    testIdx = (cvdata == i);
    trainIdx = ~testIdx;
    
    trainLabels = labelsStr(trainIdx);
    trainFeatures = features(trainIdx,:);
    
    testLabels = labelsStr(testIdx);
    testFeatures = features(testIdx,:);
    
    
    for (ic = 1:numClasses)
        disp(['SVM training for expression #' num2str(ic)]);
        trainLabelsBin = label2bin(trainLabels, ic);
        testLabelsBin = label2bin(testLabels, ic);
        
        SVMModels{ic} =  fitcsvm(trainFeatures,trainLabelsBin','BoxConstraint',0.1,'KernelFunction','linear');
        SVMModels{ic} = fitSVMPosterior(SVMModels{ic});
        [prediction, sco] = predict(SVMModels{ic},testFeatures);
    end
    
    for itf = 1 : size(testFeatures,1)
        x = testFeatures(itf,:);
        classesPred(itf) = predMultiClass1vsAll(SVMModels,x);
    end

    %scoresMultiClass(i) = sum(classesPred == testLabels')/length(testLabels)
    %rejected(i) = sum(classesPred == -1)/length(testLabels)
    
    
    %predLabels=[predLabels;classesPred];
    confcurr=confusionmat(testLabels,classesPred,'order',unique(labelsStr)');
    conf=conf+confcurr;
end

%normalize confusion matrix
for c=1:numClasses
    conf(c,:)=conf(c,:)/sum(conf(c,:));
end

imagesc(conf);
disp(['Accuracy: ' num2str(trace(conf/numClasses))]);

%figure; plot(classesPred);
%hold on; plot(grp2idx(testLabels), 'g')

% for (ic = 1:numClasses)
%     trainLabelsBin = label2bin(labelsStr, ic);
%     
%     SVMModels{ic} =  fitcsvm(features,trainLabelsBin','BoxConstraint',0.1,'KernelFunction','linear');
%     SVMModels{ic} = fitSVMPosterior(SVMModels{ic});
%     [prediction, sco] = predict(SVMModels{ic},testFeatures);
% end

% save('resSVM.mat','SVMModels','featuresOptions');
