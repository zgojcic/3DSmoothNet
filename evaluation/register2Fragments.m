function [numInliers,inlierRatio,gtFlag] = register2Fragments_PPFnet(scenePath,keypointsPath,descriptorPath,fragment1Name,fragment2Name,descriptorName)
% Script to run the evaluation of the 3DMatch data set 
% as described in the supplementary material of the 
% The Perfect Match: 3D Point Cloud Matching with Smoothed Densities
% https://arxiv.org/abs/1811.06879 
% Code is based on the 3DMatchToolbox (if you this code use please cite)
% (https://github.com/andyzeng/3dmatch-toolbox)
% 
% ---------------------------------------------------------
% Copyright (c) 2019, Zan Gojcic
% 
% This file is part of the 3DSmoothNet Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Load fragment point clouds
fragment1PointCloud = pcread(fullfile(scenePath,sprintf('%s.ply',fragment1Name)));
fragment2PointCloud = pcread(fullfile(scenePath,sprintf('%s.ply',fragment2Name)));

% Load keypoints of fragment 1
fragment1Keypoints = csvread(fullfile(keypointsPath,sprintf('%sKeypoints.txt',fragment1Name)));
fragment1Keypoints = fragment1PointCloud.Location(fragment1Keypoints+1,:);

% Load 3DMatch feature descriptors for keypoints of fragment 1
if strcmp(descriptorName,'01_3DSmoothNet') == 1
fragment1Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.150000_16_1.750000_CNN.txt',fragment1Name)));
elseif strcmp(descriptorName,'04_NoLRF') == 1
fragment1Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_%s_0.150000_16_1.750000_CNN.txt',fragment1Name,descriptorName(4:end))));
elseif strcmp(descriptorName,'05_CGF') == 1
fragment1Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.186000_CNN.txt',fragment1Name)));
elseif strcmp(descriptorName,'06_SHOT') == 1
fragment1Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.186000.txt',fragment1Name)));
elseif strcmp(descriptorName,'07_3DMatch') == 1
fid = fopen(fullfile(descriptorPath,sprintf('%s.desc.3dmatch.bin',fragment1Name)),'rb');
fragment1DescriptorData = fread(fid,'single');
fragment1NumDescriptors = fragment1DescriptorData(1);
fragment1DescriptorSize = fragment1DescriptorData(2);
fragment1Descriptors = reshape(fragment1DescriptorData(3:end),fragment1DescriptorSize,fragment1NumDescriptors)';
fclose(fid);
elseif strcmp(descriptorName,'08_FPFH') == 1
fragment1Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.093000.txt',fragment1Name)));
else
fragment1Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_%s_0.150000_16_CNN.txt',fragment1Name,descriptorName(4:end))));
end

% Load keypoints of fragment 2
fragment2Keypoints = csvread(fullfile(keypointsPath,sprintf('%sKeypoints.txt',fragment2Name)));
fragment2Keypoints = fragment2PointCloud.Location(fragment2Keypoints+1,:);

% Load 3DMatch feature descriptors for keypoints of fragment 2
if strcmp(descriptorName,'01_3DSmoothNet') == 1
fragment2Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.150000_16_1.750000_CNN.txt',fragment2Name)));
elseif strcmp(descriptorName,'04_NoLRF') == 1
fragment2Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_%s_0.150000_16_1.750000_CNN.txt',fragment2Name,descriptorName(4:end))));
elseif strcmp(descriptorName,'05_CGF') == 1
fragment2Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.186000_CNN.txt',fragment2Name)));
elseif strcmp(descriptorName,'06_SHOT') == 1
fragment2Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.186000.txt',fragment2Name)));
elseif strcmp(descriptorName,'07_3DMatch') == 1
fid = fopen(fullfile(descriptorPath,sprintf('%s.desc.3dmatch.bin',fragment2Name)),'rb');
fragment2DescriptorData = fread(fid,'single');
fragment2NumDescriptors = fragment2DescriptorData(1);
fragment2DescriptorSize = fragment2DescriptorData(2);
fragment2Descriptors = reshape(fragment2DescriptorData(3:end),fragment2DescriptorSize,fragment2NumDescriptors)';
fclose(fid);
elseif strcmp(descriptorName,'08_FPFH') == 1
fragment2Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_0.093000.txt',fragment2Name)));
else
fragment2Descriptors = csvread(fullfile(descriptorPath,sprintf('%s_%s_0.150000_16_CNN.txt',fragment2Name,descriptorName(4:end))));
end


% Find mutually closest keypoints in feature descriptor space
fragment2KDT = KDTreeSearcher(fragment2Descriptors);
fragment1KDT = KDTreeSearcher(fragment1Descriptors);
fragment1NNIdx = knnsearch(fragment2KDT,fragment1Descriptors);
fragment2NNIdx = knnsearch(fragment1KDT,fragment2Descriptors);
fragment2MatchIdx = find((1:size(fragment2NNIdx,1))' == fragment1NNIdx(fragment2NNIdx));
fragment2MatchKeypoints = fragment2Keypoints(fragment2MatchIdx,:);
fragment1MatchKeypoints = fragment1Keypoints(fragment2NNIdx(fragment2MatchIdx),:);

% Load GT transformation parameters if they exist
groundTruthTransformations = mrLoadLog(fullfile(scenePath,'gt.log'));
groundTruthIndices = reshape([groundTruthTransformations.info]',3,[])';

% Get numbers of the fragments
fragment1Number = strsplit(fragment1Name,'_');
fragment2Number = strsplit(fragment2Name,'_');
isOverlaping = find(groundTruthIndices(:,1) ==  str2num(fragment1Number{end}) & groundTruthIndices(:,2) == str2num(fragment2Number{end}));
if isempty(isOverlaping)
    numInliers = 0;
    inlierRatio = 0;
    gtFlag = 0;
else
    fragment2Points = groundTruthTransformations(isOverlaping).trans * [fragment2MatchKeypoints'; ones(1,length(fragment2MatchKeypoints))];
    fragment2Points = fragment2Points(1:3,:)';
    fragment1Points = fragment1MatchKeypoints;
    distances  = sqrt(sum((fragment1Points-fragment2Points).^2,2));
    numInliers = length(find(distances < 0.1));
    inlierRatio = numInliers/length(distances);
    gtFlag = 1;
    
end

end











