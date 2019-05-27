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
function runFragmentRegistration(i,j,k,l) 
% Location of scene files (change this and the parameters in clusterCallback.m)

sceneList = {'7-scenes-redkitchen', ...
             'sun3d-home_at-home_at_scan1_2013_jan_1', ...
             'sun3d-home_md-home_md_scan9_2012_sep_30', ...
             'sun3d-hotel_uc-scan3', ...
             'sun3d-hotel_umd-maryland_hotel1', ...
             'sun3d-hotel_umd-maryland_hotel3', ...
             'sun3d-mit_76_studyroom-76-1studyroom2', ...
             'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'};

descriptors = {'01_3DSmoothNet', ...
	'02_Occupancy', ...
	'03_Occupancy_NoLRF', ...
	'04_NoLRF',...
	'05_CGF',...
	'06_SHOT',...
	'07_3DMatch',...
	'08_FPFH'};

dims = {'16dim','32dim','64dim','128dim'};

sceneName = sceneList{j};
dim = dims{k};
descriptorName = descriptors{l};


scenePath = fullfile('../data/evaluate/input_data/3DMatch_dataset/',sceneName);
sceneDir = dir(fullfile(scenePath,'*.ply'));
numFragments = length(sceneDir);

% Get fragment pairs
fragmentPairs = {};
fragmentPairIdx = 1;
startingFragment = i;
for fragment1Idx = startingFragment:numFragments
    for fragment2Idx = (fragment1Idx+1):numFragments
        fragment1Name = sprintf('cloud_bin_%d',fragment1Idx-1);
        fragment2Name = sprintf('cloud_bin_%d',fragment2Idx-1);
        fragmentPairs{fragmentPairIdx,1} = fragment1Name;
        fragmentPairs{fragmentPairIdx,2} = fragment2Name;
        fragmentPairIdx = fragmentPairIdx + 1;
    end
end

% Run registration for all fragment pairs
for fragmentPairIdx = 1:size(fragmentPairs,1)
    clusterCallback(fragmentPairIdx,fragmentPairs,sceneName,dim,descriptorName);
end
end