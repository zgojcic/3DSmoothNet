% Script to run the evaluation of the 3DMatch data set 
% as described in the supplementary material of the 
% The Perfect Match: 3D Point Cloud Matching with Smoothed Densities
% https://arxiv.org/abs/1811.06879 
% 
% ---------------------------------------------------------
% Copyright (c) 2019, Zan Gojcic
% 
% This file is part of the 3DSmoothNet Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Change this parameters (Check the runFragmentRegistration.m for more info)
sceneIDX = 1;
descriptorIDX = 1;
dimIDX = 1;
startFragmentIDX = 1;


runFragmentRegistration(startFragmentIDX,sceneIDX,dimIDX,descriptorIDX)

%% Code can be easily paralelized, bellow the implementation that we used on
% a clustser

% for j = 1
% 	sceneIDX = j
% 	dimIDX = 2
% 	descriptorIDX = 7
% 	sceneName = scenes{j};
% 	scenePath = fullfile('/cluster/scratch/zgojcic/evaluate3dMatch/01_Data',sceneName,'/')
% 	sceneDir = dir(fullfile(scenePath,'*.ply'));
% 	numFragments = length(sceneDir);
% 
% 
% 	for i = 1: numFragments 
%  		command = ['bsub -n 1 -W  "4:00" -R "rusage[mem=8192]" matlab -nodisplay -singleCompThread -r "runFragmentRegistration_PPFnet(',num2str(i),',',num2str(sceneIDX),',',num2str(dimIDX ),',',num2str(descriptorIDX),')"'];
%  		system(command); 
% 	end
% end