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

function clusterCallback(jobID,fragmentPairs,sceneName,dim,descriptorName)

    % Configuration options (change me)
    scenePath = fullfile('../data/evaluate/input_data/3DMatch_dataset/',sceneName); % Location of scene fragment point clouds
    keypointsPath = fullfile(scenePath,'01_Keypoints');
    resultsPath = fullfile('../data/evaluate/input_data/3DMatch_dataset/registration_interim_results',descriptorName,dim,sceneName);
    descriptorsPath = fullfile('../data/evaluate/output_data/3DMatch_dataset/',descriptorName,dim,sceneName);
    
    % Add libraries
    addpath(genpath('../../core/external'));

    % List out scene fragments Location of scene files
    sceneDir = dir(fullfile(scenePath,'*.ply'));
    numFragments = length(sceneDir);
    
    
    if jobID > size(fragmentPairs,1)
        return;
    end
    
    fragment1Name = fragmentPairs{jobID,1};
    fragment2Name = fragmentPairs{jobID,2};
    fprintf('Registering %s and %s: ',fragment1Name,fragment2Name);

    % Get results file
    resultPathFull = fullfile(resultsPath,sprintf('%s-registration-results',descriptorName),sprintf('%s-%s.rt.txt',fragment1Name,fragment2Name));
    if ~exist(fullfile(resultsPath,sprintf('%s-registration-results',descriptorName)), 'dir')
      mkdir(fullfile(resultsPath,sprintf('%s-registration-results',descriptorName)));
    end
    if exist(resultPathFull,'file')
        fprintf('\n');
        return;
    end

    % Compute rigid transformation that aligns fragment 2 to fragment 1
    [numInliers,inlierRatio,gtFlag] = register2Fragments(scenePath,keypointsPath,descriptorsPath,fragment1Name,fragment2Name,descriptorName);
    fprintf('%d %f %d\n',numInliers,inlierRatio,gtFlag);

    % Save rigid transformation
    fid = fopen(resultPathFull,'w');
    fprintf(fid,'%s\t %s\t %d\t %15.8e\t %d\t',fragment1Name,fragment2Name,numInliers,inlierRatio,gtFlag);
    fclose(fid);
end