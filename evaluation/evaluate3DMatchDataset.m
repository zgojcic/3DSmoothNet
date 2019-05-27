% Evaluation Code based on the 3Dmatch-toolbox (https://github.com/andyzeng/3dmatch-toolbox)
% Script to evaluate .log files for the geometric registration benchmarks,
% in the same spirit as Choi et al 2015. Please see:
%
% http://redwood-data.org/indoor/regbasic.html
% https://github.com/qianyizh/ElasticReconstruction/tree/master/Matlab_Toolbox
clc;
clearvars;

descriptorIDX = 1;
dimIDX = 1;


descriptorName = {'01_3DSmoothNet','02_Occupancy','03_Occupancy_NoLRF','04_NoLRF','05_CGF','06_SHOT','07_3DMatch','08_FPFH'};
dim = {'16dim','32dim','64dim','128dim'};
dataPath = '../data/evaluate/input_data/3DMatch_dataset/'; % Location of scene files
intermPath = fullfile('../data/evaluate/input_data/3DMatch_dataset/registration_interim_results',descriptorName{descriptorIDX},dim{dimIDX}); % Location of intermediate registration results
    
% Real data benchmark
sceneList = {'7-scenes-redkitchen', ...
             'sun3d-home_at-home_at_scan1_2013_jan_1', ...
             'sun3d-home_md-home_md_scan9_2012_sep_30', ...
             'sun3d-hotel_uc-scan3', ...
             'sun3d-hotel_umd-maryland_hotel1', ...
             'sun3d-hotel_umd-maryland_hotel3', ...
             'sun3d-mit_76_studyroom-76-1studyroom2', ...
             'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'};

% Load Elastic Reconstruction toolbox
addpath(genpath('external'));

% Compute precision and recall
totalRecall = [];
averageNumberOfMatches = [];
for sceneIdx = 1:length(sceneList)
    resultRt = [];
    % List fragment files
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    sceneDir = dir(fullfile(scenePath,'*.ply'));
    numFragments = length(sceneDir);
    % fprintf('%s\n',scenePath);
    
    % Loop through registration results
    for fragment1Idx = 1:numFragments
        for fragment2Idx = (fragment1Idx+1):numFragments
            fragment1Name = sprintf('cloud_bin_%d',fragment1Idx-1);
            fragment2Name = sprintf('cloud_bin_%d',fragment2Idx-1);            
            resultPath = fullfile(intermPath,sceneList{sceneIdx},sprintf('%s-registration-results',descriptorName{descriptorIDX}),sprintf('%s-%s.rt.txt',fragment1Name,fragment2Name));
            fid1 = fopen(resultPath);
            data = textscan(fid1,'%s','Delimiter','\t');
            fclose(fid1);
            resultRt = [resultRt;str2num(data{1}{3}),str2num(data{1}{4}),str2num(data{1}{5}) ];
        end
    end
    indicesResults = find(resultRt(:,3) == 1);
    correctMatches = find(resultRt(indicesResults,2) > 0.05);
    totalRecall = [totalRecall, length(correctMatches)/length(indicesResults)];
    averageNumberOfMatches = [averageNumberOfMatches, floor(mean(resultRt(indicesResults,1)))];
    
    fprintf('Scene: %s \n' ,sceneList{sceneIdx});
    fprintf('Recall: %f \n' ,totalRecall(sceneIdx));
    fprintf('Average nr of matches: %i \n\n' ,averageNumberOfMatches(sceneIdx));
    
end


fprintf('Mean Recall: %f \n' ,mean(totalRecall));
fprintf('STD of Recall: %f \n' ,std(totalRecall));
