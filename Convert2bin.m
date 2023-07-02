%% Convert2bin

% Amirreza Bahramani
% IPM, Birds Lab
% April 2023

% This code will convert the raw neural data recorded with BlackRock (.ns6) to .mat and .bin
% The .bin file can be used for Plexon Offline Sorter.
% Place the code in your .ns6 folder.
% This needs NPMK toolbox which is uploaded before.

close all
clear
clc

%% PARAMETERS
saveMat = true;
saveBin = true;
saveMatAll = true;

% Add NPMK toolbox for loading .ns6 files
addpath(['C:\Users\rimaz\Documents\Personal Documents\Works\IPM\' ...
    'Birds Lab\Template Codes\MATLAB\NPMK-5.5.2.0\NPMK']);
% Write the path to raw data
dataPath = pwd;
dataSuff = '.ns6';
binDataPath = [dataPath '\bin files\'];
files = dir(fullfile(dataPath, '*.ns6'));

neuralData = cell(numel(files), 1);
rawSig = cell(numel(files), 1);
filtSig = cell(numel(files), 1);


%% LOAD .ns6 FILES & CONVERT

for sessNum = 1:numel(files)

    dataName =  files(sessNum).name;

    neuralData{sessNum} = openNSx('report', 'read', dataName, 'uv');
    selectedChannel = 1;
    rawSig{sessNum} = neuralData{sessNum}.Data(selectedChannel, :);
    fs = neuralData{sessNum}.MetaTags.SamplingFreq; % 30000 %Hz for BlackRock
    
    if saveMat
        tmp = rawSig{sessNum};
        save([dataName(1:end-4) '.mat'],'tmp', 'fs')
    end

    % convert2bin   
    if saveBin   
        
        tmpSig = double(rawSig{sessNum});
        coeff = 1/max(abs(tmpSig));
        tmpSig = int16(coeff*tmpSig*32000);
        
        fileName = [binDataPath dataName];
        fileName = strrep(fileName, dataSuff, '_16bit.bin');
        fprintf('Writing %s\n', fileName);
        fid = fopen(fileName, 'w');
        fwrite(fid, tmpSig, 'int16');
        fclose(fid);
    end
end

if saveMatAll
    save('1402-03-28_HVC.mat', 'rawSig', 'fs');
end
