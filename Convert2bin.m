%% Convert2bin

% Amirreza Bahramani
% IPM, Birds Lab
% April 2023

% This code will load the data and convert it from .ns6 to .mat and .bin

close all
clear
clc

%% PARAMETERS
addpath(['C:\Users\bahra\Documents\Personal Documents\IPM\Birds Lab\' ...
         'Template Codes\MATLAB\NPMK-5.5.2.0\NPMK']);

birdID = 'ZF003AM';
mainPath = ['..\Subjects\' birdID '\Neural\recordings\'];
dataPath = [mainPath 'raw\'];

dataSuff = '.ns5';
binDataPath = [mainPath '\bin files\'];
files = dir(fullfile(dataPath, '*.ns5'));

saveMat = true;
saveMatAll = true;

neuralData = cell(numel(files), 1);
rawSig = cell(numel(files), 1);
rawAudio = cell(numel(files), 1);
filtSig = cell(numel(files), 1);

%% LOAD .ns5 FILES & CONVERT

for sessNum = 1:numel(files)

    dataName =  files(sessNum).name;

    neuralData{sessNum} = openNSx('report', 'read', [dataPath dataName], 'uv');
    selectedChannel = 1;
    rawSig{sessNum} = neuralData{sessNum}.Data(selectedChannel, :);
    rawAudio{sessNum} = neuralData{sessNum}.Data(2, :);
    fs = neuralData{sessNum}.MetaTags.SamplingFreq; % 30000 %Hz for BlackRock

    % rawSig{sessNum} = movmean(rawSig{sessNum}, 20);

    if saveMat
        neuralSig = rawSig{sessNum};
        audioSig = rawAudio{sessNum};
        save([mainPath birdID '_neural_' sprintf('%02d', sessNum) '.mat'], 'neuralSig', 'audioSig', 'fs')
    end

    tmpSig = double(rawSig{sessNum});
    coeff = 1/max(abs(tmpSig));
    tmpSig = int16(coeff*tmpSig*32000);

    fileName = [binDataPath dataName];
    fileName = strrep(fileName, dataSuff, '_16bit_nomov.bin');
    fprintf('Writing %s\n', fileName);
    fid = fopen(fileName, 'w');
    fwrite(fid, tmpSig, 'int16');
    fclose(fid);

end