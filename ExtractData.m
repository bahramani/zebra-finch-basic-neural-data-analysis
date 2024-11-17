function [rawSig, fs, rawAudio, events] = ExtractData(dataPath, session)

    addpath(['C:\Users\rimaz\Documents\Personal Documents\Works\IPM\' ...
        'Birds Lab\Template Codes\MATLAB\NPMK-5.5.2.0\NPMK']);
    dataSuff = '.ns6';
    dataName = [dataPath 'data_raw' sprintf('%03d', session) dataSuff];
    neuralData = openNSx('report', 'read', dataName, 'uv');
    fs = neuralData.MetaTags.SamplingFreq; % 30000 %Hz
    rawSig = neuralData.Data(1, :);
    rawAudio = neuralData.Data(2, :);
    
    eventData = importdata([dataPath 'events\2023-03-26_18-28-55\trial-events-18-29-03.txt'], ',');
    events = eventData.data;
    events(:,1) = events(:,1)*1000; %convert to ms

end

