%% Behavioral_Analysis

% IPM, Birds Lab
% by Amirreza Bahramani
% June 2024

% This code is for analyzing behavioral data of the "Noisy Calls 1" project.

close all
clear
clc

% Tet/DC response
% Alternative for 1/0 response
% Tet 10 dB is higher because friend is nearby

%% PARAMETERS
addpath(genpath('I:\IPM\Birds Lab\Template Codes\chronux\'));

birdID = 'ZF003AM';
sessionNum = '01';
colorPalette = [
    0, 51, 102;     % Midnight Blue
    255, 102, 0;    % Dark Orange
    34, 139, 34;    % Forest Green
    220, 20, 60;    % Crimson Red
    0, 191, 255;    % Deep Sky Blue
    218, 165, 32;   % Goldenrod
    120, 81, 169;   % Royal Purple
    64, 224, 208;   % Turquoise
    112, 128, 144;  % Slate Gray
    107, 142, 35;   % Olive Drab
    255, 99, 71;    % Tomato
    30, 144, 255;   % Dodger Blue
    244, 164, 96;   % Sandy Brown
    255, 0, 255;    % Magenta
    46, 139, 87;    % Sea Green
    ] / 255;

valSNR = [-30, -20, -16, -12, -10, -8, -4, 0, 4, 12, 20];


saveFig = false;
saveMat = false;

%% LOAD CALL STIM
femaleCallName = 'ZF001AF_DC_02';
maleCallName = 'ZF002AM_DC_01';
[femaleCall, ~] = audioread(['..\Stim\' femaleCallName '.wav']);
[maleCall, fsCall] = audioread(['..\Stim\' maleCallName '.wav']);

powerNoiseCh = 2.4091e-04;

femaleCall = (femaleCall/mean(abs(femaleCall)))*0.01;
maleCall = (maleCall/mean(abs(maleCall)))*0.01;

femaleCallPower = sum(femaleCall.^2) / length(femaleCall);
maleCallPower = sum(maleCall.^2) / length(maleCall);

maleCall = sqrt(femaleCallPower / maleCallPower) * maleCall;
maleCallPower = sum(maleCall.^2) / length(maleCall);

%% LOAD DATA
[stimSig, fsStim] = audioread(['../Subjects/' birdID '/Behavioral/' birdID '_beh_stim_' sessionNum '.mp3']);
stimLabels = table2cell(readtable(['../Subjects/' birdID '/Behavioral/' birdID '_beh_stim_' sessionNum '_labels' '.txt']));
% stimMat = load(['../Subjects/' birdID '/Behavioral/' birdID '_beh_reponse_' sessionNum '.mat']);

% [responseSig, fsResponse] = audioread(['../Subjects/' birdID '/Behavioral/' birdID '_beh_response_' sessionNum '.wav']);
reponseLabels = table2cell(readtable(['../Subjects/' birdID '/Behavioral/' birdID '_beh_response_' sessionNum '_labels' '.txt']));

tVec = linspace(0,(length(stimSig)/fsStim),length(stimSig));

%% EXTRACT LABELS

eventTimes = cell2mat(stimLabels(:,1:2));
eventNoiseType = stimLabels(:,3);
eventSNR = strrep(stimLabels(:,4), 'dB', '');
eventSex = stimLabels(:,5);


a = cell(length(reponseLabels),1);
for i = 1:length(reponseLabels)
    a{i} = reponseLabels{i,3};
end

b = cell(length(reponseLabels),1);
for i = 1:length(reponseLabels)
    b{i} = [reponseLabels{i,1:2}];
end

mask = strcmp(a, 'dc');
dcTimes = cell2mat(b(mask));

mask = strcmp(a, 'tet');
tetTimes = cell2mat(b(mask));

mask = strcmp(a, 'song');
songTimes = cell2mat(b(mask));

%% TRIALS
timeAfterEvent = 3;

trials = struct('event_Num', {}, 'event_ID', {}, 'start_time', {}, ...
    'noise_type', {}, 'snr', {}, 'real_snr', {}, 'final_SNR', {}, 'sex', {}, ...
    'end_time', {}, 'stim_sig', {}, 'train_dc', {}, 'num_dc', {}, ...
    'train_tet', {}, 'num_tet', {}, 'train_song', {}, 'num_song', {});

for i = 1:length(eventTimes)

    trials(i).event_Num = i;
    trials(i).start_time = eventTimes(i, 1);
    trials(i).end_time = eventTimes(i, 2);

    trials(i).noise_type = eventNoiseType(i);
    trials(i).snr = eventSNR(i);
    trials(i).SNR = str2double(trials(i).snr{1}); %numeric one
    trials(i).sex = eventSex(i);

    % trials(i).stim_sig = stimSig(floor(trials(i).start_time*fsStim):floor(trials(i).end_time*fsStim));
    % f = 220.2291666666667
    % m = 221.895833333333
    jj = 577;
    switch trials(i).sex{1}
        case 'm'
            trials(i).stim_sig = stimSig(floor(trials(i).start_time*fsStim)+jj:floor((trials(i).start_time+0.221895833333333)*fsStim)-1+jj);
            trials(i).scaling_factor = sqrt(powerNoiseCh / maleCallPower * 10^(trials(i).SNR / 10));
            tmp2 = trials(i).stim_sig - trials(i).scaling_factor * maleCall;
            tmp1 = trials(i).scaling_factor * maleCall;
            trials(i).real_snr = 10 * log(sqrt((sum(tmp1.^2) / length(tmp1)) / (sum(tmp2.^2) / length(tmp2))));

        case 'f'
            trials(i).stim_sig = stimSig(floor(trials(i).start_time*fsStim)+jj:floor((trials(i).start_time+0.2202291666666667)*fsStim)-1+jj);
            trials(i).scaling_factor = sqrt(powerNoiseCh / femaleCallPower * 10^(trials(i).SNR / 10));
            tmp2 = trials(i).stim_sig - trials(i).scaling_factor * femaleCall;
            tmp1 = trials(i).scaling_factor * femaleCall;
            trials(i).real_snr = 10 * log(sqrt((sum(tmp1.^2) / length(tmp1)) / (sum(tmp2.^2) / length(tmp2))));
    end

    [~, idx] = min(abs(valSNR - trials(i).real_snr));
    trials(i).final_SNR = valSNR(idx);

    mask = (dcTimes(:,1) > eventTimes(i,1)) & (dcTimes(:,1) < eventTimes(i,1) + timeAfterEvent);
    trials(i).num_dc = length(dcTimes(mask, 1));
    trials(i).train_dc = [dcTimes(mask, 1)-eventTimes(i) dcTimes(mask, 2)-eventTimes(i)]';

    mask = (tetTimes(:,1) > eventTimes(i)) & (tetTimes(:,1) < eventTimes(i) + timeAfterEvent);
    trials(i).num_tet = length(tetTimes(mask, 1));
    trials(i).train_tet = [tetTimes(mask, 1)-eventTimes(i) tetTimes(mask, 2)-eventTimes(i)]';

    mask = (songTimes > eventTimes(i)) & (songTimes < eventTimes(i) + timeAfterEvent);
    trials(i).num_song = length(songTimes(mask));
    trials(i).train_song = (songTimes(mask) - eventTimes(i))';

    switch [trials(i).sex{1} trials(i).noise_type{1} num2str(trials(i).final_SNR)]
        %female Colored
        case ['f'	'Colored'	'20']
            trials(i).event_ID = 1;
        case ['f'	'Colored'	'12']
            trials(i).event_ID = 2;
        case ['f'	'Colored'	'8']
            trials(i).event_ID = 3;
        case ['f'	'Colored'	'4']
            trials(i).event_ID = 4;
        case ['f'	'Colored'	'0']
            trials(i).event_ID = 5;
        case ['f'	'Colored'	'-4']
            trials(i).event_ID = 6;
        case ['f'	'Colored'	'-8']
            trials(i).event_ID = 7;
        case ['f'	'Colored'	'-10']
            trials(i).event_ID = 8;
        case ['f'	'Colored'	'-12']
            trials(i).event_ID = 9;
        case ['f'	'Colored'	'-16']
            trials(i).event_ID = 10;
        case ['f'	'Colored'	'-20']
            trials(i).event_ID = 11;
        case ['f'	'Colored'	'-30']
            trials(i).event_ID = 12;
        %male Colored
        case ['m'	'Colored'	'20']
            trials(i).event_ID = 13;
        case ['m'	'Colored'	'12']
            trials(i).event_ID = 14;
        case ['m'	'Colored'	'8']
            trials(i).event_ID = 15;
        case ['m'	'Colored'	'4']
            trials(i).event_ID = 16;
        case ['m'	'Colored'	'0']
            trials(i).event_ID = 17;
        case ['m'	'Colored'	'-4']
            trials(i).event_ID = 18;
        case ['m'	'Colored'	'-8']
            trials(i).event_ID = 19;
        case ['m'	'Colored'	'-10']
            trials(i).event_ID = 20;
        case ['m'	'Colored'	'-12']
            trials(i).event_ID = 21;
        case ['m'	'Colored'	'-16']
            trials(i).event_ID = 22;
        case ['m'	'Colored'	'-20']
            trials(i).event_ID = 23;
        case ['m'	'Colored'	'-30']
            trials(i).event_ID = 24;
        %female Chorus
        case ['f'	'Chorus'	'20']
            trials(i).event_ID = 25;
        case ['f'	'Chorus'	'12']
            trials(i).event_ID = 26;
        case ['f'	'Chorus'	'8']
            trials(i).event_ID = 27;
        case ['f'	'Chorus'	'4']
            trials(i).event_ID = 28;
        case ['f'	'Chorus'	'0']
            trials(i).event_ID = 29;
        case ['f'	'Chorus'	'-4']
            trials(i).event_ID = 30;
        case ['f'	'Chorus'	'-8']
            trials(i).event_ID = 31;
        case ['f'	'Chorus'	'-10']
            trials(i).event_ID = 32;
        case ['f'	'Chorus'	'-12']
            trials(i).event_ID = 33;
        case ['f'	'Chorus'	'-16']
            trials(i).event_ID = 34;
        case ['f'	'Chorus'	'-20']
            trials(i).event_ID = 35;
        case ['f'	'Chorus'	'-30']
            trials(i).event_ID = 36;
        %male Chorus
        case ['m'	'Chorus'	'20']
            trials(i).event_ID = 37;
        case ['m'	'Chorus'	'12']
            trials(i).event_ID = 38;
        case ['m'	'Chorus'	'8']
            trials(i).event_ID = 39;
        case ['m'	'Chorus'	'4']
            trials(i).event_ID = 40;
        case ['m'	'Chorus'	'0']
            trials(i).event_ID = 41;
        case ['m'	'Chorus'	'-4']
            trials(i).event_ID = 42;
        case ['m'	'Chorus'	'-8']
            trials(i).event_ID = 43;
        case ['m'	'Chorus'	'-10']
            trials(i).event_ID = 44;
        case ['m'	'Chorus'	'-12']
            trials(i).event_ID = 45;
        case ['m'	'Chorus'	'-16']
            trials(i).event_ID = 46;
        case ['m'	'Chorus'	'-20']
            trials(i).event_ID = 47;
        case ['m'	'Chorus'	'-30']
            trials(i).event_ID = 48;
    end


end

numTrials = length(trials);
% [~, idx] = sort([trials.final_SNR]);
% trials = trials(idx);

%% RESULTS
stimIDs = sort(unique([trials.event_ID]));
numStimuli = length(stimIDs);
numBins = timeAfterEvent*2;

results = struct('event_ID', {}, 'num_played', {}, 'response_dc', {}, 'response_dc_mean', {}, ...
    'response_song', {}, 'response_song_mean', {}, 'noise_type', {}, 'snr', {}, 'sex', {});

for j = 1:numStimuli
    i = stimIDs(j);
    results(j).event_ID = i;
    results(j).num_played = length([trials([trials.event_ID]==i)]);

    results(j).response_dc = histcounts([trials([trials.event_ID]==i).train_dc],linspace(0,timeAfterEvent,numBins))';
    results(j).response_dc_mean = mean([trials([trials.event_ID]==i).num_dc]);

    results(j).response_tet = histcounts([trials([trials.event_ID]==i).train_tet],linspace(0,timeAfterEvent,numBins))';
    results(j).response_tet_mean = mean([trials([trials.event_ID]==i).num_tet]);

    results(j).response_song = histcounts([trials([trials.event_ID]==i).train_song],linspace(0,timeAfterEvent,numBins))';
    results(j).response_song_mean = mean([trials([trials.event_ID]==i).num_song]);

    results(j).noise_type = trials([trials(:).event_ID] == i).noise_type;
    % results(j).noise_type = results(i).noise_type{1};
    results(j).snr = trials([trials(:).event_ID] == i).final_SNR;
    results(j).sex = trials([trials(:).event_ID] == i).sex;
    % results(j).sex = results(i).sex{1};

end

[~, idx] = sort([results.snr]);
results = results(idx);

%% PLOTTING WITH TIME
% yylimm = [0 7];
% movMeanWin = 2;
%
% figure;
% set(gcf, 'WindowState', 'maximized');
%
% subplot(121)
% idx = strcmp({results.noise_type}, 'Colored');
% allResponseDC = [results(idx).response_dc];
% allSNR = [results(idx).snr];
% hold on;
% for i = 1:length(results)/2
%     plot(linspace(0,timeAfterEvent,numBins-1), movmean(allResponseDC(:,i),movMeanWin), 'LineWidth', 2);
%     labelsLegend{i} = sprintf('SNR = %d', allSNR(i));
% end
% hold off;
%
% legend(labelsLegend);
% ylabel('Number of DC Responses');
% xlabel('Time [s]');
% ylim(yylimm)
% title('Colored Noise');
%
% subplot(122)
% idx = strcmp({results.noise_type}, 'Chorus');
% allResponseDC = [results(idx).response_dc];
% allSNR = [results(idx).snr];
% hold on;
% for i = 1:length(results)/2
%     plot(linspace(0,timeAfterEvent,numBins-1), movmean(allResponseDC(:,i),movMeanWin), 'LineWidth', 2);
% end
% hold off;
%
% legend(labelsLegend);
% ylabel('Number of DC Responses');
% xlabel('Time [s]');
% ylim(yylimm)
% title('Chorus Noise');
%
% if saveFig
%     saveas(gcf, 'results/Beh_Time.png');
% end

%% RASTER PLOT GROUPED BY STIMULUS
meanLenDC = mean(dcTimes(:,2) - dcTimes(:,1));
meanLenTet = mean(tetTimes(:,2) - tetTimes(:,1));


figure;
t = tiledlayout(2, 2);
set(gcf, 'WindowState','maximized', 'Color', 'w');


nexttile
idx = strcmp([trials.noise_type], {'Colored'});
trials_Col = trials(idx);
% [~, idx] = sort([trials_Col.noise_snr]);
% trials_Col = trials_Col(idx);
hold on;
for i = 1:length(trials_Col)
    %This is for continued labeling
    xTmp = trials_Col(i).train_dc(1,:);
    yTmp = i * ones(size([xTmp;xTmp]));
    plot([xTmp;xTmp+meanLenDC], yTmp, 'LineWidth', 1.5, 'Color', colorPalette(valSNR == trials_Col(i).final_SNR,:));

    % xTmp = trials_Col(i).train_dc;
    % yTmp = i * ones(size(xTmp));
    % plot(xTmp, yTmp, 'LineWidth', 1.5, 'Color', colorPalette(trials_Col(i).event_ID,:));
end

box1=[0 0 .2 .2];
boxy=[0 length(trials_Col)+1 length(trials_Col)+1 0];
patch(box1,boxy,[1 0 0],'FaceAlpha',0.5)
xlim([0 2]);
ylim([0 length(trials_Col)+1])
ylabel('Trial');
xlabel('Time [s]')

%%%%
tmp = 1:length(valSNR);
tmpPlots = cell(1,length(tmp));
for i = 1:length(tmp)
    tmpPlots{i} = plot(NaN, NaN, 'color', colorPalette(tmp(i),:), 'LineWidth', 4);
end
%%%%
labelsLegend = cell(length(valSNR), 1);
for j = 1:length(labelsLegend)
    labelsLegend{j} = sprintf('SNR = %d', valSNR(j));
end

hold off;
legend([tmpPlots{:}],labelsLegend')
title('DC Response in Colored Noise');

nexttile
idx = strcmp([trials.noise_type], {'Chorus'});
trials_Cho = trials(idx);
% [~, idx] = sort([trials_Cho.noise_snr]);
% trials_Cho = trials_Cho(idx);
hold on;
for i = 1:length(trials_Cho)
    xTmp = trials_Cho(i).train_dc(1,:);
    yTmp = i * ones(size([xTmp;xTmp]));
    plot([xTmp;xTmp+meanLenDC], yTmp, 'LineWidth', 1.5, 'Color', colorPalette(valSNR == trials_Cho(i).final_SNR,:));

    % xTmp = trials_Cho(i).train_dc;
    % yTmp = i * ones(size(xTmp));
    % plot(xTmp, yTmp, 'LineWidth', 1.5, 'Color', colorPalette(trials_Cho(i).event_ID,:));
end
tmp = str2double(unique([trials_Cho.snr]));
labelsLegend = cell(length(unique([trials_Cho.snr])), 1);
for j = 1:length(unique([trials_Cho.snr]))
    labelsLegend{j} = sprintf('SNR = %d', tmp(j));
end
box1=[0 0 .2 .2];
boxy=[0 length(trials_Cho)+1 length(trials_Cho)+1 0];
patch(box1,boxy,[1 0 0],'FaceAlpha',0.5)
xlim([0 2]);
ylim([0 length(trials_Cho)+1])
ylabel('Trial');
xlabel('Time [s]')

%%%%
tmp = 1:length(valSNR);
tmpPlots = cell(1,length(tmp));
for i = 1:length(tmp)
    tmpPlots{i} = plot(NaN, NaN, 'color', colorPalette(tmp(i),:), 'LineWidth', 4);
end
%%%%

hold off;
% legend([tmpPlots{:}],labelsLegend')
title('DC Response in Chorus Noise');

if saveFig
    saveas(gcf, 'results/Beh_Raster.png');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TET
nexttile
idx = strcmp([trials.noise_type], {'Colored'});
trials_Col = trials(idx);
% [~, idx] = sort([trials_Col.noise_snr]);
% trials_Col = trials_Col(idx);
hold on;
for i = 1:length(trials_Col)
    %This is for continued labeling
    xTmp = trials_Col(i).train_tet(1,:);
    yTmp = i * ones(size([xTmp;xTmp]));
    plot([xTmp;xTmp+meanLenTet], yTmp, 'LineWidth', 1.5, 'Color', colorPalette(valSNR == trials_Col(i).final_SNR,:));

    % xTmp = trials_Col(i).train_dc;
    % yTmp = i * ones(size(xTmp));
    % plot(xTmp, yTmp, 'LineWidth', 1.5, 'Color', colorPalette(trials_Col(i).event_ID,:));
end
tmp = str2double(unique([trials_Col.snr]));
labelsLegend = cell(length(unique([trials_Col.snr])), 1);
for j = 1:length(unique([trials_Col.snr]))
    labelsLegend{j} = sprintf('SNR = %d', tmp(j));
end
box1=[0 0 .2 .2];
boxy=[0 length(trials_Col)+1 length(trials_Col)+1 0];
patch(box1,boxy,[1 0 0],'FaceAlpha',0.5)
xlim([0 2]);
ylim([0 length(trials_Col)+1])
ylabel('Trial');
xlabel('Time [s]')

%%%%
tmp = 1:length(valSNR);
tmpPlots = cell(1,length(tmp));
for i = 1:length(tmp)
    tmpPlots{i} = plot(NaN, NaN, 'color', colorPalette(tmp(i),:), 'LineWidth', 4);
end
%%%%

hold off;
% legend([tmpPlots{:}],labelsLegend')
title('Tet Response in Colored Noise');

nexttile
idx = strcmp([trials.noise_type], {'Chorus'});
trials_Cho = trials(idx);
% [~, idx] = sort([trials_Cho.noise_snr]);
% trials_Cho = trials_Cho(idx);
hold on;
for i = 1:length(trials_Cho)
    xTmp = trials_Cho(i).train_tet(1,:);
    yTmp = i * ones(size([xTmp;xTmp]));
    plot([xTmp;xTmp+meanLenTet], yTmp, 'LineWidth', 1.5, 'Color', colorPalette(valSNR == trials_Cho(i).final_SNR,:));

    % xTmp = trials_Cho(i).train_dc;
    % yTmp = i * ones(size(xTmp));
    % plot(xTmp, yTmp, 'LineWidth', 1.5, 'Color', colorPalette(trials_Cho(i).event_ID,:));
end
tmp = str2double(unique([trials_Cho.snr]));
labelsLegend = cell(length(unique([trials_Cho.snr])), 1);
for j = 1:length(unique([trials_Cho.snr]))
    labelsLegend{j} = sprintf('SNR = %d', tmp(j));
end
box1=[0 0 .2 .2];
boxy=[0 length(trials_Cho)+1 length(trials_Cho)+1 0];
patch(box1,boxy,[1 0 0],'FaceAlpha',0.5)
xlim([0 2]);
ylim([0 length(trials_Cho)+1])
ylabel('Trial');
xlabel('Time [s]')

%%%%
tmp = 1:length(valSNR);
tmpPlots = cell(1,length(tmp));
for i = 1:length(tmp)
    tmpPlots{i} = plot(NaN, NaN, 'color', colorPalette(tmp(i),:), 'LineWidth', 4);
end
%%%%

hold off;
% legend([tmpPlots{:}],labelsLegend')
title('Tet Response in Chorus Noise');

if saveFig
    saveas(gcf, 'results/Beh_Raster.png');
end

t.TileSpacing = 'compact';
t.Padding = 'compact';
set(findall(gcf,'-property','FontSize'),'FontSize',14);
fontname('Helvetica')




%% PLOTTING
yylimm = [0 2];
xxlimm = [-20 12];

figure;
t = tiledlayout(2, 2);
set(gcf, 'WindowState','maximized', 'Color', 'w');

nexttile
hold on
idx = and(strcmp([results.noise_type], 'Colored'), strcmp([results.sex], 'm'));
plot([results(idx).snr], [results(idx).response_dc_mean], 'LineWidth', 2, 'Color', colorPalette(1,:));
idx = and(strcmp([results.noise_type], {'Colored'}), strcmp([results.sex], {'f'}));
plot([results(idx).snr], [results(idx).response_dc_mean], 'LineWidth', 2, 'Color', colorPalette(2,:));
ylabel('Mean of DC Responses');
xlabel('SNR [dB]');
ylim(yylimm)
xlim(xxlimm)
legend('Male', 'Female')
xticks([-20, -16, -12, -10, -8, -4, 0, 4, 12]);
xticklabels({'-20', '-16', '-12', '-10', '-8', '-4', '0', '4', '12'});
title('DC Response in Colored Noise');

nexttile
hold on
idx = and(strcmp([results.noise_type], {'Chorus'}), strcmp([results.sex], {'m'}));
plot([results(idx).snr], [results(idx).response_dc_mean], 'LineWidth', 2, 'Color', colorPalette(1,:));
idx = and(strcmp([results.noise_type], {'Chorus'}), strcmp([results.sex], {'f'}));
plot([results(idx).snr], [results(idx).response_dc_mean], 'LineWidth', 2, 'Color', colorPalette(2,:));
ylabel('Mean of DC Responses');
xlabel('SNR [dB]');
ylim(yylimm)
xlim(xxlimm)
xticks([-20, -16, -12, -10, -8, -4, 0, 4, 12]);
xticklabels({'-20', '-16', '-12', '-10', '-8', '-4', '0', '4', '12'});
title('DC Response in Chorus Noise');

nexttile
hold on
idx = and(strcmp([results.noise_type], {'Colored'}), strcmp([results.sex], {'m'}));
plot([results(idx).snr], [results(idx).response_tet_mean], 'LineWidth', 2, 'Color', colorPalette(1,:));
idx = and(strcmp([results.noise_type], {'Colored'}), strcmp([results.sex], {'f'}));
plot([results(idx).snr], [results(idx).response_tet_mean], 'LineWidth', 2, 'Color', colorPalette(2,:));
ylabel('Mean of Tet Responses');
xlabel('SNR [dB]');
ylim(yylimm)
xlim(xxlimm)
xticks([-20, -16, -12, -10, -8, -4, 0, 4, 12]);
xticklabels({'-20', '-16', '-12', '-10', '-8', '-4', '0', '4', '12'});
title('Tet Response in Colored Noise');

nexttile
hold on
idx = and(strcmp([results.noise_type], {'Chorus'}), strcmp([results.sex], {'m'}));
plot([results(idx).snr], [results(idx).response_tet_mean], 'LineWidth', 2, 'Color', colorPalette(1,:));
idx = and(strcmp([results.noise_type], {'Chorus'}), strcmp([results.sex], {'f'}));
plot([results(idx).snr], [results(idx).response_tet_mean], 'LineWidth', 2, 'Color', colorPalette(2,:));
ylabel('Mean of Tet Responses');
xlabel('SNR [dB]');
ylim(yylimm)
xlim(xxlimm)
xticks([-20, -16, -12, -10, -8, -4, 0, 4, 12]);
xticklabels({'-20', '-16', '-12', '-10', '-8', '-4', '0', '4', '12'});
title('Tet Response in Chorus Noise');

t.TileSpacing = 'compact';
t.Padding = 'compact';
set(findall(gcf,'-property','FontSize'),'FontSize',14);
fontname('Helvetica')

%% SAVING .mat DATA
if saveMat
    save([birdID '_Beh.mat'], 'trials', 'results', 'numBins', 'timeAfterEvent', 'colorMap', 'jitterList');
end



%%
% for i = 1:length(trials_Cho)
% 
%     figure
%     subplot(131)
%     spectrogramELM(trials_Cho(i).stim_sig, fsStim, .001, 1);
%     subplot(132)
%     spectrogramELM(trials_Cho(i).scaling_factor * femaleCall, fsStim, .001, 1);
%     subplot(133)
%     spectrogramELM(trials_Cho(i).amir, fsStim, .001, 1);
% 
%     i
%     % close all
% end

%%
% 
% stim = trials_Col(28).stim_sig;
% dc = trials_Col(28).scaling_factor * femaleCall;
% 
% % Assuming 'dc' and 'stim' are already defined in your workspace
% 
% % Compute the cross-correlation
% corr = xcorr(stim, dc);
% 
% % Find the index of the maximum correlation
% [~, maxIndex] = max(corr);
% 
% % Calculate the starting index of dc in stim
% % Note: Subtracting (length(dc) - 1) to get the zero-based starting index
% startIndex = maxIndex - length(dc);
% 
% % Ensure the startIndex is within bounds
% if startIndex < 0
%     startIndex = 0;
% elseif startIndex + length(dc) > length(stim)
%     startIndex = length(stim) - length(dc);
% end
% 
% % Extract the segment of 'stim' that corresponds to 'dc'
% segment_stim = stim(startIndex + 1 : startIndex + length(dc));
% 
% % Subtract 'dc' from the identified segment in 'stim'
% noise_segment = segment_stim - dc;
% 
% % Initialize the noise signal as a copy of stim
% noise = stim;
% 
% % Replace the identified segment with the noise in the original 'stim' to get the noise
% noise(startIndex + 1 : startIndex + length(dc)) = noise_segment;
% 
% % Display the result
% disp('Noise signal:');
% disp(noise);
% 
% 
% 
% figure
% 
% subplot(211)
% plot(stim(577:end))
% subplot(212)
% plot(dc)
% 
% linkaxes([subplot(211), subplot(212)], 'x');


%%


















