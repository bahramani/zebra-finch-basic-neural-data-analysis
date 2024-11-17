%% Data_Analysis

% Amirreza Bahramani
% IPM, Birds Lab
% April 2023

% This code will convert .ns6 filse and analyze the data
% But it needs the events times
% Also this code will take spike times from Plexon Offline Sorter

% Needed functions:
% LoadSpikes
% ExtractData
% PlotErrorbar
% SNR

close all
clear
clc


%% PARAMETERES
addpath(genpath('..\libraries\chronux\'));
birdID = 'ZF003AM';
mainPath = ['..\Subjects\' birdID '\Neural\'];
dataPath = [mainPath 'recordings\'];
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

winSizeMovMean = 100;
timeBeforeEvent = 500; %ms
timeAfterEvent = 1500; %ms

sessions = [4 6 8 11];

%%
for sessNum = 11 %[4 6 8 11]

    clearvars -except sessNum birdID mainPath dataPath colorPalette winSizeMovMean timeBeforeEvent timeAfterEvent

    %% LOAD RAW NEURAL DATA & EVENT DATA & SPIKE DATA
    load([dataPath birdID '_neural_' sprintf('%02d', sessNum) '.mat'])

    tVec = linspace(0,(length(neuralSig)/fs)*1000,length(neuralSig));

    % the following data should be produced by Offline Sorter
    [spikes] = LoadSpikes(dataPath, sessNum);
    numSpikes = length(spikes);
    numUnits = length(unique([spikes.unit]));

    stimData = load([mainPath 'main stim\' birdID '_neural_stim_01.mat']);

    stimInfo = stimData.stim;

    for i = 1:length(stimInfo)
        switch stimInfo(i).sex
            case 'f'
                stimInfo(i).duration = 220.2;
            case 'm'
                stimInfo(i).duration = 221.9;
        end
    end

    %% POWER SPECTRUM DENSITY
    % ov = 0.5;      % overlap, choose around 50%-75%
    % winLenS = 5; % s  % choose around 3s
    % winLen = floor(winLenS*fs); % window length in samples
    % nff = 100000;     % number of freqs
    % freqBins = linspace(0, 300, nff);
    %
    % [pd, freqs] = pwelch(LFPSig, blackman(winLen), floor(ov*winLen), freqBins, fs);
    %
    % % Plot the PSD
    % figure
    % set(gcf, 'WindowState', 'maximized');
    % plot(freqs, 10*log10(pd), 'LineWidth', 1.5)
    % xlabel('Frequency (Hz)');
    % ylabel('Power (dB)');
    % grid on
    % title('Power Spectral Density')

    % if saveFig
    %     saveas(gcf, 'neural_analysis_figure.png');
    % end

    %% FILTERING
    filtSig = highpass(neuralSig, 250, fs, ...
        'Steepness', 0.85, 'StopbandAttenuation', 60);

    LFPSig = lowpass(neuralSig, 300, fs);
    % f0 = 50;
    % Q = 60;
    % w0 = f0/(fs/2);
    % bw = w0/Q;
    % [b, a] = iirnotch(w0, bw);
    %
    % LFPSig = filtfilt(b, a, LFPSig);

    %% UNITS' DATA MEAN WAVEFORM & ISI
    units = struct('num', {}, 'times', {},'mean_waveform', {}, 'ISI', {}, 'CV', {});

    for i = 1:numUnits
        [units(i).num] = length([spikes([spikes.unit]==i).time]);
        [units(i).times] = [spikes([spikes.unit]==i).time];
        [units(i).mean_waveform] = mean([spikes([spikes.unit]==i).waveform],2);
        [units(i).ISI] = diff([spikes([spikes.unit]==i).time]);
        [units(i).CV] = std(units(i).ISI)/mean(units(i).ISI);
    end

    %% PLOTTING UNITS' MEAN WAVEFORM & ISI
    figU = figure('WindowState','maximized', 'Color', 'w');
    tl = tiledlayout(2,numUnits);
    for i = 1:numUnits
        nexttile
        tmp = [spikes([spikes.unit]==i).waveform]';
        options.handle = gcf;
        options.color_area = colorPalette(i+5, :);
        options.color_line = colorPalette(i+5, :);
        options.alpha = 0.3;
        options.line_width = 2;
        options.x_axis = linspace(0, 1, length(units(1).mean_waveform));
        options.error = 'std';
        PlotErrorbar(tmp, options)
        grid minor
        title(['Waveform of Unit ' num2str(i)])
        xlabel("Time [ms]")
        ylabel("Voltage [\muV]")

        nexttile
        x = [linspace(0, 100)];
        histogram(units(i).ISI, x, 'FaceColor', colorPalette(i+5, :));
        grid minor
        xlim([0 50])
        title(['Inter-Spike Interval of Unit ' num2str(i)])
        xlabel("ISI [ms]")
        ylabel("Counts")
    end

    tl.TileSpacing = 'compact';
    % tl.Padding = 'compact';
    set(findall(gcf,'-property','FontSize'),'FontSize',14);
    fontname('Helvetica')

    saveas(figU, [mainPath 'results\' num2str(sessNum) '\units.png'])

    close all

    %% SAVING NEURAL AS .wav
    audiowrite([mainPath 'results\' num2str(sessNum) '\' 'stim_sig.wav'], audioSig/max(audioSig), fs);

    spikeVec = cell(1, numUnits);
    for k = 1:numUnits
        spikeVec{k} = zeros(1, length(filtSig));
        tmpSpikeTimes = units(k).times*30;
        for p = 1:length(tmpSpikeTimes)
            spikeVec{k}(ceil(tmpSpikeTimes(p)):ceil(tmpSpikeTimes(p))+length(units(k).mean_waveform)-1) = units(k).mean_waveform;
        end
        spikeVec{k} = movmean(spikeVec{k}, [winSizeMovMean 0]);
        audiowrite([mainPath 'results\' num2str(sessNum) '\' 'neural_sig_' num2str(k) '.wav'], -1 + 2 * (spikeVec{k} - min(spikeVec{k})) / (max(spikeVec{k}) - min(spikeVec{k})), fs);
    end

    %% USE AUDACITY
    jitterList = zeros(1, 20);
    jitterList(4) = 7.980;
    jitterList(6) = 3.077;
    jitterList(8) = 3.720;
    jitterList(11) = 1.536;

    events = reshape(stimData.matAllSample, length(stimData.stimID), 2);
    events = events(:, [2, 1]);

    tmp = readtable([mainPath 'main stim\' birdID '_neural_stim_01_labels.txt']);
    events(:,1) = tmp{:,1}*1000 + jitterList(sessNum)*1000;

    numTrials = length(events(:,2));
    numStim = length(unique(events(:,2)));

    %% PLOTTING SIGNAL & STIMULUS WITH EVENTS
    % figure('Name','Stimulus Plot')
    % tcl = tiledlayout(2,1);
    % set(gcf, 'WindowState', 'maximized');
    %
    % subplot(2,1,1)
    % plot(tVec, audioSig, "Color", colorPalette(3,:))
    % xline(events(:,1),'Color',colorPalette(1, :),'LineWidth',1)
    % grid minor
    % xlabel('Time [ms]')
    % ylabel('Voltage [\muV]')
    % % xlim([15 20])
    % ylim([-400 400])
    % title(['Raw Stimulus Signal of Session ' num2str(sessNum)])
    %
    % subplot(2,1,2)
    % plot(tVec, filtSig, "Color", colorPalette(5,:))
    % xline(events(:,1),'Color',colorPalette(1, :),'LineWidth',1)
    % grid minor
    % xlabel('Time [ms]')
    % ylabel('Voltage [\muV]')
    % % xlim([15 20])
    % ylim([-70 70])
    % title(['Signal vs. Stimulus Plot of Session ' num2str(sessNum)])
    % linkaxes([subplot(2,1,1), subplot(2,1,2)], 'x');
    %
    % % xlim([15 20])

    %% TRIALS INFO

    trials = struct('event_time', {}, 'event_ID', {}, 'start_time', {}, ...
        'end_time', {}, 'neural_sig', {}, ...
        'LFP_sig', {}, 'LFP_s', {}, 'LFP_f', {}, ...
        'stim_sig', {}, 'spike_times', {},'num_spikes', {}, 'snr', {});

    for i = 1:numTrials
        [trials(i).event_time] = events(i,1);
        [trials(i).event_ID] = events(i,2);
        [trials(i).start_time] = trials(i).event_time - timeBeforeEvent;
        [trials(i).end_time] = trials(i).event_time + timeAfterEvent;
        [trials(i).neural_sig] = filtSig(floor(trials(i).start_time*30):floor(trials(i).start_time*30)+(timeBeforeEvent+timeAfterEvent)*30-1);
        [trials(i).LFP_sig] = LFPSig(floor(trials(i).start_time*30):floor(trials(i).start_time*30)+(timeBeforeEvent+timeAfterEvent)*30-1);
        [trials(i).LFP_s, trials(i).LFP_f] = cwt(trials(i).LFP_sig, fs, FrequencyLimits=[1 250]);
        [trials(i).stim_sig] = audioSig(floor(trials(i).start_time*30):floor(trials(i).start_time*30)+(timeBeforeEvent+timeAfterEvent)*30-1);

        for j = 1:numUnits
            tmpTimes = units(j).times;
            %these times are locked to event start time
            [trials(i).spike_times{j}] = tmpTimes(tmpTimes > trials(i).start_time & tmpTimes < trials(i).end_time) - trials(i).start_time;
            [trials(i).num_spikes{j}] = length(trials(i).spike_times{j});
        end

        [trials(i).snr] = SNR(cell2mat(trials(i).spike_times), trials(i).neural_sig);

        disp(['Trial ', num2str(i), ' out of  ', num2str(numTrials), ' created'])
    end

    % [~, sortedIndices] = sort([trials.event_ID]);
    % trialsSorted = trials(sortedIndices);

    %% RESPONSES
    response = struct('event_ID', {}, 'num_event_played', {}, 'raster', {}, ...
        'PSTH', {}, 'PSTH_single_unit', {}, ...
        'mean_LFP', {}, 'stim_sig', {}, 'LFP_s', {}, 'LFP_f', {}, 'mean_PSTH', {});

    for i = 1:numStim
        [response(i).event_ID] = i;
        [response(i).num_event_played] = length(trials([trials.event_ID] == i));

        tmpSpikeMat = cell(1, numUnits);
        for h = 1:numUnits
            tmpSpikeMat{h} = zeros(response(i).num_event_played, timeBeforeEvent+timeAfterEvent);
        end
        tmpAllSpikeTimes = {trials([trials.event_ID] == i).spike_times};
        for j = 1:response(i).num_event_played
            for k = 1:numUnits
                tmpSpikeTimes = tmpAllSpikeTimes{1,j}{1,k};
                for p = 1:length(tmpSpikeTimes)
                    tmpSpikeMat{k}(j, ceil(tmpSpikeTimes(p))) = 1;
                end
            end
        end
        [response(i).raster] = tmpSpikeMat;

        [response(i).PSTH] = cell(1, numUnits);
        for w = 1:numUnits
            response(i).PSTH{w} = movmean((sum(response(i).raster{w})/response(i).num_event_played)*1000, [winSizeMovMean 0]);
        end
        [response(i).PSTH_single_unit] = sum(cell2mat(response(i).PSTH.'));

        [response(i).mean_LFP] = mean(reshape([trials([trials.event_ID]==i).LFP_sig]',[length(trials(1).LFP_sig), response(i).num_event_played]),2);

        [response(i).stim_sig] = trials([trials.event_ID] == i).stim_sig;

        [response(i).LFP_s] = zeros(size(trials(1).LFP_s));
        [response(i).LFP_f] = trials(1).LFP_f;
        tmpAllLFPSigs = trials([trials.event_ID] == i);
        for j = 1:response(i).num_event_played
            response(i).LFP_s = response(i).LFP_s + abs(tmpAllLFPSigs(j).LFP_s);
        end
        response(i).LFP_s = response(i).LFP_s / response(i).num_event_played;

        for w = 1:numUnits
            response(i).mean_PSTH{w} = mean(response(i).PSTH{w});
        end

        [response(i).mean_res] = cell(1, numUnits);
        for w = 1:numUnits
            tmp = response(i).raster{w};
            tmp = tmp(:, timeBeforeEvent:timeBeforeEvent+999);
            response(i).mean_res{w} = mean(sum(tmp, 2));
        end

        disp(['Response ', num2str(i), ' out of  ', num2str(numStim), ' created'])

    end

    %% SAVING
    save([mainPath 'results\' num2str(sessNum) '\' birdID '_neural_' sprintf('%02d', sessNum) '_processed_data.mat'], ...
        'response', 'numStim', 'fs', 'numUnits', 'stimInfo')

    clear("audioSig", "filtSig", "LFPSig", "neuralSig", "trials")

    %% LOADING

    dataPr = cell(1, length(sessions));
    for i = 1:length(sessions)
        sessNum = sessions(i);
        dataPr{i} = load([mainPath 'results\' num2str(sessNum) '\' birdID '_neural_' sprintf('%02d', sessNum) '_processed_data.mat']);
    end

    %% PLOTTING RESPONSES BY STIMULUS

    tVec = -timeBeforeEvent:1:timeAfterEvent-1;
    tVecStim = linspace(-timeBeforeEvent, timeAfterEvent, 30*length(tVec));

    for stimulilID = 1:1 %numStim
        figN = figure('WindowState','maximized', 'Color', 'w');
        tl = tiledlayout(2, 2);

        nexttile(tl, 1)
        spectrogramELM(response(stimulilID).stim_sig,fs,.001, 1);
        % axis xy
        clim([-20 10])
        xlabel('Time [ms]')
        ylabel('Frequency [Hz]')
        title('Stimulus Spectrogram')

        nexttile(tl, 2)
        hold on;
        for k = 1:numUnits
            spikeMat = response(stimulilID).raster{k};
            for trial = 1:size(spikeMat, 1)
                spikeIndices = find(spikeMat(trial, :) == 1)-timeBeforeEvent;
                for i = 1:length(spikeIndices)
                    x = [spikeIndices(i), spikeIndices(i)];
                    y = [trial - 0.4, trial + 0.4];
                    plot(x, y, 'LineWidth', 1, 'Color', colorPalette(k+5, :));
                end
            end
        end
        xlabel('Time');
        ylabel('Trial');
        ylim([0 size(spikeMat, 1)+1])
        xlim([-timeBeforeEvent timeAfterEvent])
        xline(0,'-','Onset', 'Color', '#A2142F', 'LineWidth',2)
        xline(stimInfo(stimulilID).duration, '-','DC ends', 'LineWidth', 2, 'Color', "#A2142F")
        title('Raster Plot');

        ax1 = nexttile(tl, 3);
        surface(tVecStim, response(stimulilID).LFP_f, abs(response(stimulilID).LFP_s))
        axis tight
        shading flat
        colormap(ax1,"parula")
        % colormap(ax1,"jet")
        xlabel("Time [ms]")
        ylabel("Frequency [Hz]")
        ylim([3 20])
        clim([0 30])
        colorbar
        xline(0,'-','Onset', 'Color', '#A2142F', 'LineWidth',2)
        xline(stimInfo(stimulilID).duration, '-','DC ends', 'LineWidth', 2, 'Color', "#A2142F")
        title('LFP Response')

        nexttile(tl, 4)
        hold on
        for k = 1:numUnits
            plot(tVec, response(stimulilID).PSTH{k}, 'Color', colorPalette(k+5, :), 'LineWidth', 2)
        end
        xline(0,'-','Onset', 'Color', '#A2142F', 'LineWidth',2)
        xline(stimInfo(stimulilID).duration, '-','DC ends', 'LineWidth', 2, 'Color', "#A2142F")
        xlabel('Time [ms]')
        ylabel('Firing Rate [Hz]')
        grid minor
        % ylim([0 max(cell2mat([response(:).PSTH]))+3])
        ylim([0 30])
        legend("Unit 1", "Unit 2")
        title('PSTH')

        title(tl, [stimInfo(stimulilID).noise_type ' ' ...
            num2str(stimInfo(stimulilID).SNR) 'dB ' stimInfo(stimulilID).sex], ...
            'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Helvetica')

        tl.TileSpacing = 'compact';
        % tl.Padding = 'compact';
        set(findall(gcf,'-property','FontSize'),'FontSize',14);
        fontname('Helvetica')

        saveas(figN, [mainPath 'results\' num2str(sessNum) '\' ...
            sprintf('%02d', stimInfo(stimulilID).SNR+abs(min([stimInfo(:).SNR]))) ...
            ' ' stimInfo(stimulilID).noise_type ' ' stimInfo(stimulilID).sex '.png'])

        close all

    end

    disp(['Session ', num2str(sessNum), ' is done!'])

    %% NEUROMETRIC

end

%% POSTER

figure3 = figure('WindowState','maximized', 'Color', 'w');
tl = tiledlayout(9, 4);
count = 0;
for sess = 2:3
    for cellnum = 1:2
        count = count + 1;
        spikeMatAll = [];

        valSNR = unique([dataPr{1}.stimInfo.SNR]);
        valSNR = valSNR(3:end);
        matSNR = repelem(valSNR, 12);

        colID = 1:length(valSNR);

        tmp = dataPr{sess}.response;
        for j = 1:7
            spikeMatAll = [spikeMatAll ; tmp(j).raster{cellnum}];

        end

        [sortedSNR, sortIdx] = sort(matSNR);
        sortedSpikeMatAll = spikeMatAll(sortIdx, :);

        if count == 4
            break
        end

        nexttile(tl, count, [7 1])
        hold on;

        for trial = 1:size(spikeMatAll, 1)
            spikeIndices = find(spikeMatAll(trial, :) == 1)-timeBeforeEvent;
            for i = 1:length(spikeIndices)
                x = [spikeIndices(i), spikeIndices(i)];
                y = [trial - 0.8, trial + 0.8];
                plot(x, y, 'LineWidth', 1, 'Color', colorPalette(valSNR == sortedSNR(trial),:));
            end
        end
        if count == 1
            % xlabel('Time [ms]');
            ylabel('Trial');
        end
        ylim([0 size(spikeMatAll, 1)+1])
        xlim([-timeBeforeEvent timeAfterEvent])
        xlim([-200 700])

        xline(0,'-','Onset', 'Color', '#A2142F', 'LineWidth',2)
        xline(220.200000000000, '-','DC ends', 'LineWidth', 2, 'Color', "#A2142F")
        title(['Sample Neuron #' num2str(count)]);

        if count == 1
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
            legend([tmpPlots{:}],flip(labelsLegend)')

        end


        nexttile(tl, 28+count, [2 1])
        hold on;
        for kk = 1:7
            tmp = dataPr{sess}.response(kk).PSTH{cellnum};
            plot(linspace(-timeBeforeEvent, timeAfterEvent, length(tmp)), tmp, 'LineWidth', 1.5, 'Color', colorPalette(kk,:))
        end
        ylim([0 30])
        xlim([-200 700])
        if count == 1
            xlabel('Time [ms]');
        end
        xline(0,'-','Onset', 'Color', '#A2142F', 'LineWidth',2, "LabelOrientation", "horizontal")
        xline(220.200000000000, '-','DC ends', 'LineWidth', 2, 'Color', "#A2142F", "LabelOrientation", "horizontal")
    end
end


PSTHall = zeros(7, 2000);

for sess = 1:4
    for k = 1:7
        PSTHall(k, :) = PSTHall(k, :) + dataPr{sess}.response(k).PSTH_single_unit/2;
    end
end

PSTHall = PSTHall/4;

nexttile(tl, 32, [2 1])
hold on;
for kk = 1:7
    plot(linspace(-timeBeforeEvent, timeAfterEvent, length(PSTHall(kk, :))), PSTHall(kk, :), 'LineWidth', 1.5, 'Color', colorPalette(kk,:))
end
% ylim([0 30])
xlim([-200 700])
title('Population PSTH (8 neurons)');
xline(0,'-','Onset', 'Color', '#A2142F', 'LineWidth',2, "LabelOrientation", "horizontal")
xline(220.200000000000, '-','DC ends', 'LineWidth', 2, 'Color', "#A2142F", "LabelOrientation", "horizontal")


tl.TileSpacing = 'compact';
tl.Padding = 'compact';
set(findall(gcf,'-property','FontSize'),'FontSize',14);
fontname('Helvetica')




%%
% figure
% for i = 1:7
%
%     spectrogramELM(dataPr{1}.response(i).stim_sig,dataPr{1}.fs,.001, 1);
%     clim([-20 10])
%     xlim([0.3 1.2])
%     xticks([])
%     set(gca, 'XTickLabel', [], 'YTickLabel', []);
%     xlabel('');
%     ylabel('');
%     set(gca, 'LooseInset', get(gca, 'TightInset'));
%     saveas(gcf, ['spec' num2str(i) '.emf']);
% end

%% Neurometric

neuromet = zeros(8*12, 7);
count = 0;

for sn = 1:7
    count = 0;
    for sess = 1:4
        for unit = 1:2
            tmp = dataPr{sess}.response(sn).raster{unit};
            for trl = 1:12
                count = count + 1;
                neuromet(count, sn) = sum(tmp(trl, 500:1000));
            end
        end
    end
end
neuromet = fliplr(neuromet);


%%

% Create the error bar plot
figure;
% tl = tiledlayout(2, 2);
set(gcf, 'WindowState','maximized', 'Color', 'w');



bandLFP = [4 30];
bandIdx = find(dataPr{1}.response(1).LFP_f >= bandLFP(1) & dataPr{1}.response(1).LFP_f <= bandLFP(2));

LFPmet = zeros(4, 7);
for sn = 1:7
    for sess = 1:4
        LFPmet(sess, sn) = mean(mean(dataPr{sess}.response(sn).LFP_s(bandIdx, :)));
    end
end
LFPmet = fliplr(LFPmet);

mean_values = mean(LFPmet);
std_error = std(LFPmet) / sqrt(size(LFPmet, 1));

% nexttile(tl, 1)
yyaxis left;
errorbar(mean_values, std_error, 'Color', colorPalette(end-4, :), LineWidth=2);
ylabel('LFP Power in 3 to 30 Hz Band');
xlim([0.5 7.5])
xticks(1:7);
xticklabels({'-12', '-10', '-8', '-4', '0', '4', '8'});
set(gca, 'YColor', colorPalette(end-4, :));

mean_values = mean(neuromet);
std_error = std(neuromet) / sqrt(size(neuromet, 1));

% nexttile(tl, 1)
yyaxis right;
errorbar(mean_values, std_error, 'Color', colorPalette(end, :), LineWidth=2);
ylabel('Spike count from 0 to 500 ms after stim');
xlim([0.5 7.5])
xticks(1:7);
xticklabels({'-12', '-10', '-8', '-4', '0', '4', '8'});
title('Neurometric Function');
xlabel('SNR [dB]')
set(gca, 'YColor', colorPalette(end, :));


% tl.TileSpacing = 'compact';
% tl.Padding = 'compact';
set(findall(gcf,'-property','FontSize'),'FontSize',14);
fontname('Helvetica')







