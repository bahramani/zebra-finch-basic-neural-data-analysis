function [spikes] = LoadSpikes(dataPath, session)

    tmp = readtable([dataPath 'datafile' sprintf('%03d', session) '_spk.xlsx']);

    spikes = struct('channel', {}, 'unit', {}, 'time', {}, 'PC', {}, 'waveform', {});

    sizeTmp = size(tmp);

    for i = 1:sizeTmp(1)
        spikes(i).channel = tmp{i, 1};
        spikes(i).unit = tmp{i, 2};
        spikes(i).time = tmp{i, 3}*1000;
        spikes(i).PC = tmp{i, 4:6};
        spikes(i).waveform = tmp{i, 7:end}';
    end

    valid_indices = ([spikes.unit] ~= 0);
    spikes = spikes(valid_indices);

end

