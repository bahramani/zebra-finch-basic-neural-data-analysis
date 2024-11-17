function [snr] = SNR(spikeTimes,filtSig)
    
    spikeSamples = floor(spikeTimes*30);
    spikeAmplitudes = zeros(length(spikeSamples), 1);
    for i = 1:length(spikeSamples)
        step = 15;
        if (spikeSamples(i)+step>length(filtSig)) || (spikeSamples(i)-step<0)
            step = 0;
        end
        spikeAmplitudes(i) = max(filtSig(spikeSamples(i)-step:spikeSamples(i)+step));
    end
    signalAmplitude = mean(spikeAmplitudes);
    
    noiseAmplitude = std(filtSig(setdiff(1:length(filtSig), spikeSamples)));
    
    snr = signalAmplitude / noiseAmplitude;

end

