function [S,Time,F] = spectrogramELM(song,fs,specDT, makePlot, fpass, BandwidthProduct, winsize)

if nargin < 7; winsize = .02; end
if nargin < 6; BandwidthProduct = 150; end
if nargin < 5; fpass = [500 8000]; end
if nargin < 4; makePlot = 0; end
if nargin < 3; specDT = .001; end

if round(specDT*fs*1e5) ~= round(specDT*fs*1e5) % 1e5 to deal with machine precision errors (eg round(40.0000)~=40); 
    error('Must choose winstep with integer number of bins')
end

winstep =specDT; 
Time = (winstep:winstep:(length(song)/fs))-winstep; 

% parameters for chronux spectrogram
params.Fs = fs;
params.fpass = fpass;
T = winsize; 
W = BandwidthProduct; % frequency bandwidth product
K = 1; %number of tapers
params.tapers = [T*W K];
movingwin = [winsize winstep]; 

% zeropad song by 1 windowsize so we can keep the spectrogram the right length
zpSong = [zeros(round(winsize*fs),1); song(:); zeros(round(winsize*fs),1)]; 

% calculate spectrogram using chronux function
[Szp,tzp,F]=mtspecgramc(zpSong,movingwin,params);

% recover part of spectrogram corresponding to original signal
tind_start = find(abs(tzp-winsize)<=winstep/2); 
tind = tind_start + Time/winstep;
if winstep<1
    tind = round(tind/10/winstep)*10*winstep; % to get rid of rounding from weird machine-precision errors
end
S = Szp(tind,:)';


% plotting stuff
if makePlot
    % cmap = jet; 
    % to make black background, set everything below threshold to threshold, then cmap(1,:) = zeros(1,3); % background = black
    % cmap(1,:) = zeros(1,3);
    cmap = 1/256*flipud([158,1,66;213,62,79;244,109,67;253,174,97;254,224,139;255,255,191;230,245,152;171,221,164;102,194,165;50,136,189;94,79,162;1 1 1]);%cbrewer spectral, modified
    % cmap = flipud(gray); 
    colormap(cmap)
    colormap(cmap);
    Plot = 10*log10(S+eps);
    Plot(Plot(:)<prctile(Plot(:),50)) = prctile(Plot(:),50);
    imagesc(Time,F/1000,Plot); axis tight; 
    set(gca, 'ydir', 'normal')
%     surf(Time, F/1000, Plot,'edgecolor','none'); axis tight; view(0,90);
    ylabel('Frequency [kHz]'); 
    xlabel('Time [s]')
    % clim([-100 -60])
    % colorbar
    shg
end