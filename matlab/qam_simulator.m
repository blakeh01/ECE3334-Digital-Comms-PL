clc
clear
close all

Fs = 92000;     % 92 kHz sample freq
freq = 3000;    % 3 kHz symbol freq

% simulate 10 periods
t = 0:1/Fs:(1/freq)*10;

%% Calculate I Q waveform for given I Q amp.

I_amp = [1, 1.5];
Q_amp = [1.5, 1];

% create I and Q waveform using sin/cos as quadatrue pair.
I_sig = I_amp * sin(2*pi*freq*t);
Q_sig = Q_amp * cos(2*pi*freq*t);

% change amplitude half way through to demo constellation map
I_sig(round(numel(t) / 2):end) = I_amp(2) * sin(2*pi*freq*t(round(numel(t) / 2):end));
Q_sig(round(numel(t) / 2):end) = Q_amp(2) * cos(2*pi*freq*t(round(numel(t) / 2):end));

% sum to create QAM signal
QAM_sig = I_sig + Q_sig;

%% Plot I, Q, QAM
figure
tiledlayout(2, 2)

ax1 = nexttile;
plot(ax1, t, I_sig);
title(ax1, "I Waveform");
xlim([0 1/freq * 10])

ax2 = nexttile;
plot(ax2, t, Q_sig);
title(ax2, "Q Waveform");
xlim([0 1/freq * 10])

ax3 = nexttile([1 2]);
plot(ax3, t, QAM_sig);
title(ax3, "I+Q Waveform")
xlim([0 1/freq * 10])

%% Recover IQ from QAM signal

% trig identity makes recovered signal 1/2 of amplitude, so multiply by 2.
I_recovered = QAM_sig .* sin(2*pi*freq*t) .* 2; 
Q_recovered = QAM_sig .* cos(2*pi*freq*t) .* 2;

% steep lowpass filter to recover just DC component
I_filtered = lowpass(I_recovered, 0.001, Steepness=0.9);
Q_filtered = lowpass(Q_recovered, 0.001, Steepness=0.9);

figure
tiledlayout(2, 2)

ax4 = nexttile;
plot(ax4, t, I_recovered);
title(ax4, "Recovered I");
xlim([0 1/freq * 10]);

ax5 = nexttile;
plot(ax5, t, I_filtered);
title(ax5, "Lowpass I");
xlim([0 1/freq*10])

ax6 = nexttile;
plot(ax6, t, Q_recovered);
title(ax6, "Recovered Q");
xlim([0 1/freq * 10]);

ax7 = nexttile;
plot(ax7, t, Q_filtered);
title(ax7, "Lowpass Q");
xlim([0 1/freq*10])
