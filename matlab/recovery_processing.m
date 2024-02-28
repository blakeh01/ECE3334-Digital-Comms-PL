clc
clear
close all

%% phase noise check

[dut, Fs] = audioread("sig_processing\5kHz_tone.wav");

t = 1:(1 / Fs):length(dut)/Fs;
freq = 5000;
amp = 0.5;
ideal = amp*cos(freq*2*pi*t);


% take FFT of orginal signal
X = fft(dut);
X_ideal = fft(ideal);

fft_time_span = length(X) / Fs;
t_fft = linspace(0, fft_time_span, length(X));

subplot(2, 2, 1)
plot(t, abs(X_ideal));
xlabel('Time (s)');
ylabel('Magnitude');
title('Magnitude of Ideal Signal');

subplot(2, 2, 2)
plot(t, rad2deg(angle(X_ideal)));
xlabel('Time (s)');
ylabel('Phase Angle (rad)');
title('Phase Angle of Ideal Signal');

subplot(2, 2, 3)
plot(t_fft, abs(X));
xlabel('Time (s)');
ylabel('Magnitude');
title('Magnitude of Received Signal');

subplot(2, 2, 4)
plot(t_fft, angle(X));
xlabel('Time (s)');
ylabel('Phase Angle (rad)');
title('Phase Angle of Received Signal');