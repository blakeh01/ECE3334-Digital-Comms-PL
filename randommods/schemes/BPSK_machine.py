"""
    BPSK example of pulse-trains being shaped using RRC
"""
import random

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

from tools.eye_diagram import plot_eye

# => DATA GENERATION - PULSE TRAINS AND PULSE SHAPING

num_symbols = 1200  # Number of symbols
Fs = 48000  # Sampling rate
Tsym = 1 / 4800  # Symbol period
L = int(Fs * Tsym)  # Upsampling rate, L samples per symbol
f_c = 5000

# generate random data
bits = np.random.randint(0, 2, num_symbols)
print(bits)

# oversample data
sig = np.array([])
for bit in bits:
    pulse = np.zeros(L)
    pulse[0] = bit * 2 - 1
    sig = np.concatenate((sig, pulse))

plt.figure(0)
plt.stem(sig)
plt.title("Generated BPSK")
plt.grid(True)
plt.show()

# Create our raised-cosine filter
num_taps = 101
beta = 0.35
t = np.arange(num_taps) - (num_taps - 1) // 2
h_rcc = np.sinc(t / L) * np.cos(np.pi * beta * t / L) / (1 - (2 * beta * t / L) ** 2)
plt.figure(1)
plt.plot(t, h_rcc, '.')
plt.title("RRC Filter Response")
plt.grid(True)
plt.show()

# RRC Matched Filter
samples = np.convolve(sig, h_rcc)
plt.figure(2)
plt.plot(samples, '.-')
for i in range(num_symbols):
    plt.plot([i * L + num_taps // 2, i * L + num_taps // 2], [0, samples[i * L + num_taps // 2]])
plt.grid(True)
plt.title("RRC Filtered Signal")
plt.show()

# add carrier wave
t = np.linspace(0, len(samples) / Fs, len(samples))
samples = np.multiply(samples, np.cos(2 * np.pi * f_c * t))

# cap volume to 1
adj_samples = np.divide(samples, np.max(samples))
attenuation_factor = 1 / np.max(samples)  # plus other factors ie. volume?

print(np.shape(adj_samples))
print("Attenuated signal by: ", attenuation_factor)
wavfile.write('BPSK_modulated_wave.wav', Fs, adj_samples)
print("Wrote .wav file!")

# FFT of the samples array
fft_result = np.fft.fft(samples)
fft_result_shifted = np.fft.fftshift(fft_result)

# frequency axis
N = len(samples)  # Number of samples
frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))

# Plot the magnitude spectrum
plt.plot(frequencies, np.abs(fft_result_shifted))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# => BEGIN CHANNEL SIMULATION
Fs, samples = wavfile.read("BPSK_modulated_wave.wav")

# samples = np.divide(samples, 32767)
# print(f"largest sample amp: {np.max(samples)}")

add_noise = True
gain_control = False
remove_carrier = True

assert Fs == 48000

if remove_carrier:
    # Define local oscillator components (in-phase and quadrature)
    t = np.linspace(0, len(samples) / Fs, len(samples))
    local_oscillator_I = np.cos(2 * np.pi * f_c * t)
    local_oscillator_Q = np.sin(2 * np.pi * f_c * t)

    # Perform quadrature demodulation
    demod_I = samples * local_oscillator_I * 2
    demod_Q = samples * local_oscillator_Q * 2

    # Low-pass filtering
    cutoff_frequency = 4000  # Hz
    filter_length = 101  # Filter length, make it odd

    nyquist_frequency = 0.5 * Fs  # Nyquist frequency
    cutoff_normalized = cutoff_frequency / nyquist_frequency
    h_lp = signal.firwin(filter_length, cutoff_normalized, window='hamming')

    h_lp /= np.sum(h_lp)  # unity gain

    demod_I_filtered = np.convolve(demod_I, h_lp, 'full')  # apply filter
    demod_Q_filtered = np.convolve(demod_Q, h_lp, 'full')  # apply filter

    # Combine into complex numbers
    samples = demod_I_filtered + 1j * demod_Q_filtered

    # t = np.linspace(0, len(samples) / Fs, len(samples))
    # samples = np.multiply(samples, np.cos(2 * np.pi * f_c * t))
    #
    # cutoff_frequency = 4000  # Hz
    # filter_length = 101  # Filter length, make it odd
    #
    # nyquist_frequency = 0.5 * Fs  # Nyquist frequency
    # cutoff_normalized = cutoff_frequency / nyquist_frequency
    # h_lp = signal.firwin(filter_length, cutoff_normalized, window='hamming')
    #
    # h_lp /= np.sum(h_lp)  # unity gain
    #
    # samples = np.convolve(samples, h_lp, 'full')  # apply filter
    # # samples = np.convolve(samples, h_rcc)  # apply filter
    #
    # FFT of the samples array
    fft_result = np.fft.fft(samples)
    fft_result_shifted = np.fft.fftshift(fft_result)

    # frequency axis
    N = len(samples)  # Number of samples
    frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))

    # Plot the magnitude spectrum
    plt.plot(frequencies, np.abs(fft_result_shifted))
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

if gain_control:
    samples = np.multiply(samples, 1 / attenuation_factor)

if add_noise:
    # Create and apply fractional delay filter and plot constellation maps
    delay = 0.4  # fractional delay, in samples
    N = 21  # number of taps
    n = np.arange(N)  # 0,1,2,3...
    h = np.sinc(n - (N - 1) / 2 - delay)  # calc filter taps
    h *= np.hamming(N)  # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h)  # normalize to get unity gain, we don't want to change the amplitude/power
    samples = np.convolve(samples, h)  # apply filter

    # Apply a freq offset
    fo = 100  # simulate freq offset
    Ts = 1 / Fs  # calc sample period
    t = np.arange(0, Ts * len(samples), Ts)  # create time vector

    t = t[0:len(samples)]
    samples = samples * np.exp(2 * np.pi * 1j * fo * t)  # freq shift

# Plot constellation
plt.plot(np.real(samples), np.imag(samples), '.')
plt.axis([-2, 2, -2, 2])
plt.title('Raw RX Constellation Map')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.grid()
plt.show()

plot_eye(np.real(samples), np.imag(samples), L, "Raw Eye Diagram")

# ==> BEGIN DEMODULATION

# TIME SYNC - Mueller and muller clock recovery
interp_factor = 64
samples_interpolated = signal.resample_poly(samples, interp_factor, 1)

mu = 0  # initial estimate of phase of sample
out = np.zeros(len(samples) + 10, dtype=np.complex64)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
i_in = 0  # input samples index
i_out = 2  # output index (let first two outputs be 0)
while i_out < len(samples) and i_in < len(samples):
    out[i_out] = samples_interpolated[
        i_in * interp_factor + int(mu * interp_factor)]  # grab what we think is the "best" sample
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j * int(np.imag(out[i_out]) > 0)
    x = (out_rail[i_out] - out_rail[i_out - 2]) * np.conj(out[i_out - 1])
    y = (out[i_out] - out[i_out - 2]) * np.conj(out_rail[i_out - 1])
    mm_val = np.real(y - x)
    mm_val = min(mm_val, 4.0)  # For the sake of there being less code to explain
    mm_val = max(mm_val, -4.0)
    mu += L + 0.3 * mm_val
    i_in += int(np.floor(mu))  # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu)  # remove the integer part of mu
    i_out += 1  # increment output index
out = out[2:i_out]  # remove the first two, and anything after i_out (that was never filled out)

_, (ax1, ax2) = plt.subplots(2, figsize=(8, 3.5))  # 7 is nearly full width
ax1.plot(sig, '.-')
ax2.plot(np.real(out[6:-7]), '.-')
ax2.plot(np.imag(out[6:-7]), '.-')
plt.show()

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # 7 is nearly full width
ax1.plot(np.real(samples), np.imag(samples), '.')
ax1.axis([-2, 2, -2, 2])
ax1.set_title('Before Time Sync')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
ax1.grid()

ax2.plot(np.real(out[32:-8]), np.imag(out[32:-8]), '.')  # leave out the ones at beginning, before sync finished
ax2.axis([-2, 2, -2, 2])
ax2.set_title('After Time Sync')
ax2.grid()
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.show()

plot_eye(np.real(out), np.imag(out), L, "After Timing Correction Eye Diagram")

# if True:
#     from matplotlib.animation import FuncAnimation
#
#     fig, ax = plt.subplots()
#     fig.set_tight_layout(True)
#     line, = ax.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], '.')
#     ax.axis([-2, 2, -2, 2])
#
#     # Add zeros at the beginning so that when gif loops it has a transition period
#     temp_out = np.concatenate((np.zeros(50), out))
#
#
#     def update(i):
#         print(i)
#         line.set_xdata([np.real(temp_out[i:i + 5])])
#         line.set_ydata([np.imag(temp_out[i:i + 5])])
#         return line, ax
#
#
#     anim = FuncAnimation(fig, update, frames=np.arange(0, len(out - 5)), interval=20)
#     anim.save('time-sync-constellation-animated.gif', dpi=80, writer='imagemagick')
#     exit()

# => SYMBOL TIMING RECOVERY USING COSTAS LOOP

samples = out  # copy for plotting

N = len(samples)
phase = 0
freq = 0
loop_bw = 0.05  # This is what to adjust, to make the feedback loop faster or slower (which impacts stability)
damping = np.sqrt(2.0) / 2.0  # Set the damping factor for a critically damped system
alpha = (4 * damping * loop_bw) / (1.0 + (2.0 * damping * loop_bw) + loop_bw ** 2)
beta = (4 * loop_bw ** 2) / (1.0 + (2.0 * damping * loop_bw) + loop_bw ** 2)
print("alpha:", alpha)
print("beta:", beta)
out = np.zeros(N, dtype=np.complex64)
freq_log = []
for i in range(N):
    out[i] = samples[i] * np.exp(-1j * phase)  # adjust the input sample by the inverse of the estimated phase offset

    error = np.real(out[i]) * np.imag(out[i])  # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

    # Limit error to the range -1 to 1
    error = min(error, 1.0)  # left out for sake of reducing code.  didnt seem to get anywhere near 1 or -1
    error = max(error, -1.0)

    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    freq_log.append(freq / 50.0 * Fs)  # see note at bottom
    phase += freq + (alpha * error)

    # Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
    while phase >= 2 * np.pi:
        phase -= 2 * np.pi
    while phase < 0:
        phase += 2 * np.pi

    # Limit frequency to range -1 to 1
    freq = min(freq, 1.0)  # didnt get anywhere near 1 or -1 in this example, leaving out for sake of understanding code
    freq = max(freq, -1.0)

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 5))  # 7 is nearly full width
fig.tight_layout(pad=2.0)  # add space between subplots
ax1.plot(np.real(samples), '.-')
ax1.plot(np.imag(samples), '.-')
ax1.set_title('Before Costas Loop')
ax2.plot(np.real(out), '.-')
ax2.plot(np.imag(out), '.-')
ax2.set_title('After Costas Loop')
plt.show()

fig, ax = plt.subplots(figsize=(7, 3))  # 7 is nearly full width
# For some reason you have to divide the steady state freq by 50,
#   to get the fraction of fs that the fo is...
#   and changing loop_bw doesn't matter
ax.plot(freq_log, '.-')
ax.set_xlabel('Sample')
ax.set_ylabel('Freq Offset')
plt.show()

plt.plot(np.real(out), np.imag(out), '.')
plt.axis([-2, 2, -2, 2])
plt.title('Raw RX Constellation Map')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.grid()
plt.show()

plot_eye(np.real(out), np.imag(out), L, "After Costas Loop Eye Diagram")


# ANIMATE M&M LOOP
