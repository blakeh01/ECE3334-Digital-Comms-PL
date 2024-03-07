"""
    QPSK example of pulse-trains being shaped using RRC and demodulated through m&m clock recovery and costas loop.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

import tools.eye_diagram

# => DATA GENERATION - PULSE TRAINS AND PULSE SHAPING

Fs = 48000  # Sampling rate
Tsym = 1 / 4800  # Symbol period
L = int(Fs * Tsym)  # Up-sampling rate, L samples per symbol
f_c = 5000

# Generate random bits
with open('../input.txt', 'r') as file:
    text = file.read()

ascii_bits = np.unpackbits(np.array([ord(c) for c in text], dtype=np.uint8))
sync_symbols = np.ones(100)  # 100 sync symbols
sync_bits = np.random.randint(0, 2, 500)   # 500 random alternating 1's and 0's (todo: change, chance @ getting preamble)

# Combine sync symbols, sync bits, and data preamble
sync_sequence = np.concatenate((sync_symbols, sync_bits, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))
print(sync_sequence)

# Combine sync sequence and ASCII bits
all_bits = np.concatenate((sync_sequence, ascii_bits))

num_symbols = len(all_bits) // 2

pulse_train_I = np.array([])
pulse_train_Q = np.array([])

for i in range(0, len(all_bits), 2):
    # Map the pair of bits to I and Q channels
    bit_I = all_bits[i] * 2 - 1
    bit_Q = all_bits[i + 1] * 2 - 1

    # Create pulses for I and Q channels
    pulse_I = np.zeros(L)
    pulse_Q = np.zeros(L)

    pulse_I[0] = bit_I
    pulse_Q[0] = bit_Q

    pulse_train_I = np.concatenate((pulse_train_I, pulse_I))
    pulse_train_Q = np.concatenate((pulse_train_Q, pulse_Q))

# Combining I and Q channels to get QPSK signal
sig = pulse_train_I + 1j * pulse_train_Q

plt.figure(0)
plt.subplot(1, 2, 1)
plt.stem(np.real(sig))
plt.title("Generated I")
plt.subplot(1, 2, 2)
plt.stem(np.imag(sig))
plt.title("Generated Q")
plt.grid(True)
plt.show()

# Create our raised-cosine filter
num_taps = 101
beta = 0.6
t = np.arange(num_taps) - (num_taps - 1) // 2
h_rcc = np.sinc(t / L) * np.cos(np.pi * beta * t / L) / (1 - (2 * beta * t / L) ** 2)
plt.figure(1)
plt.plot(t, h_rcc, '.')
plt.title("RRC Filter Response")
plt.grid(True)
plt.show()

# Match filter both I and Q

# RRC Matched Filter for I channel
samples_I = np.convolve(sig.real, h_rcc, 'full')

# RRC Matched Filter for Q channel
samples_Q = np.convolve(sig.imag, h_rcc, 'full')

plt.figure(2)

plt.subplot(1, 2, 1)
plt.plot(samples_I, '.-')
for i in range(num_symbols):
    plt.plot([i * L + num_taps // 2, i * L + num_taps // 2], [0, samples_I[i * L + num_taps // 2]])
plt.grid(True)
plt.title("RRC Filtered I")

plt.subplot(1, 2, 2)
plt.plot(samples_Q, '.-')
for i in range(num_symbols):
    plt.plot([i * L + num_taps // 2, i * L + num_taps // 2], [0, samples_Q[i * L + num_taps // 2]])
plt.grid(True)
plt.title("RRC Filtered Q")
plt.show()

# create I + Q

# add carrier wave
t = np.linspace(0, len(samples_I) / Fs, len(samples_I))
carrier_I = np.cos(2 * np.pi * f_c * t)  # Carrier wave for I component
carrier_Q = np.sin(2 * np.pi * f_c * t)  # Carrier wave for Q component

qpsk_signal = samples_I * carrier_I + samples_Q * carrier_Q

# cap volume to 1
adj_samples = np.divide(qpsk_signal, np.max(qpsk_signal))
attenuation_factor = 1 / np.max(qpsk_signal)  # plus other factors ie. volume?

print(np.shape(adj_samples))
print("Attenuated signal by: ", attenuation_factor)
wavfile.write('QPSK_modulated_wave.wav', Fs, adj_samples)
print("Wrote .wav file!")

# FFT of the samples array
fft_result = np.fft.fft(qpsk_signal)
fft_result_shifted = np.fft.fftshift(fft_result)

# frequency axis
N = len(qpsk_signal)  # Number of samples
frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / Fs))

# Plot the magnitude spectrum
plt.plot(frequencies, np.abs(fft_result_shifted))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# => BEGIN CHANNEL SIMULATION
Fs, samples = wavfile.read("erik_QPSK.wav")
print(np.shape(samples))

samples = np.divide(samples, 32767)
print(f"largest sample amp: {np.max(samples)}")

add_noise = False
gain_control = True
remove_carrier = True
coarse_freq_offset = False

assert Fs == 48000

if remove_carrier:
    # Define local oscillator components (in-phase and quadrature)
    t = np.linspace(0, len(samples) / Fs, len(samples))
    ps = 0
    local_oscillator_I = np.cos(2 * np.pi * f_c * t+ps)
    local_oscillator_Q = np.sin(2 * np.pi * f_c * t+ps)

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

    # # Apply a freq offset
    fo = 100  # simulate freq offset
    Ts = 1 / Fs  # calc sample period
    t = np.arange(0, Ts * len(samples), Ts)  # create time vector

    samples = samples * np.exp(2 * np.pi * 1j * fo * t[0:len(samples)])  # freq shift

if coarse_freq_offset:
    samples_sq = samples ** 2
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples_sq)))
    f = np.linspace(-Fs / 2.0, Fs / 2.0, len(psd))
    max_freq = f[np.argmax(psd)]

    # Shift by negative of estimated frequency
    samples = samples * np.exp(-1j * 2 * np.pi * max_freq * t / 2.0) # remember we have to divide max_freq by 2.0 because we had squared
    # Now all that's left is a small amount of freq shift, which costas loop will fix

# Plot constellation
plt.plot(np.real(samples), np.imag(samples), '.')
plt.axis([-2, 2, -2, 2])
plt.title('Raw RX Constellation Map')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.grid()
plt.show()

tools.eye_diagram.plot_eye(np.real(samples), np.imag(samples), L)

# ==> BEGIN DEMODULATION

# TIME SYNC - Mueller and muller clock recovery
interp_factor = 128
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
    mu += L + 0.1 * mm_val
    i_in += int(np.floor(mu))  # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu)  # remove the integer part of mu
    i_out += 1  # increment output index
out = out[2:i_out]  # remove the first two, and anything after i_out (that was never filled out)

_, (ax1, ax2) = plt.subplots(2, figsize=(8, 3.5))  # 7 is nearly full width
ax1.plot(np.real(sig), '.-')
ax1.plot(np.imag(sig))
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

tools.eye_diagram.plot_eye(np.real(out), np.imag(out), L)

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


def phase_detector_4(sample):
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real


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

    error = phase_detector_4(out[i])  # This is the error formula for 4th order Costas Loop (e.g. for QPSK)

    # Limit error to the range -1 to 1
    error = min(error, 1.0)
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
    freq = min(freq, 1.0)
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
plt.title('Corrected RX Constellation Map')
plt.xlabel("In-phase Component")
plt.ylabel("Quadrature Component")
plt.grid()
plt.show()


# Define QPSK constellation points
QPSK_constellation = {
    -1 - 1j: '10',
    -1 + 1j: '01',
    1 - 1j: '11',
    1 + 1j: '00',
    0 + 0j: '00'
}


# Define a function to find the nearest constellation point
def nearest_constellation_point(symbol):
    distances = {np.abs(symbol - constellation_point): constellation_point for constellation_point in QPSK_constellation}
    return distances[min(distances.keys())]


# Round the I and Q components to the nearest QPSK symbol
rounded_symbols = [nearest_constellation_point(symbol) for symbol in out]
print(rounded_symbols)

# Define the preamble
preamble = np.array([1 + 1j, 1 + 1j, 1 + 1j, -1 - 1j, -1 - 1j, -1 - 1j])

# Cross-correlation
correlation = np.correlate(rounded_symbols, preamble, mode='valid')

# Find the index where maximum correlation occurs
max_correlation_index = np.argmax(np.abs(correlation))

# Extract the preamble and following symbols
extracted_preamble_and_symbols = rounded_symbols[max_correlation_index: max_correlation_index + len(preamble)]

# Check if the extracted preamble matches the actual preamble
if np.array_equal(extracted_preamble_and_symbols, preamble):
    print("Preamble detected successfully.")
    # Extract the following symbols after the preamble
    data_symbols = rounded_symbols[max_correlation_index + len(preamble):]
    # Process the data symbols as needed
    rounded_symbols = [nearest_constellation_point(symbol) for symbol in data_symbols]

    ascii_characters = []
    for symbol in rounded_symbols:
        # Convert symbol to ASCII character
        ascii_characters.append(QPSK_constellation[symbol])

    # Concatenate ASCII characters to get the encoded string
    encoded_string = ''.join(ascii_characters)

    # Convert rounded symbols to ASCII characters
    ascii_characters = [QPSK_constellation[symbol] for symbol in rounded_symbols]

    print("Encoded string:", encoded_string)

    decoded_text = ''.join([chr(int(encoded_string[i:i + 8], 2)) for i in range(0, len(encoded_string), 8)])

    print("Decoded text:", decoded_text)
else:
    print("Preamble not found.")