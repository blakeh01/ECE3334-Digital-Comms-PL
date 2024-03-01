import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, butter, sosfilt, hilbert


def constellation_mapping(value_pair, tolerance=0.1):
    closest_key = None
    min_distance = float('inf')

    for key, mapped_value in constellation_map.items():
        distance = ((value_pair[0] - mapped_value[0]) ** 2 + (value_pair[1] - mapped_value[1]) ** 2) ** 0.5
        if distance < tolerance and distance < min_distance:
            closest_key = key
            min_distance = distance

    return closest_key


def mean_symbol(I, Q, Fs, T_sym):
    samples = len(I)
    samp_sym = int(T_sym * Fs)
    offset = int(.4 * samp_sym)
    start_broad = np.arange(0 + offset, samples - samp_sym, samp_sym)
    end_broad = np.arange(samp_sym - offset, samples, samp_sym)
    symbols_I = []
    symbols_Q = []
    for index, i in enumerate(start_broad):
        symbols_I.append(np.mean(I[start_broad[index]:end_broad[index]]))
        symbols_Q.append(np.mean(Q[start_broad[index]:end_broad[index]]))
    return [symbols_I, symbols_Q]


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx



# Constants
carrier_freq = 4800  # Carrier freq of IQ, must be multiple of Fs
T_sym = 1 * 10 ** -2 # Symbol time
F_sym = 1 / T_sym  # Symbol freq
pilot_len = 1  # length of the pilot in symbols
print("Carrier freq: ", carrier_freq, " | Symbol time: ", T_sym)

constellation_map = {
    '0000': [-0.6, 0.6],
    '0001': [-0.6, 0.2],
    '0011': [-0.6, -0.2],
    '0010': [-0.6, -0.6],
    '0100': [-0.2, 0.6],
    '0101': [-0.2, 0.2],
    '0111': [-0.2, -0.2],
    '0110': [-0.2, -0.6],
    '1100': [0.2, 0.6],
    '1101': [0.2, 0.2],
    '1111': [0.2, -0.2],
    '1110': [0.2, -0.6],
    '1000': [0.6, 0.6],
    '1001': [0.6, 0.2],
    '1011': [0.6, -0.2],
    '1010': [0.6, -0.6]
}

# --- .WAV READ ---
f = wave.open('wave.wav', 'rb')
data = f.readframes(f.getparams().nframes)
duration = f.getnframes() / f.getframerate()
Fs = f.getframerate()
print("Sound duration: ", duration, " Sample rate: ", Fs)

broadcast = np.frombuffer(data, dtype=np.int16)
broadcast = broadcast / 32768.0  # scale as amplitudes were represented as 16 bits.
f.close()

# --- IQ TIMING RECOVERY FROM PILOT SIGNAL ---
t = np.arange(0, duration, 1 / Fs)
phase_offset = 0

# generate and correct LO
rec_pilot = broadcast[0:int(T_sym * pilot_len * f.getframerate())]

num_cycles = 5
num_samples_per_cycle = int(f.getframerate() / carrier_freq)
total_samples = num_cycles * num_samples_per_cycle

# Select the middle part of the pilot signal with 5 oscillations
start_index = len(rec_pilot) // 2 - total_samples // 2
end_index = start_index + total_samples
rec_pilot = rec_pilot[start_index:end_index]

# Find the index of the maximum value in the selected pilot signal
max_index = np.argmax(rec_pilot)

# Convert the index to a time point
time_point_max = t[start_index:end_index][max_index]

# Generate time points for the cosine wave
t_lo = np.arange(time_point_max, duration, 1 / Fs)

gain = 1 / rec_pilot[max_index]

# Generate local oscillators
lo = rec_pilot[max_index] * np.cos(2 * np.pi * carrier_freq * (t_lo - time_point_max)) * gain
lo_90 = rec_pilot[max_index] * np.sin(2 * np.pi * carrier_freq * (t_lo - time_point_max)) * gain

if len(lo) < len(t):
    lo = np.pad(lo, (0, len(t) - len(lo)), 'constant', constant_values=(0))
    lo_90 = np.pad(lo_90, (0, len(t) - len(lo_90)), 'constant', constant_values=(0))

plt.plot(lo, '-', label='LO')
plt.plot(lo_90, label='LO 90')
plt.plot(rec_pilot, label='Pilot Signal')
plt.xlim([0, 250])
plt.legend()
plt.show()

# --- IQ RECOVERY ---
I_recovered = lo * broadcast * 2 * gain
Q_recovered = lo_90 * broadcast * 2 * gain

plt.plot(I_recovered)
plt.plot(Q_recovered)
plt.show()

# create LP filter
cutoff_freq = 100
num_taps = 11
fir_filter = firwin(num_taps, cutoff=cutoff_freq, fs=Fs, pass_zero=True)

I_filtered = lfilter(fir_filter, 1.0, I_recovered)
Q_filtered = lfilter(fir_filter, 1.0, Q_recovered)

# create windows
left = np.arange(0, duration, T_sym)
right = np.arange(T_sym, duration + T_sym, T_sym)

I_points = []
Q_points = []

for i in range(len(left)):
    start_idx = find_nearest(t, left[i])
    end_idx = find_nearest(t, right[i])

    I_points.append(np.median(I_filtered[start_idx:end_idx]))
    Q_points.append(np.median(Q_filtered[start_idx:end_idx]))


# Plot constellation map with average envelope-detected signals
ideal_I_values = [value[0] for value in constellation_map.values()]
ideal_Q_values = [value[1] for value in constellation_map.values()]

# Plot constellation map
plt.figure(figsize=(8, 8))
plt.scatter(I_points, Q_points, marker='o', color='b')
plt.scatter(ideal_I_values, ideal_Q_values, marker='o', color='r')
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Constellation Map (Average per Window)')
plt.grid(True)
plt.show()

# recovered = []
# for i, q in zip(I_list, Q_list):
#     data = constellation_mapping([i, q])
#     if data:
#         recovered.append(constellation_mapping([i, q]))
#
# ascii_characters = [chr(int(chunk, 2)) for chunk in [''.join(recovered[i:i+2]) for i in range(0, len(recovered), 2)]]
# ascii_string = ''.join(ascii_characters)

# print("Recovered ASCII String:", ascii_string)
#
# with open("data.txt", "r") as file:
#     file_content = file.read()
#
# if file_content == ascii_string:
#     print("\nASCII has been fully recovered!")
# else:
#     print("\nASCII was not recovered!")

for i in range(len(left)):
    plt.axvline(x=left[i], color='r', linestyle='--', linewidth=1)
    plt.axvline(x=right[i], color='r', linestyle='--', linewidth=1)

plt.plot(t, I_filtered, label='I Filtered', zorder=1)
plt.plot(t, Q_filtered, label='Q Filtered', zorder=1)
plt.title("FIR Filtered Signals")
plt.legend()
plt.show()
