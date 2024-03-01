import time
import wave

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # Fs
CHUNK = 1024  # number of samples analyzed, lower value = more computationally expensive.
THRESHOLD = 32767 / 4  # 25% amplitude threshold.
RECORD_SECONDS = 5  # number of seconds to record, this will be dependent on # of symbols & data rate.
WAVE_OUTPUT_FILENAME = "output.wav"

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open input stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


def wait_for_signal():
    frames = []
    recording = False

    while True:
        data = stream.read(CHUNK)
        # Convert data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Check if the signal crosses the threshold
        if not recording and np.max(audio_data) > THRESHOLD:
            print("Signal detected. Recording...")
            recording = True
        if recording:
            frames.append(audio_data)  # Append audio_data instead of data
        if len(frames) >= int(RATE / CHUNK * RECORD_SECONDS):
            break

    # Filter out empty arrays
    frames = [frame for frame in frames if len(frame) > 0]

    # Concatenate non-empty arrays
    return np.concatenate(frames) if frames else np.array([])


print("Listening...")
broadcast = wait_for_signal()

# while True:
#     print("Listening...")
#     broadcast = wait_for_signal()
#
#     print("Recording complete! Processing...")
#     plt.plot(broadcast)
#     plt.show()

print("* Finished recording")

# Stop stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(broadcast)

print("* Audio saved to", WAVE_OUTPUT_FILENAME)
