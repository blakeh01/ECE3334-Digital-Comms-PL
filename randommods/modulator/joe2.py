import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import soundfile as sf
import sounddevice as sd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import tools.eye_diagram

# => DATA GENERATION - PULSE TRAINS AND PULSE SHAPING

Fs = 48000  # Sampling rate
Tsym = 1 / 4800  # Symbol period
L = int(Fs * Tsym)  # Up-sampling rate, L samples per symbol
f_c = 5000

def get_QPSK_from_data(data):
    ascii_bits = np.array([])

    for i in data:
        temp = [1, 1]
        if (i == 1 or i == 1):
            temp[0] = 0
        if (i == 2 or i == 3):
            temp[1] = 0
        ascii_bits = np.concatenate([ascii_bits, temp])

    sync_symbols = np.ones(100)  # 100 sync symbols
    sync_bits = np.random.randint(0, 2, 500)   # 500 random alternating 1's and 0's (todo: change, chance @ getting preamble)

    # Combine sync symbols, sync bits, and data preamble
    sync_sequence = np.concatenate((sync_symbols, sync_bits, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))
    print(sync_sequence)

    # Combine sync sequence and ASCII bits
    all_bits = np.concatenate((sync_sequence, ascii_bits))
    print(all_bits)

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


    print(pulse_train_Q)
    print(pulse_train_I)

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

    return qpsk_signal


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text File Writer')

        layout = QVBoxLayout()

        self.symbol_label = QLabel()
        layout.addWidget(self.symbol_label)
        self.symbol_label.setText("Symbol Time in millis")
        self.symbol_label.setMaximumWidth(10000)
        self.symbol_label.setMaximumHeight(100)
        self.symbol_label.setFont(QFont("Arial", 15))
        self.symbol_label.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.symbol_label.setAlignment(Qt.AlignCenter)

        self.symbol_time = QLineEdit()
        layout.addWidget(self.symbol_time)
        self.symbol_time.setMaximumWidth(10000)
        self.symbol_time.setMaximumHeight(200)
        self.symbol_time.setFont(QFont("Arial", 20))
        self.symbol_time.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.symbol_time.setAlignment(Qt.AlignCenter)

        self.text_label = QLabel()
        layout.addWidget(self.text_label)
        self.text_label.setText("Text to transmit")
        self.text_label.setMaximumWidth(10000)
        self.text_label.setMaximumHeight(100)
        self.text_label.setFont(QFont("Arial", 15))
        self.text_label.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.text_label.setAlignment(Qt.AlignCenter)

        # textbox parameters
        self.textbox = QLineEdit()
        layout.addWidget(self.textbox)
        self.textbox.setMaximumWidth(10000)
        self.textbox.setMaximumHeight(200)
        self.textbox.setFont(QFont("Arial", 20))
        self.textbox.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.textbox.setAlignment(Qt.AlignCenter)

        # button parameters
        self.button = QPushButton('QAM4')
        self.button.setMaximumWidth(10000)
        self.button.setMaximumHeight(200)
        self.button.setFont(QFont("Arial", 20))
        self.button.clicked.connect(self.qpsk_gen)
        layout.addWidget(self.button)

        # button parameters
        self.button = QPushButton('QAM16')
        self.button.setMaximumWidth(10000)
        self.button.setMaximumHeight(200)
        self.button.setFont(QFont("Arial", 20))
        self.button.clicked.connect(self.qam16_gen)
        layout.addWidget(self.button)

        # textbox
        self.message = QLabel()
        layout.addWidget(self.message)
        self.message.setText("Messages will appear here")
        self.message.setMaximumWidth(10000)
        self.message.setMaximumHeight(100)
        self.message.setFont(QFont("Arial", 15))
        self.message.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.message.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

    def qpsk_gen(self):
        text = self.textbox.text()
        symbol_time = self.symbol_time.text()
        if (symbol_time.isnumeric() == 0):
            self.message.setText("Error: Symbol time was not a number")
            return
        symbol_time = int(symbol_time)
        text = [ord(ele) for sub in text for ele in sub]
        nums = []
        print(text)
        for i in range(len(text)):
            hex = text[i]
            part1 = (hex & 0xC0) >> 6
            part2 = (hex & 0x30) >> 4
            part3 = (hex & 0x0C) >> 2
            part4 = (hex & 0x03)
            inv_part1 = ~part1 & 0x03
            inv_part2 = ~part2 & 0x03
            inv_part3 = ~part3 & 0x03
            inv_part4 = ~part4 & 0x03
            nums.append(part1)
            nums.append(part2)
            nums.append(part3)
            nums.append(part4)
        try:
            long_arr = get_QPSK_from_data(nums)
            print(nums)
            sf.write("../schemes/erik_QPSK.wav", long_arr, Fs)
            print("wrote to file")
            self.symbol_time.setText("")
            self.textbox.setText("")
            self.message.setText("Message sent")
        except Exception as e:
            self.message.setText(e)
            return

    def qam4_gen(self):
        text = self.textbox.text()
        symbol_time = self.symbol_time.text()
        if (symbol_time.isnumeric() == 0):
            self.message.setText("Error: Symbol time was not a number")
            return
        symbol_time = int(symbol_time)
        text = [ord(ele) for sub in text for ele in sub]
        nums = []
        print(text)
        for i in range(len(text)):
            hex = text[i]
            part1 = (hex & 0xC0) >> 6
            part2 = (hex & 0x30) >> 4
            part3 = (hex & 0x0C) >> 2
            part4 = (hex & 0x03)
            inv_part1 = ~part1 & 0x03
            inv_part2 = ~part2 & 0x03
            inv_part3 = ~part3 & 0x03
            inv_part4 = ~part4 & 0x03
            nums.append(part1)
            nums.append(part2)
            nums.append(part3)
            nums.append(part4)

        qpsk = get_QPSK_from_data(nums)
        print(nums)
        sf.write("4QAM.wav", qpsk, Fs)
        sd.play(qpsk, Fs)
        print("wrote to file")
        self.symbol_time.setText("")
        self.textbox.setText("")
        self.message.setText("Message sent")

    def qam16_gen(self):
        pass


if __name__ == '__main__':
    print("hello")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
