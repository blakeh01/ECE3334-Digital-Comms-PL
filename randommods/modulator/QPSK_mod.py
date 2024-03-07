import os
import numpy as np
import matplotlib as plt
import soundfile as sf
import sounddevice as sd
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

Fs = 48000
carrier_freq = Fs / 10
Tsym = 1 * 10 ** -3
Fsym = 1 / Tsym
L = int(Fs * Tsym)  # Up-sampling rate, L samples per symbol
f_c = 5000
Fs = 48000  # Sampling rate
Tsym = 1 / 4800  # Symbol period
L = int(Fs * Tsym)  # Up-sampling rate, L samples per symbol
f_c = 5000
symbol_time = 5  # time in milis
global Time_elapsed
Time_elapsed = 0


def generate_sig(amplitudes, symbol_time):
    global Time_elapsed
    t0 = np.arange(Time_elapsed, Time_elapsed + (Tsym * symbol_time), (1 / Fs))
    Time_elapsed = Time_elapsed + symbol_time * Tsym
    return amplitudes[0] * np.cos(2 * np.pi * carrier_freq * t0) + amplitudes[1] * np.sin(2 * np.pi * carrier_freq * t0)


def get_QPSK_from_data(data):
    ascii_bits = np.array([])
    for i in data:
        temp = [1, 1]
        if i == 1 or i == 2:
            temp[0] = 0
        if i == 2 or i == 3:
            temp[1] = 0
        ascii_bits = np.concatenate([ascii_bits, temp])

    sync_symbols = np.ones(100)  # 100 sync symbols
    sync_bits = np.random.randint(0, 2,
                                  500)  # 500 random alternating 1's and 0's (todo: change, chance @ getting preamble)
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
    num_taps = 101
    beta = 0.6
    t = np.arange(num_taps) - (num_taps - 1) // 2
    h_rcc = np.sinc(t / L) * np.cos(np.pi * beta * t / L) / (1 - (2 * beta * t / L) ** 2)
    # RRC Matched Filter for I channel
    samples_I = np.convolve(sig.real, h_rcc, 'full')

    # RRC Matched Filter for Q channel
    samples_Q = np.convolve(sig.imag, h_rcc, 'full')
    t = np.linspace(0, len(samples_I) / Fs, len(samples_I))
    carrier_I = np.cos(2 * np.pi * f_c * t)  # Carrier wave for I component
    carrier_Q = np.sin(2 * np.pi * f_c * t)  # Carrier wave for Q component
    qpsk_signal = samples_I * carrier_I + samples_Q * carrier_Q
    # cap volume to 1
    adj_samples = np.divide(qpsk_signal, np.max(qpsk_signal))
    attenuation_factor = 1 / np.max(qpsk_signal)  # plus other factors ie. volume?
    return adj_samples


def get_16IQ_from_data(data):
    constellation_map = {
        0: [-.6, .6],
        1: [-.6, .2],
        3: [-.6, -.2],
        2: [-.6, -.6],
        4: [-.2, .6],
        5: [-.2, .2],
        7: [-.2, -.2],
        6: [-.2, -.6],
        12: [.2, .6],
        13: [.2, .2],
        15: [.2, -.2],
        14: [.2, -.6],
        8: [.6, .6],
        9: [.6, .2],
        11: [.6, -.2],
        10: [.6, -.6],
    }
    # print(data)
    return constellation_map[data]


def get_4IQ_from_data(data):
    constellation_map = {
        0: [.6, .6],
        1: [-.6, .6],
        3: [.6, -.6],
        2: [-.6, -.6],
    }
    return constellation_map[data]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text File Writer')

        layout = QVBoxLayout()

        # self.symbol_label = QLabel()
        # layout.addWidget(self.symbol_label)
        # self.symbol_label.setText("Symbol Time in millis")
        # self.symbol_label.setMaximumWidth(10000)
        # self.symbol_label.setMaximumHeight(100)
        # self.symbol_label.setFont(QFont("Arial",15))
        # self.symbol_label.setStyleSheet("margin-left:50%; margin-right:50%;")
        # self.symbol_label.setAlignment(Qt.AlignCenter)

        # self.symbol_time = QLineEdit()
        # layout.addWidget(self.symbol_time)
        # self.symbol_time.setMaximumWidth(10000)
        # self.symbol_time.setMaximumHeight(200)
        # self.symbol_time.setFont(QFont("Arial",20))
        # self.symbol_time.setStyleSheet("margin-left:50%; margin-right:50%;")
        # self.symbol_time.setAlignment(Qt.AlignCenter)

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
        # symbol_time = self.symbol_time.text()
        # if(symbol_time.isnumeric() == 0):
        #     self.message.setText("Error: Symbol time was not a number")
        #     return
        # symbol_time = int(symbol_time)
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
            time = len(long_arr) / Fs
            sd.play(long_arr, Fs)
            print("wrote to file")
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
        signals = []
        signals.append(generate_sig([.6, 0], 10 * symbol_time))  # pilot signal
        for i in nums:
            IQ = get_4IQ_from_data(i)
            signals.append(generate_sig(IQ, symbol_time))
        long_arr = np.concatenate(signals)
        print(nums)
        # print(long_arr)
        sf.write("4QAM.wav", long_arr, Fs)
        time = len(long_arr) / Fs
        sd.play(long_arr, Fs)
        print("wrote to file")
        self.symbol_time.setText("")
        self.textbox.setText("")
        self.message.setText("Message sent")

    def qam16_gen(self):
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
            nibble1 = (hex & 0xf0) >> 4
            nibble2 = hex & 0x0f
            inv_nibble1 = ~nibble1 & 0x0f
            inv_nibble2 = ~nibble2 & 0x0f
            nums.append(nibble1)
            nums.append(nibble2)
        signals = []
        signals.append(generate_sig([.6, 0], 10 * symbol_time))  # pilot signal
        for i in nums:
            IQ = get_16IQ_from_data(i)
            signals.append(generate_sig(IQ, symbol_time))
        long_arr = np.concatenate(signals)
        print(nums)
        sf.write("16QAM.wav", long_arr, Fs)
        time = len(long_arr) / Fs
        sd.play(long_arr, Fs)
        print("wrote to file")
        self.symbol_time.setText("")
        self.textbox.setText("")
        self.message.setText("Message sent")


if __name__ == '__main__':
    print("hello")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
