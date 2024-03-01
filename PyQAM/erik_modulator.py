import os
import sys

import numpy as np
import matplotlib as plt
import soundfile as sf
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

Fs = 48000
carrier_freq = Fs / 10
Tsym = 0.5
Fsym = 1 / Tsym
symbol_time = 1
Time_elapsed = 0


def generate_sig(amplitudes, symbol_time):
    global Time_elapsed
    t0 = np.linspace(0, Tsym * symbol_time, int(Tsym * Fs), endpoint=False)
    Time_elapsed = Time_elapsed + symbol_time * Tsym
    return amplitudes[0] * np.cos(2 * np.pi * carrier_freq * (Time_elapsed + t0)) + amplitudes[1] * np.sin(
        2 * np.pi * carrier_freq * (Time_elapsed + t0))


def get_IQ_from_data(data):
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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Text File Writer')

        layout = QVBoxLayout()

        # textbox parameters
        self.textbox = QLineEdit()
        layout.addWidget(self.textbox)
        self.textbox.setMaximumWidth(10000)
        self.textbox.setMaximumHeight(200)
        self.textbox.setFont(QFont("Arial", 20))
        self.textbox.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.textbox.setAlignment(Qt.AlignCenter)
        # button parameters
        self.button = QPushButton('Done')
        self.button.setMaximumWidth(10000)
        self.button.setMaximumHeight(200)
        self.button.setFont(QFont("Arial", 20))
        self.button.clicked.connect(self.wav_gen)

        layout.addWidget(self.button)

        self.setLayout(layout)

    def wav_gen(self):
        text = self.textbox.text()
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
            nums.append(inv_nibble1)
            nums.append(nibble2)
            nums.append(inv_nibble2)
        signals = []
        signals.append(generate_sig([1, 0], 5))
        for i in nums:
            IQ = get_IQ_from_data(i)
            signals.append(generate_sig(IQ, symbol_time))
        long_arr = np.concatenate(signals)
        sf.write("large.wav", long_arr, Fs)
        time = len(long_arr) / Fs
        print(time)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
