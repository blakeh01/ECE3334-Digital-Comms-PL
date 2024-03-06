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
carrier_freq = Fs/10
Tsym = 1* 10** -3
Fsym = 1/Tsym
symbol_time = 5 #time in milis
global Time_elapsed 
Time_elapsed = 0
def generate_sig(amplitudes,symbol_time):  
    global Time_elapsed 
    t0 = np.arange(Time_elapsed, Time_elapsed+(Tsym*symbol_time), (1/Fs))
    Time_elapsed =  Time_elapsed+symbol_time*Tsym
    return amplitudes[0]* np.cos(2*np.pi*carrier_freq*t0)+ amplitudes[1]* np.sin(2*np.pi*carrier_freq*t0)
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
    #print(data)
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

        self.symbol_label = QLabel()
        layout.addWidget(self.symbol_label)
        self.symbol_label.setText("Symbol Time in millis")
        self.symbol_label.setMaximumWidth(10000)
        self.symbol_label.setMaximumHeight(100)
        self.symbol_label.setFont(QFont("Arial",15))
        self.symbol_label.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.symbol_label.setAlignment(Qt.AlignCenter)

        self.symbol_time = QLineEdit()
        layout.addWidget(self.symbol_time)
        self.symbol_time.setMaximumWidth(10000)
        self.symbol_time.setMaximumHeight(200)
        self.symbol_time.setFont(QFont("Arial",20))
        self.symbol_time.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.symbol_time.setAlignment(Qt.AlignCenter)

        self.text_label = QLabel()
        layout.addWidget(self.text_label)
        self.text_label.setText("Text to transmit")
        self.text_label.setMaximumWidth(10000)
        self.text_label.setMaximumHeight(100)
        self.text_label.setFont(QFont("Arial",15))
        self.text_label.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.text_label.setAlignment(Qt.AlignCenter)

        #textbox parameters
        self.textbox = QLineEdit()
        layout.addWidget(self.textbox)
        self.textbox.setMaximumWidth(10000)
        self.textbox.setMaximumHeight(200)
        self.textbox.setFont(QFont("Arial",20))
        self.textbox.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.textbox.setAlignment(Qt.AlignCenter)

        #button parameters
        self.button = QPushButton('QAM4')
        self.button.setMaximumWidth(10000)
        self.button.setMaximumHeight(200)
        self.button.setFont(QFont("Arial",20))
        self.button.clicked.connect(self.qam4_gen)
        layout.addWidget(self.button)

        #button parameters
        self.button = QPushButton('QAM16')
        self.button.setMaximumWidth(10000)
        self.button.setMaximumHeight(200)
        self.button.setFont(QFont("Arial",20))
        self.button.clicked.connect(self.qam16_gen)
        layout.addWidget(self.button)
        
        #textbox 
        self.message = QLabel()
        layout.addWidget(self.message)
        self.message.setText("Messages will appear here")
        self.message.setMaximumWidth(10000)
        self.message.setMaximumHeight(100)
        self.message.setFont(QFont("Arial",15))
        self.message.setStyleSheet("margin-left:50%; margin-right:50%;")
        self.message.setAlignment(Qt.AlignCenter)


        self.setLayout(layout)
    def qam4_gen(self):
        text = self.textbox.text()
        symbol_time = self.symbol_time.text()
        if(symbol_time.isnumeric() == 0):
            self.message.setText("Error: Symbol time was not a number")
            return
        symbol_time = int(symbol_time)
        text = [ord(ele) for sub in text for ele in sub]
        nums = []
        print(text)
        for i in range(len(text)):
            hex = text[i]
            part1 = (hex&0xC0)>>6
            part2 = (hex&0x30)>>4
            part3 = (hex&0x0C)>>2
            part4 = (hex&0x03)
            inv_part1 = ~part1&0x03
            inv_part2 = ~part2&0x03
            inv_part3 = ~part3&0x03
            inv_part4 = ~part4&0x03
            nums.append(part1)
            nums.append(part2)
            nums.append(part3)
            nums.append(part4)
        signals = []
        signals.append(generate_sig([.6,0],10*symbol_time)) #pilot signal
        for i in nums:
            IQ = get_4IQ_from_data(i)
            signals.append(generate_sig(IQ,symbol_time))
        long_arr = np.concatenate(signals)
        print(nums)
        #print(long_arr)
        sf.write("4QAM.wav",long_arr,Fs)
        time = len(long_arr)/Fs
        sd.play(long_arr,Fs)
        print("wrote to file")
        self.symbol_time.setText("")
        self.textbox.setText("")
        self.message.setText("Message sent")
    def qam16_gen(self):
        text = self.textbox.text()
        symbol_time = self.symbol_time.text()
        if(symbol_time.isnumeric() == 0):
            self.message.setText("Error: Symbol time was not a number")
            return
        symbol_time = int(symbol_time)
        text = [ord(ele) for sub in text for ele in sub]
        nums = []
        print(text)
        for i in range(len(text)):
            hex = text[i]
            nibble1 = (hex&0xf0)>>4
            nibble2 = hex&0x0f
            inv_nibble1 = ~nibble1&0x0f     
            inv_nibble2 = ~nibble2&0x0f
            nums.append(nibble1)
            nums.append(nibble2)
        signals = []
        signals.append(generate_sig([.6,0],10*symbol_time)) #pilot signal
        for i in nums:
            IQ = get_16IQ_from_data(i)
            signals.append(generate_sig(IQ,symbol_time))
        long_arr = np.concatenate(signals)
        print(nums)
        sf.write("16QAM.wav",long_arr,Fs)
        time = len(long_arr)/Fs
        sd.play(long_arr,Fs)
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
