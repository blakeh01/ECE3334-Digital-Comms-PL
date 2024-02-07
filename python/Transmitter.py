import os
import numpy as np
import matplotlib as plt
import soundfile as sf

Fs = 48000
carrier_freq = Fs/10
Tsym = 1* 10**-1
Fsym = 1/Tsym
symbol_time = 5 #time in milis
global Time_elapsed 
Time_elapsed = 0
def generate_sig(amplitudes,symbol_time):  
    global Time_elapsed 
    t0 = np.arange(Time_elapsed, Time_elapsed+(Tsym*symbol_time), (1/Fs))
    Time_elapsed =  Time_elapsed+symbol_time*Tsym
    return amplitudes[0]* np.cos(2*np.pi*carrier_freq*t0)+ amplitudes[1]* np.sin(2*np.pi*carrier_freq*t0)
def get_IQ_from_data(data):
    constellation_map = {
        0: [-1, -1],
        1: [-1, 1/3],
        3: [-1, -1/3],
        2: [-1, -1],
        4: [-1, 1],
        5: [-1/3, 1/3],
        7: [-1/3, -1/3],
        6: [-1/3, -1],
        12: [1/3, 1],
        13: [1/3, 1/3],
        15: [1/3, -1/3],
        14: [1/3, -1],
        8: [1, 1],
        9: [1, 1/3],
        11: [1, -1/3],
        10: [1, -1],
    }
    print(data)
    return constellation_map[data]


content = open('Text_doc.txt', 'rb').read()
nums = []


for i in range(len(content)):
    hex = content[i]
    nibble1 = (hex&0xf0)>>4
    nibble2 = hex&0x0f
    inv_nibble1 = ~nibble1&0x0f
    inv_nibble2 = ~nibble2&0x0f
    nums.append(nibble1)
    nums.append(inv_nibble1)
    nums.append(nibble2)
    nums.append(inv_nibble2)
signals = []
signals.append(generate_sig([1,0],5))
for i in nums:
    IQ = get_IQ_from_data(i)
    signals.append(generate_sig(IQ,symbol_time))
long_arr = np.concatenate(signals)
sf.write("JOE.wav",long_arr,Fs)