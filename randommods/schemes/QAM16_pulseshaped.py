import math
import numpy as np
import matplotlib.pyplot as plt

num_symbols = 100   # number of randomly generated symbols
baud = 4800         # bps
L = 12              # samples/symbol

# 25 symbols to check for phase ambiguity
sync_phase = np.ones(50)
# 25 random symbols to sync clock
sync_clock = np.random.randint(0, 2, 50)
# data preamble symbols (todo: barker codes)
sync_preamble = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

# generate data
data = np.random.randint(0, 2, num_symbols)

sync_data = np.concatenate([sync_phase, sync_clock, sync_preamble, data])

# convert to I and Q
arr_I = []
arr_Q = []

for i in range(1, len(sync_data), 2):
    I = sync_data[i]
    Q = sync_data[i-1]

    arr_I.append(I)
    arr_Q.append(Q)

print(arr_I)
print(arr_Q)