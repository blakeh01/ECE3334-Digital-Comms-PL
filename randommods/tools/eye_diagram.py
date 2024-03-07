import numpy as np
from matplotlib import pyplot as plt


def plot_eye(I, Q, L, disp_name="Eye Diagram"):
    demod_I_reshaped = np.reshape(I[:len(I) - (len(I) % (2 * L))], (-1, 2 * L))
    demod_Q_reshaped = np.reshape(Q[:len(Q) - (len(Q) % (2 * L))], (-1, 2 * L))

    # Plot the eye diagram for I component
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    for i in range(demod_I_reshaped.shape[0]):
        plt.plot(demod_I_reshaped[i], color='b', alpha=0.5)
    plt.axvline(x=L, color='k', linestyle='--')  # Add a vertical line at the symbol sampling frequency
    plt.title(disp_name + ' (I component)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.ylim(-1, 1)
    plt.xlim(0, L * 3 / 2)  # Adjust x-axis limits to hold two symbol durations

    # Plot the eye diagram for Q component
    plt.subplot(2, 1, 2)
    for i in range(demod_Q_reshaped.shape[0]):
        plt.plot(demod_Q_reshaped[i], color='r', alpha=0.5)
    plt.axvline(x=L, color='k', linestyle='--')  # Add a vertical line at the symbol sampling frequency
    plt.title(disp_name + ' (Q component)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.ylim(-1, 1)
    plt.xlim(0, L * 3 / 2)  # Adjust x-axis limits to hold two symbol durations

    plt.tight_layout()
    plt.show()