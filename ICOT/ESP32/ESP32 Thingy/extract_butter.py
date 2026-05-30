import numpy as np
from scipy.signal import butter, filtfilt

# Sampling parameters
sr = 2500     # 8 kHz sample rate
t = np.linspace(0, 1, sr, endpoint=False)  # 1-second signal
temp = t
temp = list(temp)
temp = [float(i) for i in t]
#print(temp)

# Example signal: sum of two sine waves
sig = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*500*t) + 0.3*np.sin(2*np.pi*1500*t)

# Bandpass filter specs
low_cut = 100
high_cut = 1000
order = 5
nyq = 0.5 * sr
low = low_cut / nyq
high = high_cut / nyq

# Get Butterworth coefficients
b, a = butter(order, [low, high], btype='band')

# Apply forward-backward filter
filtered = filtfilt(b, a, sig)
temp = filtered
temp = list(temp)
temp = [float(i) for i in t]
print(temp)

'''
from scipy.signal import butter

fs = 16000           # sampling rate
low_cut = 100        # Hz
high_cut = 1000      # Hz
nyq = fs / 2

low = low_cut / nyq
high = high_cut / nyq

b, a = butter(5, [low, high], btype='band')  # order=5
print(len(b),len(a))
for i in b:
    print(i)
print()
for i in a:
    print(i)
#print("b =", list(b))
#print("a =", list(b))

'''