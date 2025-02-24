from mne.io import read_raw_edf
from matplotlib import pyplot as pp
import numpy as np
import os 

def get_all_edf(parent_path: str) -> list:
    edf_files = list()
    for file_name in os.listdir(parent_path):
        path = os.path.join(parent_path, file_name)
        print(path)
        if (file_name.endswith(".edf")):
            edf_files.append(path)
        elif os.path.isdir(path):
            edf_files.extend(get_all_edf(path))
    return edf_files

# load_file = [read_raw_edf(x) for x in get_all_edf("dataset/")]

raw = read_raw_edf("dataset/S012/S012R02.edf", preload=True)
raw_data = np.array(raw.get_data())

sampling_frequency = raw.info["sfreq"] # number of points per seconds (Hz)

def compute_fft(signal):
    return np.fft.fft(signal)

def compute_frequency(N):
    return np.fft.fftfreq(N, d=1/sampling_frequency)

for index, signal in enumerate(raw_data):
    # print(f"index {index}")
    N = len(signal) # number of samples
    # k = np.arange(N) # frequencies indices (x axis)
    # n = np.arange(N).reshape(-1, 1) # time indices (y axis)

    # W = np.exp(-2j * np.pi * k * n / N)
    # fourier_transform = np.dot(W, signal)
    fourier_transform = compute_fft(signal)
    frequencies = compute_frequency(N)

    pp.plot(frequencies[:N // 2], np.abs(fourier_transform[:N // 2]), label=f"Channel {index}")
    # frequencies = np.array([k * sampling_frequency / N for k in range(N)])

    # pp.plot(frequencies[:N // 2], np.abs(fourier_transform[:N // 2]))

pp.xlabel("Frequency (Hz)")
pp.ylabel("Magnitude")
pp.grid()
pp.xlim(0, 40)

pp.show()

# 
# print(n_entries)
# print(raw.ch)
# raw.plot(n_channels=2, scalings={"eeg":"3e-4"}, title="non filter", show=True)
# raw.filter(1, 30)
# raw.plot(n_channels=2, scalings={"eeg":"3e-4"}, title="filtered", show=True, block=True)
