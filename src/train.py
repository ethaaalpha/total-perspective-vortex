from mne.io import read_raw_edf
from mne.io import Raw
from matplotlib import pyplot as pp
from preprocessing.signal import FastFourierTransform
from preprocessing.filter import CutFilter
import numpy as np

person = str(100).zfill(3)
raws: list[Raw] = [read_raw_edf(f"dataset/S{person}/S{person}R{i:02d}.edf", preload=True) for i in range(1, 15)]

def get_filtered_data(raw: Raw, display=True) -> Raw:
    raw_filtered = CutFilter.filter(raw, 0, 30)

    if display:
        raw.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True)
        raw_filtered.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True, block=True)
    return raw_filtered

def perform_csp(raws: list[Raw]):
    return

def runner(raws: list[Raw]):
    raws_filtered = [get_filtered_data(raw, False) for raw in raws]
    perform_csp(raws)

runner(raws)



# pp.show()

# 
# print(n_entries)
# print(raw.ch)
# raw.plot(n_channels=2, scalings={"eeg":"3e-4"}, title="non filter", show=True)
# raw.filter(1, 30)
# raw.plot(n_channels=2, scalings={"eeg":"3e-4"}, title="filtered", show=True, block=True)

# def compute_fft(signal):
#     return np.fft.fft(signal)

# def compute_frequency(N):
#     return np.fft.fftfreq(N, d=1/sampling_frequency)

# for index, signal in enumerate(raw_data):
#     # print(f"index {index}")
#     N = len(signal) # number of samples
#     # k = np.arange(N) # frequencies indices (x axis)
#     # n = np.arange(N).reshape(-1, 1) # time indices (y axis)

#     # W = np.exp(-2j * np.pi * k * n / N)
#     # fourier_transform = np.dot(W, signal)
#     fourier_transform = compute_fft(signal)
#     frequencies = compute_frequency(N)

#     pp.plot(frequencies[:N // 2], np.abs(fourier_transform[:N // 2]), label=f"Channel {index}")
#     # frequencies = np.array([k * sampling_frequency / N for k in range(N)])

#     # pp.plot(frequencies[:N // 2], np.abs(fourier_transform[:N // 2]))

