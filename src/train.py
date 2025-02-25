from mne.io import read_raw_edf
from mne.io import Raw
from matplotlib import pyplot as pp
from preprocessing.signal import FastFourierTransform
from preprocessing.filter import CutFilter
from mne.decoding import CSP
import numpy as np
import mne

person = str(100).zfill(3)
raws: list[Raw] = [read_raw_edf(f"dataset/S{person}/S{person}R{i:02d}.edf", preload=True) for i in range(1, 15)]

def get_filtered_data(raw: Raw, display=True) -> Raw:
    raw_filtered = CutFilter.filter(raw, 0, 30)

    if display:
        raw.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True)
        raw_filtered.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True, block=True)
    return raw_filtered

def csp(raws: list[mne.io.Raw], epoch_duration: float = 1.0):
    epochs_list = []
    labels_list = []
    min_samples = float("inf")  # Track the shortest epoch length

    for raw in raws:
        # Extract events from annotations
        events, event_id = mne.events_from_annotations(raw)

        # Create epochs
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=0, tmax=epoch_duration,
            baseline=None, preload=True
        )
        
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        epochs_list.append(data)
        labels_list.append(epochs.events[:, -1])

        # Track the minimum epoch length
        min_samples = min(min_samples, data.shape[2])

    # Truncate all epochs to the shortest length
    epochs_list = [e[:, :, :min_samples] for e in epochs_list]  # Crop all to `min_samples`

    # Convert to numpy arrays
    X = np.concatenate(epochs_list, axis=0)  # EEG signals before CSP
    y = np.concatenate(labels_list, axis=0)  # Labels

    # Train CSP
    csp = CSP(n_components=4, log=True, cov_est='epoch')
    X_csp = csp.fit_transform(X, y)  # CSP-transformed signals

    # Visualization
    fig, axes = pp.subplots(2, 1, figsize=(10, 6))

    # Plot raw EEG signal (before CSP)
    axes[0].plot(X[0, 0, :], label="Raw Signal (Ch 1, Epoch 1)")
    axes[0].set_title("Raw EEG Signal Before CSP")
    axes[0].set_xlabel("Time (samples)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid()
    axes[0].legend()

    # Plot CSP-transformed signal
    axes[1].plot(X_csp[0, :], label="CSP Component 1")
    axes[1].set_title("EEG Signal After CSP Transformation")
    axes[1].set_xlabel("Time (samples)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid()
    axes[1].legend()

    pp.tight_layout()
    pp.show()

    return X_csp, y, csp

def perform_csp(raws: list[Raw]):
    csp(raws)

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

