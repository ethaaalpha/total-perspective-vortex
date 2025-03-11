from mne.io import read_raw_edf
from mne.io import Raw
from matplotlib import pyplot as pp
from sklearn.svm import SVC
from preprocessing.signal import FastFourierTransform
from preprocessing.filter import CutFilter
from mne.decoding import CSP
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut, ShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import numpy as np
import mne
from collections import Counter

from processing.transformers import CSPTransformer

TASK_1 = [3, 7, 11]
TASK_2 = [4, 8, 12]
TASK_3 = [5, 9, 13]
TASK_4 = [6, 10, 14] 

person = str(100).zfill(3)
raws: list[Raw] = [read_raw_edf(f"dataset/S{person}/S{person}R{i:02d}.edf") for i in range(1, 15)] # CSP only works with n >= 2 classes so baseline do not have
selected_raws = list(raws[i - 1] for i in TASK_1)

def get_filtered_data(raw: Raw, display=True) -> Raw:
    raw_filtered = CutFilter.filter(raw, 0, 20)

    if display:
        raw.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True)
        raw_filtered.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True, block=True)
    return raw_filtered

def runner(raws: list[Raw]):
    for raw in raws:
        raw.load_data()
        FastFourierTransform(raw).analyse()
    raws_filtered = [get_filtered_data(raw, False) for raw in raws]

    all_X = []
    all_Y = []

    for raw in raws_filtered:
        annotations, _ = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, annotations, tmin=0, tmax=2, baseline=(0,0), preload=True)
        X = epochs.get_data() # n_epochs, n_features, n_times
        Y = epochs.events[:, -1]
    
        all_X.append(X)
        all_Y.append(Y)

    all_X = np.concatenate(all_X, axis=0)
    all_Y = np.concatenate(all_Y, axis=0)

    pipeline = Pipeline([
        ("csp", CSPTransformer(4)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        # ("rfc", RandomForestClassifier(random_state=42))
        # ("svc", SVC(kernel="rbf", random_state=42))
    ])

    # fold = ShuffleSplit(10, random_state=42)
    fold = StratifiedKFold(shuffle=True, random_state=42)
    # fold = KFold(5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(pipeline, all_X, all_Y, cv=fold, scoring='accuracy')

    print(f'Cross-validated accuracies: {cv_scores}')
    print(f'Mean cross-validation accuracy: {cv_scores.mean() * 100:.2f}%')

runner(selected_raws)

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

