from mne.io import read_raw_edf
from mne.io import Raw
from matplotlib import pyplot as pp
from sklearn.svm import SVC, SVR
from preprocessing.signal import FastFourierTransform
from preprocessing.filter import CutFilter
from mne.decoding import CSP
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut, ShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import mne
from collections import Counter

from processing.transformers import CSPTransformer

# 1 and 2 are baselines
TASK_1 = [3, 7, 11] # open and close left or right fist
TASK_2 = [4, 8, 12] # imagine opening and closing left or right fist
TASK_3 = [5, 9, 13] # open and close both fists or both feet
TASK_4 = [6, 10, 14] # imagine opening and closing both fists or both feet
TASK_TEST = [3]

person = str(45).zfill(3)
raws: list[Raw] = [read_raw_edf(f"dataset/S{person}/S{person}R{i:02d}.edf") for i in range(1, 15)] # CSP only works with n >= 2 classes so baseline do not have
selected_raws = list(raws[i - 1] for i in TASK_1)

def get_filtered_data(raw: Raw, display=True) -> Raw:
    raw_filtered = CutFilter.filter(raw, 7, 30)

    if display:
        raw.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True)
        raw_filtered.plot(duration=15, start=0, n_channels=3, scalings={"eeg":"16e-5"}, show=True, block=True)
    return raw_filtered

def runner(raws: list[Raw]):
    for raw in raws:
        raw.load_data()
        # raw.compute_psd().plot(average=True)
        FastFourierTransform(raw).analyse()
    raws_filtered = [get_filtered_data(raw, False) for raw in raws]

    all_X = []
    all_Y = []

    for raw in raws_filtered:
        event_id = {"T1": 1, "T2": 2} # we exclude T0 since it's the "nothing activity"
        events, _ = mne.events_from_annotations(raw, event_id)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=2, baseline=(0,0), preload=True)
        X = epochs.get_data() # n_epochs, n_features, n_times
        Y = epochs.events[:, -1]
        print(Y)
        print(X.shape)
        print(f"la shape {CSPTransformer(14).fit_transform(X, Y).shape}")
        all_X.append(X)
        all_Y.append(Y)

    all_X = np.concatenate(all_X, axis=0)
    all_Y = np.concatenate(all_Y, axis=0)

    pipeline = Pipeline([
        ("csp", CSPTransformer(8)),
        # ("standard", MinMaxScaler()),
        ("scaler", StandardScaler()),
        # ("rfc", MultinomialNB())
        # ("rfc", SGDClassifier(random_state=42))
        ("gbc", GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.0001))
        # ("rfc", RandomForestClassifier(n_estimators=100, random_state=42))
        # ("svc", SVC(kernel="linear", random_state=42))
        # ("svc", LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])

    fold = StratifiedKFold(5, shuffle=True, random_state=42) # to ensure repartition of classes in each fold !

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

