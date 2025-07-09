from mne.io import Raw
from src.preprocessing.signal import FastFourierTransform
from src.preprocessing.filter import CutFilter

scalings = {"eeg": "3e-4"}

def show_fourrier(file: Raw):
    FastFourierTransform(file).analyse()

def show_standard(file: Raw):
    file.plot(
        n_channels=3, 
        duration=10, 
        scalings=scalings, 
        title="Standard")

def show_filter(file: Raw):
    file_filtered = CutFilter().filter(file, 9, 25)

    file.plot(title="Filter - Before", duration=10, n_channels=3, scalings=scalings)
    file_filtered.plot(title="Filter - After", duration=10, n_channels=3, scalings=scalings)


def show_psd_filter(file: Raw):
    file_filtered = CutFilter().filter(file, 9, 25)

    file.plot_psd(fmax=80)
    file_filtered.plot_psd(fmax=80)
