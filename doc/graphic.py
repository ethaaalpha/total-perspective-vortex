from mne.io import read_raw_edf

def example_signal():
    raw = read_raw_edf("doc/data/example.edf", preload=True)
    raw.plot(n_channels=1, duration=2, scalings={"eeg": "13e-5"}, title="EEG Example", show=True, block=True)

if __name__ == "__main__":
    example_signal()