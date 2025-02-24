from mne.io import read_raw_edf
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

raw = read_raw_edf("dataset/S001/S001R01.edf")
raw_data = np.array(raw.get_data())
print(raw.info["sfreq"])
print(np.shape(raw_data[2]))
# print(raw.ch)
# raw.plot(n_channels=2, scalings={"eeg":"3e-4"}, title="non filter", show=True)
# raw.filter(1, 30)
# raw.plot(n_channels=2, scalings={"eeg":"3e-4"}, title="filtered", show=True, block=True)
