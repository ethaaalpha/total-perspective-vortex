from mne.io.edf import read_raw_edf
from mne.io import Raw
from mne.datasets.eegbci import load_data, standardize
from mne.channels import make_standard_montage
from pathlib import Path

class DatasetImporter():
    EXP_1 = [3, 7, 11]
    EXP_2 = [4, 8, 12]
    EXP_3 = [5, 9, 13]
    EXP_4 = [6, 10, 14]
    EXP_5 = [*EXP_1, *EXP_2]
    EXP_6 = [*EXP_3, *EXP_4]
    choices = [EXP_1, EXP_2, EXP_3, EXP_4, EXP_5, EXP_6]

    def __init__(self, folder_path):
        self.folder_path = str(Path(folder_path).resolve())

    def get_subject(self, subject) -> list[Raw]:
        """Return all tasks from 1-14 of a subject"""
        return self.__format_data(load_data(subject, [run for run in range(3, 15)], path=self.folder_path))

    def get_experiment(self, subject, experiment) -> list[Raw]:
        if (experiment > 6 or experiment < 1):
            raise IndexError("Please choose an experiment in range 1-46!")
        else:
            return self.__format_data(load_data(subject, [run for run in self.choices[experiment - 1]], path=self.folder_path, update_path=True))

    def get_task(self, subject, task) -> Raw:
        return self.__format_data(load_data(subject, task, path=self.folder_path))[0]
    
    def __format_data(cls, data: list):
        format = [read_raw_edf(file) for file in data]

        for raw in format:
            standardize(raw)
            raw.set_montage(make_standard_montage("standard_1005"))
        return format
