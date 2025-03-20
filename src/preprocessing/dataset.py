from mne.io.edf import read_raw_edf
from mne.io import Raw
from mne.datasets.eegbci import load_data, standardize
from mne.channels import make_standard_montage, get_builtin_montages
from pathlib import Path
import os

class DatasetImporter():
    EXP_1 = [3, 7, 11]
    EXP_2 = [4, 8, 12]
    EXP_3 = [5, 9, 13]
    EXP_4 = [6, 10, 14]
    choices = [EXP_1, EXP_2, EXP_3, EXP_4]

    def __init__(self, folder_path):
        self.folder_path = str(Path(folder_path).resolve())

    def get_subject(self, subject) -> list[Raw]:
        """Return all tasks from 1-14 of a subject"""
        return self.__format_data(load_data(subject, [run for run in range(1, 15)], path=self.folder_path))

    def get_experience(self, subject, experience) -> list[Raw]:
        if (experience > 4 or experience < 1):
            raise IndexError("Please choose an experience in range 1-4!")
        else:
            return self.__format_data(load_data(subject, [run for run in self.choices[experience - 1]], path=self.folder_path))

    def get_task(self, subject, task) -> Raw:
        return self.__format_data(load_data(subject, task, path=self.folder_path))[0]
    
    def __format_data(cls, data: list):
        format = [read_raw_edf(file, preload=True) for file in data]

        for raw in format:
            standardize(raw)
            raw.set_montage(make_standard_montage("standard_1005"))
        return format
