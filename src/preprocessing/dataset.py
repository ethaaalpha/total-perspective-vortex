from mne.io.edf import read_raw_edf
import os

class DatasetImporter():
    EXP_1 = [3, 7, 11]
    EXP_2 = [4, 8, 12]
    EXP_3 = [5, 9, 13]
    EXP_4 = [6, 10, 14]
    choices = [EXP_1, EXP_2, EXP_3, EXP_4]

    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_subject(self, subject):
        """Return all tasks from 1-14 of a subject"""
        return [self.get_task(subject, task) for task in range(1, 15)]

    def get_experience(self, subject, experience):
        """Look for variables `EXP_1-4`, experience should be between 1-4"""

        if (experience > 4 or experience < 1):
            raise IndexError("Please choose an experience in range 1-4!")
        else:
            return [self.get_task(subject, task) for task in self.choices[experience - 1]]

    def get_task(self, subject, task):
        """struct: **folder_path/S{subject}/S{subject}R{task:02d}.edf**"""
        path = f"{self.folder_path}/S{subject}/S{subject}R{task:02d}.edf"
        return self.__load_file(path)

    def __load_file(cls, filepath: str):
        if (os.path.exists(filepath)):
            return read_raw_edf(filepath)
        else:
            raise FileNotFoundError(f"The file {filepath} do not exist!")