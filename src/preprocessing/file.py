import os

class FileManager():

    @classmethod
    def get_all_edf(cls, parent_path: str) -> list:
        edf_files = list()
        for file_name in os.listdir(parent_path):
            path = os.path.join(parent_path, file_name)
            print(path)
            if (file_name.endswith(".edf")):
                edf_files.append(path)
            elif os.path.isdir(path):
                edf_files.extend(cls.get_all_edf(path))
        return edf_files