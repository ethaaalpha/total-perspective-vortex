from mne.io import Raw
from mne import events_from_annotations, Epochs
from sklearn.model_selection import cross_val_score
from src.processing.config import Config
from src.preprocessing.filter import CutFilter
import pickle
import numpy as np

class Model():
    def ensure_config(func):
        def wrapper(cls, *args, **kwargs):
            if cls.config is None:
                raise MemoryError("A model should be loaded before calling this method.")
            return func(cls, *args, **kwargs)
        return wrapper

    @ensure_config
    def train(self, raws):
        X, Y = self.__preprocess(raws)

        self.config.pipeline.fit(X, Y)
    
    @ensure_config
    def evaluate(self, raws):
        X, Y = self.__preprocess(raws)

        return cross_val_score(self.config.pipeline, X, Y, cv=self.config.cross_validator, scoring="accuracy")

    @ensure_config
    def predict(self, raws):
        X, Y = self.__preprocess(raws)

        print(self.config.pipeline.predict(X))
        print(Y)

    @ensure_config
    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self.config, file)

    def load(self, config: Config):
        self.config = config

    def __preprocess(self, raws: list[Raw]) -> tuple[np.ndarray, np.ndarray]:
        all_X = []
        all_Y = []
        events_ids = {"T1": 1, "T2": 2} # exclude T0

        for raw in raws:
            raw = CutFilter().filter(raw, 0, 30)
            events, _ = events_from_annotations(raw, events_ids)
            epochs = Epochs(raw, events, tmin=0, tmax=2, baseline=(1, 1.5))

            all_X.append(epochs.get_data())
            all_Y.append(epochs.events[:, -1])

        return (np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0))
