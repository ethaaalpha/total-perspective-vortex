from mne.io import Raw
from mne import events_from_annotations, Epochs
from sklearn.model_selection import cross_val_score, train_test_split
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
    def train(self, raws, cv=True):
        """Return cross_val_score and score"""
        conf = self.config
        X, Y = self.__preprocess(raws)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

        self.config.pipeline.fit(X_train, Y_train)

        return [
            cross_val_score(conf.pipeline, X_train, Y_train, cv=conf.cross_validator, scoring="accuracy"), 
            conf.pipeline.score(X_test, Y_test) if cv else 0
            ]

    @ensure_config
    def predict(self, raws) -> [tuple[np.ndarray, np.ndarray]]:
        """Return predicted and real values"""
        X, Y = self.__preprocess(raws)

        return [self.config.pipeline.predict(X), Y]

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
            raw.load_data()
            raw = CutFilter().filter(raw, 7, 30)
            events, _ = events_from_annotations(raw, events_ids)
            epochs = Epochs(raw, events, tmin=-1, tmax=4, baseline=None)

            all_X.append(epochs.get_data())
            all_Y.append(epochs.events[:, -1])

        return (np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0))
