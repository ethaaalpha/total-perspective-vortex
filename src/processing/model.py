from abc import ABC, abstractmethod
from mne.io import Raw
from mne import events_from_annotations
from mne import Epochs
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from src.preprocessing.filter import CutFilter
from src.processing.transformers import CSPTransformer
import pickle
import numpy as np

class AbstractConfig(ABC):
    @abstractmethod
    def pipeline(self) -> Pipeline:
        pass

    @abstractmethod
    def cross_validator(self) -> BaseCrossValidator:
        pass

class DefaultConfig(AbstractConfig):
    def pipeline(self):
        return Pipeline([
            ("csp", CSPTransformer(6)),
            ("scaler", StandardScaler()),
            ("gbc", GradientBoostingClassifier(random_state=42))
        ])

    def cross_validator(self):
        return StratifiedKFold(5, shuffle=True, random_state=42)

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

        return cross_val_score(self.config.pipeline(), X, Y, 
            cv=self.config.cross_validator(), scoring="accuracy")

    @ensure_config
    def predict(self, raws):
        X, Y = self.__preprocess(raws)

    @ensure_config
    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self.config, file)

    def load(self, to_load: AbstractConfig):
        self.config = to_load

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
