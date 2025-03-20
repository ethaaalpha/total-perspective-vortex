from abc import ABC, abstractmethod
from mne.io import Raw
from mne import events_from_annotations
from mne import Epochs
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from src.preprocessing.filter import CutFilter
from src.processing.transformers import CSPTransformer
import numpy as np

class AbstractModel(ABC):
    def __init__(self, raws: list[Raw]):
        self.data = raws
    
    @abstractmethod
    def prepare(self):
        """Data formating and preprocessing"""
        pass

    @abstractmethod 
    def fit(self, random_state=42) -> tuple:
        """Model learning, should return the cross_val_score result function"""
        pass

class DefaultModel(AbstractModel):
    def prepare(self):
        self.prepared_X = []
        self.prepared_Y = []
        events_ids = {"T1": 1, "T2": 2}

        for raw in self.data:
            raw = CutFilter().filter(raw, 0, 30)
            events, _ = events_from_annotations(raw, events_ids)
            epochs = Epochs(raw, events, tmin=0, tmax=2, baseline=(1, 1.5))

            self.prepared_X.append(epochs.get_data())
            self.prepared_Y.append(epochs.events[:, -1])

        self.prepared_X = np.concatenate(self.prepared_X, axis=0)
        self.prepared_Y = np.concatenate(self.prepared_Y, axis=0)

    def fit(self, random_state=42):
        pipeline = Pipeline([
            ("csp", CSPTransformer(6)),
            ("scaler", StandardScaler()),
            ("gbc", GradientBoostingClassifier(random_state=random_state))
        ])

        fold = StratifiedKFold(5, shuffle=True, random_state=random_state)

        return (cross_val_score(pipeline,
            self.prepared_X, 
            self.prepared_Y, 
            cv=fold, scoring="accuracy"))