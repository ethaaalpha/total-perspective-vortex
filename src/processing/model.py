from mne.io import Raw
from mne import events_from_annotations, Epochs
from mne.preprocessing import ICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVC
from src.processing.config import Config
from src.preprocessing.filter import BandPassFilter
from mne import pick_types
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
    def train(self, raws) -> float:
        """Return training with on  test data (0.2) score"""
        pipeline = self.config.pipeline
        X, Y = self.__preprocess(raws)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=24)

        pipeline.fit(X_train, Y_train)

        return pipeline.score(X_test, Y_test)

    @ensure_config
    def cross_validation(self, raws) -> np.ndarray:
        conf = self.config
        X, Y = self.__preprocess(raws)

        return cross_val_score(
            conf.pipeline,
            X, Y,
            cv=conf.cross_validator,
            scoring="accuracy"
        )

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
        return self

    def __preprocess(self, raws: list[Raw]) -> tuple[np.ndarray, np.ndarray]:
        all_X = []
        all_Y = []

        for raw in raws:
            # raw.load_data()

            print(raw.annotations)
            raw.annotations.rename(dict(T1="hands", T2="feet"))
            raw.set_eeg_reference(projection=True)
            raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
            picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
            
            epochs = Epochs(
                raw,
                event_id=["hands", "feet"],
                tmin=-1,
                tmax=4,
                proj=True,
                picks=picks,
                baseline=None,
                preload=True,
            )
            epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
            labels = epochs.events[:, -1] - 2


        return (epochs_train.get_data(copy=False), labels)
